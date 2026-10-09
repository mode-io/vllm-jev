"""Inference-only Valen protocol over native vLLM token pooling."""

import asyncio
import json
import math
import secrets
import time
import uuid
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as functional
from safetensors.torch import load_file
from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
from vllm import PoolingParams

from .decision_template import current_template
from .media import MAX_IMAGES_PER_REQUEST, validate_text
from .media import load_image as _load_image
from .video import VideoClip, load_video


@dataclass
class ValenMedia:
    images: list = field(default_factory=list)
    video: VideoClip | None = None
    raw_ids: list | None = None
    base_length: int = 0
    video_inputs: dict = field(default_factory=dict)


def _candidates(question):
    kind = question.get("type")
    if kind == "noul":
        if question.get("criteria") is not None:
            raise ValueError("Valen Noul uses fixed true/false criteria")
        return [
            ("true", "True / 是：满足问题中的条件。"),
            ("false", "False / 否：不满足问题中的条件。"),
        ]
    criteria = question.get("criteria")
    if kind == "choice" and isinstance(criteria, dict) and 1 <= len(criteria) <= 255:
        pairs = list(criteria.items())
    elif kind == "score" and isinstance(criteria, list) and 2 <= len(criteria) <= 10:
        pairs = [
            (str(index), description) for index, description in enumerate(criteria)
        ]
    else:
        raise ValueError("unsupported Valen question or candidate count")
    if any(
        not isinstance(k, str) or not isinstance(v, str) or not k or not v
        for k, v in pairs
    ):
        raise ValueError("Valen candidates must have text keys and descriptions")
    return pairs


def _answer(kind, keys, descriptions, logits, temperature=1.0):
    if not all(math.isfinite(score) for score in logits):
        raise RuntimeError("Valen returned nonfinite decision scores")
    scores = torch.tensor(logits, dtype=torch.float32)
    probs = (scores / temperature).softmax(-1).tolist()
    total = sum(probs)
    probs = [p / total for p in probs]
    result = {"type": kind}
    if kind == "noul":
        result["noul"] = probs[keys.index("true")]
        return result
    result["probabilities"] = dict(zip(keys, probs))
    selected = max(range(len(probs)), key=probs.__getitem__)
    if kind == "choice":
        result["choice"] = keys[selected]
        result["confidence"] = (
            1.0
            if len(probs) == 1
            else max(0.0, (max(probs) - 1 / len(probs)) / (1 - 1 / len(probs)))
        )
    else:
        result["score"] = sum(index * p for index, p in enumerate(probs))
        result["legend"] = dict(zip(keys, descriptions))
        distance = sum(p * abs(index - selected) for index, p in enumerate(probs))
        uniform = sum(
            abs(index - (len(probs) - 1) / 2) for index in range(len(probs))
        ) / len(probs)
        result["confidence"] = max(0.0, 1.0 - distance / uniform)
    return result


class ValenService:
    def __init__(
        self,
        engine_client,
        model_path: Path,
        model_id: str | list[str],
        max_length: int,
        max_inflight: int = 128,
    ):
        self.engine_client = engine_client
        self.processor = Qwen3VLProcessor.from_pretrained(
            model_path, local_files_only=True
        )
        self.tokenizer = self.processor.tokenizer
        self.image_pad_id = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        self.weights = load_file(str(model_path / "valen_head.safetensors"))
        self.model_names = [model_id] if isinstance(model_id, str) else model_id
        self.model_id = self.model_names[0]
        self.max_length = max_length
        self.semaphore = asyncio.Semaphore(max_inflight)
        # Bound decoded media retained while requests wait for the engine.
        self.request_semaphore = asyncio.Semaphore(min(max_inflight, 4))
        self.compile_lock = asyncio.Lock()
        self.media_kwargs = json.loads(
            (model_path / "valen_manifest.json").read_text()
        ).get("media_kwargs", {})

    def _state(self, state, instruction_prefix=None, context_fields=None):
        media = ValenMedia()
        if isinstance(state, str):
            validate_text(state)
            messages = [{"role": "user", "content": [{"type": "text", "text": state}]}]
        elif isinstance(state, dict) and isinstance(state.get("messages"), list):
            if not state["messages"]:
                raise ValueError("Valen messages must be nonempty")
            messages = []
            image_count = 0
            for message in state["messages"]:
                if (
                    not isinstance(message, dict)
                    or not isinstance(message.get("role"), str)
                    or not message["role"]
                ):
                    raise ValueError("Valen messages require a text role")
                validate_text(message["role"])
                content = message.get("content")
                if isinstance(content, str):
                    content = [{"type": "text", "text": content}]
                if not isinstance(content, list) or not content:
                    raise ValueError(
                        "Valen message content must be text or a nonempty list"
                    )
                rendered = []
                for item in content:
                    if not isinstance(item, dict):
                        raise ValueError("Valen content items must be objects")
                    if item.get("type") == "text":
                        if not isinstance(item.get("text"), str):
                            raise ValueError("Valen text content must be a string")
                        validate_text(item["text"])
                        rendered.append({"type": "text", "text": item["text"]})
                    elif item.get("type") == "image_url":
                        if image_count >= MAX_IMAGES_PER_REQUEST:
                            raise ValueError(
                                f"Valen accepts at most {MAX_IMAGES_PER_REQUEST} images"
                            )
                        image_count += 1
                        image_url = item.get("image_url")
                        if not isinstance(image_url, dict):
                            raise ValueError("Valen image_url must contain a data URL")
                        rendered.append(
                            {
                                "type": "image",
                                "image": _load_image(image_url.get("url")),
                            }
                        )
                    elif item.get("type") == "video_url":
                        video_url = item.get("video_url")
                        if media.video is not None:
                            raise ValueError("Valen accepts at most one video")
                        if not isinstance(video_url, dict):
                            raise ValueError("Valen video_url must contain a data URL")
                        media.video = load_video(
                            video_url.get("url"), video_url.get("num_frames", 8)
                        )
                        rendered.append({"type": "video"})
                    else:
                        raise ValueError("Valen supports text, image_url and video_url")
                messages.append({"role": message["role"], "content": rendered})
        else:
            raise ValueError("Valen state must be text or a messages object")
        if context_fields is not None:
            index = 0
            while index < len(messages) and messages[index]["role"] == "system":
                index += 1
            leading, state_messages = messages[:index], messages[index:]
            blocks = {"state": state_messages}
            blocks.update(
                {
                    key: [
                        {"role": "user", "content": [{"type": "text", "text": value}]}
                    ]
                    for key, value in context_fields.items()
                }
            )
            messages = leading + current_template().arrange(blocks)
        if instruction_prefix is not None:
            index = 0
            while index < len(messages) and messages[index]["role"] == "system":
                index += 1
            messages.insert(
                index,
                {
                    "role": "user",
                    "content": [{"type": "text", "text": instruction_prefix}],
                },
            )
        media.images = [
            item["image"]
            for message in messages
            for item in message["content"]
            if item["type"] == "image"
        ]
        if media.video is not None:
            if media.images:
                raise ValueError("images and video cannot be mixed in one request")
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            media.raw_ids = self.tokenizer.encode(text, add_special_tokens=False)
            base = self.processor(
                text=text,
                videos=[media.video.frames],
                video_metadata=[dict(media.video.metadata)],
                do_sample_frames=False,
                return_tensors="pt",
                return_metadata=True,
                **self.media_kwargs,
            )
            media.video_inputs = {
                name: base[name] for name in ("pixel_values_videos", "video_grid_thw")
            }
        else:
            base = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                return_dict=True,
                return_tensors="pt",
                processor_kwargs=dict(self.media_kwargs, return_metadata=True),
            )
        ids = base["input_ids"].flatten().tolist()
        media.base_length = len(ids)
        return ids, media

    def _compile(self, payload):
        if payload.model not in (None, "vllm-jev", *self.model_names):
            raise ValueError("requested model is not loaded")
        if not 1 <= len(payload.questions) <= 64:
            raise ValueError("System One requires 1 to 64 questions")
        prepared = []
        candidate_count = 0
        for identifier, question in payload.questions.items():
            if not identifier or not isinstance(question, dict):
                raise ValueError("question IDs and definitions must be nonempty")
            kind = question.get("type")
            instructions = question.get("instructions")
            if not isinstance(instructions, str) or not instructions.strip():
                raise ValueError("Valen instructions must be nonempty text")
            validate_text(instructions)
            pairs = _candidates(question)
            for key, description in pairs:
                validate_text(key)
                validate_text(description)
            candidate_count += len(pairs)
            prepared.append((identifier, kind, instructions, pairs))
        if candidate_count > 256:
            raise ValueError("System One exceeds 256 candidates")
        if current_template().context_order is not None:
            fields = {
                "instructions": "\n".join(
                    f"Task {i + 1}: {kind}\nQuestion {i + 1}: {ins}"
                    for i, (_, kind, ins, _) in enumerate(prepared)
                ),
                "criteria": "Criteria:\n"
                + json.dumps(
                    [
                        {
                            "question": i + 1,
                            "type": kind,
                            "criteria": payload.questions[key].get("criteria"),
                        }
                        for i, (key, kind, _, _) in enumerate(prepared)
                    ],
                    ensure_ascii=False,
                    allow_nan=False,
                ),
            }
            base_ids, media = self._state(payload.state, context_fields=fields)
        elif current_template().instructions_first:
            prefix = "\n".join(
                f"Task {i + 1}: {kind}\nQuestion {i + 1}: {instructions}"
                for i, (_, kind, instructions, _) in enumerate(prepared)
            )
            base_ids, media = self._state(payload.state, prefix)
        else:
            base_ids, media = self._state(payload.state)
        questions = []
        logical_tokens = len(base_ids)
        compute_tokens = 0
        for question_index, (identifier, kind, instructions, pairs) in enumerate(
            prepared
        ):
            groups = [[pair] for pair in pairs] if kind == "score" else [pairs]
            branches = []
            for group in groups:
                suffix = self.tokenizer.encode(
                    "<|im_start|>user\n", add_special_tokens=False
                )

                def append(value):
                    suffix.extend(
                        self.tokenizer.encode(value, add_special_tokens=False)
                    )

                append(
                    f"Task: {kind}\nQuestion {question_index + 1}\nCandidates:\n"
                    if current_template().instructions_first
                    else "Task: "
                    + kind
                    + "\nQuestion: "
                    + instructions
                    + "\nCandidates:\n"
                )
                positions = []
                for key, description in group:
                    append((key + ": " if kind != "score" else "") + description)
                    positions.append(len(base_ids) + len(suffix) - 1)
                    append("\n")
                append("Decision:")
                full_ids = base_ids + suffix
                if len(full_ids) > self.max_length:
                    raise ValueError("Valen prompt exceeds model context")
                branches.append((full_ids, positions, len(full_ids) - 1))
                logical_tokens += len(suffix)
                compute_tokens += len(full_ids)
            questions.append((identifier, kind, pairs, branches))
        return questions, media, logical_tokens, compute_tokens

    async def _score(self, full_ids, positions, decision_position, media, salt):
        prompt_ids = []
        for token in full_ids:
            if (
                token != self.image_pad_id
                or not prompt_ids
                or prompt_ids[-1] != self.image_pad_id
            ):
                prompt_ids.append(token)
        prompt = {"prompt_token_ids": prompt_ids, "cache_salt": salt}
        if media.video is not None:
            prompt["prompt_token_ids"] = media.raw_ids + full_ids[media.base_length :]
            prompt["multi_modal_data"] = {
                "video": (
                    media.video.frames,
                    dict(media.video.metadata, do_sample_frames=False),
                )
            }
        elif media.images:
            prompt["multi_modal_data"] = {
                "image": media.images[0] if len(media.images) == 1 else media.images
            }
        request_id = f"valen-{uuid.uuid4().hex}"
        complete = False
        async with self.semaphore:
            try:
                outputs = self.engine_client.encode(
                    prompt=prompt,
                    pooling_params=PoolingParams(
                        task="token_embed", use_activation=False
                    ),
                    request_id=request_id,
                )
                final = None
                async for output in outputs:
                    final = output
                if final is None or not final.finished:
                    raise RuntimeError("vLLM returned no final pooling result")
                if final.prompt_token_ids != full_ids:
                    raise RuntimeError("vLLM tokenization differs from Valen protocol")
                hidden = torch.as_tensor(final.outputs.data)
                if hidden.ndim != 2 or hidden.shape != (
                    len(full_ids),
                    self.weights["candidate.weight"].shape[1],
                ):
                    raise RuntimeError(
                        "vLLM returned an incomplete token embedding sequence"
                    )
                candidate = functional.linear(
                    hidden[positions].float(), self.weights["candidate.weight"]
                )
                decision = functional.linear(
                    hidden[decision_position].float(), self.weights["decision.weight"]
                )
                logits = (candidate * decision).sum(-1) / math.sqrt(candidate.shape[-1])
                complete = True
                return logits.tolist()
            finally:
                if not complete:
                    with suppress(Exception):
                        await self.engine_client.abort(request_id)

    async def systemone(self, payload):
        async with self.request_semaphore:
            return await self._systemone(payload)

    async def _systemone(self, payload):
        from .endpoint import _gather_cancel_on_error

        started = time.perf_counter()
        # Decode outside the event loop. Serialize the processor and keep its
        # slot until a cancelled request's bounded decode has actually finished.
        async with self.compile_lock:
            compile_task = asyncio.create_task(
                asyncio.to_thread(self._compile, payload)
            )
            try:
                questions, media, logical_tokens, compute_tokens = await asyncio.shield(
                    compile_task
                )
            except asyncio.CancelledError:
                while not compile_task.done():
                    with suppress(Exception, asyncio.CancelledError):
                        await asyncio.shield(compile_task)
                with suppress(Exception):
                    compile_task.result()
                raise
        # vLLM processes raw frames itself; only the Mac runner needs these pixels.
        media.video_inputs.clear()
        salt = secrets.token_hex(16)
        answers = {}
        for identifier, kind, pairs, branches in questions:
            logits_per_branch = await _gather_cancel_on_error(
                self._score(ids, positions, decision, media, salt)
                for ids, positions, decision in branches
            )
            logits = [
                score for branch_scores in logits_per_branch for score in branch_scores
            ]
            keys = [key for key, _ in pairs]
            descriptions = [description for _, description in pairs]
            answers[identifier] = _answer(kind, keys, descriptions, logits)
        return {
            "model": self.model_id,
            "answers": answers,
            "usage": {"input_tokens": logical_tokens, "output_tokens": 0},
            "internal_usage": {"compute_tokens": compute_tokens},
            "metadata": {"inference_seconds": time.perf_counter() - started},
        }
