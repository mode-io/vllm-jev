"""Inference-only Valen protocol over native vLLM token pooling."""

import asyncio
import base64
import binascii
import io
import json
import math
import secrets
import time
import uuid
from contextlib import suppress
from pathlib import Path

import torch
import torch.nn.functional as functional
from PIL import Image
from safetensors.torch import load_file
from transformers import AutoProcessor
from vllm import PoolingParams


def _load_image(value):
    if not isinstance(value, str) or not value.startswith(
        ("data:image/png;base64,", "data:image/jpeg;base64,")
    ):
        raise ValueError("Valen images must be PNG/JPEG data URLs")
    try:
        raw = base64.b64decode(value.split(",", 1)[1], validate=True)
        if len(raw) > 8 * 1024 * 1024:
            raise ValueError("image exceeds 8 MiB")
        image = Image.open(io.BytesIO(raw))
        image.load()
        return image.convert("RGB")
    except (ValueError, OSError, binascii.Error) as error:
        raise ValueError("invalid Valen image") from error


def _candidates(question):
    kind = question.get("type")
    if kind == "noul":
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
    def __init__(self, engine_client, model_path: Path, model_id: str, max_length: int):
        self.engine_client = engine_client
        self.processor = AutoProcessor.from_pretrained(
            model_path, local_files_only=True
        )
        self.tokenizer = self.processor.tokenizer
        self.image_pad_id = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        self.weights = load_file(str(model_path / "valen_head.safetensors"))
        self.model_id = model_id
        self.max_length = max_length
        self.semaphore = asyncio.Semaphore(128)
        self.media_kwargs = json.loads(
            (model_path / "valen_manifest.json").read_text()
        ).get("media_kwargs", {})

    def _state(self, state):
        if isinstance(state, str):
            messages = [{"role": "user", "content": [{"type": "text", "text": state}]}]
        elif isinstance(state, dict) and isinstance(state.get("messages"), list):
            messages = []
            for message in state["messages"]:
                content = message["content"]
                if isinstance(content, str):
                    content = [{"type": "text", "text": content}]
                rendered = []
                for item in content:
                    if item["type"] == "text":
                        rendered.append(item)
                    elif item["type"] == "image_url":
                        rendered.append(
                            {
                                "type": "image",
                                "image": _load_image(item["image_url"]["url"]),
                            }
                        )
                    else:
                        raise ValueError("Valen supports text and images only")
                messages.append({"role": message["role"], "content": rendered})
        else:
            raise ValueError("Valen state must be text or a messages object")
        base = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs=dict(self.media_kwargs, return_metadata=True),
        )
        images = [
            item["image"]
            for message in messages
            for item in message["content"]
            if item["type"] == "image"
        ]
        return base["input_ids"].flatten().tolist(), images

    def _compile(self, payload):
        base_ids, images = self._state(payload.state)
        questions = []
        logical_tokens = len(base_ids)
        compute_tokens = 0
        for identifier, question in payload.questions.items():
            kind = question.get("type")
            instructions = question.get("instructions")
            if not isinstance(instructions, str) or not instructions.strip():
                raise ValueError("Valen instructions must be nonempty text")
            pairs = _candidates(question)
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
                    "Task: " + kind + "\nQuestion: " + instructions + "\nCandidates:\n"
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
        return questions, images, logical_tokens, compute_tokens

    async def _score(self, full_ids, positions, decision_position, images, salt):
        prompt_ids = []
        for token in full_ids:
            if (
                token != self.image_pad_id
                or not prompt_ids
                or prompt_ids[-1] != self.image_pad_id
            ):
                prompt_ids.append(token)
        prompt = {"prompt_token_ids": prompt_ids, "cache_salt": salt}
        if images:
            prompt["multi_modal_data"] = {
                "image": images[0] if len(images) == 1 else images
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
                if final is None or final.prompt_token_ids != full_ids:
                    raise RuntimeError("vLLM tokenization differs from Valen protocol")
                hidden = torch.as_tensor(final.outputs.data)
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
        started = time.perf_counter()
        questions, images, logical_tokens, compute_tokens = self._compile(payload)
        salt = secrets.token_hex(16)
        answers = {}
        for identifier, kind, pairs, branches in questions:
            logits_per_branch = await asyncio.gather(
                *(
                    self._score(ids, positions, decision, images, salt)
                    for ids, positions, decision in branches
                )
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
