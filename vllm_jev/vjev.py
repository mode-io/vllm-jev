"""Native vLLM inference for the released vjev vision readout."""

import asyncio
import json
import math
import secrets
import time
import uuid
from contextlib import suppress
from pathlib import Path

import torch
import torch.nn.functional as functional
from safetensors.torch import load_file
from transformers import AutoProcessor
from vllm import PoolingParams

from .media import MAX_IMAGES_PER_REQUEST, load_image, validate_text


def _label(index: int) -> str:
    result = ""
    while index >= 0:
        index, remainder = divmod(index, 26)
        result = chr(65 + remainder) + result
        index -= 1
    return result


class VjevService:
    def __init__(
        self,
        engine_client,
        model_path: Path,
        model_id: str | list[str],
        max_length: int,
        max_inflight: int = 128,
    ):
        self.engine_client = engine_client
        self.processor = AutoProcessor.from_pretrained(
            model_path, local_files_only=True
        )
        self.tokenizer = self.processor.tokenizer
        self.config = json.loads((model_path / "config.json").read_text())
        self.meta = json.loads((model_path / "vjev_manifest.json").read_text())
        self.head = load_file(str(model_path / "vjev_head.safetensors"))
        self.image_pad_id = self.config["image_token_id"]
        self.model_names = [model_id] if isinstance(model_id, str) else model_id
        self.model_id = self.model_names[0]
        self.max_length = min(max_length, self.meta["max_length"])
        self.semaphore = asyncio.Semaphore(max_inflight)

    def _state(self, state):
        if isinstance(state, str):
            return [], validate_text(state)
        if not isinstance(state, dict) or not isinstance(state.get("messages"), list):
            raise ValueError("vjev state must be text or a messages object")
        images, texts = [], []
        for message in state["messages"]:
            if not isinstance(message, dict) or not isinstance(
                message.get("role"), str
            ):
                raise ValueError("vjev messages require a role")
            content = message.get("content")
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            if not isinstance(content, list):
                raise ValueError("vjev message content must be text or a list")
            for item in content:
                if not isinstance(item, dict):
                    raise ValueError("vjev content items must be objects")
                if item.get("type") == "text" and isinstance(item.get("text"), str):
                    texts.append(validate_text(item["text"]))
                elif item.get("type") == "image_url":
                    image_url = item.get("image_url")
                    if not isinstance(image_url, dict):
                        raise ValueError("vjev image_url must contain a data URL")
                    if len(images) == MAX_IMAGES_PER_REQUEST:
                        raise ValueError(
                            f"vjev accepts at most {MAX_IMAGES_PER_REQUEST} images"
                        )
                    image = load_image(image_url.get("url"))
                    if max(image.size) > 512:
                        ratio = 512 / max(image.size)
                        image = image.resize(
                            (
                                max(1, int(image.width * ratio)),
                                max(1, int(image.height * ratio)),
                            )
                        )
                    images.append(image)
                else:
                    raise ValueError("vjev supports text and images only")
        if not images and not texts:
            raise ValueError("vjev state is empty")
        return images, "\n".join(texts)

    def _compile(self, payload):
        if payload.model not in (None, "vllm-jev", *self.model_names):
            raise ValueError("requested model is not loaded")
        if not 1 <= len(payload.questions) <= 64:
            raise ValueError("System One requires 1 to 64 questions")
        prepared, candidates = [], 0
        for identifier, question in payload.questions.items():
            if not identifier or not isinstance(question, dict):
                raise ValueError("question IDs and definitions must be nonempty")
            kind, instructions = question.get("type"), question.get("instructions")
            if not isinstance(instructions, str) or not instructions.strip():
                raise ValueError("vjev instructions must be nonempty text")
            validate_text(instructions)
            criteria = question.get("criteria")
            if kind == "noul":
                if criteria is not None:
                    raise ValueError("vjev Noul has no custom criteria")
                pairs = []
            elif (
                kind == "choice"
                and isinstance(criteria, dict)
                and 2 <= len(criteria) <= 255
            ):
                pairs = [
                    (key, key if value is None else value)
                    for key, value in criteria.items()
                ]
            elif (
                kind == "score"
                and isinstance(criteria, list)
                and 2 <= len(criteria) <= 10
            ):
                pairs = [(str(index), value) for index, value in enumerate(criteria)]
            else:
                raise ValueError("unsupported vjev question or candidate count")
            if any(
                not isinstance(key, str)
                or not key
                or not isinstance(value, str)
                or not value
                for key, value in pairs
            ):
                raise ValueError("vjev candidates need text labels and descriptions")
            for _, value in pairs:
                validate_text(value)
            candidates += len(pairs) if pairs else 1
            prepared.append((identifier, kind, instructions, pairs))
        if candidates > 256:
            raise ValueError("System One exceeds 256 candidates")
        images, text = self._state(payload.state)
        base = []
        if images:
            processed = self.processor.image_processor(
                images=images, return_tensors="pt"
            )
            merge = self.processor.image_processor.merge_size**2
            for index, grid in enumerate(processed["image_grid_thw"]):
                if len(images) > 1:
                    base += self.tokenizer.encode(
                        ("" if index == 0 else "\n") + f"Picture {index + 1}: ",
                        add_special_tokens=False,
                    )
                base += [self.config["vision_start_token_id"]]
                base += [self.image_pad_id] * (math.prod(map(int, grid)) // merge)
                base += [self.config["vision_end_token_id"]]
        if text:
            base += self.tokenizer.encode(text, add_special_tokens=False)
        questions, compute_tokens = [], 0
        for identifier, kind, instructions, pairs in prepared:
            ids = base + self.tokenizer.encode(
                f"\n\nQuestion: {instructions}", add_special_tokens=False
            )
            slots = []
            if kind == "noul":
                slots.append(len(ids) - 1)
            else:
                for index, (_, description) in enumerate(pairs):
                    ids += self.tokenizer.encode(
                        f"\n({_label(index)}) {description}", add_special_tokens=False
                    )
                ids += self.tokenizer.encode("\nAnswer:", add_special_tokens=False)
                for index in range(len(pairs)):
                    ids += self.tokenizer.encode(
                        f" ({_label(index)})", add_special_tokens=False
                    )
                    slots.append(len(ids) - 1)
            if len(ids) > self.max_length:
                raise ValueError("vjev prompt exceeds model context")
            questions.append((identifier, kind, pairs, ids, slots))
            compute_tokens += len(ids)
        return questions, images, len(base), compute_tokens

    async def _score(self, ids, slots, images, salt):
        prompt_ids = []
        for token in ids:
            if token != self.image_pad_id or not prompt_ids or prompt_ids[-1] != token:
                prompt_ids.append(token)
        prompt = {"prompt_token_ids": prompt_ids, "cache_salt": salt}
        if images:
            prompt["multi_modal_data"] = {
                "image": images[0] if len(images) == 1 else images
            }
        request_id = f"vjev-{uuid.uuid4().hex}"
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
                if final is None or not final.finished or final.prompt_token_ids != ids:
                    raise RuntimeError("vLLM did not return the expected vjev tokens")
                hidden = torch.as_tensor(final.outputs.data)
                if hidden.shape != (len(ids), self.head["weight"].shape[1]):
                    raise RuntimeError(
                        "vLLM returned an incomplete token embedding sequence"
                    )
                logits = functional.linear(
                    hidden[slots].float(), self.head["weight"], self.head["bias"]
                ).flatten()
                if not torch.isfinite(logits).all():
                    raise RuntimeError("vjev returned nonfinite decision scores")
                complete = True
                return logits.tolist()
            finally:
                if not complete:
                    with suppress(Exception):
                        await self.engine_client.abort(request_id)

    async def systemone(self, payload):
        from .endpoint import _gather_cancel_on_error

        started = time.perf_counter()
        questions, images, state_tokens, compute_tokens = self._compile(payload)
        salt = secrets.token_hex(16)
        logits_rows = await _gather_cancel_on_error(
            self._score(ids, slots, images, salt) for _, _, _, ids, slots in questions
        )
        answers = {}
        for (identifier, kind, pairs, _, _), logits in zip(questions, logits_rows):
            if kind == "noul":
                answers[identifier] = {
                    "type": "noul",
                    "noul": torch.sigmoid(torch.tensor(logits[0])).item(),
                }
                continue
            probs = torch.softmax(torch.tensor(logits), -1).tolist()
            keys = [key for key, _ in pairs]
            selected = max(range(len(probs)), key=probs.__getitem__)
            confidence = max(0.0, (max(probs) - 1 / len(probs)) / (1 - 1 / len(probs)))
            answer = {
                "type": kind,
                "probabilities": dict(zip(keys, probs)),
                "confidence": confidence,
            }
            if kind == "choice":
                answer["choice"] = keys[selected]
            else:
                answer["score"] = sum(
                    index * value for index, value in enumerate(probs)
                )
                answer["legend"] = dict(pairs)
            answers[identifier] = answer
        return {
            "model": self.model_id,
            "answers": answers,
            "usage": {
                "input_tokens": state_tokens
                + sum(len(ids) - state_tokens for _, _, _, ids, _ in questions),
                "output_tokens": 0,
            },
            "internal_usage": {"compute_tokens": compute_tokens},
            "metadata": {"inference_seconds": time.perf_counter() - started},
        }
