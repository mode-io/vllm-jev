"""JevAny Pointer inference over native vLLM multimodal token pooling."""

import asyncio
import json
import math
import re
import secrets
import time
import uuid
from contextlib import suppress
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoProcessor
from vllm import PoolingParams

from .decision_template import current_template
from .media import load_image

SPECIAL_RE = re.compile(r"<\|([A-Za-z0-9_]+)\|>")


async def _compile_serialized(lock, function, payload):
    async with lock:
        task = asyncio.create_task(asyncio.to_thread(function, payload))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            while not task.done():
                with suppress(Exception, asyncio.CancelledError):
                    await asyncio.shield(task)
            with suppress(Exception):
                task.result()
            raise


def render(value, indent=0):
    """Match JevAny's stable rendering of System One JSON values."""
    pad = "  " * indent
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("request values must be finite")
        return str(value)
    if isinstance(value, list):
        return "\n".join(
            f"{pad}- {render(item, indent + 1).lstrip()}" for item in value
        )
    if not isinstance(value, dict):
        raise ValueError("request values must be JSON-compatible")
    return "\n".join(
        f"{pad}{key}:\n{render(item, indent + 1)}"
        if isinstance(item, (dict, list))
        else f"{pad}{key}: {render(item)}"
        for key, item in value.items()
    )


def _safe_text(tokenizer, value):
    text = SPECIAL_RE.sub(r"<¦\1¦>", render(value))
    for token in tokenizer.all_special_tokens:
        if token.startswith(("<", "[")):
            text = text.replace(
                token, token.replace("<", "‹").replace("[", "［")
            )
    return text


def _option_text(name, description):
    return (
        name
        if description in (None, "")
        else f"{name}: {render(description)}"
    )


def _question(spec):
    kind = spec.get("type")
    instructions = _safe_json_value(spec.get("instructions"))
    criteria = spec.get("criteria")
    if kind == "noul":
        if criteria is not None and not isinstance(criteria, dict):
            raise ValueError("Noul criteria must be an object")
        criteria = criteria or {}
        unknown = set(criteria) - {"false", "true"}
        if unknown:
            raise ValueError("Noul criteria accepts only false and true")
        keys = ["false", "true"]
        options = [
            _option_text("no", criteria.get("false")),
            _option_text("yes", criteria.get("true")),
        ]
        descriptions = options
    elif kind == "choice" and isinstance(criteria, dict):
        if not 1 <= len(criteria) <= 255:
            raise ValueError("Choice requires 1 to 255 options")
        keys = list(criteria)
        if any(not isinstance(key, str) or not key for key in keys):
            raise ValueError("Choice keys must be nonempty strings")
        options = [_option_text(key, value) for key, value in criteria.items()]
        descriptions = options
    elif kind == "score" and isinstance(criteria, list):
        if not 2 <= len(criteria) <= 255:
            raise ValueError("Score requires 2 to 255 levels")
        keys = [str(index) for index in range(len(criteria))]
        options = [render(value) for value in criteria]
        descriptions = options
    else:
        raise ValueError("unsupported JevAny question")
    return kind, keys, descriptions, instructions, options


def _safe_json_value(value):
    render(value)
    return value


def _answer(kind, keys, descriptions, probabilities):
    if kind == "noul":
        return {"type": kind, "noul": probabilities[keys.index("true")]}
    selected = max(range(len(probabilities)), key=probabilities.__getitem__)
    confidence = (
        1.0
        if len(probabilities) == 1
        else max(
            0.0,
            (max(probabilities) - 1 / len(probabilities))
            / (1 - 1 / len(probabilities)),
        )
    )
    result = {
        "type": kind,
        "probabilities": dict(zip(keys, probabilities)),
        "confidence": confidence,
    }
    if kind == "choice":
        result["choice"] = keys[selected]
    else:
        result["score"] = sum(
            index * probability
            for index, probability in enumerate(probabilities)
        )
        result["legend"] = dict(zip(keys, descriptions))
        distance = sum(
            probability * abs(index - selected)
            for index, probability in enumerate(probabilities)
        )
        result["confidence"] = max(
            0.0, 1.0 - distance / max(1, len(probabilities) - 1)
        )
    return result


def pointer_probabilities(head, hidden, option_positions, decision_position, temperature):
    """Apply the released Pointer head entirely in FP32."""
    options = hidden[option_positions].float()
    decision = hidden[decision_position].float()
    query = F.linear(decision, head["q.weight"], head["q.bias"])
    keys = F.linear(options, head["k.weight"], head["k.bias"])
    logits = keys @ query / math.sqrt(query.numel())
    if "residual.0.weight" in head:
        interaction = keys * query.unsqueeze(0)
        residual = F.linear(
            interaction, head["residual.0.weight"], head["residual.0.bias"]
        )
        residual = F.gelu(residual)
        residual = F.linear(
            residual, head["residual.2.weight"], head["residual.2.bias"]
        )
        logits = logits + residual.squeeze(-1)
    if not torch.isfinite(logits).all():
        raise RuntimeError("JevAny Pointer returned nonfinite logits")
    return torch.softmax(logits / temperature, -1).tolist()


class JevAnyPointerCompiler:
    def __init__(self, checkpoint: Path, model_id, max_length: int):
        self.manifest = json.loads(
            (checkpoint / "jevany_pointer_manifest.json").read_text()
        )
        self.processor = AutoProcessor.from_pretrained(
            checkpoint, local_files_only=True
        )
        self.tokenizer = self.processor.tokenizer
        self.model_names = [model_id] if isinstance(model_id, str) else model_id
        self.model_id = self.model_names[0]
        self.max_length = min(max_length, self.manifest["max_length"])
        self.max_state = self.manifest["max_state"]
        self.markers = self.manifest["markers"]
        self.image_pad_id = self.manifest["image_token_id"]

    def compile(self, payload):
        if payload.model not in (None, "vllm-jev", *self.model_names):
            raise ValueError("requested model is not loaded")
        if len(payload.questions) != 1:
            raise ValueError("JevAny Pointer requires exactly one question")
        identifier, spec = next(iter(payload.questions.items()))
        if not identifier or not isinstance(spec, dict):
            raise ValueError("question ID and definition must be nonempty")
        kind, keys, descriptions, instructions, options = _question(spec)

        media = payload.media or []
        if not isinstance(media, list) or len(media) > 2:
            raise ValueError("JevAny Pointer accepts at most two images")
        images = []
        for item in media:
            if (
                not isinstance(item, dict)
                or item.get("type") != "image"
                or not isinstance(item.get("uri"), str)
            ):
                raise ValueError("JevAny Pointer media must contain image URIs")
            images.append(load_image(item["uri"]))

        marker = [self.tokenizer.convert_ids_to_tokens(token) for token in self.markers]
        state = _safe_text(self.tokenizer, payload.state)
        instruction = _safe_text(self.tokenizer, instructions)
        rendered_options = [_safe_text(self.tokenizer, option) for option in options]
        template = current_template()
        if template.context_order is not None:
            # Plain context may move; trained option-close/decision markers stay late.
            state = _safe_text(
                self.tokenizer,
                template.context(
                    state, instruction, json.dumps(rendered_options, ensure_ascii=False)
                ),
            )
        elif template.instructions_first:
            state = "Question: " + instruction + "\n\n" + state
        text = marker[0] + state + marker[1] + instruction
        for option in rendered_options:
            text += marker[2] + option + marker[3]
        text += marker[4]

        size = None
        if images:
            size = {
                "shortest_edge": 32 * 32,
                "longest_edge": max(128 * 128, (512 * 512) // len(images)),
            }
            batch = self.processor(
                text=[
                    (
                        self.processor.vision_start_token
                        + self.processor.image_token
                        + self.processor.vision_end_token
                    )
                    * len(images)
                    + text
                ],
                images=images,
                return_tensors="pt",
                size=size,
            )
        else:
            batch = self.tokenizer(text, return_tensors="pt")
        ids = batch["input_ids"][0].tolist()
        question_id, close_id, decide_id = (
            self.markers[1],
            self.markers[3],
            self.markers[4],
        )
        question_starts = [
            index for index, token in enumerate(ids) if token == question_id
        ]
        decisions = [index for index, token in enumerate(ids) if token == decide_id]
        if len(question_starts) != 1 or len(decisions) != 1:
            raise ValueError("JevAny delimiter layout mismatch")
        option_positions = [
            index
            for index in range(question_starts[0], decisions[0])
            if ids[index] == close_id
        ]
        if len(option_positions) != len(options):
            raise ValueError("JevAny option delimiter layout mismatch")
        text_state_tokens = len(
            self.tokenizer(
                _safe_text(self.tokenizer, payload.state), add_special_tokens=False
            ).input_ids
        ) + 1
        if text_state_tokens > self.max_state:
            raise ValueError("JevAny text state exceeds checkpoint limit")
        if len(ids) > self.max_length:
            raise ValueError("JevAny prompt exceeds checkpoint context")
        return (
            identifier,
            kind,
            keys,
            descriptions,
            ids,
            option_positions,
            decisions[0],
            images,
            size,
        )


class JevAnyPointerService:
    def __init__(
        self,
        engine_client,
        checkpoint: Path,
        model_id,
        max_length: int,
        max_inflight: int = 128,
    ):
        self.engine_client = engine_client
        self.compiler = JevAnyPointerCompiler(checkpoint, model_id, max_length)
        self.head = load_file(str(checkpoint / "jevany_pointer_head.safetensors"))
        self.model_id = self.compiler.model_id
        self.semaphore = asyncio.Semaphore(max_inflight)
        self.request_semaphore = asyncio.Semaphore(min(max_inflight, 4))
        self.compile_lock = asyncio.Lock()

    async def _states(self, ids, images, size):
        prompt_ids = []
        for token in ids:
            if (
                token != self.compiler.image_pad_id
                or not prompt_ids
                or prompt_ids[-1] != token
            ):
                prompt_ids.append(token)
        prompt = {
            "prompt_token_ids": prompt_ids,
            "cache_salt": secrets.token_hex(16),
        }
        if images:
            prompt["multi_modal_data"] = {
                "image": images[0] if len(images) == 1 else images
            }
            prompt["mm_processor_kwargs"] = {"size": size}
        request_id = f"jevany-pointer-{uuid.uuid4().hex}"
        complete = False
        async with self.semaphore:
            try:
                final = None
                async for output in self.engine_client.encode(
                    prompt=prompt,
                    pooling_params=PoolingParams(
                        task="token_embed", use_activation=False
                    ),
                    request_id=request_id,
                ):
                    final = output
                if (
                    final is None
                    or not final.finished
                    or final.prompt_token_ids != ids
                ):
                    raise RuntimeError(
                        "vLLM tokenization differs from JevAny protocol"
                    )
                hidden = torch.as_tensor(final.outputs.data)
                expected = (
                    len(ids),
                    self.compiler.manifest["hidden_size"],
                )
                if hidden.shape != expected:
                    raise RuntimeError("vLLM returned incomplete JevAny states")
                complete = True
                return hidden
            finally:
                if not complete:
                    with suppress(Exception):
                        await self.engine_client.abort(request_id)

    async def systemone(self, payload):
        started = time.perf_counter()
        async with self.request_semaphore:
            compiled = await _compile_serialized(
                self.compile_lock, self.compiler.compile, payload
            )
            (
                identifier,
                kind,
                keys,
                descriptions,
                ids,
                option_positions,
                decision_position,
                images,
                size,
            ) = compiled
            hidden = await self._states(ids, images, size)
        probabilities = pointer_probabilities(
            self.head,
            hidden,
            option_positions,
            decision_position,
            self.compiler.manifest["temperature"],
        )
        return {
            "model": self.model_id,
            "answers": {
                identifier: _answer(
                    kind, keys, descriptions, probabilities
                )
            },
            "usage": {"input_tokens": len(ids), "output_tokens": 0},
            "internal_usage": {"compute_tokens": len(ids)},
            "metadata": {"inference_seconds": time.perf_counter() - started},
        }
