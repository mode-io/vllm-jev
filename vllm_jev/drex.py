"""Experimental causal Drex rows and native vLLM pointer scoring.

Encoding/readout semantics follow Nace.AI's Apache-2.0 Drex reference at
be8ba7c8157241b9393862e9fcec104d0dc5caff (derived from Jared Palmer's Kev).
No downloaded Python is imported by serving.
"""

import asyncio
import json
import math
import re
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoTokenizer

from .decision import DecisionService, _readout_positions, _render_kev
from .decision_template import current_template
from .drex_export import (
    HEAD_FILE,
    MANIFEST_FILE,
    PROTOCOL,
    validate_template,
    verify_drex,
)


def answer(kind, keys, descriptions, probabilities):
    if (
        not probabilities
        or any(not math.isfinite(x) or not 0 <= x <= 1 for x in probabilities)
        or abs(sum(probabilities) - 1) > 1e-5
    ):
        raise RuntimeError("Drex returned invalid probabilities")
    if kind == "noul":
        return {"type": kind, "noul": round(float(probabilities[1]), 4)}
    n = len(probabilities)
    mode = max(range(n), key=probabilities.__getitem__)
    result = {
        "type": kind,
        "probabilities": {k: round(float(v), 4) for k, v in zip(keys, probabilities)},
    }
    if kind == "choice":
        confidence = 1.0 if n == 1 else (max(probabilities) - 1 / n) / (1 - 1 / n)
        result.update(choice=keys[mode], confidence=round(float(confidence), 4))
    else:
        confidence = (
            1.0
            if n == 1
            else 1
            - sum(p * abs(i - mode) for i, p in enumerate(probabilities)) / (n - 1)
        )
        result.update(
            score=round(sum(i * float(p) for i, p in enumerate(probabilities)), 4),
            legend=descriptions,
            confidence=round(float(confidence), 4),
        )
    return result


class DrexCompiler:
    protocol = PROTOCOL

    def __init__(self, checkpoint, names, max_length):
        self.manifest = json.loads(
            (Path(checkpoint) / MANIFEST_FILE).read_text(encoding="utf-8")
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            checkpoint, local_files_only=True, trust_remote_code=False
        )
        self.model_names = names
        self.max_length = min(max_length, self.manifest["max_length"])
        self.special = self.manifest["markers"]

    def compile(self, payload):
        validate_template(current_template())
        if payload.model not in (None, "vllm-jev", "drex", *self.model_names):
            raise ValueError("requested model is not loaded")
        if payload.cache_salt is not None:
            raise ValueError(
                "Drex prototype does not support prefix caching/cache_salt"
            )
        if any(getattr(payload, name, None) for name in ("images", "videos", "media")):
            raise ValueError("Drex accepts text/JSON, not image or video inputs")
        try:
            raw = json.dumps(
                [payload.state, payload.questions], ensure_ascii=False, allow_nan=False
            )
        except (ValueError, TypeError, RecursionError) as e:
            raise ValueError("Drex input must be finite JSON") from e
        if (
            len(raw.encode("utf-8")) > 2 * 1024**2
            or not 1 <= len(payload.questions) <= 64
        ):
            raise ValueError("Drex accepts at most 2 MiB and 64 questions per request")

        def tokens(value):
            text = re.sub(r"<\|([A-Za-z0-9_]+)\|>", r"<¦\1¦>", _render_kev(value))
            return self.tokenizer.encode(text, add_special_tokens=False)

        state = [self.special[0]] + tokens(payload.state)
        if len(state) >= self.max_length:
            raise ValueError("Drex state exceeds context; input is not truncated")
        prepared, count = [], 0
        for identifier, spec in payload.questions.items():
            if not identifier or not isinstance(spec, dict) or spec.get("isolated"):
                raise ValueError("Drex requires named joint question definitions")
            kind, criteria = spec.get("type"), spec.get("criteria")
            if kind == "noul":
                if criteria is not None and not isinstance(criteria, dict):
                    raise ValueError("Noul criteria must be an object")
                keys = ["false", "true"]
                options = [
                    name
                    if (criteria or {}).get(key) in (None, "")
                    else f"{name}: {_render_kev(criteria[key])}"
                    for key, name in zip(keys, ["no", "yes"])
                ]
            elif (
                kind == "choice"
                and isinstance(criteria, dict)
                and 1 <= len(criteria) <= 255
            ):
                keys = list(criteria)
                if any(not isinstance(key, str) for key in keys):
                    raise ValueError("Drex Choice keys must be strings")
                options = [
                    key if value in (None, "") else f"{key}: {_render_kev(value)}"
                    for key, value in criteria.items()
                ]
            elif (
                kind == "score"
                and isinstance(criteria, list)
                and 1 <= len(criteria) <= 255
            ):
                keys = list(map(str, range(len(criteria))))
                options = [_render_kev(value) for value in criteria]
            else:
                raise ValueError("Invalid Drex question type or candidate count")
            count += len(options)
            if count > 256:
                raise ValueError("Drex request exceeds 256 candidates")
            ids = state + [self.special[1]] + tokens(spec.get("instructions"))
            positions = []
            for option in options:
                ids += [self.special[2]] + tokens(option) + [self.special[3]]
                positions.append(len(ids) - 1)
            ids.append(self.special[4])
            positions.append(len(ids) - 1)
            if len(ids) > self.max_length:
                raise ValueError(
                    "Drex complete question row exceeds context; input is not truncated"
                )
            prepared.append(
                (
                    identifier,
                    kind,
                    keys,
                    options,
                    False,
                    [(ids, positions, len(options))],
                )
            )
        return prepared, len(state)


class DrexService(DecisionService):
    def __init__(
        self, engine_client, checkpoint, model_id, max_length, max_inflight=128
    ):
        verify_drex(checkpoint, full=False)
        self.engine_client = engine_client
        self.model_names = [model_id] if isinstance(model_id, str) else model_id
        self.model_id = self.model_names[0]
        # Native scheduler reserves the last model slot; all compiler rows must fit.
        self.compiler = DrexCompiler(checkpoint, self.model_names, max_length - 1)
        self.head = load_file(str(Path(checkpoint) / HEAD_FILE))
        self.semaphore = asyncio.Semaphore(min(max_inflight, 4))
        self.compile_lock = asyncio.Lock()

    def _probabilities(self, hidden, positions, count, kind):
        selected = hidden[positions].float()
        q = F.linear(selected[-1], self.head["q.weight"], self.head["q.bias"])
        k = F.linear(selected[:-1], self.head["k.weight"], self.head["k.bias"])
        logits = (k @ q) * (1 / math.sqrt(q.numel()))
        logits = logits / self.compiler.manifest["temperature"]
        if len(logits) != count or not torch.isfinite(logits).all():
            raise RuntimeError("Drex returned invalid pointer logits")
        return torch.softmax(logits, -1).tolist()

    async def systemone(self, payload):
        from .endpoint import _compile_serialized, _gather_cancel_on_error

        started = time.perf_counter()
        prepared, state_tokens = await _compile_serialized(
            self.compile_lock, self.compiler.compile, payload
        )
        encoded, readouts = {}, _readout_positions(prepared)
        try:
            probabilities = await _gather_cancel_on_error(
                self._score(rows[0], kind, None, encoded, readouts)
                for _, kind, _, _, _, rows in prepared
            )
            answers = {
                identifier: answer(kind, keys, descriptions, p)
                for (identifier, kind, keys, descriptions, _, _), p in zip(
                    prepared, probabilities
                )
            }
        finally:
            for task in encoded.values():
                if not task.done():
                    task.cancel()
            await asyncio.gather(*encoded.values(), return_exceptions=True)
        return {
            "model": self.model_id,
            "answers": answers,
            "usage": {
                "input_tokens": state_tokens
                + sum(len(rows[0][0]) - state_tokens for *_, rows in prepared),
                "output_tokens": 0,
            },
            "internal_usage": {"compute_tokens": sum(map(len, encoded))},
            "metadata": {
                "experimental": True,
                "released_weight_validation": False,
                "cached_tokens": 0,
                "state_bidir": False,
                "inference_seconds": time.perf_counter() - started,
            },
        }
