"""Laya decisions over vLLM's native ModernBERT token states."""

import asyncio
import json
import math
import os
import time
import uuid
import warnings
from contextlib import nullcontext, suppress
from pathlib import Path
from types import SimpleNamespace

import torch
from laya.agent import Agent
from laya.common import (
    TEMP_MAX,
    TEMP_MIN,
    DecisionModel,
    clamp_temperature,
    collate_items,
    serialize_state,
)
from safetensors.torch import load_file
from transformers import AutoTokenizer
from vllm import PoolingParams

from .endpoint import SystemOneRequest, _gather_cancel_on_error


class _TokenStates(torch.nn.Module):
    """Let Laya's original trained head consume states from vLLM."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(last_hidden_state=input_ids)


class _LayaProtocol(Agent):
    """Reuse Laya's exact prompt and probability rules without loading its encoder."""

    def __init__(self, config: dict, tokenizer):
        self.cfg = config
        self.tok = tokenizer
        self.temperature = [clamp_temperature(value) for value in config["temperature"]]
        self.temperature_by_options = {
            key: clamp_temperature(value)
            for key, value in config.get("temperature_by_options", {}).items()
        }
        self.lang_temperatures = {}
        rejected = [
            key
            for key, value in config.get("temperature_by_options", {}).items()
            if self.temperature_by_options[key] != value
        ]
        if rejected:
            warnings.warn(
                f"Laya checkpoint temperatures outside [{TEMP_MIN}, {TEMP_MAX}] "
                f"were clamped for {', '.join(rejected)}; validate confidence "
                "at these option counts.",
                RuntimeWarning,
                stacklevel=2,
            )


class LayaService:
    def __init__(self, engine_client, checkpoint: Path, model_id, max_inflight=128):
        self.engine_client = engine_client
        self.model_names = [model_id] if isinstance(model_id, str) else model_id
        self.model_id = self.model_names[0]
        config = json.loads((checkpoint / "laya_config.json").read_text())
        encoder_config = json.loads((checkpoint / "config.json").read_text())
        self.protocol = _LayaProtocol(
            config, AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
        )
        self.head = DecisionModel(
            _TokenStates(encoder_config["hidden_size"]),
            head_layers=config["head_layers"],
            n_act=len(config.get("act_costs", {})) + 1,
        )
        self.head.load_state_dict(
            load_file(str(checkpoint / "laya_head.safetensors")), strict=True
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.head.to(self.device).eval()
        self.graph_head = None
        if (
            self.model_id == "convaiinnovations/laya-multilingual"
            and self.device.type == "cuda"
            and os.environ.get("VLLM_JEV_LAYA_CUDA_GRAPH") == "1"
        ):
            from .laya_graph import LayaGraphHead

            self.graph_head = LayaGraphHead(self.head)
        self.semaphore = asyncio.Semaphore(max_inflight)

    async def _encode(self, ids: list[int], salt: str):
        request_id = f"laya-{uuid.uuid4().hex}"
        complete = False
        async with self.semaphore:
            try:
                outputs = self.engine_client.encode(
                    prompt={"prompt_token_ids": ids, "cache_salt": salt},
                    pooling_params=PoolingParams(
                        task="token_embed", use_activation=False
                    ),
                    request_id=request_id,
                )
                final = None
                async for output in outputs:
                    final = output
                if final is None or not final.finished or final.prompt_token_ids != ids:
                    raise RuntimeError("vLLM returned incomplete Laya token states")
                hidden = torch.as_tensor(final.outputs.data)
                if hidden.shape != (len(ids), self.head.type_emb.embedding_dim):
                    raise RuntimeError("vLLM returned an invalid Laya hidden shape")
                complete = True
                return hidden, int(final.num_cached_tokens)
            finally:
                if not complete:
                    with suppress(Exception):
                        await self.engine_client.abort(request_id)

    async def systemone(self, payload: SystemOneRequest) -> dict:
        started = time.perf_counter()
        if payload.model not in (
            None,
            "vllm-jev",
            "laya",
            "english",
            *self.model_names,
        ):
            raise ValueError("requested model is not loaded")
        if not isinstance(payload.state, (str, dict, list)):
            raise ValueError("Laya state must be text or JSON")
        if len(serialize_state(payload.state)) > 50_000:
            raise ValueError("Laya state exceeds 50,000 characters")
        if len(payload.questions) > 64:
            raise ValueError("System One exceeds 64 questions")
        if (
            len(json.dumps(payload.questions, ensure_ascii=False, default=str))
            > 2_000_000
        ):
            raise ValueError("Laya questions exceed 2 MiB")
        ids = list(payload.questions)
        if not ids:
            return {
                "model": self.model_id,
                "answers": {},
                "usage": {"input_tokens": 0, "output_tokens": 0},
            }
        for identifier in ids:
            if not identifier:
                raise ValueError("question IDs must be nonempty")
            definition = payload.questions[identifier]
            if definition.get("type") not in ("choice", "noul", "score"):
                raise ValueError("question type must be choice, noul or score")
            criteria = definition.get("criteria")
            if (
                definition.get("type") == "choice"
                and isinstance(criteria, list)
                and any(isinstance(value, (dict, list, set)) for value in criteria)
            ):
                raise ValueError("choice labels must be scalar values")
            self.protocol._check_question(identifier, definition)
        internal = {
            identifier: self.protocol._to_internal(payload.questions[identifier])
            for identifier in ids
        }
        items = self.protocol._encode_state(payload.state, ids, internal)
        if sum(len(item["markers"]) for item in items) > 256:
            raise ValueError("System One exceeds 256 candidates")
        salt = uuid.uuid4().hex
        rows = await _gather_cancel_on_error(
            self._encode(item["ids"], salt) for item in items
        )
        batch = collate_items([items], self.protocol.tok.pad_token_id)
        assert batch is not None
        n, length = batch["input_ids"].shape
        dtype = self.head.type_emb.weight.dtype
        hidden = torch.zeros(
            (n, length, self.head.type_emb.embedding_dim),
            device=self.device,
            dtype=dtype,
        )
        for index, ((states, _), item) in enumerate(zip(rows, items)):
            hidden[index, : len(item["ids"])] = states.to(self.device, dtype=dtype)
        autocast = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if self.device.type == "cuda"
            else nullcontext()
        )
        with torch.inference_mode(), autocast:
            readout = (
                self.graph_head
                if self.graph_head is not None
                and len(ids) == 1
                and payload.questions[ids[0]]["type"] == "choice"
                else self.head
            )
            logits, action = readout(
                hidden,
                batch["attention_mask"].to(self.device),
                batch["marker_pos"].to(self.device),
                batch["marker_mask"].to(self.device),
                batch["qtype"].to(self.device),
            )
        if not torch.isfinite(logits).all() or not torch.isfinite(action).all():
            raise RuntimeError("Laya returned nonfinite decision scores")
        answers = self.protocol._decode_answers(
            logits.float().cpu().numpy(),
            torch.softmax(action.float(), -1).cpu().numpy(),
            items,
            ids,
            internal,
            0,
        )
        if any(
            not math.isfinite(probability)
            for answer in answers.values()
            for probability in answer.get("probabilities", {}).values()
        ):
            raise RuntimeError("Laya returned nonfinite probabilities")
        return {
            "model": self.model_id,
            "answers": answers,
            "usage": {
                "input_tokens": int(batch["attention_mask"].sum()),
                "output_tokens": 0,
            },
            "metadata": {
                "cached_tokens": sum(cached for _, cached in rows),
                "inference_seconds": time.perf_counter() - started,
            },
        }
