"""Apple Silicon execution of the shared decision compiler and readout."""

import json
import time
from concurrent.futures import CancelledError
from pathlib import Path
from threading import Event

import mlx.core as mx
import torch
from mlx_lm import load

from .decision import DecisionService, _answer, _readout_positions
from .mac import _MacService


class MacDecisionService(_MacService):
    def __init__(self, checkpoint: Path, model_id: str, executor):
        super().__init__(executor)
        config = json.loads((checkpoint / "config.json").read_text())
        kind = config.get("model_type")
        overrides = {}
        if kind in ("qwen3_5_text", "qwen3_5_moe_text", "qwen3_5", "qwen3_5_moe"):
            overrides["model_type"] = "qwen3_5_moe" if "moe" in kind else "qwen3_5"
        elif kind == "qwen3":
            overrides["rope_theta"] = (
                config.get("rope_theta") or config["rope_parameters"]["rope_theta"]
            )
        elif kind != "qwen2":
            raise ValueError("this decision backbone is not supported on Apple Silicon")
        mx.set_cache_limit(512 * 1024**2)
        self.model, _ = load(str(checkpoint), model_config=overrides)
        self.reader = DecisionService(
            None, checkpoint, model_id, config.get("max_position_embeddings", 32768)
        )
        if self.reader.compiler.protocol == "thisthat_slot_v1":
            from mlx.utils import tree_map_with_path

            keep = self.model.cast_predicate
            self.model.update(
                tree_map_with_path(
                    lambda path, value: (
                        value.astype(mx.float16)
                        if keep(path) and mx.issubdtype(value.dtype, mx.floating)
                        else value
                    ),
                    self.model.parameters(),
                )
            )
        self.model_id = model_id

    def _systemone(self, payload, cancelled: Event):
        started = time.perf_counter()
        if cancelled.is_set():
            raise CancelledError()
        prepared, state_tokens = self.reader.compiler.compile(payload)
        answers = {}
        input_tokens = state_tokens
        compute_tokens = 0
        encoded = {}
        readouts = _readout_positions(prepared)
        for identifier, kind, keys, descriptions, isolated, rows in prepared:
            probabilities = []
            for ids, positions, count in rows:
                if cancelled.is_set():
                    raise CancelledError()
                key = tuple(ids)
                slots = readouts[key]
                if key not in encoded:
                    selected = self.model.model(mx.array([ids]))[
                        0, mx.array(list(slots))
                    ]
                    mx.eval(selected)
                    encoded[key] = selected
                    compute_tokens += len(ids)
                hidden = encoded[key]
                positions = [slots[p] for p in positions]
                if self.reader.compiler.protocol == "thisthat_slot_v1":
                    logits = self.model.model.embed_tokens.as_linear(
                        hidden[mx.array(positions)]
                    )[
                        0,
                        mx.array(
                            self.reader.compiler.manifest["label_token_ids"][:count]
                        ),
                    ]
                    p = mx.softmax(logits.astype(mx.float32)).tolist()
                    probabilities.append(p)
                    input_tokens += len(ids) - state_tokens
                    continue
                selected = hidden[mx.array(positions)].astype(mx.float32)
                mx.eval(selected)
                p = self.reader._probabilities(
                    torch.tensor(selected.tolist()),
                    list(range(len(positions))),
                    count,
                    kind,
                )
                probabilities.append(p)
                input_tokens += len(ids) - state_tokens
            if isolated:
                mass = [p[1] for p in probabilities]
                total = sum(mass)
                p = [x / total for x in mass] if total else [1 / len(mass)] * len(mass)
            else:
                p = probabilities[0]
            answers[identifier] = _answer(kind, keys, descriptions, p)
        return {
            "model": self.model_id,
            "answers": answers,
            "usage": {"input_tokens": input_tokens, "output_tokens": 0},
            "internal_usage": {"compute_tokens": compute_tokens},
            "metadata": {"inference_seconds": time.perf_counter() - started},
        }

    async def systemone(self, payload):
        return await self._run(self._systemone, payload)

    def close(self):
        del self.model, self.reader
        mx.clear_cache()
