"""Apple Silicon serving with Laya's published MPS runtime."""

import json
import shutil
import tempfile
import time
from concurrent.futures import CancelledError, ThreadPoolExecutor
from pathlib import Path
from threading import Event

from huggingface_hub import snapshot_download

from .endpoint import SystemOneRequest
from .laya_export import MODELS, SOURCE_FILES, verify_source
from .laya_template import ContextOrderMixin
from .mac import _MacService


def _verify_source(path: Path, model_id: str) -> None:
    verify_source(path, model_id)


def identify_source(path: Path) -> str:
    from .checkpoint import sha256

    weight_hash = sha256(path / "model.safetensors")
    for model_id, (_, expected, _, _) in MODELS.items():
        if weight_hash == expected:
            _verify_source(path, model_id)
            return model_id
    raise ValueError("unrecognized Laya checkpoint")


def prepare(model_id: str, workspace: Path) -> Path:
    revision = MODELS[model_id][0]
    output = workspace / "checkpoint" / model_id
    if output.exists():
        _verify_source(output, model_id)
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".laya-mac-", dir=output.parent))
    try:
        snapshot_download(
            repo_id=model_id,
            revision=revision,
            token=False,
            local_dir=temporary,
            allow_patterns=SOURCE_FILES,
        )
        _verify_source(temporary, model_id)
        temporary.replace(output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return output


class MacLayaService(_MacService):
    def __init__(self, checkpoint: Path, model_id: str, executor: ThreadPoolExecutor):
        super().__init__(executor)
        from laya import Agent

        class TemplateAgent(ContextOrderMixin, Agent):
            pass

        _verify_source(checkpoint, model_id)
        self.agent = TemplateAgent(str(checkpoint), device="mps", fast=False)
        if self.agent.device.type != "mps":
            raise RuntimeError("Laya requires an available Apple Metal device")
        self.model_id = model_id

    def _systemone(self, payload: SystemOneRequest, cancelled: Event) -> dict:
        if cancelled.is_set():
            raise CancelledError()
        if payload.model not in (None, "vllm-jev", "laya", self.model_id):
            raise ValueError("requested model is not loaded")
        from laya.common import serialize_state

        if not isinstance(payload.state, (str, dict, list)):
            raise ValueError("Laya state must be text or JSON")
        if len(serialize_state(payload.state)) > 50_000:
            raise ValueError("Laya state exceeds 50,000 characters")
        if len(payload.questions) > 64:
            raise ValueError("System One exceeds 64 questions")
        if len(json.dumps(payload.questions, ensure_ascii=False)) > 2_000_000:
            raise ValueError("Laya questions exceed 2 MiB")
        candidates = 0
        for question in payload.questions.values():
            if question.get("type") not in ("choice", "noul", "score"):
                raise ValueError("question type must be choice, noul or score")
            criteria = question.get("criteria")
            if (
                question.get("type") == "choice"
                and isinstance(criteria, list)
                and any(isinstance(value, (dict, list, set)) for value in criteria)
            ):
                raise ValueError("choice labels must be scalar values")
            if question.get("type") == "noul":
                candidates += 2
            elif isinstance(criteria, (dict, list)):
                candidates += len(criteria)
            else:
                candidates += 1
        if candidates > 256:
            raise ValueError("System One exceeds 256 candidates")
        started = time.perf_counter()
        result = self.agent.system_one(payload.state, payload.questions)
        result["model"] = self.model_id
        result["metadata"] = {"inference_seconds": time.perf_counter() - started}
        return result

    async def systemone(self, payload: SystemOneRequest) -> dict:
        return await self._run(self._systemone, payload)

    def close(self) -> None:
        import torch

        del self.agent
        torch.mps.empty_cache()
