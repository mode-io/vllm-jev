"""Dispatch pinned Windows sources and isolate Laya's mutable SDK runtime."""

import hashlib
import os
import shutil
import tempfile
from pathlib import Path

from filelock import FileLock
from huggingface_hub import snapshot_download

from .laya_template import ContextOrderMixin


def sha256(path: Path) -> str:
    """Hash sources without importing a tensor backend for GGUF dispatch."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _additional_adapters():
    from . import (
        windows_bruv,
        windows_decider_gguf,
        windows_decision_gguf,
        windows_jevk5,
        windows_jevk5_gguf,
        windows_jevk5_lite,
        windows_kev,
        windows_mica,
        windows_rsi,
        windows_thisthat,
    )

    return (
        windows_bruv,
        windows_decider_gguf,
        windows_decision_gguf,
        windows_jevk5,
        windows_jevk5_gguf,
        windows_jevk5_lite,
        windows_kev,
        windows_thisthat,
        windows_mica,
        windows_rsi,
    )


def _adapter_model_ids(adapter):
    """Release families share an adapter but keep distinct checkpoint identities."""
    return tuple(adapter.MODELS) if hasattr(adapter, "MODELS") else (adapter.MODEL_ID,)


def protocol_for(model_id: str) -> str:
    from . import windows_jevk5_gguf, windows_mica

    if model_id in windows_jevk5_gguf.MODELS:
        return windows_jevk5_gguf.PROTOCOL
    if model_id == windows_mica.MODEL_ID:
        return windows_mica.PROTOCOL
    from .laya_export import MODELS

    if model_id in MODELS:
        return "laya_markers_v1"
    from . import windows_tiny

    if model_id in windows_tiny.MODELS:
        return windows_tiny.PROTOCOL
    from . import windows_openjev

    if model_id == windows_openjev.MODEL_ID:
        return windows_openjev.PROTOCOL
    from . import windows_open_jev

    if model_id == windows_open_jev.MODEL_ID:
        return windows_open_jev.PROTOCOL
    from . import windows_decider

    if model_id in windows_decider.MODELS:
        return windows_decider.PROTOCOL
    for adapter in _additional_adapters():
        if model_id in _adapter_model_ids(adapter):
            return adapter.PROTOCOL
    raise ValueError(
        "Windows CPU serving supports only "
        + ", ".join(
            (
                *MODELS,
                *windows_tiny.MODELS,
                windows_openjev.MODEL_ID,
                windows_open_jev.MODEL_ID,
                *windows_decider.MODELS,
                *(
                    model
                    for adapter in _additional_adapters()
                    for model in _adapter_model_ids(adapter)
                ),
            )
        )
    )


def _require_model(model_id: str) -> None:
    from .laya_export import MODELS

    if model_id not in MODELS:
        raise ValueError("Windows CPU serving supports only " + ", ".join(MODELS))


def identify_source(checkpoint: Path) -> str:
    """Accept a verified CPU release, including the pinned Mica Q5 bundle."""
    checkpoint = Path(checkpoint).resolve()
    if any(
        (checkpoint / name).is_file()
        for name in ("decision-0.8b-Q8_0.gguf", "decision-4b-Q4_K_M.gguf")
    ):
        from .windows_decision_gguf import identify_source as identify_decision

        return identify_decision(checkpoint)
    if (checkpoint / "meta.json").is_file() and (
        checkpoint / "tower.safetensors"
    ).is_file():
        from .windows_rsi import identify_source as identify_rsi

        return identify_rsi(checkpoint)
    if (checkpoint / "decider-4b-v2.1-Q4_K_M.gguf").is_file():
        from .windows_decider_gguf import identify_source as identify_decider_gguf

        return identify_decider_gguf(checkpoint)
    if any(
        (checkpoint / name).is_file()
        for name in ("jevk5-4b-v0.3-Q5_K_M.gguf", "jevk5-9b-v0.3.3-Q5_K_M.gguf")
    ):
        from .windows_jevk5_gguf import identify_source as identify_jevk5_gguf

        return identify_jevk5_gguf(checkpoint)
    if (checkpoint / "lite_config.json").is_file():
        from .windows_jevk5_lite import identify_source as identify_lite

        return identify_lite(checkpoint)
    if (checkpoint / "jevk5_config.json").is_file():
        from .windows_jevk5 import identify_source as identify_jevk5

        return identify_jevk5(checkpoint)
    if (checkpoint / "training_config.json").is_file():
        from .windows_kev import identify_source as identify_kev

        return identify_kev(checkpoint)
    if (checkpoint / "decider_config.json").is_file():
        from .windows_decider import identify_source as identify_decider

        return identify_decider(checkpoint)
    if (checkpoint / "package/checkpoint/model.json").is_file():
        from .windows_open_jev import identify_source as identify_open_jev

        return identify_open_jev(checkpoint)
    if (checkpoint / "openjev_config.json").is_file():
        from .windows_openjev import identify_source as identify_openjev

        return identify_openjev(checkpoint)
    if (checkpoint / "calibration.json").is_file() or any(checkpoint.glob("*.gguf")):
        from .windows_mica import identify_source as identify_mica

        # These are routing hints, never proof of a supported Mica artifact.
        return identify_mica(checkpoint)
    weight = checkpoint / "model.safetensors"
    if not weight.is_file():
        raise ValueError("Windows requires an original supported source checkpoint")
    digest = sha256(weight)
    from . import windows_bruv

    if digest == windows_bruv.SOURCE_HASHES["model.safetensors"]:
        return windows_bruv.identify_source(checkpoint)
    from . import windows_thisthat

    if digest in {
        spec["source_hashes"]["model.safetensors"]
        for spec in windows_thisthat.MODELS.values()
    }:
        return windows_thisthat.identify_source(checkpoint)
    from .laya_export import MODELS, verify_source

    for model_id, (_, expected, *_) in MODELS.items():
        if digest == expected:
            verify_source(checkpoint, model_id)
            return model_id
    from . import windows_tiny

    if digest in {
        spec["source_hashes"]["model.safetensors"]
        for spec in windows_tiny.MODELS.values()
    }:
        return windows_tiny.identify_source(checkpoint)
    raise ValueError("unrecognized Windows CPU checkpoint weights")


def prepare(model_id: str, workspace: Path) -> Path:
    """Download only the pinned inference files; publish after full verification."""
    protocol = protocol_for(model_id)
    for adapter in _additional_adapters():
        if model_id in _adapter_model_ids(adapter):
            return adapter.prepare(model_id, workspace)
    if protocol == "decider_slot_v1":
        from .windows_decider import prepare as prepare_decider

        return prepare_decider(model_id, workspace)
    if protocol == "open_jev_choice":
        from .windows_open_jev import prepare as prepare_open_jev

        return prepare_open_jev(model_id, workspace)
    if protocol == "tiny_jev_marker":
        from .windows_tiny import prepare as prepare_tiny

        return prepare_tiny(model_id, workspace)
    if protocol == "openjev_branch_v03":
        from .windows_openjev import prepare as prepare_openjev

        return prepare_openjev(model_id, workspace)
    _require_model(model_id)
    from .laya_export import MODELS, SOURCE_FILES, verify_source

    output = Path(workspace).resolve() / "checkpoint" / model_id
    output.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output) + ".lock"):
        if output.exists():
            verify_source(output, model_id)
            return output
        with tempfile.TemporaryDirectory(
            prefix=".laya-download-", dir=output.parent
        ) as d:
            source = Path(d) / "source"
            snapshot_download(
                repo_id=model_id,
                revision=MODELS[model_id][0],
                token=False,
                local_dir=source,
                allow_patterns=SOURCE_FILES,
            )
            verify_source(source, model_id)
            source.replace(output)
    return output


class _CPUAgent:
    """Own the mutable SDK copy until its model has been released."""

    def __init__(self, agent, runtime):
        self.agent = agent
        self.runtime = runtime
        self.device = agent.device

    def system_one(self, state, questions):
        if self.agent is None:
            raise RuntimeError("Laya service is closed")
        return self.agent.system_one(state, questions)

    def close(self):
        self.agent = None
        self.runtime.cleanup()


def load_agent(checkpoint: Path, model_id: str, threads: int | None = None):
    """Keep SDK tokenizer normalization out of the verified source directory."""
    _require_model(model_id)
    from .laya_export import SOURCE_FILES, verify_source

    if threads is None:
        threads = min(4, os.cpu_count() or 1)
    if not 1 <= threads <= (os.cpu_count() or 1):
        raise ValueError("threads must be between 1 and the logical CPU count")
    checkpoint = Path(checkpoint).resolve()
    verify_source(checkpoint, model_id)
    import torch
    from laya import Agent

    class TemplateAgent(ContextOrderMixin, Agent):
        pass

    torch.set_num_threads(threads)
    # Checkpoints can live on read-only stores. Let tempfile select writable
    # storage (TEMP/TMP on Windows); cross-volume copies already have a fallback.
    runtime = tempfile.TemporaryDirectory(
        prefix=".laya-cpu-", ignore_cleanup_errors=True
    )
    try:
        destination = Path(runtime.name)
        for name in SOURCE_FILES:
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if name == "model.safetensors":
                try:
                    os.link(checkpoint / name, target)
                except OSError:
                    shutil.copyfile(checkpoint / name, target)
            else:
                shutil.copyfile(checkpoint / name, target)
        agent = TemplateAgent(str(destination), device="cpu", fast=False, compile=False)
        if str(agent.device) != "cpu":
            raise RuntimeError("Windows Laya requires the CPU device")
        # Keep the advertised baseline in FP32 even if the host opted into Laya AMP.
        agent.amp_enabled = False
        agent.dtype = torch.float32
        verify_source(checkpoint, model_id)
        return _CPUAgent(agent, runtime)
    except BaseException:
        runtime.cleanup()
        raise
