"""Laya runtime placement and cleanup without model downloads or tensor imports."""

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm_jev import windows_models


@pytest.mark.parametrize("load_fails", [False, True])
def test_readonly_model_store_uses_writable_temp_and_keeps_source(
    monkeypatch, tmp_path, load_fails
):
    source = tmp_path / "readonly-model-store" / "checkpoint"
    source.mkdir(parents=True)
    contents = {
        "model.safetensors": b"synthetic weight fixture",
        "tokenizer/tokenizer_config.json": b'{"original":true}',
    }
    for name, value in contents.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
    writable = tmp_path / "system-temp"
    writable.mkdir()
    real_temporary_directory = tempfile.TemporaryDirectory
    observed = {}

    def temporary_directory(*args, **kwargs):
        # Windows ACLs and root-owned CI runners cannot reliably use chmod.
        # Reject the old placement explicitly, then use a writable temp root.
        if kwargs.get("dir") == source.parent:
            raise PermissionError("model store is read-only")
        assert kwargs.get("dir") is None
        kwargs["dir"] = writable
        return real_temporary_directory(*args, **kwargs)

    def verify(path, model):
        assert Path(path) == source and model == "test/laya"
        for name, value in contents.items():
            assert (source / name).read_bytes() == value

    class SDKAgent:
        def __init__(self, path, **kwargs):
            observed["runtime"] = Path(path)
            assert (Path(path) / "model.safetensors").read_bytes() == contents[
                "model.safetensors"
            ]
            (Path(path) / "tokenizer/tokenizer_config.json").write_text(
                '{"sdk_normalized":true}', encoding="utf-8"
            )
            self.device = "cpu"
            if load_fails:
                raise RuntimeError("synthetic SDK failure")

    def unavailable_hardlink(*args):
        # System temp may be on another drive than the model store.
        raise OSError("cross-volume hardlink unavailable")

    monkeypatch.setitem(
        sys.modules,
        "vllm_jev.laya_export",
        SimpleNamespace(
            MODELS={"test/laya": None},
            SOURCE_FILES=tuple(contents),
            verify_source=verify,
        ),
    )
    monkeypatch.setitem(sys.modules, "laya", SimpleNamespace(Agent=SDKAgent))
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(set_num_threads=lambda n: None, float32=0)
    )
    monkeypatch.setattr(
        windows_models.tempfile, "TemporaryDirectory", temporary_directory
    )
    monkeypatch.setattr(windows_models.os, "link", unavailable_hardlink)
    if load_fails:
        with pytest.raises(RuntimeError, match="synthetic SDK failure"):
            windows_models.load_agent(source, "test/laya", threads=1)
    else:
        agent = windows_models.load_agent(source, "test/laya", threads=1)
        assert observed["runtime"].parent == writable
        assert observed["runtime"].exists()
        agent.close()
        agent.close()
    assert not observed["runtime"].exists()
    verify(source, "test/laya")
