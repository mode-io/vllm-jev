"""Device portability and endpoint compatibility without pretending to run NPU."""

import asyncio
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from vllm_jev import backend
from vllm_jev.ascend_server import EngineView, install_endpoint_bridge
from vllm_jev.token_score import project_token_scores


@pytest.mark.parametrize("name", ["cpu", "cuda", "ascend", "auto"])
def test_explicit_backend_preserved(monkeypatch, name):
    monkeypatch.setattr(backend.sys, "platform", "linux")
    if name == "auto":
        monkeypatch.setattr(backend.Path, "exists", lambda p: False)
    assert backend.resolve_backend(name, {}) == name


def test_bad_backend_rejected():
    with pytest.raises(ValueError):
        backend.resolve_backend("nvidia-but-maybe-cpu", {})


def test_ascend_does_not_select_cuda(monkeypatch):
    monkeypatch.setenv(backend.BACKEND_ENV, "ascend")
    monkeypatch.setattr(backend.sys, "platform", "linux")
    calls = []
    fake = SimpleNamespace(
        npu=SimpleNamespace(is_available=lambda: True),
        cuda=SimpleNamespace(is_available=lambda: calls.append("cuda") or True),
        device=lambda value: value,
    )
    monkeypatch.setitem(sys.modules, "torch", fake)
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())
    assert backend.readout_device() == "npu:0"
    assert not calls
    assert backend.export_device() == "cpu"


def test_unavailable_explicit_npu_fails_instead_of_cpu_fallback(monkeypatch):
    monkeypatch.setenv(backend.BACKEND_ENV, "ascend")
    monkeypatch.setattr(backend.sys, "platform", "linux")
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(npu=SimpleNamespace(is_available=lambda: False)),
    )
    with pytest.raises(RuntimeError, match="no usable NPU"):
        backend.readout_device()


def test_cuda_and_cpu_fallback(monkeypatch):
    monkeypatch.setenv(backend.BACKEND_ENV, "auto")
    monkeypatch.setattr(backend.Path, "exists", lambda p: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert backend.readout_device().type == "cpu"
    monkeypatch.setenv(backend.BACKEND_ENV, "cuda")
    with pytest.raises(RuntimeError, match="no usable CUDA"):
        backend.readout_device()


@pytest.mark.parametrize("shape", [(256, 16), (513, 31), (1024, 128)])
def test_portable_projection_matches_layer_norm(shape):
    x = torch.randn(shape).bfloat16()
    w, b = torch.randn(1, shape[1]), torch.randn(1)
    actual = project_token_scores(x, w, b)
    expected = F.linear(F.layer_norm(x.float(), (shape[1],)), w, b)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    assert actual.shape == (shape[0], 1)
    assert actual.dtype == torch.float32


def test_projection_handles_strided_and_constant_inputs():
    x = torch.ones(17, 600).T
    b = torch.tensor([2.0])
    torch.testing.assert_close(
        project_token_scores(x, torch.ones(1, 17), b), b.expand(600, 1)
    )


def test_short_pooling_rows_keep_original_shape():
    x = torch.randn(7, 16)
    assert project_token_scores(x, torch.ones(1, 16), torch.zeros(1)) is x


@pytest.mark.parametrize(
    "native,plugin,expected",
    [
        ("0.22.1", "0.22.1rc1", "vllm_jev.ascend_server"),
        ("0.22.1+empty", "0.22.1rc1+local", "vllm_jev.ascend_server"),
        ("0.30.0", "0.30.0rc1", "vllm.entrypoints.cli.main"),
    ],
)
def test_runtime_pair_selection(monkeypatch, native, plugin, expected):
    import importlib.metadata

    from vllm_jev import ascend_runtime

    row = ascend_runtime.RUNTIMES[native.split("+", 1)[0]]
    packages = {**row, "vllm": native, "vllm-ascend": plugin}
    monkeypatch.setattr(importlib.metadata, "version", lambda n: packages[n])
    monkeypatch.setattr(ascend_runtime, "cann_version", lambda: row["cann"])
    assert backend.server_module("ascend") == expected


def test_mixed_ascend_versions_rejected(monkeypatch):
    import importlib.metadata

    monkeypatch.setattr(importlib.metadata, "version", lambda n: "0.29.0")
    with pytest.raises(ValueError, match="Uncovered Ascend"):
        backend.server_module("ascend")
    assert backend.server_module("cuda") == "vllm.entrypoints.cli.main"


def test_legacy_bridge_preserves_engine_lifecycle_and_attaches_once():
    events = []

    async def original_init(engine, state, args, *rest, **kwargs):
        events.append("native_init")

    server = SimpleNamespace(
        build_app=lambda *a, **k: SimpleNamespace(), init_app_state=original_init
    )

    async def plugin_init(engine, state, args):
        assert engine.model_config == "config"
        assert engine.encode == "encoder"
        events.append("plugin_init")

    plugin = SimpleNamespace(
        attach_router=lambda app: events.append("attach"), init_state=plugin_init
    )
    install_endpoint_bridge(server, plugin)
    first = server.build_app
    install_endpoint_bridge(server, plugin)
    assert server.build_app is first
    server.build_app()
    engine = SimpleNamespace(
        vllm_config=SimpleNamespace(model_config="config"), encode="encoder"
    )
    asyncio.run(server.init_app_state(engine, SimpleNamespace(), SimpleNamespace()))
    assert events == ["attach", "native_init", "plugin_init"]
    assert not hasattr(engine, "model_config")
    assert EngineView(engine).encode == "encoder"


def test_ascend_cli_forwards_platform_plugin_and_uses_legacy_bridge(
    monkeypatch, tmp_path
):
    import importlib.metadata

    from vllm_jev import cli

    monkeypatch.setattr(cli.sys, "platform", "linux")
    monkeypatch.delenv("VLLM_JEV_BACKEND", raising=False)
    monkeypatch.delenv("VLLM_JEV_LAYA_CUDA_GRAPH", raising=False)
    monkeypatch.delenv("VLLM_JEV_BF16_MATMUL", raising=False)
    monkeypatch.delenv("VLLM_JEV_ONLINE_PREFIX_CACHE", raising=False)
    from vllm_jev import ascend_runtime

    packages = {**ascend_runtime.RUNTIMES["0.22.1"], "vllm": "0.22.1"}
    monkeypatch.setattr(importlib.metadata, "version", lambda n: packages[n])
    monkeypatch.setattr(ascend_runtime, "cann_version", lambda: "9.0.0")
    checkpoint = tmp_path / "export"
    checkpoint.mkdir()
    (checkpoint / "jev_manifest.json").write_text("{}")
    monkeypatch.setattr(
        cli.sys, "argv", ["vllm-jev", "serve", str(checkpoint), "--backend", "ascend"]
    )
    monkeypatch.setattr(cli.subprocess, "run", lambda *a, **k: None)
    captured = {}
    monkeypatch.setattr(
        cli.os,
        "execvpe",
        lambda exe, command, environment: captured.update(
            command=command, environment=environment
        ),
    )
    cli.main()
    assert captured["command"][2] == "vllm_jev.ascend_server"
    assert "--backend" not in captured["command"]
    assert "--async-scheduling" not in captured["command"]
    assert "--no-async-scheduling" in captured["command"]
    assert "--enforce-eager" in captured["command"]
    assert "ascend" in captured["environment"]["VLLM_PLUGINS"].split(",")
    assert captured["environment"]["VLLM_JEV_BACKEND"] == "ascend"


def test_ascend_radixflow_rejects_before_loading_a_gpu_adapter(monkeypatch):
    from argparse import Namespace

    from vllm_jev.radixflow.launch import configure

    monkeypatch.setattr(backend.sys, "platform", "linux")
    args = Namespace(radixflow=True, radixflow_policy=None)
    with pytest.raises(ValueError, match="NPU state-copy"):
        configure(args, [], {backend.BACKEND_ENV: "ascend"})
