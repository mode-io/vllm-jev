"""Mica's quantized Windows route, without loading DLLs or model weights."""

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from vllm_jev import cli, windows_dependencies, windows_models, windows_qwen
from vllm_jev.decision_template import ENVIRONMENT, current_template

MODEL = "sky7350/Mica-v0.1-4B"
PROTOCOL = "mica_gguf_v1"
BACKEND = "mica-llama-cpp-q5-b11010"


@pytest.fixture(autouse=True)
def isolated_template(monkeypatch):
    monkeypatch.delenv(ENVIRONMENT, raising=False)
    current_template.cache_clear()
    yield
    current_template.cache_clear()


class RecordingAgent:
    def __init__(self):
        self.calls = []
        self.closed = False

    def system_one(self, state, questions):
        self.calls.append((copy.deepcopy(state), copy.deepcopy(questions)))
        return {
            "answers": {"q": {"type": "noul", "noul": 0.75}},
            "usage": {"input_tokens": 8, "output_tokens": 0},
            "metadata": {
                "quantization": "Q5_K_M",
                "runtime": "llama.cpp",
                "runtime_revision": "b11010",
                "n_gpu_layers": 0,
            },
        }

    def close(self):
        self.closed = True


def payload():
    return {
        "state": {"description": "这是一段中文状态"},
        "questions": {"q": {"type": "noul", "instructions": "Is this text?"}},
    }


def test_id_preparation_selects_the_quantized_adapter(monkeypatch, tmp_path):
    from vllm_jev import windows_mica

    expected = tmp_path / "checkpoint" / MODEL
    calls = []

    def prepare(model, workspace):
        calls.append((model, workspace))
        return expected

    monkeypatch.setattr(windows_mica, "prepare", prepare)
    monkeypatch.setattr(
        windows_models,
        "snapshot_download",
        lambda **kw: pytest.fail("Mica must not use Laya preparation"),
    )
    assert windows_models.protocol_for(MODEL) == PROTOCOL
    assert windows_models.prepare(MODEL, tmp_path) == expected
    assert calls == [(MODEL, tmp_path)]


@pytest.mark.parametrize("marker", ["calibration.json", "untrusted-Q4.gguf"])
def test_local_artifact_is_not_accepted_from_filename_alone(
    tmp_path, monkeypatch, marker
):
    source = tmp_path / "中文 Mica source"
    source.mkdir()
    (source / marker).write_bytes(b"unverified")
    # An original-precision file alongside a marker must not select another
    # adapter if Mica's strict bundle verification fails.
    (source / "model.safetensors").write_bytes(b"unverified BF16")
    monkeypatch.setattr(
        windows_models,
        "sha256",
        lambda path: pytest.fail("Mica verification fell through to another model"),
    )
    with pytest.raises((ValueError, OSError)):
        windows_models.identify_source(source)


def test_http_uses_real_dispatch_and_reports_quantized_runtime(monkeypatch, tmp_path):
    from vllm_jev import windows_mica

    agent = RecordingAgent()
    calls = []

    def load(path, model, threads):
        calls.append((path, model, threads))
        return agent

    monkeypatch.setattr(windows_mica, "load_agent", load)
    source = tmp_path / "中文 checkpoint with spaces"
    app = windows_qwen.create_app(source, MODEL, threads=2)
    assert calls == []
    with TestClient(app) as client:
        assert calls == [(source, MODEL, 2)]
        health = client.get("/health").json()
        assert health == {
            "status": "ok",
            "backend": BACKEND,
            "device": "cpu",
            "experimental": True,
            "model": MODEL,
            "revision": windows_mica.REVISION,
        }
        assert client.get("/v1/models").json()["data"] == [
            {"id": MODEL, "object": "model"}
        ]
        request = payload()
        response = client.post("/v1/systemone", json=request)
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["model"] == MODEL
        assert body["metadata"]["backend"] == BACKEND
        assert body["metadata"]["quantization"] == "Q5_K_M"
        assert body["metadata"]["runtime_revision"] == "b11010"
        assert body["metadata"]["n_gpu_layers"] == 0
        assert body["metadata"]["revision"] == windows_mica.REVISION
        assert agent.calls == [(request["state"], request["questions"])]
    assert agent.closed


@pytest.mark.parametrize(
    "change",
    [
        {"model": "sky7350/Mica-v0.1-4B-Q4_K_M"},
        {"cache_salt": "namespace"},
        {"images": ["data:image/png;base64,AAAA"]},
        {"videos": ["clip.mp4"]},
        {"media": [{"type": "image", "uri": "file.png"}]},
        {
            "state": {
                "messages": [
                    {"content": [{"type": "image_url", "image_url": {"url": "x"}}]}
                ]
            }
        },
        {"state": "a" * 50_001},
        {"questions": {"q": {"type": "generation", "instructions": "Write text"}}},
        {"questions": {"q": {"type": "noul"}}},
        {
            "questions": {
                "q": {"type": "choice", "instructions": "Choose", "criteria": []}
            }
        },
    ],
)
def test_shared_http_boundary_rejects_before_inference(tmp_path, change):
    agent = RecordingAgent()
    app = windows_qwen.create_app(tmp_path, MODEL, agent_factory=lambda: agent)
    with TestClient(app) as client:
        response = client.post("/v1/systemone", json={**payload(), **change})
        assert response.status_code == 422, response.text
        assert agent.calls == []
        assert client.get("/health").status_code == 200
    assert agent.closed


@pytest.mark.parametrize("local", [False, True])
def test_cli_id_and_unicode_path_keep_modern_guard(monkeypatch, tmp_path, local):
    source = tmp_path / "中文 checkpoint with spaces"
    source.mkdir()
    calls = []
    requirements = {
        **windows_dependencies.PROFILES["modern"].requirements,
        **windows_dependencies.MICA_TOKENIZER_REQUIREMENTS,
    }
    monkeypatch.setattr(cli.sys, "platform", "win32")
    monkeypatch.setattr(windows_models, "identify_source", lambda path: MODEL)

    def version(package):
        calls.append(("version", package))
        return requirements[package]

    def prepare(model, workspace):
        assert len(calls) == len(requirements)
        calls.append(("prepare", model, workspace))
        return source

    def serve(path, model, **kw):
        calls.append(("serve", path, model, kw))

    monkeypatch.setattr(windows_dependencies.importlib.metadata, "version", version)
    monkeypatch.setattr(windows_models, "prepare", prepare)
    monkeypatch.setattr(windows_qwen, "serve", serve)
    monkeypatch.setattr(
        cli.sys,
        "argv",
        [
            "vllm-jev",
            "serve",
            str(source) if local else MODEL,
            "--workspace",
            str(tmp_path),
            "--protocol",
            PROTOCOL,
            "--cpu-threads",
            "1",
            "--host",
            "127.0.0.1",
            "--port",
            "8798",
        ],
    )
    cli.main()
    assert calls[: len(requirements)] == [("version", name) for name in requirements]
    assert calls[len(requirements) : -1] == (
        [] if local else [("prepare", MODEL, tmp_path.resolve())]
    )
    assert calls[-1] == (
        "serve",
        source.resolve(),
        MODEL,
        {"host": "127.0.0.1", "port": 8798, "threads": 1},
    )


@pytest.mark.parametrize(
    "config",
    [
        {"layout": "instructions-first"},
        {"state_template": "State: {state}"},
        {"instruction_template": "Question: {instructions}"},
        {"context_order": ["instructions", "state", "criteria"]},
    ],
)
def test_custom_template_rejected_before_cli_download_or_http_load(
    monkeypatch, tmp_path, capsys, config
):
    monkeypatch.setenv(ENVIRONMENT, json.dumps(config))
    current_template.cache_clear()
    monkeypatch.setattr(cli.sys, "platform", "win32")
    monkeypatch.setattr(cli.sys, "argv", ["vllm-jev", "serve", MODEL])
    monkeypatch.setattr(
        windows_models, "prepare", lambda *a: pytest.fail("must reject before download")
    )
    monkeypatch.setattr(
        windows_qwen, "serve", lambda *a, **k: pytest.fail("must reject before serve")
    )
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    assert "native" in capsys.readouterr().err.lower()
    with pytest.raises(ValueError, match="native"):
        windows_qwen.create_app(
            tmp_path, MODEL, agent_factory=lambda: pytest.fail("must not load")
        )


def test_legacy_environment_cannot_download_or_start_mica(monkeypatch, capsys):
    monkeypatch.setattr(cli.sys, "platform", "win32")
    monkeypatch.setattr(cli.sys, "argv", ["vllm-jev", "serve", MODEL])
    monkeypatch.setattr(
        windows_dependencies.importlib.metadata,
        "version",
        lambda package: {
            **windows_dependencies.PROFILES["legacy"].requirements,
            **windows_dependencies.MICA_TOKENIZER_REQUIREMENTS,
        }[package],
    )
    monkeypatch.setattr(
        windows_models,
        "prepare",
        lambda *a: pytest.fail("wrong environment downloaded"),
    )
    monkeypatch.setattr(
        windows_qwen, "serve", lambda *a, **kw: pytest.fail("wrong environment loaded")
    )
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    assert "windows-modern" in capsys.readouterr().err


@pytest.mark.parametrize("package", ["tokenizers", "jinja2"])
def test_mica_tokenizer_version_mismatch_rejects_before_download(
    monkeypatch, capsys, package
):
    monkeypatch.setattr(cli.sys, "platform", "win32")
    monkeypatch.setattr(cli.sys, "argv", ["vllm-jev", "serve", MODEL])
    versions = {
        **windows_dependencies.PROFILES["modern"].requirements,
        **windows_dependencies.MICA_TOKENIZER_REQUIREMENTS,
        package: "999.0.0",
    }
    monkeypatch.setattr(
        windows_dependencies.importlib.metadata, "version", versions.__getitem__
    )
    monkeypatch.setattr(
        windows_models, "prepare", lambda *a: pytest.fail("must not download")
    )
    monkeypatch.setattr(
        windows_qwen, "serve", lambda *a, **k: pytest.fail("must not serve")
    )
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    assert package in capsys.readouterr().err


def test_cli_rejects_mica_thread_budget_before_downloading(monkeypatch, capsys):
    monkeypatch.setattr(cli.sys, "platform", "win32")
    monkeypatch.setattr(cli.os, "cpu_count", lambda: 18)
    monkeypatch.setattr(
        cli.sys, "argv", ["vllm-jev", "serve", MODEL, "--cpu-threads", "5"]
    )
    monkeypatch.setattr(
        windows_models, "prepare", lambda *a: pytest.fail("must reject before download")
    )
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    assert "between 1 and 4" in capsys.readouterr().err


@pytest.mark.parametrize("platform", ["linux", "darwin"])
def test_windows_quantized_protocol_is_not_advertised_on_other_platforms(
    monkeypatch, capsys, platform
):
    monkeypatch.setattr(cli.sys, "platform", platform)
    monkeypatch.setattr(
        cli.sys, "argv", ["vllm-jev", "serve", MODEL, "--protocol", PROTOCOL]
    )
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_fresh_mica_cli_and_http_dispatch_do_not_import_tensor_backends(tmp_path):
    script = r"""
import os
import sys
from pathlib import Path
from vllm_jev import cli, windows_dependencies, windows_models, windows_qwen
from vllm_jev import windows_mica
sys.platform = "win32"
before = os.environ.get("USE_TORCH")
model = windows_mica.MODEL_ID
source = Path(sys.argv[1])
windows_mica.prepare = lambda model_id, workspace: source
windows_dependencies.require_windows_dependencies = lambda model_id: None
def serve(checkpoint, model_id, **kwargs):
    app = windows_qwen.create_app(checkpoint, model_id, agent_factory=lambda: None)
    assert app is not None
    assert "torch" not in sys.modules
    assert "transformers" not in sys.modules
windows_qwen.serve = serve
sys.argv = ["vllm-jev", "serve", model, "--cpu-threads", "1"]
cli.main()
assert os.environ.get("USE_TORCH") == before
assert "torch" not in sys.modules
assert "transformers" not in sys.modules
print("no tensor backend imported")
"""
    environment = os.environ.copy()
    environment.pop(ENVIRONMENT, None)
    environment.pop("USE_TORCH", None)
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "unloaded-source")],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "no tensor backend imported" in result.stdout


def test_explicit_native_overrides_environment_before_preparation(
    monkeypatch, tmp_path
):
    from vllm_jev import windows_mica

    monkeypatch.setenv(ENVIRONMENT, json.dumps({"layout": "instructions-first"}))
    current_template.cache_clear()
    monkeypatch.setattr(cli.sys, "platform", "win32")
    monkeypatch.setattr(
        windows_dependencies, "require_windows_dependencies", lambda model: None
    )
    calls = []

    def prepare(model, workspace):
        windows_mica.validate_template()
        assert current_template().config == {"layout": "native"}
        calls.append("prepare")
        return tmp_path

    monkeypatch.setattr(windows_models, "prepare", prepare)
    monkeypatch.setattr(windows_qwen, "serve", lambda *a, **kw: calls.append("serve"))
    monkeypatch.setattr(
        cli.sys,
        "argv",
        ["vllm-jev", "serve", MODEL, "--decision-template", "native"],
    )
    cli.main()
    assert calls == ["prepare", "serve"]


@pytest.mark.parametrize(
    "model", [model for model in windows_dependencies.MODEL_PROFILES if model != MODEL]
)
def test_other_model_cli_routes_do_not_disable_torch(monkeypatch, tmp_path, model):
    from vllm_jev import windows_laya

    monkeypatch.delenv("USE_TORCH", raising=False)
    monkeypatch.setattr(cli.sys, "platform", "win32")
    monkeypatch.setattr(
        windows_dependencies, "require_windows_dependencies", lambda model: None
    )
    monkeypatch.setattr(windows_models, "prepare", lambda model, workspace: tmp_path)
    # This routing test stops before inference. Native distribution integrity is
    # exercised by each GGUF family's dedicated tests and real acceptance driver.
    for adapter in windows_models._additional_adapters():
        if model in windows_models._adapter_model_ids(adapter) and hasattr(
            adapter, "verify_runtime"
        ):
            monkeypatch.setattr(adapter, "verify_runtime", lambda: None)
    calls = []

    def serve(*args, **kwargs):
        assert "USE_TORCH" not in os.environ
        calls.append(args[1])

    monkeypatch.setattr(windows_laya, "serve", serve)
    monkeypatch.setattr(windows_qwen, "serve", serve)
    monkeypatch.setattr(cli.sys, "argv", ["vllm-jev", "serve", model])
    cli.main()
    assert calls == [model]
    assert "USE_TORCH" not in os.environ
