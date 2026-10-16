"""CPU dispatch and HTTP boundary checks without pretrained weights."""

import sys
from pathlib import Path

import pytest

MODEL = "Mapika/decider-0.8b"
MODELS = [MODEL, "Mapika/decider-2b"]


@pytest.mark.parametrize("model", MODELS)
def test_decider_dispatch_and_marked_source_fail_closed(monkeypatch, tmp_path, model):
    from vllm_jev import windows_decider, windows_models

    assert windows_models.protocol_for(model) == "decider_slot_v1"
    monkeypatch.setattr(windows_decider, "prepare", lambda model, workspace: tmp_path)
    assert windows_models.prepare(model, tmp_path) == tmp_path
    (tmp_path / "decider_config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"unverified")

    def reject(path):
        raise ValueError("unverified Decider source")

    monkeypatch.setattr(windows_decider, "identify_source", reject)
    monkeypatch.setattr(
        windows_models,
        "sha256",
        lambda path: pytest.fail("marked source bypassed Decider verification"),
    )
    with pytest.raises(ValueError, match="unverified Decider"):
        windows_models.identify_source(tmp_path)


class RecordingAgent:
    def __init__(self):
        self.closed = False
        self.calls = []

    def system_one(self, state, questions):
        self.calls.append((state, questions))
        return {"answers": {"q": {"type": "noul", "noul": 0.75}}}

    def close(self):
        self.closed = True


@pytest.mark.parametrize("model", MODELS)
def test_decider_http_identity_validation_and_lifecycle(monkeypatch, tmp_path, model):
    from fastapi.testclient import TestClient

    from vllm_jev import windows_decider, windows_qwen

    agent = RecordingAgent()
    loads = []

    def load(path, model, threads):
        loads.append((path, model, threads))
        return agent

    monkeypatch.setattr(windows_decider, "load_agent", load)
    app = windows_qwen.create_app(tmp_path, model, threads=2)
    assert not loads
    payload = {
        "state": {"message": "请退款"},
        "questions": {"q": {"type": "noul", "instructions": "Refund?"}},
    }
    with TestClient(app) as client:
        assert loads == [(tmp_path, model, 2)]
        assert client.get("/health").json() == {
            "status": "ok",
            "backend": "decider-pytorch",
            "device": "cpu",
            "experimental": True,
            "model": model,
            "revision": windows_decider.MODELS[model]["revision"],
        }
        result = client.post("/v1/systemone", json=payload)
        assert result.status_code == 200, result.text
        assert result.json()["model"] == model
        for extra in (
            {"model": "unloaded/model"},
            {"cache_salt": "unsupported"},
            {"images": ["https://invalid.example/image.png"]},
        ):
            assert (
                client.post("/v1/systemone", json={**payload, **extra}).status_code
                == 422
            )
        assert len(agent.calls) == 1
    assert agent.closed


@pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("model", MODELS)
def test_unified_cli_routes_decider_without_vllm(monkeypatch, tmp_path, local, model):
    from vllm_jev import cli, windows_models, windows_qwen

    source = tmp_path / "中文 checkpoint with spaces"
    source.mkdir()
    events = []
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(windows_models, "identify_source", lambda path: model)
    monkeypatch.setattr(windows_models, "prepare", lambda model, workspace: source)
    monkeypatch.setattr(
        windows_qwen, "serve", lambda *args, **kwargs: events.append((args, kwargs))
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["vllm-jev", "serve", str(source) if local else model, "--cpu-threads", "1"],
    )
    cli.main()
    assert events == [
        ((source, model), {"host": "127.0.0.1", "port": 8795, "threads": 1})
    ]


def test_decision_compiler_does_not_import_vllm():
    import subprocess

    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from vllm_jev.decision import DecisionCompiler; "
            "assert 'vllm' not in sys.modules; assert 'mlx' not in sys.modules",
        ],
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
