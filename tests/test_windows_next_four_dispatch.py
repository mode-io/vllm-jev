"""The new families keep explicit identities through the shared Windows entry."""

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from vllm_jev import windows_bruv, windows_decision_gguf, windows_rsi
from vllm_jev.decision_template import ENVIRONMENT, current_template
from vllm_jev.windows_models import protocol_for
from vllm_jev.windows_qwen import create_app

MODELS = [
    (
        windows_decision_gguf.MODEL_ID,
        windows_decision_gguf,
        "decision-llama-cpp-0.3.35",
    ),
    (
        windows_decision_gguf.MODEL_ID_4B,
        windows_decision_gguf,
        "decision-llama-cpp-0.3.35",
    ),
    (windows_bruv.MODEL_ID, windows_bruv, "bruv-pytorch"),
    (windows_rsi.MODEL_ID, windows_rsi, "rsi-pytorch"),
]


@pytest.fixture(autouse=True)
def native_template(monkeypatch):
    monkeypatch.delenv(ENVIRONMENT, raising=False)
    current_template.cache_clear()
    yield
    current_template.cache_clear()


@pytest.mark.parametrize("model,adapter,backend", MODELS)
def test_shared_service_preserves_identity_and_owned_close(
    tmp_path, model, adapter, backend
):
    calls = []
    fake = SimpleNamespace(
        close=lambda: calls.append("close"),
        system_one=lambda *_: pytest.fail("invalid input reached inference"),
    )
    app = create_app(tmp_path, model, agent_factory=lambda: fake)
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        health = response.json()
        assert health["model"] == model
        assert health["backend"] == backend
        assert health["device"] == "cpu" and health["experimental"] is True
        rejected = client.post(
            "/v1/systemone",
            json={
                "state": "x",
                "model": "unrelated/model",
                "questions": {
                    "q": {
                        "type": "choice",
                        "instructions": "Choose",
                        "criteria": ["a", "b"],
                    }
                },
            },
        )
        assert rejected.status_code == 422
        assert "requested model is not loaded" in str(rejected.json())
    assert calls == ["close"]
    assert protocol_for(model) == adapter.PROTOCOL


@pytest.mark.parametrize("model,adapter,backend", MODELS)
@pytest.mark.parametrize(
    "setting",
    ['{"layout":"instructions-first"}', '{"state_template":"Wrapped {state}"}'],
)
def test_unsupported_templates_fail_before_loading(
    monkeypatch, tmp_path, model, adapter, backend, setting
):
    monkeypatch.setenv(ENVIRONMENT, setting)
    current_template.cache_clear()
    with pytest.raises(ValueError, match="native"):
        create_app(tmp_path, model, agent_factory=lambda: pytest.fail("model loaded"))


@pytest.mark.parametrize("model,adapter,backend", MODELS)
def test_cli_routes_new_id_to_its_backend_before_any_download(
    monkeypatch, tmp_path, model, adapter, backend
):
    from vllm_jev import cli, windows_dependencies, windows_models, windows_qwen

    monkeypatch.setattr(cli.sys, "platform", "win32")
    monkeypatch.setattr(
        cli.sys, "argv", ["vllm-jev", "serve", model, "--workspace", str(tmp_path)]
    )
    seen = []
    monkeypatch.setattr(
        windows_dependencies,
        "require_windows_dependencies",
        lambda m: seen.append(("dependencies", m)),
    )
    monkeypatch.setattr(
        windows_models, "prepare", lambda m, w: seen.append(("prepare", m)) or tmp_path
    )
    monkeypatch.setattr(
        windows_qwen, "serve", lambda checkpoint, m, **kwargs: seen.append(("serve", m))
    )
    if hasattr(adapter, "verify_runtime"):
        monkeypatch.setattr(adapter, "verify_runtime", lambda: None)
    cli.main()
    assert seen == [("dependencies", model), ("prepare", model), ("serve", model)]
