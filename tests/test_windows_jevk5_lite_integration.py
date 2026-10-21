"""Check Lite's public route and early rejection without loading model weights."""

import json

import pytest
from fastapi.testclient import TestClient

from vllm_jev import cli, windows_dependencies, windows_models, windows_qwen
from vllm_jev.decision_template import ENVIRONMENT, current_template

MODEL = "alibiserikbay/JevK5-Lite"
PROTOCOL = "jevk5_lite_v1"


@pytest.fixture(autouse=True)
def isolated_template(monkeypatch):
    monkeypatch.delenv(ENVIRONMENT, raising=False)
    current_template.cache_clear()
    yield
    current_template.cache_clear()


def test_lite_marker_requires_verification_and_never_falls_through(
    tmp_path, monkeypatch
):
    (tmp_path / "lite_config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"not verified")
    monkeypatch.setattr(
        windows_models,
        "sha256",
        lambda path: pytest.fail("must not fall through to another model"),
    )
    with pytest.raises((ValueError, OSError)):
        windows_models.identify_source(tmp_path)


@pytest.mark.parametrize("local", [False, True])
def test_installed_cli_route_uses_modern_profile(monkeypatch, tmp_path, local):
    source = tmp_path / "中文 checkpoint with spaces"
    source.mkdir()
    calls = []
    requirements = windows_dependencies.PROFILES["modern"].requirements
    monkeypatch.setattr(cli.sys, "platform", "win32")
    monkeypatch.setattr(windows_models, "identify_source", lambda path: MODEL)

    def version(package):
        calls.append(("version", package))
        return requirements[package]

    def prepare(model, workspace):
        assert len(calls) == len(requirements)
        calls.append(("prepare", model, workspace))
        return source

    monkeypatch.setattr(windows_dependencies.importlib.metadata, "version", version)
    monkeypatch.setattr(windows_models, "prepare", prepare)
    monkeypatch.setattr(
        windows_qwen, "serve", lambda *a, **kw: calls.append(("serve", a, kw))
    )
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
            "2",
            "--port",
            "8798",
        ],
    )
    cli.main()
    assert calls[: len(requirements)] == [("version", p) for p in requirements]
    assert calls[len(requirements) : -1] == (
        [] if local else [("prepare", MODEL, tmp_path.resolve())]
    )
    assert calls[-1] == (
        "serve",
        (source.resolve(), MODEL),
        {"host": "127.0.0.1", "port": 8798, "threads": 2},
    )


@pytest.mark.parametrize(
    "setting",
    [
        {"layout": "instructions-first"},
        {"state_template": "State: {state}"},
        {"instruction_template": "Task: {instructions}"},
        {"context_order": ["state", "instructions", "criteria"]},
    ],
)
def test_custom_templates_fail_before_download_or_model_load(
    monkeypatch, tmp_path, capsys, setting
):
    monkeypatch.setenv(ENVIRONMENT, json.dumps(setting))
    current_template.cache_clear()
    monkeypatch.setattr(cli.sys, "platform", "win32")
    monkeypatch.setattr(cli.sys, "argv", ["vllm-jev", "serve", MODEL])
    monkeypatch.setattr(
        windows_models, "prepare", lambda *a: pytest.fail("must not download")
    )
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    assert "native" in capsys.readouterr().err.lower()
    with pytest.raises(ValueError, match="native"):
        windows_qwen.create_app(
            tmp_path, MODEL, agent_factory=lambda: pytest.fail("must not load")
        )


def test_legacy_environment_rejects_before_download(monkeypatch, capsys):
    monkeypatch.setattr(cli.sys, "platform", "win32")
    monkeypatch.setattr(cli.sys, "argv", ["vllm-jev", "serve", MODEL])
    monkeypatch.setattr(
        windows_dependencies.importlib.metadata,
        "version",
        windows_dependencies.PROFILES["legacy"].requirements.__getitem__,
    )
    monkeypatch.setattr(
        windows_models, "prepare", lambda *a: pytest.fail("must not download")
    )
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    assert "windows-modern" in capsys.readouterr().err


def test_thread_limit_rejected_before_download(monkeypatch, capsys):
    monkeypatch.setattr(cli.sys, "platform", "win32")
    monkeypatch.setattr(cli.os, "cpu_count", lambda: 18)
    monkeypatch.setattr(
        cli.sys, "argv", ["vllm-jev", "serve", MODEL, "--cpu-threads", "5"]
    )
    monkeypatch.setattr(
        windows_models, "prepare", lambda *a: pytest.fail("must not download")
    )
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    assert "between 1 and 4" in capsys.readouterr().err


@pytest.mark.parametrize("platform", ["linux", "darwin"])
def test_lite_protocol_is_windows_only(monkeypatch, capsys, platform):
    monkeypatch.setattr(cli.sys, "platform", platform)
    monkeypatch.setattr(
        cli.sys, "argv", ["vllm-jev", "serve", MODEL, "--protocol", PROTOCOL]
    )
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_http_dispatch_identity_media_boundary_and_lifetime(tmp_path, monkeypatch):
    from vllm_jev import windows_jevk5_lite as lite

    class Agent:
        calls = 0
        closed = False

        def system_one(self, state, questions):
            self.calls += 1
            return {"answers": {"q": {"type": "noul", "noul": 0.75}}}

        def close(self):
            self.closed = True

    agent = Agent()
    monkeypatch.setattr(lite, "load_agent", lambda *a: agent)
    app = windows_qwen.create_app(tmp_path, MODEL, threads=2)
    request = {
        "state": "文本",
        "questions": {"q": {"type": "noul", "instructions": "Is this text?"}},
    }
    with TestClient(app) as client:
        assert client.get("/health").json() == {
            "status": "ok",
            "backend": "jevk5-lite-pytorch",
            "device": "cpu",
            "experimental": True,
            "model": MODEL,
            "revision": lite.REVISION,
        }
        assert client.get("/v1/models").json()["data"] == [
            {"id": MODEL, "object": "model"}
        ]
        response = client.post("/v1/systemone", json=request)
        assert response.status_code == 200, response.text
        assert response.json()["metadata"]["revision"] == lite.REVISION
        assert agent.calls == 1
        for change in (
            {"model": "alibiserikbay/JevK5-2B"},
            {"images": ["image.png"]},
            {"videos": ["video.mp4"]},
            {"media": [{"type": "image", "uri": "image.png"}]},
            {"cache_salt": "namespace"},
        ):
            assert (
                client.post("/v1/systemone", json={**request, **change}).status_code
                == 422
            )
        assert agent.calls == 1
    assert agent.closed
