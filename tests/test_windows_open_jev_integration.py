"""Windows 2B routing and HTTP contracts, using recording agents, no weights."""

import copy
import subprocess
import sys
from pathlib import Path

import pytest

from vllm_jev.decision_template import ENVIRONMENT, current_template

ROOT = Path(__file__).resolve().parents[1]
MODEL = "ZefanCai/Open-Jev-2B"
UNSUPPORTED = "ZefanCai/Open-Jev-9B"
TINY = "lostargon/Tiny-Jev"
OPENJEV_06B = "IamBusy/OpenJev-0.6B"
LAYA = "convaiinnovations/laya-multilingual"


@pytest.fixture(autouse=True)
def isolated_template(monkeypatch):
    monkeypatch.delenv(ENVIRONMENT, raising=False)
    current_template.cache_clear()
    yield
    current_template.cache_clear()


@pytest.fixture
def local_source(tmp_path):
    source = tmp_path / "中文 checkpoint with spaces"
    marker = source / "package/checkpoint/model.json"
    marker.parent.mkdir(parents=True)
    marker.write_text("{}", encoding="utf-8")
    return source


class RecordingAgent:
    def __init__(self):
        self.calls = []
        self.closed = False

    def system_one(self, state, questions):
        self.calls.append((copy.deepcopy(state), copy.deepcopy(questions)))
        return {
            "answers": {"observed": {"type": "noul", "noul": 0.5}},
            "usage": {"input_tokens": 1, "output_tokens": 0},
        }

    def close(self):
        self.closed = True


def budget_questions(choice_count=255, noul_count=1):
    questions = {
        "choice": {
            "type": "choice",
            "instructions": "Choose a candidate.",
            "criteria": {f"candidate_{index}": None for index in range(choice_count)},
        }
    }
    questions.update(
        {
            f"noul_{index}": {"type": "noul", "instructions": "Is this correct?"}
            for index in range(noul_count)
        }
    )
    return questions


def test_dispatch_protocol_and_prepare_use_pinned_2b_adapter(monkeypatch, tmp_path):
    from vllm_jev import windows_models, windows_open_jev

    assert windows_models.protocol_for(MODEL) == "open_jev_choice"
    calls = []
    expected = tmp_path / "checkpoint/ZefanCai/Open-Jev-2B"

    def prepare(model_id, workspace):
        calls.append((model_id, workspace))
        return expected

    monkeypatch.setattr(windows_open_jev, "prepare", prepare)
    monkeypatch.setattr(
        windows_models,
        "snapshot_download",
        lambda **kwargs: pytest.fail("2B preparation entered the Laya downloader"),
    )
    assert windows_models.prepare(MODEL, tmp_path) == expected
    assert calls == [(MODEL, tmp_path)]


@pytest.mark.parametrize("valid", [True, False])
def test_identification_delegates_to_strict_2b_verifier(
    monkeypatch, local_source, valid
):
    from vllm_jev import windows_models, windows_open_jev

    # A misleading legacy weight name must never bypass the 2B source verifier.
    (local_source / "model.safetensors").write_bytes(b"not a verified release")
    calls = []

    def verify(path, model_id=MODEL):
        calls.append((path, model_id))
        if not valid:
            raise ValueError("synthetic pinned 2B checksum mismatch")
        return {}

    monkeypatch.setattr(windows_open_jev, "verify_source", verify)
    monkeypatch.setattr(
        windows_models,
        "sha256",
        lambda path: pytest.fail("2B marker fell through to legacy weight detection"),
    )
    if valid:
        assert windows_models.identify_source(local_source) == MODEL
    else:
        with pytest.raises(ValueError, match="pinned 2B checksum"):
            windows_models.identify_source(local_source)
    assert calls == [(local_source.resolve(), MODEL)]


def test_unsupported_9b_never_prepares_or_loads(monkeypatch, tmp_path):
    from vllm_jev import windows_models, windows_open_jev, windows_qwen

    def forbidden(*args, **kwargs):
        pytest.fail("unsupported 9B reached preparation or model loading")

    monkeypatch.setattr(windows_open_jev, "prepare", forbidden)
    monkeypatch.setattr(windows_open_jev, "load_agent", forbidden)
    monkeypatch.setattr(windows_models, "snapshot_download", forbidden)
    with pytest.raises(ValueError, match="supports only"):
        windows_models.protocol_for(UNSUPPORTED)
    with pytest.raises(ValueError):
        windows_models.prepare(UNSUPPORTED, tmp_path)
    with pytest.raises(ValueError, match="unsupported"):
        windows_qwen.create_app(tmp_path, UNSUPPORTED, agent_factory=forbidden)


@pytest.mark.parametrize(
    "model,choice_count,noul_count,status",
    [
        (MODEL, 255, 1, 200),
        (MODEL, 255, 2, 422),
        (TINY, 255, 1, 422),
        (TINY, 254, 1, 200),
        (OPENJEV_06B, 255, 1, 422),
        (LAYA, 255, 1, 422),
    ],
)
def test_http_candidate_budget_uses_the_selected_protocol(
    model, choice_count, noul_count, status
):
    from fastapi.testclient import TestClient

    from vllm_jev import windows_laya, windows_qwen

    agent = RecordingAgent()
    factory = windows_laya.create_app if model == LAYA else windows_qwen.create_app
    app = factory(Path("unused-checkpoint"), model, agent_factory=lambda: agent)
    questions = budget_questions(choice_count, noul_count)
    with TestClient(app) as client:
        response = client.post(
            "/v1/systemone",
            json={
                "model": model,
                "state": {"message": "请判断"},
                "questions": questions,
            },
        )
        assert response.status_code == status, response.text
        if status == 200:
            assert agent.calls == [({"message": "请判断"}, questions)]
            assert response.json()["model"] == model
        else:
            assert "256" in response.json()["detail"]
            assert not agent.calls
        assert client.get("/health").status_code == 200
    assert agent.closed


def test_http_2b_uses_actual_adapter_loader_and_identity(monkeypatch, local_source):
    from fastapi.testclient import TestClient

    from vllm_jev import windows_open_jev, windows_qwen

    agent = RecordingAgent()
    loads = []

    def load(path, model_id, threads):
        loads.append((path, model_id, threads))
        return agent

    monkeypatch.setattr(windows_open_jev, "load_agent", load)
    app = windows_qwen.create_app(local_source, MODEL, threads=2)
    assert not loads  # Initialization belongs to lifespan, not module/app import.
    with TestClient(app) as client:
        assert loads == [(local_source, MODEL, 2)]
        health = client.get("/health").json()
        assert health == {
            "status": "ok",
            "backend": "open-jev-pytorch",
            "device": "cpu",
            "experimental": True,
            "model": MODEL,
            "revision": windows_open_jev.REVISION,
        }
        assert client.get("/v1/models").json()["data"] == [
            {"id": MODEL, "object": "model"}
        ]
        response = client.post(
            "/v1/systemone",
            json={
                "state": "state",
                "questions": {"q": {"type": "noul", "instructions": "Yes?"}},
            },
        )
        assert response.status_code == 200
        assert response.json()["metadata"]["backend"] == "open-jev-pytorch"
        assert response.json()["metadata"]["revision"] == windows_open_jev.REVISION
    assert agent.closed


@pytest.mark.parametrize("local_path", [False, True])
def test_cli_routes_2b_model_id_and_unicode_source(
    monkeypatch, tmp_path, local_source, local_path
):
    # Import platform-sensitive existing Laya dependencies before simulating win32.
    from vllm_jev import (
        cli,
        windows_laya,
        windows_models,
        windows_open_jev,
        windows_qwen,
    )

    prepared, verified, served = [], [], []
    monkeypatch.setattr(cli.sys, "platform", "win32")
    monkeypatch.setattr(cli.os, "cpu_count", lambda: 2)
    monkeypatch.setattr(
        cli.sys,
        "argv",
        [
            "vllm-jev",
            "serve",
            str(local_source) if local_path else MODEL,
            "--workspace",
            str(tmp_path),
            "--protocol",
            "open_jev_choice",
            "--host",
            "127.0.0.1",
            "--port",
            "18798",
        ],
    )

    def verify(path, model_id=MODEL):
        verified.append((path, model_id))
        return {}

    def prepare(model_id, workspace):
        prepared.append((model_id, workspace))
        return local_source

    monkeypatch.setattr(windows_open_jev, "verify_source", verify)
    monkeypatch.setattr(windows_models, "prepare", prepare)
    monkeypatch.setattr(
        windows_qwen,
        "serve",
        lambda path, model_id, **kwargs: served.append((path, model_id, kwargs)),
    )
    monkeypatch.setattr(
        windows_laya, "serve", lambda *a, **kw: pytest.fail("2B routed into Laya")
    )
    monkeypatch.setattr(
        cli.subprocess,
        "run",
        lambda *a, **kw: pytest.fail("native backend subprocess launched"),
    )
    cli.main()
    assert served == [
        (local_source, MODEL, {"host": "127.0.0.1", "port": 18798, "threads": 2})
    ]
    assert prepared == ([] if local_path else [(MODEL, tmp_path)])
    assert verified == ([(local_source.resolve(), MODEL)] if local_path else [])


@pytest.mark.parametrize(
    "target,protocol", [(UNSUPPORTED, "auto"), (MODEL, "tiny_jev_marker")]
)
def test_cli_rejects_unsupported_model_or_wrong_protocol_before_loading(
    monkeypatch, tmp_path, target, protocol
):
    from vllm_jev import cli, windows_models, windows_qwen

    monkeypatch.setattr(cli.sys, "platform", "win32")
    monkeypatch.setattr(
        cli.sys,
        "argv",
        [
            "vllm-jev",
            "serve",
            target,
            "--workspace",
            str(tmp_path),
            "--protocol",
            protocol,
        ],
    )

    def forbidden(*args, **kwargs):
        pytest.fail("invalid model/protocol reached preparation or serving")

    monkeypatch.setattr(windows_models, "prepare", forbidden)
    monkeypatch.setattr(windows_qwen, "serve", forbidden)
    with pytest.raises(SystemExit) as failure:
        cli.main()
    assert failure.value.code == 2


def test_2b_app_and_http_need_no_framework_or_external_sdk():
    code = r"""
import builtins
import sys
from pathlib import Path

blocked = {'torch', 'transformers', 'peft', 'jev', 'vllm', 'mlx'}
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in blocked:
        raise AssertionError('unexpected framework/SDK import: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from fastapi.testclient import TestClient
from vllm_jev import cli, windows_open_jev, windows_qwen

class Agent:
    def system_one(self, state, questions):
        return {'answers': {'q': {'type': 'noul', 'noul': 0.5}}, 'usage': {'input_tokens': 0, 'output_tokens': 0}}

app = windows_qwen.create_app(Path('unused'), windows_open_jev.MODEL_ID, agent_factory=Agent)
with TestClient(app) as client:
    assert client.get('/health').json()['backend'] == 'open-jev-pytorch'
    response = client.post('/v1/systemone', json={'state': 'state', 'questions': {'q': {'type': 'noul', 'instructions': 'Yes?'}}})
    assert response.status_code == 200, response.text
assert not blocked.intersection(sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
