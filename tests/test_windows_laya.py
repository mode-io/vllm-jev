"""Windows CPU boundaries; no model download or inference is required.

HTTP checks use the real app and a recording synchronous agent. They exercise
validation, dispatch and cancellation, not model quality or GPU equivalence.
"""

import asyncio
import copy
import hashlib
import importlib.util
import os
import subprocess
import sys
import threading
import tomllib
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from packaging.markers import default_environment
from packaging.requirements import Requirement

from vllm_jev.decision_template import ENVIRONMENT, current_template

ROOT = Path(__file__).resolve().parents[1]
MODEL = "convaiinnovations/laya-multilingual"
QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "Choose the responsible department.",
        "criteria": {"billing": "Payments", "technical": "Software faults"},
    },
    "refund": {"type": "noul", "instructions": "Is a refund requested?"},
    "urgency": {
        "type": "score",
        "instructions": "How urgent is it?",
        "criteria": ["Can wait", "Soon", "Today"],
    },
}
RESULT = {
    "answers": {
        "department": {
            "type": "choice",
            "choice": "billing",
            "probabilities": {"billing": 0.8, "technical": 0.2},
        },
        "refund": {"type": "noul", "noul": 0.9},
        "urgency": {
            "type": "score",
            "score": 1.4,
            "probabilities": {"0": 0.1, "1": 0.4, "2": 0.5},
        },
    },
    "usage": {"input_tokens": 42, "output_tokens": 0},
}


@pytest.fixture(autouse=True)
def isolated_template(monkeypatch):
    monkeypatch.delenv(ENVIRONMENT, raising=False)
    current_template.cache_clear()
    yield
    current_template.cache_clear()


class RecordingAgent:
    def __init__(self):
        self.calls = []

    def system_one(self, state, questions):
        self.calls.append((copy.deepcopy(state), copy.deepcopy(questions)))
        return copy.deepcopy(RESULT)


def payload(**updates):
    return {
        "state": "I was charged twice. Please refund the duplicate payment.",
        "questions": copy.deepcopy(QUESTIONS),
        **updates,
    }


def app_for(agent, **options):
    from vllm_jev.windows_laya import create_app

    return create_app(
        Path("unused-test-checkpoint"), MODEL, agent_factory=lambda: agent, **options
    )


async def request_once(agent, body):
    app = app_for(agent)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            return await client.post("/v1/systemone", json=body)


def test_windows_dependency_resolution_excludes_native_backends():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    environment = default_environment()
    environment.update(sys_platform="win32", platform_system="Windows")
    requirements = [Requirement(value) for value in project["project"]["dependencies"]]
    selected = {
        value.name.lower(): value
        for value in requirements
        if value.marker is None or value.marker.evaluate(environment)
    }
    assert not {"vllm", "mlx", "mlx-vlm", "vllm-metal"} & selected.keys()
    assert {"torch", "laya", "fastapi", "uvicorn", "peft"} <= selected.keys()
    assert "5.10.2" in selected["transformers"].specifier
    assert "0.19.1" in selected["peft"].specifier


def test_windows_import_and_http_do_not_require_vllm_or_mlx():
    # A fresh interpreter prevents a prior test's sys.modules cache masking an
    # accidental import of the Linux endpoint or an Apple-only dependency.
    code = """
import builtins
import sys
from pathlib import Path
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'vllm', 'mlx', 'mlx_vlm', 'vllm_metal'}:
        raise AssertionError('unexpected native backend import: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from fastapi.testclient import TestClient
from vllm_jev import cli, windows_laya, windows_models
class Agent:
    def system_one(self, state, questions):
        return {'answers': {'q': {'type': 'noul', 'noul': 0.5}}}
app = windows_laya.create_app(Path('unused'), 'convaiinnovations/laya-multilingual',
                              agent_factory=Agent)
with TestClient(app) as client:
    assert client.get('/health').status_code == 200
    assert client.post('/v1/systemone', json={'state': 'text', 'questions': {
        'q': {'type': 'noul', 'instructions': 'Is it urgent?'}}}).status_code == 200
assert not any(name == 'vllm' or name.startswith('vllm.') for name in sys.modules)
"""
    environment = os.environ.copy()
    environment.pop(ENVIRONMENT, None)
    environment["PYTHONPATH"] = str(ROOT)
    result = subprocess.run(
        [sys.executable, "-B", "-c", code],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "state", ["中文退款请求", {"message": "refund", "n": 2}, ["a", "b"]]
)
def test_http_preserves_sdk_answers_and_marks_cpu_backend(state):
    agent = RecordingAgent()
    response = asyncio.run(request_once(agent, payload(state=state)))
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["answers"] == RESULT["answers"]
    assert result["usage"] == RESULT["usage"]
    assert result["model"] == MODEL
    assert result["metadata"]["device"] == "cpu"
    assert result["metadata"]["experimental"] is True
    assert "laya" in result["metadata"]["backend"]
    assert agent.calls == [(state, QUESTIONS)]


@pytest.mark.parametrize(
    "updates",
    [
        {"model": "another-model"},
        {"state": 42},
        {"state": None},
        {"state": "x" * 50_001},
        {"questions": {"q": {"type": [], "instructions": "Choose."}}},
        {"questions": {"q": {"type": "chat", "instructions": "Choose."}}},
        {
            "questions": {
                "q": {
                    "type": "choice",
                    "instructions": "Choose.",
                    "criteria": [{"a": 1}],
                }
            }
        },
        {
            "questions": {
                "q": {"type": "choice", "instructions": "Choose.", "criteria": [1, 2]}
            }
        },
        {
            "questions": {
                "q": {
                    "type": "choice",
                    "instructions": "Choose.",
                    "criteria": [None, True],
                }
            }
        },
        {
            "questions": {
                str(i): {"type": "noul", "instructions": "Choose."} for i in range(65)
            }
        },
        {
            "questions": {
                "q": {
                    "type": "choice",
                    "instructions": "Choose.",
                    "criteria": [str(i) for i in range(257)],
                }
            }
        },
        {"images": ["data:image/png;base64,AA=="]},
        {"videos": ["data:video/mp4;base64,AA=="]},
        {"media": [{"type": "image", "uri": "data:image/png;base64,AA=="}]},
        {
            "state": {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": "data:image/png;base64,AA=="},
                            }
                        ],
                    }
                ]
            }
        },
        {"cache_salt": "caller-a"},
        {"cache_salt": ""},
    ],
)
def test_invalid_or_unsupported_inputs_never_reach_inference(updates):
    agent = RecordingAgent()
    response = asyncio.run(request_once(agent, payload(**updates)))
    assert response.status_code == 422, response.text
    assert not agent.calls


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity"])
def test_nonfinite_json_is_rejected_without_inference(value):
    agent = RecordingAgent()

    async def exercise():
        app = app_for(agent)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                body = (
                    '{"state":{"value":'
                    + value
                    + '},"questions":{"q":{"type":"noul"}}}'
                )
                return await client.post(
                    "/v1/systemone",
                    content=body,
                    headers={"Content-Type": "application/json"},
                )

    response = asyncio.run(exercise())
    assert response.status_code == 422, response.text
    assert not agent.calls


def test_two_concurrent_requests_queue_and_both_succeed():
    started, release = threading.Event(), threading.Event()

    class BlockingAgent(RecordingAgent):
        def system_one(self, state, questions):
            if state == "first":
                started.set()
                if not release.wait(timeout=5):
                    raise AssertionError("test did not release first request")
            return super().system_one(state, questions)

    agent = BlockingAgent()

    async def exercise():
        app = app_for(agent)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                first = asyncio.create_task(
                    client.post("/v1/systemone", json=payload(state="first"))
                )
                second = None
                try:
                    assert await asyncio.to_thread(started.wait, 2)
                    second = asyncio.create_task(
                        client.post("/v1/systemone", json=payload(state="second"))
                    )
                    await asyncio.sleep(0.02)
                    assert not second.done()
                    assert (
                        await asyncio.wait_for(client.get("/health"), 0.5)
                    ).status_code == 200
                    release.set()
                    responses = await asyncio.wait_for(asyncio.gather(first, second), 2)
                    assert [response.status_code for response in responses] == [
                        200,
                        200,
                    ]
                    assert [state for state, _ in agent.calls] == ["first", "second"]
                finally:
                    release.set()
                    await asyncio.gather(
                        *[task for task in (first, second) if task is not None],
                        return_exceptions=True,
                    )

    asyncio.run(exercise())


def test_cancelled_work_holds_capacity_until_consumed_then_http_recovers():
    started, release = threading.Event(), threading.Event()

    class BlockingAgent(RecordingAgent):
        def system_one(self, state, questions):
            if state == "blocking":
                started.set()
                if not release.wait(timeout=5):
                    raise AssertionError("test did not release the blocked CPU call")
            return super().system_one(state, questions)

    agent = BlockingAgent()

    async def exercise():
        app = app_for(agent, max_pending=2)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                first = asyncio.create_task(
                    client.post("/v1/systemone", json=payload(state="blocking"))
                )
                queued = None
                try:
                    assert await asyncio.to_thread(started.wait, 2)
                    health = await asyncio.wait_for(client.get("/health"), 0.5)
                    assert health.status_code == 200
                    queued = asyncio.create_task(
                        client.post(
                            "/v1/systemone",
                            json=payload(state="cancel before execution"),
                        )
                    )
                    await asyncio.sleep(0.02)
                    busy = await asyncio.wait_for(
                        client.post("/v1/systemone", json=payload(state="overflow")),
                        0.5,
                    )
                    assert busy.status_code == 503
                    queued.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await queued
                    first.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await first
                    # Cancelling either caller cannot free its executor slot
                    # before the synchronous worker has consumed that work.
                    still_busy = await asyncio.wait_for(
                        client.post("/v1/systemone", json=payload(state="still full")),
                        0.5,
                    )
                    assert still_busy.status_code == 503
                    assert (
                        await asyncio.wait_for(client.get("/health"), 0.5)
                    ).status_code == 200
                    release.set()

                    async def wait_until_consumed():
                        while app.state.laya_service.pending:
                            await asyncio.sleep(0)

                    await asyncio.wait_for(wait_until_consumed(), 2)
                    survivor = await asyncio.wait_for(
                        client.post("/v1/systemone", json=payload(state="survivor")),
                        2,
                    )
                    assert survivor.status_code == 200
                    recovered = await client.post(
                        "/v1/systemone", json=payload(state="recovered")
                    )
                    assert recovered.status_code == 200
                    assert [state for state, _ in agent.calls] == [
                        "blocking",
                        "survivor",
                        "recovered",
                    ]
                finally:
                    release.set()
                    tasks = [task for task in (first, queued) if task is not None]
                    for task in tasks:
                        if not task.done():
                            task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)

    asyncio.run(exercise())


def test_shutdown_finishes_running_work_before_closing_agent():
    started, release, closed = threading.Event(), threading.Event(), threading.Event()

    class BlockingAgent(RecordingAgent):
        def system_one(self, state, questions):
            started.set()
            if not release.wait(5):
                raise AssertionError("test did not release worker")
            assert not closed.is_set(), "agent was freed during inference"
            return super().system_one(state, questions)

        def close(self):
            closed.set()

    async def exercise():
        app = app_for(BlockingAgent())
        lifespan = app.router.lifespan_context(app)
        await lifespan.__aenter__()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            request = asyncio.create_task(client.post("/v1/systemone", json=payload()))
            shutdown = None
            try:
                assert await asyncio.to_thread(started.wait, 2)
                shutdown = asyncio.create_task(lifespan.__aexit__(None, None, None))
                await asyncio.sleep(0.02)
                assert not closed.is_set()
                assert not shutdown.done()
                assert (await client.get("/health")).status_code == 503
            finally:
                release.set()
                if shutdown is None:
                    shutdown = asyncio.create_task(lifespan.__aexit__(None, None, None))
                await asyncio.wait_for(shutdown, 2)
                await asyncio.gather(request, return_exceptions=True)
        assert closed.is_set()

    asyncio.run(exercise())


def test_loading_failure_stops_its_executor_thread():
    from vllm_jev.windows_laya import create_app

    threads = []

    def fail():
        threads.append(threading.current_thread())
        raise RuntimeError("synthetic model loading failure")

    async def exercise():
        app = create_app(Path("unused"), MODEL, agent_factory=fail)
        with pytest.raises(RuntimeError, match="synthetic model loading failure"):
            async with app.router.lifespan_context(app):
                pytest.fail("a failed load must never enter ready lifespan")
        assert app.state.laya_service is None
        assert len(threads) == 1 and not threads[0].is_alive()

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "extra",
    [
        ["--gpu-memory-utilization", "0.1"],
        ["--radixflow", "true"],
        ["--cpu-threads", "0"],
        ["--protocol", "open_jev_choice"],
    ],
)
def test_windows_cli_rejects_incompatible_options_before_preparing(
    monkeypatch, tmp_path, extra
):
    from vllm_jev import cli, windows_models

    monkeypatch.setattr(cli.sys, "platform", "win32")
    monkeypatch.setattr(
        cli.sys,
        "argv",
        ["vllm-jev", "serve", MODEL, "--workspace", str(tmp_path), *extra],
    )
    monkeypatch.setattr(
        windows_models,
        "prepare",
        lambda *args, **kwargs: pytest.fail("unexpected preparation"),
    )
    monkeypatch.setattr(
        cli.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("unexpected native subprocess"),
    )
    monkeypatch.setattr(
        cli.os,
        "execvpe",
        lambda *args, **kwargs: pytest.fail("unexpected native server"),
    )
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2


def test_windows_cli_routes_original_checkpoint_to_cpu_server(monkeypatch, tmp_path):
    from vllm_jev import cli, windows_laya, windows_models

    checkpoint = tmp_path / "source"
    checkpoint.mkdir()
    (checkpoint / "rl_agent_config.json").write_text("{}", encoding="utf-8")
    observed = {}
    monkeypatch.setattr(cli.sys, "platform", "win32")
    monkeypatch.setattr(
        cli.sys,
        "argv",
        [
            "vllm-jev",
            "serve",
            str(checkpoint),
            "--workspace",
            str(tmp_path),
            "--cpu-threads",
            "2",
            "--port",
            "18795",
        ],
    )
    monkeypatch.setattr(windows_models, "identify_source", lambda path: MODEL)
    monkeypatch.setattr(
        windows_models,
        "prepare",
        lambda *args, **kwargs: pytest.fail("local source was redownloaded"),
    )
    monkeypatch.setattr(
        windows_laya,
        "serve",
        lambda path, model, **kwargs: observed.update(
            checkpoint=path, model=model, **kwargs
        ),
    )
    monkeypatch.setattr(
        cli.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("unexpected native subprocess"),
    )
    monkeypatch.setattr(
        cli.os,
        "execvpe",
        lambda *args, **kwargs: pytest.fail("unexpected native server"),
    )
    cli.main()
    assert observed == {
        "checkpoint": checkpoint.resolve(),
        "model": MODEL,
        "threads": 2,
        "port": 18795,
        "host": "127.0.0.1",
    }


@pytest.mark.parametrize("logical_cpus,expected_threads", [(1, 1), (2, 2), (None, 1)])
def test_windows_cli_default_threads_fit_small_or_unknown_cpu_count(
    monkeypatch, tmp_path, logical_cpus, expected_threads
):
    from vllm_jev import cli, windows_laya, windows_models

    observed = {}
    monkeypatch.setattr(cli.sys, "platform", "win32")
    monkeypatch.setattr(cli.os, "cpu_count", lambda: logical_cpus)
    monkeypatch.setattr(
        cli.sys, "argv", ["vllm-jev", "serve", MODEL, "--workspace", str(tmp_path)]
    )
    monkeypatch.setattr(windows_models, "prepare", lambda model, workspace: tmp_path)
    monkeypatch.setattr(
        windows_laya, "serve", lambda *args, **kwargs: observed.update(kwargs)
    )
    cli.main()
    assert observed["threads"] == expected_threads


@pytest.fixture
def small_verified_source(tmp_path, monkeypatch):
    """Use the production hash verifier against a tiny synthetic release."""
    from vllm_jev import laya_export

    source = tmp_path / "原始 source"
    files = {
        "model.safetensors": b"small-test-weight-file",
        "encoder/config.json": b'{"model_type":"modernbert"}',
        "rl_agent_config.json": b'{"head_layers":2,"max_len":1024,"head_max_len":256}',
        "tokenizer/tokenizer.json": b'{"test":"tokenizer"}',
        "tokenizer/tokenizer_config.json": b'{"original":true}',
    }
    for name, contents in files.items():
        destination = source / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(contents)
    digests = {name: hashlib.sha256(value).hexdigest() for name, value in files.items()}
    monkeypatch.setitem(
        laya_export.MODELS,
        MODEL,
        ("a" * 40, digests["model.safetensors"], 1024, 256),
    )
    monkeypatch.setitem(
        laya_export.SOURCE_HASHES,
        MODEL,
        tuple(
            digests[name]
            for name in (
                "encoder/config.json",
                "rl_agent_config.json",
                "tokenizer/tokenizer.json",
                "tokenizer/tokenizer_config.json",
            )
        ),
    )
    return source, files


def test_preparation_verifies_before_publishing_and_reuses_only_valid_cache(
    tmp_path, monkeypatch, small_verified_source
):
    from vllm_jev import windows_models

    source, files = small_verified_source
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        for name, data in files.items():
            path = Path(kwargs["local_dir"]) / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)

    monkeypatch.setattr(windows_models, "snapshot_download", download)
    workspace = tmp_path / "workspace"
    checkpoint = windows_models.prepare(MODEL, workspace)
    assert windows_models.identify_source(checkpoint) == MODEL
    assert calls[0]["repo_id"] == MODEL
    assert calls[0]["revision"] == "a" * 40
    assert calls[0]["token"] is False
    assert set(calls[0]["allow_patterns"]) == set(files)
    assert windows_models.prepare(MODEL, workspace) == checkpoint
    assert len(calls) == 1
    (checkpoint / "model.safetensors").write_bytes(b"unexpected release")
    with pytest.raises(ValueError, match="pinned release"):
        windows_models.prepare(MODEL, workspace)
    assert len(calls) == 1, "corrupt existing source must not be silently replaced"
    assert (source / "model.safetensors").read_bytes() == files["model.safetensors"]


def test_failed_download_verification_leaves_no_published_or_partial_model(
    tmp_path, monkeypatch, small_verified_source
):
    from vllm_jev import windows_models

    _, files = small_verified_source

    def download(**kwargs):
        for name, data in files.items():
            path = Path(kwargs["local_dir"]) / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        (Path(kwargs["local_dir"]) / "model.safetensors").write_bytes(b"truncated")

    monkeypatch.setattr(windows_models, "snapshot_download", download)
    workspace = tmp_path / "workspace"
    with pytest.raises(ValueError, match="pinned release"):
        windows_models.prepare(MODEL, workspace)
    assert not (workspace / "checkpoint" / MODEL).exists()
    assert not list(workspace.rglob(".laya-download-*"))


@pytest.mark.parametrize(
    "digest_value,accepted",
    [
        ("424b69444bf7b5809dc2cd2e36d0bd71b8055124dd24274d6db3c655d38205e7", True),
        ("6c6b2d8e3c84ce0e671c129cd6b374b235d6f9863042a5836358d00a89bbb5a1", True),
        ("2c0c4d82d4b4bc6b4ac40b2375e067a1645f78b381a9774248d47915f33d751f", False),
    ],
)
def test_only_pinned_raw_and_legacy_lf_tokenizer_hashes_are_accepted(
    monkeypatch, small_verified_source, digest_value, accepted
):
    from vllm_jev import laya_export

    source, _ = small_verified_source
    original_hash = laya_export.sha256
    hashes = laya_export.SOURCE_HASHES[MODEL]
    monkeypatch.setitem(
        laya_export.SOURCE_HASHES,
        MODEL,
        (
            *hashes[:-1],
            "424b69444bf7b5809dc2cd2e36d0bd71b8055124dd24274d6db3c655d38205e7",
        ),
    )
    monkeypatch.setattr(
        laya_export,
        "sha256",
        lambda path: (
            digest_value
            if path == source / "tokenizer/tokenizer_config.json"
            else original_hash(path)
        ),
    )
    if accepted:
        laya_export.verify_source(source, MODEL)
    else:
        with pytest.raises(ValueError, match="pinned release"):
            laya_export.verify_source(source, MODEL)
    # The legacy exception is specific to tokenizer metadata, not model weights.
    monkeypatch.setattr(
        laya_export,
        "sha256",
        lambda path: (
            "6c6b2d8e3c84ce0e671c129cd6b374b235d6f9863042a5836358d00a89bbb5a1"
            if path == source / "model.safetensors"
            else original_hash(path)
        ),
    )
    with pytest.raises(ValueError, match="pinned release"):
        laya_export.verify_source(source, MODEL)


@pytest.mark.parametrize("fails", [False, True])
def test_sdk_tokenizer_mutation_never_changes_source_and_runtime_is_cleaned(
    monkeypatch, small_verified_source, fails
):
    from vllm_jev import windows_models

    source, files = small_verified_source
    observed = {}

    class SDKAgent:
        def __init__(self, path, **kwargs):
            observed.update(path=Path(path), kwargs=kwargs)
            self.device = "cpu"
            self.amp_enabled = True
            (Path(path) / "tokenizer/tokenizer_config.json").write_text(
                '{"normalized_by_sdk":true}', encoding="utf-8"
            )
            if fails:
                raise RuntimeError("synthetic SDK load failure")

        def system_one(self, state, questions):
            return copy.deepcopy(RESULT)

    monkeypatch.setitem(sys.modules, "laya", SimpleNamespace(Agent=SDKAgent))
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(set_num_threads=lambda count: None, float32="float32"),
    )
    if fails:
        with pytest.raises(RuntimeError, match="synthetic SDK load failure"):
            windows_models.load_agent(source, MODEL, threads=1)
    else:
        agent = windows_models.load_agent(source, MODEL, threads=1)
        assert observed["path"] != source
        assert observed["path"].exists()
        assert agent.agent.amp_enabled is False
        assert agent.system_one("state", QUESTIONS) == RESULT
        agent.close()
        agent.close()  # App cleanup may follow a caller-initiated close.
        with pytest.raises(RuntimeError, match="closed"):
            agent.system_one("state", QUESTIONS)
    assert observed["kwargs"] == {"device": "cpu", "fast": False, "compile": False}
    assert not observed["path"].exists()
    for name, contents in files.items():
        assert (source / name).read_bytes() == contents
    assert windows_models.identify_source(source) == MODEL


@pytest.mark.parametrize("local_path", [False, True])
def test_validation_driver_owns_and_cleans_cli_workspace(
    tmp_path, monkeypatch, small_verified_source, local_path
):
    source, files = small_verified_source
    spec = importlib.util.spec_from_file_location(
        "windows_validation_driver", ROOT / "scripts/check_windows_laya.py"
    )
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    workspace = tmp_path / "validation"
    workspace.mkdir()
    observed = {}

    def run(args, reference, environment, report, round_number):
        clone = args.workspace / "checkpoint" / MODEL
        observed["workspace"] = args.workspace
        assert args.source == (clone if local_path else None)
        assert os.path.samefile(
            clone / "model.safetensors", source / "model.safetensors"
        )
        assert not os.path.samefile(
            clone / "tokenizer/tokenizer_config.json",
            source / "tokenizer/tokenizer_config.json",
        )
        mutable = clone.parent / ".laya-cpu-owned"
        mutable.mkdir()
        (mutable / "tokenizer.json").write_text(
            "temporary SDK metadata", encoding="utf-8"
        )
        report["service_runs"].append(
            {"owned_process_started": True, "owned_process_stopped": True}
        )

    monkeypatch.setattr(driver, "validate_service_run", run)
    args = SimpleNamespace(
        workspace=workspace, model=MODEL, source=source if local_path else None
    )
    reference = {
        "source": str(source),
        "source_sha256": {
            name: hashlib.sha256(value).hexdigest() for name, value in files.items()
        },
    }
    report = {"service_runs": []}
    driver.validate_service(args, reference, {}, report, 1)
    assert not observed["workspace"].exists()
    assert report["service_runs"][0]["private_workspace_removed"] is True
    assert report["service_runs"][0]["source_unchanged"] is True
    assert not list(workspace.iterdir())
    for name, contents in files.items():
        assert (source / name).read_bytes() == contents
