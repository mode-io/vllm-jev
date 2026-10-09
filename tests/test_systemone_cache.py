"""System One request and cancellation tests with an instrumented engine."""

import asyncio
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def endpoint():
    # Keep these HTTP tests runnable without vLLM or a GPU. Cache behavior is
    # covered separately with vLLM's cache manager and a live server.
    stubs = {
        "vllm": SimpleNamespace(PoolingParams=SimpleNamespace),
        "vllm.entrypoints.serve.utils.api_utils": SimpleNamespace(
            with_cancellation=lambda function: function,
            load_aware_call=lambda function: function,
        ),
    }
    path = Path(__file__).parents[1] / "vllm_jev" / "endpoint.py"
    spec = importlib.util.spec_from_file_location("vllm_jev.endpoint", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)
    return module


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return "\n".join(message["content"] for message in messages)

    def encode(self, text, **kwargs):
        return list(text.encode())


class RecordingEngine:
    def __init__(self, blocked=False):
        self.calls = []
        self.aborted = []
        self.blocked = blocked
        self.started = asyncio.Queue()
        self.release = {}

    async def encode(self, *, prompt, pooling_params, request_id, priority):
        self.calls.append((prompt, pooling_params, request_id))
        self.release[request_id] = asyncio.Event()
        self.started.put_nowait(request_id)
        if self.blocked:
            await self.release[request_id].wait()
        ids = prompt.get("prompt_token_ids") or Tokenizer().encode(prompt["prompt"])
        yield SimpleNamespace(
            finished=True,
            outputs=SimpleNamespace(data=np.array([0.5])),
            prompt_token_ids=ids,
            num_cached_tokens=0,
        )

    async def abort(self, request_id):
        self.aborted.append(request_id)


def app_for(endpoint, service, backend="vllm_jev_service"):
    app = FastAPI()
    setattr(app.state, backend, service)
    endpoint.JevEndpointPlugin().attach_router(app)
    return app


def payload(**fields):
    return {
        "state": "A customer requests a refund.",
        "questions": {
            "urgent": {"type": "noul", "instructions": "Is it urgent?"},
            "action": {
                "type": "choice",
                "instructions": "What next?",
                "criteria": {"refund": "Refund", "review": "Review"},
            },
            "severity": {
                "type": "score",
                "instructions": "How severe?",
                "criteria": ["Low", "High"],
            },
        },
        **fields,
    }


@pytest.mark.parametrize("protocol", ["open_jev_choice", "openjev_branch_v03"])
@pytest.mark.parametrize("salt", [None, "caller-a", " x ", "x" * 256])
def test_namespace_reaches_every_question(endpoint, protocol, salt):
    engine = RecordingEngine()
    service = endpoint._JevService(engine, Tokenizer(), protocol=protocol)
    client = TestClient(app_for(endpoint, service))
    fields = {} if salt is None else {"cache_salt": salt}
    salts = []
    for _ in range(2):
        engine.calls.clear()
        response = client.post("/v1/systemone", json=payload(**fields))
        assert response.status_code == 200, response.text
        assert set(response.json()["answers"]) == {"urgent", "action", "severity"}
        assert len(engine.calls) == (5 if protocol == "open_jev_choice" else 6)
        request_salts = {prompt["cache_salt"] for prompt, _, _ in engine.calls}
        assert len(request_salts) == 1
        salts.append(request_salts.pop())
        assert all(params.task == "classify" for _, params, _ in engine.calls)
        assert len({request_id for _, _, request_id in engine.calls}) == len(
            engine.calls
        )
    if salt is None:
        assert salts[0] != salts[1]
    else:
        assert salts == [salt, salt]


@pytest.mark.parametrize("salt", ["", " \t\n", "x" * 257, 1, True, [], {}])
def test_invalid_namespace_is_rejected_before_inference(endpoint, salt):
    engine = RecordingEngine()
    service = endpoint._JevService(engine, Tokenizer())
    response = TestClient(app_for(endpoint, service)).post(
        "/v1/systemone", json=payload(cache_salt=salt)
    )
    assert response.status_code == 422
    assert not engine.calls


def test_null_namespace_keeps_request_isolation(endpoint):
    engine = RecordingEngine()
    service = endpoint._JevService(engine, Tokenizer())
    client = TestClient(app_for(endpoint, service))
    for _ in range(2):
        assert (
            client.post("/v1/systemone", json=payload(cache_salt=None)).status_code
            == 200
        )
    assert len({prompt["cache_salt"] for prompt, _, _ in engine.calls}) == 2


@pytest.mark.parametrize(
    "backend",
    [
        "vllm_decision_service",
        "vllm_clef_service",
        "vllm_rsijev_service",
        "vllm_laya_service",
        "vllm_vjev_service",
        "vllm_valen_service",
        "vllm_jev_service",
    ],
)
def test_unsupported_backend_rejects_namespace(endpoint, backend):
    # Mac services also use these dispatch slots, without the native capability.
    service = SimpleNamespace()
    response = TestClient(app_for(endpoint, service, backend)).post(
        "/v1/systemone", json=payload(cache_salt="caller-a")
    )
    assert response.status_code == 422
    assert "sequence-classification" in response.json()["detail"]


def test_token_embedding_backend_rejects_namespace(endpoint):
    engine = RecordingEngine()
    service = endpoint._JevService(engine, Tokenizer(), protocol="tiny_jev_marker")
    response = TestClient(app_for(endpoint, service)).post(
        "/v1/systemone", json=payload(cache_salt="caller-a")
    )
    assert response.status_code == 422
    assert not engine.calls


def test_requests_overlap_and_cancellation_is_scoped(endpoint):
    async def run():
        engine = RecordingEngine(blocked=True)
        service = endpoint._JevService(engine, Tokenizer(), max_inflight=4)
        transport = httpx.ASGITransport(app=app_for(endpoint, service))
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            questions = {
                f"a{i}": {"type": "noul", "instructions": f"Question {i}?"}
                for i in range(1, 5)
            }
            first = asyncio.create_task(
                client.post(
                    "/v1/systemone",
                    json=payload(questions=questions, cache_salt="caller-a"),
                )
            )
            second = None
            try:
                first_ids = [await engine.started.get() for _ in range(4)]
                second = asyncio.create_task(
                    client.post(
                        "/v1/systemone",
                        json=payload(
                            questions={
                                "a5": {"type": "noul", "instructions": "Question 5?"}
                            },
                            cache_salt="caller-a",
                        ),
                    )
                )
                engine.release[first_ids[1]].set()
                second_id = await engine.started.get()
                assert second_id not in first_ids
                assert not first.done()
                assert not second.done()
                assert {call[0]["cache_salt"] for call in engine.calls} == {"caller-a"}
                first.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await first
                assert set(engine.aborted) == set(first_ids) - {first_ids[1]}
                assert second_id not in engine.aborted
                engine.release[second_id].set()
                response = await second
                assert response.status_code == 200
                assert set(response.json()["answers"]) == {"a5"}
            finally:
                tasks = [task for task in (first, second) if task is not None]
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)

    asyncio.run(asyncio.wait_for(run(), timeout=5))


@pytest.fixture
def configured_template(monkeypatch):
    from vllm_jev.decision_template import ENVIRONMENT, current_template

    def set_config(config):
        import json

        monkeypatch.setenv(ENVIRONMENT, json.dumps(config))
        current_template.cache_clear()

    yield set_config
    current_template.cache_clear()


def test_template_applies_once_to_all_typed_questions(endpoint, configured_template):
    configured_template(
        {
            "layout": "instructions-first",
            "instruction_template": "POLICY\n{instructions}",
        }
    )
    engine = RecordingEngine()
    service = endpoint._JevService(engine, Tokenizer())
    client = TestClient(app_for(endpoint, service))
    response = client.post("/v1/systemone", json=payload())
    assert response.status_code == 200, response.text
    assert len(engine.calls) == 5
    for prompt, _, _ in engine.calls:
        text = prompt["prompt"]
        assert text.startswith("Question: POLICY\n")
        assert text.count("POLICY") == 1
        assert text.index("Question:") < text.index("Context:")


@pytest.mark.parametrize(
    "backend",
    [
        "vllm_clef_service",
        "vllm_rsijev_service",
        "vllm_laya_service",
        "vllm_decision_service",
        "vllm_valen_service",
        "vllm_vjev_service",
    ],
)
def test_shared_instruction_template_preserves_media_and_schema(
    endpoint, configured_template, backend
):
    configured_template({"instruction_template": "PREFIX {instructions} SUFFIX"})
    seen = []

    class Service:
        async def systemone(self, request):
            seen.append(request)
            return {"ok": True}

    client = TestClient(app_for(endpoint, Service(), backend))
    body = payload(
        state={
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,abc"},
                        }
                    ],
                }
            ]
        }
    )
    response = client.post("/v1/systemone", json=body)
    assert response.status_code == 200, response.text
    assert seen[0].state == body["state"]
    for key, q in body["questions"].items():
        assert (
            seen[0].questions[key]["instructions"]
            == "PREFIX " + q["instructions"] + " SUFFIX"
        )
        assert seen[0].questions[key].get("criteria") == q.get("criteria")
    assert body["questions"]["urgent"]["instructions"] == "Is it urgent?"


def test_layout_rejected_for_marker_backend(endpoint, configured_template):
    configured_template({"layout": "instructions-first"})
    engine = RecordingEngine()
    service = endpoint._JevService(engine, Tokenizer(), protocol="openjev_branch_v03")
    response = TestClient(app_for(endpoint, service)).post(
        "/v1/systemone", json=payload()
    )
    assert response.status_code == 422
    assert not engine.calls


def test_invalid_batch_template_input_starts_no_engine_work(
    endpoint, configured_template
):
    configured_template({"state_template": "Evidence: {state}"})
    engine = RecordingEngine(blocked=True)
    service = endpoint._JevService(engine, Tokenizer())
    client = TestClient(app_for(endpoint, service))
    request = {"state": "plain evidence", "question": "Which?", "options": ["A", "B"]}
    response = client.post(
        "/plugins/vllm-jev/batch",
        json={"requests": [request, {**request, "state": {"messages": []}}]},
    )
    assert response.status_code == 422
    assert engine.calls == []
