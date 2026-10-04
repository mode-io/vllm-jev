"""Request-validation regressions; no model weights or GPU required."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm_jev import clef, rsijev


@pytest.fixture
def endpoint(monkeypatch):
    pytest.importorskip("vllm")
    from vllm_jev import endpoint

    # Exercise the actual FastAPI request schema and routing, without vLLM's
    # unrelated server-load and disconnect state in these CPU-only tests.
    monkeypatch.setattr(endpoint, "with_cancellation", lambda function: function)
    monkeypatch.setattr(endpoint, "load_aware_call", lambda function: function)
    return endpoint


class RecordingService:
    def __init__(self, validate=None):
        self.calls = []
        self.validate = validate

    async def systemone(self, payload):
        self.calls.append(payload)
        if self.validate:
            self.validate(payload)
        return {"answers": {}}


def client_for(endpoint, backend, service):
    app = FastAPI()
    setattr(app.state, backend, service)
    endpoint.JevEndpointPlugin().attach_router(app)
    return TestClient(app)


def payload(**fields):
    return {
        "state": "A short state",
        "questions": {"q": {"type": "noul", "instructions": "Is it urgent?"}},
        **fields,
    }


@pytest.mark.parametrize(
    "backend",
    [
        "vllm_decision_service",
        "vllm_rsijev_service",
        "vllm_laya_service",
        "vllm_vjev_service",
        "vllm_valen_service",
        "vllm_jev_service",
    ],
)
@pytest.mark.parametrize("field", ["images", "videos"])
def test_non_clef_rejects_top_level_media_before_dispatch(endpoint, backend, field):
    service = RecordingService()
    client = client_for(endpoint, backend, service)
    response = client.post("/v1/systemone", json=payload(**{field: ["unused media"]}))
    assert response.status_code == 422
    assert "top-level images and videos" in response.json()["detail"]
    assert not service.calls


@pytest.mark.parametrize(
    "media",
    [{}, {"images": None, "videos": None}, {"images": [], "videos": []}],
)
def test_non_clef_accepts_no_top_level_media(endpoint, media):
    service = RecordingService()
    client = client_for(endpoint, "vllm_rsijev_service", service)
    assert client.post("/v1/systemone", json=payload(**media)).status_code == 200
    assert len(service.calls) == 1


def test_clef_receives_top_level_media(endpoint):
    service = RecordingService()
    client = client_for(endpoint, "vllm_clef_service", service)
    response = client.post(
        "/v1/systemone",
        json=payload(images=["image payload"], videos=["video payload"]),
    )
    assert response.status_code == 200
    assert service.calls[0].images == ["image payload"]
    assert service.calls[0].videos == ["video payload"]


@pytest.mark.parametrize("kind", [[], {}, None, True, 1])
def test_clef_invalid_question_type_is_a_validation_error(kind):
    with pytest.raises(ValueError, match="type must be noul, choice or score"):
        clef.validate_question("q", {"type": kind})


@pytest.mark.parametrize("role", [[], {}, None, True, 1])
def test_rsijev_invalid_message_role_is_a_validation_error(role):
    with pytest.raises(ValueError, match="unsupported message role"):
        rsijev.state_parts([{"role": role, "content": "hello"}])


@pytest.mark.parametrize("kind", [[], {}])
def test_clef_malformed_type_returns_422(endpoint, kind):
    service = RecordingService(
        lambda request: clef.validate_question("q", request.questions["q"])
    )
    client = client_for(endpoint, "vllm_clef_service", service)
    response = client.post(
        "/v1/systemone", json=payload(questions={"q": {"type": kind}})
    )
    assert response.status_code == 422


@pytest.mark.parametrize("role", [[], {}])
def test_rsijev_malformed_role_returns_422(endpoint, role):
    service = RecordingService(lambda request: rsijev.state_parts(request.state))
    client = client_for(endpoint, "vllm_rsijev_service", service)
    response = client.post(
        "/v1/systemone", json=payload(state=[{"role": role, "content": "hello"}])
    )
    assert response.status_code == 422
