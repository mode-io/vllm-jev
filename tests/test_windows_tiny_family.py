"""Pinned family identity and dispatch; numerical loading is tested in test_windows_qwen."""

import hashlib
import json

import pytest

from vllm_jev import windows_dependencies, windows_models, windows_qwen, windows_tiny


@pytest.mark.parametrize("model_id", tuple(windows_tiny.MODELS))
def test_release_source_identification_requires_all_matching_files(
    tmp_path, monkeypatch, model_id
):
    spec = dict(windows_tiny.MODELS[model_id])
    config = {
        "model_type": "tiny_jev",
        "architectures": ["TinyJevModel"],
        "hidden_size": spec["hidden_size"],
        "temperature": spec["temperature"],
        "opt_token": "<opt>",
    }
    hashes = {}
    for name in spec["source_hashes"]:
        payload = (
            json.dumps(config).encode()
            if name == "config.json"
            else (model_id + name).encode()
        )
        (tmp_path / name).write_bytes(payload)
        hashes[name] = hashlib.sha256(payload).hexdigest()
    spec["source_hashes"] = hashes
    monkeypatch.setitem(windows_tiny.MODELS, model_id, spec)
    assert windows_models.identify_source(tmp_path) == model_id
    wrong = next(name for name in windows_tiny.MODELS if name != model_id)
    with pytest.raises(ValueError, match="checksum"):
        windows_tiny.verify_source(tmp_path, wrong)
    (tmp_path / "modeling_tiny_jev.py").write_text("tampered")
    with pytest.raises(ValueError, match="checksum"):
        windows_tiny.identify_source(tmp_path)


def test_new_family_release_identity_and_dependency_dispatch():
    model_id = "lostargon/Tiny-Jev-1.7B"
    spec = windows_tiny.MODELS[model_id]
    assert spec["revision"] == "c41bcf3f191407c779bd2ebce1b992dd1469437f"
    assert spec["temperature"] == 1.15
    assert windows_models.protocol_for(model_id) == "tiny_jev_marker"
    assert windows_dependencies.profile_for(model_id).name == "legacy"
    assert windows_dependencies.profile_for("alibiserikbay/JevK5-9B").name == "modern"
    assert windows_models.protocol_for("alibiserikbay/JevK5-9B") == "jevk5_gguf_v1"
    agent = windows_tiny.TinyAgent(
        object(), None, None, None, model_id=model_id, temperature=1.15
    )
    response = agent.system_one(
        "state",
        {"q": {"type": "choice", "instructions": "Pick", "criteria": {"only": None}}},
    )
    assert response["model"] == model_id
    assert response["metadata"]["temperature"] == 1.15
    agent.close()


@pytest.mark.parametrize(
    "model_id", ["lostargon/Tiny-Jev-1.7B", "alibiserikbay/JevK5-9B"]
)
def test_http_health_retains_selected_family_identity(tmp_path, monkeypatch, model_id):
    from fastapi.testclient import TestClient

    class Agent:
        def close(self):
            pass

    with TestClient(
        windows_qwen.create_app(tmp_path, model_id, agent_factory=Agent)
    ) as client:
        result = client.get("/health")
        assert result.status_code == 200
        assert result.json()["model"] == model_id
