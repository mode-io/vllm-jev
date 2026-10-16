"""Bruv contract and numerical checks without loading released weights."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm_jev import windows_bruv as adapter
from vllm_jev.decision_template import DecisionTemplate, current_template

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = {
    "choice": {
        "type": "choice",
        "instructions": "Choose.",
        "criteria": {"billing": "Refund", "other": None},
    },
    "noul": {
        "type": "noul",
        "instructions": "Refund?",
        "criteria": {"true": "Refund requested"},
    },
    "score": {
        "type": "score",
        "instructions": "Urgency?",
        "criteria": ["Low", "Medium", "High"],
    },
}


@pytest.fixture(autouse=True)
def native_template(monkeypatch):
    monkeypatch.delenv("VLLM_JEV_DECISION_TEMPLATE", raising=False)
    current_template.cache_clear()
    yield
    current_template.cache_clear()


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return list(text.encode("utf-8"))

    def decode(self, ids):
        return bytes(ids).decode("utf-8")

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }
        return "\n".join(m["content"] for m in messages) + "\nAssistant:\n"


def make_agent():
    import torch

    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(256, 4)
            self.embedding.weight.data.copy_(
                torch.arange(1024).reshape(256, 4).float() / 1024
            )
            self.calls = []

        def forward(self, *, input_ids, attention_mask, use_cache):
            assert (
                use_cache is False and attention_mask.all() and input_ids.shape[0] == 1
            )
            self.calls.append(input_ids.tolist())
            values = self.embedding(input_ids)
            positions = torch.arange(1, len(input_ids[0]) + 1)[None, :, None]
            return SimpleNamespace(last_hidden_state=values.cumsum(1) / positions)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Backbone()
            self.lm_head = torch.nn.Linear(4, 256, bias=False)
            self.lm_head.weight.data.copy_(
                torch.arange(1024).reshape(256, 4).float() / 100
            )

    return adapter.BruvAgent(Model(), Tokenizer(), 2)


def test_original_scoring_math_and_systemone_mapping():
    import numpy as np

    agent = make_agent()
    result = agent.system_one({"message": "Refund please", "amount": 10}, QUESTIONS)
    rows = agent.model.model.calls
    assert len(rows) == 3
    for index, (kind, count) in enumerate((("choice", 2), ("noul", 2), ("score", 3))):
        ids = np.array(rows[index][0], dtype=np.int64)
        hidden = np.array(
            [[(int(t) * 4 + j) / 1024 for j in range(4)] for t in ids], dtype=np.float32
        ).sum(0) / len(ids)
        logits = np.array(
            [
                sum(float(hidden[j]) * ((letter * 4 + j) / 100) for j in range(4))
                for letter in range(65, 65 + count)
            ],
            dtype=np.float64,
        )
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        answer = result["answers"][kind]
        observed = (
            [answer["noul"]]
            if kind == "noul"
            else list(answer["probabilities"].values())
        )
        expected = [probabilities[1]] if kind == "noul" else probabilities
        np.testing.assert_allclose(observed, expected, atol=2e-6, rtol=0)
    score = result["answers"]["score"]
    assert score["legend"] == {"0": "Low", "1": "Medium", "2": "High"}
    assert score["score"] == pytest.approx(
        sum(i * p for i, p in enumerate(score["probabilities"].values()))
    )
    assert result["metadata"]["temperature"] == 1.0
    assert result["metadata"]["use_cache"] is False
    assert result["metadata"]["systemone_mapping"] == "bruv_joint_v1"


def test_prompt_is_author_json_and_no_thinking():
    agent = make_agent()
    state = ["中文", {"a": True}]
    ids = agent._encode(state, "Pick.", ["No", "Yes"])
    text = bytes(ids).decode("utf-8")
    assert text.startswith(adapter.DIRECT_SYSTEM + "\n")
    payload = json.loads(text.split("\n")[1])
    assert payload == {
        "evidence": state,
        "criterion": "Pick.",
        "options": [
            {"letter": "A", "description": "No"},
            {"letter": "B", "description": "Yes"},
        ],
    }


@pytest.mark.parametrize(
    "question",
    [
        {"type": "choice", "instructions": "q", "criteria": ["a"]},
        {
            "type": "choice",
            "instructions": "q",
            "criteria": [str(i) for i in range(17)],
        },
        {"type": "choice", "instructions": "q", "criteria": ["a", "a"]},
        {"type": "choice", "instructions": "q", "criteria": ["", "b"]},
        {"type": "choice", "instructions": "q", "criteria": {"a": {}, "b": None}},
        {"type": "noul", "instructions": "q", "criteria": {"true": 1}},
        {"type": "noul", "instructions": "q", "criteria": {"maybe": "x"}},
        {"type": "score", "instructions": "q", "criteria": ["x"] * 11},
        {"type": "score", "instructions": "q", "criteria": ["x", 1]},
        {"type": "score", "instructions": "q", "criteria": ["x", ""]},
        {"type": "noul", "instructions": ""},
        {"type": "noul", "instructions": None},
        {"type": "noul", "instructions": "q", "isolated": False},
        {"type": "unknown", "instructions": "q"},
        None,
    ],
)
def test_question_rejection_before_forward(question):
    agent = make_agent()
    with pytest.raises(ValueError):
        agent.system_one("s", {"valid": QUESTIONS["noul"], "invalid": question})
    assert agent.model.model.calls == []


@pytest.mark.parametrize(
    "state",
    [1, None, float("nan"), {"a": float("nan")}, "a" * 50001],
    ids=["integer", "null", "nan", "json_nan", "overlong"],
)
def test_state_rejection(state):
    agent = make_agent()
    with pytest.raises(ValueError):
        agent.system_one(state, QUESTIONS)
    assert not agent.model.model.calls


def test_full_request_limit_preflight():
    agent = make_agent()
    oversized = {**QUESTIONS["noul"], "instructions": "a" * 2000}
    with pytest.raises(ValueError, match="1024"):
        agent.system_one("s", {"valid": QUESTIONS["noul"], "late_invalid": oversized})
    assert not agent.model.model.calls
    for questions in (
        {},
        {"": QUESTIONS["noul"]},
        {str(i): QUESTIONS["noul"] for i in range(65)},
        {
            str(i): {
                "type": "choice",
                "instructions": "q",
                "criteria": [str(j) for j in range(16)],
            }
            for i in range(17)
        },
    ):
        with pytest.raises(ValueError):
            agent.system_one("s", questions)
    assert not agent.model.model.calls


def test_repeated_and_closed_model():
    agent = make_agent()
    request = {"first": QUESTIONS["noul"], "same": copy.deepcopy(QUESTIONS["noul"])}
    first = agent.system_one("s", request)
    assert len(agent.model.model.calls) == 1
    assert first["answers"]["first"] == first["answers"]["same"]
    second = agent.system_one("s", request)
    assert len(agent.model.model.calls) == 2
    assert first["answers"] == second["answers"]
    agent.close()
    agent.close()
    with pytest.raises(RuntimeError, match="closed"):
        agent.system_one("s", request)


def test_fp32_is_enforced():
    agent = make_agent()
    agent.model.half()
    with pytest.raises(ValueError, match="FP32"):
        agent.system_one("s", QUESTIONS)


def test_nonfinite_logits_are_rejected():
    agent = make_agent()
    agent.model.lm_head.weight.data[65, 0] = float("nan")
    with pytest.raises(RuntimeError, match="nonfinite"):
        agent.system_one("s", QUESTIONS)


@pytest.mark.parametrize(
    "config",
    [
        {"layout": "instructions-first"},
        {"state_template": "{state}"},
        {"instruction_template": "{instructions}"},
        {"context_order": ["state", "instructions", "criteria"]},
    ],
)
def test_native_template_only(config):
    with pytest.raises(ValueError, match="native"):
        adapter.validate_template(DecisionTemplate(config))
    adapter.validate_template(DecisionTemplate({}))
    adapter.validate_template(DecisionTemplate({"layout": "native"}))


@pytest.mark.parametrize("threads", [True, 0, 5, -1, 1.5])
def test_thread_guard_precedes_source(threads):
    with pytest.raises(ValueError, match="threads"):
        adapter.load_agent(Path("does-not-exist"), threads=threads)


def test_release_identity_and_source_integrity(tmp_path, monkeypatch):
    config = {
        "model_type": "qwen3_5_text",
        "architectures": ["Qwen3_5ForCausalLM"],
        "hidden_size": 1024,
        "num_hidden_layers": 24,
        "vocab_size": 248320,
        "tie_word_embeddings": True,
    }
    data = {
        "config.json": json.dumps(config).encode(),
        "model.safetensors": b"fixed-test-weights",
    }
    pins = {name: hashlib.sha256(value).hexdigest() for name, value in data.items()}
    monkeypatch.setattr(adapter, "SOURCE_HASHES", pins)
    for name, value in data.items():
        (tmp_path / name).write_bytes(value)
    assert adapter.verify_source(tmp_path) == config
    assert adapter.identify_source(tmp_path) == adapter.MODEL_ID
    with pytest.raises(ValueError, match="unsupported"):
        adapter.verify_source(tmp_path, "other/model")
    (tmp_path / "evil.py").write_text("raise Exception()")
    with pytest.raises(ValueError, match="unexpected"):
        adapter.verify_source(tmp_path)
    (tmp_path / "evil.py").unlink()
    (tmp_path / "model.safetensors").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        adapter.verify_source(tmp_path)


def test_reference_has_independent_pins_and_product_import_block():
    spec = importlib.util.spec_from_file_location(
        "bruv_test_reference", ROOT / "scripts/windows_bruv_reference.py"
    )
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    assert reference.SOURCE_HASHES == adapter.SOURCE_HASHES
    assert reference.SDK_REVISION == "d10751e8fd3c2a917fea81fb350593eeed3c482d"
    with pytest.raises(ImportError, match="forbidden"):
        reference.BlockProductImports().find_spec("vllm_jev.windows_bruv")
    records = reference.mapped_records({"state": "s", "questions": QUESTIONS})
    assert records[0]["options"] == [
        {"label": "A", "description": "billing: Refund"},
        {"label": "B", "description": "other"},
    ]
    assert records[1]["options"] == [
        {"label": "A", "description": "No"},
        {"label": "B", "description": "Yes: Refund requested"},
    ]
    assert records[2]["options"][2] == {"label": "C", "description": "High"}
