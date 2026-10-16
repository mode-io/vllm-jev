"""RSI release identity, no-truncation and independently checked readout rules."""

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm_jev import windows_rsi as adapter
from vllm_jev.decision_template import ENVIRONMENT, DecisionTemplate, current_template

QUESTIONS = {
    "choice": {
        "type": "choice",
        "instructions": "Route?",
        "criteria": {"billing": "Refund", "tech": None},
    },
    "noul": {"type": "noul", "instructions": "Refund?"},
    "score": {
        "type": "score",
        "instructions": "Urgency?",
        "criteria": ["low", "medium", "high"],
    },
}


@pytest.fixture(autouse=True)
def native_template(monkeypatch):
    monkeypatch.delenv(ENVIRONMENT, raising=False)
    current_template.cache_clear()
    yield
    current_template.cache_clear()


class Tokenizer:
    def __call__(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return {"input_ids": [ord(c) % 250 + 1 for c in text]}


@pytest.mark.parametrize("config", [{}, {"layout": "native"}])
def test_native_templates_only(config):
    adapter.validate_template(DecisionTemplate(config))


@pytest.mark.parametrize(
    "config",
    [
        {"layout": "instructions-first"},
        {"state_template": "prefix {state}"},
        {"instruction_template": "{instructions}!"},
        {"context_order": ["state", "instructions", "criteria"]},
    ],
)
def test_modified_templates_rejected(config):
    with pytest.raises(ValueError, match="native"):
        adapter.validate_template(DecisionTemplate(config))


def test_author_layout_and_spans():
    tokens = Tokenizer()
    row = adapter.encode_question(tokens, "状态", "问?", ["a", "b"], ["退款", ""])
    pieces = ["状态\n\n问?\nOptions:\n", "\n- a: 退款", "\n- b", "\n\nAnswer:"]
    ids = sum([tokens(p)["input_ids"] for p in pieces], [])
    starts = [len(pieces[0]), len(pieces[0]) + len(pieces[1])]
    ends = [starts[0] + len(pieces[1]), starts[1] + len(pieces[2])]
    assert row == (ids, list(zip(starts, ends)), len(ids) - 1)


def test_empty_state_has_no_leading_blank_lines():
    row = adapter.encode_question(Tokenizer(), "  ", "Q", ["a", "b"], ["", ""])
    assert row[0][0] == Tokenizer()("Q")["input_ids"][0]


def test_complete_prompt_boundary_and_no_truncation(monkeypatch):
    original = adapter.encode_question(Tokenizer(), "s", "Q", ["a", "b"], ["", ""])
    monkeypatch.setattr(adapter, "MAX_SEQUENCE_TOKENS", len(original[0]))
    assert (
        adapter.encode_question(Tokenizer(), "s", "Q", ["a", "b"], ["", ""]) == original
    )
    with pytest.raises(ValueError, match="truncation"):
        adapter.encode_question(Tokenizer(), "ss", "Q", ["a", "b"], ["", ""])


def test_independent_question_semantics():
    parsed = adapter.parse_questions(QUESTIONS)
    assert parsed[0] == (
        "choice",
        "choice",
        "Route?",
        ["billing", "tech"],
        ["Refund", ""],
    )
    assert parsed[1][3:] == (["false", "true"], ["No", "Yes"])
    assert parsed[2][3:] == (["0", "1", "2"], ["low", "medium", "high"])
    q = {
        "x": {
            "type": "choice",
            "instructions": {"task": "路由"},
            "criteria": {"a": {"值": 1}, "b": ""},
        }
    }
    assert adapter.parse_questions(q)[0][2:] == (
        '{"task":"路由"}',
        ["a", "b"],
        ['{"值":1}', ""],
    )


@pytest.mark.parametrize(
    "questions",
    [
        {},
        {str(i): QUESTIONS["noul"] for i in range(65)},
        {
            "q": {
                "type": "choice",
                "instructions": "?",
                "criteria": {str(i): "" for i in range(81)},
            }
        },
        {"q": {"type": "choice", "instructions": "?", "criteria": ["same", "same"]}},
        {"q": {"type": "choice", "instructions": "?", "criteria": ["", "b"]}},
        {"q": {"type": "score", "instructions": "?", "criteria": list(range(11))}},
        {"q": {"type": "noul", "instructions": "?", "criteria": {"maybe": "?"}}},
        {"q": {"type": "noul", "instructions": "?", "isolated": False}},
        {"q": {"type": "choice", "criteria": {"a": None, "b": None}}},
        {"q": {"type": "other", "instructions": "?"}},
    ],
)
def test_rejected_question_contracts(questions):
    with pytest.raises(ValueError):
        adapter.parse_questions(questions)


def test_candidate_budget():
    question = {
        "type": "choice",
        "instructions": "?",
        "criteria": {str(i): "" for i in range(64)},
    }
    assert len(adapter.parse_questions({str(i): question for i in range(4)})) == 4
    with pytest.raises(ValueError, match="256"):
        adapter.parse_questions({str(i): question for i in range(5)})


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_json_rejected(bad):
    q = copy.deepcopy(QUESTIONS)
    q["score"]["criteria"][0] = bad
    with pytest.raises(ValueError):
        adapter.parse_questions(q)


def test_checkpoint_identity_rejects_extra_and_modified_files(tmp_path, monkeypatch):
    recipe = {
        "readout": "option_xattn",
        "readout_layer": -1,
        "layout": "state_first",
        "option_pool": "mean",
        "max_options": 80,
        "residual": False,
        "logit_cap": None,
        "head_input_norm": False,
        "arch_extra": {},
    }
    (tmp_path / "base").mkdir()
    values = {
        "meta.json": json.dumps({"base_model": adapter.BASE_ID, "spec": recipe}),
        "base/config.json": "{}",
    }
    for name, value in values.items():
        (tmp_path / name).write_text(value, encoding="utf-8")
    hashes = {
        name: hashlib.sha256(value.encode()).hexdigest()
        for name, value in values.items()
    }
    monkeypatch.setattr(adapter, "SOURCE_HASHES", hashes)
    assert adapter.identify_source(tmp_path) == adapter.MODEL_ID
    (tmp_path / "evil.py").write_text("raise RuntimeError", encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected"):
        adapter.verify_source(tmp_path)
    (tmp_path / "evil.py").unlink()
    (tmp_path / "base/config.json").write_text("bad", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        adapter.verify_source(tmp_path)


def tiny_agent():
    import torch

    class Tower(torch.nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(81)
            self.embedding = torch.nn.Embedding(256, 8)
            self.calls = []

        def forward(self, input_ids, attention_mask, use_cache, return_dict):
            assert use_cache and return_dict and attention_mask.all()
            self.calls.append(input_ids)
            return SimpleNamespace(
                last_hidden_state=self.embedding(input_ids).cumsum(1)
            )

    return adapter.RsiAgent(Tower(), Tokenizer(), adapter.make_scorer(8))


def test_all_rows_preflight_before_forward():
    agent = tiny_agent()
    q = copy.deepcopy(QUESTIONS)
    q["score"]["instructions"] = "long " * 1000
    with pytest.raises(ValueError, match="complete prompt"):
        agent.system_one("refund", q)
    assert agent.tower.calls == []


def test_readout_matches_original_release_equations():
    import torch
    import torch.nn.functional as F

    agent = tiny_agent()
    row = adapter.encode_question(
        agent.tokenizer, "refund", "Route?", ["a", "b"], ["refund", "tech"]
    )
    ids, spans, decision = row
    h = agent.tower.embedding(torch.tensor([ids])).cumsum(1)
    # Independent per-option slice means and explicit projected multi-head attention.
    options = torch.zeros(1, 80, 8)
    for i, (a, b) in enumerate(spans):
        options[0, i] = h[0, a:b].mean(0)
    mask = torch.arange(80).view(1, -1) < 2
    scorer = agent.scorer
    q, k, v = (
        scorer.q(h[:, decision]).unsqueeze(1),
        scorer.k(options),
        scorer.v(options),
    )
    w, b = scorer.attn.in_proj_weight, scorer.attn.in_proj_bias
    pq, pk, pv = [
        F.linear(x, wi, bi) for x, wi, bi in zip((q, k, v), w.chunk(3), b.chunk(3))
    ]
    pq = pq.view(1, 1, 4, 2).transpose(1, 2)
    pk, pv = [x.view(1, 80, 4, 2).transpose(1, 2) for x in (pk, pv)]
    attention = torch.softmax(
        ((pq @ pk.transpose(-2, -1)) / (2**0.5)).masked_fill(
            ~mask[:, None, None, :], float("-inf")
        ),
        -1,
    )
    ctx = (attention @ pv).transpose(1, 2).reshape(1, 1, 8)
    ctx = scorer.attn.out_proj(ctx)
    expected = torch.softmax(scorer.proj(v + ctx)[0, :2, 0], -1)
    assert agent._probabilities(*row) == pytest.approx(expected.tolist(), abs=1e-6)


def test_repeated_rows_deduplicated_and_close():
    agent = tiny_agent()
    q = {"a": QUESTIONS["choice"], "b": QUESTIONS["choice"]}
    result = agent.system_one("refund", q)
    assert result["answers"]["a"] == result["answers"]["b"]
    assert len(agent.tower.calls) == 1
    agent.close()
    agent.close()
    with pytest.raises(RuntimeError, match="closed"):
        agent.system_one("refund", q)


def test_independent_reference_pins():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        spec = importlib.util.spec_from_file_location(
            "_test_rsi_ref", scripts / "windows_rsi_reference.py"
        )
        ref = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ref)
    finally:
        sys.path.remove(str(scripts))
    assert ref.SOURCE_HASHES == adapter.SOURCE_HASHES
    assert ref.REVISION == adapter.REVISION
    assert len(ref.SDK_HASHES) == 7
