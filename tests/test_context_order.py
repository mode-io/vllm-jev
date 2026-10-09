"""Logical order, protected readout, and protocol compatibility checks."""

import itertools
import json

import pytest
from test_model_template_layouts import Tokens, text

from vllm_jev.decision_template import ENVIRONMENT, DecisionTemplate, current_template

ORDERS = list(itertools.permutations(["state", "instructions", "criteria"]))


@pytest.fixture(autouse=True)
def reset(monkeypatch):
    monkeypatch.delenv(ENVIRONMENT, raising=False)
    current_template.cache_clear()
    yield
    current_template.cache_clear()


def configure(monkeypatch, order):
    monkeypatch.setenv(ENVIRONMENT, json.dumps({"context_order": list(order)}))
    current_template.cache_clear()


@pytest.mark.parametrize("order", ORDERS)
def test_order_is_real_and_final_cue_stays_late(monkeypatch, order):
    from vllm_jev.prompt import branch_token_ids, candidate_prompts, tiny_token_ids

    configure(monkeypatch, order)
    values = {"state": "EVIDENCE", "instructions": "POLICY", "criteria": "LEFT"}
    rendered = candidate_prompts("EVIDENCE", "POLICY", ["LEFT", "RIGHT"])[0]
    assert [rendered.index(values[k]) for k in order] == sorted(
        rendered.index(values[k]) for k in order
    )
    assert rendered.endswith("Answer Yes or No.")
    for row in branch_token_ids(Tokens(), "EVIDENCE", "POLICY", ["LEFT", "RIGHT"]):
        rendered = text(row)
        assert [rendered.index(values[k]) for k in order] == sorted(
            rendered.index(values[k]) for k in order
        )
    row, slots = tiny_token_ids(Tokens(), "EVIDENCE", "POLICY", ["LEFT", "RIGHT"])
    rendered = text(row)
    assert [rendered.index(values[k]) for k in order] == sorted(
        rendered.index(values[k]) for k in order
    )
    assert all(pos > rendered.index("EVIDENCE") for pos in slots)


@pytest.mark.parametrize(
    "bad",
    [
        None,
        [],
        ["state"],
        ["state", "state", "criteria"],
        ["state", "instructions", "unknown"],
        [["state"], "instructions", "criteria"],
        "state",
    ],
)
def test_invalid_orders_rejected(bad):
    with pytest.raises(ValueError, match="context_order"):
        DecisionTemplate({"context_order": bad})


@pytest.mark.parametrize(
    "extra",
    [
        {"layout": "native"},
        {"layout": "instructions-first"},
        {"choice_prompt": "{state} {instructions} {candidate}"},
        {"noul_prompt": "{state} {instructions}"},
    ],
)
def test_ambiguous_controls_rejected(extra):
    with pytest.raises(ValueError, match="cannot be combined"):
        DecisionTemplate(
            {"context_order": ["instructions", "criteria", "state"], **extra}
        )


@pytest.mark.parametrize("order", ORDERS)
def test_clef_all_orders_preserve_state_conditioned_readouts(monkeypatch, order):
    from vllm_jev.clef import encode

    configure(monkeypatch, order)
    qs = {
        "q": {
            "type": "choice",
            "instructions": "POLICY",
            "criteria": {"a": "LEFT", "b": "RIGHT"},
        }
    }
    ids, encoded = encode(Tokens(), qs, "EVIDENCE", [100001], 4096)
    rendered = text(ids)
    values = {"state": "EVIDENCE", "instructions": "POLICY", "criteria": "LEFT"}
    assert [rendered.index(values[k]) for k in order] == sorted(
        rendered.index(values[k]) for k in order
    )
    q = encoded[0]
    assert q.question_span[0] > max(rendered.index(v) for v in values.values())
    assert text(ids[slice(*q.question_span)]) == "POLICY"
    assert all(a > rendered.index("EVIDENCE") for a, b in q.option_spans)
    assert ids.count(100001) == 1


@pytest.mark.parametrize("order", ORDERS)
def test_laya_marker_offsets_and_embedded_separator(monkeypatch, order):
    from vllm_jev.laya_template import ContextOrderMixin

    configure(monkeypatch, order)

    class Native:
        def _encode_state(self, state, qids, questions):
            return [
                {
                    "ids": [1, 10, 2, 3, 2, 20, 3, 21, 2]
                    + ([30, 2, 31] if state else [])
                    + [2],
                    "markers": [3, 6],
                    "qtype": 0,
                }
            ]

    class Protocol(ContextOrderMixin, Native):
        pass

    item = Protocol()._encode_state("data", ["q"], {})[0]
    assert len(item["ids"]) == 13
    assert [item["ids"][p] for p in item["markers"]] == [3, 3]
    assert item["ids"].count(2) == 5
    native = Native()._encode_state("data", ["q"], {})[0]
    assert sorted(item["ids"]) == sorted(native["ids"])


@pytest.mark.parametrize("order", ORDERS)
def test_native_label_protocols_and_late_pointer_positions(monkeypatch, order):
    from vllm_jev.decision import DecisionCompiler
    from vllm_jev.decision_protocols import (
        jevk5_prompt,
        mica_prompt,
        packed_slots,
        task_prompt,
    )

    configure(monkeypatch, order)
    values = {"state": "EVIDENCE", "instructions": "POLICY", "criteria": "LEFT"}
    tok = Tokens()
    rendered = [
        task_prompt(tok, "EVIDENCE", "POLICY", ["a", "b"], ["LEFT", "RIGHT"]),
        jevk5_prompt(tok, "EVIDENCE", "POLICY", ["LEFT", "RIGHT"]),
        mica_prompt(
            tok,
            "EVIDENCE",
            "POLICY",
            ["a", "b"],
            ["LEFT", "RIGHT"],
            "choice",
            ["A", "B"],
        ),
    ]
    for value in rendered:
        assert [value.index(values[k]) for k in order] == sorted(
            value.index(values[k]) for k in order
        )
        assert value.endswith("<assistant>")
    row, slots = packed_slots(
        tok,
        "EVIDENCE",
        [("POLICY", ["LEFT", "RIGHT"]), ("SECOND", ["UP", "DOWN"])],
        ["A", "B"],
        [100, 101],
    )
    assert all(p > text(row).index("EVIDENCE") for p in slots)
    service = object.__new__(DecisionCompiler)
    service.tokenizer = tok
    service.protocol = "kev_pointer_v1"
    service.special = [100001, 100002, 100003, 100004, 100005]
    service.max_length = 4096
    row, slots, _ = service._row(service._base("EVIDENCE"), "POLICY", ["LEFT", "RIGHT"])
    assert [row[p] for p in slots] == [100004, 100004, 100005]
    assert row.count(100001) == 1
    assert all(p > text(row).index("EVIDENCE") for p in slots)


@pytest.mark.parametrize("order", ORDERS)
def test_rsi_mean_spans_stay_after_all_context(monkeypatch, order):
    from vllm_jev.rsijev import encode_question

    configure(monkeypatch, order)
    row, spans, pos = encode_question(
        Tokens(), "EVIDENCE", "POLICY", ["a", "b"], {"a": "LEFT", "b": "RIGHT"}, 4096
    )
    rendered = text(row)
    values = {"state": "EVIDENCE", "instructions": "POLICY", "criteria": "LEFT"}
    assert [rendered.index(values[k]) for k in order] == sorted(
        rendered.index(values[k]) for k in order
    )
    assert [text(row[a:b]) for a, b in spans] == ["\n- a: LEFT", "\n- b: RIGHT"]
    assert min(a for a, b in spans) > max(rendered.index(v) for v in values.values())
    assert pos == len(row) - 1
    with pytest.raises(ValueError, match="ordered"):
        encode_question(
            Tokens(), "EVIDENCE", "POLICY", ["a", "b"], {"a": "LEFT", "b": "RIGHT"}, 32
        )


@pytest.mark.parametrize("order", ORDERS)
def test_valen_keeps_leading_system_and_ordered_typed_messages(monkeypatch, order):
    import torch

    from vllm_jev.valen import ValenService

    configure(monkeypatch, order)
    seen = []

    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            seen.extend(messages)
            rendered = "\n".join(p["text"] for m in messages for p in m["content"])
            return {"input_ids": torch.tensor([Tokens().encode(rendered)])}

    service = object.__new__(ValenService)
    service.processor = Processor()
    service.tokenizer = Tokens()
    service.media_kwargs = {}
    ids, media = service._state(
        {
            "messages": [
                {"role": "system", "content": "SYSTEM"},
                {"role": "user", "content": "EVIDENCE"},
            ]
        },
        context_fields={"instructions": "POLICY", "criteria": "LEFT"},
    )
    assert seen[0]["role"] == "system" and seen[0]["content"][0]["text"] == "SYSTEM"
    values = {"state": "EVIDENCE", "instructions": "POLICY", "criteria": "LEFT"}
    assert [text(ids).index(values[k]) for k in order] == sorted(
        text(ids).index(values[k]) for k in order
    )
    assert media.base_length == len(ids)


def test_valen_context_criteria_use_supplied_values(monkeypatch):
    from types import SimpleNamespace

    from vllm_jev.valen import ValenMedia, ValenService

    configure(monkeypatch, ["instructions", "criteria", "state"])
    service = object.__new__(ValenService)
    service.model_names = ["test"]
    service.max_length = 4096
    service.tokenizer = Tokens()
    seen = []

    def state(value, **kwargs):
        seen.append(kwargs["context_fields"])
        return Tokens().encode("EVIDENCE"), ValenMedia(base_length=8)

    service._state = state
    payload = SimpleNamespace(
        model=None,
        state="EVIDENCE",
        questions={
            "route": {
                "type": "choice",
                "instructions": "POLICY",
                "criteria": {"a": "LEFT", "b": "RIGHT"},
            },
            "check": {"type": "noul", "instructions": "CONDITION"},
        },
    )
    rows, _, _, _ = service._compile(payload)
    criteria = json.loads(seen[0]["criteria"].split("\n", 1)[1])
    assert criteria == [
        {"question": 1, "type": "choice", "criteria": {"a": "LEFT", "b": "RIGHT"}},
        {"question": 2, "type": "noul", "criteria": None},
    ]
    assert "Task 2: noul" in seen[0]["instructions"]
    assert "True /" not in seen[0]["criteria"]
    assert "Question: CONDITION" in text(rows[1][3][0][0])
