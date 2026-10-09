"""Protocol-level layout checks; these do not measure model accuracy."""

import json

import pytest

from vllm_jev.decision_template import ENVIRONMENT, current_template


class Tokens:
    unk_token_id = -1

    def encode(self, text, **kwargs):
        return list(map(ord, text))

    def __call__(self, text, **kwargs):
        class Result(dict):
            @property
            def input_ids(self):
                return self["input_ids"]

        return Result(input_ids=self.encode(text))

    def apply_chat_template(self, messages, **kwargs):
        return "".join(m["content"] for m in messages) + "<assistant>"

    def convert_tokens_to_ids(self, text):
        return 100000


def text(ids):
    return "".join(chr(i) for i in ids)


@pytest.fixture(autouse=True)
def layout(monkeypatch):
    monkeypatch.setenv(ENVIRONMENT, '{"layout":"instructions-first"}')
    current_template.cache_clear()
    yield
    current_template.cache_clear()


def test_scalar_and_marker_prompts_keep_readouts_after_evidence():
    from vllm_jev.prompt import branch_token_ids, tiny_token_ids

    tok = Tokens()
    rows = branch_token_ids(tok, "EVIDENCE", "POLICY", ["LEFT", "RIGHT"])
    for row, candidate in zip(rows, ["LEFT", "RIGHT"]):
        rendered = text(row)
        assert (
            rendered.index("POLICY")
            < rendered.index("EVIDENCE")
            < rendered.index(candidate)
        )
        assert rendered.endswith("<assistant>")
    row, positions = tiny_token_ids(tok, "EVIDENCE", "POLICY", ["LEFT", "RIGHT"])
    assert text(row).index("POLICY") < text(row).index("EVIDENCE")
    assert all(p > text(row).index("EVIDENCE") for p in positions)
    assert [row[p] for p in positions] == [ord("?"), ord("?")]


def test_json_and_mica_layouts_keep_native_answers():
    from vllm_jev.decision_protocols import jevk5_prompt, mica_prompt, task_prompt

    tok = Tokens()
    for rendered in [
        task_prompt(tok, "EVIDENCE", "POLICY", ["x", "y"], ["LEFT", "RIGHT"]),
        jevk5_prompt(tok, "EVIDENCE", "POLICY", ["LEFT", "RIGHT"]),
        mica_prompt(
            tok,
            "EVIDENCE",
            "POLICY",
            ["x", "y"],
            ["LEFT", "RIGHT"],
            "choice",
            ["A", "B"],
        ),
    ]:
        assert rendered.index("POLICY") < rendered.index("EVIDENCE")
        assert rendered.endswith("<assistant>")


def test_packed_slots_remain_causal_and_numbered():
    from vllm_jev.decision_protocols import packed_slots

    row, slots = packed_slots(
        Tokens(),
        "EVIDENCE",
        [("FIRST", ["a", "b"]), ("SECOND", ["c", "d"])],
        ["A", "B"],
        [100, 101],
    )
    rendered = text(row)
    assert rendered.index("FIRST") < rendered.index("EVIDENCE")
    assert rendered.index("SECOND") < rendered.index("EVIDENCE")
    assert all(p > rendered.index("EVIDENCE") for p in slots)
    assert [rendered[: p + 1].splitlines()[-1] for p in slots] == [
        "Answer 1: (",
        "Answer 2: (",
    ]


def test_rsi_option_means_and_overflow_are_protected():
    from vllm_jev.rsijev import encode_question, encode_questions

    tok = Tokens()
    args = (tok, "EVIDENCE", "POLICY", ["a", "b"], {"a": "LEFT", "b": "RIGHT"}, 512)
    row, spans, decision = encode_question(*args)
    assert text(row).index("POLICY") < text(row).index("EVIDENCE")
    assert [text(row[a:b]) for a, b in spans] == ["\n- a: LEFT", "\n- b: RIGHT"]
    assert decision == len(row) - 1 and text(row).endswith("Answer:")
    assert encode_questions(
        tok, "EVIDENCE", [("q", "choice", "POLICY", args[3], args[4])], 512
    ) == [(row, spans, decision)]
    with pytest.raises(ValueError, match="instructions-first"):
        encode_question(*args[:-1], 32)


def test_clef_readout_schema_stays_after_evidence():
    from vllm_jev.clef import encode

    questions = {
        "q": {
            "type": "choice",
            "instructions": "POLICY",
            "criteria": {"a": "LEFT", "b": "RIGHT"},
        }
    }
    row, qs = encode(Tokens(), questions, "EVIDENCE", [100001, 100002], 4096)
    rendered = text(row)
    assert rendered.index("POLICY") < rendered.index("EVIDENCE")
    assert text(row[slice(*qs[0].question_span)]) == "POLICY"
    assert qs[0].question_span[0] > rendered.index("EVIDENCE")
    assert all(a > rendered.index("EVIDENCE") for a, _ in qs[0].option_spans)
    assert [json.loads(text(row[a:b])) for a, b in qs[0].option_spans] == [
        {"option_id": "a", "description": "LEFT"},
        {"option_id": "b", "description": "RIGHT"},
    ]
    assert row.count(100001) == row.count(100002) == 1
    assert rendered.endswith("JOINT SCHEMA DECISIONS:")


def test_kev_specials_and_slot_readout_remain_present():
    from vllm_jev.decision import DecisionCompiler

    service = object.__new__(DecisionCompiler)
    service.tokenizer = Tokens()
    service.protocol = "kev_pointer_v1"
    service.special = [100001, 100002, 100003, 100004, 100005]
    service.max_length = 512
    row, slots, n = service._row(service._base("EVIDENCE"), "POLICY", ["LEFT", "RIGHT"])
    assert row[0] == service.special[1]
    assert text(row).index("POLICY") < text(row).index("EVIDENCE")
    assert [row[p] for p in slots] == [
        service.special[3],
        service.special[3],
        service.special[4],
    ]
    assert n == 2


def test_valen_instruction_message_follows_system_and_precedes_state():
    import torch

    from vllm_jev.valen import ValenService

    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            assert [m["role"] for m in messages] == ["system", "user", "user"]
            assert messages[0]["content"][0]["text"] == "SYSTEM"
            rendered = "\n".join(p["text"] for m in messages for p in m["content"])
            assert rendered.index("POLICY") < rendered.index("EVIDENCE")
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
        "POLICY",
    )
    assert media.base_length == len(ids)
    assert text(ids) == "SYSTEM\nPOLICY\nEVIDENCE"


def test_vjev_keeps_noul_late_readout():
    from types import SimpleNamespace

    from vllm_jev.vjev import VjevService

    service = object.__new__(VjevService)
    service.model_names = ["test"]
    service.tokenizer = Tokens()
    service.max_length = 1024
    service._state = lambda _: ([], "EVIDENCE")
    payload = SimpleNamespace(
        model=None,
        state="EVIDENCE",
        questions={
            "n": {"type": "noul", "instructions": "POLICY"},
            "c": {
                "type": "choice",
                "instructions": "POLICY",
                "criteria": {"a": "LEFT", "b": "RIGHT"},
            },
        },
    )
    rows, _, _, _ = service._compile(payload)
    for _, kind, _, ids, slots in rows:
        rendered = text(ids)
        assert rendered.index("POLICY") < rendered.index("EVIDENCE")
        assert all(p > rendered.index("EVIDENCE") for p in slots)
        if kind == "noul":
            assert rendered.endswith("Question: POLICY")
