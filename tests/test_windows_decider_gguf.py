"""CPU GGUF contracts without loading released weights."""

import copy
import hashlib
from types import SimpleNamespace

import pytest

from vllm_jev import windows_decider_gguf as adapter
from vllm_jev.decision_template import ENVIRONMENT, current_template


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


class Readout:
    threads = 4

    def __init__(self):
        self.calls, self.closed = [], 0

    def read(self, ids):
        import torch

        self.calls.append(ids)
        # A known nonuniform raw vector exposes temperature and Noul ordering.
        return torch.arange(255, dtype=torch.float32) / 10

    def close(self):
        self.closed += 1


@pytest.fixture
def agent(monkeypatch):
    monkeypatch.delenv(ENVIRONMENT, raising=False)
    current_template.cache_clear()
    agent = adapter.DeciderGGUFAgent(
        Tokenizer(), [(f"L{i}", i) for i in range(255)], Readout()
    )
    yield agent
    agent.close()
    current_template.cache_clear()


QUESTIONS = {
    "c": {"type": "choice", "instructions": "Route?", "criteria": ["a", "b"]},
    "n": {"type": "noul", "instructions": "True?"},
    "s": {"type": "score", "instructions": "Level?", "criteria": ["0: Low", "1: High"]},
}


def test_type_temperatures_and_isolated_score(agent):
    import math

    answer = agent.system_one("Text", QUESTIONS)["answers"]
    assert answer["c"]["probabilities"]["b"] == pytest.approx(
        1 / (1 + math.exp(-0.1 / 1.110))
    )
    assert answer["n"]["noul"] == pytest.approx(1 / (1 + math.exp(-0.1 / 1.560)))
    assert answer["s"]["probabilities"] == {"0": 0.5, "1": 0.5}
    assert len(agent.readout.calls) == 4
    decoded = ["".join(map(chr, ids)) for ids in agent.readout.calls]
    assert "Proposed answer: Low\nDoes the proposed answer fit?" in decoded[2]
    assert "0: Low" not in decoded[2]


def test_score_uses_score_temperature_even_for_binary_rows(agent):
    import torch

    calls = []

    def varying(ids):
        calls.append(ids)
        logits = torch.zeros(255)
        logits[1] = len(calls)
        return logits

    agent.readout.read = varying
    actual = agent.system_one("Text", {"s": QUESTIONS["s"]})["answers"]["s"]
    fits = torch.sigmoid(torch.tensor([1.0, 2.0]) / 1.287).tolist()
    assert actual["probabilities"]["1"] == pytest.approx(fits[1] / sum(fits))


def test_explicit_nonisolated_score(agent):
    q = copy.deepcopy(QUESTIONS["s"])
    q["isolated"] = False
    actual = agent.system_one("Text", {"s": q})["answers"]["s"]
    assert actual["probabilities"]["1"] > 0.5
    assert len(agent.readout.calls) == 1
    assert "Proposed answer" not in "".join(map(chr, agent.readout.calls[0]))


def test_late_overflow_rejected_before_any_native_call(agent):
    questions = copy.deepcopy(QUESTIONS)
    questions["s"]["instructions"] = "x" * 1100
    with pytest.raises(ValueError, match="context"):
        agent.system_one("Text", questions)
    assert agent.readout.calls == []


def test_zero_fit_does_not_invent_uniform_probabilities(agent):
    import torch

    def read(ids):
        logits = torch.zeros(255)
        logits[1] = -10000
        return logits

    agent.readout.read = read
    actual = agent.system_one("Text", {"s": QUESTIONS["s"]})["answers"]["s"]
    assert actual["probabilities"] == {"0": 0.0, "1": 0.0}
    assert actual["confidence"] == actual["score"] == 0.0


def test_close_is_idempotent_and_disables_inference(agent):
    readout = agent.readout
    agent.close()
    agent.close()
    assert readout.closed == 1
    with pytest.raises(RuntimeError, match="closed"):
        agent.system_one("Text", QUESTIONS)


def test_native_partial_allocation_cleanup():
    events = []
    native = adapter.NativeReadout.__new__(adapter.NativeReadout)
    native.L = SimpleNamespace(
        llama_batch_free=lambda x: events.append(("batch", x)),
        llama_free=lambda x: events.append(("context", x)),
        llama_model_free=lambda x: events.append(("model", x)),
    )
    native.batch, native.context, native.model = None, None, "owned"
    native.close()
    native.close()
    assert events == [("model", "owned")]


def test_non_native_template_is_rejected():
    with pytest.raises(ValueError, match="native"):
        adapter.validate_template(
            SimpleNamespace(config={"layout": "instructions-first"})
        )


def test_dispatch_is_distinct_from_bf16_and_other_gguf(monkeypatch, tmp_path):
    from vllm_jev.windows_models import identify_source, protocol_for

    (tmp_path / adapter.GGUF_FILENAME).write_bytes(b"routing hint")
    monkeypatch.setattr(adapter, "verify_source", lambda *a: {})
    assert identify_source(tmp_path) == adapter.MODEL_ID
    assert protocol_for(adapter.MODEL_ID) == adapter.PROTOCOL


def test_source_allowlist_and_bytes_are_checked_before_loading(monkeypatch, tmp_path):
    content = {"decider_config.json": b"{}", adapter.GGUF_FILENAME: b"GGUF fixture"}
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in content.items()}
    monkeypatch.setattr(adapter, "SOURCE_HASHES", hashes)
    for name, data in content.items():
        (tmp_path / name).write_bytes(data)
    assert adapter.verify_source(tmp_path) == {}
    (tmp_path / adapter.GGUF_FILENAME).write_bytes(b"wrong release")
    with pytest.raises(ValueError, match="checksum"):
        adapter.verify_source(tmp_path)
    (tmp_path / adapter.GGUF_FILENAME).write_bytes(content[adapter.GGUF_FILENAME])
    (tmp_path / "downloaded_model.py").write_text("raise RuntimeError()")
    with pytest.raises(ValueError, match="unexpected"):
        adapter.verify_source(tmp_path)


@pytest.mark.parametrize("override", ["", "unverified.dll"])
def test_runtime_override_rejected_before_backend_import(monkeypatch, override):
    monkeypatch.setattr(
        adapter.importlib.metadata,
        "distribution",
        lambda _: SimpleNamespace(version="0.3.35"),
    )
    monkeypatch.setenv("LLAMA_CPP_LIB_PATH", override)
    with pytest.raises(ValueError, match="overrides"):
        adapter.verify_runtime()


@pytest.mark.parametrize("threads", [0, 5, True])
def test_threads_rejected_before_load(tmp_path, threads):
    with pytest.raises(ValueError, match="threads"):
        adapter.load_agent(tmp_path, threads=threads)
