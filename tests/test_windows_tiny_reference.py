"""Independent-reference integrity checks; no released weights are loaded."""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "_tiny_reference_test", ROOT / "scripts/windows_tiny_reference.py"
)
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


def test_reference_import_boundary():
    finder = reference.BlockProductImports()
    for name in ("vllm_jev", "vllm_jev.windows_tiny", "vllm_jev.prompt"):
        with pytest.raises(ImportError, match="forbidden"):
            finder.find_spec(name)
    assert finder.find_spec("transformers") is None


def test_reference_executes_only_hash_verified_author_file(tmp_path, monkeypatch):
    file = tmp_path / "modeling_tiny_jev.py"
    file.write_text("raise RuntimeError('untrusted source executed')", encoding="utf-8")
    monkeypatch.setattr(reference, "SOURCE_HASHES", {file.name: "0" * 64})
    with pytest.raises(ValueError, match="unverified"):
        reference.original_module(tmp_path)


def test_source_pin_rejects_each_modified_component(tmp_path, monkeypatch):
    pins = {}
    for name in reference.SOURCE_HASHES:
        file = tmp_path / name
        file.write_bytes(name.encode())
        pins[name] = reference.digest(file)
    monkeypatch.setattr(reference, "SOURCE_HASHES", pins)
    assert reference.verify(tmp_path) == pins
    for name in pins:
        file = tmp_path / name
        before = file.read_bytes()
        file.write_bytes(before + b"altered")
        with pytest.raises(ValueError, match="pinned"):
            reference.verify(tmp_path)
        file.write_bytes(before)


def test_author_mapping_preserves_unicode_order_and_custom_noul():
    question = {
        "type": "choice",
        "instructions": {"中文": "选择", "a": 1},
        "criteria": {"乙": {"z": False, "a": ["中文"]}, "甲": "Second"},
    }
    value = reference.author_definition(question)
    assert list(value["criteria"]) == ["乙", "甲"]
    assert value["criteria"]["乙"] == '{"a": ["中文"], "z": false}'
    assert value["instructions"] == '{"a": 1, "中文": "选择"}'
    assert question["instructions"] == {"中文": "选择", "a": 1}
    noul = reference.author_definition(
        {
            "type": "noul",
            "instructions": "Refund?",
            "criteria": {"false": "No request", "true": "Request"},
        }
    )
    assert noul == {
        "kind": "noul",
        "instructions": "Refund?\nYes means: Request\nNo means: No request",
    }


def test_reference_observer_reads_author_positions_and_logits():
    import torch

    row = reference.observe_forward(
        None,
        (
            torch.tensor([[10, 20, 30, 40]]),
            torch.ones(1, 4, dtype=torch.long),
            torch.tensor([[1, 3]]),
            torch.tensor([[True, True]]),
        ),
        torch.tensor([[1.0, -2.0]]),
    )
    assert row == {
        "input_ids": [10, 20, 30, 40],
        "positions": [1, 3],
        "calibrated_logits": [1.0, -2.0],
    }
    with pytest.raises(RuntimeError, match="unpadded"):
        reference.observe_forward(
            None,
            (
                torch.tensor([[10, 20]]),
                torch.tensor([[1, 0]]),
                torch.tensor([[0]]),
                torch.tensor([[True]]),
            ),
            torch.tensor([[1.0]]),
        )


@pytest.mark.skipif(sys.platform != "win32", reason="Windows memory counters")
def test_reference_lifetime_memory_counters_include_current_usage():
    sample = reference.process_memory()
    assert sample["peak_working_set_bytes"] >= sample["working_set_bytes"] > 0
    assert sample["peak_commit_bytes"] >= sample["private_bytes"] > 0
