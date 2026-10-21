"""Decider CPU contracts; no released weights or network are used."""

import copy
import hashlib
import itertools
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm_jev import windows_decider as adapter
from vllm_jev.decision_template import ENVIRONMENT, current_template

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = {
    "route": {
        "type": "choice",
        "instructions": "Route?",
        "criteria": {"bill": "Refund", "tech": None},
    },
    "yes": {"type": "noul", "instructions": "Refund?"},
    "urgency": {
        "type": "score",
        "instructions": "Urgent?",
        "criteria": ["0: Later", "1: Soon", "2: Now"],
    },
}


@pytest.fixture(autouse=True)
def isolated_template(monkeypatch):
    monkeypatch.delenv(ENVIRONMENT, raising=False)
    current_template.cache_clear()
    yield
    current_template.cache_clear()


class Tokenizer:
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return [ord(c) % 1023 + 1 for c in text]


def make_agent(model_id=adapter.MODEL_ID):
    import torch

    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(810)
            self.embedding = torch.nn.Embedding(1024, 16, dtype=torch.bfloat16)
            self.calls = []

        def get_input_embeddings(self):
            return self.embedding

        def forward(self, *, input_ids, attention_mask, use_cache, return_dict):
            assert not use_cache and return_dict
            self.calls.append((input_ids.clone(), attention_mask.clone()))
            # A deterministic context-sensitive state makes readout-position and
            # independent-Score-row mistakes visible without a model download.
            embeddings = self.embedding(input_ids).float() * attention_mask[..., None]
            states = (
                embeddings.cumsum(1) / attention_mask.cumsum(1).clamp_min(1)[..., None]
            )
            return SimpleNamespace(last_hidden_state=states.bfloat16())

    model = Backbone()
    torch.manual_seed(812)
    weight = torch.randn(255, 16).bfloat16()
    manifest = {
        "max_length": 1024,
        "max_options": 255,
        "temperature": adapter.MODELS[model_id]["temperature"],
        "temperatures": dict(adapter.MODELS[model_id]["temperatures"]),
        "isolated_levels": True,
        "labels": list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + [f"L{i}" for i in range(229)],
        "label_token_ids": list(range(255)),
    }
    return adapter.DeciderAgent(model, Tokenizer(), weight, manifest, model_id)


def independent_probability(agent, state, instructions, options, temperature=1.03):
    """Independently render the author's narrow row and BF16 readout."""
    import torch
    import torch.nn.functional as F

    text = "Context:\n" + state
    text += "\n\nQuestion: " + instructions + "\nOptions:"
    text += "".join(f"\n({chr(65 + i)}) {v}" for i, v in enumerate(options))
    text += "\nAnswer: ("
    ids = torch.tensor(Tokenizer().encode(text))
    hidden = agent.model.embedding(ids).float().sum(0) / len(ids)
    logits = F.linear(hidden.bfloat16()[None], agent.weight).float()[0]
    logits[len(options) :] = -float("inf")
    return torch.softmax(logits / temperature, -1)[: len(options)].tolist(), ids


def test_lightweight_import_does_not_import_backends():
    code = "import sys; import vllm_jev.windows_decider; assert not any(n in sys.modules for n in ('torch','transformers','vllm','mlx'))"
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)


def test_compiler_import_does_not_import_vllm():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import vllm_jev.decision; assert 'vllm' not in sys.modules",
        ],
        cwd=ROOT,
        check=True,
    )


def test_three_types_preserve_bf16_projection_and_isolated_levels():
    agent = make_agent()
    state = "A duplicate charge."
    result = agent.system_one(state, QUESTIONS)
    route, _ = independent_probability(agent, state, "Route?", ["bill: Refund", "tech"])
    yes, _ = independent_probability(agent, state, "Refund?", ["no", "yes"])
    fits = [
        independent_probability(
            agent,
            state,
            f"Urgent?\nProposed answer: {level}\nDoes the proposed answer fit?",
            ["no", "yes"],
        )[0][1]
        for level in ["Later", "Soon", "Now"]
    ]
    score = [value / sum(fits) for value in fits]
    assert list(result["answers"]["route"]["probabilities"].values()) == pytest.approx(
        route, abs=1e-7
    )
    assert result["answers"]["yes"]["noul"] == pytest.approx(yes[1], abs=1e-7)
    assert list(
        result["answers"]["urgency"]["probabilities"].values()
    ) == pytest.approx(score, abs=1e-7)
    assert result["answers"]["urgency"]["score"] == pytest.approx(
        sum(i * p for i, p in enumerate(score))
    )
    assert len(agent.model.calls) == 5
    for ids, mask in agent.model.calls:
        assert ids.shape[0] == 1 and ids.shape[1] % 64 == 0
        real = int(mask.sum())
        assert mask[0, :real].all() and not mask[0, real:].any()
        assert not ids[0, real:].any()
    assert result["metadata"]["head_dtype"] == "bfloat16"
    assert result["metadata"]["padding_multiple"] == 64
    assert result["usage"]["output_tokens"] == 0


def test_score_isolation_opt_out_and_list_choice():
    agent = make_agent()
    questions = {
        "pick": {
            "type": "choice",
            "instructions": "Pick?",
            "criteria": ["left", "right"],
        },
        "scale": {
            "type": "score",
            "instructions": "How?",
            "criteria": ["low", "high"],
            "isolated": False,
        },
    }
    result = agent.system_one("state", questions)
    assert set(result["answers"]["pick"]["probabilities"]) == {"left", "right"}
    assert len(agent.model.calls) == 2
    expected, _ = independent_probability(agent, "state", "How?", ["0: low", "1: high"])
    assert list(result["answers"]["scale"]["probabilities"].values()) == pytest.approx(
        expected
    )


def test_2b_preserves_per_type_temperature_for_isolated_score_rows():
    agent = make_agent(adapter.MODEL_ID_2B)
    state = "A duplicate charge."
    result = agent.system_one(state, QUESTIONS)
    route, _ = independent_probability(
        agent, state, "Route?", ["bill: Refund", "tech"], temperature=1.164
    )
    yes, _ = independent_probability(
        agent, state, "Refund?", ["no", "yes"], temperature=1.624
    )
    fits = [
        independent_probability(
            agent,
            state,
            f"Urgent?\nProposed answer: {level}\nDoes the proposed answer fit?",
            ["no", "yes"],
            temperature=1.124,
        )[0][1]
        for level in ["Later", "Soon", "Now"]
    ]
    assert result["model"] == adapter.MODEL_ID_2B
    assert list(result["answers"]["route"]["probabilities"].values()) == pytest.approx(
        route, abs=1e-7
    )
    assert result["answers"]["yes"]["noul"] == pytest.approx(yes[1], abs=1e-7)
    assert list(
        result["answers"]["urgency"]["probabilities"].values()
    ) == pytest.approx([p / sum(fits) for p in fits], abs=1e-7)
    assert len(agent.model.calls) == 5


def test_2b_listwise_score_keeps_score_temperature():
    agent = make_agent(adapter.MODEL_ID_2B)
    question = {**QUESTIONS["urgency"], "isolated": False}
    result = agent.system_one("state", {"s": question})
    expected, _ = independent_probability(
        agent,
        "state",
        "Urgent?",
        ["0: 0: Later", "1: 1: Soon", "2: 2: Now"],
        temperature=1.124,
    )
    assert list(result["answers"]["s"]["probabilities"].values()) == pytest.approx(
        expected, abs=1e-7
    )
    assert len(agent.model.calls) == 1


def test_duplicate_rows_reuse_only_within_request():
    agent = make_agent()
    questions = {"a": QUESTIONS["yes"], "b": copy.deepcopy(QUESTIONS["yes"])}
    first = agent.system_one("context", questions)
    assert len(agent.model.calls) == 1
    assert first["answers"]["a"] == first["answers"]["b"]
    agent.system_one("context", questions)
    assert len(agent.model.calls) == 2


def test_json_array_indices_and_descriptions_preserve_original_rendering():
    agent = make_agent()
    state = {"记录": [{"文本": str(i)} for i in range(8)]}
    questions = {
        "问": {
            "type": "choice",
            "instructions": {"task": "选择"},
            "criteria": {"甲": {"类别": 1}, "乙": None},
        }
    }
    result = agent.system_one(state, questions)
    annotated = {"记录": [{"_index": i, "文本": str(i)} for i in range(8)]}
    expected, ids = independent_probability(
        agent,
        json.dumps(annotated, ensure_ascii=False),
        json.dumps({"task": "选择"}, ensure_ascii=False),
        ["甲: " + json.dumps({"类别": 1}, ensure_ascii=False), "乙"],
    )
    assert list(result["answers"]["问"]["probabilities"].values()) == pytest.approx(
        expected
    )
    assert agent.model.calls[0][0][0, : len(ids)].tolist() == ids.tolist()


@pytest.mark.parametrize(
    "order", list(itertools.permutations(["state", "instructions", "criteria"]))
)
def test_context_orders_keep_decider_readout_blocks(monkeypatch, order):
    monkeypatch.setenv(ENVIRONMENT, json.dumps({"context_order": list(order)}))
    agent = make_agent()
    agent.system_one("STATE", {"q": QUESTIONS["route"]})
    blocks = {
        "state": "Context:\nSTATE\n\n",
        "instructions": "Question: Route?\n\n",
        "criteria": 'Criteria: ["bill: Refund", "tech"]\n\n',
    }
    prefix = "".join(blocks[key] for key in order)
    row = (
        prefix + "\n\nQuestion: Route?\nOptions:\n(A) bill: Refund\n(B) tech\nAnswer: ("
    )
    expected = Tokenizer().encode(row)
    ids, mask = agent.model.calls[0]
    assert ids[0, : int(mask.sum())].tolist() == expected


def test_instructions_first_keeps_answer_slot(monkeypatch):
    monkeypatch.setenv(ENVIRONMENT, json.dumps({"layout": "instructions-first"}))
    agent = make_agent()
    agent.system_one("STATE", {"q": QUESTIONS["yes"]})
    expected = Tokenizer().encode(
        "Question: Refund?\n\nContext:\nSTATE\nOptions:\n(A) no\n(B) yes\nAnswer: ("
    )
    ids, mask = agent.model.calls[0]
    assert ids[0, : int(mask.sum())].tolist() == expected


def test_wide_choices_insert_single_label_ids():
    agent = make_agent()
    agent.system_one(
        "state",
        {
            "q": {
                "type": "choice",
                "instructions": "Pick?",
                "criteria": {str(i): None for i in range(11)},
            }
        },
    )
    tokenizer = Tokenizer()
    expected = tokenizer.encode("Context:\nstate\n\nQuestion: Pick?\nOptions:")
    for i in range(11):
        expected += tokenizer.encode("\n(") + [i] + tokenizer.encode(f") {i}")
    expected += tokenizer.encode("\nAnswer: (")
    ids, mask = agent.model.calls[0]
    assert ids[0, : int(mask.sum())].tolist() == expected


@pytest.mark.parametrize(
    "questions",
    [
        {},
        {"x": {"type": "score", "instructions": "x", "criteria": ["a"] * 11}},
        {"x": {"type": "choice", "instructions": "x", "criteria": ["a", "a"]}},
        {"x": {"type": "noul", "instructions": "x", "criteria": {"bad": "x"}}},
        {
            "x": {
                "type": "score",
                "instructions": "x",
                "criteria": ["a", "b"],
                "isolated": "false",
            }
        },
        {
            "x": {
                "type": "choice",
                "instructions": float("nan"),
                "criteria": {"a": None, "b": None},
            }
        },
    ],
)
def test_invalid_schema_does_not_run_any_row(questions):
    agent = make_agent()
    with pytest.raises(ValueError):
        agent.system_one("context", questions)
    assert not agent.model.calls


def test_long_later_question_is_rejected_before_first_forward():
    agent = make_agent()
    questions = {
        "a": QUESTIONS["yes"],
        "b": {"type": "noul", "instructions": "x" * 1024},
    }
    with pytest.raises(ValueError, match="context"):
        agent.system_one("context", questions)
    assert not agent.model.calls


def test_precision_is_actual_and_close_rejects_inference():
    import torch

    agent = make_agent()
    agent.weight = agent.weight.float()
    with pytest.raises(ValueError, match="BF16"):
        agent.system_one("context", QUESTIONS)
    assert not agent.model.calls
    agent.weight = agent.weight.bfloat16()
    agent.model.embedding.weight = torch.nn.Parameter(
        agent.model.embedding.weight.float()
    )
    with pytest.raises(ValueError, match="BF16"):
        agent.system_one("context", QUESTIONS)
    agent.close()
    with pytest.raises(RuntimeError, match="closed"):
        agent.system_one("context", QUESTIONS)


def fake_source(tmp_path, monkeypatch, model_id=adapter.MODEL_ID):
    specification = adapter.MODELS[model_id]
    config = {
        "model_type": "qwen3_5_text",
        "architectures": ["Qwen3_5ForCausalLM"],
        "hidden_size": specification["hidden_size"],
        "tie_word_embeddings": True,
    }
    calibration = {
        "layout": "plain",
        "neutralize_none": False,
        "schema_first": False,
        "max_options": 255,
        "isolated_levels": True,
        "version": specification["version"],
        "temperature": specification["temperature"],
        "temperature_by_type": specification["temperatures"],
    }
    files = {
        "config.json": json.dumps(config).encode(),
        "decider_config.json": json.dumps(calibration).encode(),
        "model.safetensors": b"fixture weights",
    }
    hashes = {name: hashlib.sha256(value).hexdigest() for name, value in files.items()}
    monkeypatch.setitem(
        adapter.MODELS,
        model_id,
        {**specification, "files": hashes},
    )
    for name, value in files.items():
        (tmp_path / name).write_bytes(value)
    return files


def test_source_verification_is_read_only_and_rejects_override(tmp_path, monkeypatch):
    files = fake_source(tmp_path, monkeypatch)
    assert adapter.identify_source(tmp_path) == adapter.MODEL_ID
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == files
    (tmp_path / "adapter_config.json").write_text("{}")
    with pytest.raises(ValueError, match="unexpected"):
        adapter.verify_source(tmp_path)


def test_source_checksum_rejects_changed_weights(tmp_path, monkeypatch):
    fake_source(tmp_path, monkeypatch)
    (tmp_path / "model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        adapter.verify_source(tmp_path)


def test_2b_identification_selects_config_before_hashing_weights_once(
    tmp_path, monkeypatch
):
    fake_source(tmp_path, monkeypatch, adapter.MODEL_ID_2B)
    original = hashlib.file_digest
    opened = []

    def digest(stream, algorithm):
        opened.append(Path(stream.name).name)
        return original(stream, algorithm)

    monkeypatch.setattr(adapter.hashlib, "file_digest", digest)
    assert adapter.identify_source(tmp_path) == adapter.MODEL_ID_2B
    assert opened.count("model.safetensors") == 1
    assert opened[0] == "config.json"
    with pytest.raises(ValueError, match="checksum"):
        adapter.verify_source(tmp_path, adapter.MODEL_ID)


def test_unknown_config_does_not_hash_weights(tmp_path, monkeypatch):
    fake_source(tmp_path, monkeypatch)
    (tmp_path / "config.json").write_text("{}")
    original = hashlib.file_digest
    opened = []

    def digest(stream, algorithm):
        opened.append(Path(stream.name).name)
        return original(stream, algorithm)

    monkeypatch.setattr(adapter.hashlib, "file_digest", digest)
    with pytest.raises(ValueError, match="unrecognized"):
        adapter.identify_source(tmp_path)
    assert opened == ["config.json"]


@pytest.mark.parametrize(
    "replacement",
    [
        {},
        {"score": 1.624},
        {"choice": 1.164, "noul": 1.624, "score": 1.124, "bool": 1.624},
    ],
)
def test_2b_calibration_identity_rejects_wrong_type_map(
    tmp_path, monkeypatch, replacement
):
    fake_source(tmp_path, monkeypatch, adapter.MODEL_ID_2B)
    path = tmp_path / "decider_config.json"
    value = json.loads(path.read_text())
    value["temperature_by_type"] = replacement
    path.write_text(json.dumps(value))
    specification = adapter.MODELS[adapter.MODEL_ID_2B]
    specification["files"]["decider_config.json"] = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()
    with pytest.raises(ValueError, match="identity"):
        adapter.verify_source(tmp_path, adapter.MODEL_ID_2B)


@pytest.mark.parametrize("model_id", [adapter.MODEL_ID, adapter.MODEL_ID_2B])
def test_download_failure_cleans_staging_and_does_not_publish(
    tmp_path, monkeypatch, model_id
):
    def interrupted(**kwargs):
        assert kwargs["repo_id"] == model_id
        assert kwargs["revision"] == adapter.MODELS[model_id]["revision"]
        assert set(kwargs["allow_patterns"]) == set(adapter.MODELS[model_id]["files"])
        Path(kwargs["local_dir"]).mkdir()
        (Path(kwargs["local_dir"]) / "partial").write_bytes(b"partial")
        raise OSError("network interrupted")

    monkeypatch.setattr(adapter, "snapshot_download", interrupted)
    with pytest.raises(OSError, match="interrupted"):
        adapter.prepare(model_id, tmp_path)
    parent = tmp_path / "checkpoint" / "Mapika"
    assert not (tmp_path / "checkpoint" / model_id).exists()
    assert not list(parent.glob(".decider-download-*"))


def test_existing_source_is_never_redownloaded(tmp_path, monkeypatch):
    source = tmp_path / "checkpoint" / adapter.MODEL_ID
    source.mkdir(parents=True)
    files = fake_source(source, monkeypatch)
    monkeypatch.setattr(
        adapter, "snapshot_download", lambda **kw: pytest.fail("unexpected download")
    )
    assert adapter.prepare(adapter.MODEL_ID, tmp_path) == source
    assert {p.name: p.read_bytes() for p in source.iterdir()} == files
