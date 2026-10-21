"""JevK5 CPU contracts; entirely offline and without released model weights."""

import copy
import hashlib
import importlib.util
import itertools
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm_jev import windows_jevk5 as adapter
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
        "criteria": ["Later", "Soon", "Now"],
    },
}


@pytest.fixture(autouse=True)
def isolated_template(monkeypatch):
    monkeypatch.delenv(ENVIRONMENT, raising=False)
    current_template.cache_clear()
    yield
    current_template.cache_clear()


class Tokenizer:
    pad_token_id = None

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return [ord(char) % 1023 + 1 for char in text]

    def apply_chat_template(
        self, messages, *, tokenize, add_generation_prompt, enable_thinking
    ):
        assert not tokenize and add_generation_prompt and not enable_thinking
        return (
            "".join(
                f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
                for message in messages
            )
            + "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )


def make_agent():
    import torch

    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(815)
            self.embedding = torch.nn.Embedding(1024, 16, dtype=torch.bfloat16)
            self.calls = []

        def get_input_embeddings(self):
            return self.embedding

        def forward(self, *, input_ids, use_cache):
            assert not use_cache
            self.calls.append(input_ids.clone())
            embeddings = self.embedding(input_ids).float()
            positions = torch.arange(1, input_ids.shape[1] + 1)[None, :, None]
            return SimpleNamespace(
                last_hidden_state=(embeddings.cumsum(1) / positions).bfloat16()
            )

    torch.manual_seed(813)
    weight = torch.randn(16, 16).bfloat16()
    return adapter.JevK5Agent(Backbone(), Tokenizer(), weight)


def independent_row(agent, state, instructions, options):
    """Spell out the published prompt and readout without the product compiler."""
    import numpy as np
    import torch

    system = "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. Respond with only its uppercase letter, with no explanation or reasoning."
    payload = {
        "evidence": state,
        "criterion": instructions,
        "options": [
            {"letter": chr(65 + i), "description": text}
            for i, text in enumerate(options)
        ],
    }
    text = (
        "<|im_start|>system\n"
        + system
        + "<|im_end|>\n<|im_start|>user\n"
        + json.dumps(payload, ensure_ascii=False)
        + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    ids = Tokenizer().encode(text)
    hidden = agent.model.embedding(torch.tensor(ids)).float().sum(0) / len(ids)
    logits = (hidden.bfloat16()[None] @ agent.weight.T)[
        0, : len(options)
    ].float().numpy() / 1.42
    probability = np.exp(logits - logits.max())
    probability /= probability.sum()
    return probability.tolist(), ids


def test_lightweight_import():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import vllm_jev.windows_jevk5; assert not any(n in sys.modules for n in ('torch','transformers','vllm','mlx'))",
        ],
        cwd=ROOT,
        check=True,
    )


def test_three_types_use_author_prompt_projection_and_numpy_probability():
    agent = make_agent()
    state = {"文本": "A duplicate charge.", "rows": list(range(8))}
    result = agent.system_one(state, QUESTIONS)
    expected = [
        independent_row(agent, state, "Route?", ["bill: Refund", "tech: tech"]),
        independent_row(
            agent,
            state,
            "Refund?",
            ["true: The proposition is true.", "false: The proposition is false."],
        ),
        independent_row(agent, state, "Urgent?", ["0: Later", "1: Soon", "2: Now"]),
    ]
    assert len(agent.model.calls) == 3
    for actual, (_, ids) in zip(agent.model.calls, expected):
        assert actual.tolist() == [ids]  # No padding and no synthetic JSON indices.
    assert list(result["answers"]["route"]["probabilities"].values()) == expected[0][0]
    assert result["answers"]["yes"]["noul"] == expected[1][0][0]
    score = result["answers"]["urgency"]
    assert list(score["probabilities"].values()) == expected[2][0]
    assert score["score"] == sum(i * p for i, p in enumerate(expected[2][0]))
    assert score["legend"] == {"0": "0: Later", "1": "1: Soon", "2": "2: Now"}
    assert result["usage"] == {
        "input_tokens": sum(len(ids) for _, ids in expected),
        "output_tokens": 0,
    }
    assert result["metadata"]["padding_multiple"] == 1
    assert result["metadata"]["readout_dtype"] == "bfloat16"


def test_choice_list_and_duplicate_rows_only_reuse_within_request():
    agent = make_agent()
    question = {"type": "choice", "instructions": "Pick?", "criteria": ["a", "b"]}
    result = agent.system_one("state", {"a": question, "b": copy.deepcopy(question)})
    assert len(agent.model.calls) == 1
    assert result["answers"]["a"] == result["answers"]["b"]
    agent.system_one("state", {"a": question})
    assert len(agent.model.calls) == 2


@pytest.mark.parametrize("count", [2, 10, 16])
def test_all_supported_choice_widths(count):
    agent = make_agent()
    q = {
        "type": "choice",
        "instructions": "Pick?",
        "criteria": [str(i) for i in range(count)],
    }
    result = agent.system_one("state", {"q": q})
    expected, _ = independent_row(
        agent, "state", "Pick?", [f"{i}: {i}" for i in range(count)]
    )
    assert list(result["answers"]["q"]["probabilities"].values()) == expected


@pytest.mark.parametrize(
    "order", list(itertools.permutations(["state", "instructions", "criteria"]))
)
def test_context_order_keeps_readout_protocol(monkeypatch, order):
    monkeypatch.setenv(ENVIRONMENT, json.dumps({"context_order": list(order)}))
    agent = make_agent()
    agent.system_one("STATE", {"q": QUESTIONS["route"]})
    text = "".join(chr(token - 1) for token in agent.model.calls[0][0].tolist())
    keys = {
        "state": '"evidence"',
        "instructions": '"criterion"',
        "criteria": '"options"',
    }
    positions = [text.index(keys[key]) for key in order]
    assert positions == sorted(positions)
    assert text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


@pytest.mark.parametrize(
    "question",
    [
        {"type": "choice", "instructions": "?", "criteria": ["same", "same"]},
        {"type": "choice", "instructions": "?", "criteria": {"only": None}},
        {
            "type": "choice",
            "instructions": "?",
            "criteria": {str(i): None for i in range(17)},
        },
        {"type": "choice", "instructions": "?", "criteria": {"a": {}, "b": None}},
        {"type": "choice", "instructions": "?", "criteria": {"": None, "b": None}},
        {"type": "noul", "instructions": "?", "criteria": {"maybe": "perhaps"}},
        {"type": "noul", "instructions": "?", "criteria": {"true": 3}},
        {"type": "score", "instructions": "?", "criteria": ["a"]},
        {"type": "score", "instructions": "?", "criteria": ["a"] * 11},
        {
            "type": "score",
            "instructions": "?",
            "criteria": ["a", "b"],
            "isolated": True,
        },
        {"type": "score", "instructions": "?", "criteria": ["a", 3]},
        {"type": "unknown", "instructions": "?"},
        {"type": "noul", "instructions": {}},
        {"type": "noul", "instructions": " "},
    ],
)
def test_invalid_later_question_prevents_any_forward(question):
    agent = make_agent()
    with pytest.raises(ValueError):
        agent.system_one("state", {"valid": QUESTIONS["yes"], "invalid": question})
    assert not agent.model.calls


@pytest.mark.parametrize(
    "state",
    [
        None,
        True,
        3,
        "x" * 50_001,
        {"x": float("nan")},
        {"messages": [{"content": [{"type": "image_url", "image_url": "x"}]}]},
    ],
    ids=["null", "boolean", "number", "oversize", "nonfinite", "media"],
)
def test_invalid_state_prevents_forward(state):
    agent = make_agent()
    with pytest.raises(ValueError):
        agent.system_one(state, {"q": QUESTIONS["yes"]})
    assert not agent.model.calls


def test_overlong_complete_prompt_is_rejected_not_truncated():
    agent = make_agent()
    with pytest.raises(ValueError, match="context"):
        agent.system_one("x" * 800, {"q": QUESTIONS["route"]})
    assert not agent.model.calls


def test_precision_failures_and_close():
    agent = make_agent()
    agent.weight = agent.weight.float()
    with pytest.raises(ValueError, match="BF16"):
        agent.system_one("state", QUESTIONS)
    agent = make_agent()
    agent.weight[0, 0] = float("nan")
    with pytest.raises(ValueError, match="readout"):
        agent.system_one("state", QUESTIONS)
    agent.close()
    with pytest.raises(RuntimeError, match="closed"):
        agent.system_one("state", QUESTIONS)


def fake_source(root, monkeypatch):
    config = {
        "architectures": ["Qwen3_5ForCausalLM"],
        "model_type": "qwen3_5_text",
        "hidden_size": 2048,
        "num_hidden_layers": 24,
        "dtype": "bfloat16",
        "tie_word_embeddings": True,
    }
    files = {
        "config.json": json.dumps(config).encode(),
        "jevk5_config.json": b'{"temperature": 1.42}',
        "model.safetensors": b"fixture weights",
    }
    for name, content in files.items():
        (root / name).write_bytes(content)
    monkeypatch.setattr(
        adapter,
        "SOURCE_HASHES",
        {name: hashlib.sha256(content).hexdigest() for name, content in files.items()},
    )
    return files


@pytest.mark.parametrize(
    "name,directory",
    [
        ("adapter_config.json", False),
        ("chat_template.jinja", False),
        ("templates", True),
        (".cache", False),
    ],
)
def test_source_rejects_unpinned_loader_behavior_before_import(
    tmp_path, monkeypatch, name, directory
):
    fake_source(tmp_path, monkeypatch)
    target = tmp_path / name
    target.mkdir() if directory else target.write_text("{}")
    with pytest.raises(ValueError, match="unexpected"):
        adapter.load_agent(tmp_path)


def test_source_identification_hashes_large_weight_once_and_stays_readonly(
    tmp_path, monkeypatch
):
    files = fake_source(tmp_path, monkeypatch)
    (tmp_path / ".cache").mkdir()
    original, calls = hashlib.file_digest, []

    def digest(stream, algorithm):
        calls.append(Path(stream.name).name)
        return original(stream, algorithm)

    monkeypatch.setattr(adapter.hashlib, "file_digest", digest)
    assert adapter.identify_source(tmp_path) == adapter.MODEL_ID
    assert calls[0] == "jevk5_config.json" and calls.count("model.safetensors") == 1
    assert {name: (tmp_path / name).read_bytes() for name in files} == files
    (tmp_path / "model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        adapter.verify_source(tmp_path)


def test_unknown_calibration_never_hashes_weights(tmp_path, monkeypatch):
    fake_source(tmp_path, monkeypatch)
    (tmp_path / "jevk5_config.json").write_text("{}")
    calls, original = [], hashlib.file_digest

    def digest(stream, algorithm):
        calls.append(Path(stream.name).name)
        return original(stream, algorithm)

    monkeypatch.setattr(adapter.hashlib, "file_digest", digest)
    with pytest.raises(ValueError, match="unrecognized"):
        adapter.identify_source(tmp_path)
    assert calls == ["jevk5_config.json"]


def test_download_uses_pins_and_never_publishes_partial_source(tmp_path, monkeypatch):
    def interrupted(**kwargs):
        assert (
            kwargs["repo_id"] == adapter.MODEL_ID
            and kwargs["revision"] == adapter.REVISION
        )
        assert set(kwargs["allow_patterns"]) == set(adapter.SOURCE_HASHES)
        kwargs["local_dir"].mkdir()
        (kwargs["local_dir"] / "partial").write_bytes(b"partial")
        raise OSError("interrupted")

    monkeypatch.setattr(adapter, "snapshot_download", interrupted)
    with pytest.raises(OSError, match="interrupted"):
        adapter.prepare(adapter.MODEL_ID, tmp_path)
    assert not (tmp_path / "checkpoint" / adapter.MODEL_ID).exists()
    assert not list(
        (tmp_path / "checkpoint" / "alibiserikbay").glob(".jevk5-download-*")
    )


def test_existing_source_never_redownloads(tmp_path, monkeypatch):
    source = tmp_path / "checkpoint" / adapter.MODEL_ID
    source.mkdir(parents=True)
    fake_source(source, monkeypatch)
    monkeypatch.setattr(
        adapter, "snapshot_download", lambda **kw: pytest.fail("unexpected download")
    )
    assert adapter.prepare(adapter.MODEL_ID, tmp_path) == source


@pytest.mark.parametrize("threads", [True, 0, -1, 1.5])
def test_bad_threads_rejected_before_source_access(tmp_path, threads):
    with pytest.raises(ValueError, match="threads"):
        adapter.load_agent(tmp_path, threads=threads)


def load_runner():
    spec = importlib.util.spec_from_file_location(
        "jevk5_reference_for_test", ROOT / "scripts/windows_jevk5_reference.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_independent_runner_has_matching_pins_and_blocks_product():
    runner = load_runner()
    assert runner.SOURCE_HASHES == adapter.SOURCE_HASHES
    assert all(len(value) == 64 for value in runner.SDK_HASHES.values())
    with pytest.raises(RuntimeError, match="product"):
        runner.BlockProductImports().find_spec("vllm_jev.windows_jevk5")


def test_runner_rejects_duplicate_or_nonfinite_cases(tmp_path):
    runner = load_runner()
    source = tmp_path / "cases.json"
    source.write_text('[{"id":"x","id":"y","state":"x","questions":{}}]')
    with pytest.raises(ValueError, match="Duplicate"):
        runner.read_cases(source)
    source.write_text('[{"id":"x","state":NaN,"questions":{}}]')
    with pytest.raises(ValueError, match="Nonfinite"):
        runner.read_cases(source)


def test_runner_only_copies_verified_sdk_python(tmp_path, monkeypatch):
    runner = load_runner()
    source, target = tmp_path / "source", tmp_path / "target"
    (source / "jevk5/__pycache__").mkdir(parents=True)
    payload = b"# original source\n"
    (source / "jevk5/__init__.py").write_bytes(payload)
    (source / "jevk5/__pycache__/evil.pyc").write_bytes(b"not executed")
    monkeypatch.setattr(
        runner, "SDK_HASHES", {"jevk5/__init__.py": hashlib.sha256(payload).hexdigest()}
    )
    runner.copy_sdk(source, target)
    assert (target / "jevk5/__init__.py").read_bytes() == payload
    assert not (target / "jevk5/__pycache__").exists()
    (source / "jevk5/__init__.py").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA256"):
        runner.verify_sdk(source)


def test_real_tiny_qwen35_backbone_matches_author_readout():
    import numpy as np
    import torch
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    torch.set_num_threads(1)
    torch.manual_seed(816)
    config = Qwen3_5TextConfig(
        vocab_size=1024,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        layer_types=["linear_attention", "full_attention"],
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        pad_token_id=None,
    )
    config._attn_implementation = "sdpa"
    model = Qwen3_5ForCausalLM(config).bfloat16().eval()
    weight = model.lm_head.weight[:16].detach().clone()
    agent = adapter.JevK5Agent(model.model, Tokenizer(), weight)
    ids = [2, 3, 4, 5, 6]
    actual = agent._readout(ids)
    with torch.inference_mode():
        tokens = torch.tensor([ids])
        hidden = model.model(input_ids=tokens, use_cache=False).last_hidden_state
        expected = (
            (hidden[torch.arange(1), torch.tensor([len(ids) - 1])] @ weight.T)[0]
            .float()
            .numpy()
        )
    np.testing.assert_array_equal(actual, expected)
