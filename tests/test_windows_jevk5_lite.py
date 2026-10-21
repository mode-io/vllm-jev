"""Offline JevK5-Lite contracts; no released encoder weights are loaded."""

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm_jev import windows_jevk5_lite as adapter
from vllm_jev.decision_template import ENVIRONMENT, DecisionTemplate, current_template

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = {
    "choice": {
        "type": "choice",
        "instructions": "route_request",
        "criteria": {"billing": "refund", "technical": None},
    },
    "noul": {"type": "noul", "instructions": "Should refund?"},
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
    cls_token_id = 0
    sep_token_id = 1
    unk_token_id = 2

    def convert_tokens_to_ids(self, token):
        return {"[TASK]": 3, "[LABEL]": 4}.get(token, self.unk_token_id)

    def __call__(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return {"input_ids": [ord(char) % 500 + 5 for char in text]}


def make_agent():
    import torch

    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(512, 16)
            self.calls = []

        def get_input_embeddings(self):
            return self.embedding

        def forward(self, *, input_ids, attention_mask):
            assert input_ids.shape[0] == 1
            assert torch.equal(attention_mask, torch.ones_like(input_ids))
            self.calls.append(input_ids.clone())
            h = self.embedding(input_ids)
            return SimpleNamespace(last_hidden_state=h + h.mean(1, keepdim=True))

    torch.manual_seed(523)
    return adapter.LiteAgent(Encoder(), Tokenizer(), adapter.make_scorer(16))


def independent_row(agent, state, task, labels):
    """Spell out the original one-head layout and FP32 MLP/softmax."""
    import torch
    import torch.nn.functional as F

    text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)

    def piece(value):
        return [ord(char) % 500 + 5 for char in value]

    schema = [3] + piece(task.replace("_", " ") + " (one)")
    positions = []
    for label in labels:
        positions.append(len(schema) + 1)
        schema += [4] + piece(label)
    body = piece(text)
    ids = [0] + schema + [1] + body + [1]
    with torch.inference_mode():
        h = agent.model.embedding(torch.tensor([ids]))
        h = (h + h.mean(1, keepdim=True))[0]
        start = len(schema) + 2
        pooled = h[start : start + max(1, len(body))].mean(0, keepdim=True)
        markers = h[positions]
        pooled = pooled.expand_as(markers)
        features = torch.cat([markers, pooled, markers * pooled], -1)
        first = F.linear(features, agent.scorer[0].weight, agent.scorer[0].bias)
        logits = (
            F.linear(F.gelu(first), agent.scorer[2].weight, agent.scorer[2].bias)
            .squeeze(-1)
            .float()
        )
        p = torch.softmax(logits / 1.8660659830736146, -1).tolist()
    return p, ids


def test_module_import_does_not_load_tensor_runtimes():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import vllm_jev.windows_jevk5_lite; "
            "assert not any(x in sys.modules for x in "
            "('torch','transformers','vllm','mlx','laya'))",
        ],
        check=True,
        cwd=ROOT,
    )


@pytest.mark.parametrize("state", ["Two charges.", {"中文": "重复收费", "n": 2}, ""])
def test_three_types_exact_author_layout_and_fp32_math(state):
    agent = make_agent()
    result = agent.system_one(state, QUESTIONS)
    expected = [
        independent_row(agent, state, "route_request", ["refund", "technical"]),
        independent_row(agent, state, "Should refund?", ["true", "false"]),
        independent_row(agent, state, "Urgency?", ["low", "medium", "high"]),
    ]
    assert len(agent.model.calls) == 3
    for call, (_, ids) in zip(agent.model.calls, expected):
        assert call.tolist() == [ids]
    answers = result["answers"]
    assert list(answers["choice"]["probabilities"].values()) == expected[0][0]
    assert answers["noul"]["noul"] == expected[1][0][0]
    assert list(answers["score"]["probabilities"].values()) == expected[2][0]
    assert answers["score"]["score"] == sum(i * p for i, p in enumerate(expected[2][0]))
    assert answers["score"]["legend"] == {"0": "low", "1": "medium", "2": "high"}
    assert result["usage"]["input_tokens"] == sum(len(ids) for _, ids in expected)
    assert result["internal_usage"]["compute_tokens"] == result["usage"]["input_tokens"]
    assert result["metadata"]["padding"] == "none"
    assert result["metadata"]["parameter_dtypes"] == ["torch.float32"]
    assert result["metadata"]["revision"] == adapter.REVISION


@pytest.mark.parametrize(
    ("kind", "criteria", "labels"),
    [
        ("choice", {"a": "", "b": None}, ["a", "b"]),
        ("choice", ["a", "b"], ["a", "b"]),
        ("choice", {"a": "same", "b": "same"}, ["a: same", "b: same"]),
        ("choice", {"a": None, "b": "a"}, ["a: a", "b: a"]),
        ("score", ["same", "same"], ["0: same", "1: same"]),
        ("score", ["", ""], ["0: ", "1: "]),
    ],
)
def test_author_empty_description_and_duplicate_label_fallback(kind, criteria, labels):
    agent = make_agent()
    question = {"type": kind, "instructions": "Pick?", "criteria": criteria}
    response = agent.system_one("data", {"q": question})
    expected, ids = independent_row(agent, "data", "Pick?", labels)
    assert agent.model.calls[0].tolist() == [ids]
    assert list(response["answers"]["q"]["probabilities"].values()) == expected


@pytest.mark.parametrize(
    "question",
    [
        {"type": "noul", "instructions": "?", "criteria": {"true": "custom"}},
        {"type": "noul", "instructions": "?", "criteria": []},
        {"type": "noul", "instructions": {}},
        {"type": "noul", "instructions": " "},
        {"type": "choice", "instructions": "?", "criteria": ["a"]},
        {"type": "choice", "instructions": "?", "criteria": ["a", "a"]},
        {"type": "choice", "instructions": "?", "criteria": ["a", 4]},
        {"type": "choice", "instructions": "?", "criteria": {"": None, "a": None}},
        {"type": "choice", "instructions": "?", "criteria": {"a": {}, "b": None}},
        {"type": "score", "instructions": "?", "criteria": ["a"]},
        {"type": "score", "instructions": "?", "criteria": ["a"] * 11},
        {"type": "score", "instructions": "?", "criteria": ["a", 3]},
        {
            "type": "score",
            "instructions": "?",
            "criteria": ["a", "b"],
            "isolated": True,
        },
        {"type": "other", "instructions": "?"},
    ],
)
def test_invalid_later_question_performs_no_forward(question):
    agent = make_agent()
    with pytest.raises(ValueError):
        agent.system_one("state", {"valid": QUESTIONS["noul"], "invalid": question})
    assert not agent.model.calls


def test_unresolved_rendered_collision_is_rejected_like_original_classifier():
    agent = make_agent()
    # A collision elsewhere triggers prefixing, after which these first two
    # different keys collide. The author _heads check rejects this second clash.
    question = {
        "type": "choice",
        "instructions": "?",
        "criteria": {"a": "b: c", "a: b": "c", "x": "same", "y": "same"},
    }
    with pytest.raises(ValueError, match="rendered labels"):
        agent.system_one("state", {"q": question})
    assert not agent.model.calls


@pytest.mark.parametrize("state", [True, 42, None, {"a": float("nan")}, [float("inf")]])
def test_invalid_state_performs_no_forward(state):
    agent = make_agent()
    with pytest.raises(ValueError):
        agent.system_one(state, QUESTIONS)
    assert not agent.model.calls


def test_complete_row_budget_512_and_no_partial_forward_on_late_overflow():
    agent = make_agent()
    q = {"type": "noul", "instructions": "?"}
    width = len(agent._encode("", "?", ["true", "false"])[0])
    agent.system_one("x" * (512 - width), {"q": q})
    assert agent.model.calls[0].shape == (1, 512)
    agent.model.calls.clear()
    with pytest.raises(ValueError, match="512 tokens"):
        agent.system_one(
            "x" * (512 - width),
            {"ok": q, "too_long": {"type": "noul", "instructions": "??"}},
        )
    assert not agent.model.calls


def test_same_question_is_independently_evaluated_and_close_is_idempotent():
    agent = make_agent()
    agent.system_one("state", {"a": QUESTIONS["noul"], "b": QUESTIONS["noul"]})
    assert len(agent.model.calls) == 2
    agent.system_one("state", {"c": QUESTIONS["noul"]})
    assert len(agent.model.calls) == 3
    agent.close()
    agent.close()
    with pytest.raises(RuntimeError, match="closed"):
        agent.system_one("state", QUESTIONS)


@pytest.mark.parametrize(
    "config", [{"layout": "instructions-first"}, {"state_template": "{state}"}]
)
def test_non_native_template_fails_before_any_inference(config):
    with pytest.raises(ValueError, match="default/native"):
        adapter.validate_template(DecisionTemplate(config))


def test_dtype_and_nonfinite_head_rejected():
    import torch

    agent = make_agent()
    with pytest.raises(ValueError, match="FP32"):
        adapter.LiteAgent(agent.model.bfloat16(), Tokenizer(), agent.scorer)
    agent = make_agent()
    with torch.no_grad():
        agent.scorer[0].weight[0, 0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        adapter.LiteAgent(agent.model, Tokenizer(), agent.scorer)


def test_actual_small_deberta_forward_has_no_padding_or_history():
    import torch
    from transformers import DebertaV2Config, DebertaV2Model

    torch.set_num_threads(1)
    torch.manual_seed(531)
    model = DebertaV2Model(
        DebertaV2Config(
            vocab_size=512,
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=4,
            intermediate_size=32,
            max_position_embeddings=512,
            relative_attention=True,
            pos_att_type=["p2c", "c2p"],
            position_buckets=32,
        )
    )
    agent = adapter.LiteAgent(model, Tokenizer(), adapter.make_scorer(16))
    seen = []

    def observe(_model, _args, kwargs):
        assert set(kwargs) == {"input_ids", "attention_mask"}
        assert kwargs["input_ids"].shape[0] == 1
        assert torch.equal(
            kwargs["attention_mask"], torch.ones_like(kwargs["input_ids"])
        )
        seen.append(kwargs["input_ids"].shape[1])

    hook = model.register_forward_pre_hook(observe, with_kwargs=True)
    result = agent.system_one("真实小模型", QUESTIONS)
    hook.remove()
    assert len(seen) == 3
    assert result["metadata"]["threads"] == 1
    for key in ("choice", "score"):
        probabilities = list(result["answers"][key]["probabilities"].values())
        assert sum(probabilities) == pytest.approx(1.0, abs=1e-6)
    assert all(0 < width <= 512 for width in seen)


@pytest.fixture
def source(tmp_path, monkeypatch):
    root = tmp_path / "source"
    root.mkdir()
    contents = {
        "config.json": {
            "model_type": "deberta-v2",
            "architectures": ["DebertaV2Model"],
            "hidden_size": 1024,
            "num_hidden_layers": 24,
            "vocab_size": 128003,
            "dtype": "float32",
        },
        "lite_config.json": {
            "format": "jevk5-lite/1",
            "base": "microsoft/deberta-v3-large",
            "max_len": 512,
            "temperature_single": adapter.TEMPERATURE,
            "temperature_multi": 1.319507910772894,
        },
        "tokenizer_config.json": {
            "tokenizer_class": "DebertaV2Tokenizer",
            "backend": "tokenizers",
        },
    }
    for name in adapter.SOURCE_FILES:
        data = (
            json.dumps(contents[name]).encode() if name in contents else name.encode()
        )
        (root / name).write_bytes(data)
    hashes = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()
    }
    monkeypatch.setattr(adapter, "SOURCE_HASHES", hashes)
    return root


def test_source_identity_full_hash_and_unexpected_files(source):
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    assert adapter.identify_source(source) == adapter.MODEL_ID
    assert {p.name: p.read_bytes() for p in source.iterdir()} == before
    with pytest.raises(ValueError, match="unsupported Windows"):
        adapter.verify_source(source, "alibiserikbay/JevK5")
    (source / "model.safetensors").write_bytes(b"substituted")
    with pytest.raises(ValueError, match="checksum"):
        adapter.identify_source(source)
    (source / "model.safetensors").write_bytes(before["model.safetensors"])
    (source / "modeling_custom.py").write_text("pass", encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected"):
        adapter.verify_source(source)


def test_prepare_uses_fixed_revision_atomic_download_and_reuses_verified_source(
    source, tmp_path, monkeypatch
):
    import shutil

    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        shutil.copytree(source, kwargs["local_dir"])

    monkeypatch.setattr(adapter, "snapshot_download", download)
    output = adapter.prepare(adapter.MODEL_ID, tmp_path / "workspace")
    assert adapter.prepare(adapter.MODEL_ID, tmp_path / "workspace") == output
    assert len(calls) == 1
    assert calls[0]["revision"] == adapter.REVISION
    assert calls[0]["token"] is False
    assert set(calls[0]["allow_patterns"]) == set(adapter.SOURCE_FILES)
    assert not list(output.parent.glob(".jevk5-lite-download-*"))


@pytest.mark.parametrize("failure", [RuntimeError("network"), KeyboardInterrupt()])
def test_failed_download_cleans_private_partial_files(tmp_path, monkeypatch, failure):
    def download(**kwargs):
        kwargs["local_dir"].mkdir()
        (kwargs["local_dir"] / "partial").write_bytes(b"partial")
        raise failure

    monkeypatch.setattr(adapter, "snapshot_download", download)
    with pytest.raises(type(failure)):
        adapter.prepare(adapter.MODEL_ID, tmp_path)
    parent = tmp_path / "checkpoint" / "alibiserikbay"
    assert not (parent / "JevK5-Lite").exists()
    assert not list(parent.glob(".jevk5-lite-download-*"))


@pytest.mark.parametrize("threads", [0, 5, True, "4", -1])
def test_invalid_thread_count_precedes_source_loading(tmp_path, threads):
    with pytest.raises(ValueError, match="threads"):
        adapter.load_agent(tmp_path, threads=threads)


def test_loader_disables_remote_code_and_requires_complete_weights(source, monkeypatch):
    import torch
    import transformers
    from safetensors import torch as safetensors_torch

    calls = {}
    model = SimpleNamespace(
        config=SimpleNamespace(
            model_type="deberta-v2", hidden_size=1024, num_hidden_layers=24
        ),
        get_input_embeddings=lambda: SimpleNamespace(
            weight=SimpleNamespace(shape=(128003, 1024))
        ),
    )

    def load_model(path, **kwargs):
        calls["model"] = (path, kwargs)
        return model, {}

    def load_tokenizer(path, **kwargs):
        calls["tokenizer"] = (path, kwargs)
        return Tokenizer()

    class Head:
        def load_state_dict(self, tensors, strict):
            assert strict and tensors["weight"].dtype == torch.float32

    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", load_model)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", load_tokenizer)
    monkeypatch.setattr(
        safetensors_torch,
        "load_file",
        lambda *args, **kwargs: {"weight": torch.zeros(1)},
    )
    monkeypatch.setattr(adapter, "make_scorer", lambda hidden: Head())
    monkeypatch.setattr(
        adapter, "LiteAgent", lambda model, tokenizer, scorer, model_id: model
    )
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    assert adapter.load_agent(source, threads=1) is model
    assert calls["model"][1] == {
        "dtype": torch.float32,
        "attn_implementation": "eager",
        "device_map": {"": "cpu"},
        "local_files_only": True,
        "trust_remote_code": False,
        "output_loading_info": True,
    }
    assert calls["tokenizer"][1] == {
        "local_files_only": True,
        "trust_remote_code": False,
    }
    assert {p.name: p.read_bytes() for p in source.iterdir()} == before
    monkeypatch.setattr(
        transformers.AutoModel,
        "from_pretrained",
        lambda *a, **k: (model, {"missing_keys": ["weight"]}),
    )
    with pytest.raises(ValueError, match="not loaded completely"):
        adapter.load_agent(source, threads=1)


def test_question_and_global_candidate_limits_precede_forward():
    agent = make_agent()
    with pytest.raises(ValueError, match="1 to 64"):
        agent.system_one("state", {})
    q = {
        "type": "choice",
        "instructions": "?",
        "criteria": [str(i) for i in range(129)],
    }
    with pytest.raises(ValueError, match="256 candidates"):
        agent.system_one("state", {"one": q, "two": copy.deepcopy(q)})
    assert not agent.model.calls
