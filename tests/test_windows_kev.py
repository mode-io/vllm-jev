"""Kev CPU contracts, using deterministic tiny states and no released weights."""

import copy
import hashlib
import importlib.util
import itertools
import json
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm_jev import windows_kev as adapter
from vllm_jev.decision_template import ENVIRONMENT, current_template

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = {
    "route": {
        "type": "choice",
        "instructions": "Route?",
        "criteria": {"bill": "Refund", "tech": None},
    },
    "yes": {
        "type": "noul",
        "instructions": "Refund?",
        "criteria": {"true": "Return funds"},
    },
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
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return [ord(character) % 1018 + 6 for character in text]

    def convert_tokens_to_ids(self, token):
        return adapter.MARKERS.index(token) + 1


def make_agent(model_id=adapter.MODEL_ID):
    import torch

    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            generator = torch.Generator().manual_seed(915)
            self.embedding = torch.nn.Embedding.from_pretrained(
                torch.randn(1024, 16, generator=generator)
            )
            self.calls = []

        def get_input_embeddings(self):
            return self.embedding

        def forward(
            self, *, input_ids, position_ids, attention_mask, use_cache, return_dict
        ):
            # Deliberately no past_key_values argument: each forward owns a new
            # author-compatible cache, which the adapter never retains/reuses.
            assert use_cache and return_dict
            assert input_ids.shape[0] == 1
            assert torch.equal(
                position_ids, torch.arange(input_ids.shape[1]).unsqueeze(0)
            )
            assert attention_mask.all()
            self.calls.append(input_ids.clone())
            embeddings = self.embedding(input_ids)
            states = embeddings.cumsum(1) / (position_ids + 1).unsqueeze(-1)
            return SimpleNamespace(last_hidden_state=states)

    generator = torch.Generator().manual_seed(916)
    head = {
        name: torch.randn(shape, generator=generator) * 0.2
        for name, shape in {
            "q.weight": (256, 16),
            "q.bias": (256,),
            "k.weight": (256, 16),
            "k.bias": (256,),
        }.items()
    }
    return adapter.KevAgent(
        Backbone(),
        Tokenizer(),
        head,
        adapter.MODELS[model_id]["temperature"],
        model_id=model_id,
    )


def render(value, indent=0):
    """Independent reconstruction of the pinned author's JSON flattening."""
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    padding = "  " * indent
    if isinstance(value, list):
        return "\n".join(
            padding + "- " + render(child, indent + 1).lstrip() for child in value
        )
    return "\n".join(
        padding + key + ":\n" + render(child, indent + 1)
        if isinstance(child, (dict, list))
        else padding + key + ": " + render(child)
        for key, child in value.items()
    )


def author_row(state, question):
    def tokens(value):
        return Tokenizer().encode(re.sub(r"<\|([A-Za-z0-9_]+)\|>", r"<¦\1¦>", value))

    def option(key, value):
        return key if value in (None, "") else key + ": " + render(value)

    kind, criteria = question["type"], question.get("criteria")
    if kind == "noul":
        criteria = criteria or {}
        options = [
            option("no", criteria.get("false")),
            option("yes", criteria.get("true")),
        ]
    elif kind == "choice":
        options = [option(key, value) for key, value in criteria.items()]
    else:
        options = [render(value) for value in criteria]
    ids = (
        [1] + tokens(render(state)) + [2] + tokens(render(question.get("instructions")))
    )
    positions = []
    for text in options:
        ids += [3] + tokens(text) + [4]
        positions.append(len(ids) - 1)
    ids.append(5)
    positions.append(len(ids) - 1)
    return ids, positions


def independent_probabilities(agent, state, question):
    import torch
    import torch.nn.functional as F

    ids, positions = author_row(state, question)
    states = (
        agent.model.embedding(torch.tensor(ids)).cumsum(0)
        / torch.arange(1, len(ids) + 1)[:, None]
    )
    query = F.linear(
        states[positions[-1]], agent.head["q.weight"], agent.head["q.bias"]
    )
    keys = F.linear(
        states[positions[:-1]], agent.head["k.weight"], agent.head["k.bias"]
    )
    logits = (keys @ query) * (1 / 16)
    return torch.softmax(logits / agent.temperature, -1).tolist(), ids, positions


def test_import_is_lightweight():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import vllm_jev.windows_kev; assert not any(n in sys.modules for n in ('torch','transformers','vllm','mlx'))",
        ],
        cwd=ROOT,
        check=True,
    )


@pytest.mark.parametrize(
    "state",
    [
        "Duplicate charge <|box_end|>",
        {"用户": "退款", "事件": ["重复扣款", {"金额": 10}], "ok": True},
        ["Unicode ✓", None, False],
    ],
)
@pytest.mark.parametrize("model_id", [adapter.MODEL_ID, adapter.MODEL_ID_06])
def test_exact_marker_rows_and_fp32_pointer_semantics(state, model_id):
    agent = make_agent(model_id)
    result = agent.system_one(state, QUESTIONS)
    assert result["model"] == model_id
    assert result["metadata"]["temperature"] == adapter.MODELS[model_id]["temperature"]
    for index, (identifier, question) in enumerate(QUESTIONS.items()):
        expected, ids, _ = independent_probabilities(agent, state, question)
        answer = result["answers"][identifier]
        if question["type"] == "noul":
            assert answer["noul"] == pytest.approx(expected[1], abs=1e-8)
        else:
            assert list(answer["probabilities"].values()) == pytest.approx(
                expected, abs=1e-8
            )
        assert agent.model.calls[index].tolist() == [ids]
    probabilities = result["answers"]["urgency"]["probabilities"]
    assert result["answers"]["urgency"]["score"] == sum(
        int(k) * p for k, p in probabilities.items()
    )
    assert result["metadata"]["backbone_dtype"] == "float32"
    assert result["metadata"]["head_dtype"] == "float32"
    assert result["metadata"]["row_batch_size"] == 1
    assert result["usage"]["output_tokens"] == 0


@pytest.mark.parametrize(
    "criteria",
    [None, {}, {"true": "yes description"}, {"false": {"否": [False, None]}}],
)
def test_noul_partial_criteria_and_absent_instructions(criteria):
    agent = make_agent()
    question = {"type": "noul", "criteria": criteria}
    expected, ids, _ = independent_probabilities(agent, "state", question)
    result = agent.system_one("state", {"q": question})
    assert result["answers"]["q"]["noul"] == pytest.approx(expected[1])
    assert agent.model.calls[0].tolist() == [ids]


def test_joint_score_duplicates_and_singletons():
    agent = make_agent()
    questions = {
        "collision": {"type": "choice", "criteria": {"a": "b", "a: b": None}},
        "score": {"type": "score", "criteria": [{"level": 1}, {"level": 1}]},
        "one": {"type": "score", "criteria": ["Only"]},
        "singleton": {"type": "choice", "criteria": ["Only"]},
    }
    result = agent.system_one("state", questions)
    assert len(agent.model.calls) == 3  # Score/Choice singleton rows are identical.
    assert result["answers"]["one"]["score"] == 0
    assert result["answers"]["one"]["confidence"] == 1
    assert result["answers"]["singleton"]["probabilities"] == {"Only": 1.0}
    assert list(result["answers"]["score"]["legend"].values()) == [
        "level: 1",
        "level: 1",
    ]
    # Identical descriptions occupy distinct causal positions; never deduplicate
    # options or assume their pointer probabilities must equal each other.
    expected, _, _ = independent_probabilities(agent, "state", questions["collision"])
    assert list(
        result["answers"]["collision"]["probabilities"].values()
    ) == pytest.approx(expected)


def test_repeated_rows_share_work_within_one_request_only():
    agent = make_agent()
    one = QUESTIONS["route"]
    agent.system_one("state", {"a": one, "b": copy.deepcopy(one)})
    assert len(agent.model.calls) == 1
    agent.system_one("state", {"a": one})
    assert len(agent.model.calls) == 2


@pytest.mark.parametrize(
    "question",
    [
        {"type": "noul", "criteria": {"yes": "unsupported"}},
        {"type": "choice", "criteria": []},
        {"type": "choice", "criteria": ["a", "a"]},
        {"type": "choice", "criteria": {"": "bad"}},
        {"type": "score", "criteria": []},
        {"type": "score", "criteria": ["a"], "isolated": True},
        {"type": "score", "criteria": [float("nan")]},
        {"type": "other", "criteria": ["a"]},
        {"type": "noul", "instructions": "x" * 1024},
    ],
)
def test_all_inputs_rejected_before_any_forward(question):
    agent = make_agent()
    with pytest.raises((ValueError, TypeError)):
        agent.system_one("state", {"good": QUESTIONS["route"], "bad": question})
    assert not agent.model.calls


@pytest.mark.parametrize(
    "state",
    [
        None,
        1,
        True,
        {"n": float("inf")},
        "x" * 50_001,
        "x" * 1024,
        {"messages": [{"content": [{"type": "image_url", "image_url": "x"}]}]},
    ],
    ids=[
        "null",
        "number",
        "bool",
        "nonfinite",
        "character-budget",
        "token-budget",
        "media",
    ],
)
def test_invalid_state_is_atomic(state):
    agent = make_agent()
    with pytest.raises((ValueError, TypeError)):
        agent.system_one(state, QUESTIONS)
    assert not agent.model.calls


def test_question_and_candidate_budgets():
    agent = make_agent()
    for questions in (
        {},
        {str(i): QUESTIONS["yes"] for i in range(65)},
        {"a": {"type": "score", "criteria": [""] * 255}, "b": {"type": "noul"}},
    ):
        with pytest.raises(ValueError):
            agent.system_one("", questions)
    assert not agent.model.calls


@pytest.mark.parametrize(
    "order", list(itertools.permutations(["state", "instructions", "criteria"]))
)
def test_context_order_preserves_pointer_positions(monkeypatch, tmp_path, order):
    monkeypatch.setenv(ENVIRONMENT, json.dumps({"context_order": order}))
    current_template.cache_clear()
    agent = make_agent()
    result = agent.system_one("state", QUESTIONS)
    assert len(result["answers"]) == 3
    for ids in agent.model.calls:
        assert ids[0, -1].item() == 5
        assert (ids == 4).sum() >= 2


def test_instructions_first_layout(monkeypatch):
    monkeypatch.setenv(ENVIRONMENT, '{"layout":"instructions-first"}')
    current_template.cache_clear()
    agent = make_agent()
    agent.system_one("state", {"q": QUESTIONS["route"]})
    assert agent.model.calls[0][0, 0].item() == 2


def test_precision_and_close_are_truthful():
    import torch

    agent = make_agent()
    agent.model.embedding.to(torch.bfloat16)
    with pytest.raises(ValueError, match="FP32"):
        agent.system_one("state", QUESTIONS)
    agent = make_agent()
    agent.head["q.weight"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        agent.system_one("state", QUESTIONS)
    agent = make_agent()
    agent.close()
    agent.close()
    with pytest.raises(RuntimeError, match="closed"):
        agent.system_one("state", QUESTIONS)


def small_source(tmp_path, monkeypatch):
    source = tmp_path / "source"
    files = {
        "base/config.json": {
            "model_type": "qwen3_5",
            "text_config": {
                "model_type": "qwen3_5_text",
                "hidden_size": 1024,
                "num_hidden_layers": 24,
            },
        },
        "adapter_config.json": {
            "base_model_name_or_path": adapter.BASE_ID,
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "task_type": "FEATURE_EXTRACTION",
            "target_modules": sorted(adapter.LORA_TARGETS),
            "bias": "none",
        },
        "training_config.json": {
            "base_revision": adapter.BASE_REVISION,
            "args": {"base": adapter.BASE_ID},
        },
        "base/weights.safetensors": {"fixture": True},
    }
    hashes = {}
    for name, value in files.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(adapter, "SOURCE_HASHES", hashes)
    monkeypatch.setitem(adapter.MODELS[adapter.MODEL_ID], "source_hashes", hashes)
    monkeypatch.setattr(adapter, "SOURCE_FILES", tuple(hashes))
    return source, hashes


def test_source_identity_bytes_and_allowlist(tmp_path, monkeypatch):
    source, hashes = small_source(tmp_path, monkeypatch)
    assert adapter.identify_source(source) == adapter.MODEL_ID
    with pytest.raises(ValueError, match="unsupported"):
        adapter.verify_source(source, "another/model")
    rogue = source / "base/adapter_config.json"
    rogue.write_text("{}")
    with pytest.raises(ValueError, match="unexpected"):
        adapter.verify_source(source)
    rogue.unlink()
    (source / "base/weights.safetensors").write_text("changed")
    with pytest.raises(ValueError, match="checksum"):
        adapter.verify_source(source)


def test_download_exception_cleans_temporary_source(tmp_path, monkeypatch):
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        kwargs["local_dir"].mkdir(parents=True)
        (kwargs["local_dir"] / "partial").write_bytes(b"partial")
        raise RuntimeError("connection interrupted")

    monkeypatch.setattr(adapter, "snapshot_download", download)
    with pytest.raises(RuntimeError, match="connection"):
        adapter.prepare(adapter.MODEL_ID, tmp_path)
    parent = tmp_path / "checkpoint/jaredpalmer"
    assert not list(parent.glob(".kev-download-*"))
    assert not (parent / "kev-0.8b").exists()
    assert calls[0]["revision"] == adapter.REVISION
    assert calls[0]["token"] is False


def test_loader_checks_complete_lora_then_merges_fp32_without_writing_source(
    tmp_path, monkeypatch
):
    """Exercise real PEFT safetensors loading/merging on a tiny linear backbone."""
    import torch
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM, AutoTokenizer, PretrainedConfig

    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = PretrainedConfig(hidden_size=1024)
            self.config.model_type = "qwen3_5_text"
            self.embedding = torch.nn.Embedding(1024, 16)
            self.q_proj = torch.nn.Linear(16, 16, bias=False)

        def get_input_embeddings(self):
            return self.embedding

    torch.manual_seed(917)
    original = Backbone()
    tuning = {
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj"],
    }
    trained = get_peft_model(
        copy.deepcopy(original), LoraConfig(task_type="FEATURE_EXTRACTION", **tuning)
    )
    with torch.no_grad():
        for name, value in trained.named_parameters():
            if "lora_" in name:
                value.copy_(torch.randn_like(value) * 0.01)
    expected_state = get_peft_model_state_dict(trained, save_embedding_layers=False)
    save_file(expected_state, str(tmp_path / "adapter_model.safetensors"))
    expected_merged = trained.merge_and_unload().q_proj.weight.detach().clone()
    head = make_agent().head
    torch.save(
        {
            "base": adapter.BASE_ID,
            "base_revision": adapter.BASE_REVISION,
            "lora": 16,
            "head_dim": 256,
            "option_isolation": False,
            "special_embeddings": False,
            "weights_dtype": "fp32",
            "temperature": adapter.TEMPERATURE,
            "head": head,
        },
        tmp_path / "head.pt",
    )
    before = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.iterdir()
    }
    monkeypatch.setattr(adapter, "verify_source", lambda *args: tuning)
    seen = {}

    def load(path, **kwargs):
        seen.update(kwargs)
        assert path == tmp_path / "base"
        return SimpleNamespace(model=copy.deepcopy(original)), {}

    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", load)
    monkeypatch.setattr(
        AutoTokenizer, "from_pretrained", lambda path, **kwargs: Tokenizer()
    )
    agent = adapter.load_agent(tmp_path, threads=1)
    assert torch.equal(agent.model.q_proj.weight, expected_merged)
    assert not any("lora_" in name for name, _ in agent.model.named_parameters())
    assert seen["dtype"] == torch.float32
    assert seen["attn_implementation"] == "eager"
    assert seen["local_files_only"] is True and seen["trust_remote_code"] is False
    assert agent.model.config.use_cache is True
    assert before == {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.iterdir()
    }
    # A missing adapter tensor must fail, even when PEFT's non-strict setter
    # could otherwise leave a randomly initialized adapter behind.
    expected_state.pop(next(iter(expected_state)))
    save_file(expected_state, str(tmp_path / "adapter_model.safetensors"))
    with pytest.raises(ValueError, match="LoRA keys"):
        adapter.load_agent(tmp_path, threads=1)


@pytest.mark.parametrize("threads", [0, True, -1, 100000])
def test_loader_rejects_thread_configuration_before_file_access(tmp_path, threads):
    with pytest.raises(ValueError, match="threads"):
        adapter.load_agent(tmp_path, threads=threads)


def reference_module():
    path = ROOT / "scripts/windows_kev_reference.py"
    spec = importlib.util.spec_from_file_location("kev_reference_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_independent_reference_pins_match_and_block_product():
    reference = reference_module()
    assert reference.SOURCE_HASHES == adapter.SOURCE_HASHES
    assert reference.REVISION == adapter.REVISION
    assert reference.BASE_REVISION == adapter.BASE_REVISION
    with pytest.raises(RuntimeError, match="product"):
        reference.BlockProductImports().find_spec("vllm_jev.windows_kev")


@pytest.mark.parametrize(
    "payload",
    [
        '[{"id":"x","id":"y","state":"s","questions":{"q":{}}}]',
        '[{"id":"x","state":NaN,"questions":{"q":{}}}]',
        "[]",
    ],
)
def test_reference_cases_reject_ambiguous_data(tmp_path, payload):
    path = tmp_path / "cases.json"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError):
        reference_module().read_cases(path)
