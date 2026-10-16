"""Open-Jev-2B CPU contracts using small random models, never released weights."""

import copy
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from vllm_jev import windows_open_jev as adapter
from vllm_jev.decision_template import ENVIRONMENT, current_template

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = {
    "route": {
        "type": "choice",
        "instructions": "Who can help?",
        "criteria": {"billing": "Refunds", "tech": None},
    },
    "refund": {"type": "noul", "instructions": "Refund requested?"},
    "urgency": {
        "type": "score",
        "instructions": "How urgent?",
        "criteria": ["Later", "Soon", "Today"],
    },
}
TEMPERATURE = 1.518796342858676


@pytest.fixture(autouse=True)
def isolated_template(monkeypatch):
    monkeypatch.delenv(ENVIRONMENT, raising=False)
    current_template.cache_clear()
    yield
    current_template.cache_clear()


class CharacterTokenizer:
    pad_token_id = 0
    chat_template = "test-template"

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return [1 + ord(letter) % 126 for letter in text]

    def apply_chat_template(
        self, messages, *, tokenize, add_generation_prompt, enable_thinking
    ):
        assert not tokenize and add_generation_prompt and not enable_thinking
        assert len(messages) == 1 and messages[0]["role"] == "user"
        return "<user>" + messages[0]["content"] + "</user><assistant>"


def text_config():
    from transformers import Qwen3_5TextConfig

    config = Qwen3_5TextConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        layer_types=["linear_attention", "full_attention"],
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 1.0,
            "mrope_section": [2, 3, 3],
        },
        pad_token_id=0,
        use_cache=False,
    )
    config._attn_implementation = "sdpa"
    return config


def attach_lora(model):
    import torch
    from peft import LoraConfig, get_peft_model

    model = get_peft_model(
        model,
        LoraConfig(
            r=2,
            lora_alpha=4,
            lora_dropout=0.0,
            bias="none",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "in_proj_qkv",
                "out_proj",
            ],
        ),
    )
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_B" in name:
                parameter.normal_(0, 0.08)
    return model.eval()


@pytest.fixture
def agent():
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "Qwen3_5TextModel"):
        pytest.skip("requires the isolated Qwen3.5 CPU environment")
    torch.set_num_threads(2)
    torch.manual_seed(742)
    model = attach_lora(
        transformers.Qwen3_5TextModel(text_config()).to(dtype=torch.bfloat16)
    )
    value = adapter.OpenJev2BAgent(
        model,
        CharacterTokenizer(),
        torch.randn(1, 64),
        torch.tensor([0.37]),
        temperature=TEMPERATURE,
    )
    yield value
    value.close()


def original_scores(agent, state, question, options=None):
    """Independent author's default prompt and final-token linear readout."""
    import torch

    state = (
        state
        if isinstance(state, str)
        else json.dumps(state, ensure_ascii=False, sort_keys=True)
    )
    question = (
        question
        if isinstance(question, str)
        else json.dumps(question, ensure_ascii=False, sort_keys=True)
    )
    prefix = f"Context:\n{state}\n\nQuestion: {question}\n"
    raw = (
        [prefix + "Is the answer to this question yes? Answer Yes or No."]
        if options is None
        else [
            prefix
            + f"Proposed answer: {option}\nIs this proposed answer correct? Answer Yes or No."
            for option in options
        ]
    )
    values = []
    with torch.inference_mode():
        for prompt in raw:
            ids = agent.tokenizer.encode("<user>" + prompt + "</user><assistant>")
            inputs = torch.tensor([ids], dtype=torch.long)
            hidden = (
                agent.model(
                    input_ids=inputs,
                    attention_mask=torch.ones_like(inputs),
                    use_cache=False,
                )
                .last_hidden_state[:, -1]
                .float()
            )
            # Preserve the author's [batch, hidden] nn.Linear call shape.
            head = torch.nn.Linear(hidden.shape[-1], 1, dtype=torch.float32)
            head.weight = torch.nn.Parameter(agent.weight)
            head.bias = torch.nn.Parameter(agent.bias)
            values.append(float(head(hidden)[0, 0].item()))
    return values


def independent_probabilities(scores):
    weights = [math.exp((value - max(scores)) / TEMPERATURE) for value in scores]
    return [value / sum(weights) for value in weights]


@pytest.mark.parametrize(
    "state", ["Please refund me.", {"message": "请退款。", "priority": 2}]
)
def test_real_hybrid_bf16_matches_independent_readout(agent, state):
    import torch

    expected = {
        "route": independent_probabilities(
            original_scores(agent, state, "Who can help?", ["billing: Refunds", "tech"])
        ),
        "refund": 1
        / (
            1
            + math.exp(
                -original_scores(agent, state, "Refund requested?")[0] / TEMPERATURE
            )
        ),
        "urgency": independent_probabilities(
            original_scores(agent, state, "How urgent?", ["Later", "Soon", "Today"])
        ),
    }
    calls = []

    def observe(module, args, kwargs):
        calls.append(kwargs["input_ids"].shape)
        assert kwargs["input_ids"].shape[0] == 1
        assert torch.all(kwargs["attention_mask"] == 1)
        assert not kwargs["use_cache"] and kwargs["return_dict"]

    handle = agent.model.register_forward_pre_hook(observe, with_kwargs=True)
    try:
        result = agent.system_one(state, QUESTIONS)
    finally:
        handle.remove()
    assert len(calls) == 6
    answers = result["answers"]
    for name in ("route", "urgency"):
        assert list(answers[name]["probabilities"].values()) == pytest.approx(
            expected[name], abs=1e-7
        )
    assert answers["refund"]["noul"] == pytest.approx(expected["refund"], abs=1e-7)
    assert answers["route"]["confidence"] == pytest.approx(
        2 * max(expected["route"]) - 1
    )
    p = expected["urgency"]
    mode = max(range(3), key=p.__getitem__)
    assert answers["urgency"]["score"] == pytest.approx(
        sum(i * value for i, value in enumerate(p))
    )
    assert answers["urgency"]["confidence"] == pytest.approx(
        max(0, 1 - sum(value * abs(i - mode) for i, value in enumerate(p)) / (2 / 3))
    )
    assert answers["urgency"]["legend"] == {"0": "Later", "1": "Soon", "2": "Today"}
    assert result["usage"]["input_tokens"] == sum(shape[1] for shape in calls)
    assert result["metadata"]["candidate_sequences"] == 6
    assert result["metadata"]["backbone_dtype"] == "bfloat16"
    assert result["metadata"]["head_dtype"] == "float32"
    assert result["metadata"]["candidate_batch_size"] == 1
    assert result["metadata"]["max_sequence_tokens"] == 1024


def test_nonzero_lora_changes_real_forward(agent):
    adapted = original_scores(agent, "Account", "Who?", ["billing"])
    with agent.model.disable_adapter():
        plain = original_scores(agent, "Account", "Who?", ["billing"])
    assert adapted != plain


def test_single_choice_does_not_encode_or_forward(agent, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("singleton must not encode or forward")

    monkeypatch.setattr(agent, "_encode", forbidden)
    monkeypatch.setattr(agent, "_scores", forbidden)
    result = agent.system_one(
        "state",
        {"only": {"type": "choice", "instructions": "Pick", "criteria": {"yes": None}}},
    )
    assert result["answers"]["only"] == {
        "type": "choice",
        "choice": "yes",
        "probabilities": {"yes": 1.0},
        "confidence": 1.0,
    }
    assert result["usage"]["input_tokens"] == 0


@pytest.mark.parametrize(
    "change",
    [
        {"type": "choice", "instructions": "Pick", "criteria": []},
        {"type": "score", "instructions": "Rate", "criteria": ["one", True]},
        {"type": "score", "instructions": "Rate", "criteria": ["one"]},
        {"type": "noul", "instructions": "Yes?", "criteria": {"unknown": "Yes"}},
        {"type": "noul", "instructions": " "},
        {"type": "noul", "instructions": {"bad": float("nan")}},
        {
            "type": "choice",
            "instructions": "A" * 1024,
            "criteria": {"yes": None, "no": None},
        },
    ],
)
def test_all_questions_are_checked_before_any_forward(agent, monkeypatch, change):
    def forbidden(*args):
        pytest.fail("invalid later question must prevent every forward")

    monkeypatch.setattr(agent, "_scores", forbidden)
    questions = copy.deepcopy(QUESTIONS)
    questions["last"] = change
    with pytest.raises(ValueError):
        agent.system_one("state", questions)


def test_1024_complete_prompt_boundary_no_truncation(agent, monkeypatch):
    monkeypatch.setattr(agent.tokenizer, "encode", lambda *args, **kwargs: [1] * 1024)
    assert len(agent._encode("prompt")) == 1024
    monkeypatch.setattr(agent.tokenizer, "encode", lambda *args, **kwargs: [1] * 1025)
    with pytest.raises(ValueError, match="1024.*no truncation"):
        agent._encode("prompt")


@pytest.mark.parametrize(
    "criteria, suffix",
    [
        ({}, ""),
        ({"true": {"refund": True}}, '\nYes means: {"refund": true}'),
        ({"false": "No refund"}, "\nNo means: No refund"),
        (
            {"false": "No refund", "true": {"refund": True}},
            '\nYes means: {"refund": true}\nNo means: No refund',
        ),
    ],
)
def test_noul_custom_criteria_is_one_branch_with_bias(
    agent, monkeypatch, criteria, suffix
):
    captured = []
    original = agent.tokenizer.apply_chat_template

    def record(messages, **kwargs):
        captured.append(messages[0]["content"])
        return original(messages, **kwargs)

    monkeypatch.setattr(agent.tokenizer, "apply_chat_template", record)
    question = {
        "type": "noul",
        "instructions": {"ask": "退款?"},
        "criteria": criteria,
    }
    result = agent.system_one("state", {"q": question})
    assert len(captured) == 1
    assert captured[0] == (
        'Context:\nstate\n\nQuestion: {"ask": "退款?"}'
        + suffix
        + "\nIs the answer to this question yes? Answer Yes or No."
    )
    assert result["metadata"]["candidate_sequences"] == 1
    with_bias = result["answers"]["q"]["noul"]
    agent.bias.zero_()
    assert (
        agent.system_one("state", {"q": question})["answers"]["q"]["noul"] != with_bias
    )


@pytest.mark.parametrize("score, expected", [(-1e8, 0.0), (1e8, 1.0), (0.0, 0.5)])
def test_noul_sigmoid_extremes(agent, monkeypatch, score, expected):
    monkeypatch.setattr(agent, "_scores", lambda branches: [score])
    result = agent.system_one("s", {"q": QUESTIONS["refund"]})
    assert result["answers"]["q"]["noul"] == expected


def test_choice_tie_uses_input_order_not_description_sort(agent, monkeypatch):
    monkeypatch.setattr(agent, "_scores", lambda branches: [0.0] * len(branches))
    result = agent.system_one(
        "s",
        {
            "q": {
                "type": "choice",
                "instructions": "Pick",
                "criteria": {"z": "Zulu", "a": "Alpha"},
            }
        },
    )
    assert result["answers"]["q"]["choice"] == "z"
    assert result["answers"]["q"]["confidence"] == 0


@pytest.mark.parametrize(
    "criteria",
    [
        ["same", "same"],
        [{"priority": 1, "message": "退款"}, {"message": "退款", "priority": 1}],
    ],
)
def test_repeated_score_descriptions_keep_independent_levels(
    agent, monkeypatch, criteria
):
    prompts = []
    original = agent.tokenizer.apply_chat_template

    def record(messages, **kwargs):
        prompts.append(messages[0]["content"])
        return original(messages, **kwargs)

    monkeypatch.setattr(agent.tokenizer, "apply_chat_template", record)
    result = agent.system_one(
        "state", {"q": {"type": "score", "instructions": "Rate", "criteria": criteria}}
    )
    assert len(prompts) == 2 and prompts[0] == prompts[1]
    assert result["answers"]["q"] == {
        "type": "score",
        "score": 0.5,
        "probabilities": {"0": 0.5, "1": 0.5},
        "confidence": 0.0,
        "legend": {"0": criteria[0], "1": criteria[1]},
    }
    assert result["metadata"]["candidate_sequences"] == 2


def test_choice_rendered_collision_preserves_both_keys(agent, monkeypatch):
    prompts = []
    original = agent.tokenizer.apply_chat_template

    def record(messages, **kwargs):
        prompts.append(messages[0]["content"])
        return original(messages, **kwargs)

    monkeypatch.setattr(agent.tokenizer, "apply_chat_template", record)
    result = agent.system_one(
        "state",
        {
            "q": {
                "type": "choice",
                "instructions": "Pick",
                "criteria": {"a": "b", "a: b": None},
            }
        },
    )
    assert len(prompts) == 2 and prompts[0] == prompts[1]
    assert "Proposed answer: a: b\n" in prompts[0]
    assert result["answers"]["q"] == {
        "type": "choice",
        "choice": "a",
        "probabilities": {"a": 0.5, "a: b": 0.5},
        "confidence": 0.0,
    }


def test_precision_is_observed_and_env_cannot_change_it(agent, monkeypatch):
    monkeypatch.setenv("JEV_TORCH_DTYPE", "float32")
    result = agent.system_one("s", {"q": QUESTIONS["refund"]})
    assert result["metadata"]["backbone_dtype"] == "bfloat16"
    agent.model.float()
    with pytest.raises(ValueError, match="BF16"):
        agent.system_one("s", {"q": QUESTIONS["refund"]})


@pytest.mark.parametrize("invalid", ["dtype", "meta", "nan", "shape"])
def test_invalid_actual_head_is_rejected_before_serving(agent, invalid):
    import torch

    weight, bias = agent.weight, agent.bias
    if invalid == "dtype":
        weight = weight.to(torch.bfloat16)
    elif invalid == "meta":
        weight = torch.empty_like(weight, device="meta")
    elif invalid == "nan":
        bias = torch.tensor([float("nan")])
    else:
        weight = weight[:, :-1]
    with pytest.raises(ValueError):
        adapter.OpenJev2BAgent(
            agent.model, agent.tokenizer, weight, bias, temperature=TEMPERATURE
        )


def test_candidate_budget_counts_noul_as_one():
    questions = {
        "choice": {
            "type": "choice",
            "instructions": "Pick",
            "criteria": {f"v{i}": None for i in range(255)},
        },
        "noul": QUESTIONS["refund"],
    }
    assert adapter._questions(questions)[1] == 256
    questions["extra"] = QUESTIONS["refund"]
    with pytest.raises(ValueError, match="256"):
        adapter._questions(questions)


@pytest.mark.parametrize(
    "config",
    [
        {"layout": "instructions-first"},
        {"context_order": ["instructions", "criteria", "state"]},
        {
            "choice_prompt": "{instructions}|{state}|{candidate}",
            "noul_prompt": "{instructions}?{state}",
        },
        {
            "choice_prompt": "{options}|{instructions}|{state}|{candidate}",
            "noul_prompt": "{instructions}?{state}",
        },
    ],
)
def test_template_prompt_variants_are_applied(agent, monkeypatch, config):
    monkeypatch.setenv(ENVIRONMENT, json.dumps(config))
    current_template.cache_clear()
    prompts = []
    monkeypatch.setattr(agent, "_encode", lambda raw: prompts.append(raw) or [1])
    monkeypatch.setattr(agent, "_scores", lambda branches: [0.0] * len(branches))
    agent.system_one(
        "STATE",
        {
            "q": {
                "type": "choice",
                "instructions": "INSTRUCTION",
                "criteria": {"a": None, "b": None},
            },
            "n": {"type": "noul", "instructions": "INSTRUCTION"},
        },
    )
    assert len(prompts) == 3
    assert all(
        prompt.index("INSTRUCTION") < prompt.index("STATE") for prompt in prompts
    )
    if "{options}" in config.get("choice_prompt", ""):
        assert prompts[0].startswith('["a", "b"]|')
        assert prompts[1].startswith('["a", "b"]|')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def source(tmp_path, monkeypatch):
    root = tmp_path / "中文 checkpoint"
    data = {name: b"test bytes" for name in adapter.SOURCE_HASHES}
    data["package/checkpoint/model.json"] = json.dumps(
        {
            "model_id": adapter.BASE_MODEL_ID,
            "revision": adapter.BASE_REVISION,
            "max_length": 4096,
            "lora_rank": 8,
        }
    ).encode()
    data["release-manifest.json"] = json.dumps(
        {
            "repo_id": adapter.MODEL_ID,
            "base_model": adapter.BASE_MODEL_ID,
            "base_revision": adapter.BASE_REVISION,
        }
    ).encode()
    data["base/config.json"] = json.dumps(
        {"model_type": "qwen3_5", "text_config": {"model_type": "qwen3_5_text"}}
    ).encode()
    for name, content in data.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    monkeypatch.setattr(
        adapter,
        "SOURCE_HASHES",
        {name: hashlib.sha256(content).hexdigest() for name, content in data.items()},
    )
    return root


def test_source_verifies_all_pinned_artifacts_without_modification(source):
    before = {name: digest(source / name) for name in adapter.SOURCE_HASHES}
    assert adapter.identify_source(source) == adapter.MODEL_ID
    assert before == {name: digest(source / name) for name in adapter.SOURCE_HASHES}
    (source / "LICENSE-CODE").write_text("changed")
    with pytest.raises(ValueError, match="checksum.*LICENSE-CODE"):
        adapter.verify_source(source)


@pytest.mark.parametrize(
    "name",
    [
        "base/adapter_config.json",
        "base/added_tokens.json",
        "base/chat_templates/custom.jinja",
        "package/checkpoint/adapter/adapter_model.bin",
        "package/checkpoint/tokenizer.json",
    ],
)
def test_source_rejects_loader_override_files(source, name):
    path = source / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("override")
    with pytest.raises(ValueError, match="unexpected"):
        adapter.verify_source(source)


def test_failed_download_does_not_publish_or_leave_temporary_source(
    tmp_path, monkeypatch
):
    def fail(**kwargs):
        kwargs["local_dir"].mkdir(parents=True)
        (kwargs["local_dir"] / "partial").write_text("partial")
        raise OSError("network failure")

    monkeypatch.setattr(adapter, "snapshot_download", fail)
    with pytest.raises(OSError, match="network failure"):
        adapter.prepare(adapter.MODEL_ID, tmp_path)
    parent = tmp_path / "checkpoint/ZefanCai"
    assert not (parent / "Open-Jev-2B").exists()
    assert not list(parent.glob(".open-jev-download-*"))


def test_import_does_not_load_torch_transformers_or_vllm():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import vllm_jev.windows_open_jev; assert not any(x in sys.modules for x in ('torch', 'transformers', 'peft', 'vllm', 'mlx'))",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_real_small_original_wrapper_and_adapter_load(tmp_path, monkeypatch):
    import torch
    import transformers
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

    torch.set_num_threads(2)
    torch.manual_seed(157)
    source = tmp_path / "模型 with spaces"
    base = source / "base"
    package = source / "package/checkpoint"
    package.mkdir(parents=True)
    config = Qwen3_5Config(
        text_config=text_config().to_dict(),
        vision_config={
            "depth": 1,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 4,
            "out_hidden_size": 64,
            "num_position_embeddings": 16,
            "patch_size": 2,
            "temporal_patch_size": 2,
            "spatial_merge_size": 2,
        },
    )
    full = Qwen3_5ForConditionalGeneration(config).to(dtype=torch.bfloat16)
    full.save_pretrained(base)
    model = attach_lora(full.model.language_model)
    model.save_pretrained(package / "adapter")
    torch.save(
        {"weight": torch.randn(1, 64), "bias": torch.tensor([0.2])}, package / "head.pt"
    )
    (package / "temperature.json").write_text(json.dumps({"temperature": TEMPERATURE}))
    del model, full
    before = {
        str(path.relative_to(source)): digest(path)
        for path in source.rglob("*")
        if path.is_file()
    }
    checked = []
    monkeypatch.setattr(
        adapter,
        "verify_source",
        lambda path, model_id: checked.append((path, model_id)),
    )

    def tokenizer_loader(path, **kwargs):
        assert path == base
        assert kwargs == {"local_files_only": True, "trust_remote_code": False}
        return CharacterTokenizer()

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", tokenizer_loader)
    original_loader = transformers.AutoModelForImageTextToText.from_pretrained

    def inspect_loader(path, **kwargs):
        assert path == base
        assert kwargs["dtype"] == torch.bfloat16
        assert kwargs["device_map"] == {"": "cpu"}
        assert kwargs["attn_implementation"] == "sdpa"
        assert kwargs["local_files_only"] and not kwargs["trust_remote_code"]
        assert kwargs["output_loading_info"]
        return original_loader(path, **kwargs)

    monkeypatch.setattr(
        transformers.AutoModelForImageTextToText, "from_pretrained", inspect_loader
    )
    monkeypatch.setenv("JEV_TORCH_DTYPE", "float32")
    loaded = adapter.load_agent(source, threads=2)
    try:
        assert checked == [(source.resolve(), adapter.MODEL_ID)]
        assert loaded.model.get_input_embeddings().weight.dtype == torch.bfloat16
        assert all(
            parameter.device.type == "cpu" and not parameter.is_meta
            for parameter in loaded.model.parameters()
        )
        result = loaded.system_one("Refund", {"q": QUESTIONS["refund"]})
        assert 0 <= result["answers"]["q"]["noul"] <= 1
        assert result["metadata"]["head_dtype"] == "float32"
    finally:
        loaded.close()
    assert before == {
        str(path.relative_to(source)): digest(path)
        for path in source.rglob("*")
        if path.is_file()
    }
