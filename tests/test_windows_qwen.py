"""Windows Qwen boundaries and real tiny-random Qwen3 forward regression tests.

These tests never download checkpoints. Random weights test the adapter's
numerical mechanics; the separate CLI driver checks pretrained model parity.
"""

import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm_jev.decision_template import ENVIRONMENT, current_template

ROOT = Path(__file__).resolve().parents[1]
MODEL = "lostargon/Tiny-Jev"
QUESTIONS = {
    "route": {
        "type": "choice",
        "instructions": "Who can help?",
        "criteria": {"billing": "Refunds", "tech": "Login"},
    },
    "refund": {"type": "noul", "instructions": "Refund requested?"},
    "urgency": {
        "type": "score",
        "instructions": "How urgent?",
        "criteria": ["Later", "Soon", "Today"],
    },
}


@pytest.fixture(autouse=True)
def isolated_template(monkeypatch):
    monkeypatch.delenv(ENVIRONMENT, raising=False)
    current_template.cache_clear()
    yield
    current_template.cache_clear()


class CharacterTokenizer:
    """A deterministic tokenizer with one newline token and a distinct marker."""

    unk_token_id = None

    def encode(self, text, add_special_tokens=False):
        return [1 if letter == "\n" else 2 + ord(letter) % 61 for letter in text]

    def convert_tokens_to_ids(self, token):
        return 63 if token == "<opt>" else None


@pytest.fixture
def tiny_agent():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "Qwen3Model"):
        pytest.skip("Qwen3 requires the Windows Qwen dependency environment")
    from vllm_jev.windows_tiny import TinyAgent

    torch.manual_seed(231)
    config = transformers.Qwen3Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=4096,
        rope_theta=1_000_000,
        attention_dropout=0.0,
        use_cache=False,
    )
    config._attn_implementation = "sdpa"
    model = transformers.Qwen3Model(config).eval()
    weight = torch.randn(1, config.hidden_size)
    bias = torch.tensor([0.37])
    agent = TinyAgent(model, CharacterTokenizer(), weight, bias, temperature=0.95)
    yield agent
    agent.close()


def original_readout(agent, state, question, options):
    """Independent transcription of the pinned original's encode/forward math.

    It deliberately avoids project tiny_token_ids, _scores and choice_result.
    The pretrained driver executes the complete original source separately.
    """
    import torch
    from torch.nn import functional as functional

    text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
    tokenizer = agent.tokenizer
    ids = tokenizer.encode(
        "<state>\n"
        + text
        + "\n</state>\n<question>\n"
        + question
        + "\n</question>\n<options>\n"
    )
    positions = []
    for option in options:
        ids.extend(tokenizer.encode(option + " - correct?"))
        positions.append(len(ids) - 1)
        ids.append(63)
        ids.extend(tokenizer.encode("\n"))
    ids.extend(tokenizer.encode("</options>"))
    tokens = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        hidden = agent.model(
            input_ids=tokens, attention_mask=torch.ones_like(tokens), use_cache=False
        ).last_hidden_state
        selected = hidden[:, positions, :].float()
        normalized = functional.layer_norm(selected, (hidden.shape[-1],))
        logits = functional.linear(normalized, agent.weight, agent.bias).squeeze(-1)
        probabilities = torch.softmax(logits / 0.95, dim=-1)[0].tolist()
    return probabilities


@pytest.mark.parametrize(
    "state", ["Please refund me.", {"message": "请退款。", "priority": 2}]
)
def test_real_qwen3_forward_matches_original_marker_math(tiny_agent, state):
    result = tiny_agent.system_one(state, QUESTIONS)
    answers = result["answers"]
    for identifier, definition in QUESTIONS.items():
        kind = definition["type"]
        criteria = definition.get("criteria")
        if kind == "choice":
            options = [
                f"{label}: {description}" for label, description in criteria.items()
            ]
            keys = list(criteria)
        elif kind == "score":
            keys = options = criteria
        else:
            options = ["no", "yes"]
        expected = original_readout(
            tiny_agent, state, definition["instructions"], options
        )
        actual = answers[identifier]
        assert actual["type"] == kind
        if kind == "noul":
            assert actual["noul"] == pytest.approx(expected[1], abs=1e-6)
        else:
            assert actual["probabilities"] == pytest.approx(
                dict(zip(keys, expected)), abs=1e-6
            )
            assert actual["confidence"] == pytest.approx(max(expected), abs=1e-6)
            if kind == "choice":
                assert (
                    actual["choice"]
                    == keys[max(range(len(expected)), key=expected.__getitem__)]
                )
            else:
                assert actual["score"] == pytest.approx(
                    sum(i * p for i, p in enumerate(expected)), abs=1e-6
                )
                assert actual["legend"] == dict(zip(keys, criteria))
    assert result["usage"]["input_tokens"] > 0
    assert result["usage"]["output_tokens"] == 0
    assert result["metadata"]["cached_tokens"] == 0
    assert all(parameter.grad is None for parameter in tiny_agent.model.parameters())


def test_single_choice_does_not_forward_and_noul_criteria_change_prompt(
    tiny_agent, monkeypatch
):
    def forbidden(*args, **kwargs):
        raise AssertionError("a deterministic single candidate must not run a forward")

    with monkeypatch.context() as scoped:
        scoped.setattr(tiny_agent, "_scores", forbidden)
        result = tiny_agent.system_one(
            "state",
            {
                "only": {
                    "type": "choice",
                    "instructions": "Pick",
                    "criteria": {"one": None},
                }
            },
        )
    assert result["answers"]["only"]["probabilities"] == {"one": 1.0}
    question = {
        "type": "noul",
        "instructions": "Escalate?",
        "criteria": {"true": "Urgent", "false": "Can wait"},
    }
    result = tiny_agent.system_one("state", {"q": question})
    expected = original_readout(
        tiny_agent,
        "state",
        "Escalate?\nYes means: Urgent\nNo means: Can wait",
        ["no", "yes"],
    )
    assert result["answers"]["q"]["noul"] == pytest.approx(expected[1], abs=1e-6)


def test_schema_failure_is_atomic_before_any_forward(tiny_agent, monkeypatch):
    calls = []
    monkeypatch.setattr(tiny_agent, "_scores", lambda *args: calls.append(args))
    questions = copy.deepcopy(QUESTIONS)
    questions["later_bad_question"] = {
        "type": "score",
        "instructions": "Rate",
        "criteria": ["Only one"],
    }
    with pytest.raises(ValueError, match="2 to 10"):
        tiny_agent.system_one("state", questions)
    assert not calls


@pytest.mark.parametrize(
    "update",
    [
        {"questions": {}},
        {
            "questions": {
                "q": {"type": "choice", "instructions": "Pick", "criteria": ["a", "b"]}
            }
        },
        {
            "questions": {
                "q": {"type": "score", "instructions": "Rate", "criteria": ["One"]}
            }
        },
        {
            "questions": {
                "q": {
                    "type": "noul",
                    "instructions": "True?",
                    "criteria": {"true": "Yes"},
                }
            }
        },
        {
            "questions": {
                "q": {
                    "type": "choice",
                    "instructions": "Pick",
                    "criteria": {"a": 3, "b": "fine"},
                }
            }
        },
        {"cache_salt": "caller"},
        {"images": ["image.png"]},
        {"model": "unloaded/model"},
        {"state": 42},
    ],
)
def test_tiny_http_rejects_invalid_input_before_forward(
    tiny_agent, monkeypatch, update
):
    from fastapi.testclient import TestClient

    from vllm_jev.windows_qwen import create_app

    calls = []
    monkeypatch.setattr(tiny_agent, "_scores", lambda *args: calls.append(args))
    app = create_app(Path("unused"), MODEL, agent_factory=lambda: tiny_agent)
    with TestClient(app) as client:
        response = client.post(
            "/v1/systemone", json={"state": "state", "questions": QUESTIONS, **update}
        )
        assert response.status_code == 422, response.text
        assert client.get("/health").json()["backend"] == "tiny-jev-pytorch"
    assert not calls


@pytest.mark.parametrize("model_id", ["lostargon/Tiny-Jev", "lostargon/Tiny-Jev-1.7B"])
def test_tiny_load_preserves_rotary_config_source_and_never_executes_remote_code(
    tmp_path, monkeypatch, model_id
):
    import torch
    from safetensors.torch import save_file
    from transformers import Qwen3Config, Qwen3Model

    from vllm_jev import windows_tiny

    config = Qwen3Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        rope_theta=1_000_000,
    )
    backbone = Qwen3Model(config)
    stored_dtype = (
        torch.bfloat16 if model_id == windows_tiny.MODEL_ID_17B else torch.float32
    )
    parameters = {
        f"model.{key}": value.to(dtype=stored_dtype)
        for key, value in backbone.state_dict().items()
    }
    parameters.update(
        {
            # The published 1.7B release mixes a BF16 backbone with an FP32 head.
            "head.weight": torch.ones(1, 16, dtype=torch.float32),
            "head.bias": torch.zeros(1, dtype=torch.float32),
        }
    )
    save_file(parameters, tmp_path / "model.safetensors")
    backbone.load_state_dict(
        {
            key.removeprefix("model."): value
            for key, value in parameters.items()
            if key.startswith("model.")
        }
    )
    backbone.eval()
    backbone.config._attn_implementation = "sdpa"
    original = config.to_dict()
    original.pop("rope_theta", None)
    original.update(
        model_type="tiny_jev",
        architectures=["TinyJevModel"],
        rope_parameters={"rope_type": "default", "rope_theta": 1_000_000},
        dtype="bfloat16",
        temperature=0.95,
        cue=" - correct?",
        opt_token="<opt>",
    )
    (tmp_path / "config.json").write_text(json.dumps(original), encoding="utf-8")
    (tmp_path / "modeling_tiny_jev.py").write_text(
        "raise AssertionError('remote code ran')", encoding="utf-8"
    )
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"extra_special_tokens": ["<opt>"]}), encoding="utf-8"
    )
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    monkeypatch.setattr(windows_tiny, "verify_source", lambda *args: original.copy())

    class Tokenizer(CharacterTokenizer):
        def __init__(self, **kwargs):
            assert kwargs["additional_special_tokens"] == ["<opt>"]
            assert "extra_special_tokens" not in kwargs

        def __len__(self):
            return 64

        def encode(self, text, add_special_tokens=False):
            return [198] if text == "\n" else super().encode(text)

    monkeypatch.setattr("transformers.Qwen2TokenizerFast", Tokenizer)
    source_heads = {}
    if model_id == windows_tiny.MODEL_ID_17B:
        import safetensors

        original_safe_open = safetensors.safe_open

        @contextmanager
        def observe_source_storage(*args, **kwargs):
            with original_safe_open(*args, **kwargs) as source:

                class ObservedSource:
                    def keys(self):
                        return source.keys()

                    def get_tensor(self, name):
                        tensor = source.get_tensor(name)
                        if name.startswith("head."):
                            source_heads[name] = tensor
                        return tensor

                yield ObservedSource()

        monkeypatch.setattr(safetensors, "safe_open", observe_source_storage)
    agent = windows_tiny.load_agent(tmp_path, model_id=model_id, threads=1)
    try:
        for name, tensor in (("head.weight", agent.weight), ("head.bias", agent.bias)):
            torch.testing.assert_close(tensor, parameters[name], rtol=0, atol=0)
            if source_heads:
                # FP32 serving tensors must not keep the source mmap alive.
                assert tensor.data_ptr() != source_heads[name].data_ptr()
        # Transformers 5 stores RoPE in rope_parameters; compare the actual
        # frequencies to the original backbone, independent of config spelling.
        torch.testing.assert_close(
            agent.model.rotary_emb.inv_freq,
            backbone.rotary_emb.inv_freq,
            rtol=0,
            atol=0,
        )
        assert agent.model.config.use_cache is False
        assert agent.model.config._attn_implementation == "sdpa"
        assert all(t.dtype == torch.float32 for t in agent.model.parameters())
        assert agent.model_id == model_id
        assert all(t.device.type == "cpu" for t in agent.model.buffers())
        for key, expected in backbone.state_dict().items():
            torch.testing.assert_close(
                agent.model.state_dict()[key], expected, rtol=0, atol=0
            )
        with torch.inference_mode():
            token_ids = torch.tensor([[1, 2, 3, 4]])
            actual = agent.model(input_ids=token_ids, use_cache=False).last_hidden_state
            expected = backbone(input_ids=token_ids, use_cache=False).last_hidden_state
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert not agent.model.training
        assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before
    finally:
        agent.close()


def test_lightweight_qwen_import_never_loads_backends_or_torch():
    code = """
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'transformers', 'vllm', 'mlx'}:
        raise AssertionError('unexpected backend import: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from vllm_jev import windows_tiny, windows_qwen
assert windows_tiny.MODEL_ID == 'lostargon/Tiny-Jev'
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", code], cwd=ROOT, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("local_source", [False, True])
def test_cli_routes_tiny_to_qwen_without_native_backend(
    legacy_windows_dependencies, monkeypatch, tmp_path, local_source
):
    from vllm_jev import cli, windows_laya, windows_models, windows_qwen

    checkpoint = tmp_path / "中文路径 with spaces"
    checkpoint.mkdir()
    target = str(checkpoint) if local_source else MODEL
    observed = {}
    prepared = []
    monkeypatch.setattr(cli.sys, "platform", "win32")
    monkeypatch.setattr(cli.os, "cpu_count", lambda: 2)
    monkeypatch.setattr(
        cli.sys,
        "argv",
        [
            "vllm-jev",
            "serve",
            target,
            "--workspace",
            str(tmp_path),
            "--protocol",
            "tiny_jev_marker",
        ],
    )
    monkeypatch.setattr(windows_models, "identify_source", lambda path: MODEL)

    def prepare(model, workspace):
        prepared.append((model, workspace))
        return checkpoint

    monkeypatch.setattr(windows_models, "prepare", prepare)
    monkeypatch.setattr(
        windows_qwen,
        "serve",
        lambda path, model, **kwargs: observed.update(path=path, model=model, **kwargs),
    )
    monkeypatch.setattr(
        windows_laya,
        "serve",
        lambda *args, **kwargs: pytest.fail("Tiny routed into Laya"),
    )
    monkeypatch.setattr(
        cli.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("native subprocess launched"),
    )
    cli.main()
    assert observed["path"] == checkpoint
    assert observed["model"] == MODEL
    assert observed["threads"] == 2
    assert prepared == ([] if local_source else [(MODEL, tmp_path)])


@pytest.fixture
def driver():
    spec = importlib.util.spec_from_file_location(
        "check_windows_qwen", ROOT / "scripts/check_windows_qwen.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reference_refuses_unverified_executable_source(driver, tmp_path):
    (tmp_path / "modeling_tiny_jev.py").write_text(
        "raise AssertionError('code executed')", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="unverified"):
        driver.original_tiny_module(tmp_path)


def test_reference_comparison_preserves_labels_probability_and_score(driver):
    expected = {
        "department": {
            "choice": "billing",
            "probabilities": {"billing": 0.7, "technical": 0.3},
            "confidence": 0.7,
        },
        "refund": {"noul": 0.8},
        "urgency": {
            "score": 1.3,
            "probabilities": dict(
                zip(driver.QUESTIONS["urgency"]["criteria"], [0.2, 0.3, 0.5])
            ),
            "confidence": 0.5,
        },
    }
    actual = {
        key: {"type": driver.QUESTIONS[key]["type"], **copy.deepcopy(value)}
        for key, value in expected.items()
    }
    actual["urgency"]["legend"] = {
        level: level for level in driver.QUESTIONS["urgency"]["criteria"]
    }
    assert driver.compare_answers(expected, actual) == 0
    for field, value in [
        ("score", float("nan")),
        ("score", 1.5),
        ("probabilities", {"0": 0.2, "1": 0.3, "2": 0.5}),
    ]:
        changed = copy.deepcopy(actual)
        changed["urgency"][field] = value
        with pytest.raises(AssertionError):
            driver.compare_answers(expected, changed)
    actual["department"]["choice"] = "technical"
    with pytest.raises(AssertionError, match="label differs"):
        driver.compare_answers(expected, actual)


def test_driver_stops_owned_process_before_removing_private_workspace(
    driver, tmp_path, monkeypatch
):
    source = tmp_path / "原始 source"
    source.mkdir()
    (source / "model.safetensors").write_bytes(b"tiny weights")
    (source / "config.json").write_bytes(b"{}")
    hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source.iterdir()
    }
    monkeypatch.setattr(driver, "SOURCE_HASHES", hashes)
    unrelated = tmp_path / "tiny-cli-validation-unrelated"
    unrelated.mkdir()
    (unrelated / "keep").write_text("owned by another run", encoding="utf-8")
    observed = {}

    class Process:
        stopped = False

        def __init__(self, command, **kwargs):
            observed["workspace"] = Path(command[command.index("--workspace") + 1])
            assert Path(command[5]).is_dir(), (
                "local-source CLI route must use its private clone"
            )

        def poll(self):
            return 0 if self.stopped else None

        def terminate(self):
            assert observed["workspace"].exists()
            observed["terminated"] = True

        def wait(self, timeout):
            self.stopped = True

    monkeypatch.setattr(driver.subprocess, "Popen", Process)

    def fail_http(*args, **kwargs):
        raise RuntimeError("synthetic startup error")

    monkeypatch.setattr(driver, "http", fail_http)
    args = SimpleNamespace(
        workspace=tmp_path,
        source=source,
        threads=1,
        port=18796,
        output=tmp_path / "result.json",
        startup_timeout=1,
    )
    report = {"service_runs": []}
    with pytest.raises(RuntimeError, match="synthetic startup"):
        driver.service_run(
            args, {"source": str(source), "source_sha256": hashes}, {}, report, 1
        )
    check = report["service_runs"][0]
    assert observed["terminated"]
    assert check["owned_process_stopped"] and check["private_workspace_removed"]
    assert check["source_unchanged"]
    assert not observed["workspace"].exists()
    assert (unrelated / "keep").read_text(encoding="utf-8") == "owned by another run"
    assert driver.verify_source(source) == hashes
