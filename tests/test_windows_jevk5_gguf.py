import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm_jev import windows_jevk5_gguf as adapter
from vllm_jev.windows_models import protocol_for

QUESTIONS = {
    "route": {
        "type": "choice",
        "instructions": "Choose a department.",
        "criteria": {"bill": "Billing", "tech": "Technical"},
    },
    "refund": {"type": "noul", "instructions": "Does this need a refund?"},
    "urgency": {
        "type": "score",
        "instructions": "How urgent?",
        "criteria": ["Low", "High"],
    },
}


class Server:
    threads = 4

    def __init__(self, *, oversized=False):
        self.calls = []
        self.oversized = oversized
        self.closed = False

    def post(self, path, body):
        self.calls.append((path, body))
        if path == "/tokenize":
            return {
                "tokens": [10]
                * (1025 if self.oversized and len(self.calls) == 2 else 5)
            }
        return {
            "tokens_evaluated": 5,
            "truncated": False,
            "completion_probabilities": [
                {
                    "top_logprobs": [
                        {"token": "A", "logprob": -1.0},
                        {"token": "B", "logprob": -2.0},
                    ]
                }
            ],
        }

    def close(self):
        self.closed = True


def test_calibration_preserves_missing_letter_floor():
    out = {
        "completion_probabilities": [
            {
                "top_logprobs": [
                    {"token": "A", "logprob": -1},
                    {"token": "other", "logprob": -4},
                ]
            }
        ]
    }
    probabilities, missing = adapter.read_probabilities(out, 3)
    weights = [1, math.exp(-5 / 1.22), math.exp(-5 / 1.22)]
    assert probabilities == pytest.approx(
        [w / sum(weights) for w in weights], abs=1e-15
    )
    assert missing == 2


def test_late_overlength_question_is_rejected_before_any_inference():
    server = Server(oversized=True)
    with pytest.raises(ValueError, match="1024"):
        adapter.JevK5GGUFAgent(server).system_one("Test", QUESTIONS)
    assert all(path == "/tokenize" for path, body in server.calls)


def test_state_order_native_prompt_and_type_mapping():
    server = Server()
    agent = adapter.JevK5GGUFAgent(server)
    result = agent.system_one({"message": "退款 🙏"}, QUESTIONS)
    prompts = [body["content"] for path, body in server.calls if path == "/tokenize"]
    assert all(
        prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
        for prompt in prompts
    )
    assert '"evidence": {"message": "退款 🙏"}' in prompts[0]
    assert prompts[1].index("true: The proposition is true.") < prompts[1].index(
        "false: The proposition is false."
    )
    answers = result["answers"]
    assert answers["route"]["choice"] == "bill"
    assert answers["refund"]["noul"] == answers["route"]["probabilities"]["bill"]
    assert answers["urgency"]["legend"] == {"0": "Low", "1": "High"}
    assert result["metadata"]["quantization"] == "Q5_K_M"
    assert result["metadata"]["runtime_generated_tokens"] == 3
    agent.close()
    agent.close()
    assert server.closed
    with pytest.raises(RuntimeError, match="closed"):
        agent.system_one("test", QUESTIONS)


@pytest.mark.parametrize(
    "change",
    [
        {"type": "choice", "criteria": [str(i) for i in range(17)]},
        {"type": "choice", "criteria": ["repeat", "repeat"]},
        {"type": "score", "criteria": ["low", "high"], "isolated": True},
        {"type": "noul", "criteria": {"invalid": "bad"}},
    ],
)
def test_unsupported_requests_fail_before_runtime(change):
    server = Server()
    question = {"instructions": "Decide", **change}
    with pytest.raises(ValueError):
        adapter.JevK5GGUFAgent(server).system_one("state", {"q": question})
    assert not server.calls


def test_source_rejects_unexpected_native_libraries_before_hashing(tmp_path):
    (tmp_path / "runtime").mkdir()
    (tmp_path / "runtime/foreign.dll").write_bytes(b"foreign")
    with pytest.raises(ValueError, match="unexpected"):
        adapter.verify_source(tmp_path)


def test_native_only_and_model_dispatch():
    assert protocol_for(adapter.MODEL_ID) == adapter.PROTOCOL
    adapter.validate_template(SimpleNamespace(config={"layout": "native"}))
    with pytest.raises(ValueError, match="native"):
        adapter.validate_template(SimpleNamespace(config={"instructions_first": True}))


def test_request_candidate_budget():
    question = copy.deepcopy(QUESTIONS["route"])
    question["criteria"] = [str(i) for i in range(16)]
    with pytest.raises(ValueError, match="256"):
        adapter.rows_for("state", {str(i): question for i in range(17)})


@pytest.mark.parametrize(
    "model,temperature", [(adapter.MODEL_ID, 1.22), (adapter.MODEL_ID_9B, 1.316)]
)
def test_family_calibration_identity_and_all_typed_answers(model, temperature):
    server = Server()
    result = adapter.JevK5GGUFAgent(server, model).system_one(
        "Refund requested.", QUESTIONS
    )
    expected = 1.0 / (1.0 + math.exp(-1.0 / temperature))
    assert result["model"] == result["metadata"]["model"] == model
    assert result["metadata"]["temperature"] == temperature
    assert result["metadata"]["artifact_filename"] == adapter.MODELS[model]["filename"]
    assert result["answers"]["route"]["probabilities"]["bill"] == pytest.approx(
        expected, abs=1e-15
    )
    assert result["answers"]["refund"]["noul"] == pytest.approx(expected, abs=1e-15)
    assert result["answers"]["urgency"]["score"] == pytest.approx(
        1.0 - expected, abs=1e-15
    )
    assert all(
        body["n_probs"] == 40 and body["cache_prompt"] is False
        for path, body in server.calls
        if path == "/completion"
    )


def test_nine_b_missing_letters_use_its_own_temperature():
    output = {
        "completion_probabilities": [
            {
                "top_logprobs": [
                    {"token": "A", "logprob": -1.0},
                    {"token": "Z", "logprob": -4.0},
                ]
            }
        ]
    }
    probabilities, missing = adapter.read_probabilities(output, 16, adapter.MODEL_ID_9B)
    denominator = 1 + 15 * math.exp(-5.0 / 1.316)
    assert missing == 15
    assert probabilities == pytest.approx(
        [1 / denominator] + [math.exp(-5.0 / 1.316) / denominator] * 15, abs=1e-15
    )


@pytest.mark.parametrize("model", tuple(adapter.MODELS))
def test_source_family_identity_cannot_mix_models_or_ignore_tampering(
    tmp_path, monkeypatch, model
):
    filename = adapter.MODELS[model]["filename"]
    content = b"small unit test model bytes"
    digest = hashlib.sha256(content).hexdigest()
    monkeypatch.setitem(
        adapter.MODELS,
        model,
        {
            **adapter.MODELS[model],
            "files": {filename: digest},
        },
    )
    monkeypatch.setattr(adapter, "RUNTIME_HASHES", {"llama-server.exe": digest})
    (tmp_path / "runtime").mkdir()
    (tmp_path / "runtime/llama-server.exe").write_bytes(content)
    weight = tmp_path / filename
    weight.write_bytes(content)
    assert adapter.identify_source(tmp_path) == model
    other = next(item for item in adapter.MODELS if item != model)
    with pytest.raises(ValueError, match="unexpected"):
        adapter.verify_source(tmp_path, other)
    other_weight = tmp_path / adapter.MODELS[other]["filename"]
    other_weight.write_bytes(content)
    with pytest.raises(ValueError, match="exactly one"):
        adapter.identify_source(tmp_path)
    other_weight.unlink()
    weight.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        adapter.identify_source(tmp_path)


@pytest.mark.parametrize("model", tuple(adapter.MODELS))
def test_loader_verifies_selected_release_before_starting_its_weight(
    tmp_path, monkeypatch, model
):
    from vllm_jev import windows_llama_server

    calls = []
    monkeypatch.setattr(
        adapter,
        "verify_source",
        lambda path, identity: calls.append(("verify", path, identity)),
    )
    monkeypatch.setattr(
        windows_llama_server,
        "CPUServer",
        lambda executable, weight, threads: (
            calls.append(("server", executable, weight, threads)) or Server()
        ),
    )
    agent = adapter.load_agent(tmp_path, model, threads=2)
    assert agent.model_id == model
    assert calls == [
        ("verify", tmp_path, model),
        (
            "server",
            tmp_path / "runtime/llama-server.exe",
            tmp_path / adapter.MODELS[model]["filename"],
            2,
        ),
    ]
    agent.close()


@pytest.fixture
def driver(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        "jevk5_family_driver_test", scripts / "check_windows_jevk5_gguf.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reference_pins_are_independent_and_match_fixed_releases(driver):
    reference = driver.reference
    assert adapter.source_hashes() == adapter.SOURCE_HASHES
    assert reference.source_hashes() == reference.SOURCE_HASHES
    for model in adapter.MODELS:
        assert adapter.source_hashes(model) == reference.source_hashes(model)
    text = Path(reference.__file__).read_text(encoding="utf-8")
    assert "from vllm_jev" not in text and "import vllm_jev" not in text
    filename = "jevk5-9b-v0.3.3-Q5_K_M.gguf"
    assert (
        reference.source_hashes(adapter.MODEL_ID_9B)[filename]
        == "da4a4becd04c3beabd4cefbd09b546ce2648740d923adc8d14ae1d0cb921486e"
    )
    assert adapter.GGUF_FILENAME not in reference.source_hashes(adapter.MODEL_ID_9B)


@pytest.mark.parametrize(
    "model,available,commit",
    [(adapter.MODEL_ID, 6.0, 8.0), (adapter.MODEL_ID_9B, 10.0, 12.0)],
)
def test_acceptance_uses_model_specific_memory_gates(driver, model, available, commit):
    args = SimpleNamespace(
        model=model,
        min_available_gib=None,
        min_commit_available_gib=None,
        abort_below_available_gib=1.5,
    )
    driver.apply_memory_policy(args)
    assert (args.min_available_gib, args.min_commit_available_gib) == (
        available,
        commit,
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("min_available_gib", 9.9),
        ("min_commit_available_gib", 11.9),
        ("abort_below_available_gib", 1.49),
    ],
)
def test_nine_b_memory_policy_cannot_be_weakened(driver, field, value):
    args = SimpleNamespace(
        model=adapter.MODEL_ID_9B,
        min_available_gib=10.0,
        min_commit_available_gib=12.0,
        abort_below_available_gib=1.5,
    )
    setattr(args, field, value)
    with pytest.raises(ValueError, match="9B requires"):
        driver.apply_memory_policy(args)


def test_nine_b_acceptance_rejects_four_b_metadata_and_outputs(driver):
    result = adapter.JevK5GGUFAgent(Server(), adapter.MODEL_ID_9B).system_one(
        "test", QUESTIONS
    )
    # This fake response represents a fresh GGUF CLI process. Other tests may
    # have imported Torch into pytest; that parent state is not the fake CLI's.
    # The real acceptance driver must still require torch_imported=False.
    result["metadata"]["torch_imported"] = False
    expected = copy.deepcopy(result["answers"])
    driver.check_response(200, result, expected, 4, adapter.MODEL_ID_9B)
    for field, value in [
        ("temperature", 1.22),
        ("artifact_filename", adapter.GGUF_FILENAME),
        ("model", adapter.MODEL_ID),
    ]:
        changed = copy.deepcopy(result)
        changed["metadata"][field] = value
        with pytest.raises(ValueError, match="identity differs"):
            driver.check_response(200, changed, expected, 4, adapter.MODEL_ID_9B)
    result["model"] = adapter.MODEL_ID
    with pytest.raises(ValueError, match="response model"):
        driver.check_response(200, result, expected, 4, adapter.MODEL_ID_9B)


def test_nine_b_rejects_reference_report_from_another_temperature(driver, tmp_path):
    reference = driver.reference
    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps(driver.CASES), encoding="utf-8")
    args = SimpleNamespace(
        model=adapter.MODEL_ID_9B, threads=4, reference_python=tmp_path / "python.exe"
    )
    report = {
        "status": "passed",
        "model": args.model,
        "revision": reference.REVISION,
        "sdk_commit": reference.SDK_COMMIT,
        "source_hashes": reference.source_hashes(args.model),
        "sdk_hashes": reference.SDK_HASHES,
        "runner_sha256": driver.common.digest(driver.RUNNER),
        "cases_sha256": driver.common.digest(cases),
        "threads": 4,
        "temperature": 1.22,
        "knockout_temperature": 1.05,
        "source_unchanged": True,
        "sdk_unchanged": True,
        "server_stopped": True,
        "product_imported": False,
        "torch_imported": False,
    }
    path = tmp_path / "author.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="temperature"):
        driver.validate_reference(path, args, cases)
