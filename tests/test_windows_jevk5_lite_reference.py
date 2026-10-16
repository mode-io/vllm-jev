"""Independent reference provenance and no-truncation gates, without weights."""

import copy
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "independent_jevk5_lite_reference",
    Path(__file__).resolve().parents[1] / "scripts/windows_jevk5_lite_reference.py",
)
reference = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = reference
SPEC.loader.exec_module(reference)

CASE = {
    "id": "中文 case",
    "state": {"ticket": "Please refund the duplicate charge."},
    "questions": {
        "intent": {
            "type": "choice",
            "instructions": "Route this ticket",
            "criteria": {"billing": "", "refund": "Money back", "other": "Money back"},
        },
        "refund": {"type": "noul", "instructions": "Was a refund requested?"},
        "urgency": {
            "type": "score",
            "instructions": "How urgent?",
            "criteria": ["low", "high"],
        },
    },
}


def write_cases(tmp_path, cases):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(cases, ensure_ascii=False), encoding="utf-8")
    return path


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    source, sdk = tmp_path / "source", tmp_path / "sdk"
    source.mkdir()
    sdk.mkdir()
    sources, authors = {}, {}
    for name in reference.SOURCE_HASHES:
        path = source / name
        path.write_bytes(("fixed-checkpoint:" + name).encode())
        sources[name] = reference.sha256(path)
    for name in reference.SDK_HASHES:
        path = sdk / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(("# fixed-author:" + name).encode())
        authors[name] = reference.sha256(path)
    monkeypatch.setattr(reference, "SOURCE_HASHES", sources)
    monkeypatch.setattr(reference, "SDK_HASHES", authors)
    return source, sdk


def test_fixed_release_and_source_pins():
    assert reference.MODEL_ID == "alibiserikbay/JevK5-Lite"
    assert reference.REVISION == "315ee211f828a899161477c534cea23e84ba3568"
    assert reference.SDK_COMMIT == "f26426d16f59e8bbe1470e5b162cc89329e29b29"
    assert len(reference.SOURCE_HASHES) == 7
    assert set(reference.SDK_HASHES) == {
        "jevk5/__init__.py",
        "jevk5/prompt.py",
        "jevk5/lite.py",
        "bench/jevk5_lite.py",
    }
    for digest in [*reference.SOURCE_HASHES.values(), *reference.SDK_HASHES.values()]:
        assert len(digest) == 64
        assert len(bytes.fromhex(digest)) == 32


@pytest.mark.parametrize(
    "name", ["model.safetensors", "scorer.safetensors", "tokenizer.json"]
)
def test_checkpoint_tampering_is_rejected(inputs, name):
    source, _ = inputs
    (source / name).write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA256"):
        reference.verify_source(source)


@pytest.mark.parametrize(
    "name", ["adapter_config.json", "chat_template.jinja", "inference.py", ".cache"]
)
def test_extra_checkpoint_loader_file_is_rejected(inputs, name):
    source, _ = inputs
    (source / name).write_text("untrusted", encoding="utf-8")
    with pytest.raises(ValueError, match="Unexpected"):
        reference.verify_source(source)


def test_missing_scorer_fails_before_loading(inputs):
    source, _ = inputs
    (source / "scorer.safetensors").unlink()
    with pytest.raises(ValueError, match="Missing"):
        reference.verify_source(source)


def test_private_copy_contains_only_pinned_bytes(inputs, tmp_path):
    source, sdk = inputs
    (source / ".cache").mkdir()
    (source / ".cache" / "unused.json").write_text("cache", encoding="utf-8")
    (sdk / "jevk5" / "runtime.py").write_text(
        "raise AssertionError('unused')", encoding="utf-8"
    )
    checkpoint, private_sdk = reference.copy_inputs(source, sdk, tmp_path / "private")
    assert reference.verify_source(checkpoint) == reference.SOURCE_HASHES
    assert reference.verify_sdk(private_sdk, isolated=True) == reference.SDK_HASHES
    assert not (checkpoint / ".cache").exists()
    assert not (private_sdk / "jevk5/runtime.py").exists()
    assert (private_sdk / "bench/jevk5_lite.py").read_bytes() == (
        sdk / "bench/jevk5_lite.py"
    ).read_bytes()
    # Configuration mutations in the private loader cannot rewrite the source.
    (checkpoint / "tokenizer.json").write_text("changed", encoding="utf-8")
    assert reference.verify_source(source) == reference.SOURCE_HASHES
    with pytest.raises(ValueError, match="SHA256"):
        reference.verify_source(checkpoint)


def test_private_sdk_added_module_fails_unchanged_check(inputs, tmp_path):
    _, sdk = inputs
    assert reference.verify_sdk(sdk, isolated=True) == reference.SDK_HASHES
    (sdk / "jevk5" / "extra.py").write_text("# new", encoding="utf-8")
    with pytest.raises(ValueError, match="Unexpected private"):
        reference.verify_sdk(sdk, isolated=True)


def test_original_bench_tampering_fails_before_import(inputs):
    _, sdk = inputs
    (sdk / "bench/jevk5_lite.py").write_text(
        "raise AssertionError('never')", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="SHA256"):
        reference.load_bench(sdk)


def test_cases_preserve_chinese_null_and_duplicate_descriptions(tmp_path):
    case = copy.deepcopy(CASE)
    case["questions"]["intent"]["criteria"]["billing"] = None
    path = write_cases(tmp_path, [case])
    assert reference.read_cases(path) == [case]


@pytest.mark.parametrize(
    "text",
    [
        '[{"id":"x","id":"y","state":"s","questions":{}}]',
        '[{"id":"x","state":{"n":NaN},"questions":{}}]',
        '[{"id":"x","state":{"n":Infinity},"questions":{}}]',
        '[{"id":"x","state":{"n":1e999},"questions":{}}]',
    ],
)
def test_duplicate_and_nonfinite_json_rejected(tmp_path, text):
    path = tmp_path / "bad.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        reference.read_cases(path)


@pytest.mark.parametrize(
    "question",
    [
        {"type": "noul", "instructions": "q", "criteria": {"true": "custom"}},
        {"type": "noul", "instructions": "q", "criteria": []},
        {"type": "choice", "instructions": "q", "criteria": ["same", "same"]},
        {"type": "choice", "instructions": "q", "criteria": ["only"]},
        {"type": "choice", "instructions": "q", "criteria": {"a": 1, "b": "b"}},
        {"type": "choice", "instructions": "q", "criteria": {"": "a", "b": "b"}},
        {
            "type": "score",
            "instructions": "q",
            "criteria": ["low", "high"],
            "isolated": True,
        },
        {"type": "score", "instructions": "q", "criteria": [1, 2]},
        {"type": "noul", "instructions": " "},
        {"type": "other", "instructions": "q"},
    ],
)
def test_unsupported_question_rejected_before_author(tmp_path, question):
    with pytest.raises(ValueError):
        reference.read_cases(
            write_cases(tmp_path, [{**CASE, "questions": {"q": question}}])
        )


class FakeTokenizer:
    def __call__(self, text, **kwargs):
        return {"input_ids": [7] * len(text.split())}


class PreflightAuthor:
    tok = FakeTokenizer()

    def __init__(self, count, dropped=0):
        self.count, self.dropped = count, dropped

    def encode(self, text, heads):
        task, labels, _, _ = heads[0]
        body = len(self.tok(text)["input_ids"]) - self.dropped
        return [1] * self.count, [2, 4], [(0, len(labels))], (8, 8 + body)


def fake_bench():
    return types.SimpleNamespace(
        lite_head=lambda q: (q["instructions"], ["true", "false"], ["true", "false"]),
        state_text=lambda state: state,
    )


def test_no_truncation_and_512_boundary():
    case = {
        **CASE,
        "state": "full state",
        "questions": {"q": CASE["questions"]["refund"]},
    }
    rows = reference.preflight_cases([case], fake_bench(), PreflightAuthor(512))
    assert len(rows[case["id"]]["q"]["input_ids"]) == 512
    for count, dropped in [(513, 0), (500, 1), (0, 0)]:
        with pytest.raises(ValueError, match="complete input"):
            reference.preflight_cases(
                [case], fake_bench(), PreflightAuthor(count, dropped)
            )


def good_result():
    row = {
        "option_ids": ["true", "false"],
        "labels": ["true", "false"],
        "input_ids": [1, 2],
    }
    result = reference.DecisionResult(
        adapter="jevk5_lite",
        ok=True,
        probs={"yes": 0.7, "no": 0.3},
        raw={
            "probabilities": {"true": 0.7, "false": 0.3},
            "labels": {"true": "true", "false": "false"},
            "truncated_state_tokens": 0,
            "temperature_single": reference.TEMPERATURE_SINGLE,
        },
        usage={"input_tokens": 2},
    )
    return result, row


def test_noul_preserves_author_true_first_order():
    result, row = good_result()
    assert reference.result_probabilities(result, {"type": "noul"}, row) == [0.7, 0.3]


@pytest.mark.parametrize(
    "change", ["failed", "swapped", "truncated", "temperature", "nan", "sum", "missing"]
)
def test_invalid_author_evidence_cannot_be_success(change):
    result, row = good_result()
    if change == "failed":
        result.ok, result.error = False, "load failed"
    elif change == "swapped":
        result.probs = {"yes": 0.3, "no": 0.7}
    elif change == "truncated":
        result.raw["truncated_state_tokens"] = 1
    elif change == "temperature":
        result.raw["temperature_single"] = 1.0
    elif change == "nan":
        result.raw["probabilities"]["true"] = float("nan")
    elif change == "sum":
        result.raw["probabilities"]["true"] = 0.8
    else:
        del result.raw["probabilities"]["false"]
    with pytest.raises(RuntimeError):
        reference.result_probabilities(result, {"type": "noul"}, row)


def main_args(inputs, tmp_path, monkeypatch, *, check=False):
    source, sdk = inputs
    cases = write_cases(tmp_path, [CASE])
    output = tmp_path / "report.json"
    monkeypatch.setattr(
        reference.sys,
        "argv",
        [
            "reference.py",
            "--source",
            str(source),
            "--sdk-source",
            str(sdk),
            "--cases-json",
            str(cases),
            "--output",
            str(output),
            "--threads",
            "1",
            *(["--check-only"] if check else []),
        ],
    )
    monkeypatch.setattr(reference, "versions", lambda: {"torch": "fixture"})
    monkeypatch.chdir(tmp_path)
    return output


def test_check_only_never_loads_and_never_claims_measured(
    inputs, tmp_path, monkeypatch
):
    output = main_args(inputs, tmp_path, monkeypatch, check=True)
    monkeypatch.setattr(reference, "infer", lambda *a: pytest.fail("no model may load"))
    reference.main()
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "source_checked"
    assert report["measured"] is False
    assert report["cases"] == []
    assert report["source_hashes"] == reference.SOURCE_HASHES
    assert report["sdk_hashes"] == reference.SDK_HASHES
    assert report["environment"] == {
        "python": sys.version,
        "executable": sys.executable,
    }
    assert report["driver_sha256"] == reference.sha256(reference.__file__)


@pytest.mark.parametrize(
    "failure", ["partial", "unmeasured", "changed_private", "changed_original"]
)
def test_failed_or_changed_reference_cannot_report_passed(
    inputs, tmp_path, monkeypatch, failure
):
    output = main_args(inputs, tmp_path, monkeypatch)

    def simulate(args, report, cases, sdk, checkpoint):
        report["cases"] = [] if failure == "partial" else list(cases)
        report["measured"] = failure != "unmeasured"
        if failure == "changed_private":
            (sdk / "bench/jevk5_lite.py").write_text("changed", encoding="utf-8")
        if failure == "changed_original":
            (args.source / "config.json").write_text("changed", encoding="utf-8")

    monkeypatch.setattr(reference, "infer", simulate)
    with pytest.raises((ValueError, RuntimeError)):
        reference.main()
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "failed"
    assert not list(tmp_path.glob("jevk5-lite-reference-*"))


def test_success_requires_all_hash_readbacks(inputs, tmp_path, monkeypatch):
    output = main_args(inputs, tmp_path, monkeypatch)

    def simulate(args, report, cases, sdk, checkpoint):
        assert sdk != args.sdk_source and checkpoint != args.source
        report.update(measured=True, cases=list(cases))

    monkeypatch.setattr(reference, "infer", simulate)
    reference.main()
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "passed"
    assert all(
        report[key] is True
        for key in (
            "source_unchanged",
            "sdk_unchanged",
            "private_source_unchanged",
            "private_sdk_unchanged",
        )
    )
    assert not list(tmp_path.glob("jevk5-lite-reference-*"))


def test_product_imports_are_blocked():
    with pytest.raises(RuntimeError, match="cannot import product"):
        reference.BlockProductImports().find_spec("vllm_jev.windows_jevk5_lite")
    assert reference.BlockProductImports().find_spec("json") is None


def test_original_pinned_bench_mapping_without_model(tmp_path, monkeypatch):
    """Optional real-author contract probe; all four executed bytes are verified."""
    sdk_path = os.environ.get("JEVK5_LITE_SDK_SOURCE")
    if not sdk_path:
        pytest.skip("set JEVK5_LITE_SDK_SOURCE to a pinned author checkout")
    source = Path(sdk_path).resolve()
    reference.verify_sdk(source)
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    prefix = "_jevk5_lite_author_bench"
    for key in list(sys.modules):
        if key == prefix or key.startswith(prefix + "."):
            monkeypatch.delitem(sys.modules, key)
    try:
        bench = reference.load_bench(source)
        assert bench.state_text({"中文": "测试"}) == '{"中文": "测试"}'
        assert bench.lite_head(CASE["questions"]["intent"])[1:] == (
            ["billing: billing", "refund: Money back", "other: Money back"],
            ["billing", "refund", "other"],
        )
        assert bench.lite_head(CASE["questions"]["refund"])[1:] == (
            ["true", "false"],
            ["true", "false"],
        )
        assert bench.lite_head(CASE["questions"]["urgency"])[1:] == (
            ["low", "high"],
            ["0", "1"],
        )

        # Execute the original adapter.run and its output bookkeeping too. Only
        # the heavyweight model is a deterministic test double in this test.
        class FakeLite:
            temperature_single = reference.TEMPERATURE_SINGLE
            tok = FakeTokenizer()

            def classify(self, text, tasks):
                return {
                    name: {
                        "labels": [labels[-1]],
                        "probabilities": dict(
                            zip(
                                labels,
                                [0.1, 0.2, 0.7] if len(labels) == 3 else [0.2, 0.8],
                            )
                        ),
                    }
                    for name, labels in tasks.items()
                }

            def encode(self, text, heads):
                size = len(heads[0][1])
                body = len(self.tok(text)["input_ids"])
                return [1] * 12, list(range(size)), [(0, size)], (8, 8 + body)

        adapter = bench.JevK5LiteAdapter(endpoint="unused", model=reference.MODEL_ID)
        adapter._loaded = FakeLite()
        rows = reference.preflight_cases([CASE], bench, adapter._loaded)[CASE["id"]]
        for identifier, question in CASE["questions"].items():
            result = adapter.run(
                types.SimpleNamespace(state=CASE["state"], question=question)
            )
            expected = [0.1, 0.2, 0.7] if identifier == "intent" else [0.2, 0.8]
            assert (
                reference.result_probabilities(result, question, rows[identifier])
                == expected
            )
    finally:
        for key in list(sys.modules):
            if key == prefix or key.startswith(prefix + "."):
                sys.modules.pop(key)
