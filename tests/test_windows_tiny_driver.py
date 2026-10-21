"""Acceptance evidence must reject drift, malformed vectors, and wrong identity."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "_tiny_driver_test", ROOT / "scripts/check_windows_tiny.py"
)
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


@pytest.fixture
def evidence(tmp_path):
    inputs = driver.cases()
    path = tmp_path / "author.json"
    cases_path = tmp_path / "cases.json"
    driver.common.save(cases_path, inputs)
    reference = driver.reference
    value = {
        "status": "passed",
        "model": driver.MODEL,
        "revision": reference.REVISION,
        "runner_sha256": reference.digest(reference.__file__),
        "cases_sha256": reference.digest(cases_path),
        "source_hashes": reference.SOURCE_HASHES,
        "source_unchanged": True,
        "versions": reference.VERSIONS,
        "runtime": {
            "device": "cpu",
            "dtype": "float32",
            "attention": "sdpa",
            "use_cache": False,
            "threads": 4,
            "row_batch_size": 1,
            "temperature": 1.15,
            "max_sequence_tokens": 4096,
        },
        "cases": [],
    }
    for case in inputs:
        observed = {"id": case["id"], "rows": [], "answers": {}}
        for identifier, question in case["questions"].items():
            count = 2 if question["type"] == "noul" else len(question["criteria"])
            p = [1 / count] * count
            observed["rows"].append(
                {
                    "id": identifier,
                    "definition": reference.author_definition(question),
                    "input_ids": list(range(count * 2)),
                    "positions": list(range(count)),
                    "calibrated_logits": [0.0] * count,
                    "probabilities": p,
                }
            )
            observed["answers"][identifier] = driver.canonical(question, p)
        value["cases"].append(observed)
    driver.common.save(path, value)
    return path, cases_path, inputs, value


def test_full_reference_evidence_accepts_consistent_fixture(evidence):
    path, cases_path, inputs, value = evidence
    assert driver.validate_reference(path, cases_path, inputs, 4) == value


@pytest.mark.parametrize(
    "mutation",
    [
        lambda v: v.update(revision="wrong"),
        lambda v: v.update(source_unchanged=False),
        lambda v: v["runtime"].update(dtype="bfloat16"),
        lambda v: v["runtime"].update(temperature=0.95),
        lambda v: v["cases"][0]["rows"][0].update(probabilities=[0.1, 0.9]),
        lambda v: v["cases"][0]["rows"][0].update(positions=[1, 1]),
        lambda v: v["cases"][0]["rows"][0].update(input_ids=[True, 1, 2]),
        lambda v: v["cases"][0]["rows"][0]["definition"].update(
            instructions="different"
        ),
        lambda v: v["cases"][0]["answers"]["department"].update(confidence=0.9),
    ],
)
def test_modified_reference_evidence_rejected(evidence, mutation):
    path, cases_path, inputs, value = evidence
    mutation(value)
    driver.common.save(path, value)
    with pytest.raises((ValueError, AssertionError)):
        driver.validate_reference(path, cases_path, inputs, 4)


def test_nonfinite_and_duplicate_json_evidence_rejected(evidence):
    path, cases_path, inputs, value = evidence
    path.write_text('{"status": "passed", "status": "failed"}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        driver.validate_reference(path, cases_path, inputs, 4)
    value["cases"][0]["rows"][0]["probabilities"][0] = float("nan")
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError):
        driver.validate_reference(path, cases_path, inputs, 4)


def test_response_checks_full_probabilities_identity_and_token_counts(evidence):
    _, _, inputs, value = evidence
    original = value["cases"][0]
    response = {
        "model": driver.MODEL,
        "answers": copy.deepcopy(original["answers"]),
        "usage": {
            "input_tokens": sum(len(row["input_ids"]) for row in original["rows"]),
            "output_tokens": 0,
        },
        "metadata": {
            "model": driver.MODEL,
            "revision": driver.reference.REVISION,
            "backend": "tiny-jev-pytorch",
            "device": "cpu",
            "experimental": True,
            "method": "marker_token_head",
            "temperature": 1.15,
            "cached_tokens": 0,
            "candidate_sequences": sum(
                len(row["probabilities"]) for row in original["rows"]
            ),
        },
    }
    assert (
        driver.check_response(200, response, original["answers"], original, inputs[0])
        == 0
    )
    for mutation in (
        lambda v: v["metadata"].update(temperature=0.95),
        lambda v: v["usage"].update(output_tokens=1),
        lambda v: v["answers"]["department"]["probabilities"].update(billing=0.9),
    ):
        changed = copy.deepcopy(response)
        mutation(changed)
        with pytest.raises(AssertionError):
            driver.check_response(
                200, changed, original["answers"], original, inputs[0]
            )


def test_memory_guard_precedes_process_launch(monkeypatch, tmp_path):
    from types import SimpleNamespace

    monkeypatch.setattr(driver.common, "memory_gate", lambda args: {"passed": False})
    monkeypatch.setattr(
        driver.helpers.subprocess,
        "Popen",
        lambda *args, **kwargs: pytest.fail("model started before memory gate"),
    )
    with pytest.raises(RuntimeError, match="insufficient memory"):
        driver.helpers.child_run(
            SimpleNamespace(), [], {}, tmp_path / "unused.log", {}, lambda *args: None
        )


def test_main_records_preflight_failure_without_starting_a_model(monkeypatch, tmp_path):
    from types import SimpleNamespace

    args = SimpleNamespace(
        source=tmp_path / "source",
        workspace=tmp_path / "work",
        output=tmp_path / "result.json",
        reference_python=Path(driver.sys.executable),
        console_entrypoint=tmp_path / "vllm-jev.exe",
        reference_report=None,
        reference_only=False,
        threads=1,
        port=0,
        timeout=30,
        min_available_gib=10,
        min_commit_available_gib=20,
        abort_below_available_gib=1.5,
    )
    monkeypatch.setattr(driver, "parse_args", lambda: args)
    monkeypatch.setattr(
        driver.reference, "verify", lambda path: driver.reference.SOURCE_HASHES
    )

    def stop_before_launch(*args, **kwargs):
        raise RuntimeError("insufficient memory: preflight sentinel")

    monkeypatch.setattr(driver.helpers, "reference_run", stop_before_launch)
    with pytest.raises(RuntimeError, match="preflight sentinel"):
        driver.main()
    report = json.loads(args.output.read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert report["measured"] is False
    assert report["service_runs"] == []
    assert report["driver_sha256"] == driver.reference.digest(driver.__file__)
    assert report["memory_policy"]["min_commit_available_gib"] == 20


@pytest.mark.parametrize(
    "extra",
    [
        ["--reference-only", "--reference-report", "saved.json"],
        ["--min-available-gib", "1.5", "--abort-below-available-gib", "1.5"],
    ],
)
def test_conflicting_validation_modes_and_memory_thresholds_rejected(
    monkeypatch, tmp_path, extra
):
    arguments = ["check_windows_tiny.py"]
    for option in (
        "source",
        "reference-python",
        "workspace",
        "output",
        "console-entrypoint",
    ):
        arguments.extend(["--" + option, str(tmp_path / option)])
    monkeypatch.setattr(driver.sys, "platform", "win32")
    monkeypatch.setattr(driver.sys, "argv", arguments + extra)
    with pytest.raises(SystemExit) as error:
        driver.parse_args()
    assert error.value.code == 2
