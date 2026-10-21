"""Acceptance-driver guards, using synthetic data and no model inference."""

import copy
import hashlib
import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest


@pytest.fixture
def driver():
    path = Path(__file__).resolve().parents[1] / "scripts/check_windows_decider.py"
    spec = importlib.util.spec_from_file_location("decider_driver_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def source(driver, tmp_path, monkeypatch):
    source = tmp_path / "中文 source"
    source.mkdir()
    hashes = {}
    for name in driver.SOURCE_HASHES:
        content = f"fixture {name}\n".encode()
        (source / name).write_bytes(content)
        hashes[name] = hashlib.sha256(content).hexdigest()
    monkeypatch.setattr(driver, "SOURCE_HASHES", hashes)
    return source


def author_case(request):
    case = copy.deepcopy(request)
    rows, index, answers, counts = [], [], {}, []
    for identifier, question in case["questions"].items():
        if question["type"] == "score":
            index.append([identifier, "iso", len(rows), len(question["criteria"])])
            rows += [[0.8, 0.2], [0.4, 0.6], [0.8, 0.2]]
            counts += [2, 2, 2]
            answers[identifier] = {
                "type": "score",
                "score": 1.0,
                "confidence": 0.6,
                "legend": {str(i): text for i, text in enumerate(question["criteria"])},
                "probabilities": {"0": 0.2, "1": 0.6, "2": 0.2},
            }
        else:
            index.append([identifier, "list", len(rows), 1])
            count = len(question["criteria"]) if question["type"] == "choice" else 2
            rows.append([0.75, *([0.25 / (count - 1)] * (count - 1))])
            counts.append(count)
            if question["type"] == "noul":
                answers[identifier] = {"type": "noul", "noul": 0.25}
            else:
                probabilities = [p / sum(rows[-1]) for p in rows[-1]]
                answers[identifier] = {
                    "type": "choice",
                    "choice": next(iter(question["criteria"])),
                    "confidence": round(max(probabilities), 4),
                    "probabilities": dict(
                        zip(question["criteria"], (round(p, 4) for p in probabilities))
                    ),
                }
    case.update(row_probabilities=rows, row_index=index)
    case.update(
        answers=answers,
        response={"answers": copy.deepcopy(answers)},
        prompt_rows=[
            {
                "ids": [1, 2, 3],
                "slots": [2],
                "nopts": [count],
                "padded_sequence_length": 64,
            }
            for count in counts
        ],
        forward_input_shapes=[[1, 64] for _ in rows],
        readout_dtypes=["torch.float32"] * len(rows),
    )
    return case


def test_independent_probability_mapping_uses_project_confidence_not_author_maxp(
    driver,
):
    actual = driver.canonical_answers(author_case(driver.CASES[0]))
    assert actual["department"]["choice"] == "billing"
    assert actual["department"]["confidence"] == 0.5
    assert actual["refund"] == {"type": "noul", "noul": 0.25}
    assert actual["urgency"]["score"] == 1.0
    assert actual["urgency"]["confidence"] == pytest.approx(0.4)
    assert actual["urgency"]["probabilities"] == {"0": 0.2, "1": 0.6, "2": 0.2}


def test_wide_choice_is_really_wider_than_ten_and_preserves_label_order(driver):
    case = author_case(driver.CASES[-1])
    actual = driver.canonical_answers(case)["category"]
    assert len(actual["probabilities"]) == 12
    assert list(actual["probabilities"]) == list(
        case["questions"]["category"]["criteria"]
    )
    assert actual["choice"] == "refund"


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.1, True])
def test_nonfinite_negative_or_boolean_probabilities_cannot_pass(driver, bad):
    case = author_case(driver.CASES[0])
    case["row_probabilities"][0][0] = bad
    with pytest.raises(ValueError, match="finite"):
        driver.canonical_answers(case)


def test_wrong_question_row_mapping_and_nonzero_masked_options_are_rejected(driver):
    case = author_case(driver.CASES[0])
    case["row_index"][0][0] = "other_question"
    with pytest.raises(ValueError, match="row index"):
        driver.canonical_answers(case)
    case = author_case(driver.CASES[0])
    case["row_probabilities"][0] = [0.7, 0.2, 0.1]
    with pytest.raises(ValueError, match="masked options"):
        driver.canonical_answers(case)


def test_clone_only_pins_files_and_never_hardlinks_configuration(
    driver, source, tmp_path
):
    (source / "README.md").write_text("not required to load", encoding="utf-8")
    (source / ".cache").mkdir()
    clone = tmp_path / "private clone"
    driver.clone_source(source, clone)
    assert {p.name for p in clone.iterdir()} == set(driver.SOURCE_HASHES)
    assert driver.verify_source(clone) == driver.SOURCE_HASHES
    original = (source / "config.json").read_bytes()
    (clone / "config.json").write_text("runtime rewrite", encoding="utf-8")
    assert (source / "config.json").read_bytes() == original


@pytest.mark.parametrize(
    "extra", ["adapter_config.json", "special_tokens_map.json", "templates"]
)
def test_unverified_loader_override_is_rejected_before_hashing(
    driver, source, monkeypatch, extra
):
    (source / extra).write_text("override", encoding="utf-8")
    digest = Mock(side_effect=AssertionError("should reject layout first"))
    monkeypatch.setattr(driver, "digest", digest)
    with pytest.raises(ValueError, match="unexpected"):
        driver.verify_source(source)
    digest.assert_not_called()


def test_memory_gate_uses_available_physical_and_commit_separately(driver, monkeypatch):
    args = SimpleNamespace(min_available_gib=4, min_commit_available_gib=6)
    monkeypatch.setattr(
        driver,
        "system_memory",
        lambda: {"available_bytes": 3 * 2**30, "commit_available_bytes": 40 * 2**30},
    )
    assert driver.memory_gate(args)["passed"] is False
    monkeypatch.setattr(
        driver,
        "system_memory",
        lambda: {"available_bytes": 5 * 2**30, "commit_available_bytes": 5 * 2**30},
    )
    assert driver.memory_gate(args)["passed"] is False
    monkeypatch.setattr(
        driver,
        "system_memory",
        lambda: {"available_bytes": 5 * 2**30, "commit_available_bytes": 7 * 2**30},
    )
    assert driver.memory_gate(args)["passed"] is True


def test_low_memory_watchdog_stops_only_the_supplied_owned_tree(driver, monkeypatch):
    import psutil

    monkeypatch.setattr(
        psutil, "virtual_memory", lambda: SimpleNamespace(available=100)
    )
    tree, record = Mock(), {}
    monitor = driver.MemoryMonitor(
        tree, SimpleNamespace(abort_below_available_gib=0.75), record
    )
    monitor.start()
    monitor.thread.join(timeout=3)
    assert monitor.stop() is False
    tree.stop.assert_called_once_with()
    assert record["memory_monitor"]["aborted"] is True


def test_failed_process_cleanup_preserves_private_workspace(driver, tmp_path):
    temporary = SimpleNamespace(name=str(tmp_path / "owned"), cleanup=Mock())
    tree = SimpleNamespace(stop=Mock(return_value=False))
    record = {}
    assert driver.finish_run(temporary, tree, object(), record, None) is False
    temporary.cleanup.assert_not_called()
    assert record["retained_workspace"] == temporary.name


def test_mutated_private_source_fails_even_after_successful_process_cleanup(
    driver, source
):
    (source / "config.json").write_text("modified", encoding="utf-8")
    temporary = SimpleNamespace(name=str(source), cleanup=Mock())
    tree = SimpleNamespace(stop=lambda: True)
    record = {}
    assert driver.finish_run(temporary, tree, object(), record, source) is False
    assert record["source_unchanged"] is False
    temporary.cleanup.assert_called_once()


@pytest.fixture
def reference(driver, tmp_path, monkeypatch):
    runner = tmp_path / "reference.py"
    runner.write_text("# trusted fixture runner\n", encoding="utf-8")
    packages = {
        name: "test version" for name in ("torch", "transformers", "safetensors")
    }
    monkeypatch.setattr(driver.common, "environment", lambda: {"packages": packages})
    report = {
        "status": "passed",
        "model_id": driver.MODEL,
        "revision": driver.REVISION,
        "source_unchanged": True,
        "sdk_unchanged": True,
        "measured": True,
        "sdk_hashes": driver.SDK_HASHES,
        "checkpoint_hashes": driver.SOURCE_HASHES,
        "driver_sha256": driver.digest(runner),
        "reference_kind": "original_hf_release_cpu_eager",
        "environment": {"packages": packages.copy()},
        "runtime": {
            "threads": 4,
            "dtype": "bfloat16",
            "row_batch_size": 1,
            "padding_multiple": 64,
            "graphs": False,
            "actual_parameter_devices": ["cpu"],
            "actual_embedding_dtype": "torch.bfloat16",
            "actual_head_dtype": "torch.bfloat16",
            "actual_readout_dtype": "torch.float32",
            "independent": True,
            "isolated_levels": True,
            "layout": "state_first",
            "temperature": 1.03,
            "max_fwd_tokens": 1,
            "max_state_tokens": 32768,
            "cross_request_cache": False,
            "raw_probabilities_capture": "original_return_frame.flatp",
        },
        "cases": [author_case(request) for request in driver.CASES],
    }
    return (
        SimpleNamespace(reference_driver=runner, threads=4),
        tmp_path / "reference.json",
        report,
    )


def test_reference_mapping_preserves_original_author_response(driver, reference):
    args, path, report = reference
    driver.save(path, report)
    checked = driver.validate_reference(args, path)
    assert checked["cases"][0]["answers"] == report["cases"][0]["answers"]
    assert checked["cases"][0]["project_answers"]["department"]["confidence"] == 0.5


@pytest.fixture
def reference_2b(driver, reference):
    args, path, report = reference
    args.model = "Mapika/decider-2b"
    profile = driver.PROFILES[args.model]
    report.update(
        model_id=args.model,
        revision=profile["revision"],
        sdk_hashes=copy.deepcopy(profile["sdk_hashes"]),
        checkpoint_hashes=copy.deepcopy(profile["source_hashes"]),
        reference_kind=profile["reference_kind"],
    )
    report["runtime"].update(
        temperature=1.145,
        temperature_by_type={"choice": 1.164, "noul": 1.624, "score": 1.124},
        model_layout="plain",
        chat_template_applied=False,
        actual_linear_input_dtype="torch.bfloat16",
        actual_linear_weight_dtype="torch.bfloat16",
    )
    for case in report["cases"]:
        temperatures, types = [], []
        for identifier, _, _, count in case["row_index"]:
            kind = case["questions"][identifier]["type"]
            temperatures.extend(
                [report["runtime"]["temperature_by_type"][kind]] * count
            )
            types.extend([[kind]] * count)
            answer = case["answers"][identifier]
            if kind == "choice":
                answer.update(
                    confidence=0.7273 if identifier == "category" else 0.5,
                    x_p_max=0.75,
                )
            elif kind == "score":
                answer.update(confidence=0.4, x_p_max=0.6)
        case.update(
            response={"answers": copy.deepcopy(case["answers"])},
            row_temperatures=temperatures,
            linear_input_dtypes=["torch.bfloat16"] * len(types),
            linear_weight_dtypes=["torch.bfloat16"] * len(types),
        )
        for row, kind in zip(case["prompt_rows"], types):
            row["types"] = kind
    return args, path, report


def test_2b_reference_uses_v11_confidence_and_original_score_row_temperature(
    driver, reference_2b
):
    args, path, report = reference_2b
    driver.save(path, report)
    checked = driver.validate_reference(args, path)
    mixed = checked["cases"][0]
    assert mixed["row_temperatures"] == [1.164, 1.624, 1.124, 1.124, 1.124]
    assert mixed["answers"]["department"]["confidence"] == 0.5
    assert mixed["answers"]["department"]["x_p_max"] == 0.75
    assert mixed["project_answers"]["urgency"]["confidence"] == pytest.approx(0.4)
    assert "x_p_max" not in mixed["project_answers"]["department"]
    assert (
        len(checked["cases"][-1]["project_answers"]["category"]["probabilities"]) == 12
    )


@pytest.mark.parametrize(
    "change",
    [
        "score_temperature",
        "score_row_type",
        "old_confidence",
        "max_probability",
        "linear_operand",
        "chat_layout",
        "global_temperature",
    ],
)
def test_2b_reference_rejects_wrong_calibration_presentation_or_actual_operands(
    driver, reference_2b, change
):
    args, path, report = reference_2b
    case = report["cases"][0]
    if change == "score_temperature":
        case["row_temperatures"][2] = 1.624
    elif change == "score_row_type":
        case["prompt_rows"][2]["types"] = ["noul"]
    elif change == "old_confidence":
        case["answers"]["department"]["confidence"] = 0.75
    elif change == "max_probability":
        case["answers"]["department"]["x_p_max"] = 0.5
    elif change == "linear_operand":
        case["linear_input_dtypes"][0] = "torch.float32"
    elif change == "chat_layout":
        report["runtime"]["chat_template_applied"] = True
    else:
        report["runtime"]["temperature"] = 1.03
    case["response"]["answers"] = copy.deepcopy(case["answers"])
    driver.save(path, report)
    with pytest.raises((ValueError, AssertionError)):
        driver.validate_reference(args, path)


@pytest.mark.parametrize(
    "field", ["model_id", "revision", "sdk_hashes", "checkpoint_hashes"]
)
def test_2b_reference_cannot_reuse_any_other_release_provenance(
    driver, reference_2b, field
):
    args, path, report = reference_2b
    old = driver.PROFILES[driver.MODEL]
    report[field] = {
        "model_id": driver.MODEL,
        "revision": old["revision"],
        "sdk_hashes": old["sdk_hashes"],
        "checkpoint_hashes": old["source_hashes"],
    }[field]
    driver.save(path, report)
    with pytest.raises(ValueError, match=f"mismatch: {field}"):
        driver.validate_reference(args, path)


@pytest.mark.parametrize("model", ["Mapika/decider-0.8b", "Mapika/decider-2b"])
def test_reference_launch_passes_selected_profile_to_repository_runner(
    driver, tmp_path, monkeypatch, model
):
    args = SimpleNamespace(
        model=model,
        workspace=tmp_path,
        source=tmp_path / "original",
        reference_python=Path(sys.executable),
        reference_driver=driver.REFERENCE_RUNNER,
        sdk_source=tmp_path / "sdk",
        output=tmp_path / "report.json",
        threads=4,
    )
    profile = driver.PROFILES[model]
    monkeypatch.setattr(
        driver, "clone_source", lambda source, target, selected: target.mkdir()
    )
    monkeypatch.setattr(
        driver, "verify_source", lambda source, selected: profile["source_hashes"]
    )
    launch = Mock(side_effect=RuntimeError("intercept before process creation"))
    monkeypatch.setattr(driver.subprocess, "Popen", launch)
    report = {}
    with pytest.raises(RuntimeError, match="intercept before"):
        driver.run_reference(args, {}, report, tmp_path / "reference.json")
    command = launch.call_args.args[0]
    assert Path(command[2]) == driver.REFERENCE_RUNNER
    assert command[command.index("--model") + 1] == model
    assert command[command.index("--max-fwd-tokens") + 1] == "1"
    assert report["reference_process"]["private_workspace_removed"] is True
    assert not list(tmp_path.glob("decider-reference-*"))


@pytest.mark.parametrize(
    "model,override,expected",
    [
        ("Mapika/decider-0.8b", [], (4.0, 6.0)),
        ("Mapika/decider-2b", [], (6.0, 8.0)),
        (
            "Mapika/decider-2b",
            ["--min-available-gib", "7", "--min-commit-available-gib", "9"],
            (7.0, 9.0),
        ),
    ],
)
def test_cli_model_specific_memory_gates_precede_any_source_or_model_loading(
    driver, tmp_path, monkeypatch, model, override, expected
):
    output = tmp_path / "gate.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            driver.__file__,
            "--model",
            model,
            "--source",
            str(tmp_path / "not_present"),
            "--workspace",
            str(tmp_path / "workspace"),
            "--output",
            str(output),
            "--sdk-source",
            str(tmp_path / "not_present_sdk"),
            *override,
        ],
    )
    monkeypatch.setattr(driver.sys, "platform", "win32")
    captured = []

    def gate(args):
        captured.append((args.min_available_gib, args.min_commit_available_gib))
        assert args.abort_below_available_gib == 0.75
        assert args.reference_driver == driver.REFERENCE_RUNNER.resolve()
        return {"passed": False}

    monkeypatch.setattr(driver, "memory_gate", gate)
    load = Mock(side_effect=AssertionError("memory gate must run first"))
    monkeypatch.setattr(driver, "verify_source", load)
    driver.main()
    assert captured == [expected]
    assert driver.read_json(output)["status"] == "skipped"
    load.assert_not_called()


def test_legacy_reference_reuse_checks_old_runner_without_launching_it(
    driver, reference, tmp_path, monkeypatch
):
    args, reference_path, original = reference
    driver.save(reference_path, original)
    output = tmp_path / "reuse.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            driver.__file__,
            "--model",
            driver.MODEL,
            "--source",
            str(tmp_path / "source"),
            "--workspace",
            str(tmp_path / "workspace"),
            "--output",
            str(output),
            "--sdk-source",
            str(tmp_path / "sdk"),
            "--reference-driver",
            str(args.reference_driver),
            "--reference-report",
            str(reference_path),
        ],
    )
    monkeypatch.setattr(driver.sys, "platform", "win32")
    monkeypatch.setattr(driver, "memory_gate", lambda args: {"passed": True})
    monkeypatch.setattr(driver, "verify_source", lambda *args: driver.SOURCE_HASHES)
    monkeypatch.setattr(driver, "verify_sdk", lambda *args: driver.SDK_HASHES)
    monkeypatch.setattr(driver.socket, "socket", MagicMock())
    launch = Mock(side_effect=AssertionError("historical reference must not rerun"))
    monkeypatch.setattr(driver, "run_reference", launch)
    product = Mock(side_effect=RuntimeError("stop before product model launch"))
    monkeypatch.setattr(driver, "service_run", product)
    with pytest.raises(RuntimeError, match="stop before product"):
        driver.main()
    saved = driver.read_json(output)
    assert saved["reference_provenance"]["reused"] is True
    assert saved["reference_provenance"]["rerun_this_invocation"] is False
    assert saved["reference_provenance"]["sha256"] == driver.digest(reference_path)
    product.assert_called_once()
    launch.assert_not_called()


@pytest.mark.parametrize(
    "change",
    [
        "empty_answers",
        "wrong_probability",
        "unpadded_shape",
        "different_readout",
        "overlong",
    ],
)
def test_reference_observations_must_match_original_response_and_actual_shapes(
    driver, reference, change
):
    args, path, report = reference
    case = report["cases"][0]
    if change == "empty_answers":
        case["answers"] = case["response"]["answers"] = {}
    elif change == "wrong_probability":
        case["answers"]["department"]["probabilities"]["billing"] = 0.8
        case["response"]["answers"] = copy.deepcopy(case["answers"])
    elif change == "unpadded_shape":
        case["forward_input_shapes"][0] = [1, 3]
    elif change == "different_readout":
        case["prompt_rows"][0]["slots"] = [0]
    else:
        case["prompt_rows"][0]["ids"] = [1] * 1025
    driver.save(path, report)
    with pytest.raises((ValueError, AssertionError)):
        driver.validate_reference(args, path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("dtype", "float32"),
        ("threads", 2),
        ("graphs", True),
        ("actual_parameter_devices", ["cuda:0"]),
        ("row_batch_size", 2),
        ("padding_multiple", 1),
    ],
)
def test_reference_reuse_cannot_cross_runtime_configurations(
    driver, reference, field, value
):
    args, path, report = reference
    report["runtime"][field] = value
    driver.save(path, report)
    with pytest.raises(ValueError, match="runtime mismatch"):
        driver.validate_reference(args, path)


@pytest.mark.parametrize(
    "change", ["source", "sdk", "runner", "cases", "package", "unmeasured"]
)
def test_reference_reuse_binds_source_implementation_cases_and_environment(
    driver, reference, change
):
    args, path, report = reference
    report = copy.deepcopy(report)
    if change == "source":
        report["checkpoint_hashes"]["config.json"] = "wrong"
    elif change == "sdk":
        report["sdk_hashes"]["decider/model.py"] = "wrong"
    elif change == "runner":
        report["driver_sha256"] = "wrong"
    elif change == "cases":
        report["cases"][0]["state"] = "Different input"
    elif change == "package":
        report["environment"]["packages"]["transformers"] = "different"
    else:
        report["measured"] = False
    driver.save(path, report)
    with pytest.raises(ValueError, match="mismatch"):
        driver.validate_reference(args, path)


@pytest.mark.parametrize(
    "payload", ['{"status":"failed","status":"passed"}', '{"probability":NaN}']
)
def test_ambiguous_or_nonfinite_reference_json_is_rejected(driver, tmp_path, payload):
    path = tmp_path / "bad.json"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError):
        driver.read_json(path)


def test_import_does_not_import_torch_transformers_or_inference_backends(driver):
    code = """
import importlib.abc, importlib.util, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','transformers','peft','vllm','mlx','vllm_jev','decider'}:
            raise AssertionError('unexpected heavy import: ' + fullname)
sys.meta_path.insert(0, Block())
spec=importlib.util.spec_from_file_location('driver', sys.argv[1])
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert len(module.CASES)==3
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", code, driver.__file__],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
