"""Shared dispatch and independently specified acceptance boundaries."""

import importlib.util
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from vllm_jev import windows_models, windows_qwen

SPEC = importlib.util.spec_from_file_location(
    "candidate_acceptance",
    Path(__file__).resolve().parents[1] / "scripts/check_windows_candidates.py",
)
driver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(driver)


@pytest.fixture
def valid_reference(tmp_path, monkeypatch, request):
    import copy
    import json
    from types import SimpleNamespace

    profile = getattr(request, "param", "jevk5")
    cases = driver.cases_for(profile)
    if profile != "jevk5_lite":
        cases = cases[:2]
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(json.dumps(cases, ensure_ascii=False), encoding="utf-8")
    evidence = {
        "model": driver.PROFILES[profile][0],
        "revision": "release-pin-" + profile,
        "sdk_revision": "author-pin",
        "driver_sha256": "runner-sha",
        "source_hashes": {"weight": "source-sha"},
        "sdk_hashes": {"author.py": "sdk-sha"},
        "versions": {"torch": "2.8.0+cpu", "transformers": "5.17.0", "numpy": "2.4.0"},
    }
    monkeypatch.setattr(
        driver, "_reference_evidence", lambda *args: copy.deepcopy(evidence)
    )
    report = {
        "status": "passed",
        "measured": True,
        "source_unchanged": True,
        "sdk_unchanged": True,
        "model_id": evidence["model"],
        **{
            name: copy.deepcopy(value)
            for name, value in evidence.items()
            if name != "model"
        },
        "input_sha256": driver.common.digest(cases_path),
        "runtime": {
            "threads": 4,
            "dtype": "bfloat16",
            "parameter_devices": ["cpu"],
            "parameter_dtypes": ["torch.bfloat16"],
            "head_dtype": "torch.bfloat16",
            "head_shape": [16, 2048],
            "row_batch_size": 1,
            "padding": "none",
            "use_cache": False,
            "graphs": False,
            "max_sequence_tokens": 1024,
            "temperature": 1.42,
        },
        "cases": [
            {
                **copy.deepcopy(case),
                "probabilities": {
                    name: [1 / (2 if q["type"] == "noul" else len(q["criteria"]))]
                    * (2 if q["type"] == "noul" else len(q["criteria"]))
                    for name, q in case["questions"].items()
                },
            }
            for case in cases
        ],
    }
    if profile in driver.THISTHAT_PROFILES:
        evidence["helper_sha256"] = report["helper_sha256"] = "helper-pin"
        evidence["environment_identity"] = {
            "python": "3.12",
            "executable": "author.exe",
        }
        report["environment"] = dict(evidence["environment_identity"])
        report["runtime"] = {
            "threads": 4,
            "actual_parameter_devices": ["cpu"],
            "actual_parameter_dtypes": ["torch.float32"],
            "actual_embedding_dtype": "torch.float32",
            "actual_head_dtype": "torch.float32",
            "interop_threads": 1,
            "attention_implementation": "sdpa",
            "config_use_cache": True,
            "layout": "state_first",
            "max_state_tokens": 1536,
            "temperature": 1.0,
            "padding_multiple": 64,
            "packed_questions": True,
            "cross_request_cache": False,
        }
    elif profile == "jevk5_lite":
        evidence["environment_identity"] = {
            "python": "3.12",
            "executable": "lite-author.exe",
        }
        report["environment"] = dict(evidence["environment_identity"])
        report["private_source_unchanged"] = True
        report["private_sdk_unchanged"] = True
        report["runtime"] = {
            "threads": 4,
            "dtype": "float32",
            "parameter_devices": ["cpu"],
            "parameter_dtypes": ["torch.float32"],
            "head_dtype": "torch.float32",
            "row_batch_size": 1,
            "padding": "none",
            "use_cache": False,
            "graphs": False,
            "max_sequence_tokens": 512,
            "temperature_single": 1.8660659830736146,
            "temperature_multi": 1.319507910772894,
            "truncation": False,
            "packed_questions": False,
        }
    path = tmp_path / "reference.json"
    args = SimpleNamespace(profile=profile, threads=4)

    def validate(value=None):
        path.write_text(
            json.dumps(report if value is None else value, ensure_ascii=False),
            encoding="utf-8",
        )
        return driver.validate_reference(
            path,
            cases,
            args=args,
            runner=tmp_path / "runner.py",
            cases_path=cases_path,
            source_hashes=evidence["source_hashes"],
        )

    return report, validate, cases_path, evidence


def test_reference_accepts_exact_completed_evidence(valid_reference):
    report, validate, _, _ = valid_reference
    assert validate() == report


@pytest.mark.parametrize(
    "name,value",
    [
        ("status", "running"),
        ("measured", False),
        ("measured", 1),
        ("source_unchanged", False),
        ("sdk_unchanged", None),
        ("model_id", "other/model"),
        ("revision", "new-pin"),
        ("sdk_revision", "new-author"),
        ("driver_sha256", "tampered-runner"),
        ("input_sha256", "different-request"),
        ("source_hashes", {"weight": "changed"}),
        ("sdk_hashes", {}),
        ("versions", {"torch": "2.9.0"}),
        ("cases", []),
    ],
)
def test_reference_rejects_partial_stale_or_tampered_evidence(
    valid_reference, name, value
):
    report, validate, _, _ = valid_reference
    report[name] = value
    with pytest.raises(ValueError):
        validate()


@pytest.mark.parametrize(
    "name,value",
    [
        ("threads", 2),
        ("threads", True),
        ("dtype", "float32"),
        ("parameter_devices", ["xpu"]),
        ("parameter_dtypes", ["torch.float32"]),
        ("head_dtype", "torch.float32"),
        ("head_shape", [2, 2048]),
        ("row_batch_size", 2),
        ("padding", "right_to_multiple_of_64"),
        ("use_cache", True),
        ("graphs", True),
        ("max_sequence_tokens", 2048),
        ("temperature", 1.0),
    ],
)
def test_reference_rejects_runtime_mismatch(valid_reference, name, value):
    report, validate, _, _ = valid_reference
    report["runtime"][name] = value
    with pytest.raises(ValueError):
        validate()


@pytest.mark.parametrize(
    "change",
    [
        "row_order",
        "state",
        "question",
        "question_order",
        "option_order",
        "missing_state",
        "probability",
        "nonfinite",
        "duplicate_row",
    ],
)
def test_reference_requires_exact_cases_and_raw_probabilities(valid_reference, change):
    report, validate, _, _ = valid_reference
    row = report["cases"][0]
    if change == "row_order":
        report["cases"].reverse()
    elif change == "state":
        row["state"] = "different evidence"
    elif change == "question":
        row["questions"]["refund"]["instructions"] = "A different question?"
    elif change == "question_order":
        row["questions"] = dict(reversed(list(row["questions"].items())))
    elif change == "option_order":
        question = row["questions"]["department"]
        question["criteria"] = dict(reversed(list(question["criteria"].items())))
    elif change == "missing_state":
        del row["state"]
    elif change == "probability":
        row["probabilities"]["refund"] = [0.1, 0.1]
    elif change == "nonfinite":
        row["probabilities"]["refund"] = [float("nan"), 0.5]
    else:
        report["cases"][1] = row
    with pytest.raises(ValueError):
        validate()


def test_reference_rejects_conflicting_identity_aliases(valid_reference):
    report, validate, _, _ = valid_reference
    report["model"] = "other/model"
    with pytest.raises(ValueError, match="conflicting"):
        validate()
    del report["model"]
    report["sdk_commit"] = "different-pin"
    with pytest.raises(ValueError, match="conflicting"):
        validate()


def test_reference_rechecks_actual_request_file(valid_reference):
    import json

    report, validate, path, _ = valid_reference
    changed = json.loads(path.read_text(encoding="utf-8"))
    changed[0]["state"] = "changed"
    path.write_text(json.dumps(changed), encoding="utf-8")
    report["input_sha256"] = driver.common.digest(path)
    with pytest.raises(ValueError, match="selected cases"):
        validate()


def test_reference_evidence_checks_current_pins_and_sdk_bytes(tmp_path, monkeypatch):
    from types import SimpleNamespace

    sdk = tmp_path / "sdk"
    sdk.mkdir()
    (sdk / "author.py").write_text("# pinned source\n")
    runner = tmp_path / "runner.py"
    runner.write_text(
        "MODEL_ID='model'\nREVISION='release'\nSDK_COMMIT='commit'\nSOURCE_HASHES={'weight':'expected'}\nSDK_HASHES={'author.py':"
        + repr(driver.common.digest(sdk / "author.py"))
        + "}\n"
    )
    args = SimpleNamespace(
        profile="jevk5", sdk_source=sdk, reference_python=tmp_path / "python.exe"
    )
    monkeypatch.setattr(
        driver, "_reference_versions", lambda *args: {"torch": "2.8.0+cpu"}
    )
    result = driver._reference_evidence(args, runner, {"weight": "expected"})
    assert result["driver_sha256"] == driver.common.digest(runner)
    with pytest.raises(ValueError, match="release pins"):
        driver._reference_evidence(args, runner, {"weight": "changed"})
    (sdk / "author.py").write_text("# altered\n")
    with pytest.raises(ValueError, match="SDK"):
        driver._reference_evidence(args, runner, {"weight": "expected"})


def test_independent_answer_formulas():
    case = {
        "questions": {
            "c": {"type": "choice", "criteria": {"a": "A", "b": "B"}},
            "n": {"type": "noul"},
            "s": {"type": "score", "criteria": ["low", "high"]},
        }
    }
    p = {"c": [0.25, 0.75], "n": [0.1, 0.9], "s": [0.25, 0.75]}
    result = driver.canonical(case, p, 1)
    assert result["c"] == {
        "type": "choice",
        "probabilities": {"a": 0.25, "b": 0.75},
        "choice": "b",
        "confidence": 0.5,
    }
    assert result["n"] == {"type": "noul", "noul": 0.9}
    assert result["s"]["score"] == 0.75
    assert result["s"]["confidence"] == 0.5
    assert driver.canonical(case, p, 0)["n"]["noul"] == 0.1


@pytest.mark.parametrize(
    "p", [[0.1], [0.1, 0.1], [float("nan"), 0.5], [True, 0], [-0.1, 1.1]]
)
def test_reference_rejects_invalid_probability_evidence(p):
    with pytest.raises(ValueError):
        driver.canonical({"questions": {"n": {"type": "noul"}}}, {"n": p}, 1)


def test_workspace_preserved_when_owned_process_not_stopped(tmp_path):
    record = {"owned_process_stopped": False}
    with driver.private_workspace(tmp_path, record) as directory:
        (directory / "evidence.txt").write_text("retained")
    assert directory.is_dir()
    assert record["private_workspace_removed"] is False


@pytest.mark.parametrize(
    "model,protocol,backend",
    [
        ("alibiserikbay/JevK5-2B", "jevk5_letters_v1", "jevk5-pytorch"),
        ("alibiserikbay/JevK5-Lite", "jevk5_lite_v1", "jevk5-lite-pytorch"),
        ("jaredpalmer/kev-0.8b", "kev_pointer_v1", "kev-pytorch"),
        ("flock-io/this-that-model-1.2", "thisthat_slot_v1", "thisthat-pytorch"),
        ("flock-io/this-that-model-1.1", "thisthat_slot_v1", "thisthat-pytorch"),
        ("flock-io/this-that-model-1.0", "thisthat_slot_v1", "thisthat-pytorch"),
    ],
)
def test_shared_dispatch_and_http_identity(tmp_path, model, protocol, backend):
    class Agent:
        closed = False

        def system_one(self, state, questions):
            return {"answers": {"q": {"type": "noul", "noul": 0.75}}}

        def close(self):
            self.closed = True

    agent = Agent()
    assert windows_models.protocol_for(model) == protocol
    app = windows_qwen.create_app(tmp_path, model, agent_factory=lambda: agent)
    with TestClient(app) as client:
        health = client.get("/health").json()
        assert health["backend"] == backend
        payload = {
            "state": "hello",
            "questions": {"q": {"type": "noul", "instructions": "Is this a greeting?"}},
        }
        response = client.post("/v1/systemone", json=payload)
        assert response.status_code == 200
        assert response.json()["model"] == model
        assert response.json()["metadata"]["device"] == "cpu"
        if model.startswith("flock-io/"):
            from vllm_jev import windows_thisthat

            revision = windows_thisthat.MODELS[model]["revision"]
            assert health["revision"] == revision
            assert response.json()["metadata"]["revision"] == revision
            if model != windows_thisthat.MODEL_ID:
                assert revision != windows_thisthat.REVISION
        elif model == "alibiserikbay/JevK5-Lite":
            assert health["revision"] == "315ee211f828a899161477c534cea23e84ba3568"
            assert response.json()["metadata"]["revision"] == health["revision"]
        assert (
            client.post(
                "/v1/systemone", json={**payload, "images": ["invalid"]}
            ).status_code
            == 422
        )
    assert agent.closed


@pytest.mark.parametrize("profile", list(driver.PROFILES))
@pytest.mark.parametrize("local", [False, True])
def test_cli_checks_dependency_profile_before_prepare_or_serve(
    monkeypatch, tmp_path, profile, local
):
    from vllm_jev import cli, windows_dependencies

    model = driver.PROFILES[profile][0]
    source = tmp_path / "中文 checkpoint with spaces"
    source.mkdir()
    events = []
    monkeypatch.setattr(cli.sys, "platform", "win32")
    monkeypatch.setattr(windows_models, "identify_source", lambda path: model)

    def check(model_id):
        events.append(("dependencies", model_id))

    def prepare(model_id, workspace):
        events.append(("prepare", model_id))
        return source

    monkeypatch.setattr(windows_dependencies, "require_windows_dependencies", check)
    monkeypatch.setattr(windows_models, "prepare", prepare)
    monkeypatch.setattr(
        windows_qwen,
        "serve",
        lambda path, model_id, **kw: events.append(("serve", model_id)),
    )
    monkeypatch.setattr(
        cli.sys,
        "argv",
        ["vllm-jev", "serve", str(source) if local else model, "--cpu-threads", "1"],
    )
    cli.main()
    assert events == [("dependencies", model)] + (
        [] if local else [("prepare", model)]
    ) + [("serve", model)]


def test_cli_dependency_failure_prevents_download_and_model_start(monkeypatch, capsys):
    from vllm_jev import cli, windows_dependencies

    monkeypatch.setattr(cli.sys, "platform", "win32")
    monkeypatch.setattr(
        cli.sys, "argv", ["vllm-jev", "serve", driver.PROFILES["kev"][0]]
    )

    def reject(model_id):
        raise ValueError("Use an isolated windows-modern environment")

    monkeypatch.setattr(windows_dependencies, "require_windows_dependencies", reject)
    monkeypatch.setattr(
        windows_models, "prepare", lambda *a: pytest.fail("must not download")
    )
    monkeypatch.setattr(
        windows_qwen, "serve", lambda *a, **kw: pytest.fail("must not load")
    )
    with pytest.raises(SystemExit) as failure:
        cli.main()
    assert failure.value.code == 2
    assert "windows-modern" in capsys.readouterr().err


def test_single_candidate_choice_and_score_have_defined_confidence():
    case = {
        "questions": {
            "choice": {"type": "choice", "criteria": {"only": "Only option"}},
            "score": {"type": "score", "criteria": ["Only level"]},
        }
    }
    result = driver.canonical(case, {"choice": [1.0], "score": [1.0]}, 1, "kev")
    assert result["choice"] == {
        "type": "choice",
        "choice": "only",
        "confidence": 1.0,
        "probabilities": {"only": 1.0},
    }
    assert result["score"] == {
        "type": "score",
        "score": 0.0,
        "confidence": 1.0,
        "probabilities": {"0": 1.0},
        "legend": {"0": "Only level"},
    }


def test_reference_parent_removes_scratch_after_hard_child_timeout(
    tmp_path, monkeypatch
):
    """Kill a real lightweight Python child that cannot run its finalizer."""
    import json
    import os
    import subprocess
    import sys
    import time
    from types import SimpleNamespace

    class Monitor:
        def __init__(self, *args):
            pass

        def start(self):
            pass

        def stop(self):
            return True

    monkeypatch.setattr(driver.common, "memory_gate", lambda args: {"passed": True})
    monkeypatch.setattr(driver.common, "MemoryMonitor", Monitor)
    args = SimpleNamespace(workspace=tmp_path)
    record = {}
    code = (
        "import json,os,pathlib,time; "
        "root=pathlib.Path.cwd(); "
        "scratch=root/'kev-reference-aborted'; scratch.mkdir(); "
        "(scratch/'weights-link-marker').write_text('owned'); "
        "(root/'ready.json').write_text(json.dumps({'cwd':str(root),'hf':os.environ['HF_HOME'],'tmp':os.environ['TEMP']})); "
        "time.sleep(60)"
    )

    def action(process, tree):
        directory = Path(record["workspace"])
        deadline = time.monotonic() + 10
        while not (directory / "ready.json").exists() and time.monotonic() < deadline:
            assert process.poll() is None
            time.sleep(0.01)
        observed = json.loads((directory / "ready.json").read_text())
        assert Path(observed["cwd"]) == directory
        assert Path(observed["hf"]).is_relative_to(directory)
        assert Path(observed["tmp"]).is_relative_to(directory)
        process.wait(timeout=0.05)

    with pytest.raises(subprocess.TimeoutExpired):
        driver.reference_run(
            args,
            [sys.executable, "-c", code],
            os.environ.copy(),
            tmp_path / "author.log",
            record,
            action,
        )
    assert args.workspace == tmp_path  # Do not redirect later product runs.
    assert record["owned_process_stopped"] is True
    assert record["private_workspace_removed"] is True
    assert not Path(record["workspace"]).exists()


def test_reference_workspace_is_retained_when_child_cleanup_is_unproven(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    args = SimpleNamespace(workspace=tmp_path)
    record = {}

    def failed_child(args, command, environment, log, record, action):
        (args.workspace / "evidence").write_text("may still be in use")
        record.update(launcher_pid=123, owned_process_stopped=False)
        raise RuntimeError("could not stop owned process")

    monkeypatch.setattr(driver, "child_run", failed_child)
    with pytest.raises(RuntimeError, match="could not stop"):
        driver.reference_run(args, [], {}, tmp_path / "log", record, None)
    assert record["private_workspace_removed"] is False
    assert Path(record["retained_workspace"]).is_dir()


@pytest.mark.parametrize("failure", ["http", "precision", "probability"])
def test_failed_response_is_saved_before_comparison(tmp_path, failure):
    import json

    observations = []
    report = {"status": "running", "service_runs": [{"cases": observations}]}
    output = tmp_path / "acceptance.json"
    expected = {"q": {"type": "noul", "noul": 0.75}}
    response = {
        "metadata": {
            "backbone_dtype": "bfloat16" if failure == "precision" else "float32"
        },
        "answers": {"q": {"type": "noul", "noul": 0.1}},
    }
    with pytest.raises((AssertionError, ValueError, RuntimeError)):
        driver.record_and_compare_response(
            output,
            report,
            observations,
            status=422 if failure == "http" else 200,
            response=response,
            seconds=1.2,
            expected=expected,
            dtype="float32",
            case_id="failed-case",
        )
    saved = json.loads(output.read_text(encoding="utf-8"))["service_runs"][0]["cases"]
    assert len(saved) == 1
    assert saved[0]["id"] == "failed-case"
    assert saved[0]["response"] == response
    assert saved[0]["status"] == (422 if failure == "http" else 200)


@pytest.mark.parametrize(
    "valid_reference", sorted(driver.THISTHAT_PROFILES), indirect=True
)
def test_thisthat_reference_accepts_selected_release_only(valid_reference):
    report, validate, _, evidence = valid_reference
    assert validate() == report
    for profile in driver.THISTHAT_PROFILES:
        other = driver.PROFILES[profile][0]
        if other == evidence["model"]:
            continue
        report["model_id"] = other
        with pytest.raises(ValueError, match="model identity"):
            validate()
    report["model_id"] = evidence["model"]
    report["revision"] = "another-release"
    with pytest.raises(ValueError, match="revision"):
        validate()


@pytest.mark.parametrize("valid_reference", ["thisthat11"], indirect=True)
@pytest.mark.parametrize(
    "field", ["helper_sha256", "source_hashes", "environment", "versions"]
)
def test_thisthat_reused_reference_requires_current_evidence(valid_reference, field):
    report, validate, _, _ = valid_reference
    report[field] = {} if field != "helper_sha256" else "old-helper"
    with pytest.raises(ValueError, match="evidence|environment"):
        validate()


@pytest.mark.parametrize("profile", list(driver.PROFILES))
def test_reference_command_selects_shared_runner_and_explicit_release(
    tmp_path, profile
):
    from types import SimpleNamespace

    args = SimpleNamespace(
        profile=profile,
        reference_python=tmp_path / "author python.exe",
        source=tmp_path / "中文 source",
        sdk_source=tmp_path / "sdk",
        threads=4,
    )
    runner = driver.reference_runner(profile)
    command = driver.reference_command(
        args, runner, tmp_path / "cases", tmp_path / "out"
    )
    assert command[:3] == [str(args.reference_python), "-B", str(runner)]
    assert command[command.index("--source") + 1] == str(args.source)
    if profile in driver.THISTHAT_PROFILES:
        assert runner.name == "windows_thisthat_reference.py"
        assert command[command.index("--model") + 1] == driver.PROFILES[profile][0]
        assert driver.profile_family(profile) == "thisthat"
    else:
        assert "--model" not in command
        assert runner.name == f"windows_{profile}_reference.py"


@pytest.mark.parametrize("profile", sorted(driver.THISTHAT_PROFILES))
def test_reference_evidence_uses_selected_release_pins(tmp_path, monkeypatch, profile):
    from types import SimpleNamespace

    sdk = tmp_path / "sdk"
    sdk.mkdir()
    author = sdk / "author.py"
    author.write_text("# author\n", encoding="utf-8")
    models = {
        item[0]: {"revision": key, "source_hashes": {"weight": key}}
        for key, item in driver.PROFILES.items()
        if key in driver.THISTHAT_PROFILES
    }
    runner = tmp_path / "runner.py"
    runner.write_text(
        f"MODELS={models!r}\nSDK_REVISION='sdk-pin'\n"
        f"SDK_HASHES={{'author.py':{driver.common.digest(author)!r}}}\n"
        "def model_profile(model): return MODELS[model]\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        driver, "_reference_versions", lambda *a: {"torch": "2.8.0+cpu"}
    )
    monkeypatch.setattr(
        driver, "_reference_python_identity", lambda *a: {"python": "3.12"}
    )
    args = SimpleNamespace(
        profile=profile, sdk_source=sdk, reference_python=Path("python")
    )
    result = driver._reference_evidence(args, runner, {"weight": profile})
    assert result["model"] == driver.PROFILES[profile][0]
    assert result["revision"] == profile
    assert result["source_hashes"] == {"weight": profile}
    with pytest.raises(ValueError, match="release pins"):
        driver._reference_evidence(args, runner, {"weight": "different release"})


def test_installed_console_launch_has_no_module_or_pythonpath_fallback(
    tmp_path, monkeypatch
):
    scripts = tmp_path / "Scripts"
    scripts.mkdir()
    python, executable = scripts / "python.exe", scripts / "vllm-jev.exe"
    executable.write_bytes(b"console-entrypoint-fixture")
    monkeypatch.setattr(driver.sys, "executable", str(python))
    monkeypatch.setattr(
        driver,
        "_installed_entrypoint_identity",
        lambda _: {
            "python": str(python),
            "module": str(driver.ROOT / "vllm_jev/cli.py"),
            "entrypoints": {"vllm-jev": "vllm_jev.cli:main"},
        },
    )
    prefix, identity = driver.product_entrypoint(executable)
    assert prefix == [str(executable)]
    assert identity["mode"] == "installed_console"
    assert identity["executable_sha256"] == driver.common.digest(executable)
    assert (
        driver.product_environment({"PYTHONPATH": "wrong checkout"}, identity)[
            "PYTHONPATH"
        ]
        == ""
    )
    legacy, identity = driver.product_entrypoint()
    assert legacy == [str(python), "-B", "-m", "vllm_jev.cli"]
    assert identity["mode"] == "python_module"


@pytest.mark.parametrize(
    "failure", ["other_environment", "other_module", "other_target"]
)
def test_console_launch_rejects_an_unrelated_installation(
    tmp_path, monkeypatch, failure
):
    python = tmp_path / "python.exe"
    executable = tmp_path / "vllm-jev.exe"
    executable.write_bytes(b"fixture")
    monkeypatch.setattr(driver.sys, "executable", str(python))
    installed = {
        "python": str(python),
        "module": str(driver.ROOT / "vllm_jev/cli.py"),
        "entrypoints": {"vllm-jev": "vllm_jev.cli:main"},
    }
    if failure == "other_environment":
        installed["python"] = str(tmp_path / "other/python.exe")
    elif failure == "other_module":
        installed["module"] = str(tmp_path / "other/cli.py")
    else:
        installed["entrypoints"]["vllm-jev"] = "other:main"
    monkeypatch.setattr(driver, "_installed_entrypoint_identity", lambda _: installed)
    with pytest.raises(ValueError, match="this checkout"):
        driver.product_entrypoint(executable)


@pytest.mark.parametrize(
    "field,value",
    [
        ("revision", "1.2-stale-revision"),
        ("backend", "thisthat11-pytorch"),
        ("head_dtype", "bfloat16"),
    ],
)
def test_product_metadata_rejects_wrong_release_backend_or_precision(
    tmp_path, field, value
):
    identity = {
        "model": driver.PROFILES["thisthat11"][0],
        "revision": "1.1-revision",
        "backend": "thisthat-pytorch",
        "head_dtype": "float32",
    }
    response = {
        "model": identity["model"],
        "metadata": {
            **identity,
            "backbone_dtype": "float32",
            field: value,
        },
        "answers": {"q": {"type": "noul", "noul": 0.5}},
    }
    with pytest.raises(ValueError, match="metadata differs"):
        driver.record_and_compare_response(
            tmp_path / "report.json",
            {},
            [],
            status=200,
            response=response,
            seconds=0.1,
            expected=response["answers"],
            dtype="float32",
            identity=identity,
        )


def test_lite_cases_add_label_boundaries_without_changing_accepted_requests(tmp_path):
    # The SHA is the request artifact accepted with all three This-That releases.
    accepted_sha = "1645878620128e612e60be3fc4626e799c35830b780e5efafc8ed93de7ffac58"
    for profile in set(driver.PROFILES) - {"jevk5_lite"}:
        path = tmp_path / f"{profile}.json"
        driver.common.save(path, driver.cases_for(profile))
        assert driver.common.digest(path) == accepted_sha
    cases = driver.cases_for("jevk5_lite")
    assert len(cases) == 4
    assert len(cases[2]["questions"]["category"]["criteria"]) > 10
    for case in cases:
        for question in case["questions"].values():
            if question["type"] == "noul":
                assert "criteria" not in question
    criteria = cases[-1]["questions"]["category"]["criteria"]
    assert "" in criteria.values()
    assert len(set(criteria.values())) < len(criteria)
    score = cases[-1]["questions"]["repeated_score"]
    assert score["type"] == "score"
    assert score["criteria"] == ["same", "same", "different"]
    # Callers cannot contaminate historical cases or a later invocation.
    cases[0]["state"] = "modified"
    assert driver.cases_for("jevk5_lite")[0]["state"] != "modified"


@pytest.mark.parametrize("valid_reference", ["jevk5_lite"], indirect=True)
def test_lite_reference_accepts_only_selected_identity_and_complete_evidence(
    valid_reference,
):
    report, validate, _, _ = valid_reference
    assert validate() == report
    assert len(report["cases"]) == 4
    report["model_id"] = "alibiserikbay/JevK5-2B"
    with pytest.raises(ValueError, match="model identity"):
        validate()


@pytest.mark.parametrize("valid_reference", ["jevk5_lite"], indirect=True)
@pytest.mark.parametrize(
    "field,value",
    [
        ("dtype", "bfloat16"),
        ("parameter_devices", ["cuda:0"]),
        ("head_dtype", "torch.bfloat16"),
        ("row_batch_size", 2),
        ("padding", "right_to_multiple_of_64"),
        ("use_cache", True),
        ("max_sequence_tokens", 1024),
        ("temperature_single", 1.42),
        ("temperature_multi", 1.8660659830736146),
        ("truncation", True),
        ("packed_questions", True),
    ],
)
def test_lite_reference_rejects_decoder_defaults_and_altered_precision(
    valid_reference, field, value
):
    report, validate, _, _ = valid_reference
    report["runtime"][field] = value
    with pytest.raises(ValueError, match="precision|runtime setting"):
        validate()


@pytest.mark.parametrize("valid_reference", ["jevk5_lite"], indirect=True)
@pytest.mark.parametrize(
    "field",
    ["environment", "versions", "private_source_unchanged", "private_sdk_unchanged"],
)
def test_lite_reference_reuse_requires_unchanged_environment_and_private_inputs(
    valid_reference, field
):
    report, validate, _, _ = valid_reference
    report[field] = {} if field in ("environment", "versions") else False
    with pytest.raises(ValueError, match="environment|has not proved"):
        validate()


def test_lite_noul_true_is_first_and_score_is_expected_level():
    case = {
        "questions": {
            "n": {"type": "noul"},
            "s": {"type": "score", "criteria": ["low", "medium", "high"]},
        }
    }
    answers = driver.canonical(
        case,
        {"n": [0.8, 0.2], "s": [0.2, 0.3, 0.5]},
        driver.PROFILES["jevk5_lite"][2],
        "jevk5_lite",
    )
    assert answers["n"]["noul"] == 0.8
    assert answers["s"]["score"] == 1.3
    assert answers["s"]["legend"] == {"0": "low", "1": "medium", "2": "high"}


def test_lite_repeated_score_prefixes_all_author_labels_without_changing_other_profiles():
    case = {
        "questions": {"s": {"type": "score", "criteria": ["same", "same", "different"]}}
    }
    probabilities = {"s": [0.2, 0.3, 0.5]}
    result = driver.canonical(case, probabilities, 0, "jevk5_lite")["s"]
    assert result["score"] == 1.3
    assert result["legend"] == {
        "0": "0: same",
        "1": "1: same",
        "2": "2: different",
    }
    assert result["probabilities"] == {"0": 0.2, "1": 0.3, "2": 0.5}
    assert driver.canonical(case, probabilities, 1, "thisthat")["s"]["legend"] == {
        "0": "same",
        "1": "same",
        "2": "different",
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("revision", "JevK5-2B-revision"),
        ("backend", "jevk5_lite-pytorch"),
        ("head_dtype", "bfloat16"),
        ("softmax_backend", "numpy"),
        ("temperature", 1.42),
        ("threads", 2),
        ("max_sequence_tokens", 1024),
        ("truncated_state_tokens", 1),
        ("cache_reuse", True),
    ],
)
def test_lite_product_metadata_is_checked_before_numerical_pass(tmp_path, field, value):
    identity = driver.product_identity("jevk5_lite", "lite-revision", 4)
    response = {
        "model": "alibiserikbay/JevK5-Lite",
        "metadata": {**identity, "backbone_dtype": "float32", field: value},
        "answers": {"n": {"type": "noul", "noul": 0.8}},
    }
    with pytest.raises(ValueError, match="metadata differs"):
        driver.record_and_compare_response(
            tmp_path / "report.json",
            {},
            [],
            status=200,
            response=response,
            seconds=0.1,
            expected=response["answers"],
            dtype="float32",
            identity=identity,
        )


def test_lite_thread_budget_rejects_before_workspace_or_reference_start(
    tmp_path, monkeypatch
):
    workspace = tmp_path / "must-not-create"
    monkeypatch.setattr(driver.sys, "platform", "win32")
    monkeypatch.setattr(driver.os, "cpu_count", lambda: 18)
    monkeypatch.setattr(
        driver.sys,
        "argv",
        [
            "acceptance",
            "--profile",
            "jevk5_lite",
            "--source",
            str(tmp_path / "source"),
            "--sdk-source",
            str(tmp_path / "sdk"),
            "--reference-python",
            str(tmp_path / "python.exe"),
            "--workspace",
            str(workspace),
            "--output",
            str(tmp_path / "report.json"),
            "--threads",
            "5",
        ],
    )
    monkeypatch.setattr(
        driver, "hashes", lambda *a: pytest.fail("must not read weights")
    )
    with pytest.raises(SystemExit) as failure:
        driver.main()
    assert failure.value.code == 2
    assert not workspace.exists()
