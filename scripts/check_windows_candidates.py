"""Serial original-author / native Windows CLI acceptance for new text models.

Requires already downloaded, pinned sources and an isolated author environment.
Runs no installer and stops only process trees created by this driver.
"""

import argparse
import concurrent.futures
import copy
import importlib.util
import json
import math
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "_candidate_process_helpers", Path(__file__).with_name("check_windows_decider.py")
)
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)

PROFILES = {
    "jevk5": ("alibiserikbay/JevK5-2B", "bfloat16", 0, 6.0, 8.0),
    "jevk5_lite": ("alibiserikbay/JevK5-Lite", "float32", 0, 6.0, 8.0),
    "kev": ("jaredpalmer/kev-0.8b", "float32", 1, 5.0, 7.0),
    "kev06": ("jaredpalmer/kev-0.6b", "float32", 1, 5.0, 7.0),
    "thisthat": ("flock-io/this-that-model-1.2", "float32", 1, 9.0, 12.0),
    "thisthat11": ("flock-io/this-that-model-1.1", "float32", 1, 9.0, 12.0),
    "thisthat10": ("flock-io/this-that-model-1.0", "float32", 1, 9.0, 12.0),
}
THISTHAT_PROFILES = frozenset({"thisthat", "thisthat11", "thisthat10"})
CASES = common.CASES
TOLERANCE = 1e-5
LITE_TEMPERATURE = 1.8660659830736146


def cases_for(profile):
    """Keep accepted older request bytes stable while extending the Lite domain."""
    cases = copy.deepcopy(CASES)
    if profile == "jevk5_lite":
        for case in cases:
            for question in case["questions"].values():
                if question["type"] == "noul":
                    question.pop("criteria", None)
        cases.append(
            {
                "id": "empty_and_duplicate_choice",
                "state": "My parcel has not arrived. Please check the delivery status.",
                "questions": {
                    "category": {
                        "type": "choice",
                        "instructions": "Choose the issue that matches this request.",
                        "criteria": {
                            "delivery": "",
                            "billing": "Payment issue",
                            "duplicate_billing": "Payment issue",
                            "other": "Other request",
                        },
                    },
                    "repeated_score": {
                        "type": "score",
                        "instructions": "Rate the delivery issue using the ordered levels.",
                        "criteria": ["same", "same", "different"],
                    },
                },
            }
        )
    return cases


def profile_family(profile):
    return "thisthat" if profile in THISTHAT_PROFILES else profile


def backend_for(profile):
    if profile == "kev06":
        return "kev-pytorch"
    return (
        "jevk5-lite-pytorch"
        if profile == "jevk5_lite"
        else f"{profile_family(profile)}-pytorch"
    )


def reference_runner(profile):
    return Path(__file__).with_name(f"windows_{profile_family(profile)}_reference.py")


def reference_command(args, runner, cases_path, output):
    command = [str(args.reference_python), "-B", str(runner)]
    if args.profile in THISTHAT_PROFILES:
        command += ["--model", PROFILES[args.profile][0]]
    return command + [
        "--source",
        str(args.source),
        "--sdk-source",
        str(args.sdk_source),
        "--cases-json",
        str(cases_path),
        "--output",
        str(output),
        "--threads",
        str(args.threads),
    ]


def _installed_entrypoint_identity(python):
    code = (
        "import importlib.metadata,importlib.util,json,sys; "
        "d=importlib.metadata.distribution('vllm-jev'); "
        "print(json.dumps({'python':sys.executable,"
        "'module':importlib.util.find_spec('vllm_jev.cli').origin,"
        "'entrypoints':{e.name:e.value for e in d.entry_points "
        "if e.group=='console_scripts'}}))"
    )
    result = subprocess.run(
        [str(python), "-I", "-c", code],
        text=True,
        capture_output=True,
        check=True,
        timeout=30,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    return _strict_reference_json(result.stdout)


def product_entrypoint(console_entrypoint=None):
    """Distinguish an installed console launch from the legacy module launch."""
    python = Path(sys.executable).resolve()
    module = ROOT / "vllm_jev/cli.py"
    identity = {
        "mode": "python_module",
        "python": str(python),
        "module": str(module),
        "module_sha256": common.digest(module),
    }
    if console_entrypoint is None:
        return [str(python), "-B", "-m", "vllm_jev.cli"], identity
    entrypoint = Path(console_entrypoint).resolve()
    if (
        not entrypoint.is_file()
        or entrypoint.name.lower() != "vllm-jev.exe"
        or entrypoint.parent != python.parent
    ):
        raise ValueError("console entrypoint must be vllm-jev.exe beside this Python")
    installed = _installed_entrypoint_identity(python)
    if (
        Path(installed["python"]).resolve() != python
        or Path(installed["module"]).resolve() != module.resolve()
        or installed["entrypoints"].get("vllm-jev") != "vllm_jev.cli:main"
    ):
        raise ValueError("installed console entrypoint does not target this checkout")
    identity.update(
        mode="installed_console",
        executable=str(entrypoint),
        executable_sha256=common.digest(entrypoint),
        installed_entrypoint="vllm_jev.cli:main",
    )
    return [str(entrypoint)], identity


def product_environment(environment, entrypoint):
    # An installed-entrypoint acceptance must resolve the actual installed
    # distribution; a PYTHONPATH override would silently validate another copy.
    return {
        **environment,
        "PYTHONPATH": "" if entrypoint["mode"] == "installed_console" else str(ROOT),
    }


def canonical(case, probabilities, true_index, profile=None):
    """Independent documented API formulas; never call product answer code."""
    if set(probabilities) != set(case["questions"]):
        raise ValueError("reference question IDs differ")
    answers = {}
    for key, question in case["questions"].items():
        kind, criteria = question["type"], question.get("criteria")
        count = 2 if kind == "noul" else len(criteria)
        p = probabilities[key]
        if not isinstance(p, list) or len(p) != count:
            raise ValueError("reference probability count differs")
        if any(
            type(x) not in (int, float) or not math.isfinite(x) or not 0 <= x <= 1
            for x in p
        ):
            raise ValueError("reference probabilities invalid")
        if not math.isclose(sum(p), 1, abs_tol=1e-6):
            raise ValueError("reference probability mass differs")
        if kind == "noul":
            answers[key] = {"type": kind, "noul": p[true_index]}
            continue
        selected = max(range(count), key=p.__getitem__)
        keys = list(criteria) if kind == "choice" else [str(i) for i in range(count)]
        result = {"type": kind, "probabilities": dict(zip(keys, p))}
        if kind == "choice":
            result.update(
                choice=keys[selected],
                confidence=1.0
                if count == 1
                else max(0, (count * p[selected] - 1) / (count - 1)),
            )
        else:
            distance = sum(abs(i - (count - 1) / 2) for i in range(count)) / count
            result.update(
                score=sum(i * x for i, x in enumerate(p)),
                confidence=1.0
                if count == 1
                else max(
                    0,
                    1 - sum(x * abs(i - selected) for i, x in enumerate(p)) / distance,
                ),
                legend=dict(
                    zip(
                        keys,
                        [f"{i}: {value}" for i, value in enumerate(criteria)]
                        if profile == "jevk5"
                        or (profile == "jevk5_lite" and len(set(criteria)) < count)
                        else criteria,
                    )
                ),
            )
        answers[key] = result
    return answers


def hashes(source):
    return {
        p.relative_to(source).as_posix(): common.digest(p)
        for p in source.rglob("*")
        if p.is_file() and ".cache" not in p.relative_to(source).parts
    }


def clone(source, destination, expected):
    destination.mkdir(parents=True)
    for name in expected:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if (source / name).stat().st_size > 32 * 1024**2:
            os.link(source / name, target)
        else:
            shutil.copyfile(source / name, target)
    if hashes(destination) != expected:
        raise RuntimeError("source clone checksum differs")


def _strict_reference_json(text):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate reference JSON key: {key}")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(f"nonfinite reference JSON: {value}")

    return json.loads(text, object_pairs_hook=unique, parse_constant=invalid)


def _ordered_json(value):
    # Both option order and JSON-state field order affect the original prompts.
    # Plain dict equality would accept reordered questions/options or bool == 1.
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _reference_versions(python, packages):
    """Inspect the selected author environment without importing any backend."""
    code = (
        "import importlib.metadata,json,sys; "
        "print(json.dumps({n:importlib.metadata.version(n) for n in sys.argv[1:]}))"
    )
    result = subprocess.run(
        [str(python), "-I", "-c", code, *packages],
        text=True,
        capture_output=True,
        check=True,
        timeout=30,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    return _strict_reference_json(result.stdout)


def _reference_python_identity(python):
    result = subprocess.run(
        [
            str(python),
            "-I",
            "-c",
            "import json,sys; print(json.dumps({'python':sys.version,'executable':sys.executable}))",
        ],
        text=True,
        capture_output=True,
        check=True,
        timeout=30,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    return _strict_reference_json(result.stdout)


def _reference_evidence(args, runner, source_hashes):
    """Read independent runner pins and current bytes, never product inference."""
    definition = importlib.util.spec_from_file_location(
        "_candidate_reference_pins", runner
    )
    reference = importlib.util.module_from_spec(definition)
    definition.loader.exec_module(reference)
    if args.profile in THISTHAT_PROFILES:
        model = PROFILES[args.profile][0]
        selected = reference.model_profile(model)
        revision, expected_source = selected["revision"], selected["source_hashes"]
    else:
        model, revision = reference.MODEL_ID, reference.REVISION
        expected_source = reference.SOURCE_HASHES
    if source_hashes != expected_source:
        raise ValueError("current source files do not match independent release pins")
    sdk = {name: common.digest(args.sdk_source / name) for name in reference.SDK_HASHES}
    if sdk != reference.SDK_HASHES:
        raise ValueError("current author SDK files do not match fixed pins")
    packages = ["torch", "transformers", "accelerate", "huggingface_hub"]
    packages.append("peft" if args.profile in ("kev", "kev06") else "numpy")
    if args.profile in THISTHAT_PROFILES or args.profile == "jevk5_lite":
        packages += ["safetensors", "tokenizers"]
    evidence = {
        "model": model,
        "revision": revision,
        "sdk_revision": getattr(
            reference, "SDK_REVISION", getattr(reference, "SDK_COMMIT", None)
        ),
        "source_hashes": expected_source,
        "sdk_hashes": sdk,
        "driver_sha256": common.digest(runner),
        "versions": _reference_versions(args.reference_python, packages),
    }
    if hasattr(reference, "BASE_REVISION"):
        evidence["base_revision"] = reference.BASE_REVISION
    if hasattr(reference, "common"):
        evidence["helper_sha256"] = common.digest(Path(reference.common.__file__))
    if args.profile in THISTHAT_PROFILES or args.profile == "jevk5_lite":
        evidence["environment_identity"] = _reference_python_identity(
            args.reference_python
        )
    return evidence


def _validate_reference_runtime(value, profile, threads):
    runtime = value.get("runtime")
    if (
        not isinstance(runtime, dict)
        or type(runtime.get("threads")) is not int
        or runtime["threads"] != threads
    ):
        raise ValueError("reference CPU thread configuration differs")
    dtype = PROFILES[profile][1]
    devices = runtime.get("parameter_devices", runtime.get("actual_parameter_devices"))
    dtypes = runtime.get("parameter_dtypes", runtime.get("actual_parameter_dtypes"))
    head = runtime.get("head_dtype", runtime.get("actual_head_dtype"))
    if devices != ["cpu"] or dtypes != ["torch." + dtype] or head != "torch." + dtype:
        raise ValueError("reference CPU parameter/head precision differs")
    if profile == "jevk5_lite":
        expected = {
            "dtype": "float32",
            "row_batch_size": 1,
            "padding": "none",
            "use_cache": False,
            "graphs": False,
            "max_sequence_tokens": 512,
            "temperature_single": LITE_TEMPERATURE,
            "temperature_multi": 1.319507910772894,
            "truncation": False,
            "packed_questions": False,
        }
    elif profile in THISTHAT_PROFILES:
        expected = {
            "actual_embedding_dtype": "torch.float32",
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
    else:
        expected = {
            "dtype": dtype,
            "row_batch_size": 1,
            "padding": "none",
            "use_cache": False,
            "graphs": False,
            "max_sequence_tokens": 1024,
        }
        if profile == "jevk5":
            expected.update(temperature=1.42, head_shape=[16, 2048])
        else:
            expected.update(use_cache=True, cache_reuse=False)
            if profile == "kev06":
                expected.update(
                    packed_questions=True,
                    temperature=1.0,
                    attention_implementation="eager",
                )
    for name, expected_value in expected.items():
        if _ordered_json(runtime.get(name)) != _ordered_json(expected_value):
            raise ValueError(f"reference runtime setting differs: {name}")


def validate_reference(path, cases, *, args, runner, cases_path, source_hashes):
    """Reject stale, partial or mismatched baselines before any product CLI runs."""
    value = _strict_reference_json(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("status") != "passed":
        raise ValueError("reference did not pass")
    for name in ("measured", "source_unchanged", "sdk_unchanged"):
        if value.get(name) is not True:
            raise ValueError(f"reference has not proved {name}")
    if args.profile == "jevk5_lite":
        for name in ("private_source_unchanged", "private_sdk_unchanged"):
            if value.get(name) is not True:
                raise ValueError(f"reference has not proved {name}")
    evidence = _reference_evidence(args, runner, source_hashes)
    model = PROFILES[args.profile][0]
    if evidence["model"] != model or value.get("model_id", value.get("model")) != model:
        raise ValueError("reference model identity differs")
    if "model" in value and value["model"] != model:
        raise ValueError("reference contains conflicting model identity")
    for name in (
        "revision",
        "driver_sha256",
        "source_hashes",
        "sdk_hashes",
        "base_revision",
        "helper_sha256",
    ):
        if name in evidence and value.get(name) != evidence[name]:
            raise ValueError(f"reference evidence differs: {name}")
    if value.get("sdk_revision", value.get("sdk_commit")) != evidence["sdk_revision"]:
        raise ValueError("reference SDK revision differs")
    if "sdk_commit" in value and value["sdk_commit"] != evidence["sdk_revision"]:
        raise ValueError("reference contains conflicting SDK identity")
    if value.get("input_sha256") != common.digest(cases_path):
        raise ValueError("reference request-file SHA256 differs")
    if _ordered_json(
        _strict_reference_json(cases_path.read_text(encoding="utf-8"))
    ) != _ordered_json(cases):
        raise ValueError("reference request file differs from selected cases")
    recorded_versions = value.get("versions")
    if not isinstance(recorded_versions, dict) or any(
        recorded_versions.get(name) != version
        for name, version in evidence["versions"].items()
    ):
        raise ValueError("reference environment versions differ")
    if "environment_identity" in evidence:
        recorded_environment = value.get("environment", {})
        if any(
            recorded_environment.get(name) != expected_value
            for name, expected_value in evidence["environment_identity"].items()
        ):
            raise ValueError("reference Python environment identity differs")
    rows = value.get("cases")
    if not isinstance(rows, list) or len(rows) != len(cases):
        raise ValueError("reference case count differs")
    for row, case in zip(rows, cases):
        if not isinstance(row, dict) or any(name not in row for name in case):
            raise ValueError("reference is missing original case content")
        if _ordered_json({name: row[name] for name in case}) != _ordered_json(case):
            raise ValueError("reference case content or order differs")
        canonical(case, row.get("probabilities", {}), PROFILES[args.profile][2])
    if "selected_case_ids" in value and value["selected_case_ids"] != [
        case["id"] for case in cases
    ]:
        raise ValueError("reference selected case IDs differ")
    _validate_reference_runtime(value, args.profile, args.threads)
    return value


@contextmanager
def private_workspace(parent, record):
    directory = Path(tempfile.mkdtemp(prefix="candidate-cli-", dir=parent)).resolve()
    if directory.parent != parent.resolve() or not directory.name.startswith(
        "candidate-cli-"
    ):
        raise RuntimeError("private workspace escaped acceptance root")
    try:
        yield directory
    finally:
        if record.get("owned_process_stopped") is False or (
            record.get("owned_process_stopped") is not True and "launcher_pid" in record
        ):
            record["retained_workspace"] = str(directory)
            record["private_workspace_removed"] = False
        else:
            shutil.rmtree(directory)
            record["private_workspace_removed"] = not directory.exists()


def child_run(args, command, environment, log, record, action):
    gate = common.memory_gate(args)
    record["memory_gate"] = gate
    if not gate["passed"]:
        raise RuntimeError("insufficient memory before startup; no model started")
    tree = monitor = None
    with log.open("w", encoding="utf-8") as stream:
        try:
            process = subprocess.Popen(
                command,
                cwd=args.workspace,
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            record.update(launcher_pid=process.pid, owned_process_stopped=False)
            tree = common.OwnedProcessTree(process, record)
            monitor = common.MemoryMonitor(tree, args, record)
            monitor.start()
            return action(process, tree)
        finally:
            okay = True
            try:
                if monitor:
                    okay = monitor.stop()
            except BaseException as error:
                okay = False
                record["monitor_cleanup_error"] = str(error)
            finally:
                stopped = tree.stop() if tree else "launcher_pid" not in record
                # A tree-constructor failure must still stop our known Popen.
                if tree is None and "launcher_pid" in record:
                    process.terminate()
                    process.wait(timeout=15)
                    record["owned_process_stopped"] = False
            if not okay or not stopped:
                raise RuntimeError("owned process cleanup or memory guard failed")


def reference_run(args, command, environment, log, record, action):
    """The parent owns reference scratch files even after a forced child exit."""
    with private_workspace(args.workspace, record) as directory:
        record["workspace"] = str(directory)
        temporary = directory / "tmp"
        temporary.mkdir()
        isolated_args = argparse.Namespace(**vars(args))
        isolated_args.workspace = directory
        isolated_environment = {
            **environment,
            "HF_HOME": str(directory / "hf-home"),
            "HF_HUB_CACHE": str(directory / "hf-home/hub"),
            "TEMP": str(temporary),
            "TMP": str(temporary),
            "TMPDIR": str(temporary),
        }
        return child_run(
            isolated_args, command, isolated_environment, log, record, action
        )


def record_and_compare_response(
    output,
    report,
    observations,
    *,
    status,
    response,
    seconds,
    expected,
    dtype,
    case_id=None,
    repeated_answers=None,
    identity=None,
):
    """Keep the actual failed response before numerical/identity assertions."""
    observation = {"status": status, "seconds": seconds, "response": response}
    if case_id is not None:
        observation["id"] = case_id
    observations.append(observation)
    common.save(output, report)
    assert status == 200, response
    assert response["metadata"]["backbone_dtype"] == dtype, response
    if identity is not None:
        if response.get("model") != identity["model"]:
            raise ValueError("product response model identity differs")
        for name, value in identity.items():
            if _ordered_json(response["metadata"].get(name)) != _ordered_json(value):
                raise ValueError(f"product response metadata differs: {name}")
    difference = common.compare_answers(expected, response["answers"], TOLERANCE)
    observation["max_absolute_difference"] = difference
    if repeated_answers is not None:
        common.compare_answers(repeated_answers, response["answers"], 0)
    return observation


def product_identity(profile, revision, threads):
    if profile == "kev06":
        return {
            "model": PROFILES[profile][0],
            "revision": revision,
            "backend": "kev-pytorch",
            "device": "cpu",
            "experimental": True,
            "backbone_dtype": "float32",
            "head_dtype": "float32",
            "logits_dtype": "float32",
            "temperature": 1.0,
            "use_cache": True,
            "cache_reuse": False,
            "attention_implementation": "eager",
            "lora_merged": True,
            "row_batch_size": 1,
            "max_sequence_tokens": 1024,
        }
    if profile == "jevk5_lite":
        return {
            "model": PROFILES[profile][0],
            "revision": revision,
            "backend": backend_for(profile),
            "device": "cpu",
            "experimental": True,
            "head_dtype": "float32",
            "readout_dtype": "float32",
            "logits_dtype": "float32",
            "softmax_dtype": "float32",
            "softmax_backend": "torch",
            "temperature": LITE_TEMPERATURE,
            "threads": threads,
            "row_batch_size": 1,
            "padding_multiple": 1,
            "padding": "none",
            "packed_questions": False,
            "max_sequence_tokens": 512,
            "truncated_state_tokens": 0,
            "cache_reuse": False,
            "graphs": False,
            "parameter_devices": ["cpu"],
            "parameter_dtypes": ["torch.float32"],
        }
    if profile in THISTHAT_PROFILES:
        return {
            "model": PROFILES[profile][0],
            "revision": revision,
            "backend": "thisthat-pytorch",
            "device": "cpu",
            "experimental": True,
            "head_dtype": "float32",
            "readout_dtype": "float32",
            "logits_dtype": "float32",
            "softmax_dtype": "float32",
            "softmax_backend": "numpy",
            "row_batch_size": 1,
            "padding_multiple": 64,
            "max_sequence_tokens": 1024,
            "packed_questions": True,
            "temperature": 1.0,
        }
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, choices=PROFILES)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--sdk-source", required=True, type=Path)
    parser.add_argument("--reference-python", required=True, type=Path)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reference-only", action="store_true")
    parser.add_argument("--reference-report", type=Path)
    parser.add_argument(
        "--console-entrypoint",
        type=Path,
        help="Installed vllm-jev.exe beside this Python; default retains python -m",
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--port", type=int, default=18818)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--abort-below-available-gib", type=float, default=0.75)
    args = parser.parse_args()
    model, dtype, true_index, args.min_available_gib, args.min_commit_available_gib = (
        PROFILES[args.profile]
    )
    cases = cases_for(args.profile)
    if sys.platform != "win32":
        parser.error("native Windows is required")
    maximum_threads = (
        min(4, os.cpu_count() or 1)
        if args.profile == "jevk5_lite"
        else (os.cpu_count() or 1)
    )
    if not 1 <= args.threads <= maximum_threads or not 1 <= args.port <= 65535:
        parser.error("invalid threads/port")
    for field in ("timeout", "abort_below_available_gib"):
        if not math.isfinite(getattr(args, field)) or getattr(args, field) <= 0:
            parser.error(f"{field} must be positive and finite")
    for field in (
        "source",
        "sdk_source",
        "reference_python",
        "workspace",
        "output",
        "reference_report",
        "console_entrypoint",
    ):
        if getattr(args, field) is not None:
            setattr(args, field, getattr(args, field).resolve())
    if args.reference_report == args.output:
        parser.error("output cannot replace reference")
    if any(
        args.workspace.is_relative_to(root) for root in (args.source, args.sdk_source)
    ):
        parser.error("validation workspace must be outside source and SDK")
    outputs = [
        args.output,
        *[
            args.output.with_name(args.output.stem + suffix)
            for suffix in (
                "-author.json",
                "-cases.json",
                "-author.log",
                "-server-1.log",
                "-server-2.log",
            )
        ],
    ]
    for path in outputs:
        if any(
            path.is_relative_to(root) for root in (args.source, args.sdk_source)
        ) or path in (args.reference_python, args.console_entrypoint):
            parser.error(
                "acceptance output must not overwrite source, SDK, or interpreter"
            )
    args.workspace.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "running",
        "measured": False,
        "model": model,
        "profile": args.profile,
        "dtype": dtype,
        "threads": args.threads,
        "source": str(args.source),
        "tolerance": TOLERANCE,
        "driver_sha256": common.digest(Path(__file__)),
        "environment": common.common.environment(),
        "service_runs": [],
        "scope": (
            "Four synthetic cases including empty/repeated Choice descriptions, original raw probabilities, independent project presentation, two fresh CLI routes. Not accuracy/throughput evaluation."
            if args.profile == "jevk5_lite"
            else "Three synthetic cases, original raw probabilities, independent project presentation, two fresh CLI routes. Not accuracy/throughput evaluation."
        ),
    }
    before = None
    try:
        with socket.socket() as s:
            s.bind(("127.0.0.1", args.port))
        before = hashes(args.source)
        report["source_sha256"] = before
        env = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith("JEV_") and k != "VLLM_JEV_DECISION_TEMPLATE"
        }
        env.update(
            PYTHONDONTWRITEBYTECODE="1",
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            HF_HOME=str(args.workspace / "hf-home"),
            OMP_NUM_THREADS=str(args.threads),
            MKL_NUM_THREADS=str(args.threads),
            PYTHONIOENCODING="utf-8",
            PYTHONPATH="",
        )
        reference_path = args.reference_report or args.output.with_name(
            args.output.stem + "-author.json"
        )
        cases_path = args.output.with_name(args.output.stem + "-cases.json")
        common.save(cases_path, cases)
        runner = reference_runner(args.profile)
        report["reference_runner_sha256"] = common.digest(runner)
        if not args.reference_report:
            command = reference_command(args, runner, cases_path, reference_path)
            report["reference_process"] = {"command": command}

            def reference_action(process, tree):
                if tree.wait(args.timeout) != 0:
                    raise RuntimeError("original reference failed; inspect author log")

            reference_run(
                args,
                command,
                env,
                args.output.with_name(args.output.stem + "-author.log"),
                report["reference_process"],
                reference_action,
            )
        reference = validate_reference(
            reference_path,
            cases,
            args=args,
            runner=runner,
            cases_path=cases_path,
            source_hashes=before,
        )
        report["reference_report"] = str(reference_path)
        report["reference_sha256"] = common.digest(reference_path)
        report["reference_reused"] = bool(args.reference_report)
        report["revision"] = reference["revision"]
        expected = [
            canonical(case, row["probabilities"], true_index, args.profile)
            for case, row in zip(cases, reference["cases"])
        ]
        if not args.reference_only:
            prefix, entrypoint = product_entrypoint(args.console_entrypoint)
            report["product_entrypoint"] = entrypoint
            identity = product_identity(
                args.profile, reference["revision"], args.threads
            )
            for number in (1, 2):
                record = {
                    "route": "model_id" if number == 1 else "unicode_local_path",
                    "cases": [],
                    "memory": [],
                }
                report["service_runs"].append(record)
                with private_workspace(args.workspace, record) as directory:
                    workspace = Path(directory)
                    source = (
                        workspace / "checkpoint" / model
                        if number == 1
                        else workspace / "本地 模型" / model.split("/")[-1]
                    )
                    clone(args.source, source, before)
                    command = [
                        *prefix,
                        "serve",
                        model if number == 1 else str(source),
                        "--workspace",
                        str(workspace),
                        "--cpu-threads",
                        str(args.threads),
                        "--host",
                        "127.0.0.1",
                        "--port",
                        str(args.port),
                    ]
                    record["command"] = command
                    record["working_directory"] = str(workspace)
                    service_environment = product_environment(env, entrypoint)
                    record["pythonpath"] = service_environment["PYTHONPATH"]
                    started = time.perf_counter()

                    def service_action(process, tree):
                        url = f"http://127.0.0.1:{args.port}"
                        deadline = time.monotonic() + 600
                        while time.monotonic() < deadline:
                            if process.poll() is not None:
                                raise RuntimeError(
                                    "CLI exited before ready; inspect server log"
                                )
                            try:
                                status, health, _ = common.http(
                                    url, "/health", timeout=1
                                )
                                if status == 200:
                                    break
                            except (OSError, ValueError):
                                pass
                            time.sleep(0.2)
                        else:
                            raise TimeoutError("server startup timeout")
                        owned = tree.listener(args.port)
                        assert health["model"] == model and health["device"] == "cpu"
                        assert (
                            health["backend"] == backend_for(args.profile)
                            and health["experimental"] is True
                        )
                        if identity is not None:
                            assert health["revision"] == reference["revision"]
                        record.update(
                            health=health, ready_seconds=time.perf_counter() - started
                        )
                        status, models, _ = common.http(url, "/v1/models")
                        assert status == 200 and models["data"][0]["id"] == model
                        record["memory"].append(
                            common.memory_sample(owned, "ready", started)
                        )
                        for i, case in enumerate(cases):
                            payload = {
                                "model": model,
                                "state": case["state"],
                                "questions": case["questions"],
                            }
                            status, response, elapsed = common.http(
                                url, "/v1/systemone", payload, 600
                            )
                            record_and_compare_response(
                                args.output,
                                report,
                                record["cases"],
                                status=status,
                                response=response,
                                seconds=elapsed,
                                expected=expected[i],
                                dtype=dtype,
                                identity=identity,
                                case_id=case["id"],
                                repeated_answers=(
                                    report["service_runs"][0]["cases"][i]["response"][
                                        "answers"
                                    ]
                                    if number == 2
                                    else None
                                ),
                            )
                            record["memory"].append(
                                common.memory_sample(owned, case["id"], started)
                            )
                            common.save(args.output, report)
                            print(
                                json.dumps(
                                    {
                                        "event": "case_completed",
                                        "profile": args.profile,
                                        "round": number,
                                        "case": case["id"],
                                        "seconds": elapsed,
                                    }
                                ),
                                flush=True,
                            )
                        base = {
                            "model": model,
                            "state": cases[0]["state"],
                            "questions": cases[0]["questions"],
                        }
                        invalid = {
                            "cache_salt": {**base, "cache_salt": "x"},
                            "images": {
                                **base,
                                "images": ["data:image/png;base64,AA=="],
                            },
                            "videos": {
                                **base,
                                "videos": ["data:video/mp4;base64,AA=="],
                            },
                            "model": {**base, "model": "unloaded/model"},
                            "empty": {**base, "questions": {}},
                            "type": {
                                **base,
                                "questions": {
                                    "q": {"type": "bad", "instructions": "x"}
                                },
                            },
                            "overlength": {**base, "state": "token " * 3000},
                        }
                        if args.profile == "jevk5_lite":
                            invalid["noul_criteria"] = {
                                **base,
                                "questions": {
                                    "q": {
                                        "type": "noul",
                                        "instructions": "Does the request ask for a refund?",
                                        "criteria": {
                                            "true": "Custom yes",
                                            "false": "Custom no",
                                        },
                                    }
                                },
                            }
                        record["rejections"] = {}
                        for key, payload in invalid.items():
                            status, response, _ = common.http(
                                url, "/v1/systemone", payload, 30
                            )
                            record["rejections"][key] = {
                                "status": status,
                                "response": response,
                            }
                            assert status == 422, (key, response)
                        with concurrent.futures.ThreadPoolExecutor(
                            max_workers=2
                        ) as pool:
                            futures = [
                                pool.submit(
                                    common.http, url, "/v1/systemone", base, 600
                                )
                                for _ in range(2)
                            ]
                            time.sleep(0.1)
                            unfinished = sum(not f.done() for f in futures)
                            status, _, elapsed = common.http(url, "/health", timeout=5)
                            after = sum(not f.done() for f in futures)
                            record["concurrent_health"] = {
                                "status": status,
                                "seconds": elapsed,
                                "unfinished_before": unfinished,
                                "unfinished_after": after,
                            }
                            assert status == 200 and unfinished and after
                            record["concurrent_requests"] = []
                            for f in futures:
                                status, response, elapsed = f.result()
                                record_and_compare_response(
                                    args.output,
                                    report,
                                    record["concurrent_requests"],
                                    status=status,
                                    response=response,
                                    seconds=elapsed,
                                    expected=expected[0],
                                    dtype=dtype,
                                    identity=identity,
                                    repeated_answers=record["cases"][0]["response"][
                                        "answers"
                                    ],
                                )
                        record["memory"].append(
                            common.memory_sample(owned, "completed", started)
                        )

                    isolated_args = argparse.Namespace(**vars(args))
                    isolated_args.workspace = workspace
                    child_run(
                        isolated_args,
                        command,
                        service_environment,
                        args.output.with_name(
                            args.output.stem + f"-server-{number}.log"
                        ),
                        record,
                        service_action,
                    )
                    assert hashes(source) == before
                    record["source_unchanged"] = True
                record.update(
                    status="passed", private_workspace_removed=not workspace.exists()
                )
        report.update(status="passed", measured=True)
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if before is not None:
            report["source_unchanged"] = hashes(args.source) == before
            if not report["source_unchanged"]:
                report["status"] = "failed"
        common.save(args.output, report)


if __name__ == "__main__":
    main()
