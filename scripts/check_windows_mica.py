"""Mica Q5: independent author reference, then two installed Windows CLI runs.

No downloads or installation. All model processes run serially. Only owned
process trees are stopped by the 1.5 GiB available-memory watchdog.
"""

import argparse
import concurrent.futures
import importlib.util
import json
import math
import os
import socket
import subprocess
import sys
import time
from pathlib import Path


def load_script(name, filename):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).with_name(filename)
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reference = load_script("_mica_independent_reference", "windows_mica_reference.py")
helpers = load_script("_mica_process_helpers", "check_windows_candidates.py")
common = helpers.common
ROOT = Path(__file__).resolve().parents[1]
MODEL = reference.MODEL_ID
BACKEND = "mica-llama-cpp-q5-b11010"
TOLERANCE = 1e-5
# The unmodified author requests max(n_ctx, 2048), but b11010 clamps the
# effective batch to this profile's 1024-token context (llama_n_batch).
REQUESTED_N_BATCH = 2048
EFFECTIVE_N_BATCH = 1024
CASES = common.CASES
RUNNER = Path(__file__).with_name("windows_mica_reference.py")
HELPERS = (
    "check_windows_candidates.py",
    "check_windows_decider.py",
    "check_windows_open_jev.py",
    "windows_decider_reference.py",
)
METADATA = {
    "tokenizer_runtime": "tokenizers-jinja2",
    "tokenizers_version": "0.23.3",
    "jinja2_version": "3.1.6",
    "torch_imported": False,
    "transformers_imported": False,
    "backend": BACKEND,
    "device": "cpu",
    "experimental": True,
    "revision": reference.REVISION,
    "quantization": "Q5_K_M",
    "runtime": "llama.cpp",
    "runtime_revision": reference.RUNTIME_REVISION,
    "n_gpu_layers": 0,
    "context_tokens": 1024,
    "n_seq_max": 1,
    "n_batch": EFFECTIVE_N_BATCH,
    "row_batch_size": 1,
    "n_ubatch": 512,
    "flash_attn": -1,
    "prefix": False,
    "session": None,
    "cache_reuse": False,
    "weight_format": "GGUF",
    "logits_dtype": "float32",
    "softmax_dtype": "float64",
    "max_sequence_tokens": 1024,
}


def canonical(case, probabilities):
    return helpers.canonical(case, probabilities, 1)


def command_for_reference(args, cases, output):
    return [
        str(args.reference_python),
        "-B",
        str(RUNNER),
        "--source",
        str(args.source),
        "--sdk-source",
        str(args.sdk_source),
        "--cases-json",
        str(cases),
        "--output",
        str(output),
        "--threads",
        str(args.threads),
    ]


def validate_runtime(runtime, threads):
    expected = {
        "tokenizer_runtime": "transformers-tokenizer-only",
        "torch_imported": False,
        "torch_installed": False,
        "runtime": "llama.cpp",
        "runtime_revision": "b11010",
        "quantization": "Q5_K_M",
        "device": "cpu",
        "n_gpu_layers": 0,
        "threads": threads,
        "context_tokens": 1024,
        "n_seq_max": 1,
        "n_batch": EFFECTIVE_N_BATCH,
        "n_ubatch": 512,
        "flash_attn": -1,
        "max_sequence_tokens": 1024,
        "row_batch_size": 1,
        "share_prefix": False,
        "session_id": None,
        "temperature": reference.TEMPERATURE,
        "temperatures": {
            kind: reference.TEMPERATURE for kind in ("noul", "choice", "score")
        },
        "noul_logit_bias": 0.0,
    }
    if not isinstance(runtime, dict):
        raise ValueError("reference runtime is missing")
    for name, value in expected.items():
        if reference.ordered(runtime.get(name)) != reference.ordered(value):
            raise ValueError(f"reference runtime differs: {name}")
    if (
        type(runtime.get("backend_registration_count")) is not int
        or runtime["backend_registration_count"] < 1
    ):
        raise ValueError("reference did not observe a registered backend")


def reference_versions(python):
    code = (
        "import importlib.metadata as m,json,sys\n"
        "versions={}\n"
        "for name in sys.argv[1:]:\n"
        " try: versions[name]=m.version(name)\n"
        " except m.PackageNotFoundError: versions[name]='not-installed'\n"
        "print(json.dumps(versions))\n"
    )
    result = subprocess.run(
        [str(python), "-I", "-c", code, *reference.PACKAGES],
        text=True,
        capture_output=True,
        check=True,
        timeout=30,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    versions = reference.strict_json(result.stdout)
    if versions.get("torch") != "not-installed":
        raise ValueError("author reference environment must not install Torch")
    return versions


def validate_reference(path, args, cases_path, source_hashes):
    value = reference.strict_json(path.read_text(encoding="utf-8"))
    expected = {
        "status": "passed",
        "measured": True,
        "model_id": MODEL,
        "revision": reference.REVISION,
        "sdk_revision": reference.SDK_REVISION,
        "runtime_revision": reference.RUNTIME_REVISION,
        "source_hashes": source_hashes,
        "sdk_hashes": reference.verify_sdk(args.sdk_source),
        "driver_sha256": reference.digest(RUNNER),
        "input_sha256": reference.digest(cases_path),
        "versions": reference_versions(args.reference_python),
        "environment_identity": helpers._reference_python_identity(
            args.reference_python
        ),
        "source_unchanged": True,
        "sdk_unchanged": True,
        "isolated_source_unchanged": True,
        "isolated_sdk_unchanged": True,
        "judge_closed": True,
        "private_workspace_removed": True,
        "runtime_source_isolated": False,
        "isolated_source_hashes": {
            name: checksum
            for name, checksum in reference.SOURCE_HASHES.items()
            if not name.startswith("runtime/")
        },
    }
    if source_hashes != reference.SOURCE_HASHES:
        raise ValueError("source differs from the independent release pins")
    for name, expected_value in expected.items():
        if reference.ordered(value.get(name)) != reference.ordered(expected_value):
            raise ValueError(f"reference provenance differs: {name}")
    validate_runtime(value.get("runtime"), args.threads)
    rows = value.get("cases")
    if not isinstance(rows, list) or len(rows) != len(CASES):
        raise ValueError("reference cases are incomplete")
    for case, row in zip(CASES, rows):
        if reference.ordered(
            {name: row.get(name) for name in case}
        ) != reference.ordered(case):
            raise ValueError("reference case content/order differs")
        canonical(case, row.get("probabilities", {}))
        if set(row.get("raw_logits", {})) != set(case["questions"]) or set(
            row.get("evidence", {})
        ) != set(case["questions"]):
            raise ValueError("missing original logits/prompt observations")
        for identifier, question in case["questions"].items():
            logits = row["raw_logits"][identifier]
            p = row["probabilities"][identifier]
            if (
                not isinstance(logits, list)
                or len(logits) != len(p)
                or any(
                    type(x) not in (int, float) or not math.isfinite(x) for x in logits
                )
            ):
                raise ValueError("invalid observed logits")
            scaled = [x / reference.TEMPERATURE for x in logits]
            weights = [math.exp(x - max(scaled)) for x in scaled]
            recomputed = [x / sum(weights) for x in weights]
            common.compare_answers(recomputed, p, 1e-12)
            evidence = row["evidence"][identifier]
            ids, labels = evidence.get("token_ids"), evidence.get("label_ids")
            if (
                not isinstance(ids, list)
                or not 1 <= len(ids) <= 1024
                or any(type(token) is not int or token < 0 for token in ids)
                or not isinstance(labels, list)
                or len(labels) != len(p)
                or len(set(labels)) != len(labels)
                or any(type(token) is not int or token < 0 for token in labels)
                or not isinstance(evidence.get("prompt"), str)
                or not evidence["prompt"]
                or evidence.get("row", {}).get("id") != identifier
                or evidence["row"].get("kind") != question["type"]
            ):
                raise ValueError("invalid original prompt/label observations")
            stats = evidence.get("prefix_cache_stats", {})
            if stats != {
                "restores": 0,
                "restored_prefix_tokens": 0,
                "prefix_tokens_decoded": 0,
                "checkpoint_bytes": 0,
            }:
                raise ValueError("reference used prefix/session caching")
    return value


def verify_metadata(response, threads):
    if response.get("model") != MODEL:
        raise ValueError("product model identity differs")
    metadata = response.get("metadata", {})
    for name, value in {**METADATA, "threads": threads}.items():
        if reference.ordered(metadata.get(name)) != reference.ordered(value):
            raise ValueError(f"product metadata differs: {name}")


def record_response(
    output,
    report,
    observations,
    status,
    response,
    seconds,
    expected,
    threads,
    *,
    case_id=None,
    repeated=None,
):
    observation = {"status": status, "response": response, "seconds": seconds}
    if case_id is not None:
        observation["id"] = case_id
    observations.append(observation)
    common.save(output, report)
    if status != 200:
        raise ValueError(f"product request failed: {response}")
    verify_metadata(response, threads)
    observation["max_absolute_difference"] = common.compare_answers(
        expected, response["answers"], TOLERANCE
    )
    if repeated is not None:
        common.compare_answers(repeated, response["answers"], 0)
    common.save(output, report)


def reference_run(args, command, environment, log, record):
    def action(process, tree):
        record["memory"] = []
        started, deadline = time.perf_counter(), time.monotonic() + args.timeout
        while process.poll() is None:
            tree.refresh()
            for owned in tree.living():
                try:
                    record["memory"].append(
                        common.memory_sample(owned, "author_running", started)
                    )
                except tree.psutil.NoSuchProcess:
                    pass
            if time.monotonic() >= deadline:
                raise TimeoutError("author reference timeout")
            time.sleep(1)
        if process.returncode != 0:
            raise RuntimeError("original reference failed; inspect author log")

    helpers.reference_run(args, command, environment, log, record, action)


def service_run(args, environment, prefix, identity, report, expected, number):
    record = {
        "route": "model_id" if number == 1 else "unicode_local_path",
        "cases": [],
        "memory": [],
    }
    report["service_runs"].append(record)
    with helpers.private_workspace(args.workspace, record) as workspace:
        source = workspace / ("checkpoint" if number == 1 else "中文 模型") / MODEL
        reference.clone_pinned(
            args.source, source, reference.SOURCE_HASHES, link_weight=True
        )
        command = [
            *prefix,
            "serve",
            MODEL if number == 1 else str(source),
            "--workspace",
            str(workspace),
            "--cpu-threads",
            str(args.threads),
            "--host",
            "127.0.0.1",
            "--port",
            str(args.port),
        ]
        child_env = helpers.product_environment(environment, identity)
        record.update(
            command=command,
            working_directory=str(workspace),
            pythonpath=child_env["PYTHONPATH"],
        )
        started = time.perf_counter()

        def action(process, tree):
            url = f"http://127.0.0.1:{args.port}"
            deadline = time.monotonic() + args.timeout
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError("CLI exited before ready; inspect server log")
                try:
                    status, health, _ = common.http(url, "/health", timeout=1)
                    if status == 200:
                        break
                except (OSError, ValueError):
                    pass
                time.sleep(0.2)
            else:
                raise TimeoutError("CLI startup timeout")
            owned = tree.listener(args.port)
            for key, value in {
                "model": MODEL,
                "backend": BACKEND,
                "device": "cpu",
                "revision": reference.REVISION,
                "experimental": True,
            }.items():
                if reference.ordered(health.get(key)) != reference.ordered(value):
                    raise ValueError(f"health identity differs: {key}")
            record.update(health=health, ready_seconds=time.perf_counter() - started)
            status, models, _ = common.http(url, "/v1/models")
            if status != 200 or models["data"][0]["id"] != MODEL:
                raise ValueError("model listing differs")
            record["memory"].append(common.memory_sample(owned, "ready", started))
            for index, case in enumerate(CASES):
                payload = {
                    "model": MODEL,
                    "state": case["state"],
                    "questions": case["questions"],
                }
                status, response, elapsed = common.http(
                    url, "/v1/systemone", payload, args.timeout
                )
                record_response(
                    args.output,
                    report,
                    record["cases"],
                    status,
                    response,
                    elapsed,
                    expected[index],
                    args.threads,
                    case_id=case["id"],
                    repeated=report["service_runs"][0]["cases"][index]["response"][
                        "answers"
                    ]
                    if number == 2
                    else None,
                )
                record["memory"].append(
                    common.memory_sample(owned, case["id"], started)
                )
                print(
                    json.dumps(
                        {
                            "event": "product_case_completed",
                            "route": record["route"],
                            "case": case["id"],
                            "seconds": elapsed,
                        }
                    ),
                    flush=True,
                )
            base = {
                "model": MODEL,
                "state": CASES[0]["state"],
                "questions": CASES[0]["questions"],
            }
            invalid = {
                "cache_salt": {**base, "cache_salt": "unavailable"},
                "media": {**base, "images": ["data:image/png;base64,AA=="]},
                "video": {**base, "videos": ["data:video/mp4;base64,AA=="]},
                "unknown_model": {**base, "model": "wrong/model"},
                "empty_questions": {**base, "questions": {}},
                "unknown_type": {
                    **base,
                    "questions": {"q": {"type": "bad", "instructions": "x"}},
                },
                "overlength": {**base, "state": "token " * 3000},
            }
            record["rejections"] = {}
            for name, payload in invalid.items():
                status, response, elapsed = common.http(
                    url, "/v1/systemone", payload, 60
                )
                record["rejections"][name] = {
                    "status": status,
                    "response": response,
                    "seconds": elapsed,
                }
                common.save(args.output, report)
                if status != 422:
                    raise ValueError(
                        f"invalid request was not rejected: {name}: {response}"
                    )
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                futures = [
                    pool.submit(common.http, url, "/v1/systemone", base, args.timeout)
                    for _ in range(2)
                ]
                time.sleep(0.1)
                before = sum(not item.done() for item in futures)
                status, _, seconds = common.http(url, "/health", timeout=5)
                after = sum(not item.done() for item in futures)
                record["concurrent_health"] = {
                    "status": status,
                    "seconds": seconds,
                    "unfinished_before": before,
                    "unfinished_after": after,
                }
                if status != 200 or not before or not after:
                    raise ValueError(
                        "health was not observed during outstanding inference"
                    )
                record["concurrent_requests"] = []
                for future in futures:
                    status, response, elapsed = future.result()
                    record_response(
                        args.output,
                        report,
                        record["concurrent_requests"],
                        status,
                        response,
                        elapsed,
                        expected[0],
                        args.threads,
                        repeated=record["cases"][0]["response"]["answers"],
                    )
            record["memory"].append(common.memory_sample(owned, "completed", started))

        child_args = argparse.Namespace(**vars(args))
        child_args.workspace = workspace
        helpers.child_run(
            child_args,
            command,
            child_env,
            args.output.with_name(args.output.stem + f"-server-{number}.log"),
            record,
            action,
        )
        record["source_unchanged"] = (
            reference.verify_source(source) == reference.SOURCE_HASHES
        )
    record["status"] = "passed"
    common.save(args.output, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "sdk-source", "reference-python", "workspace", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument(
        "--console-entrypoint",
        type=Path,
        help="Defaults to installed vllm-jev.exe beside this Python",
    )
    parser.add_argument("--reference-only", action="store_true")
    parser.add_argument("--reference-report", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--port", type=int, default=18824)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--min-available-gib", type=float, default=6)
    parser.add_argument("--min-commit-available-gib", type=float, default=8)
    parser.add_argument("--abort-below-available-gib", type=float, default=1.5)
    args = parser.parse_args()
    if sys.platform != "win32":
        parser.error("native Windows is required")
    if (
        not 1 <= args.threads <= min(4, os.cpu_count() or 1)
        or not 1 <= args.port <= 65535
    ):
        parser.error("invalid threads/port")
    for field in (
        "timeout",
        "min_available_gib",
        "min_commit_available_gib",
        "abort_below_available_gib",
    ):
        if not math.isfinite(getattr(args, field)) or getattr(args, field) <= 0:
            parser.error(f"{field} must be positive and finite")
    if args.abort_below_available_gib >= args.min_available_gib:
        parser.error("abort threshold must be below startup available-memory threshold")
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
    if args.reference_report == args.output or any(
        args.workspace.is_relative_to(root) for root in (args.source, args.sdk_source)
    ):
        parser.error("workspace/output must not replace input sources")
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
        ) or path in (
            args.reference_python,
            args.console_entrypoint,
            args.reference_report,
        ):
            parser.error("output would overwrite an input")
    args.workspace.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "running",
        "measured": False,
        "model": MODEL,
        "revision": reference.REVISION,
        "quantization": "Q5_K_M",
        "runtime_revision": "b11010",
        "threads": args.threads,
        "source": str(args.source),
        "tolerance": TOLERANCE,
        "driver_sha256": common.digest(Path(__file__)),
        "helper_sha256": {
            name: common.digest(Path(__file__).with_name(name)) for name in HELPERS
        },
        "environment": common.common.environment(),
        "service_runs": [],
        "scope": "Same pinned Q5/CPU runtime author comparison; not BF16 quality equivalence or throughput benchmarking.",
    }
    before = sdk_before = None
    try:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", args.port))
        before, sdk_before = (
            reference.verify_source(args.source),
            reference.verify_sdk(args.sdk_source),
        )
        report.update(source_hashes=before, sdk_hashes=sdk_before)
        environment = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith("JEV_") and key != "VLLM_JEV_DECISION_TEMPLATE"
        }
        environment.update(
            PYTHONDONTWRITEBYTECODE="1",
            PYTHONIOENCODING="utf-8",
            PYTHONPATH="",
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            HF_HOME=str(args.workspace / "hf-home"),
            OMP_NUM_THREADS=str(args.threads),
            MKL_NUM_THREADS=str(args.threads),
        )
        cases_path = args.output.with_name(args.output.stem + "-cases.json")
        common.save(cases_path, CASES)
        reference_path = args.reference_report or args.output.with_name(
            args.output.stem + "-author.json"
        )
        if not args.reference_report:
            command = command_for_reference(args, cases_path, reference_path)
            report["reference_process"] = {"command": command}
            reference_run(
                args,
                command,
                environment,
                args.output.with_name(args.output.stem + "-author.log"),
                report["reference_process"],
            )
        baseline = validate_reference(reference_path, args, cases_path, before)
        report.update(
            reference_report=str(reference_path),
            reference_sha256=common.digest(reference_path),
            reference_reused=bool(args.reference_report),
            reference_runner_sha256=common.digest(RUNNER),
        )
        expected = [
            canonical(case, row["probabilities"])
            for case, row in zip(CASES, baseline["cases"])
        ]
        if not args.reference_only:
            console = args.console_entrypoint or Path(sys.executable).with_name(
                "vllm-jev.exe"
            )
            prefix, identity = helpers.product_entrypoint(console)
            report["product_entrypoint"] = identity
            for number in (1, 2):
                service_run(
                    args, environment, prefix, identity, report, expected, number
                )
        report.update(status="passed", measured=True)
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        try:
            report["source_unchanged"] = (
                before is not None and reference.verify_source(args.source) == before
            )
            report["sdk_unchanged"] = (
                sdk_before is not None
                and reference.verify_sdk(args.sdk_source) == sdk_before
            )
        except BaseException as error:
            report.update(source_unchanged=False, integrity_error=str(error))
        if not report.get("source_unchanged") or not report.get("sdk_unchanged"):
            report["status"] = "failed"
        common.save(args.output, report)


if __name__ == "__main__":
    main()
