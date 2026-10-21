"""Validate the Windows CPU CLI against an independent, direct Laya SDK run.

Run from the installed Windows environment. Reference inference and the CLI
service run in separate, sequential processes so two models are never resident
together. This is a bounded synthetic parity check, not an accuracy benchmark.

Example:
  python scripts/check_windows_laya.py --model convaiinnovations/laya-multilingual \
    --source C:/models/laya-multilingual --workspace C:/jev-validation \
    --output C:/jev-validation/multilingual-result.json

Omit --source to exercise pinned downloading and the model-ID CLI route. Repeat
separately for convaiinnovations/laya and convaiinnovations/laya-typed-decisions.
"""

import argparse
import concurrent.futures
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import socket
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = (
    "convaiinnovations/laya",
    "convaiinnovations/laya-multilingual",
    "convaiinnovations/laya-typed-decisions",
)
QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "Choose the department that should handle this request.",
        "criteria": {"billing": "Payments and refunds", "technical": "Software issues"},
    },
    "refund": {"type": "noul", "instructions": "Does the customer request a refund?"},
    "urgency": {
        "type": "score",
        "instructions": "Rate how urgently the request needs attention.",
        "criteria": ["Can wait", "Needs attention soon", "Immediate blocking issue"],
    },
}
CASES = (
    ("text_refund", "I was charged twice. Please refund the duplicate payment today."),
    ("text_login", "I cannot sign in. My password reset link no longer works."),
    (
        "json_state",
        {"message": "Please refund my duplicate charge.", "currency": "USD"},
    ),
    ("chinese_state", "本月被重复扣款了，请退还多扣的费用。"),
)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def environment():
    packages = {}
    for name in ("torch", "transformers", "laya", "fastapi", "uvicorn", "vllm", "mlx"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": packages,
    }


def reference(args):
    """Use the published SDK directly; never project load_agent/template code."""
    sys.path.insert(0, str(ROOT))
    from vllm_jev.laya_export import MODELS as pinned_models
    from vllm_jev.laya_export import SOURCE_FILES, verify_source
    from vllm_jev.windows_models import identify_source, prepare

    source = (
        args.source.resolve() if args.source else prepare(args.model, args.workspace)
    )
    if identify_source(source) != args.model:
        raise ValueError("the local source does not match --model")
    verify_source(source, args.model)
    before = {name: digest(source / name) for name in SOURCE_FILES}
    result = {
        "status": "running",
        "environment": environment(),
        "model": args.model,
        "revision": pinned_models[args.model][0],
        "source": str(source),
        "source_sha256": before,
        "threads": args.threads,
        "reference": "published laya.Agent, no project load_agent or ContextOrderMixin",
        "cases": [],
    }
    import torch
    from laya import Agent

    torch.set_num_threads(args.threads)
    with tempfile.TemporaryDirectory(
        prefix="laya-sdk-reference-", dir=args.workspace
    ) as folder:
        runtime = Path(folder)
        for name in SOURCE_FILES:
            destination = runtime / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if name == "model.safetensors":
                try:
                    os.link(source / name, destination)
                except OSError:
                    shutil.copyfile(source / name, destination)
            else:
                shutil.copyfile(source / name, destination)
        started = time.perf_counter()
        agent = Agent(str(runtime), device="cpu", fast=False, compile=False)
        agent.amp_enabled = False
        agent.dtype = torch.float32
        result["cold_load_seconds"] = time.perf_counter() - started
        if str(agent.device) != "cpu":
            raise RuntimeError("reference did not load on CPU")
        for name, state in CASES:
            started = time.perf_counter()
            response = agent.system_one(state, QUESTIONS)
            result["cases"].append(
                {
                    "id": name,
                    "state": state,
                    "response": response,
                    "seconds": time.perf_counter() - started,
                }
            )
        del agent
    after = {name: digest(source / name) for name in SOURCE_FILES}
    result["source_unchanged"] = before == after
    if before != after:
        raise RuntimeError("SDK reference changed the original source")
    result["status"] = "passed"
    save(args.output, result)


def http(url, path, payload=None, timeout=60):
    body = (
        None
        if payload is None
        else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    )
    request = urllib.request.Request(
        url + path,
        data=body,
        headers={"Content-Type": "application/json; charset=utf-8"},
    )
    started = time.perf_counter()
    try:
        response = urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as error:
        response = error
    with response:
        return response.status, json.load(response), time.perf_counter() - started


def compare(expected, actual, tolerance=1e-6):
    """Preserve exact labels/keys and bound numeric differences recursively."""
    if isinstance(expected, dict):
        return (
            isinstance(actual, dict)
            and expected.keys() == actual.keys()
            and all(
                compare(value, actual[key], tolerance)
                for key, value in expected.items()
            )
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(expected) == len(actual)
            and all(
                compare(left, right, tolerance) for left, right in zip(expected, actual)
            )
        )
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        return isinstance(actual, (int, float)) and abs(expected - actual) <= tolerance
    return expected == actual


def validate_service(args, reference_result, child_environment, report, round_number):
    # The CLI's mutable runtime is created beside its source. Keep both in a
    # directory owned solely by this run, including when Windows termination
    # cannot execute the service's Python lifespan cleanup.
    owned = tempfile.TemporaryDirectory(
        prefix="laya-cli-validation-", dir=args.workspace, delete=False
    )
    run_index = len(report["service_runs"])
    try:
        workspace = Path(owned.name)
        clone = workspace / "checkpoint" / args.model
        original = Path(reference_result["source"])
        for name, expected in reference_result["source_sha256"].items():
            destination = clone / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if name == "model.safetensors":
                try:
                    os.link(original / name, destination)
                except OSError:
                    shutil.copyfile(original / name, destination)
            else:
                shutil.copyfile(original / name, destination)
            if digest(destination) != expected:
                raise AssertionError("private source copy failed hash verification")
        private_args = argparse.Namespace(**vars(args))
        private_args.workspace = workspace
        private_args.source = clone if args.source else None
        validate_service_run(
            private_args, reference_result, child_environment, report, round_number
        )
        checks = report["service_runs"][-1]
        checks["source_unchanged"] = all(
            digest(clone / name) == value
            for name, value in reference_result["source_sha256"].items()
        )
        if not checks["source_unchanged"]:
            raise AssertionError("CLI serving changed its private source files")
    finally:
        checks = (
            report["service_runs"][run_index]
            if len(report["service_runs"]) > run_index
            else {}
        )
        if not checks.get("owned_process_started") or checks.get(
            "owned_process_stopped"
        ):
            owned.cleanup()
            checks["private_workspace_removed"] = True
        else:
            checks["private_workspace_removed"] = False
            checks["retained_workspace"] = owned.name


def validate_service_run(
    args, reference_result, child_environment, report, round_number
):
    target = str(args.source.resolve()) if args.source else args.model
    command = [
        sys.executable,
        "-B",
        "-m",
        "vllm_jev.cli",
        "serve",
        target,
        "--workspace",
        str(args.workspace),
        "--cpu-threads",
        str(args.threads),
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
    ]
    log = args.output.with_name(args.output.stem + f"-server-{round_number}.log")
    url = f"http://127.0.0.1:{args.port}"
    checks = {"round": round_number, "command": command, "log": str(log), "cases": []}
    report["service_runs"].append(checks)
    with log.open("w", encoding="utf-8") as output:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=child_environment,
            stdout=output,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        checks["owned_process_started"] = True
        try:
            deadline = time.monotonic() + args.startup_timeout
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"CLI exited during startup; inspect {log}")
                try:
                    status, health, _ = http(url, "/health", timeout=1)
                    if status == 200:
                        break
                except (OSError, ValueError):
                    pass
                time.sleep(0.2)
            else:
                raise TimeoutError(f"CLI health did not become ready; inspect {log}")
            if health.get("model") != args.model or health.get("device") != "cpu":
                raise AssertionError(f"unexpected health identity: {health}")
            if health.get("revision") != reference_result["revision"]:
                raise AssertionError("CLI/reference model revision differs")
            checks["health"] = health
            for expected in reference_result["cases"]:
                status, response, elapsed = http(
                    url,
                    "/v1/systemone",
                    {
                        "model": args.model,
                        "state": expected["state"],
                        "questions": QUESTIONS,
                    },
                )
                equal = status == 200 and compare(
                    expected["response"]["answers"], response.get("answers")
                )
                checks["cases"].append(
                    {
                        "id": expected["id"],
                        "status": status,
                        "seconds": elapsed,
                        "answers_match_sdk": equal,
                        "response": response,
                    }
                )
                if not equal:
                    raise AssertionError(
                        f"CLI/reference answers differ: {expected['id']}"
                    )
            base = {"state": CASES[0][1], "questions": QUESTIONS}
            invalid = {
                "cache_salt": {**base, "cache_salt": "synthetic-caller"},
                "image": {**base, "images": ["data:image/png;base64,AA=="]},
                "video": {**base, "videos": ["data:video/mp4;base64,AA=="]},
                "question": {**base, "questions": {"q": {"type": "unsupported"}}},
            }
            checks["rejections"] = {}
            for name, body in invalid.items():
                status, response, _ = http(url, "/v1/systemone", body)
                checks["rejections"][name] = {"status": status, "response": response}
                if status != 422:
                    raise AssertionError(f"unsupported {name} was not rejected")
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as clients:
                pending = [
                    clients.submit(http, url, "/v1/systemone", base) for _ in range(2)
                ]
                status, health, elapsed = http(url, "/health", timeout=5)
                checks["concurrent_health"] = {"status": status, "seconds": elapsed}
                responses = [future.result() for future in pending]
            checks["concurrent_answers_match_sdk"] = all(
                status == 200
                and compare(
                    reference_result["cases"][0]["response"]["answers"],
                    response.get("answers"),
                )
                for status, response, _ in responses
            )
            if status != 200 or not checks["concurrent_answers_match_sdk"]:
                raise AssertionError("concurrent requests or health failed")
            checks["status"] = "passed"
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
            checks["owned_process_stopped"] = process.poll() is not None
            checks["shutdown_scope"] = (
                "Owned process terminated; this does not validate graceful shutdown. "
                "The private CLI workspace is removed only after process exit."
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18795)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--startup-timeout", type=float, default=180)
    parser.add_argument(
        "--reference-child", action="store_true", help=argparse.SUPPRESS
    )
    args = parser.parse_args()
    args.workspace = args.workspace.resolve()
    args.output = args.output.resolve()
    if args.threads < 1 or not 1 <= args.port <= 65535:
        parser.error("threads must be positive and port must be in 1..65535")
    args.workspace.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.reference_child:
        reference(args)
        return
    report = {
        "status": "running",
        "model": args.model,
        "environment": environment(),
        "measured": False,
        "service_runs": [],
        "scope": "Synthetic Windows CPU CLI/SDK parity, text and JSON, Choice/Noul/Score; not quality or performance benchmarking.",
        "answer_absolute_tolerance": 1e-6,
    }
    if sys.platform != "win32":
        report.update(status="skipped", reason="This check requires native Windows.")
        save(args.output, report)
        print(json.dumps({"status": "skipped", "output": str(args.output)}))
        return
    child_environment = os.environ.copy()
    child_environment["PYTHONPATH"] = str(ROOT)
    child_environment.update(
        USE_TF="0", USE_FLAX="0", LAYA_CPU_AMP="", HF_HUB_DISABLE_TELEMETRY="1"
    )
    child_environment.pop("VLLM_JEV_DECISION_TEMPLATE", None)
    reference_output = args.output.with_name(args.output.stem + "-sdk-reference.json")
    try:
        # Refuse an occupied port; do not reuse or stop somebody else's server.
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", args.port))
        reference_command = [
            sys.executable,
            "-B",
            str(Path(__file__).resolve()),
            "--reference-child",
            "--model",
            args.model,
            "--workspace",
            str(args.workspace),
            "--output",
            str(reference_output),
            "--threads",
            str(args.threads),
        ]
        if args.source:
            reference_command.extend(["--source", str(args.source.resolve())])
        subprocess.run(reference_command, cwd=ROOT, env=child_environment, check=True)
        reference_result = json.loads(reference_output.read_text(encoding="utf-8"))
        report["sdk_reference"] = str(reference_output)
        report["revision"] = reference_result["revision"]
        report["measured"] = True
        for round_number in (1, 2):
            validate_service(
                args, reference_result, child_environment, report, round_number
            )
        source = Path(reference_result["source"])
        report["source_unchanged"] = all(
            digest(source / name) == value
            for name, value in reference_result["source_sha256"].items()
        )
        if not report["source_unchanged"]:
            raise AssertionError("CLI serving changed original model files")
        report.update(status="passed", measured=True)
        report["observed_http_median_seconds"] = statistics.median(
            row["seconds"] for run in report["service_runs"] for row in run["cases"]
        )
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save(args.output, report)
        print(json.dumps({"status": report["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
