"""Compare the Windows Tiny-Jev CLI with its pinned original HF implementation.

Example (run in the installed Windows CPU environment):
  python scripts/check_windows_qwen.py --model lostargon/Tiny-Jev \
    --workspace C:/jev-validation --output C:/jev-validation/tiny-result.json \
    --reference-python C:/tiny-transformers-5.17/Scripts/python.exe

Use --source to exercise a local checkpoint path, including a Unicode path.
Reference and service processes run sequentially, with two fresh CLI starts.
This checks bounded synthetic parity, not model quality or throughput.
"""

import argparse
import concurrent.futures
import hashlib
import importlib.metadata
import importlib.util
import json
import math
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
MODEL = "lostargon/Tiny-Jev"
REVISION = "62449fb3272d68fb8c8fe8a33082566643a76d0b"
# Independent pins: the reference may execute only these original source bytes.
SOURCE_HASHES = {
    "model.safetensors": "513411b17cb0eca54839ca3948b4353e7f65710723ffcde8d13e3cb04328633e",
    "config.json": "3897b8c2f7ee8e566449b170f2f8a9cc2dde447fb474be50a155485d69859937",
    "tokenizer.json": "e3fccc77adda606e2a461ffd17ca5a55a5a75ac40a5aa0a91c5307fb31b1efac",
    "tokenizer_config.json": "5e44295ef654157e02ca593d0be8c07ed11265d4553e7920005d721ff5a9feab",
    "chat_template.jinja": "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8",
    "modeling_tiny_jev.py": "5c2d146e2f92ffcb7bb8a0927b180a3d9458b5fc3c4f5d99835302910802e05d",
}
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
TOLERANCE = 1e-5


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def verify_source(source):
    observed = {name: digest(source / name) for name in SOURCE_HASHES}
    if observed != SOURCE_HASHES:
        changed = [name for name in observed if observed[name] != SOURCE_HASHES[name]]
        raise ValueError(f"checkpoint differs from the pinned Tiny release: {changed}")
    return observed


def environment():
    packages = {}
    for name in (
        "torch",
        "transformers",
        "peft",
        "laya",
        "fastapi",
        "uvicorn",
        "vllm",
        "mlx",
    ):
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


def original_tiny_module(source):
    """Hash-check the original published module before executing it locally."""
    path = source / "modeling_tiny_jev.py"
    if digest(path) != SOURCE_HASHES[path.name]:
        raise ValueError("refusing to execute unverified reference source code")
    spec = importlib.util.spec_from_file_location("tiny_jev_original_reference", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def reference(args):
    """No project loader, prompt builder, head or inference code is imported."""
    source = args.source.resolve()
    before = verify_source(source)
    import torch
    from transformers import AutoTokenizer

    if importlib.metadata.version("transformers") != "5.17.0":
        raise RuntimeError(
            "The unmodified Tiny reference requires Transformers 5.17.0; pass --reference-python."
        )
    torch.set_num_threads(args.threads)
    original = original_tiny_module(source)
    # Use the release's native Transformers version. In particular, do not
    # repeat production's TF4 compatibility mapping for RoPE or token metadata.
    config = original.TinyJevConfig.from_pretrained(str(source), local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(str(source), local_files_only=True)
    started = time.perf_counter()
    model = original.TinyJevModel.from_pretrained(
        str(source),
        config=config,
        dtype=torch.float32,
        attn_implementation="sdpa",
        local_files_only=True,
    ).eval()
    if next(model.parameters()).device.type != "cpu":
        raise RuntimeError("reference did not load on CPU")
    result = {
        "status": "running",
        "environment": environment(),
        "model": MODEL,
        "revision": REVISION,
        "source": str(source),
        "source_sha256": before,
        "threads": args.threads,
        "cold_load_seconds": time.perf_counter() - started,
        "reference": "Pinned original modeling_tiny_jev.TinyJevModel.decide; no project inference implementation",
        "inference": {
            "dtype": "float32",
            "attention": "sdpa",
            "config_compatibility_patches": False,
            "tokenizer_compatibility_patches": False,
        },
        "cases": [],
    }
    definitions = [
        {"kind": value["type"], **{k: v for k, v in value.items() if k != "type"}}
        for value in QUESTIONS.values()
    ]
    for name, state in CASES:
        started = time.perf_counter()
        decisions = model.decide(tokenizer, state, definitions)
        result["cases"].append(
            {
                "id": name,
                "state": state,
                "raw_decisions": decisions,
                "answers": dict(zip(QUESTIONS, decisions)),
                "seconds": time.perf_counter() - started,
            }
        )
    del model
    result["source_unchanged"] = before == verify_source(source)
    result["status"] = "passed"
    save(args.output, result)


def http(url, path, payload=None, timeout=90):
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


def compare_answers(expected, actual, tolerance=TOLERANCE):
    """Compare typed decision values, not unrelated HTTP/reference envelopes."""
    if not isinstance(actual, dict) or set(expected) != set(actual):
        raise AssertionError("answer identifiers differ")
    differences = []

    def compare(left, right):
        if isinstance(left, dict):
            if not isinstance(right, dict) or left.keys() != right.keys():
                raise AssertionError("probability labels differ")
            for key in left:
                compare(left[key], right[key])
        elif isinstance(left, (int, float)) and not isinstance(left, bool):
            if (
                not isinstance(right, (int, float))
                or isinstance(right, bool)
                or not math.isfinite(right)
            ):
                raise AssertionError("expected a finite numeric decision")
            difference = abs(left - right)
            differences.append(difference)
            if difference > tolerance:
                raise AssertionError(
                    f"decision differs by {difference}, tolerance {tolerance}"
                )
        elif left != right:
            raise AssertionError(f"decision label differs: {left!r} != {right!r}")

    for identifier, decision in expected.items():
        value = actual[identifier]
        definition = QUESTIONS[identifier]
        if value.get("type") != definition["type"]:
            raise AssertionError("typed HTTP answer has an incorrect type")
        for field, expected_value in decision.items():
            compare(expected_value, value.get(field))
        if definition["type"] == "score":
            compare(
                {level: level for level in definition["criteria"]}, value.get("legend")
            )
    return max(differences, default=0.0)


def clone_source(source, destination):
    for name, expected in SOURCE_HASHES.items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if name == "model.safetensors":
            try:
                os.link(source / name, target)
            except OSError:
                shutil.copyfile(source / name, target)
        else:
            shutil.copyfile(source / name, target)
        if digest(target) != expected:
            raise AssertionError("private source copy failed verification")


def service_run(args, expected, child_environment, report, round_number):
    owned = tempfile.TemporaryDirectory(
        prefix="tiny-cli-validation-", dir=args.workspace, delete=False
    )
    workspace = Path(owned.name)
    clone = workspace / "checkpoint" / MODEL
    check = {"round": round_number, "cases": [], "owned_process_started": False}
    report["service_runs"].append(check)
    process = None
    try:
        clone_source(Path(expected["source"]), clone)
        command = [
            sys.executable,
            "-B",
            "-m",
            "vllm_jev.cli",
            "serve",
            str(clone) if args.source else MODEL,
            "--workspace",
            str(workspace),
            "--cpu-threads",
            str(args.threads),
            "--host",
            "127.0.0.1",
            "--port",
            str(args.port),
        ]
        log = args.output.with_name(args.output.stem + f"-server-{round_number}.log")
        check.update(command=command, log=str(log))
        url = f"http://127.0.0.1:{args.port}"
        with log.open("w", encoding="utf-8") as output:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=child_environment,
                stdout=output,
                stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            check["owned_process_started"] = True
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
                raise TimeoutError(f"CLI did not become ready; inspect {log}")
            for key, value in {
                "model": MODEL,
                "device": "cpu",
                "revision": REVISION,
                "backend": "tiny-jev-pytorch",
                "experimental": True,
            }.items():
                if health.get(key) != value:
                    raise AssertionError(f"unexpected health {key}: {health}")
            check["health"] = health
            for case in expected["cases"]:
                status, response, elapsed = http(
                    url,
                    "/v1/systemone",
                    {"model": MODEL, "state": case["state"], "questions": QUESTIONS},
                )
                row = {
                    "id": case["id"],
                    "status": status,
                    "seconds": elapsed,
                    "response": response,
                }
                check["cases"].append(row)
                if status != 200:
                    raise AssertionError(f"valid request failed: {case['id']}")
                row["max_absolute_difference"] = compare_answers(
                    case["answers"], response.get("answers")
                )
                row["answers_match_original"] = True
            base = {"state": CASES[0][1], "questions": QUESTIONS}
            invalid = {
                "cache_salt": {**base, "cache_salt": "synthetic-caller"},
                "image": {**base, "images": ["data:image/png;base64,AA=="]},
                "video": {**base, "videos": ["data:video/mp4;base64,AA=="]},
                "model": {**base, "model": "unloaded/model"},
                "question": {**base, "questions": {"q": {"type": "unsupported"}}},
            }
            check["rejections"] = {}
            for name, body in invalid.items():
                status, response, _ = http(url, "/v1/systemone", body)
                check["rejections"][name] = {"status": status, "response": response}
                if status != 422:
                    raise AssertionError(f"unsupported {name} was not rejected")
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as clients:
                pending = [
                    clients.submit(http, url, "/v1/systemone", base) for _ in range(2)
                ]
                health_status, _, elapsed = http(url, "/health", timeout=5)
                check["concurrent_health"] = {
                    "status": health_status,
                    "seconds": elapsed,
                }
                responses = [future.result() for future in pending]
            check["concurrent_max_absolute_difference"] = []
            for status, response, _ in responses:
                if status != 200:
                    raise AssertionError("concurrent request failed")
                check["concurrent_max_absolute_difference"].append(
                    compare_answers(
                        expected["cases"][0]["answers"], response.get("answers")
                    )
                )
            if health_status != 200:
                raise AssertionError("concurrent health failed")
            check["status"] = "passed"
    finally:
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
            check["owned_process_stopped"] = process.poll() is not None
        if not check["owned_process_started"] or check.get("owned_process_stopped"):
            try:
                if clone.exists():
                    check["source_unchanged"] = (
                        verify_source(clone) == expected["source_sha256"]
                    )
            finally:
                owned.cleanup()
                check["private_workspace_removed"] = True
        else:
            check.update(private_workspace_removed=False, retained_workspace=owned.name)
        check["shutdown_scope"] = (
            "Owned process terminated; graceful shutdown is not validated."
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=[MODEL], required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18796)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--startup-timeout", type=float, default=240)
    parser.add_argument(
        "--reference-python",
        type=Path,
        default=Path(sys.executable),
        help="Python environment matching the original release (Transformers 5.17).",
    )
    parser.add_argument(
        "--reference-child", action="store_true", help=argparse.SUPPRESS
    )
    args = parser.parse_args()
    args.workspace, args.output = args.workspace.resolve(), args.output.resolve()
    if args.threads < 1 or not 1 <= args.port <= 65535:
        parser.error("threads must be positive and port must be in 1..65535")
    args.workspace.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.reference_child:
        if args.source is None:
            parser.error("reference child requires an already verified --source")
        reference(args)
        return
    report = {
        "status": "running",
        "model": MODEL,
        "revision": REVISION,
        "environment": environment(),
        "measured": False,
        "service_runs": [],
        "scope": "Synthetic native Windows CPU CLI/original HF parity; text and JSON; Choice/Noul/Score. Not quality or performance benchmarking.",
        "answer_absolute_tolerance": TOLERANCE,
    }
    if sys.platform != "win32":
        report.update(status="skipped", reason="This check requires native Windows.")
        save(args.output, report)
        print(json.dumps({"status": "skipped", "output": str(args.output)}))
        return
    child_environment = os.environ.copy()
    child_environment.update(
        PYTHONPATH=str(ROOT), USE_TF="0", USE_FLAX="0", HF_HUB_DISABLE_TELEMETRY="1"
    )
    child_environment.pop("VLLM_JEV_DECISION_TEMPLATE", None)
    reference_output = args.output.with_name(
        args.output.stem + "-original-reference.json"
    )
    source = None
    try:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", args.port))
        if args.source:
            source = args.source.resolve()
        else:
            sys.path.insert(0, str(ROOT))
            from vllm_jev.windows_models import prepare

            source = prepare(MODEL, args.workspace)
        verify_source(source)
        command = [
            str(args.reference_python.resolve()),
            "-B",
            str(Path(__file__).resolve()),
            "--reference-child",
            "--model",
            MODEL,
            "--source",
            str(source),
            "--workspace",
            str(args.workspace),
            "--output",
            str(reference_output),
            "--threads",
            str(args.threads),
        ]
        subprocess.run(command, cwd=ROOT, env=child_environment, check=True)
        expected = json.loads(reference_output.read_text(encoding="utf-8"))
        report.update(original_reference=str(reference_output), measured=True)
        for round_number in (1, 2):
            service_run(args, expected, child_environment, report, round_number)
        report["observed_http_median_seconds"] = statistics.median(
            case["seconds"] for run in report["service_runs"] for case in run["cases"]
        )
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if source is not None:
            try:
                report["source_unchanged"] = verify_source(source) == SOURCE_HASHES
            except (OSError, ValueError) as error:
                report.update(
                    status="failed", source_unchanged=False, source_error=str(error)
                )
        save(args.output, report)
        print(json.dumps({"status": report["status"], "output": str(args.output)}))
        if report["status"] == "failed" and sys.exc_info()[0] is None:
            raise RuntimeError("Validation failed; inspect the saved report.")


if __name__ == "__main__":
    main()
