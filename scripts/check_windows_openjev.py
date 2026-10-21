"""Compare the Windows OpenJev CLI with its pinned, unmodified official SDK.

Example (run in the installed Windows CPU environment):
  python scripts/check_windows_openjev.py --model IamBusy/OpenJev-0.6B \
    --workspace C:/jev-validation --output C:/jev-validation/openjev-result.json \
    --sdk-source C:/reference/OpenJev-v0.3.2

Use --source to exercise a local checkpoint path, including a Unicode path.
Reference and service processes run sequentially, with two fresh CLI starts.
This checks bounded synthetic parity, not model quality or throughput.
"""

import argparse
import concurrent.futures
import hashlib
import importlib.metadata
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
MODEL = "IamBusy/OpenJev-0.6B"
REVISION = "b4bf78e7c053407b5d54a71bcc682ef5bbe59e72"
BASE_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
SDK_COMMIT = "5a827c549035cc4231ee77c5dd2bfcf7e42543ac"
SDK_HASHES = {
    "__init__.py": "8631bbe293672808aaaa317b64674498a490919e3f969be0ac7cbbcd508c4c8c",
    "branch_model.py": "7fbd5828a677bc3e7d6220310c27c792048894f5f7396ae72c4d7383dd6d6af2",
    "schema.py": "7149f1f19b688adb53af6b16bdb5b8c59c2d19f6ea0b43e7c1975b7a5dfd7b34",
    "io.py": "eb0fda0cf84c040dd127138fe5c362ab0f7fe18fddb48ca667ba1b82b401ac56",
    "identity.py": "f899349dc9f1d6c7d5521f6e62777a47132c52f562986aa202ed66ec31daf30c",
    "download.py": "235ac3b5f823ffdd5886eb55364d047969fc1b56d6180ff504621f33eff4a6cd",
}
# Independent release pins include the adapter, calibration, head and base.
SOURCE_HASHES = {
    "MANIFEST.json": "8a0df9df5e30f2e7b980de40460dd81ddf5cbc21ad263baf7ea256100a86584f",
    "LICENSE": "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30",
    "README.md": "058c4af0d3a2a127705a95ceacc40a08cd855da226c0f73c9772863266ca15b9",
    "RESULTS.md": "b57c93eacd96cb86f6a5ff51f07e158dce6658d592bd009c25c3f39175667363",
    "THIRD_PARTY.md": "2edf4dfbe190a24cf81c9d9fc3b3b5f4e3d7e936c66348732b6d6bc293b16c6c",
    "adapter_config.json": "ebfa81c887c626f1782ba3d5f5303373c47120f8a32ca3a81ef14fdb532c8cf8",
    "adapter_model.safetensors": "112e80f7ccc835b57266f9af6cfdf18eaf3e99c02c91a71941d5d1ac89c5ec07",
    "calibration-v03.json": "114210f173f334fa1b40c3d1b99356111fec708891801800286e7a5b6abe555b",
    "example_request.json": "ae3dc5feca0dd426ceece7c3d191b9969002d0e5e1346cfbc8ef02fb0c01ee93",
    "head.safetensors": "5f9e0780f6c8ace63a43fdbd58cfd2293e81d22d637ed139221a578fa7a9ac8d",
    "openjev_config.json": "a3db71d948690c661fbf9406ad0b96ae54e78a3ee3e282528857d214f66c98c4",
    "base/model.safetensors": "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b",
    "base/config.json": "660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd",
    "base/generation_config.json": "2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2",
    "base/tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    "base/tokenizer_config.json": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
    "base/vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
    "base/merges.txt": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
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
        raise ValueError(
            f"checkpoint differs from the pinned OpenJev release: {changed}"
        )
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


def copy_verified_sdk(sdk_source, destination):
    """Copy only pinned source, excluding installed modules and bytecode caches."""
    package = sdk_source / "src" / "openjev"
    for name, expected in SDK_HASHES.items():
        if digest(package / name) != expected:
            raise ValueError(f"refusing to execute unverified SDK source: {name}")
    target = destination / "openjev"
    target.mkdir(parents=True)
    for name, expected in SDK_HASHES.items():
        shutil.copyfile(package / name, target / name)
        if digest(target / name) != expected:
            raise ValueError(f"SDK source copy failed verification: {name}")
    return target


def reference(args):
    """No project loader, prompt builder, head or inference code is imported."""
    source = args.source.resolve()
    before = verify_source(source)
    result = {
        "status": "running",
        "environment": environment(),
        "model": MODEL,
        "revision": REVISION,
        "source": str(source),
        "source_sha256": before,
        "threads": args.threads,
        "base_revision": BASE_REVISION,
        "sdk_commit": SDK_COMMIT,
        "sdk_source_sha256": SDK_HASHES,
        "reference": "Unmodified OpenJev v0.3.2 OpenJevModel.from_pretrained().predict(cached=False); no project inference implementation",
        "inference": {
            "dtype": "float32",
            "attention": "sdpa",
            "cached": False,
            "sdk_patches": False,
        },
        "cases": [],
    }
    with tempfile.TemporaryDirectory(
        prefix="openjev-sdk-reference-", dir=args.workspace
    ) as folder:
        reference_source = Path(folder) / "checkpoint"
        clone_source(source, reference_source)
        result["reference_source_isolated"] = True
        package = copy_verified_sdk(args.sdk_source.resolve(), Path(folder))
        sys.path.insert(0, folder)
        import openjev
        import torch

        if Path(openjev.__file__).resolve() != package / "__init__.py":
            raise RuntimeError("reference imported an unrelated installed SDK")
        if (
            importlib.metadata.version("transformers") != "4.57.6"
            or importlib.metadata.version("peft") != "0.18.1"
        ):
            raise RuntimeError(
                "The reference requires Transformers 4.57.6 and PEFT 0.18.1."
            )
        started = time.perf_counter()
        model = openjev.OpenJevModel.from_pretrained(
            str(reference_source),
            base_model_path=reference_source / "base",
            local_files_only=True,
        )
        torch.set_num_threads(args.threads)
        result["cold_load_seconds"] = time.perf_counter() - started
        if (
            model.device != "cpu"
            or next(model.backbone.parameters()).dtype != torch.float32
        ):
            raise RuntimeError("reference did not load on FP32 CPU")
        for name, state in CASES:
            started = time.perf_counter()
            response = model.predict(state, QUESTIONS, cached=False)
            if (
                response["usage"]["shared_prefix"]
                or not response["calibration"]["applied"]
            ):
                raise AssertionError(
                    "reference must use uncached inference and delivered calibration"
                )
            result["cases"].append(
                {
                    "id": name,
                    "state": state,
                    "response": response,
                    "answers": response["answers"],
                    "seconds": time.perf_counter() - started,
                }
            )
        del model
        result["reference_source_unchanged"] = before == verify_source(reference_source)
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
                or not math.isfinite(left)
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
            if field == "legend":
                if not isinstance(expected_value, list):
                    raise AssertionError("official Score legend should be a list")
                expected_value = {
                    str(index): level for index, level in enumerate(expected_value)
                }
            compare(expected_value, value.get(field))
    return max(differences, default=0.0)


def clone_source(source, destination):
    for name, expected in SOURCE_HASHES.items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if name.endswith(".safetensors"):
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
        prefix="openjev-cli-validation-", dir=args.workspace, delete=False
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
                "backend": "openjev-pytorch",
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
    parser.add_argument(
        "--sdk-source",
        type=Path,
        required=True,
        help="Original OpenJev v0.3.2 source root containing src/openjev.",
    )
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18797)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--startup-timeout", type=float, default=240)
    parser.add_argument(
        "--reference-python",
        type=Path,
        default=Path(sys.executable),
        help="Reference environment with Transformers 4.57.6 and PEFT 0.18.1.",
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
        "scope": "Synthetic native Windows CPU CLI/official SDK uncached parity; text and JSON; Choice/Noul/Score. SDK Score legend list is mapped to the HTTP index-keyed legend object. Not quality or performance benchmarking.",
        "answer_absolute_tolerance": TOLERANCE,
    }
    if sys.platform != "win32":
        report.update(status="skipped", reason="This check requires native Windows.")
        save(args.output, report)
        print(json.dumps({"status": "skipped", "output": str(args.output)}))
        return
    child_environment = os.environ.copy()
    child_environment.update(
        PYTHONPATH=str(ROOT),
        USE_TF="0",
        USE_FLAX="0",
        HF_HUB_DISABLE_TELEMETRY="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
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
            "--sdk-source",
            str(args.sdk_source.resolve()),
        ]
        subprocess.run(command, cwd=ROOT, env=child_environment, check=True)
        expected = json.loads(reference_output.read_text(encoding="utf-8"))
        report.update(
            original_reference=str(reference_output),
            reference_source_isolated=expected["reference_source_isolated"],
            measured=True,
        )
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
