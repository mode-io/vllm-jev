"""Compare two fresh Windows CLI runs with an independently executed author SDK.

The repository reference runner executes pinned, unmodified Decider Python source.
Its raw softmax probabilities are observed without replacing the author's code;
this script independently checks the project's documented answer presentation.
No mode downloads models, installs packages, or closes unrelated applications.
"""

import argparse
import concurrent.futures
import ctypes
import importlib.util
import json
import math
import os
import shutil
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = "Mapika/decider-0.8b"
TOLERANCE = 1e-5
MAX_SEQUENCE_TOKENS = 1024
REFERENCE_RUNNER = Path(__file__).with_name("windows_decider_reference.py")
_reference_spec = importlib.util.spec_from_file_location(
    "_decider_author_profiles", REFERENCE_RUNNER
)
reference_profiles = importlib.util.module_from_spec(_reference_spec)
_reference_spec.loader.exec_module(reference_profiles)
# These release pins belong to the independent author harness, never the product loader.
PROFILES = {
    model: {
        **profile,
        "model": model,
        "temperature": 1.145 if profile["typed_temperatures"] else 1.03,
        "temperature_by_type": {"choice": 1.164, "noul": 1.624, "score": 1.124}
        if profile["typed_temperatures"]
        else {},
        "min_available_gib": 6.0 if model == "Mapika/decider-2b" else 4.0,
        "min_commit_available_gib": 8.0 if model == "Mapika/decider-2b" else 6.0,
    }
    for model, profile in reference_profiles.PROFILES.items()
}
REVISION = PROFILES[MODEL]["revision"]
SOURCE_HASHES, SDK_HASHES = (
    PROFILES[MODEL]["source_hashes"],
    PROFILES[MODEL]["sdk_hashes"],
)


def profile_for(args):
    return PROFILES[getattr(args, "model", MODEL)]


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
CASES = [
    {"id": name, "state": state, "questions": QUESTIONS}
    for name, state in (
        (
            "text_refund",
            "I was charged twice. Please refund the duplicate payment today.",
        ),
        (
            "chinese_json",
            {
                "message": "本月被重复扣款了，请退还多扣的费用。谢谢 🙏",
                "order": "订单 A-12",
            },
        ),
    )
]
CASES.append(
    {
        "id": "wide_choice",
        "state": "My password reset link no longer works. I cannot sign in.",
        "questions": {
            "category": {
                "type": "choice",
                "instructions": "Choose the best support category.",
                "criteria": dict(
                    zip(
                        (
                            "refund",
                            "invoice",
                            "password",
                            "delivery",
                            "returns",
                            "cancel",
                            "upgrade",
                            "privacy",
                            "feedback",
                            "outage",
                            "sales",
                            "other",
                        ),
                        (
                            "Refunds",
                            "Billing invoices",
                            "Password resets and sign-in",
                            "Delivery tracking",
                            "Product returns",
                            "Subscription cancellation",
                            "Account upgrades",
                            "Personal data requests",
                            "Product feedback",
                            "Service outages",
                            "Sales questions",
                            "Other issues",
                        ),
                    )
                ),
            }
        },
    }
)

# Reuse audited process ownership and HTTP utilities only, never model code.
_spec = importlib.util.spec_from_file_location(
    "_windows_acceptance_process_utils",
    Path(__file__).with_name("check_windows_open_jev.py"),
)
common = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(common)
digest, save, http = common.digest, common.save, common.http
compare_answers = common.compare_answers


class OwnedProcessTree(common.OwnedProcessTree):
    """Serialize the shared ownership tracker with the low-memory watchdog."""

    def __init__(self, *args):
        self.lock = threading.RLock()
        super().__init__(*args)

    def refresh(self):
        with self.lock:
            return super().refresh()

    def listener(self, port):
        with self.lock:
            return super().listener(port)

    def stop(self):
        with self.lock:
            return super().stop()


class MemoryMonitor:
    def __init__(self, tree, args, record):
        self.tree, self.record = tree, record
        self.threshold = int(args.abort_below_available_gib * 2**30)
        self.event = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.result = {
            "abort_below_available_bytes": self.threshold,
            "aborted": False,
            "samples": 0,
        }
        record["memory_monitor"] = self.result

    def start(self):
        self.thread.start()

    def run(self):
        import psutil

        while not self.event.is_set():
            try:
                available = psutil.virtual_memory().available
                self.result["samples"] += 1
                self.result["minimum_available_bytes"] = min(
                    available, self.result.get("minimum_available_bytes", available)
                )
                if available < self.threshold:
                    self.result.update(
                        aborted=True,
                        reason="Available physical memory fell below the configured threshold",
                    )
                    self.tree.stop()
                    return
            except Exception as error:
                self.result.update(aborted=True, error=str(error))
                self.tree.stop()
                return
            self.event.wait(0.5)

    def stop(self):
        self.event.set()
        self.thread.join(timeout=35)
        if self.thread.is_alive():
            self.result.update(aborted=True, error="memory watchdog did not stop")
        if self.result["aborted"]:
            self.record["status"] = "failed"
        return not self.result["aborted"]


def verify_source(source, profile=None):
    hashes = SOURCE_HASHES if profile is None else profile["source_hashes"]
    for entry in source.iterdir():
        if entry.name == ".cache" and entry.is_dir():
            continue
        if entry.name in {"README.md", "LICENSE", ".gitattributes"} and entry.is_file():
            continue
        if entry.name not in hashes or not entry.is_file():
            raise ValueError(f"unexpected Decider source entry: {entry.name}")
    for name in hashes:
        if not (source / name).is_file():
            raise ValueError(f"missing Decider source file: {name}")
    observed = {name: digest(source / name) for name in hashes}
    if observed != hashes:
        raise ValueError("Decider checkpoint does not match the fixed release hashes")
    return observed


def verify_sdk(source, profile=None):
    hashes = SDK_HASHES if profile is None else profile["sdk_hashes"]
    if {name: digest(source / name) for name in hashes} != hashes:
        raise ValueError("author SDK source does not match the fixed release hashes")


def clone_source(source, destination, profile=None):
    hashes = SOURCE_HASHES if profile is None else profile["source_hashes"]
    verify_source(source, profile)
    destination.mkdir(parents=True)
    for name in hashes:
        if name.endswith(".safetensors"):
            try:
                os.link(source / name, destination / name)
                continue
            except OSError:
                pass
        shutil.copyfile(source / name, destination / name)
    verify_source(destination, profile)


def system_memory():
    import psutil

    memory = psutil.virtual_memory()
    result = {"available_bytes": memory.available, "total_bytes": memory.total}
    if os.name != "nt":
        return result

    class PerformanceInformation(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            *[
                (name, ctypes.c_size_t)
                for name in (
                    "CommitTotal",
                    "CommitLimit",
                    "CommitPeak",
                    "PhysicalTotal",
                    "PhysicalAvailable",
                    "SystemCache",
                    "KernelTotal",
                    "KernelPaged",
                    "KernelNonpaged",
                    "PageSize",
                )
            ],
            ("HandleCount", ctypes.c_ulong),
            ("ProcessCount", ctypes.c_ulong),
            ("ThreadCount", ctypes.c_ulong),
        ]

    info = PerformanceInformation()
    info.cb = ctypes.sizeof(info)
    get_info = ctypes.WinDLL("psapi", use_last_error=True).GetPerformanceInfo
    get_info.argtypes = [ctypes.POINTER(PerformanceInformation), ctypes.c_ulong]
    get_info.restype = ctypes.c_bool
    if not get_info(ctypes.byref(info), info.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    result.update(
        committed_bytes=info.CommitTotal * info.PageSize,
        commit_limit_bytes=info.CommitLimit * info.PageSize,
        commit_available_bytes=(info.CommitLimit - info.CommitTotal) * info.PageSize,
    )
    return result


def memory_gate(args):
    sample = system_memory()
    sample.update(
        minimum_available_bytes=int(args.min_available_gib * 2**30),
        minimum_commit_available_bytes=int(args.min_commit_available_gib * 2**30),
        policy="Conservative experiment precheck, not a measured peak-memory guarantee.",
    )
    sample["passed"] = (
        sample["available_bytes"] >= sample["minimum_available_bytes"]
        and sample.get("commit_available_bytes", 0)
        >= sample["minimum_commit_available_bytes"]
    )
    return sample


def memory_sample(process, stage, started):
    sample = common.memory_sample(stage, started, process)
    info = process.memory_info()
    if os.name == "nt" and not hasattr(info, "peak_pagefile"):
        raise RuntimeError("Windows process peak commit counter is unavailable")
    sample["peak_commit_bytes"] = getattr(info, "peak_pagefile", None)
    sample["system"] = system_memory()
    return sample


def _distribution(values, count):
    if not isinstance(values, list) or len(values) < count:
        raise ValueError("incomplete author row probability vector")
    if any(
        type(p) not in (float, int) or not math.isfinite(p) or not 0 <= p <= 1
        for p in values
    ):
        raise ValueError("author row probabilities must be finite and within [0, 1]")
    if any(values[count:]) or not math.isclose(sum(values), 1.0, abs_tol=1e-6):
        raise ValueError("author row probability mass or masked options are invalid")
    return values[:count]


def canonical_answers(case):
    """Map observed author row probabilities to the project's public wire formulas."""
    result, expected_index, position = {}, [], 0
    rows = case["row_probabilities"]
    for identifier, question in case["questions"].items():
        kind, criteria = question["type"], question.get("criteria")
        count = len(criteria) if kind != "noul" else 2
        isolated = kind == "score"
        row_count = count if isolated else 1
        expected_index.append(
            [identifier, "iso" if isolated else "list", position, row_count]
        )
        if isolated:
            fits = [_distribution(rows[position + j], 2)[1] for j in range(count)]
            total = sum(fits)
            if total <= 0:
                raise ValueError("isolated Score has no positive probability mass")
            probabilities = [p / total for p in fits]
        else:
            probabilities = _distribution(rows[position], count)
        position += row_count
        if kind == "noul":
            result[identifier] = {"type": "noul", "noul": probabilities[1]}
            continue
        selected = max(range(count), key=probabilities.__getitem__)
        keys = list(criteria) if kind == "choice" else [str(i) for i in range(count)]
        answer = {"type": kind, "probabilities": dict(zip(keys, probabilities))}
        if kind == "choice":
            answer.update(
                choice=keys[selected],
                confidence=max(
                    0.0, (count * probabilities[selected] - 1) / (count - 1)
                ),
            )
        else:
            distance = sum(abs(i - (count - 1) / 2) for i in range(count)) / count
            answer.update(
                score=sum(i * p for i, p in enumerate(probabilities)),
                confidence=max(
                    0.0,
                    1
                    - sum(p * abs(i - selected) for i, p in enumerate(probabilities))
                    / distance,
                ),
                legend=dict(zip(keys, criteria)),
            )
        result[identifier] = answer
    if case["row_index"] != expected_index or len(rows) != position:
        raise ValueError("author row index does not match the requested question order")
    return result


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(f"non-finite JSON value: {value}")

    return json.loads(
        path.read_bytes(), object_pairs_hook=pairs, parse_constant=invalid
    )


def validate_author_observations(case, profile=None):
    """Bind the observed raw rows to the author's returned, rounded answers."""
    expected = canonical_answers(case)
    if set(case["answers"]) != set(expected):
        raise ValueError("reference author answer IDs differ")
    for identifier, project in expected.items():
        author = case["answers"][identifier]
        kind = project["type"]
        if author.get("type") != kind:
            raise ValueError("reference author answer type differs")
        if kind == "noul":
            index = next(row for row in case["row_index"] if row[0] == identifier)
            probabilities = case["row_probabilities"][index[2]][:2]
            wanted = {"noul": round(probabilities[1] / sum(probabilities), 4)}
        else:
            probabilities = list(project["probabilities"].values())
            normalized = [p / sum(probabilities) for p in probabilities]
            wanted = {
                "probabilities": dict(
                    zip(project["probabilities"], (round(p, 4) for p in normalized))
                ),
                "confidence": round(max(normalized), 4),
            }
            if profile is not None and profile["typed_temperatures"]:
                # The author's confidence helper normalizes once more, while
                # probabilities, expected Score and x_p_max use the first pass.
                confidence_probabilities = [p / sum(normalized) for p in normalized]
                count = len(confidence_probabilities)
                selected = max(range(count), key=confidence_probabilities.__getitem__)
                if kind == "choice":
                    confidence = max(
                        0.0,
                        (count * confidence_probabilities[selected] - 1) / (count - 1),
                    )
                else:
                    distance = (
                        sum(abs(i - (count - 1) / 2) for i in range(count)) / count
                    )
                    confidence = max(
                        0.0,
                        1
                        - sum(
                            p * abs(i - selected)
                            for i, p in enumerate(confidence_probabilities)
                        )
                        / distance,
                    )
                wanted.update(
                    confidence=round(min(1.0, confidence), 4),
                    x_p_max=round(max(normalized), 4),
                )
            if kind == "choice":
                wanted["choice"] = project["choice"]
            else:
                wanted.update(
                    score=round(sum(i * p for i, p in enumerate(normalized)), 2),
                    legend=project["legend"],
                )
        for key, value in wanted.items():
            compare_answers(value, author.get(key), tolerance=0)
    prompts = case["prompt_rows"]
    shapes = case["forward_input_shapes"]
    if len(prompts) != len(case["row_probabilities"]) or len(shapes) != len(prompts):
        raise ValueError("reference prompt/forward count differs")
    for prompt, shape in zip(prompts, shapes):
        ids = prompt["ids"]
        length = len(ids)
        padded = (length + 63) // 64 * 64
        if (
            not length
            or padded > MAX_SEQUENCE_TOKENS
            or any(type(token) is not int or token < 0 for token in ids)
        ):
            raise ValueError(
                "reference prompt exceeds the product's complete-row token bound"
            )
        if (
            prompt["slots"] != [length - 1]
            or prompt["padded_sequence_length"] != padded
            or shape != [1, padded]
        ):
            raise ValueError(
                "reference readout position or actual padded shape differs"
            )
    if case["readout_dtypes"] != ["torch.float32"] * len(prompts):
        raise ValueError("reference observed readout dtype differs")
    if profile is not None and profile["typed_temperatures"]:
        temperatures, types = [], []
        for identifier, _, _, count in case["row_index"]:
            kind = case["questions"][identifier]["type"]
            temperatures.extend([profile["temperature_by_type"][kind]] * count)
            types.extend([[kind]] * count)
        if (
            case.get("row_temperatures") != temperatures
            or [row.get("types") for row in prompts] != types
        ):
            raise ValueError("reference per-row type/temperature differs")
        for key in ("linear_input_dtypes", "linear_weight_dtypes"):
            if case.get(key) != ["torch.bfloat16"] * len(prompts):
                raise ValueError("reference linear operand dtype differs")
    return expected


def validate_reference(args, path):
    profile = profile_for(args)
    expected = read_json(path)
    required = {
        "status": "passed",
        "model_id": profile["model"],
        "revision": profile["revision"],
        "source_unchanged": True,
        "sdk_unchanged": True,
        "measured": True,
        "sdk_hashes": profile["sdk_hashes"],
        "checkpoint_hashes": profile["source_hashes"],
        "driver_sha256": digest(args.reference_driver),
        "reference_kind": profile["reference_kind"],
    }
    for key, value in required.items():
        if type(expected.get(key)) is not type(value) or expected[key] != value:
            raise ValueError(f"reference report mismatch: {key}")
    runtime = expected["runtime"]
    for key, value in {
        "threads": args.threads,
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
        "temperature": profile["temperature"],
        "max_fwd_tokens": 1,
        "max_state_tokens": 32768,
        "cross_request_cache": False,
        "raw_probabilities_capture": "original_return_frame.flatp",
    }.items():
        if type(runtime.get(key)) is not type(value) or runtime[key] != value:
            raise ValueError(f"reference runtime mismatch: {key}")
    if profile["typed_temperatures"]:
        for key, value in {
            "model_layout": "plain",
            "chat_template_applied": False,
            "temperature_by_type": profile["temperature_by_type"],
            "actual_linear_input_dtype": "torch.bfloat16",
            "actual_linear_weight_dtype": "torch.bfloat16",
        }.items():
            if type(runtime.get(key)) is not type(value) or runtime[key] != value:
                raise ValueError(f"reference runtime mismatch: {key}")
    current = common.environment()
    for package in ("torch", "transformers", "safetensors"):
        if (
            expected["environment"]["packages"].get(package)
            != current["packages"][package]
        ):
            raise ValueError(f"reference package mismatch: {package}")
    if len(expected["cases"]) != len(CASES):
        raise ValueError("reference case count mismatch")
    for row, requested in zip(expected["cases"], CASES):
        for key, value in requested.items():
            if json.dumps(row[key], ensure_ascii=False) != json.dumps(
                value, ensure_ascii=False
            ):
                raise ValueError(f"reference case mismatch: {key}")
        if row["answers"] != row["response"]["answers"]:
            raise ValueError("reference original response and recorded answers differ")
        row["project_answers"] = validate_author_observations(row, profile)
    return expected


def finish_run(temporary, tree, process, record, source, profile=None):
    """Keep only this run's workspace if any owned child may still be alive."""
    stopped = tree.stop() if tree is not None else process is None
    record.update(private_workspace=temporary.name, owned_process_stopped=stopped)
    if not stopped:
        record.update(
            status="failed",
            private_workspace_removed=False,
            retained_workspace=temporary.name,
        )
        return False
    okay = True
    if source is not None:
        try:
            record["source_unchanged"] = verify_source(source, profile) == (
                SOURCE_HASHES if profile is None else profile["source_hashes"]
            )
        except Exception as error:
            okay = False
            record.update(
                status="failed", source_unchanged=False, preservation_error=str(error)
            )
    try:
        temporary.cleanup()
        record["private_workspace_removed"] = True
    except Exception as error:
        okay = False
        record.update(
            status="failed",
            private_workspace_removed=False,
            retained_workspace=temporary.name,
            cleanup_error=str(error),
        )
    return okay


def run_reference(args, environment, report, output):
    profile = profile_for(args)
    record = {"status": "running"}
    report["reference_process"] = record
    temporary = tempfile.TemporaryDirectory(
        prefix="decider-reference-", dir=args.workspace, delete=False
    )
    process = tree = source = monitor = None
    try:
        source = Path(temporary.name) / "checkpoint"
        clone_source(args.source, source, profile)
        cases = Path(temporary.name) / "cases.json"
        save(cases, CASES)
        command = [
            str(args.reference_python),
            "-B",
            str(args.reference_driver),
            "--model",
            profile["model"],
            "--source",
            str(source),
            "--sdk-source",
            str(args.sdk_source),
            "--cases-json",
            str(cases),
            "--output",
            str(output),
            "--threads",
            str(args.threads),
            "--dtype",
            "bfloat16",
            "--max-fwd-tokens",
            "1",
        ]
        record["command"] = command
        log = args.output.with_name(args.output.stem + "-reference.log")
        with log.open("w", encoding="utf-8") as stream:
            process = subprocess.Popen(
                command,
                cwd=temporary.name,
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            tree = OwnedProcessTree(process, record)
            monitor = MemoryMonitor(tree, args, record)
            monitor.start()
            if tree.wait(args.reference_timeout) != 0:
                raise RuntimeError(f"original Decider reference failed; inspect {log}")
        record["status"] = "passed"
    except BaseException as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        monitored = monitor.stop() if monitor is not None else True
        okay = finish_run(temporary, tree, process, record, source, profile)
        if (not okay or not monitored) and sys.exc_info()[0] is None:
            raise RuntimeError("reference cleanup or source preservation failed")


def verify_response(response, profile=None):
    profile = PROFILES[MODEL] if profile is None else profile
    if response.get("model") != profile["model"]:
        raise AssertionError("response model differs")
    metadata = response.get("metadata", {})
    for key, value in {
        "backend": "decider-pytorch",
        "device": "cpu",
        "revision": profile["revision"],
        "experimental": True,
        "backbone_dtype": "bfloat16",
        "head_dtype": "bfloat16",
        "readout_dtype": "bfloat16",
        "logits_dtype": "float32",
        "row_batch_size": 1,
        "padding_multiple": 64,
        "max_sequence_tokens": MAX_SEQUENCE_TOKENS,
    }.items():
        if metadata.get(key) != value:
            raise AssertionError(f"product runtime metadata differs: {key}")


def service_run(args, expected, environment, report, number):
    profile = profile_for(args)
    model = profile["model"]
    record = {
        "status": "running",
        "round": number,
        "route": "model_id" if number == 1 else "local_unicode_path",
        "cases": [],
        "memory": [],
    }
    report["service_runs"].append(record)
    record["memory_gate"] = memory_gate(args)
    if not record["memory_gate"]["passed"]:
        record["status"] = "skipped"
        raise RuntimeError("available memory precheck failed before product startup")
    temporary = tempfile.TemporaryDirectory(
        prefix="decider-cli-", dir=args.workspace, delete=False
    )
    process = tree = source = monitor = None
    started = time.perf_counter()
    try:
        workspace = Path(temporary.name)
        source = (
            workspace / "checkpoint" / model
            if number == 1
            else workspace / "本地 模型" / model.split("/")[-1]
        )
        clone_source(args.source, source, profile)
        command = [
            sys.executable,
            "-B",
            "-m",
            "vllm_jev.cli",
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
        log = args.output.with_name(args.output.stem + f"-server-{number}.log")
        url = f"http://127.0.0.1:{args.port}"
        with log.open("w", encoding="utf-8") as stream:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env={**environment, "PYTHONPATH": str(ROOT)},
                stdout=stream,
                stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            tree = OwnedProcessTree(process, record)
            monitor = MemoryMonitor(tree, args, record)
            monitor.start()
            deadline = time.monotonic() + args.startup_timeout
            while time.monotonic() < deadline:
                tree.refresh()
                if process.poll() is not None:
                    raise RuntimeError(f"CLI exited before readiness; inspect {log}")
                try:
                    status, health, _ = http(url, "/health", timeout=1)
                    if status == 200:
                        break
                except (OSError, ValueError):
                    pass
                time.sleep(0.2)
            else:
                raise TimeoutError("Decider CLI readiness timeout")
            for key, value in {
                "model": model,
                "revision": profile["revision"],
                "backend": "decider-pytorch",
                "device": "cpu",
                "experimental": True,
            }.items():
                if health.get(key) != value:
                    raise AssertionError(f"health identity differs: {key}")
            owned = tree.listener(args.port)
            record.update(health=health, ready_seconds=time.perf_counter() - started)
            record["memory"].append(memory_sample(owned, "ready", started))
            for case in expected["cases"]:
                payload = {
                    "model": model,
                    "state": case["state"],
                    "questions": case["questions"],
                }
                status, response, elapsed = http(
                    url, "/v1/systemone", payload, args.request_timeout
                )
                row = {
                    "id": case["id"],
                    "status": status,
                    "seconds": elapsed,
                    "response": response,
                }
                record["cases"].append(row)
                if status != 200:
                    raise AssertionError(f"valid request failed: {case['id']}")
                verify_response(response, profile)
                row["max_absolute_difference"] = compare_answers(
                    case["project_answers"], response.get("answers"), TOLERANCE
                )
                if number == 2:
                    previous = report["service_runs"][0]["cases"][
                        len(record["cases"]) - 1
                    ]
                    compare_answers(
                        previous["response"]["answers"],
                        response["answers"],
                        tolerance=0,
                    )
                record["memory"].append(memory_sample(owned, case["id"], started))
                save(args.output, report)
                print(
                    json.dumps(
                        {
                            "event": "service_case_completed",
                            "round": number,
                            "id": case["id"],
                            "seconds": elapsed,
                        }
                    ),
                    flush=True,
                )
            first = expected["cases"][0]
            base = {
                "model": model,
                "state": first["state"],
                "questions": first["questions"],
            }
            invalid = {
                "cache_salt": {**base, "cache_salt": "caller"},
                "image": {**base, "images": ["data:image/png;base64,AA=="]},
                "video": {**base, "videos": ["data:video/mp4;base64,AA=="]},
                "model": {**base, "model": "unloaded/model"},
                "question_type": {**base, "questions": {"q": {"type": "unsupported"}}},
                "empty_questions": {**base, "questions": {}},
                "overlength": {**base, "state": "token " * (MAX_SEQUENCE_TOKENS + 100)},
            }
            record["rejections"] = {}
            for key, payload in invalid.items():
                status, response, elapsed = http(url, "/v1/systemone", payload, 15)
                record["rejections"][key] = {
                    "status": status,
                    "seconds": elapsed,
                    "response": response,
                }
                if status != 422:
                    raise AssertionError(f"unsupported input accepted: {key}")
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as clients:
                pending = [
                    clients.submit(
                        http, url, "/v1/systemone", base, args.request_timeout
                    )
                    for _ in range(2)
                ]
                time.sleep(0.05)
                unfinished = sum(not future.done() for future in pending)
                status, health, elapsed = http(url, "/health", timeout=5)
                after = sum(not future.done() for future in pending)
                record["concurrent_health"] = {
                    "status": status,
                    "seconds": elapsed,
                    "unfinished_before": unfinished,
                    "unfinished_after": after,
                }
                responses = [future.result() for future in pending]
            if status != 200 or unfinished < 1 or after < 1:
                raise AssertionError(
                    "health was not observed while decision requests were unfinished"
                )
            record["concurrent_requests"] = []
            for status, response, elapsed in responses:
                if status != 200:
                    raise AssertionError("queued concurrent request failed")
                verify_response(response, profile)
                difference = compare_answers(
                    first["project_answers"], response.get("answers"), TOLERANCE
                )
                compare_answers(
                    record["cases"][0]["response"]["answers"],
                    response["answers"],
                    tolerance=0,
                )
                record["concurrent_requests"].append(
                    {
                        "status": status,
                        "seconds": elapsed,
                        "max_absolute_difference": difference,
                        "deterministic": True,
                    }
                )
            record["memory"].append(memory_sample(owned, "after_concurrency", started))
            record.update(
                status="passed",
                peak_wset_bytes=max(s["peak_wset_bytes"] for s in record["memory"]),
                peak_commit_bytes=max(s["peak_commit_bytes"] for s in record["memory"]),
            )
    except BaseException as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        monitored = monitor.stop() if monitor is not None else True
        okay = finish_run(temporary, tree, process, record, source, profile)
        record["wall_seconds"] = time.perf_counter() - started
        if (not okay or not monitored) and sys.exc_info()[0] is None:
            raise RuntimeError("product cleanup or source preservation failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=list(PROFILES), default=MODEL)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sdk-source", type=Path, required=True)
    parser.add_argument(
        "--reference-driver",
        type=Path,
        default=REFERENCE_RUNNER,
        help="Repository runner by default. For legacy report reuse, select the original external runner to verify its recorded SHA.",
    )
    parser.add_argument("--reference-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--reference-report", type=Path)
    parser.add_argument("--reference-only", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--port", type=int, default=18808)
    parser.add_argument("--startup-timeout", type=float, default=600)
    parser.add_argument("--request-timeout", type=float, default=300)
    parser.add_argument("--reference-timeout", type=float, default=1800)
    parser.add_argument(
        "--min-available-gib",
        type=float,
        help="Startup available RAM minimum: 4 GiB for 0.8B, 6 GiB for 2B.",
    )
    parser.add_argument(
        "--min-commit-available-gib",
        type=float,
        help="Startup commit headroom minimum: 6 GiB for 0.8B, 8 GiB for 2B.",
    )
    parser.add_argument("--abort-below-available-gib", type=float, default=0.75)
    args = parser.parse_args()
    profile = profile_for(args)
    for key in ("min_available_gib", "min_commit_available_gib"):
        if getattr(args, key) is None:
            setattr(args, key, profile[key])
    if args.reference_only and args.reference_report:
        parser.error("reference-only and reference-report are mutually exclusive")
    for key in (
        "startup_timeout",
        "request_timeout",
        "reference_timeout",
        "min_available_gib",
        "min_commit_available_gib",
        "abort_below_available_gib",
    ):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            parser.error(f"{key} must be positive and finite")
    if args.abort_below_available_gib >= args.min_available_gib:
        parser.error("abort threshold must be below the startup memory threshold")
    if not 1 <= args.threads <= (os.cpu_count() or 1) or not 1 <= args.port <= 65535:
        parser.error("invalid CPU threads or port")
    for key in (
        "source",
        "workspace",
        "output",
        "sdk_source",
        "reference_driver",
        "reference_python",
        "reference_report",
    ):
        if getattr(args, key) is not None:
            setattr(args, key, getattr(args, key).resolve())
    if args.reference_report == args.output:
        parser.error("output must not overwrite the reference report")
    report = {
        "status": "running",
        "model": profile["model"],
        "revision": profile["revision"],
        "measured": False,
        "environment": common.environment(),
        "threads": args.threads,
        "scope": "Three synthetic cases: English mixed, Chinese JSON mixed, and 12-option Choice. Original BF16 probability parity with project confidence formulas; not task quality or throughput.",
        "source": str(args.source),
        "source_sha256": profile["source_hashes"],
        "sdk_sha256": profile["sdk_hashes"],
        "service_runs": [],
        "tolerance": TOLERANCE,
        "reference_only": args.reference_only,
        "driver_sha256": digest(Path(__file__)),
        "process_helper_sha256": digest(Path(common.__file__)),
        "presentation_difference": "Author rounds probabilities to 4 decimals and Score to 2. Product uses unrounded probabilities and project confidence formulas. "
        + (
            "Author v11 uses the same confidence formulas and reports max probability as x_p_max."
            if profile["typed_temperatures"]
            else "Author 0.8b reports max probability as confidence, unlike the project."
        ),
    }
    started, before = time.perf_counter(), None
    try:
        if sys.platform != "win32":
            report.update(status="skipped", reason="Native Windows is required")
            return
        report["memory_gate"] = memory_gate(args)
        if not report["memory_gate"]["passed"]:
            report.update(
                status="skipped",
                reason="Insufficient currently available memory; no model process was started",
            )
            return
        args.workspace.mkdir(parents=True, exist_ok=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        before = verify_source(args.source, profile)
        verify_sdk(args.sdk_source, profile)
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", args.port))
        environment = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith("JEV_") and key != "VLLM_JEV_DECISION_TEMPLATE"
        }
        environment.update(
            PYTHONPATH="",
            PYTHONDONTWRITEBYTECODE="1",
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            HF_HUB_DISABLE_TELEMETRY="1",
            TOKENIZERS_PARALLELISM="false",
            USE_TF="0",
            USE_FLAX="0",
        )
        output = args.reference_report or args.output.with_name(
            args.output.stem + "-original-reference.json"
        )
        if not args.reference_report:
            run_reference(args, environment, report, output)
        expected = validate_reference(args, output)
        report.update(
            measured=True,
            original_reference=str(output),
            reference_provenance={
                "sha256": digest(output),
                "reused": bool(args.reference_report),
                "rerun_this_invocation": not bool(args.reference_report),
            },
            reference_runtime=expected["runtime"],
        )
        if not args.reference_only:
            for number in (1, 2):
                service_run(args, expected, environment, report, number)
            rows = [r for run in report["service_runs"] for r in run["cases"]]
            concurrent = [
                r for run in report["service_runs"] for r in run["concurrent_requests"]
            ]
            report["summary"] = {
                "valid_requests": len(rows) + len(concurrent),
                "sequential_requests": len(rows),
                "sequential_median_seconds": statistics.median(
                    r["seconds"] for r in rows
                ),
                "maximum_absolute_difference": max(
                    r["max_absolute_difference"] for r in rows + concurrent
                ),
                "peak_wset_bytes": max(
                    r["peak_wset_bytes"] for r in report["service_runs"]
                ),
                "peak_commit_bytes": max(
                    r["peak_commit_bytes"] for r in report["service_runs"]
                ),
            }
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if before is not None:
            try:
                report["source_unchanged"] = (
                    verify_source(args.source, profile) == before
                )
                verify_sdk(args.sdk_source, profile)
            except Exception as error:
                report.update(
                    status="failed",
                    source_unchanged=False,
                    preservation_error=str(error),
                )
        report["wall_seconds"] = time.perf_counter() - started
        save(args.output, report)
        print(json.dumps({"status": report["status"], "output": str(args.output)}))
        if report["status"] == "failed" and sys.exc_info()[0] is None:
            raise RuntimeError("validation failed; inspect the saved report")


if __name__ == "__main__":
    main()
