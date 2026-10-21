"""Serial RSI v1 author and installed Windows CPU CLI acceptance.

Runs fixed real weights in two separate environments; never installs anything.
Only process trees created by this driver are stopped.
"""

import argparse
import concurrent.futures
import copy
import json
import math
import os
import socket
import sys
import time
from pathlib import Path

import check_windows_candidates as helpers
import windows_rsi_reference as reference

common = helpers.common
MODEL = reference.MODEL_ID
BACKEND = "rsi-pytorch"
RUNNER = Path(__file__).with_name("windows_rsi_reference.py")
TOLERANCE = 1e-5
CASES = copy.deepcopy(common.CASES)
CASES.extend(
    [
        {
            "id": "max_80_options",
            "state": "Choose option k42.",
            "questions": {
                "selection": {
                    "type": "choice",
                    "instructions": "Choose the named option.",
                    "criteria": {f"k{i}": "" for i in range(80)},
                }
            },
        },
        {
            "id": "json_criteria_and_score_10",
            "state": [{"message": "请立即退款", "id": 7}],
            "questions": {
                "route": {
                    "type": "choice",
                    "instructions": {"task": "选择部门"},
                    "criteria": {"退款": {"工作": "退还多扣的钱"}, "技术": None},
                },
                "urgency": {
                    "type": "score",
                    "instructions": "Rate urgency from zero to nine.",
                    "criteria": [f"Level {i}" for i in range(10)],
                },
                "refund": {
                    "type": "noul",
                    "instructions": "Is a refund requested?",
                    "criteria": {"false": "No refund", "true": "Refund"},
                },
            },
        },
    ]
)


def ordered(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def validate_reference(path, args, cases_path):
    import numpy as np

    report = helpers._strict_reference_json(path.read_text(encoding="utf-8"))
    for flag in (
        "measured",
        "source_unchanged",
        "sdk_unchanged",
        "private_source_unchanged",
        "private_sdk_unchanged",
        "private_base_unchanged",
        "private_workspace_removed",
    ):
        if report.get(flag) is not True:
            raise ValueError(f"Original reference evidence missing: {flag}")
    identity = {
        "status": "passed",
        "model": MODEL,
        "revision": reference.REVISION,
        "base_revision": reference.BASE_REVISION,
        "sdk_revision": reference.SDK_REVISION,
        "runner_sha256": common.digest(RUNNER),
        "reference_helpers_sha256": common.digest(Path(reference.common.__file__)),
        "cases_sha256": common.digest(cases_path),
        "source_hashes": reference.SOURCE_HASHES,
        "sdk_hashes": reference.SDK_HASHES,
    }
    for name, value in identity.items():
        if ordered(report.get(name)) != ordered(value):
            raise ValueError(f"Original reference identity differs: {name}")
    for name, value in {
        "device": "cpu",
        "tower_dtype": "float32",
        "scorer_dtype": "float32",
        "temperature": 1.0,
        "attention_implementation": "sdpa",
        "use_cache": True,
        "threads": args.threads,
        "batch_size": 1,
        "max_options": 80,
    }.items():
        if ordered(report.get("runtime", {}).get(name)) != ordered(value):
            raise ValueError(f"Original runtime differs: {name}")
    installed = helpers._reference_versions(
        args.reference_python, tuple(report["versions"])
    )
    if (
        installed != report["versions"]
        or installed["torch"] != "2.13.0+cpu"
        or installed["transformers"] != "5.17.0"
    ):
        raise ValueError("Original reference environment changed")
    if report["executable"].casefold() != str(args.reference_python).casefold():
        raise ValueError("Original reference interpreter differs")
    expected, audits = [], []
    for case, observed in zip(CASES, report["cases"], strict=True):
        if (
            ordered({key: observed.get(key) for key in case}) != ordered(case)
            or not observed["state_not_truncated"]
            or not observed["original_forward_inputs_verified"]
        ):
            raise ValueError("Original reference changed input or truncated it")
        deltas = []
        for name, question in case["questions"].items():
            n = 2 if question["type"] == "noul" else len(question["criteria"])
            logits = np.asarray(observed["raw_logits"][name], dtype=np.float32)
            if logits.shape != (n,) or not np.isfinite(logits).all():
                raise ValueError("Original raw logits are invalid")
            exp = np.exp(logits - logits.max())
            softmax = exp / exp.sum(dtype=np.float32)
            p = observed["probabilities"][name]
            if len(p) != n or any(not math.isfinite(v) or v < 0 for v in p):
                raise ValueError("Original full option probabilities are invalid")
            delta = float(np.max(np.abs(softmax - p)))
            if delta > 1e-6 or abs(sum(p) - 1) > 1e-6:
                raise ValueError("Original probabilities disagree with FP32 softmax")
            deltas.append(delta)
            readout = observed["original_readouts"][name]
            keys = (
                ["false", "true"]
                if question["type"] == "noul"
                else list(question["criteria"])
                if question["type"] == "choice"
                else [str(i) for i in range(n)]
            )
            if (
                readout["choice"] != keys[max(range(n), key=p.__getitem__)]
                or abs(readout["score"] - sum(i * v for i, v in enumerate(p))) > 1e-7
            ):
                raise ValueError("Original typed readout differs")
            if question["type"] == "noul" and readout["noul"] != p[1]:
                raise ValueError("Original Noul is not p(true)")
        expected.append(
            helpers.canonical(case, observed["probabilities"], true_index=1)
        )
        audits.append({"id": case["id"], "softmax_max_delta": max(deltas)})
    return report, expected, audits


def check_response(status, response, expected, threads):
    if status != 200 or response.get("model") != MODEL:
        raise ValueError(f"System One failed: {status}: {response}")
    for key, value in {
        "backend": BACKEND,
        "model": MODEL,
        "device": "cpu",
        "experimental": True,
        "revision": reference.REVISION,
        "backbone_dtype": "float32",
        "head_dtype": "float32",
        "logits_dtype": "float32",
        "row_batch_size": 1,
        "temperature": 1.0,
        "max_sequence_tokens": 1024,
        "option_pool": "mean",
        "readout": "option_xattn",
        "attention_implementation": "sdpa",
        "use_cache": True,
        "cache_reuse": False,
        "scorer_slots": 80,
        "threads": threads,
    }.items():
        if ordered(response.get("metadata", {}).get(key)) != ordered(value):
            raise ValueError(f"Product metadata differs: {key}")
    if response.get("usage", {}).get("output_tokens") != 0:
        raise ValueError("Decision engine generated tokens")
    return common.compare_answers(expected, response.get("answers"), TOLERANCE)


def compare_encoding(args, environment, report, original, cases_path):
    """Compare product blocks with observed author inputs after its process exits."""
    probe = Path(__file__).with_name("windows_rsi_encoding.py")
    output = args.output.with_name(args.output.stem + "-encoding.json")
    command = [
        str(Path(sys.executable)),
        "-B",
        str(probe),
        "--source",
        str(args.source),
        "--cases-json",
        str(cases_path),
        "--output",
        str(output),
    ]
    record = {"command": command}
    report["encoding_process"] = record

    def wait_probe(process, tree):
        if tree.wait(min(args.timeout, 180)) != 0:
            raise RuntimeError("Product tokenizer probe failed; inspect its log")

    helpers.reference_run(
        args,
        command,
        environment,
        args.output.with_name(args.output.stem + "-encoding.log"),
        record,
        wait_probe,
    )
    result = helpers._strict_reference_json(output.read_text(encoding="utf-8"))
    module = Path(__file__).resolve().parents[1] / "vllm_jev/windows_rsi.py"
    for key, value in {
        "status": "passed",
        "model": MODEL,
        "revision": reference.REVISION,
        "python": sys.executable,
        "product_module": str(module),
        "product_sha256": common.digest(module),
        "probe_sha256": common.digest(probe),
        "cases_sha256": common.digest(cases_path),
        "tokenizer_sha256": reference.SOURCE_HASHES["base/tokenizer.json"],
    }.items():
        if ordered(result.get(key)) != ordered(value):
            raise ValueError(f"Product tokenizer probe identity differs: {key}")
    count = 0
    for case, author, product in zip(
        CASES, original["cases"], result["cases"], strict=True
    ):
        if case["id"] != product["id"]:
            raise ValueError("Product tokenizer case order differs")
        for key, expected, actual in zip(
            case["questions"], author["encoded"], product["encoded"], strict=True
        ):
            if set(actual) != {
                "id",
                "input_ids",
                "option_span",
                "decision_index",
                "options",
                "mode",
            }:
                raise ValueError("Product tokenizer probe is missing encoded fields")
            if actual.pop("id") != key or any(
                ordered(actual[name]) != ordered(expected[name]) for name in actual
            ):
                raise ValueError(
                    "Product tokens, spans, decision position or option order differ"
                )
            count += 1
    report["encoding_comparison"] = {
        "status": "passed",
        "questions": count,
        "exact": True,
        "report": str(output),
        "sha256": common.digest(output),
    }


def rejection_cases(base):
    def question(kind, criteria, **extra):
        return {
            **base,
            "questions": {
                "q": {
                    "type": kind,
                    "instructions": "Choose.",
                    "criteria": criteria,
                    **extra,
                }
            },
        }

    return {
        "cache_salt": {**base, "cache_salt": "unexpected-cache"},
        "images": {**base, "images": ["data:image/png;base64,AA=="]},
        "videos": {**base, "videos": ["data:video/mp4;base64,AA=="]},
        "embedded_media": {
            **base,
            "state": {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": "data:image/png;base64,AA=="},
                            }
                        ],
                    }
                ],
            },
        },
        "wrong_model": {**base, "model": "wrong/model"},
        "empty_questions": {**base, "questions": {}},
        "overlength_state": {**base, "state": "word " * 3000},
        "later_overlength_question": {
            **base,
            "questions": {
                "short": {"type": "noul", "instructions": "Is a refund requested?"},
                "too_long": {
                    "type": "choice",
                    "instructions": "Pick.",
                    "criteria": {"a": "word " * 3000, "b": "short"},
                },
            },
        },
        "choice_81": question("choice", [str(i) for i in range(81)]),
        "duplicate_choices": question("choice", ["x", "x"]),
        "empty_choices": question("choice", []),
        "illegal_isolated": question("score", ["low", "high"], isolated=False),
        "score_11": question("score", [str(i) for i in range(11)]),
        "noul_bad_criteria": question("noul", {"maybe": "uncertain"}),
        "questions_65": {
            **base,
            "questions": {
                str(i): {"type": "noul", "instructions": "Choose."} for i in range(65)
            },
        },
        "candidates_320": {
            **base,
            "questions": {
                str(i): {
                    "type": "choice",
                    "instructions": "Choose.",
                    "criteria": [str(j) for j in range(80)],
                }
                for i in range(4)
            },
        },
    }


def service_run(args, environment, report, expected, number):
    record = {
        "round": number,
        "status": "running",
        "cases": [],
        "rejections": {},
        "concurrent_requests": [],
        "memory": [],
    }
    report["service_runs"].append(record)
    with helpers.private_workspace(args.workspace, record) as workspace:
        source = (
            workspace / "checkpoint" / MODEL
            if number == 1
            else workspace / "中文 本地 模型" / "RSI v1"
        )
        helpers.clone(args.source, source, reference.SOURCE_HASHES)
        command = [
            str(args.console_entrypoint),
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
        record["command"] = command
        child_args = argparse.Namespace(**vars(args))
        child_args.workspace = workspace
        started = time.perf_counter()

        def action(process, tree):
            url = f"http://127.0.0.1:{args.port}"
            deadline = time.monotonic() + 300
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(
                        "Installed CLI exited before readiness; inspect its log"
                    )
                try:
                    status, health, _ = common.http(url, "/health", timeout=1)
                    if status == 200:
                        break
                except (OSError, ValueError):
                    pass
                time.sleep(0.2)
            else:
                raise TimeoutError("Installed CLI readiness timed out")
            owned = tree.listener(args.port)
            for key, value in {
                "model": MODEL,
                "revision": reference.REVISION,
                "backend": BACKEND,
                "device": "cpu",
                "experimental": True,
            }.items():
                if health.get(key) != value:
                    raise ValueError(f"CLI health identity differs: {key}")
            record.update(health=health, ready_seconds=time.perf_counter() - started)
            status, models, _ = common.http(url, "/v1/models")
            if status != 200 or models.get("data", [{}])[0].get("id") != MODEL:
                raise ValueError("Installed CLI model listing differs")
            record["models"] = models
            record["memory"].append(common.memory_sample(owned, "ready", started))
            for case, answer in zip(CASES, expected, strict=True):
                payload = {
                    "model": MODEL,
                    "state": case["state"],
                    "questions": case["questions"],
                }
                status, response, seconds = common.http(
                    url, "/v1/systemone", payload, args.timeout
                )
                observation = {
                    "id": case["id"],
                    "status": status,
                    "response": response,
                    "seconds": seconds,
                }
                record["cases"].append(observation)
                common.save(args.output, report)
                observation["comparison"] = check_response(
                    status, response, answer, args.threads
                )
                print(
                    json.dumps(
                        {
                            "event": "case_passed",
                            "round": number,
                            "case": case["id"],
                            "seconds": seconds,
                        }
                    ),
                    flush=True,
                )
            base = {
                "model": MODEL,
                "state": CASES[0]["state"],
                "questions": CASES[0]["questions"],
            }
            for key, payload in rejection_cases(base).items():
                status, response, seconds = common.http(
                    url, "/v1/systemone", payload, 60
                )
                record["rejections"][key] = {
                    "status": status,
                    "response": response,
                    "seconds": seconds,
                }
                common.save(args.output, report)
                if status != 422:
                    raise ValueError(
                        f"Invalid request was not rejected: {key}: {response}"
                    )
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                futures = [
                    pool.submit(common.http, url, "/v1/systemone", base, args.timeout)
                    for _ in range(2)
                ]
                time.sleep(0.1)
                unfinished = sum(not future.done() for future in futures)
                status, _, seconds = common.http(url, "/health", timeout=5)
                after = sum(not future.done() for future in futures)
                record["concurrent_health"] = {
                    "status": status,
                    "seconds": seconds,
                    "unfinished_before": unfinished,
                    "unfinished_after": after,
                }
                if status != 200 or not unfinished or not after:
                    raise ValueError(
                        "Health did not remain available during concurrent real inference"
                    )
                for future in futures:
                    status, response, seconds = future.result()
                    observation = {
                        "status": status,
                        "response": response,
                        "seconds": seconds,
                    }
                    record["concurrent_requests"].append(observation)
                    common.save(args.output, report)
                    observation["comparison"] = check_response(
                        status, response, expected[0], args.threads
                    )
            tree.refresh()
            record["memory"].append(common.memory_sample(owned, "completed", started))
            record["all_process_memory"] = [
                common.memory_sample(item, "completed", started)
                for item in tree.living()
            ]

        helpers.child_run(
            child_args,
            command,
            environment,
            args.output.with_name(args.output.stem + f"-server-{number}.log"),
            record,
            action,
        )
        record["source_unchanged"] = (
            reference.verify_checkpoint(source) == reference.SOURCE_HASHES
        )
        if not record["source_unchanged"]:
            raise ValueError("CLI changed its private checkpoint")
    record["status"] = "passed"
    common.save(args.output, report)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "sdk-source", "reference-python", "workspace", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument(
        "--console-entrypoint",
        type=Path,
        default=Path(sys.executable).with_name("vllm-jev.exe"),
    )
    parser.add_argument("--reference-only", action="store_true")
    parser.add_argument("--reference-report", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--port", type=int, default=18835)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--min-available-gib", type=float, default=6)
    parser.add_argument("--min-commit-available-gib", type=float, default=8)
    parser.add_argument("--abort-below-available-gib", type=float, default=1.5)
    args = parser.parse_args()
    for name in (
        "source",
        "sdk_source",
        "reference_python",
        "workspace",
        "output",
        "console_entrypoint",
        "reference_report",
    ):
        if getattr(args, name) is not None:
            setattr(args, name, getattr(args, name).resolve())
    if (
        sys.platform != "win32"
        or not 1 <= args.threads <= 4
        or not 1 <= args.port <= 65535
    ):
        parser.error("Native Windows, 1-4 threads and a valid port are required")
    if any(
        not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0
        for key in (
            "timeout",
            "min_available_gib",
            "min_commit_available_gib",
            "abort_below_available_gib",
        )
    ):
        parser.error("Memory and timeout policies must be finite positive values")
    if (
        args.min_available_gib < 6
        or args.min_commit_available_gib < 8
        or not 1.5 <= args.abort_below_available_gib < args.min_available_gib
    ):
        parser.error("Keep the 6 GiB available / 8 GiB commit / 1.5 GiB abort policy")
    if args.reference_only and args.reference_report:
        parser.error("reference-only and reference-report are mutually exclusive")
    outputs = {
        args.output,
        args.output.with_name(args.output.stem + "-cases.json"),
        args.output.with_name(args.output.stem + "-author.json"),
        args.output.with_name(args.output.stem + "-encoding.json"),
    }
    if outputs & {args.reference_python, args.console_entrypoint} or (
        args.reference_report is not None and args.reference_report in outputs
    ):
        parser.error("Acceptance output must not overwrite an interpreter or reference")
    if any(
        args.workspace.is_relative_to(root) or args.output.is_relative_to(root)
        for root in (args.source, args.sdk_source)
    ):
        parser.error("Outputs must not overwrite original inputs")
    return args


def main():
    args = parse_args()
    args.workspace.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "running",
        "model": MODEL,
        "revision": reference.REVISION,
        "service_runs": [],
        "source": str(args.source),
        "tolerance": TOLERANCE,
        "driver_sha256": common.digest(Path(__file__)),
        "environment": common.common.environment(),
        "reference_only": args.reference_only,
    }
    try:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", args.port))
        report["source_hashes"] = reference.verify_checkpoint(args.source)
        report["sdk_hashes"] = reference.verify_sdk(args.sdk_source)
        environment = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith("JEV_") and k != "VLLM_JEV_DECISION_TEMPLATE"
        }
        environment.update(
            PYTHONPATH="",
            PYTHONDONTWRITEBYTECODE="1",
            PYTHONIOENCODING="utf-8",
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            TOKENIZERS_PARALLELISM="false",
            HF_HOME=str(args.workspace / "hf-home"),
            NO_PROXY="127.0.0.1,localhost",
        )
        cases_path = args.output.with_name(args.output.stem + "-cases.json")
        common.save(cases_path, CASES)
        refpath = args.reference_report or args.output.with_name(
            args.output.stem + "-author.json"
        )
        if args.reference_report is None:
            command = [
                str(args.reference_python),
                "-B",
                str(RUNNER),
                "--source",
                str(args.source),
                "--sdk-source",
                str(args.sdk_source),
                "--cases-json",
                str(cases_path),
                "--output",
                str(refpath),
                "--threads",
                str(args.threads),
            ]
            record = {"command": command}
            report["reference_process"] = record

            def wait_reference(process, tree):
                if tree.wait(args.timeout) != 0:
                    raise RuntimeError(
                        "Original author reference failed; inspect its log"
                    )

            helpers.reference_run(
                args,
                command,
                environment,
                args.output.with_name(args.output.stem + "-author.log"),
                record,
                wait_reference,
            )
        original, expected, audits = validate_reference(refpath, args, cases_path)
        report.update(
            reference_report=str(refpath),
            reference_sha256=common.digest(refpath),
            reference_reused=args.reference_report is not None,
            reference_audits=audits,
        )
        if not args.reference_only:
            _, identity = helpers.product_entrypoint(args.console_entrypoint)
            report["product_entrypoint"] = identity
            environment = helpers.product_environment(environment, identity)
            compare_encoding(args, environment, report, original, cases_path)
            for number in (1, 2):
                service_run(args, environment, report, expected, number)
        report.update(
            status="passed",
            measured=True,
            totals={
                "installed_cli_starts": len(report["service_runs"]),
                "valid_http_requests": sum(
                    len(r["cases"]) + len(r["concurrent_requests"])
                    for r in report["service_runs"]
                ),
                "rejected_http_requests": sum(
                    len(r["rejections"]) for r in report["service_runs"]
                ),
                "original_reference_cases": len(audits),
            },
        )
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        try:
            report["source_unchanged"] = reference.verify_checkpoint(
                args.source
            ) == report.get("source_hashes")
            report["sdk_unchanged"] = reference.verify_sdk(
                args.sdk_source
            ) == report.get("sdk_hashes")
        except Exception as error:
            report.update(status="failed", preservation_error=str(error))
        if report["status"] == "passed" and not all(
            report.get(k) for k in ("source_unchanged", "sdk_unchanged")
        ):
            report.update(status="failed", error="Source preservation failed")
        report["parent_torch_imported"] = "torch" in sys.modules
        if report["parent_torch_imported"]:
            report.update(status="failed", error="Driver unexpectedly imported Torch")
        common.save(args.output, report)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
