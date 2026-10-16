"""Real CPU acceptance against independent public-model-card reference (not author SDK)."""

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
import windows_decision_gguf_reference as reference

common = helpers.common
MODEL = "evalengine/decision-0.8b-gguf"
BACKEND = "decision-llama-cpp-0.3.35"
RUNNER = Path(__file__).with_name("windows_decision_gguf_reference.py")
TOLERANCE = 1e-5
EXPECTED_CONTEXT = None
CASES = copy.deepcopy(common.CASES)
CASES += [
    {
        "id": "all_24_options",
        "state": "The requested number is 23.",
        "questions": {
            "number": {
                "type": "choice",
                "instructions": "Select the requested number.",
                "criteria": {str(i): f"Number {i}" for i in range(24)},
            }
        },
    },
    {
        "id": "explicit_mapping",
        "state": [{"message": "Please refund this duplicate payment."}],
        "questions": {
            "n": {
                "type": "noul",
                "instructions": "Is a refund requested?",
                "criteria": {
                    "true": "A refund is requested",
                    "false": "No refund requested",
                },
            },
            "s": {
                "type": "score",
                "instructions": "Rate urgency.",
                "criteria": ["Low", "Medium", "High"],
                "isolated": False,
            },
            "list": {
                "type": "choice",
                "instructions": "Select relevant topic.",
                "criteria": ["refund", "delivery"],
            },
        },
    },
]


CASES.append(
    {
        "id": "empty_and_duplicate_descriptions",
        "state": "The message requests a refund.",
        "questions": {
            "c": {
                "type": "choice",
                "instructions": "Select a matching topic.",
                "criteria": {"refund": "", "a": "same", "b": "same"},
            },
            "n": {
                "type": "noul",
                "instructions": "Is a refund requested?",
                "criteria": {"true": "", "false": ""},
            },
            "s": {
                "type": "score",
                "instructions": "Rate the urgency.",
                "criteria": ["same", "same"],
            },
        },
    }
)


def ordered(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def validate_reference(path, args, cases_path):
    global EXPECTED_CONTEXT
    import numpy as np

    report = json.loads(path.read_text(encoding="utf-8"))
    checks = {
        "status": "passed",
        "model": MODEL,
        "revision": reference.MODELS[MODEL]["revision"],
        "reference_kind": "independent_public_model_card",
        "original_author_sdk": False,
        "driver_sha256": common.digest(RUNNER),
        "input_sha256": common.digest(cases_path),
        "source_hashes": reference.verify_checkpoint(args.source, MODEL),
        "source_unchanged": True,
        "llama_runtime_unchanged": True,
        "forbidden_modules_imported": False,
        "measured": True,
    }
    for key, value in checks.items():
        if ordered(report.get(key)) != ordered(value):
            raise ValueError(f"Independent reference identity differs: {key}")
    if Path(report["environment"]["executable"]).resolve() != args.reference_python:
        raise ValueError("Reference interpreter differs")
    runtime = reference.verify_llama_wheel(args.llama_wheel)
    if report["llama_runtime"]["package_hashes"] != runtime["package_hashes"]:
        raise ValueError("Product and reference native runtime bytes differ")
    if report["runtime"] != {
        "device": "cpu",
        "threads": args.threads,
        "n_gpu_layers": 0,
        "n_ctx": 2048,
        "max_sequence_tokens": 1024,
        "temperature": 1.0,
        "memory_clear_before_decode": True,
        "readout": "all_option_logits",
        "softmax_dtype": "float32",
    }:
        raise ValueError("Independent reference runtime semantics differ")
    if len(report["cases"]) != len(CASES):
        raise ValueError("Independent reference case count differs")
    EXPECTED_CONTEXT = report["native_context"]
    if set(EXPECTED_CONTEXT) != set(reference.CONTEXT_FIELDS):
        raise ValueError("Reference native context configuration is incomplete")
    answers, audits, nonuniform_rows = [], [], 0
    for case, observed in zip(CASES, report["cases"], strict=True):
        if ordered({k: observed[k] for k in case}) != ordered(case):
            raise ValueError("Reference case or option order differs")
        mapped = reference.task_rows(case)
        if len(observed["rows"]) != len(mapped):
            raise ValueError("Reference rows are incomplete")
        raw_probabilities = {}
        for (identifier, task), row in zip(mapped, observed["rows"], strict=True):
            prompt = (
                "<|im_start|>system\n"
                + report["system_from_pinned_model_card"]
                + "<|im_end|>\n<|im_start|>user\n"
                + json.dumps(task, ensure_ascii=False, allow_nan=False)
                + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            )
            if (
                row["id"] != identifier
                or row["prompt"] != prompt
                or not 1 <= len(row["prompt_tokens"]) <= 1024
            ):
                raise ValueError(
                    "Embedded template and documented no-thinking prompt differ"
                )
            count = len(task["options"])
            if len(row["label_ids"]) != count or len(set(row["label_ids"])) != count:
                raise ValueError("Reference is missing candidate token IDs")
            logits = np.asarray(row["logits"], dtype=np.float32)
            if logits.shape != (count,) or not np.isfinite(logits).all():
                raise ValueError("Reference lacks complete finite logits")
            nonuniform_rows += int(np.ptp(logits) > 1e-6)
            weights = np.exp(logits - logits.max(), dtype=np.float32)
            probabilities = (weights / weights.sum(dtype=np.float32)).tolist()
            if not np.allclose(probabilities, row["probabilities"], atol=1e-7, rtol=0):
                raise ValueError("Reference FP32 normalization differs")
            raw_probabilities[identifier] = probabilities
        if ordered(raw_probabilities) != ordered(observed["probabilities"]):
            raise ValueError("Reference presentation differs from raw logits")
        answers.append(helpers.canonical(case, raw_probabilities, 1))
        audits.append(
            {
                "id": case["id"],
                "complete_logits_verified": True,
                "embedded_template_verified": True,
            }
        )
    if not nonuniform_rows:
        raise ValueError(
            "All reference logits are uniform; check the native output buffer"
        )
    return answers, audits


def check_response(status, response, expected, threads):
    if status != 200:
        raise ValueError(f"System One failed: {status}: {response}")
    if response.get("model") != MODEL:
        raise ValueError("Product response model differs")
    meta = response.get("metadata", {})
    for key, value in {
        "backend": BACKEND,
        "model": MODEL,
        "device": "cpu",
        "experimental": True,
        "revision": reference.MODELS[MODEL]["revision"],
        "quantization": reference.MODELS[MODEL]["quantization"],
        "logits_dtype": "float32",
        "row_batch_size": 1,
        "threads": threads,
        "n_gpu_layers": 0,
        "max_sequence_tokens": 1024,
        "context_tokens": 2048,
        "native_context": EXPECTED_CONTEXT,
    }.items():
        if ordered(meta.get(key)) != ordered(value):
            raise ValueError(f"Product identity differs: {key}: {meta.get(key)!r}")
    if response.get("usage", {}).get("output_tokens") != 0:
        raise ValueError("Decision engine unexpectedly generated output tokens")
    return common.compare_answers(expected, response.get("answers"), TOLERANCE)


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
        "choice_25": question("choice", [str(i) for i in range(25)]),
        "duplicate_choices": question("choice", ["x", "x"]),
        "empty_choices": question("choice", []),
        "illegal_isolated": question("score", ["low", "high"], isolated=True),
        "score_11": question("score", [str(i) for i in range(11)]),
        "noul_bad_criteria": question("noul", {"maybe": "uncertain"}),
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
            else workspace / "中文 本地 模型" / "Decision 模型"
        )
        helpers.clone(args.source, source, reference.MODELS[MODEL]["files"])
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
                "revision": reference.MODELS[MODEL]["revision"],
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
            reference.verify_checkpoint(source, MODEL)
            == reference.MODELS[MODEL]["files"]
        )
        if not record["source_unchanged"]:
            raise ValueError("CLI changed its private checkpoint")
    record["status"] = "passed"
    common.save(args.output, report)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=reference.MODELS, required=True)
    for name in ("source", "reference-python", "llama-wheel", "workspace", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument(
        "--console-entrypoint",
        type=Path,
        default=Path(sys.executable).with_name("vllm-jev.exe"),
    )
    parser.add_argument("--reference-report", type=Path)
    parser.add_argument("--reference-only", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--port", type=int, default=18839)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--min-available-gib", type=float)
    parser.add_argument("--min-commit-available-gib", type=float)
    parser.add_argument("--abort-below-available-gib", type=float, default=1.5)
    args = parser.parse_args()
    floors = (4, 6) if "0.8b" in args.model else (6, 8)
    args.min_available_gib = args.min_available_gib or floors[0]
    args.min_commit_available_gib = args.min_commit_available_gib or floors[1]
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.resolve())
    if (
        sys.platform != "win32"
        or not 1 <= args.threads <= 4
        or not 1 <= args.port <= 65535
    ):
        parser.error("Native Windows, 1-4 threads and a valid port are required")
    if any(
        not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0
        for name in (
            "timeout",
            "min_available_gib",
            "min_commit_available_gib",
            "abort_below_available_gib",
        )
    ):
        parser.error("Finite positive timeout/memory policies required")
    if (
        args.min_available_gib < floors[0]
        or args.min_commit_available_gib < floors[1]
        or not 1.5 <= args.abort_below_available_gib < args.min_available_gib
    ):
        parser.error("Memory thresholds cannot relax the model's supported policy")
    if args.reference_only and args.reference_report:
        parser.error("reference-only and reference-report are mutually exclusive")
    if (
        args.workspace.is_relative_to(args.source)
        or args.output.is_relative_to(args.source)
        or args.output == args.llama_wheel
        or (args.reference_report is not None and args.reference_report == args.output)
    ):
        parser.error("Validation outputs must not overwrite inputs")
    return args


def main():
    global MODEL
    args = parse_args()
    MODEL = args.model
    args.workspace.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "running",
        "model": MODEL,
        "revision": reference.MODELS[MODEL]["revision"],
        "quantization": reference.MODELS[MODEL]["quantization"],
        "service_runs": [],
        "reference_kind": "independent_public_model_card",
        "original_author_sdk": False,
        "tolerance": TOLERANCE,
        "driver_sha256": common.digest(Path(__file__)),
        "environment": common.common.environment(),
    }
    try:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", args.port))
        report["source_hashes"] = reference.verify_checkpoint(args.source, MODEL)
        report["official_runtime"] = reference.verify_llama_wheel(args.llama_wheel)
        env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith("JEV_") and key != "VLLM_JEV_DECISION_TEMPLATE"
        }
        env.update(
            PYTHONPATH="",
            PYTHONDONTWRITEBYTECODE="1",
            PYTHONIOENCODING="utf-8",
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            HF_HOME=str(args.workspace / "hf-home"),
            NO_PROXY="127.0.0.1,localhost",
        )
        cases_path = args.output.with_name(args.output.stem + "-cases.json")
        common.save(cases_path, CASES)
        reference_path = args.reference_report or args.output.with_name(
            args.output.stem + "-reference.json"
        )
        if args.reference_report is None:
            command = [
                str(args.reference_python),
                "-B",
                str(RUNNER),
                "--model",
                MODEL,
                "--source",
                str(args.source),
                "--llama-wheel",
                str(args.llama_wheel),
                "--cases-json",
                str(cases_path),
                "--output",
                str(reference_path),
                "--threads",
                str(args.threads),
            ]
            record = {"command": command}
            report["reference_process"] = record

            def wait_reference(process, tree):
                if tree.wait(args.timeout) != 0:
                    raise RuntimeError(
                        "Independent model-card reference failed; inspect log"
                    )

            helpers.reference_run(
                args,
                command,
                env,
                args.output.with_name(args.output.stem + "-reference.log"),
                record,
                wait_reference,
            )
        expected, audits = validate_reference(reference_path, args, cases_path)
        report.update(
            reference_report=str(reference_path),
            reference_sha256=common.digest(reference_path),
            reference_reused=args.reference_report is not None,
            reference_audits=audits,
        )
        if not args.reference_only:
            _, identity = helpers.product_entrypoint(args.console_entrypoint)
            report["product_entrypoint"] = identity
            env = helpers.product_environment(env, identity)
            for number in (1, 2):
                service_run(args, env, report, expected, number)
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
                "independent_reference_cases": len(audits),
            },
        )
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        try:
            report["source_unchanged"] = reference.verify_checkpoint(
                args.source, MODEL
            ) == report.get("source_hashes")
            report["runtime_unchanged"] = reference.verify_llama_wheel(
                args.llama_wheel
            ) == report.get("official_runtime")
        except Exception as error:
            report.update(source_unchanged=False, preservation_error=str(error))
        if report["status"] == "passed" and not all(
            report.get(k) for k in ["source_unchanged", "runtime_unchanged"]
        ):
            report.update(
                status="failed", error="Source or runtime preservation failed"
            )
        common.save(args.output, report)
    if report["status"] != "passed":
        raise RuntimeError(report.get("error", "Acceptance failed"))


if __name__ == "__main__":
    main()
