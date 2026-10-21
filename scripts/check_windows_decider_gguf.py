"""Real Windows CPU Q4 acceptance through an installed CLI and original SDK.

The reference executes the unmodified author's GGUF engine. This driver audits
its raw FP32 logits, typed temperatures and rounded wire answers independently,
then compares two fresh installed CLI launches. It never imports product model
code to produce a baseline or installs/downloads any dependencies or weights.
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
import windows_decider_gguf_reference as reference

common = helpers.common
MODEL = reference.MODEL_ID
BACKEND = "decider-llama-cpp-q4-0.3.35"
RUNNER = Path(__file__).with_name("windows_decider_gguf_reference.py")
TOLERANCE = 1e-5
CALIBRATION_TOLERANCE = 1e-6
CASES = copy.deepcopy(common.CASES)
CASES.extend(
    [
        {
            "id": "explicit_nonisolated_score",
            "state": "The payment outage blocks every customer and must be fixed immediately.",
            "questions": {
                "severity": {
                    "type": "score",
                    "instructions": "Rate the severity of the issue.",
                    "criteria": [
                        "Low impact",
                        "Some customers affected",
                        "All customers blocked",
                    ],
                    "isolated": False,
                }
            },
        },
        {
            "id": "indexed_json",
            "state": {
                "records": [
                    {"message": "No refund needed", "number": i} for i in range(7)
                ]
                + [{"message": "订单被重复扣款，请立即退款。", "number": 7}],
                "target": "records[7].message",
            },
            "questions": {
                "department": {
                    "type": "choice",
                    "instructions": "Choose the department for records[7].message.",
                    "criteria": {
                        "billing": "Payments and refunds",
                        "technical": "Software issues",
                    },
                },
                "refund": {
                    "type": "noul",
                    "instructions": "Does records[7].message request a refund?",
                    "criteria": {
                        "false": "No refund requested",
                        "true": "A refund is requested",
                    },
                },
                "urgency": {
                    "type": "score",
                    "instructions": "Rate the urgency of records[7].message.",
                    "criteria": [
                        "0: Can wait",
                        "1: Needs attention soon",
                        "2: Immediate request",
                    ],
                },
            },
        },
    ]
)


def ordered(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def expected_rows(case):
    index, types, counts = [], [], []
    for identifier, question in case["questions"].items():
        kind = question["type"]
        count = 2 if kind == "noul" else len(question["criteria"])
        isolated = kind == "score" and question.get("isolated", True)
        nrows = count if isolated else 1
        index.append([identifier, "iso" if isolated else "list", len(types), nrows])
        types.extend([kind] * nrows)
        counts.extend([2 if isolated else count] * nrows)
    return index, types, counts


def independent_answers(case, observed):
    """Map raw author rows using request labels, never product or SDK helpers."""
    index, _, _ = expected_rows(case)
    if observed.get("row_index") != index:
        raise ValueError(
            "Original answer-row order differs from the requested questions"
        )
    probabilities, details = {}, {}
    for identifier, mode, start, count in index:
        question = case["questions"][identifier]
        nopts = 2 if question["type"] == "noul" else len(question["criteria"])
        if mode == "iso":
            fits = [
                float(observed["row_probabilities"][start + i][1]) for i in range(count)
            ]
            mass = sum(fits)
            if mass <= 0:
                raise ValueError("Acceptance case has no positive isolated Score mass")
            p = [fit / mass for fit in fits]
            details[identifier] = {"level_fit": fits, "fit_mass": mass}
        else:
            p = list(observed["row_probabilities"][start][:nopts])
        probabilities[identifier] = p
    return helpers.canonical(case, probabilities, true_index=1), details


def validate_wire(case, observed, expected, details):
    """Bind unrounded observations to actual author system_one API output."""
    raw = observed["response"]["answers"]
    if ordered(raw) != ordered(observed["answers"]) or set(raw) != set(expected):
        raise ValueError("Actual author wire response and recorded answers differ")
    _, _, counts = expected_rows(case)
    if len(observed["row_probabilities"]) != len(counts):
        raise ValueError("Author probability row count differs")
    for identifier, answer in expected.items():
        original = raw[identifier]
        kind = answer["type"]
        if original.get("type") != kind:
            raise ValueError("Author answer type differs")
        row = next(item for item in observed["row_index"] if item[0] == identifier)
        if kind == "noul":
            p = observed["row_probabilities"][row[2]][:2]
            wanted = {"noul": round(p[1] / sum(p), 4)}
        else:
            p = list(answer["probabilities"].values())
            p = [value / sum(p) for value in p]
            selected = max(range(len(p)), key=p.__getitem__)
            confidence_p = [value / sum(p) for value in p]
            n = len(p)
            if kind == "choice":
                confidence = (n * max(confidence_p) - 1) / (n - 1)
            else:
                distance = sum(abs(i - (n - 1) / 2) for i in range(n)) / n
                confidence = (
                    1
                    - sum(
                        value * abs(i - selected)
                        for i, value in enumerate(confidence_p)
                    )
                    / distance
                )
            wanted = {
                "probabilities": dict(
                    zip(answer["probabilities"], (round(value, 4) for value in p))
                ),
                "confidence": round(max(0.0, min(1.0, confidence)), 4),
                "x_p_max": round(max(p), 4),
            }
            if kind == "choice":
                wanted["choice"] = answer["choice"]
            else:
                wanted.update(
                    score=round(sum(i * value for i, value in enumerate(p)), 2),
                    legend=answer["legend"],
                )
        for key, value in wanted.items():
            if ordered(original.get(key)) != ordered(value):
                raise ValueError(
                    f"Author raw probabilities do not explain original wire: {identifier}.{key}"
                )
        if identifier in details:
            values = details[identifier]
            if original.get("level_fit") != {
                str(i): round(p, 4) for i, p in enumerate(values["level_fit"])
            }:
                raise ValueError("Original isolated Score fits differ")
            if original.get("fit_mass") != round(values["fit_mass"], 4):
                raise ValueError("Original isolated Score mass differs")


def validate_logits(case, observed):
    """Recompute calibrated probabilities in NumPy FP32 from observed logits."""
    import numpy as np

    index, types, counts = expected_rows(case)
    prompts = observed.get("prompt_rows", [])
    decodes = observed.get("decode_calls", [])
    rows = observed.get("row_probabilities", [])
    if (
        len(prompts) != len(types)
        or len(decodes) != len(types)
        or len(rows) != len(types)
    ):
        raise ValueError("Reference prompts, decodes and raw rows are incomplete")
    if observed.get("row_index") != index or observed.get("forwards") != len(types):
        raise ValueError("Reference did not execute one independent decode per row")
    wanted_temperatures = [reference.TEMPERATURES[kind] for kind in types]
    if observed.get("row_temperatures") != wanted_temperatures:
        raise ValueError("Reference typed temperatures differ")
    max_delta = 0.0
    for prompt, decode, raw, kind, count, temperature in zip(
        prompts, decodes, rows, types, counts, wanted_temperatures, strict=True
    ):
        ids = prompt.get("ids", [])
        if not 1 <= len(ids) <= 1024 or prompt.get("slots") != [len(ids) - 1]:
            raise ValueError(
                "Reference complete prompt or answer slot exceeds the supported bound"
            )
        if any(type(token) is not int or token < 0 for token in ids):
            raise ValueError("Reference contains invalid prompt token ids")
        if (
            prompt.get("nopts") != [count]
            or prompt.get("types") != [kind]
            or prompt.get("perms") != [list(range(count))]
        ):
            raise ValueError("Reference candidate order or row type differs")
        if (
            decode.get("rows") != [{"ids": ids, "slots": [len(ids) - 1]}]
            or decode.get("readout_dtype") != "float32"
        ):
            raise ValueError("Reference decode does not match its complete prompt")
        logits = np.asarray(decode.get("raw_letter_logits"), dtype=np.float32)
        if logits.shape != (1, 255) or not np.isfinite(logits).all():
            raise ValueError("Reference lacks a finite 255-column FP32 projection")
        if len(raw) != 255 or any(
            type(value) not in (int, float) or not math.isfinite(value) or value < 0
            for value in raw
        ):
            raise ValueError("Reference contains invalid probabilities")
        if any(raw[count:]) or abs(sum(raw[:count]) - 1.0) > 1e-6:
            raise ValueError("Reference masked softmax probability mass differs")
        scaled = logits[0, :count] / np.float32(temperature)
        weight = np.exp(scaled - np.max(scaled))
        calculated = weight / np.sum(weight, dtype=np.float32)
        delta = max(
            abs(float(a) - b) for a, b in zip(calculated, raw[:count], strict=True)
        )
        max_delta = max(max_delta, delta)
        if delta > CALIBRATION_TOLERANCE:
            raise ValueError(
                f"Independent FP32 calibration differs from author softmax: {delta}"
            )
    softmax_calls = observed.get("softmax_calls", [])
    if sum(item.get("shape", [0])[0] for item in softmax_calls) != len(types):
        raise ValueError("Missing author softmax call observations")
    for event in softmax_calls:
        if (
            event.get("input_dtype") != "torch.float32"
            or event.get("output_dtype") != "torch.float32"
            or event.get("device") != "cpu"
            or event.get("shape", [0, 0])[1] != 255
        ):
            raise ValueError("Reference did not use a full-width CPU FP32 softmax")
    return max_delta


def validate_reference(path, args, cases_path):
    report = helpers._strict_reference_json(path.read_text(encoding="utf-8"))
    checks = {
        "status": "passed",
        "model_id": MODEL,
        "revision": reference.REVISION,
        "sdk_revision": reference.SDK_REVISION,
        "reference_kind": "original_decider_sdk_4b_q4_cpu_gguf",
        "checkpoint_hashes": reference.SOURCE_HASHES,
        "sdk_hashes": reference.verify_sdk(args.sdk_source),
        "driver_sha256": common.digest(RUNNER),
        "input_sha256": common.digest(cases_path),
        "source_unchanged": True,
        "sdk_unchanged": True,
        "llama_runtime_unchanged": True,
        "sdk_source_isolated": True,
        "measured": True,
        "forbidden_modules_imported": False,
        "selected_case_ids": [case["id"] for case in CASES],
    }
    for key, value in checks.items():
        if ordered(report.get(key)) != ordered(value):
            raise ValueError(f"Original reference identity differs: {key}")
    env = report.get("environment", {})
    if Path(env.get("executable", "")).resolve() != args.reference_python:
        raise ValueError("Original reference interpreter differs")
    package_names = (
        "torch",
        "transformers",
        "numpy",
        "tokenizers",
        "llama-cpp-python",
        "psutil",
    )
    installed = helpers._reference_versions(args.reference_python, package_names)
    if any(
        env.get("packages", {}).get(name) != installed[name] for name in package_names
    ):
        raise ValueError(
            "Original reference package versions differ from its selected environment"
        )
    wheel = report.get("llama_runtime", {})
    if (
        wheel.get("version") != "0.3.35"
        or wheel.get("wheel_sha256") != reference.WHEEL_SHA256
    ):
        raise ValueError("Original reference runtime is not the official pinned wheel")
    current_runtime = reference.verify_llama_wheel(args.llama_wheel)
    if wheel.get("package_hashes") != current_runtime["package_hashes"]:
        raise ValueError(
            "Reference and product environments do not contain the same official runtime bytes"
        )
    runtime = report.get("runtime", {})
    for key, value in {
        "device": "cpu",
        "n_gpu_layers": 0,
        "threads": args.threads,
        "torch_threads": args.threads,
        "interop_threads": 1,
        "n_ctx": 2048,
        "n_seq_max": 1,
        "row_batch_size": 1,
        "independent": True,
        "padding": "none",
        "actual_readout_dtype": "float32",
        "softmax_dtype": "torch.float32",
        "layout": "state_first",
        "model_layout": "plain",
        "chat_template_applied": False,
        "isolated_levels": True,
        "temperature": 1.099,
        "temperature_by_type": reference.TEMPERATURES,
        "max_sequence_tokens": 1024,
        "use_graphs": False,
        "cross_request_cache": False,
        "memory_clear_before_decode": True,
    }.items():
        if ordered(runtime.get(key)) != ordered(value):
            raise ValueError(f"Original runtime observation differs: {key}")
    if len(report.get("cases", [])) != len(CASES):
        raise ValueError("Original reference case count differs")
    expected, audits = [], []
    for case, observed in zip(CASES, report["cases"], strict=True):
        if ordered({key: observed.get(key) for key in case}) != ordered(case):
            raise ValueError("Original reference input or option order differs")
        if not observed.get("state_not_truncated") or not observed.get(
            "preflight_complete_before_decode"
        ):
            raise ValueError("Original reference did not preflight the complete prompt")
        delta = validate_logits(case, observed)
        answers, details = independent_answers(case, observed)
        validate_wire(case, observed, answers, details)
        # The runner's separately mapped, unrounded presentation is supplementary
        # evidence; the acceptance expectation is rebuilt from raw observations.
        common.compare_answers(answers, observed["canonical_answers"], 1e-6)
        expected.append(answers)
        audits.append(
            {
                "id": case["id"],
                "raw_fp32_calibration_max_delta": delta,
                "author_wire_verified": True,
            }
        )
    return report, expected, audits


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
        "revision": reference.REVISION,
        "quantization": "Q4_K_M",
        "logits_dtype": "float32",
        "row_batch_size": 1,
        "threads": threads,
        "n_gpu_layers": 0,
        "max_sequence_tokens": 1024,
        "context_tokens": 2048,
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
        "choice_256": question("choice", [str(i) for i in range(256)]),
        "duplicate_choices": question("choice", ["x", "x"]),
        "empty_choices": question("choice", []),
        "illegal_isolated": question("score", ["low", "high"], isolated="false"),
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
            else workspace / "中文 本地 模型" / "Decider Q4"
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
    for name in (
        "source",
        "sdk-source",
        "reference-python",
        "llama-wheel",
        "workspace",
        "output",
    ):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument(
        "--console-entrypoint",
        type=Path,
        default=Path(sys.executable).with_name("vllm-jev.exe"),
    )
    parser.add_argument("--reference-only", action="store_true")
    parser.add_argument("--reference-report", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--port", type=int, default=18829)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--min-available-gib", type=float, default=6)
    parser.add_argument("--min-commit-available-gib", type=float, default=8)
    parser.add_argument("--abort-below-available-gib", type=float, default=1.5)
    args = parser.parse_args()
    for name in (
        "source",
        "sdk_source",
        "reference_python",
        "llama_wheel",
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
        parser.error(
            "Keep at least the 6 GiB available / 8 GiB commit / 1.5 GiB abort policy"
        )
    if args.reference_only and args.reference_report:
        parser.error("reference-only and reference-report are mutually exclusive")
    if (
        args.reference_report
        in (args.output, args.output.with_name(args.output.stem + "-cases.json"))
        or args.output == args.llama_wheel
        or any(
            args.workspace.is_relative_to(root) or args.output.is_relative_to(root)
            for root in (args.source, args.sdk_source)
        )
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
        "quantization": "Q4_K_M",
        "tolerance": TOLERANCE,
        "calibration_tolerance": CALIBRATION_TOLERANCE,
        "driver_sha256": common.digest(Path(__file__)),
        "environment": common.common.environment(),
        "reference_only": args.reference_only,
    }
    try:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", args.port))
        report["source_hashes"] = reference.verify_checkpoint(args.source)
        report["sdk_hashes"] = reference.verify_sdk(args.sdk_source)
        report["official_runtime"] = reference.verify_llama_wheel(args.llama_wheel)
        environment = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith("JEV_") and key != "VLLM_JEV_DECISION_TEMPLATE"
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
                "--llama-wheel",
                str(args.llama_wheel),
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
        _, expected, audits = validate_reference(refpath, args, cases_path)
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
            for number in (1, 2):
                service_run(args, environment, report, expected, number)
        report.update(
            status="passed",
            measured=True,
            totals={
                "installed_cli_starts": len(report["service_runs"]),
                "valid_http_requests": sum(
                    len(run["cases"]) + len(run["concurrent_requests"])
                    for run in report["service_runs"]
                ),
                "rejected_http_requests": sum(
                    len(run["rejections"]) for run in report["service_runs"]
                ),
                "original_reference_cases": len(audits),
            },
        )
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        # Record preservation failures without replacing the original error or
        # losing the partial report when an incomplete input was rejected.
        try:
            report["source_unchanged"] = reference.verify_checkpoint(
                args.source
            ) == report.get("source_hashes")
            report["sdk_unchanged"] = reference.verify_sdk(
                args.sdk_source
            ) == report.get("sdk_hashes")
        except Exception as error:
            report.update(
                source_unchanged=False,
                preservation_error=f"{type(error).__name__}: {error}",
            )
        if report.get("status") == "passed" and not all(
            report.get(key) for key in ("source_unchanged", "sdk_unchanged")
        ):
            report.update(
                status="failed", error="Original checkpoint/SDK preservation failed"
            )
        common.save(args.output, report)
    if report["status"] != "passed":
        raise RuntimeError(report.get("error", "Acceptance failed"))


if __name__ == "__main__":
    main()
