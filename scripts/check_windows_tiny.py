"""Serial independent author / installed Windows CLI acceptance for Tiny 1.7B.

Requires an already verified checkpoint and isolated reference environment.
No downloads, installations, or unrelated process termination are performed.
"""

import argparse
import concurrent.futures
import copy
import importlib.util
import json
import math
import os
import socket
import subprocess
import sys
import time
from pathlib import Path


def load(name, filename):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).with_name(filename)
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


helpers = load("_tiny_acceptance_helpers", "check_windows_candidates.py")
reference = load("_tiny_author_reference", "windows_tiny_reference.py")
common = helpers.common
MODEL = reference.MODEL_ID
TOLERANCE = 1e-5


def cases():
    result = copy.deepcopy(common.CASES)
    result.extend(
        [
            {
                "id": "single_choice",
                "state": "There is one valid routing option.",
                "questions": {
                    "route": {
                        "type": "choice",
                        "instructions": "Select the available route.",
                        "criteria": {"only": "The only available option"},
                    }
                },
            },
            {
                "id": "custom_noul_ten_levels",
                "state": {"request": "请马上处理重复扣款的问题", "priority": 8},
                "questions": {
                    "refund": {
                        "type": "noul",
                        "instructions": "Is a refund requested?",
                        "criteria": {
                            "false": "No refund requested",
                            "true": "Refund requested",
                        },
                    },
                    "urgency": {
                        "type": "score",
                        "instructions": "Rate urgency from low to high.",
                        "criteria": [f"Level {i}" for i in range(10)],
                    },
                },
            },
            {
                "id": "unicode_json_descriptions",
                "state": ["订单 A-12 尚未送到", {"delayed": True, "tracking": None}],
                "questions": {
                    "category": {
                        "type": "choice",
                        "instructions": {"任务": "选择负责部门", "language": "zh"},
                        "criteria": {
                            "物流": {"范围": ["配送", "包裹"], "priority": 2},
                            "账单": {"范围": ["退款"], "priority": 1},
                            "其他": "",
                        },
                    }
                },
            },
        ]
    )
    return result


def invalid_cases(base):
    def questions(value):
        return {**base, "questions": value}

    choice = {
        "type": "choice",
        "instructions": "Choose.",
        "criteria": {"a": "A", "b": "B"},
    }
    return {
        "images": {**base, "images": ["data:image/png;base64,AA=="]},
        "videos": {**base, "videos": ["data:video/mp4;base64,AA=="]},
        "media": {**base, "media": [{"type": "image", "url": "x"}]},
        "wrong_model": {**base, "model": "not-loaded/model"},
        "cache_salt": {**base, "cache_salt": "x"},
        "state_type": {**base, "state": 1},
        "empty_questions": questions({}),
        "empty_id": questions({"": choice}),
        "invalid_type": questions({"q": {**choice, "type": "other"}}),
        "empty_choice": questions({"q": {**choice, "criteria": {}}}),
        "empty_label": questions({"q": {**choice, "criteria": {"": "A", "b": "B"}}}),
        "choice_256": questions(
            {"q": {**choice, "criteria": {str(i): "Option" for i in range(256)}}}
        ),
        "invalid_description": questions(
            {"q": {**choice, "criteria": {"a": 4, "b": "B"}}}
        ),
        "score_1": questions({"q": {**choice, "type": "score", "criteria": ["low"]}}),
        "score_11": questions(
            {"q": {**choice, "type": "score", "criteria": [str(i) for i in range(11)]}}
        ),
        "score_repeated": questions(
            {"q": {**choice, "type": "score", "criteria": ["low", "low"]}}
        ),
        "noul_criteria": questions(
            {"q": {**choice, "type": "noul", "criteria": {"maybe": "Maybe"}}}
        ),
        "questions_65": questions({str(i): choice for i in range(65)}),
        "candidate_sum_258": questions(
            {
                str(i): {**choice, "criteria": {str(j): "Option" for j in range(129)}}
                for i in range(2)
            }
        ),
        "schema_token_limit": questions(
            {"q": {**choice, "instructions": "token " * 5000}}
        ),
        "late_invalid_schema": questions(
            {
                "valid": choice,
                "invalid": {**choice, "instructions": "token " * 5000},
            }
        ),
    }


def canonical(question, p):
    """Independent reconstruction of this release's typed answer presentation."""
    kind, criteria = question["type"], question.get("criteria")
    if kind == "noul":
        return {"type": kind, "noul": p[1]}
    keys = (
        list(criteria) if kind == "choice" else [reference.render(v) for v in criteria]
    )
    selected = max(range(len(p)), key=p.__getitem__)
    result = {
        "type": kind,
        "probabilities": dict(zip(keys, p)),
        "confidence": p[selected],
    }
    if kind == "choice":
        result["choice"] = keys[selected]
    else:
        result["score"] = sum(i * value for i, value in enumerate(p))
        result["legend"] = dict(zip(keys, criteria))
    return result


def validate_reference(path, cases_path, inputs, threads):
    value = helpers._strict_reference_json(path.read_text(encoding="utf-8"))
    expected = {
        "status": "passed",
        "model": MODEL,
        "revision": reference.REVISION,
        "runner_sha256": reference.digest(reference.__file__),
        "cases_sha256": reference.digest(cases_path),
        "source_hashes": reference.SOURCE_HASHES,
        "source_unchanged": True,
        "versions": reference.VERSIONS,
        "runtime": {
            "device": "cpu",
            "dtype": "float32",
            "attention": "sdpa",
            "use_cache": False,
            "threads": threads,
            "row_batch_size": 1,
            "temperature": 1.15,
            "max_sequence_tokens": 4096,
        },
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise ValueError(f"author evidence differs: {key}")
    if len(value["cases"]) != len(inputs):
        raise ValueError("author case count differs")
    for case, observed in zip(inputs, value["cases"]):
        if observed["id"] != case["id"] or len(observed["rows"]) != len(
            case["questions"]
        ):
            raise ValueError("author case/question identity differs")
        reconstructed = {}
        for (identifier, question), row in zip(
            case["questions"].items(), observed["rows"]
        ):
            if row["id"] != identifier or helpers._ordered_json(
                row["definition"]
            ) != helpers._ordered_json(reference.author_definition(question)):
                raise ValueError("author input definition differs")
            count = 2 if question["type"] == "noul" else len(question["criteria"])
            ids, positions, logits, p = (
                row["input_ids"],
                row["positions"],
                row["calibrated_logits"],
                row["probabilities"],
            )
            if (
                not isinstance(ids, list)
                or not 1 <= len(ids) <= 4096
                or any(type(token) is not int or token < 0 for token in ids)
                or len(positions) != count
                or len(logits) != count
                or len(p) != count
                or any(
                    type(pos) is not int or not 0 <= pos < len(ids) for pos in positions
                )
                or positions != sorted(set(positions))
                or any(
                    type(x) not in (int, float) or not math.isfinite(x)
                    for x in logits + p
                )
                or any(not 0 <= x <= 1 for x in p)
            ):
                raise ValueError("invalid author token/logit/probability evidence")
            exp = [math.exp(x - max(logits)) for x in logits]
            softmax = [x / sum(exp) for x in exp]
            if max(abs(a - b) for a, b in zip(softmax, p)) > 1e-6:
                raise ValueError("author softmax evidence differs")
            reconstructed[identifier] = canonical(question, p)
        common.compare_answers(reconstructed, observed["answers"], 1e-6)
    return value


def check_prompt_parity(source, inputs, author):
    """Only this product-side probe imports product encoding, never the reference."""
    from transformers import Qwen2TokenizerFast

    from vllm_jev import windows_tiny
    from vllm_jev.prompt import tiny_token_ids

    config = json.loads((source / "tokenizer_config.json").read_text(encoding="utf-8"))
    for key in ("tokenizer_class", "backend", "is_local", "local_files_only"):
        config.pop(key, None)
    config["additional_special_tokens"] = config.pop("extra_special_tokens")
    tokenizer = Qwen2TokenizerFast(
        tokenizer_file=str(source / "tokenizer.json"), **config
    )
    checked = singleton = 0
    for case, observed in zip(inputs, author["cases"]):
        prepared, _ = windows_tiny._questions(case["questions"])
        for item, row in zip(prepared, observed["rows"]):
            identifier, _, question, _, _, options = item
            if identifier != row["id"]:
                raise ValueError("question order differs")
            if len(options) == 1:
                # Product intentionally returns certainty without executing a prompt.
                singleton += 1
                continue
            ids, positions = tiny_token_ids(tokenizer, case["state"], question, options)
            if ids != row["input_ids"] or positions != row["positions"]:
                raise ValueError(
                    "product prompt tokens/positions differ from original author"
                )
            checked += 1
    return {"rows": checked, "singleton_shortcuts": singleton}


def check_response(status, response, expected, author_case, case):
    if status != 200:
        raise AssertionError(f"valid request failed: {response}")
    identity = {
        "model": MODEL,
        "revision": reference.REVISION,
        "backend": "tiny-jev-pytorch",
        "device": "cpu",
        "experimental": True,
        "method": "marker_token_head",
        "temperature": 1.15,
        "cached_tokens": 0,
    }
    if response.get("model") != MODEL:
        raise AssertionError("response model identity differs")
    for key, value in identity.items():
        if response.get("metadata", {}).get(key) != value:
            raise AssertionError(f"response metadata differs: {key}")
    count = sum(len(row["probabilities"]) for row in author_case["rows"])
    token_count = sum(
        len(row["input_ids"])
        for row in author_case["rows"]
        if len(row["probabilities"]) > 1
    )
    if response["metadata"].get("candidate_sequences") != count or response.get(
        "usage"
    ) != {
        "input_tokens": token_count,
        "output_tokens": 0,
    }:
        raise AssertionError("candidate/token accounting differs")
    return common.compare_answers(expected, response.get("answers"), TOLERANCE)


def service_run(args, environment, report, inputs, author, number, prefix, entrypoint):
    record = {
        "route": "model_id" if number == 1 else "unicode_local_path",
        "cases": [],
        "memory": [],
    }
    report["service_runs"].append(record)
    with helpers.private_workspace(args.workspace, record) as directory:
        source = (
            directory / "checkpoint" / MODEL
            if number == 1
            else directory / "本地 模型" / "Tiny 1.7B"
        )
        helpers.clone(args.source, source, reference.SOURCE_HASHES)
        command = [
            *prefix,
            "serve",
            MODEL if number == 1 else str(source),
            "--workspace",
            str(directory),
            "--cpu-threads",
            str(args.threads),
            "--host",
            "127.0.0.1",
            "--port",
            str(args.port),
        ]
        record["command"] = command
        started = time.perf_counter()

        def action(process, tree):
            url = f"http://127.0.0.1:{args.port}"
            deadline = time.monotonic() + args.timeout
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(
                        "Tiny CLI exited before ready; inspect server log"
                    )
                try:
                    status, health, _ = common.http(url, "/health", timeout=1)
                    if status == 200:
                        break
                except (OSError, ValueError):
                    pass
                time.sleep(0.2)
            else:
                raise TimeoutError("Tiny CLI startup timeout")
            owned = tree.listener(args.port)
            for key, value in {
                "model": MODEL,
                "revision": reference.REVISION,
                "backend": "tiny-jev-pytorch",
                "device": "cpu",
                "experimental": True,
            }.items():
                if health.get(key) != value:
                    raise AssertionError(f"health identity differs: {key}")
            record.update(health=health, ready_seconds=time.perf_counter() - started)
            status, models, _ = common.http(url, "/v1/models")
            if status != 200 or models["data"][0]["id"] != MODEL:
                raise AssertionError("model listing differs")
            record["memory"].append(common.memory_sample(owned, "ready", started))

            def observe(case, original, rows):
                status, response, elapsed = common.http(
                    url,
                    "/v1/systemone",
                    {
                        "model": MODEL,
                        "state": case["state"],
                        "questions": case["questions"],
                    },
                    args.timeout,
                )
                row = {
                    "id": case["id"],
                    "status": status,
                    "response": response,
                    "seconds": elapsed,
                }
                rows.append(row)
                common.save(args.output, report)
                row["max_absolute_difference"] = check_response(
                    status, response, original["answers"], original, case
                )
                return row

            for index, (case, original) in enumerate(zip(inputs, author["cases"])):
                row = observe(case, original, record["cases"])
                if number == 2:
                    common.compare_answers(
                        report["service_runs"][0]["cases"][index]["response"][
                            "answers"
                        ],
                        row["response"]["answers"],
                        0,
                    )
                record["memory"].append(
                    common.memory_sample(owned, case["id"], started)
                )
                print(
                    json.dumps(
                        {
                            "event": "case_completed",
                            "round": number,
                            "case": case["id"],
                            "seconds": row["seconds"],
                        }
                    ),
                    flush=True,
                )
            base = {
                "model": MODEL,
                "state": inputs[0]["state"],
                "questions": inputs[0]["questions"],
            }
            record["rejections"] = {}
            for name, payload in invalid_cases(base).items():
                status, response, _ = common.http(url, "/v1/systemone", payload, 60)
                record["rejections"][name] = {"status": status, "response": response}
                common.save(args.output, report)
                if status != 422:
                    raise AssertionError(f"unsupported input was not rejected: {name}")
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                futures = [
                    pool.submit(common.http, url, "/v1/systemone", base, args.timeout)
                    for _ in range(2)
                ]
                time.sleep(0.1)
                unfinished = sum(not future.done() for future in futures)
                status, _, elapsed = common.http(url, "/health", timeout=5)
                after = sum(not future.done() for future in futures)
                record["concurrent_health"] = {
                    "status": status,
                    "seconds": elapsed,
                    "unfinished_before": unfinished,
                    "unfinished_after": after,
                }
                if status != 200 or not unfinished or not after:
                    raise AssertionError(
                        "health was not observed during pending inference"
                    )
                record["concurrent_requests"] = []
                for future in futures:
                    status, response, elapsed = future.result()
                    difference = check_response(
                        status,
                        response,
                        author["cases"][0]["answers"],
                        author["cases"][0],
                        inputs[0],
                    )
                    common.compare_answers(
                        record["cases"][0]["response"]["answers"],
                        response["answers"],
                        0,
                    )
                    record["concurrent_requests"].append(
                        {
                            "status": status,
                            "response": response,
                            "seconds": elapsed,
                            "max_absolute_difference": difference,
                        }
                    )
            record["memory"].append(common.memory_sample(owned, "completed", started))

        isolated = argparse.Namespace(**vars(args))
        isolated.workspace = directory
        helpers.child_run(
            isolated,
            command,
            helpers.product_environment(environment, entrypoint),
            args.output.with_name(args.output.stem + f"-server-{number}.log"),
            record,
            action,
        )
        reference.verify(source)
        record["source_unchanged"] = True
    record["status"] = "passed"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "source",
        "reference-python",
        "workspace",
        "output",
        "console-entrypoint",
    ):
        parser.add_argument("--" + name, required=True, type=Path)
    reference_mode = parser.add_mutually_exclusive_group()
    reference_mode.add_argument("--reference-report", type=Path)
    reference_mode.add_argument("--reference-only", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--port", type=int, default=18836)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--min-available-gib", type=float, default=10)
    # The validated FP32 loader peaks near 16.7 GiB of Windows commit, even
    # though its physical working set is below 7 GiB. Keep separate headroom.
    parser.add_argument("--min-commit-available-gib", type=float, default=20)
    parser.add_argument("--abort-below-available-gib", type=float, default=1.5)
    args = parser.parse_args()
    if sys.platform != "win32" or not 1 <= args.threads <= min(4, os.cpu_count() or 1):
        parser.error("native Windows and 1-4 CPU threads required")
    if not 1 <= args.port <= 65535:
        parser.error("invalid port")
    for name in (
        "timeout",
        "min_available_gib",
        "min_commit_available_gib",
        "abort_below_available_gib",
    ):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error("memory/timeout limits must be finite and positive")
    if args.abort_below_available_gib >= args.min_available_gib:
        parser.error("abort threshold must be below the startup memory threshold")
    for name in (
        "source",
        "reference_python",
        "workspace",
        "output",
        "console_entrypoint",
        "reference_report",
    ):
        if getattr(args, name) is not None:
            setattr(args, name, getattr(args, name).resolve())
    if args.workspace.is_relative_to(args.source) or args.output.is_relative_to(
        args.source
    ):
        parser.error("workspace/output must not overlap source")
    if args.output in (
        args.reference_report,
        args.reference_python,
        args.console_entrypoint,
    ):
        parser.error("output must not overwrite reference or executable")
    return args


def main():
    args = parse_args()
    args.workspace.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    inputs = cases()
    cases_path = args.output.with_name(args.output.stem + "-cases.json")
    author_path = args.reference_report or args.output.with_name(
        args.output.stem + "-author.json"
    )
    report = {
        "status": "running",
        "measured": False,
        "model": MODEL,
        "revision": reference.REVISION,
        "driver_sha256": common.digest(Path(__file__)),
        "service_runs": [],
        "tolerance": TOLERANCE,
        "memory_policy": {
            "min_available_gib": args.min_available_gib,
            "min_commit_available_gib": args.min_commit_available_gib,
            "abort_below_available_gib": args.abort_below_available_gib,
        },
        "scope": "Original author decide FP32/SDPA, full typed probabilities, exact tokens and marker positions, installed model-ID and Unicode-path routes; bounded synthetic parity, not model quality.",
        "cancellation_scope": "Shared service cancellation is covered by isolated regression tests; no real-weight disconnect timing claim.",
    }
    try:
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", args.port))
        reference.verify(args.source)
        common.save(cases_path, inputs)
        environment = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith("JEV_") and k != "VLLM_JEV_DECISION_TEMPLATE"
        }
        environment.update(
            PYTHONDONTWRITEBYTECODE="1",
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            HF_HOME=str(args.workspace / "hf"),
            OMP_NUM_THREADS=str(args.threads),
            MKL_NUM_THREADS=str(args.threads),
            PYTHONIOENCODING="utf-8",
            PYTHONPATH="",
        )
        if not args.reference_report:
            command = [
                str(args.reference_python),
                "-B",
                reference.__file__,
                "--source",
                str(args.source),
                "--cases-json",
                str(cases_path),
                "--output",
                str(author_path),
                "--threads",
                str(args.threads),
            ]
            record = report["reference_process"] = {"command": command}

            def run_reference(process, tree):
                if tree.wait(args.timeout) != 0:
                    raise RuntimeError("author reference failed; inspect author log")

            helpers.reference_run(
                args,
                command,
                environment,
                args.output.with_name(args.output.stem + "-author.log"),
                record,
                run_reference,
            )
        author = validate_reference(author_path, cases_path, inputs, args.threads)
        report.update(
            reference_report=str(author_path),
            reference_sha256=common.digest(author_path),
            reference_reused=bool(args.reference_report),
        )
        if not args.reference_only:
            prompt_path = args.output.with_name(args.output.stem + "-prompt-check.json")
            command = [
                sys.executable,
                "-B",
                str(Path(__file__).resolve()),
                "--prompt-check",
                str(args.source),
                str(cases_path),
                str(author_path),
                str(prompt_path),
            ]
            subprocess.run(
                command,
                env={**environment, "PYTHONPATH": str(helpers.ROOT)},
                cwd=args.workspace,
                check=True,
                timeout=120,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            prompt = helpers._strict_reference_json(
                prompt_path.read_text(encoding="utf-8")
            )
            if (
                prompt.get("status") != "passed"
                or prompt.get("cases_sha256") != common.digest(cases_path)
                or prompt.get("reference_sha256") != common.digest(author_path)
            ):
                raise ValueError("prompt probe evidence differs")
            report["prompt_check"] = prompt
            prefix, entrypoint = helpers.product_entrypoint(args.console_entrypoint)
            report["product_entrypoint"] = entrypoint
            for number in (1, 2):
                service_run(
                    args,
                    environment,
                    report,
                    inputs,
                    author,
                    number,
                    prefix,
                    entrypoint,
                )
        report.update(status="passed", measured=True)
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["source_unchanged"] = (
            reference.verify(args.source) == reference.SOURCE_HASHES
        )
        common.save(args.output, report)


if __name__ == "__main__":
    if len(sys.argv) == 6 and sys.argv[1] == "--prompt-check":
        source, cases_path, author_path, output = map(Path, sys.argv[2:])
        if output.resolve().is_relative_to(source.resolve()) or output.resolve() in (
            cases_path.resolve(),
            author_path.resolve(),
        ):
            raise ValueError("prompt evidence cannot overwrite its inputs")
        rows = check_prompt_parity(
            source,
            json.loads(cases_path.read_text(encoding="utf-8")),
            json.loads(author_path.read_text(encoding="utf-8")),
        )
        common.save(
            output,
            {
                "status": "passed",
                **rows,
                "cases_sha256": common.digest(cases_path),
                "reference_sha256": common.digest(author_path),
            },
        )
    else:
        main()
