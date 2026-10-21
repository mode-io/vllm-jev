"""Serial real-weight Bruv author reference and two installed Windows CLI routes."""

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
from types import SimpleNamespace


def load(name, filename):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).with_name(filename)
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


helpers = load("_bruv_acceptance_helpers", "check_windows_candidates.py")
reference = load("_bruv_author_reference", "windows_bruv_reference.py")
common = helpers.common
MODEL = reference.MODEL_ID
TOLERANCE = 1e-5


def cases():
    result = copy.deepcopy(common.CASES)
    result[2]["id"] = "sixteen_options"
    result[2]["questions"]["category"]["criteria"].update(
        {
            "security": "Security incident",
            "account": "Account settings",
            "docs": "Documentation",
            "legal": "Legal requests",
        }
    )
    result.append(
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
        }
    )
    result.append(
        {
            "id": "list_state_duplicate_descriptions",
            "state": ["The delivery is late", {"tracking": None, "delayed": True}],
            "questions": {
                "category": {
                    "type": "choice",
                    "instructions": "Select the best category.",
                    "criteria": {
                        "delivery": "Delivery issue",
                        "duplicate": "Delivery issue",
                        "other": "",
                    },
                },
                "severity": {
                    "type": "score",
                    "instructions": "Rate the problem.",
                    "criteria": ["Small", "Small", "Large"],
                },
            },
        }
    )
    return result


def invalid_cases(base):
    def questions(value):
        return {**base, "questions": value}

    choice = {"type": "choice", "instructions": "Choose.", "criteria": ["a", "b"]}
    return {
        "images": {**base, "images": ["data:image/png;base64,AA=="]},
        "videos": {**base, "videos": ["data:video/mp4;base64,AA=="]},
        "media": {**base, "media": [{"type": "image", "url": "x"}]},
        "model": {**base, "model": "not-loaded/model"},
        "cache_salt": {**base, "cache_salt": "x"},
        "state_type": {**base, "state": 1},
        "full_prompt_limit": {**base, "state": "token " * 2000},
        "empty": questions({}),
        "empty_id": questions({"": choice}),
        "invalid_type": questions({"q": {**choice, "type": "other"}}),
        "one_choice": questions({"q": {**choice, "criteria": ["a"]}}),
        "seventeen_choice": questions(
            {"q": {**choice, "criteria": [str(i) for i in range(17)]}}
        ),
        "duplicate_choice": questions({"q": {**choice, "criteria": ["a", "a"]}}),
        "invalid_description": questions(
            {"q": {**choice, "criteria": {"a": {}, "b": "B"}}}
        ),
        "isolated": questions({"q": {**choice, "isolated": False}}),
        "eleven_score": questions(
            {
                "q": {
                    "type": "score",
                    "instructions": "Rate.",
                    "criteria": [str(i) for i in range(11)],
                }
            }
        ),
        "noul_criteria": questions(
            {
                "q": {
                    "type": "noul",
                    "instructions": "Yes?",
                    "criteria": {"maybe": "Maybe"},
                }
            }
        ),
        "sixty_five_questions": questions({str(i): choice for i in range(65)}),
        "candidate_limit": questions(
            {
                str(i): {**choice, "criteria": [str(j) for j in range(16)]}
                for i in range(17)
            }
        ),
        "late_invalid": questions(
            {"valid": choice, "invalid": {**choice, "instructions": "token " * 2000}}
        ),
    }


def verify_reference(path, cases_path, inputs, args):
    import numpy as np

    result = helpers._strict_reference_json(path.read_text(encoding="utf-8"))
    expected = {
        "status": "passed",
        "model": MODEL,
        "revision": reference.REVISION,
        "sdk_revision": reference.SDK_REVISION,
        "runner_sha256": reference.digest(Path(reference.__file__)),
        "cases_sha256": reference.digest(cases_path),
        "source_hashes": reference.SOURCE_HASHES,
        "sdk_hashes": reference.SDK_HASHES,
        "versions": reference.VERSIONS,
        "source_unchanged": True,
        "sdk_unchanged": True,
        "private_workspace_removed": True,
    }
    for key, value in expected.items():
        if result.get(key) != value:
            raise ValueError(f"author evidence differs: {key}")
    if (
        result["runtime"]["dtype"] != "float32"
        or result["runtime"]["threads"] != args.threads
    ):
        raise ValueError("author runtime differs")
    if len(result["cases"]) != len(inputs):
        raise ValueError("author case count differs")
    for case, observed in zip(inputs, result["cases"]):
        if case["id"] != observed["id"] or [
            row["record"] for row in observed["rows"]
        ] != reference.mapped_records(case):
            raise ValueError("author records differ")
        if set(observed["probabilities"]) != set(case["questions"]):
            raise ValueError("author question IDs differ")
        for row in observed["rows"]:
            count = len(row["record"]["options"])
            logits = np.array(row["logits"], dtype=np.float32)
            if (
                logits.shape != (count,)
                or not np.isfinite(logits).all()
                or not 1 <= len(row["input_ids"]) <= 1024
            ):
                raise ValueError("author raw logits or token count invalid")
            exp = np.exp(logits - logits.max())
            expected_p = exp / exp.sum()
            if (
                not np.allclose(expected_p, row["probabilities"], rtol=0, atol=1e-6)
                or row["probabilities"] != observed["probabilities"][row["id"]]
            ):
                raise ValueError("independent author softmax check failed")
    return result


def check_prompt_parity(source, inputs, author):
    """Tokenizer-only product check after the independent author report exists."""
    from transformers import AutoTokenizer

    from vllm_jev import windows_bruv

    agent = windows_bruv.BruvAgent.__new__(windows_bruv.BruvAgent)
    agent.tokenizer = AutoTokenizer.from_pretrained(
        source, local_files_only=True, trust_remote_code=False
    )
    label_ids = [
        agent.tokenizer.encode(c, add_special_tokens=False)[0]
        for c in "ABCDEFGHIJKLMNOP"
    ]
    agent.label_ids = SimpleNamespace(tolist=lambda: label_ids)
    checked = 0
    for case, observed in zip(inputs, author["cases"]):
        for (identifier, question), row in zip(
            case["questions"].items(), observed["rows"]
        ):
            _, _, _, instruction, options = windows_bruv._question(question)
            if (
                identifier != row["id"]
                or agent._encode(case["state"], instruction, options)
                != row["input_ids"]
            ):
                raise ValueError("product prompt tokens differ from original author")
            checked += 1
    return checked


def prompt_check_process(args, cases_path, author_path, environment):
    """Transformers imports Torch; let a short child release it before serving."""
    output = args.output.with_name(args.output.stem + "-prompt-check.json")
    command = [
        str(Path(sys.executable).resolve()),
        "-B",
        str(Path(__file__).resolve()),
        "--prompt-check",
        str(args.source),
        str(cases_path),
        str(author_path),
        str(output),
    ]
    completed = subprocess.run(
        command,
        cwd=args.workspace,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=180,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    if completed.returncode:
        raise RuntimeError(
            "Bruv prompt parity child failed: " + completed.stderr[-2000:]
        )
    result = json.loads(output.read_text(encoding="utf-8"))
    if (
        result.get("status") != "passed"
        or result.get("cases_sha256") != common.digest(cases_path)
        or result.get("reference_sha256") != common.digest(author_path)
    ):
        raise ValueError("Bruv prompt parity evidence differs")
    if "torch" in sys.modules:
        raise RuntimeError("acceptance parent imported Torch")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "source",
        "sdk-source",
        "reference-python",
        "workspace",
        "output",
        "console-entrypoint",
    ):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--reference-report", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--port", type=int, default=18832)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--min-available-gib", type=float, default=6)
    parser.add_argument("--min-commit-available-gib", type=float, default=8)
    parser.add_argument("--abort-below-available-gib", type=float, default=1.5)
    args = parser.parse_args()
    if sys.platform != "win32" or not 1 <= args.threads <= min(4, os.cpu_count() or 1):
        parser.error("requires native Windows and 1-4 CPU threads")
    if not 1 <= args.port <= 65535:
        parser.error("invalid port")
    for field in (
        "timeout",
        "min_available_gib",
        "min_commit_available_gib",
        "abort_below_available_gib",
    ):
        if not math.isfinite(getattr(args, field)) or getattr(args, field) <= 0:
            parser.error("memory and timeout limits must be finite and positive")
    for field in (
        "source",
        "sdk_source",
        "reference_python",
        "workspace",
        "output",
        "console_entrypoint",
        "reference_report",
    ):
        if getattr(args, field) is not None:
            setattr(args, field, getattr(args, field).resolve())
    for root in (args.source, args.sdk_source):
        if args.workspace.is_relative_to(root) or args.output.is_relative_to(root):
            parser.error("workspace/output must be outside author sources")
    if args.reference_report == args.output:
        parser.error("output cannot replace reference report")
    args.workspace.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    inputs = cases()
    cases_path = args.output.with_name(args.output.stem + "-cases.json")
    author_path = args.reference_report or args.output.with_name(
        args.output.stem + "-author.json"
    )
    report = {
        "status": "running",
        "model": MODEL,
        "revision": reference.REVISION,
        "driver_sha256": common.digest(Path(__file__)),
        "tolerance": TOLERANCE,
        "service_runs": [],
        "scope": "Same released weights, explicit FP32 author score_batch; project Choice/Noul/Score mapping; fixed synthetic cases only, not BF16 parity or benchmark accuracy.",
    }
    try:
        with socket.socket() as s:
            s.bind(("127.0.0.1", args.port))
        reference.verify(args.source, reference.SOURCE_HASHES)
        reference.verify(args.sdk_source, reference.SDK_HASHES)
        report["source_hashes"] = reference.SOURCE_HASHES
        report["sdk_hashes"] = reference.SDK_HASHES
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
                "--sdk-source",
                str(args.sdk_source),
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
        author = verify_reference(author_path, cases_path, inputs, args)
        report["reference_report"] = str(author_path)
        report["reference_sha256"] = common.digest(author_path)
        report["reference_reused"] = bool(args.reference_report)
        report["prompt_check"] = prompt_check_process(
            args, cases_path, author_path, environment
        )
        report["exact_author_prompt_rows"] = report["prompt_check"]["rows"]
        expected = [
            helpers.canonical(case, observed["probabilities"], 1)
            for case, observed in zip(inputs, author["cases"])
        ]
        prefix, entrypoint = helpers.product_entrypoint(args.console_entrypoint)
        report["product_entrypoint"] = entrypoint
        identity = {
            "model": MODEL,
            "revision": reference.REVISION,
            "backend": "bruv-pytorch",
            "device": "cpu",
            "experimental": True,
            "backbone_dtype": "float32",
            "head_dtype": "float32",
            "logits_dtype": "float32",
            "temperature": 1.0,
            "attention_implementation": "eager",
            "use_cache": False,
            "cache_reuse": False,
            "row_batch_size": 1,
            "max_sequence_tokens": 1024,
            "threads": args.threads,
            "prompt_format": "kevala-direct-options-v1",
            "systemone_mapping": "bruv_joint_v1",
        }
        for number in (1, 2):
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
                    else directory / "本地 模型" / "Bruv"
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

                def service_action(process, tree):
                    url = f"http://127.0.0.1:{args.port}"
                    deadline = time.monotonic() + 600
                    while time.monotonic() < deadline:
                        if process.poll() is not None:
                            raise RuntimeError("Bruv CLI exited before ready")
                        try:
                            status, health, _ = common.http(url, "/health", timeout=1)
                            if status == 200:
                                break
                        except (OSError, ValueError):
                            pass
                        time.sleep(0.2)
                    else:
                        raise TimeoutError("Bruv server startup timeout")
                    owned = tree.listener(args.port)
                    assert all(
                        health[k] == identity[k]
                        for k in (
                            "model",
                            "revision",
                            "backend",
                            "device",
                            "experimental",
                        )
                    )
                    record.update(
                        health=health, ready_seconds=time.perf_counter() - started
                    )
                    status, models, _ = common.http(url, "/v1/models")
                    assert status == 200 and models["data"][0]["id"] == MODEL
                    record["memory"].append(
                        common.memory_sample(owned, "ready", started)
                    )
                    for i, case in enumerate(inputs):
                        payload = {
                            "model": MODEL,
                            "state": case["state"],
                            "questions": case["questions"],
                        }
                        status, response, elapsed = common.http(
                            url, "/v1/systemone", payload, 600
                        )
                        helpers.record_and_compare_response(
                            args.output,
                            report,
                            record["cases"],
                            status=status,
                            response=response,
                            seconds=elapsed,
                            expected=expected[i],
                            dtype="float32",
                            identity=identity,
                            case_id=case["id"],
                            repeated_answers=report["service_runs"][0]["cases"][i][
                                "response"
                            ]["answers"]
                            if number == 2
                            else None,
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
                                    "seconds": elapsed,
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
                        status, response, _ = common.http(
                            url, "/v1/systemone", payload, 30
                        )
                        record["rejections"][name] = {
                            "status": status,
                            "response": response,
                        }
                        assert status == 422, (name, response)
                    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                        futures = [
                            pool.submit(common.http, url, "/v1/systemone", base, 600)
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
                        for future in futures:
                            status, response, elapsed = future.result()
                            helpers.record_and_compare_response(
                                args.output,
                                report,
                                record["concurrent_requests"],
                                status=status,
                                response=response,
                                seconds=elapsed,
                                expected=expected[0],
                                dtype="float32",
                                identity=identity,
                                repeated_answers=record["cases"][0]["response"][
                                    "answers"
                                ],
                            )
                    record["memory"].append(
                        common.memory_sample(owned, "completed", started)
                    )

                isolated = argparse.Namespace(**vars(args))
                isolated.workspace = directory
                helpers.child_run(
                    isolated,
                    command,
                    helpers.product_environment(environment, entrypoint),
                    args.output.with_name(args.output.stem + f"-server-{number}.log"),
                    record,
                    service_action,
                )
                reference.verify(source, reference.SOURCE_HASHES)
                record["source_unchanged"] = True
            record["status"] = "passed"
        report.update(status="passed", measured=True)
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["source_unchanged"] = (
            reference.verify(args.source, reference.SOURCE_HASHES)
            == reference.SOURCE_HASHES
        )
        report["sdk_unchanged"] = (
            reference.verify(args.sdk_source, reference.SDK_HASHES)
            == reference.SDK_HASHES
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
                "rows": rows,
                "cases_sha256": common.digest(cases_path),
                "reference_sha256": common.digest(author_path),
            },
        )
    else:
        main()
