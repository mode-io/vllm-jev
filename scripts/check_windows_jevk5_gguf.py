"""Real Q5 CPU acceptance through installed CLI and independent author SDK."""

import argparse
import concurrent.futures
import copy
import json
import math
import os
import socket
import sys
import tempfile
import time
from pathlib import Path

import check_windows_candidates as helpers
import windows_jevk5_gguf_reference as reference

common = helpers.common
MODEL = reference.MODEL_ID
BACKEND = "jevk5-llama-cpp-q5-b11010"
RUNNER = Path(__file__).with_name("windows_jevk5_gguf_reference.py")
TOLERANCE = 1e-5
CASES = copy.deepcopy(common.CASES)
CASES.append(
    {
        "id": "boundary_16",
        "state": ["Delivery is late.", {"language": "中文"}],
        "questions": {
            "category": {
                "type": "choice",
                "instructions": "Pick the matching topic.",
                "criteria": ["delivery"] + [f"alternative{i}" for i in range(15)],
            },
            "truth": {
                "type": "noul",
                "instructions": "Is delivery late?",
                "criteria": {"true": "Late delivery", "false": "On time"},
            },
            "scale": {
                "type": "score",
                "instructions": "Rate delay severity.",
                "criteria": ["low", "low", "high"],
            },
        },
    }
)


def validate_reference(path, args, cases_path):
    report = json.loads(path.read_text(encoding="utf-8"))
    model = getattr(args, "model", MODEL)
    specification = reference.MODELS[model]
    checks = {
        "status": "passed",
        "model": model,
        "revision": reference.REVISION,
        "sdk_commit": reference.SDK_COMMIT,
        "source_hashes": reference.source_hashes(model),
        "sdk_hashes": reference.SDK_HASHES,
        "runner_sha256": common.digest(RUNNER),
        "cases_sha256": common.digest(cases_path),
        "threads": args.threads,
        "temperature": specification["temperature"],
        "knockout_temperature": specification["knockout_temperature"],
        "source_unchanged": True,
        "sdk_unchanged": True,
        "server_stopped": True,
        "product_imported": False,
        "torch_imported": False,
    }
    for key, value in checks.items():
        if report.get(key) != value:
            raise ValueError(f"author baseline identity differs: {key}")
    if Path(report.get("python", "")).resolve() != args.reference_python:
        raise ValueError("author interpreter differs")
    if len(report.get("cases", [])) != len(CASES):
        raise ValueError("incomplete author cases")
    expected = []
    for case, observed in zip(CASES, report["cases"], strict=True):
        if json.dumps(observed["case"], ensure_ascii=False) != json.dumps(
            case, ensure_ascii=False
        ):
            raise ValueError("author inputs or their order differ")
        for name, question in case["questions"].items():
            event = observed["observations"][name]
            output, tokens = event["output"], event["tokens"]
            if (
                output.get("truncated")
                or output.get("tokens_evaluated") != len(tokens)
                or not 1 <= len(tokens) <= 1024
            ):
                raise ValueError("author prompt not completely evaluated")
            seen = {
                item["token"]: item["logprob"]
                for item in output["completion_probabilities"][0]["top_logprobs"]
            }
            count = 2 if question["type"] == "noul" else len(question["criteria"])
            z = [seen.get(chr(65 + i), min(seen.values()) - 2) for i in range(count)]
            w = [math.exp((x - max(z)) / specification["temperature"]) for x in z]
            if (
                max(
                    abs(a - b / sum(w))
                    for a, b in zip(observed["probabilities"][name], w, strict=True)
                )
                > 1e-12
            ):
                raise ValueError(
                    "author output does not match observed logits/calibration"
                )
        expected.append(helpers.canonical(case, observed["probabilities"], 0))
    return report, expected


def check_response(status, response, expected, threads, model=MODEL):
    if status != 200:
        raise ValueError(f"System One failed: {status}: {response}")
    specification = reference.MODELS[model]
    if response.get("model") != model:
        raise ValueError("product response model differs")
    meta = response["metadata"]
    for key, value in {
        "backend": BACKEND,
        "model": model,
        "device": "cpu",
        "experimental": True,
        "revision": reference.REVISION,
        "quantization": "Q5_K_M",
        "threads": threads,
        "n_gpu_layers": 0,
        "temperature": specification["temperature"],
        "top_k": 40,
        "cache_reuse": False,
        "torch_imported": False,
        "artifact_repository": "alibiserikbay/JevK5-GGUF",
        "artifact_filename": specification["filename"],
    }.items():
        if meta.get(key) != value:
            raise ValueError(f"product identity differs: {key}: {meta.get(key)!r}")
    return common.compare_answers(expected, response["answers"], TOLERANCE)


def process_memory_sample(process, stage, started):
    """Keep native child identity beside its OS lifetime memory counters."""
    import psutil

    sample = common.memory_sample(process, stage, started)
    sample["process_identity_errors"] = []
    for key, getter in (
        ("process_name", process.name),
        ("executable_basename", lambda: Path(process.exe()).name),
        ("create_time", process.create_time),
    ):
        try:
            sample[key] = getter()
        except (psutil.Error, OSError, RuntimeError) as error:
            sample[key] = None
            sample["process_identity_errors"].append(
                f"{key}: {type(error).__name__}: {error}"
            )
    return sample


def service_run(args, environment, report, expected, number):
    model = args.model
    source_hashes = reference.source_hashes(model)
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
            workspace / "checkpoint" / model
            if number == 1
            else workspace / "中文 本地 模型" / "JevK5"
        )
        helpers.clone(args.source, source, source_hashes)
        command = [
            str(args.console_entrypoint),
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
        child_args = argparse.Namespace(**vars(args))
        child_args.workspace = workspace
        started = time.perf_counter()

        def action(process, tree):
            url = f"http://127.0.0.1:{args.port}"
            deadline = time.monotonic() + 300
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError("installed CLI exited before ready; inspect log")
                try:
                    status, health, _ = common.http(url, "/health", timeout=1)
                    if status == 200:
                        break
                except (OSError, ValueError):
                    pass
                time.sleep(0.2)
            else:
                raise TimeoutError("CLI startup timed out")
            owned = tree.listener(args.port)
            assert (
                health["model"] == model
                and health["revision"] == reference.REVISION
                and health["backend"] == BACKEND
            )
            record.update(health=health, ready_seconds=time.perf_counter() - started)
            status, models, _ = common.http(url, "/v1/models")
            assert status == 200 and models["data"][0]["id"] == model
            record["memory"].append(common.memory_sample(owned, "ready", started))
            for case, answer in zip(CASES, expected, strict=True):
                payload = {
                    "model": model,
                    "state": case["state"],
                    "questions": case["questions"],
                }
                status, response, seconds = common.http(
                    url, "/v1/systemone", payload, args.timeout
                )
                delta = check_response(status, response, answer, args.threads, model)
                record["cases"].append(
                    {
                        "id": case["id"],
                        "status": status,
                        "response": response,
                        "seconds": seconds,
                        "comparison": delta,
                    }
                )
                common.save(args.output, report)
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
                "model": model,
                "state": CASES[0]["state"],
                "questions": CASES[0]["questions"],
            }
            invalid = {
                "cache_salt": {**base, "cache_salt": "x"},
                "images": {**base, "images": ["data:image/png;base64,AA=="]},
                "videos": {**base, "videos": ["data:video/mp4;base64,AA=="]},
                "model": {**base, "model": "wrong/model"},
                "empty": {**base, "questions": {}},
                "overlength": {**base, "state": "word " * 3000},
                "choice_17": {
                    **base,
                    "questions": {
                        "q": {
                            "type": "choice",
                            "instructions": "Pick",
                            "criteria": [str(i) for i in range(17)],
                        }
                    },
                },
                "isolated_score": {
                    **base,
                    "questions": {
                        "q": {
                            "type": "score",
                            "instructions": "Rate",
                            "criteria": ["low", "high"],
                            "isolated": True,
                        }
                    },
                },
                "duplicate_choices": {
                    **base,
                    "questions": {
                        "q": {
                            "type": "choice",
                            "instructions": "Pick",
                            "criteria": ["x", "x"],
                        }
                    },
                },
            }
            for key, payload in invalid.items():
                status, response, seconds = common.http(
                    url, "/v1/systemone", payload, 60
                )
                record["rejections"][key] = {
                    "status": status,
                    "response": response,
                    "seconds": seconds,
                }
                if status != 422:
                    raise ValueError(f"rejection failed: {key}: {response}")
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
                assert status == 200 and unfinished and after
                for future in futures:
                    status, response, seconds = future.result()
                    check_response(status, response, expected[0], args.threads, model)
                    record["concurrent_requests"].append(
                        {"status": status, "response": response, "seconds": seconds}
                    )
            tree.refresh()
            record["memory"].append(common.memory_sample(owned, "completed", started))
            record["all_process_memory"] = [
                process_memory_sample(item, "completed", started)
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
            reference.verify_source(source, model) == source_hashes
        )
    record["status"] = "passed"
    common.save(args.output, report)


def apply_memory_policy(args):
    """Keep 4B defaults; never let a 9B acceptance bypass its larger gate."""
    minimum_available, minimum_commit = (
        (10.0, 12.0) if args.model == reference.MODEL_ID_9B else (6.0, 8.0)
    )
    if args.min_available_gib is None:
        args.min_available_gib = minimum_available
    if args.min_commit_available_gib is None:
        args.min_commit_available_gib = minimum_commit
    if args.model == reference.MODEL_ID_9B and (
        args.min_available_gib < minimum_available
        or args.min_commit_available_gib < minimum_commit
        or args.abort_below_available_gib < 1.5
    ):
        raise ValueError(
            "9B requires at least 10 GiB available / 12 GiB commit headroom "
            "and a 1.5 GiB available-memory abort threshold"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(reference.MODELS), default=MODEL)
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
    parser.add_argument("--port", type=int, default=18826)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--min-available-gib", type=float)
    parser.add_argument("--min-commit-available-gib", type=float)
    parser.add_argument("--abort-below-available-gib", type=float, default=1.5)
    args = parser.parse_args()
    try:
        apply_memory_policy(args)
    except ValueError as error:
        parser.error(str(error))
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
        parser.error("native Windows, 1-4 threads and valid port required")
    if (
        any(
            not math.isfinite(getattr(args, k)) or getattr(args, k) <= 0
            for k in (
                "timeout",
                "min_available_gib",
                "min_commit_available_gib",
                "abort_below_available_gib",
            )
        )
        or args.abort_below_available_gib >= args.min_available_gib
    ):
        parser.error("invalid memory/timeout policy")
    if args.reference_report == args.output or any(
        args.workspace.is_relative_to(root) or args.output.is_relative_to(root)
        for root in (args.source, args.sdk_source)
    ):
        parser.error("outputs must not overwrite original inputs")
    args.workspace.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "running",
        "model": args.model,
        "revision": reference.REVISION,
        "service_runs": [],
        "source": str(args.source),
        "quantization": "Q5_K_M",
        "memory_policy": {
            "min_available_gib": args.min_available_gib,
            "min_commit_available_gib": args.min_commit_available_gib,
            "abort_below_available_gib": args.abort_below_available_gib,
        },
        "tolerance": TOLERANCE,
        "driver_sha256": common.digest(Path(__file__)),
        "environment": common.common.environment(),
    }
    try:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", args.port))
        report["source_hashes"] = reference.verify_source(args.source, args.model)
        report["sdk_hashes"] = reference.verify(args.sdk_source, reference.SDK_HASHES)
        env = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith("JEV_") and k != "VLLM_JEV_DECISION_TEMPLATE"
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
        refpath = args.reference_report or args.output.with_name(
            args.output.stem + "-author.json"
        )
        if args.reference_report is None:
            with tempfile.TemporaryDirectory(
                prefix="author-", dir=args.workspace
            ) as directory:
                command = [
                    str(args.reference_python),
                    str(RUNNER),
                    "--model",
                    args.model,
                    "--source",
                    str(args.source),
                    "--sdk-source",
                    str(args.sdk_source),
                    "--workspace",
                    directory,
                    "--cases",
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
                        raise RuntimeError("author process failed; inspect author log")

                helpers.child_run(
                    args,
                    command,
                    env,
                    args.output.with_name(args.output.stem + "-author.log"),
                    record,
                    wait_reference,
                )
        reference_report, expected = validate_reference(refpath, args, cases_path)
        report.update(
            reference_report=str(refpath),
            reference_sha256=common.digest(refpath),
            reference_reused=args.reference_report is not None,
        )
        if not args.reference_only:
            _, identity = helpers.product_entrypoint(args.console_entrypoint)
            report["product_entrypoint"] = identity
            for number in (1, 2):
                service_run(args, env, report, expected, number)
        report.update(status="passed", measured=True)
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["source_unchanged"] = reference.verify_source(
            args.source, args.model
        ) == reference.source_hashes(args.model)
        report["sdk_unchanged"] = (
            reference.verify(args.sdk_source, reference.SDK_HASHES)
            == reference.SDK_HASHES
        )
        common.save(args.output, report)


if __name__ == "__main__":
    main()
