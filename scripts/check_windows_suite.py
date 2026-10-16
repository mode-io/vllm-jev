"""Run the lightweight Windows tests serially in their four pinned environments.

This runner installs nothing and downloads nothing. It uses synthetic models,
fake HTTP/runtime adapters and pinned SDK source checks, not released weights.
Use the separate model acceptance scripts for real-weight validation.
"""

import argparse
import hashlib
import json
import os
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEGACY_NODES = (
    "tests/test_windows_open_jev_integration.py::"
    "test_cli_routes_2b_model_id_and_unicode_source[False]",
    "tests/test_windows_open_jev_integration.py::"
    "test_cli_routes_2b_model_id_and_unicode_source[True]",
    "tests/test_windows_openjev.py::"
    "test_cli_routes_model_id_and_unicode_local_bundle[False]",
    "tests/test_windows_openjev.py::"
    "test_cli_routes_model_id_and_unicode_local_bundle[True]",
    "tests/test_windows_laya.py::"
    "test_windows_cli_routes_original_checkpoint_to_cpu_server",
    "tests/test_windows_laya.py::"
    "test_windows_cli_default_threads_fit_small_or_unknown_cpu_count[1-1]",
    "tests/test_windows_laya.py::"
    "test_windows_cli_default_threads_fit_small_or_unknown_cpu_count[2-2]",
    "tests/test_windows_laya.py::"
    "test_windows_cli_default_threads_fit_small_or_unknown_cpu_count[None-1]",
    "tests/test_windows_decider_integration.py::"
    "test_unified_cli_routes_decider_without_vllm[Mapika/decider-0.8b-False]",
    "tests/test_windows_decider_integration.py::"
    "test_unified_cli_routes_decider_without_vllm[Mapika/decider-0.8b-True]",
    "tests/test_windows_decider_integration.py::"
    "test_unified_cli_routes_decider_without_vllm[Mapika/decider-2b-False]",
    "tests/test_windows_decider_integration.py::"
    "test_unified_cli_routes_decider_without_vllm[Mapika/decider-2b-True]",
)
LEGACY_FILES = (
    "tests/test_windows_qwen.py",
    "tests/test_windows_tiny_driver.py",
    "tests/test_windows_tiny_reference.py",
    "tests/test_windows_tiny_family.py",
)
PROFILE_FILES = {
    "bruv": ("tests/test_windows_bruv.py",),
    "rsi": ("tests/test_windows_rsi.py",),
}
SDK_FILES = (
    "jevk5/__init__.py",
    "jevk5/prompt.py",
    "jevk5/lite.py",
    "bench/jevk5_lite.py",
)
OUTCOMES = ("passed", "skipped", "failure", "error")


def save(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def selections():
    files = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "tests").glob("test_windows_*.py")
    }
    reserved = set(LEGACY_FILES).union(*PROFILE_FILES.values())
    required = reserved | {node.split("::", 1)[0] for node in LEGACY_NODES}
    if missing := required - files:
        raise ValueError("Missing test files: " + ", ".join(sorted(missing)))
    return {
        "legacy-tiny": [*LEGACY_NODES, *LEGACY_FILES],
        "modern": [
            *sorted(files - reserved),
            *(f"--deselect={node}" for node in LEGACY_NODES),
        ],
        **{name: list(values) for name, values in PROFILE_FILES.items()},
    }


def make_jobs(args):
    interpreters = {
        "legacy-tiny": args.legacy_python,
        "modern": args.modern_python,
        "bruv": args.bruv_python,
        "rsi": args.rsi_python,
    }
    jobs = []
    for name, selected in selections().items():
        xml = args.output / f"{name}.xml"
        jobs.append(
            {
                "name": name,
                "command": [
                    str(interpreters[name]),
                    "-X",
                    "utf8",
                    "-B",
                    "-m",
                    "pytest",
                    *selected,
                    "-q",
                    "--disable-warnings",
                    "-p",
                    "no:cacheprovider",
                    f"--basetemp={args.output / (name + '-temp')}",
                    f"--junitxml={xml}",
                ],
                "xml": str(xml),
                "log": str(args.output / f"{name}.log"),
                "status": "planned",
            }
        )
    return jobs


def summarize(jobs):
    cases, profiles = {}, []
    for job in jobs:
        counts = dict.fromkeys(OUTCOMES, 0)
        row = {
            "name": job["name"],
            "returncode": job.get("returncode"),
            "report": job["xml"],
            "counts": counts,
            "errors": [],
        }
        if job.get("error"):
            row["errors"].append(job["error"])
        if job.get("returncode") != 0:
            row["errors"].append("pytest did not exit successfully")
        try:
            raw = Path(job["xml"]).read_bytes()
            row["sha256"] = hashlib.sha256(raw).hexdigest()
            tree = ET.fromstring(raw)
            for node in tree.iter("testcase"):
                identity = node.get("classname", "") + "::" + node.get("name", "")
                outcome = next(
                    (
                        name
                        for name in ("error", "failure", "skipped")
                        if node.find(name) is not None
                    ),
                    "passed",
                )
                counts[outcome] += 1
                cases.setdefault(identity, []).append(
                    {"profile": job["name"], "outcome": outcome}
                )
            if not sum(counts.values()):
                row["errors"].append("JUnit report contains no test cases")
        except (OSError, ET.ParseError) as error:
            row["errors"].append(f"Cannot read JUnit report: {error}")
        row["status"] = (
            "failed"
            if row["errors"] or any(counts[name] for name in OUTCOMES[1:])
            else "passed"
        )
        profiles.append(row)
    unique = dict.fromkeys(OUTCOMES, 0)
    for events in cases.values():
        outcomes = {event["outcome"] for event in events}
        outcome = next(
            name
            for name in ("error", "failure", "skipped", "passed")
            if name in outcomes
        )
        unique[outcome] += 1
    duplicates = sorted(key for key, events in cases.items() if len(events) != 1)
    return {
        "status": (
            "passed"
            if len(profiles) == 4
            and all(row["status"] == "passed" for row in profiles)
            and not duplicates
            else "failed"
        ),
        "created_unix": time.time(),
        "profiles": profiles,
        "unique_test_counts": unique,
        "duplicated_test_ids": duplicates,
        "scope": (
            "Lightweight Windows tests only: synthetic models, fake HTTP/runtime "
            "adapters and pinned SDK source checks. No released model weights."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("legacy", "modern", "bruv", "rsi"):
        parser.add_argument(
            f"--{name}-python",
            type=Path,
            required=True,
            help=f"Python executable in the isolated windows-{name} environment.",
        )
    parser.add_argument(
        "--jevk5-sdk-source",
        type=Path,
        required=True,
        help="Pinned JevK5 author SDK checkout; tests verify its source hashes.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New or empty result directory; existing reports are never overwritten.",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Write plan.json without starting Python test processes.",
    )
    args = parser.parse_args()
    try:
        for name in ("legacy", "modern", "bruv", "rsi"):
            path = getattr(args, f"{name}_python").resolve(strict=True)
            if not path.is_file():
                raise ValueError(f"{name} Python executable is not a file: {path}")
            setattr(args, f"{name}_python", path)
        args.jevk5_sdk_source = args.jevk5_sdk_source.resolve(strict=True)
        for name in SDK_FILES:
            if not (args.jevk5_sdk_source / name).is_file():
                raise ValueError(f"Missing pinned SDK source: {name}")
        args.output = args.output.resolve()
        if args.output.exists() and (
            not args.output.is_dir() or any(args.output.iterdir())
        ):
            raise ValueError(f"Output must be new or empty: {args.output}")
        jobs = make_jobs(args)
        args.output.mkdir(parents=True, exist_ok=True)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    manifest = {
        "status": "planned",
        "created_unix": time.time(),
        "repository": str(ROOT),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "jevk5_sdk_source": str(args.jevk5_sdk_source),
        "jobs": jobs,
    }
    save(args.output / "plan.json", manifest)
    if args.plan_only:
        print(f"Planned four profiles: {args.output / 'plan.json'}")
        return 0
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("JEV_")
        and key
        not in ("VLLM_JEV_DECISION_TEMPLATE", "PYTEST_ADDOPTS", "PYTEST_PLUGINS")
    }
    environment.update(
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONUTF8="1",
        PYTHONIOENCODING="utf-8",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        JEVK5_LITE_SDK_SOURCE=str(args.jevk5_sdk_source),
    )
    manifest["status"] = "running"
    interrupted = False
    try:
        for job in jobs:
            job.update(status="running", started_unix=time.time())
            save(args.output / "execution.json", manifest)
            print(f"Running {job['name']}...", flush=True)
            try:
                with Path(job["log"]).open("x", encoding="utf-8") as log:
                    result = subprocess.run(
                        job["command"],
                        cwd=ROOT,
                        env=environment,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=1800,
                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                    )
                job["returncode"] = result.returncode
                checked = summarize([job])
                job["status"] = (
                    "passed"
                    if checked["profiles"][0]["status"] == "passed"
                    and not checked["duplicated_test_ids"]
                    else "failed"
                )
            except (OSError, subprocess.TimeoutExpired) as error:
                job.update(status="failed", error=f"{type(error).__name__}: {error}")
            finally:
                job["finished_unix"] = time.time()
                save(args.output / "execution.json", manifest)
            print(f"{job['name']}: {job['status']}", flush=True)
    except KeyboardInterrupt:
        interrupted = True
        manifest["error"] = "Interrupted by user"
    finally:
        summary = summarize(jobs)
        if interrupted:
            summary["status"] = "failed"
        manifest["status"] = summary["status"]
        save(args.output / "execution.json", manifest)
        save(args.output / "summary.json", summary)
    counts = summary["unique_test_counts"]
    print(
        f"{summary['status']}: {counts['passed']} unique tests passed; "
        f"report: {args.output / 'summary.json'}"
    )
    return 130 if interrupted else (0 if summary["status"] == "passed" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
