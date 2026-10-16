"""Validate Open-Jev-2B against pinned, unmodified author code on Windows CPU.

Run --reference-only first, using a separate reference Python environment.
The default subsequently compares two fresh product CLI processes, sequentially.
No mode downloads weights or installs packages. --source is the product source
root containing package/checkpoint and base; --hf-cache is a standard Hub cache.
BF16 and FP32 are separate numerical configurations, never interchangeable baselines.
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
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = "ZefanCai/Open-Jev-2B"
REVISION = "0c7aa498b1627be8da4acf34c863ff0ee0a92785"
BASE_MODEL = "Qwen/Qwen3.5-2B"
BASE_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
SDK_COMMIT = "bd4118882f733574a3250a4b65fe4d884130c08b"
SDK_HASHES = {
    "__init__.py": "039d6c25d5ebea9724711756aff885f44d2226abbeea4abb99dfbc7b0b35ce7e",
    "model.py": "333bed84cf2801ba33b053891e9ff9563d082f3da87ad6ad1681a2b1bcb4c6aa",
    "serving.py": "ff23f5132bc15b524a15b56f3dbdcc5117174b15fc4c36ba6495aa0f3080a45a",
    "api.py": "f22aa50c00d64aa91c2d415c1973f59bdaad11d1de51308ae5454326b8cbf350",
    "metrics.py": "cb460c78b877a24708a0a8f8ca6b51f9602491f6ea24cd14e9a828c03d92f78b",
}
# Author package/manifest.json, independently checked against the pinned HF tree.
CHECKPOINT_FILES = {
    "adapter/adapter_config.json": (
        1170,
        "d27624bdbbe984cf175e10c7fa3d9beaea9876231ce943a248ad40e12742fe74",
    ),
    "adapter/adapter_model.safetensors": (
        10042480,
        "2d23935b1a7380db444abac572c04646918ba794e59002d1588236182a3ca18f",
    ),
    "head.pt": (
        9997,
        "3532cd576c58d5ad5bf17c3e9f2df4be8c70e08c07fa6bb7fa673dcd7b401f2a",
    ),
    "model.json": (
        186,
        "d985eafa4d635113ee8cc912e0512a95aac0bed3426e9256931235b75cb3d6c5",
    ),
    "temperature.json": (
        161,
        "090fb330a616338210a924d1c370c934e934f30f53f92c8a52a5cc7852cadd99",
    ),
}
SOURCE_HASHES = {
    **{
        f"package/checkpoint/{name}": value[1]
        for name, value in CHECKPOINT_FILES.items()
    },
    "LICENSE": "bbedc3fda3305820b977265f01b8619d87570a6739de3a5582c3464840f1e57a",
    "LICENSE-CODE": "ef2f6f6db605a2f611bc2d037c59276f615cdea76ac6e7551a28f71db890c9d2",
    "release-manifest.json": "fd4c1b0785b3b57fa05b1aa06d7e4daf08445ffcd27e9bd2da6fca949823407e",
    "UPSTREAM.md": "8d77b0b07ec0e34af6dfa238d05d2f95109926df91b468bfb238276610d1c420",
    "base/chat_template.jinja": "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80",
    "base/config.json": "ed1c1723241f23f7f4e23430759cbd7dcfb4103cbdfe052bfe7626b57c2615b4",
    "base/LICENSE": "bbedc3fda3305820b977265f01b8619d87570a6739de3a5582c3464840f1e57a",
    "base/merges.txt": "a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d",
    "base/model.safetensors.index.json": "aca8afed9da75b0f050b408d270766fd77627f1af401e240f61c3b47d0db02f9",
    "base/tokenizer.json": "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
    "base/tokenizer_config.json": "49e2b6e395f959f077f1e992b338919c0d4a9732fc6e613995e06557f843500c",
    "base/vocab.json": "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003",
    "base/model.safetensors-00001-of-00001.safetensors": "aa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1",
}
# Git blob IDs bind non-LFS raw bytes; the large LFS files use SHA-256.
# Source: HF /api/models/Qwen/Qwen3.5-2B/tree/<BASE_REVISION>.
BASE_FILES = {
    ".gitattributes": (1570, "git", "52373fe24473b1aa44333d318f578ae6bf04b49b"),
    "LICENSE": (11544, "git", "f938136e3adacfd92be087f6e113b5d6d97f678f"),
    "README.md": (62814, "git", "2efa49d346c547ea81cd3d1310f414df25934f45"),
    "chat_template.jinja": (7755, "git", "0ef09f214eaa6d9bca297988afc1454b5827b2c7"),
    "config.json": (2908, "git", "d30a15be1bbfd610f58ad80f1fbeb6778f58a80f"),
    "merges.txt": (3353259, "git", "a494e019ca1502219fd0128658b979e5f05ae8e8"),
    "model.safetensors-00001-of-00001.safetensors": (
        4548221488,
        "sha256",
        "aa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1",
    ),
    "model.safetensors.index.json": (
        64460,
        "git",
        "69a41758e1138259916d08b99b56fada441fd502",
    ),
    "preprocessor_config.json": (
        390,
        "git",
        "2ea84a437d448ff71b08df68fdd949d5cc4ebb64",
    ),
    "tokenizer.json": (
        12807982,
        "sha256",
        "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
    ),
    "tokenizer_config.json": (16709, "git", "fae3ce993e07c092ad024dde45e592379fde91bb"),
    "video_preprocessor_config.json": (
        385,
        "git",
        "3ba673a5ad7d4d13f54155ecd38b2a94a6dac8fe",
    ),
    "vocab.json": (6722759, "git", "0aa0ce0658d60ac4a5d609f4eadb0e8e43514176"),
}
BASE_REQUIRED = {
    "LICENSE",
    "config.json",
    "chat_template.jinja",
    "tokenizer_config.json",
    "tokenizer.json",
    "merges.txt",
    "vocab.json",
    "model.safetensors.index.json",
    "model.safetensors-00001-of-00001.safetensors",
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
MAX_SEQUENCE_TOKENS = 1024


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_checkpoint(checkpoint):
    observed = {}
    for name, (size, expected) in CHECKPOINT_FILES.items():
        path = checkpoint / name
        if not path.is_file() or path.stat().st_size != size:
            raise ValueError(f"missing or incorrect Open-Jev checkpoint file: {name}")
        observed[name] = digest(path)
        if observed[name] != expected:
            raise ValueError(f"Open-Jev checkpoint checksum mismatch: {name}")
    return observed


def verify_base(base):
    """Pin every present base file and reject unverified loader overrides."""
    if not base.is_dir():
        raise ValueError(f"base snapshot is missing: {base}")
    observed = {}
    for path in base.iterdir():
        if path.name == ".cache" and path.is_dir():
            continue
        if not path.is_file() or path.name not in BASE_FILES:
            raise ValueError(f"unexpected base source entry: {path.name}")
        size, algorithm, expected = BASE_FILES[path.name]
        if path.stat().st_size != size:
            raise ValueError(f"base file size mismatch: {path.name}")
        sha256 = hashlib.sha256()
        git = hashlib.sha1(f"blob {size}\0".encode()) if algorithm == "git" else None
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                sha256.update(block)
                if git is not None:
                    git.update(block)
        actual = git.hexdigest() if git is not None else sha256.hexdigest()
        if actual != expected:
            raise ValueError(f"base checksum mismatch: {path.name}")
        observed[path.name] = sha256.hexdigest()
    if missing := BASE_REQUIRED - observed.keys():
        raise ValueError(f"base source files are missing: {sorted(missing)}")
    return observed


def verify_source(source):
    # Product inputs have a narrower layout than the author's standard HF cache.
    # Reject overrides and extra official files before starting the reference.
    for directory in ("base", "package/checkpoint", "package/checkpoint/adapter"):
        path = source / directory
        if not path.is_dir():
            raise ValueError(f"product source directory is missing: {directory}")
        allowed = {
            name.removeprefix(directory + "/").split("/")[0]
            for name in SOURCE_HASHES
            if name.startswith(directory + "/")
        }
        for entry in path.iterdir():
            if directory == "base" and entry.name == ".cache" and entry.is_dir():
                continue
            relative = entry.relative_to(source).as_posix()
            if entry.name not in allowed or (
                relative in SOURCE_HASHES and not entry.is_file()
            ):
                raise ValueError(f"unexpected product source entry: {relative}")
    # Fail on a missing clone artifact before spending time hashing the weights.
    for name in SOURCE_HASHES:
        if not (source / name).is_file():
            raise ValueError(f"source artifact missing: {name}")
    checkpoint = verify_checkpoint(source / "package/checkpoint")
    base = verify_base(source / "base")
    observed = {
        **{f"package/checkpoint/{name}": value for name, value in checkpoint.items()},
        **{f"base/{name}": value for name, value in base.items()},
    }
    for name, expected in SOURCE_HASHES.items():
        if name not in observed:
            observed[name] = digest(source / name)
        if observed[name] != expected:
            raise ValueError(f"source checksum mismatch: {name}")
    return observed


def environment():
    packages = {}
    for name in (
        "torch",
        "transformers",
        "peft",
        "accelerate",
        "safetensors",
        "psutil",
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


def memory_sample(stage, started, pid=None):
    import psutil

    process = pid if hasattr(pid, "memory_info") else psutil.Process(pid)
    if not process.is_running():
        raise RuntimeError(f"memory sampling process exited: {process.pid}")
    information = process.memory_info()
    if os.name == "nt" and not hasattr(information, "peak_wset"):
        raise RuntimeError("psutil did not expose Windows peak working set")
    return {
        "stage": stage,
        "pid": process.pid,
        "wall_seconds": time.perf_counter() - started,
        "rss_bytes": information.rss,
        "peak_wset_bytes": getattr(information, "peak_wset", None),
        "private_bytes": getattr(information, "private", None),
    }


class OwnedProcessTree:
    """Track this Popen's descendants, retaining identities after parent exit."""

    def __init__(self, process, record):
        import psutil

        self.psutil = psutil
        self.process = process
        self.record = record
        self.members = {}
        self.discovery_complete = True
        record.update(launcher_pid=process.pid, owned_processes=[], stop_errors=[])
        try:
            self.remember(psutil.Process(process.pid))
        except psutil.Error as error:
            self.discovery_complete = False
            record["stop_errors"].append(f"initial process discovery: {error}")
        self.refresh()

    def remember(self, process):
        identity = (process.pid, process.create_time())
        self.members.setdefault(identity, process)
        self.record["owned_processes"] = [
            {"pid": pid, "create_time": created} for pid, created in self.members
        ]

    def refresh(self):
        for process in list(self.members.values()):
            try:
                if process.is_running():
                    for child in process.children(recursive=True):
                        self.remember(child)
            except self.psutil.NoSuchProcess:
                continue
            except self.psutil.Error as error:
                self.discovery_complete = False
                self.record["stop_errors"].append(f"process discovery: {error}")

    def living(self):
        living = []
        for process in self.members.values():
            try:
                # is_running checks creation time, so a reused PID is never killed.
                if (
                    process.is_running()
                    and process.status() != self.psutil.STATUS_ZOMBIE
                ):
                    living.append(process)
            except self.psutil.NoSuchProcess:
                continue
            except self.psutil.Error as error:
                self.discovery_complete = False
                self.record["stop_errors"].append(f"process exit check: {error}")
        return living

    def listener(self, port):
        self.refresh()
        listeners = {
            item.pid
            for item in self.psutil.net_connections(kind="tcp")
            if item.status == self.psutil.CONN_LISTEN and item.laddr.port == port
        }
        live = {item.pid: item for item in self.living()}
        if len(listeners) != 1 or not listeners.issubset(live):
            raise RuntimeError(
                f"port {port} listener is not exactly one owned process: {listeners}"
            )
        pid = listeners.pop()
        self.record.update(listener_pid=pid, memory_process_pid=pid)
        return live[pid]

    def wait(self, timeout):
        deadline = time.monotonic() + timeout
        while True:
            self.refresh()
            result = self.process.poll()
            if result is not None:
                return result
            if time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired(self.process.args, timeout)
            time.sleep(min(0.2, max(0, deadline - time.monotonic())))

    def stop(self):
        """Never raise from cleanup or touch processes outside the recorded tree."""
        try:
            self.refresh()
            for operation, timeout in (("terminate", 20), ("kill", 10)):
                living = self.living()
                for process in reversed(living):
                    try:
                        getattr(process, operation)()
                    except self.psutil.NoSuchProcess:
                        continue
                    except Exception as error:
                        self.record["stop_errors"].append(
                            f"{operation} pid {process.pid}: {type(error).__name__}: {error}"
                        )
                try:
                    _, alive = self.psutil.wait_procs(living, timeout=timeout)
                    if alive:
                        self.record["stop_errors"].append(
                            f"{operation} wait timed out for {[item.pid for item in alive]}"
                        )
                except Exception as error:
                    self.record["stop_errors"].append(
                        f"{operation} wait: {type(error).__name__}: {error}"
                    )
                self.refresh()
                if not self.living():
                    break
            try:
                self.process.wait(timeout=1)
            except Exception as error:
                self.record["stop_errors"].append(
                    f"launcher wait: {type(error).__name__}: {error}"
                )
            remaining = self.living()
            self.record["remaining_owned_pids"] = [item.pid for item in remaining]
            stopped = (
                self.discovery_complete
                and not remaining
                and self.process.poll() is not None
            )
        except Exception as error:
            self.record["stop_errors"].append(
                f"process cleanup: {type(error).__name__}: {error}"
            )
            stopped = False
        self.record["owned_process_stopped"] = stopped
        self.record["shutdown_scope"] = (
            "Only the recorded process tree was stopped; graceful shutdown not validated."
        )
        return stopped


def finish_owned_run(temporary, tree, process, record, source=None, before=None):
    """Keep the workspace if any owned process may survive; preserve prior errors."""
    record["private_workspace"] = temporary.name
    stopped = tree.stop() if tree is not None else process is None
    record["owned_process_stopped"] = stopped
    if not stopped:
        record.update(
            status="failed",
            retained_workspace=temporary.name,
            private_workspace_removed=False,
            cleanup_error="Could not confirm every owned process exited; workspace retained.",
        )
        return False
    okay = True
    if source is not None and before is not None:
        try:
            record["source_unchanged"] = verify_source(source) == before
            if not record["source_unchanged"]:
                raise AssertionError("product service modified original source files")
        except Exception as error:
            okay = False
            record.update(status="failed", preservation_error=str(error))
    try:
        temporary.cleanup()
        record["private_workspace_removed"] = True
    except Exception as error:
        okay = False
        record.update(
            status="failed",
            retained_workspace=temporary.name,
            private_workspace_removed=False,
            cleanup_error=f"{type(error).__name__}: {error}",
        )
    return okay


def copy_verified_sdk(source, destination):
    package = source / "jev"
    target = destination / "jev"
    target.mkdir(parents=True)
    for name, expected in SDK_HASHES.items():
        if digest(package / name) != expected:
            raise ValueError(f"refusing to execute unverified author source: {name}")
        shutil.copyfile(package / name, target / name)
        if digest(target / name) != expected:
            raise ValueError(f"author source copy failed verification: {name}")
    return target


def selected_cases(args):
    return [
        (name, state) for name, state in CASES if not args.case or name in args.case
    ]


def reference(args):
    """Execute only unmodified author inference, never project inference modules."""
    started = time.perf_counter()
    report = {
        "status": "running",
        "environment": environment(),
        "model": MODEL,
        "revision": REVISION,
        "base_revision": BASE_REVISION,
        "sdk_commit": SDK_COMMIT,
        "sdk_source_sha256": SDK_HASHES,
        "source": str(args.source),
        "cases": [],
        "memory": [],
        "reference": "Unmodified jev.serving.load_predictor().predict(), Torch CPU",
        "inference": {
            "dtype": args.dtype,
            "head_dtype": "float32",
            "batch_size": 1,
            "prefix_cache": False,
            "backend": "torch",
            "sdk_patches": False,
            "max_sequence_tokens": MAX_SEQUENCE_TOKENS,
        },
        "threads": args.threads,
        "selected_case_ids": [name for name, _ in selected_cases(args)],
        "measured": False,
    }
    source_before = cache_before = None
    snapshot = args.hf_cache / "models--Qwen--Qwen3.5-2B/snapshots" / BASE_REVISION
    try:
        source_before = verify_source(args.source)
        cache_before = verify_base(snapshot)
        report.update(
            source_sha256=source_before,
            hf_snapshot=str(snapshot),
            hf_snapshot_sha256=cache_before,
        )
        report["memory"].append(memory_sample("before_load", started))
        private = args.reference_workspace
        if not private.is_dir() or any(private.iterdir()):
            raise ValueError("reference requires an empty parent-owned workspace")
        package = copy_verified_sdk(args.sdk_source, private)
        checkpoint = private / "checkpoint"
        for name in CHECKPOINT_FILES:
            target = checkpoint / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(args.source / "package/checkpoint" / name, target)
        verify_checkpoint(checkpoint)
        sys.path.insert(0, str(private))
        import jev
        import torch
        from jev.serving import load_predictor

        if Path(jev.__file__).resolve() != package / "__init__.py":
            raise RuntimeError("reference imported an unrelated installed jev")
        versions = report["environment"]["packages"]
        if (
            versions["transformers"] != "5.10.2"
            or versions["peft"] != "0.19.1"
            or versions["accelerate"] != "1.13.0"
            or versions["torch"].split("+")[0] != "2.8.0"
        ):
            raise RuntimeError(
                "reference requires Torch 2.8.0, Transformers 5.10.2, PEFT 0.19.1 and Accelerate 1.13.0"
            )
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        load_started = time.perf_counter()
        print(
            json.dumps({"event": "reference_loading", "dtype": args.dtype}),
            flush=True,
        )
        predictor = load_predictor(
            checkpoint=checkpoint,
            device="cpu",
            max_length=MAX_SEQUENCE_TOKENS,
            batch_size=1,
            prefix_cache=False,
            backend="torch",
        )
        report["cold_load_seconds"] = time.perf_counter() - load_started
        model = predictor.scorer.model
        backbone = model.backbone
        expected_dtype = getattr(torch, args.dtype)
        if (
            any(parameter.device.type != "cpu" for parameter in model.parameters())
            or backbone.get_input_embeddings().weight.dtype != expected_dtype
            or model.head.weight.dtype != torch.float32
        ):
            raise RuntimeError("reference loaded on an unexpected device or dtype")
        report["inference"].update(
            actual_parameter_devices=sorted(
                {p.device.type for p in model.parameters()}
            ),
            actual_embedding_dtype=str(backbone.get_input_embeddings().weight.dtype),
            actual_head_dtype=str(model.head.weight.dtype),
            actual_backbone_dtypes=sorted(
                {str(p.dtype) for p in backbone.parameters()}
            ),
            backbone_parameter_bytes=sum(
                p.numel() * p.element_size() for p in backbone.parameters()
            ),
            head_parameter_bytes=sum(
                p.numel() * p.element_size() for p in model.head.parameters()
            ),
            actual_text_backbone=type(backbone.get_base_model()).__name__,
            cuda_available=torch.cuda.is_available(),
        )
        report["memory"].append(memory_sample("after_load", started))
        save(args.output, report)
        for name, state in selected_cases(args):
            case_started = time.perf_counter()
            response = predictor.predict({"state": state, "questions": QUESTIONS})
            if response["metadata"]["prefix_cache"]["enabled"]:
                raise AssertionError("reference unexpectedly enabled prefix caching")
            report["cases"].append(
                {
                    "id": name,
                    "state": state,
                    "questions": QUESTIONS,
                    "response": response,
                    "answers": response["answers"],
                    "seconds": time.perf_counter() - case_started,
                }
            )
            report["memory"].append(memory_sample(f"after_{name}", started))
            print(
                json.dumps(
                    {
                        "event": "reference_case_completed",
                        "case": name,
                        "seconds": report["cases"][-1]["seconds"],
                    }
                ),
                flush=True,
            )
            save(args.output, report)
        report["private_checkpoint_unchanged"] = verify_checkpoint(checkpoint) == {
            name: value[1] for name, value in CHECKPOINT_FILES.items()
        }
        if any(
            digest(package / name) != expected for name, expected in SDK_HASHES.items()
        ):
            raise RuntimeError("private author source changed during inference")
        if any(
            name == "vllm_jev" or name.startswith("vllm_jev.") for name in sys.modules
        ):
            raise RuntimeError("reference process imported product inference modules")
        del predictor, model, backbone
        report.update(status="passed", measured=True)
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        try:
            if source_before is not None:
                report["source_unchanged"] = verify_source(args.source) == source_before
            if cache_before is not None:
                report["hf_snapshot_unchanged"] = verify_base(snapshot) == cache_before
            if (
                report.get("source_unchanged") is False
                or report.get("hf_snapshot_unchanged") is False
            ):
                report["status"] = "failed"
            report["memory"].append(memory_sample("final", started))
            report["peak_wset_bytes"] = max(
                (sample["peak_wset_bytes"] or 0 for sample in report["memory"]),
                default=0,
            )
        except (OSError, ValueError) as error:
            report.update(status="failed", preservation_error=str(error))
        report["wall_seconds"] = time.perf_counter() - started
        save(args.output, report)
        if report["status"] == "failed" and sys.exc_info()[0] is None:
            raise RuntimeError("Reference validation failed; inspect the report.")


def compare_answers(expected, actual, tolerance=TOLERANCE):
    differences = []

    def compare(left, right):
        if isinstance(left, dict):
            if not isinstance(right, dict) or left.keys() != right.keys():
                raise AssertionError("answer fields or probability labels differ")
            for key in left:
                compare(left[key], right[key])
        elif isinstance(left, list):
            if not isinstance(right, list) or len(left) != len(right):
                raise AssertionError("answer list shapes differ")
            for a, b in zip(left, right):
                compare(a, b)
        elif isinstance(left, (int, float)) and not isinstance(left, bool):
            if (
                not isinstance(right, (int, float))
                or isinstance(right, bool)
                or not math.isfinite(left)
                or not math.isfinite(right)
            ):
                raise AssertionError("answer must contain finite numeric values")
            difference = abs(left - right)
            differences.append(difference)
            if difference > tolerance:
                raise AssertionError(
                    f"decision differs by {difference}; fixed tolerance {tolerance}"
                )
        elif left != right:
            raise AssertionError(f"decision label differs: {left!r} != {right!r}")

    compare(expected, actual)
    return max(differences, default=0.0)


def http(url, path, payload=None, timeout=300):
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode()
    request = urllib.request.Request(
        url + path,
        data=data,
        headers={"Content-Type": "application/json; charset=utf-8"},
    )
    started = time.perf_counter()
    try:
        response = urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as error:
        response = error
    with response:
        return response.status, json.load(response), time.perf_counter() - started


def verify_product_metadata(response):
    metadata = response.get("metadata", {})
    for name, expected in {
        "backbone_dtype": "bfloat16",
        "head_dtype": "float32",
        "candidate_batch_size": 1,
        "max_sequence_tokens": MAX_SEQUENCE_TOKENS,
    }.items():
        if metadata.get(name) != expected:
            raise AssertionError(
                f"unexpected product inference metadata {name}: {metadata}"
            )


def clone_source(source, destination):
    verified = verify_source(source)
    for name in verified:
        original, target = source / name, destination / name
        if not original.is_file():
            raise ValueError(f"source artifact missing: {name}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if name.endswith(".safetensors"):
            try:
                os.link(original, target)
                continue
            except OSError:
                pass
        shutil.copyfile(original, target)
    if verify_source(destination) != verified:
        raise ValueError("private service source failed verification")


def service_run(args, expected, child_environment, report, round_number):
    process = None
    started = time.perf_counter()
    route = "model_id" if round_number == 1 else "local_unicode_path"
    check = {
        "status": "running",
        "round": round_number,
        "route": route,
        "cases": [],
        "memory": [],
        "selected_case_ids": [case["id"] for case in expected["cases"]],
    }
    report["service_runs"].append(check)
    temporary = tempfile.TemporaryDirectory(
        prefix="open-jev-cli-", dir=args.workspace, delete=False
    )
    workspace = Path(temporary.name)
    tree = before = source = None
    try:
        source = (
            workspace / "checkpoint" / MODEL
            if route == "model_id"
            else workspace / "本地 模型" / "Open-Jev 2B"
        )
        clone_source(args.source, source)
        before = verify_source(source)
        command = [
            sys.executable,
            "-B",
            "-m",
            "vllm_jev.cli",
            "serve",
            MODEL if route == "model_id" else str(source),
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
            tree = OwnedProcessTree(process, check)
            deadline = time.monotonic() + args.startup_timeout
            while time.monotonic() < deadline:
                tree.refresh()
                if process.poll() is not None:
                    raise RuntimeError(
                        f"product CLI exited before readiness; inspect {log}"
                    )
                try:
                    status, health, _ = http(url, "/health", timeout=1)
                    if status == 200:
                        break
                except (OSError, ValueError):
                    pass
                time.sleep(0.2)
            else:
                raise TimeoutError(f"product CLI readiness timeout; inspect {log}")
            for key, value in {
                "model": MODEL,
                "device": "cpu",
                "revision": REVISION,
                "experimental": True,
            }.items():
                if health.get(key) != value:
                    raise AssertionError(f"unexpected health field {key}: {health}")
            check["health"] = health
            serving_process = tree.listener(args.port)
            check["memory"].append(
                memory_sample("after_ready", started, serving_process)
            )
            for case in expected["cases"]:
                status, response, elapsed = http(
                    url,
                    "/v1/systemone",
                    {
                        "model": MODEL,
                        "state": case["state"],
                        "questions": case["questions"],
                    },
                    timeout=args.request_timeout,
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
                verify_product_metadata(response)
                row["max_absolute_difference"] = compare_answers(
                    case["answers"], response.get("answers")
                )
                row["answers_match_original"] = True
                check["memory"].append(
                    memory_sample(f"after_{case['id']}", started, serving_process)
                )
                save(args.output, report)
                print(
                    json.dumps(
                        {
                            "event": "service_case_completed",
                            "round": round_number,
                            "route": route,
                            "id": case["id"],
                            "seconds": elapsed,
                            "max_difference": row["max_absolute_difference"],
                        }
                    ),
                    flush=True,
                )
            first = expected["cases"][0]
            base = {
                "model": MODEL,
                "state": first["state"],
                "questions": first["questions"],
            }
            invalid = {
                "cache_salt": {**base, "cache_salt": "synthetic-caller"},
                "image": {**base, "images": ["data:image/png;base64,AA=="]},
                "video": {**base, "videos": ["data:video/mp4;base64,AA=="]},
                "model": {**base, "model": "unloaded/model"},
                "question_type": {
                    **base,
                    "questions": {"q": {"type": "unsupported"}},
                },
                "empty_questions": {**base, "questions": {}},
                "overlength": {
                    **base,
                    "state": "token " * (MAX_SEQUENCE_TOKENS + 100),
                },
            }
            check["rejections"] = {}
            for name, payload in invalid.items():
                status, response, elapsed = http(
                    url, "/v1/systemone", payload, timeout=15
                )
                check["rejections"][name] = {
                    "status": status,
                    "response": response,
                    "seconds": elapsed,
                }
                if status != 422:
                    raise AssertionError(f"unsupported {name} was not rejected")
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as clients:
                pending = [
                    clients.submit(
                        http, url, "/v1/systemone", base, args.request_timeout
                    )
                    for _ in range(2)
                ]
                time.sleep(0.1)
                unfinished = sum(not future.done() for future in pending)
                status, health, elapsed = http(url, "/health", timeout=5)
                unfinished_after_health = sum(not future.done() for future in pending)
                check["concurrent_health"] = {
                    "status": status,
                    "seconds": elapsed,
                    "response": health,
                    "unfinished_decision_requests": unfinished,
                    "unfinished_decision_requests_after_health": unfinished_after_health,
                }
                responses = [future.result() for future in pending]
            if status != 200 or unfinished < 1 or unfinished_after_health < 1:
                raise AssertionError(
                    "health was not verified while decision requests were unfinished"
                )
            check["concurrent_requests"] = []
            for status, response, elapsed in responses:
                if status != 200:
                    raise AssertionError("concurrent decision request failed")
                verify_product_metadata(response)
                difference = compare_answers(first["answers"], response.get("answers"))
                check["concurrent_requests"].append(
                    {
                        "status": status,
                        "seconds": elapsed,
                        "max_absolute_difference": difference,
                    }
                )
            check["memory"].append(
                memory_sample("after_concurrent_requests", started, serving_process)
            )
            check["peak_wset_bytes"] = max(
                sample["peak_wset_bytes"] or 0 for sample in check["memory"]
            )
            check["status"] = "passed"
    except BaseException as error:
        check.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        active_error = sys.exc_info()[0] is not None
        okay = finish_owned_run(temporary, tree, process, check, source, before)
        check["wall_seconds"] = time.perf_counter() - started
        try:
            save(args.output, report)
        except Exception as error:
            check["progress_write_error"] = str(error)
            if not active_error:
                raise
        if not okay and sys.exc_info()[0] is None:
            raise RuntimeError(
                "Service cleanup failed; inspect retained workspace details."
            )


def run_reference_process(args, command, child_environment, log, report):
    # Import before launch so missing monitoring support cannot leave an untracked child.
    import psutil  # noqa: F401

    temporary = tempfile.TemporaryDirectory(
        prefix="open-jev-author-", dir=args.workspace, delete=False
    )
    command = [*command, "--reference-workspace", temporary.name]
    record = {"status": "running"}
    report.update(
        reference_command=command,
        reference_log=str(log),
        reference_process=record,
    )
    process = tree = None
    try:
        with log.open("w", encoding="utf-8") as output:
            process = subprocess.Popen(
                command,
                cwd=args.workspace,
                env=child_environment,
                stdout=output,
                stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            tree = OwnedProcessTree(process, record)
            result = tree.wait(args.reference_timeout)
            if result:
                raise subprocess.CalledProcessError(result, command)
            record["status"] = "passed"
    except BaseException as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        okay = finish_owned_run(temporary, tree, process, record)
        if not okay and sys.exc_info()[0] is None:
            raise RuntimeError(
                "Reference cleanup failed; inspect retained workspace details."
            )


def validate_reference_report(args, path, source_hashes):
    """Accept only a complete matching author run, before any service is started."""

    def require(condition, field):
        if not condition:
            raise ValueError(f"reference report mismatch or incomplete field: {field}")

    def equal(actual, expected, field):
        require(type(actual) is type(expected) and actual == expected, field)

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f"duplicate JSON key {key}")
            result[key] = value
        return result

    def finite(value, field, minimum=0):
        require(
            type(value) in (int, float) and math.isfinite(value) and value >= minimum,
            field,
        )

    def same_json(actual, expected, field):
        # Preserve object order too: JSON state/question order can affect prompts.
        equal(
            json.dumps(actual, ensure_ascii=False),
            json.dumps(expected, ensure_ascii=False),
            field,
        )

    def invalid_constant(value):
        raise ValueError(f"non-finite reference JSON constant: {value}")

    raw = path.read_bytes()
    expected = json.loads(
        raw, object_pairs_hook=unique_object, parse_constant=invalid_constant
    )
    require(isinstance(expected, dict), "report object")
    equal(expected.get("status"), "passed", "status")
    for flag in (
        "measured",
        "source_unchanged",
        "hf_snapshot_unchanged",
        "private_checkpoint_unchanged",
    ):
        equal(expected.get(flag), True, flag)
    require(
        not any(
            key in expected for key in ("error", "preservation_error", "source_error")
        ),
        "errors",
    )
    for key, value in {
        "model": MODEL,
        "revision": REVISION,
        "base_revision": BASE_REVISION,
        "sdk_commit": SDK_COMMIT,
        "sdk_source_sha256": SDK_HASHES,
        "source_sha256": source_hashes,
        "threads": args.threads,
        "reference": "Unmodified jev.serving.load_predictor().predict(), Torch CPU",
        "selected_case_ids": [name for name, _ in selected_cases(args)],
    }.items():
        equal(expected.get(key), value, key)
    equal(source_hashes, SOURCE_HASHES, "current source hashes")
    for name, checksum in SDK_HASHES.items():
        equal(
            digest(args.sdk_source / "jev" / name),
            checksum,
            f"current author source {name}",
        )
    snapshot = args.hf_cache / "models--Qwen--Qwen3.5-2B/snapshots" / BASE_REVISION
    equal(
        expected.get("hf_snapshot_sha256"),
        verify_base(snapshot),
        "current HF snapshot hashes",
    )
    runtime = expected.get("environment", {})
    current = environment()
    require(
        str(runtime.get("platform", "")).startswith("Windows-"),
        "reference Windows environment",
    )
    require(str(runtime.get("python", "")).startswith("3.12."), "reference Python 3.12")
    for name, version in {
        "torch": "2.8.0+cpu",
        "transformers": "5.10.2",
        "peft": "0.19.1",
        "accelerate": "1.13.0",
    }.items():
        equal(
            runtime.get("packages", {}).get(name), version, f"reference package {name}"
        )
        equal(current["packages"].get(name), version, f"current package {name}")
    inference = expected.get("inference", {})
    for key, value in {
        "dtype": args.dtype,
        "head_dtype": "float32",
        "batch_size": 1,
        "prefix_cache": False,
        "backend": "torch",
        "sdk_patches": False,
        "max_sequence_tokens": MAX_SEQUENCE_TOKENS,
        "actual_embedding_dtype": "torch.bfloat16",
        "actual_head_dtype": "torch.float32",
        "actual_backbone_dtypes": ["torch.bfloat16", "torch.float32"],
        "actual_text_backbone": "Qwen3_5TextModel",
        "cuda_available": False,
    }.items():
        equal(inference.get(key), value, f"inference.{key}")
    equal(args.dtype, "bfloat16", "product comparison dtype")
    # Earlier driver reports predate this field but already checked every parameter
    # was on CPU before recording actual dtypes. Their CPU wheel and reference marker
    # are required above; never infer CPU from cuda_available alone.
    if "actual_parameter_devices" in inference:
        equal(
            inference["actual_parameter_devices"], ["cpu"], "actual parameter devices"
        )
    finite(inference.get("backbone_parameter_bytes"), "backbone bytes", 1)
    equal(inference.get("head_parameter_bytes"), 8196, "FP32 head bytes")
    checkpoint_digest = hashlib.sha256()
    for name in sorted(CHECKPOINT_FILES):
        checkpoint_digest.update(str(Path(name)).encode() + b"\0")
        checkpoint_digest.update(
            (args.source / "package/checkpoint" / name).read_bytes()
        )
    cases = expected.get("cases")
    require(
        isinstance(cases, list) and len(cases) == len(selected_cases(args)),
        "case count",
    )
    for case, (name, state) in zip(cases, selected_cases(args)):
        require(isinstance(case, dict), "case object")
        equal(case.get("id"), name, "case order/ID")
        same_json(case.get("state"), state, f"{name} state")
        same_json(case.get("questions"), QUESTIONS, f"{name} questions")
        response = case.get("response", {})
        equal(response.get("model"), BASE_MODEL, f"{name} response model")
        answers = case.get("answers")
        require(
            isinstance(answers, dict) and answers.keys() == QUESTIONS.keys(),
            f"{name} answers",
        )
        same_json(answers, response.get("answers"), f"{name} recorded answers")
        compare_answers(answers, answers, tolerance=0)
        validate_recorded_answers(answers)
        metadata = response.get("metadata", {})
        for key, value in {
            "method": "lora_decision_head",
            "temperature": 1.518796342858676,
            "candidate_sequences": 6,
            "prefix_cache": {"enabled": False, "mode": "independent_candidates"},
            "base_revision": BASE_REVISION,
            "max_length": MAX_SEQUENCE_TOKENS,
            "backbone_parameter_dtypes": ["torch.bfloat16", "torch.float32"],
            "checkpoint_sha256": checkpoint_digest.hexdigest(),
        }.items():
            equal(metadata.get(key), value, f"{name} metadata.{key}")
        equal(
            response.get("usage", {}).get("output_tokens"), 0, f"{name} output tokens"
        )
        finite(response.get("usage", {}).get("input_tokens"), f"{name} input tokens", 1)
        finite(case.get("seconds"), f"{name} seconds")
        finite(metadata.get("inference_seconds"), f"{name} inference seconds")
    memory = expected.get("memory")
    stages = [
        "before_load",
        "after_load",
        *[f"after_{name}" for name, _ in selected_cases(args)],
        "final",
    ]
    require(isinstance(memory, list), "memory samples")
    equal([sample.get("stage") for sample in memory], stages, "completed memory stages")
    for sample in memory:
        finite(sample.get("peak_wset_bytes"), "Windows peak working set", 1)
    equal(
        expected.get("peak_wset_bytes"),
        max(sample["peak_wset_bytes"] for sample in memory),
        "peak working set",
    )
    finite(expected.get("cold_load_seconds"), "cold load seconds")
    finite(expected.get("wall_seconds"), "wall seconds")
    checksum = hashlib.sha256(raw).hexdigest()
    equal(digest(path), checksum, "report changed during validation")
    return expected, {
        "path": str(path),
        "sha256": checksum,
        "reused": True,
        "rerun_this_invocation": False,
    }


def validate_recorded_answers(answers):
    """Reject incomplete or internally inconsistent cached decision outputs."""

    def number(value, minimum, maximum):
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or not minimum <= value <= maximum
        ):
            raise ValueError("reference report has an invalid decision value")

    for name, question in QUESTIONS.items():
        answer = answers[name]
        kind = question["type"]
        fields = (
            {"type", "noul"}
            if kind == "noul"
            else {
                "type",
                "probabilities",
                "confidence",
                "choice" if kind == "choice" else "score",
            }
        )
        if kind == "score":
            fields.add("legend")
        if (
            not isinstance(answer, dict)
            or answer.keys() != fields
            or answer["type"] != kind
        ):
            raise ValueError("reference report has incomplete answer fields")
        if kind == "noul":
            number(answer["noul"], 0, 1)
            continue
        labels = (
            list(question["criteria"])
            if kind == "choice"
            else [str(index) for index in range(len(question["criteria"]))]
        )
        probabilities = answer["probabilities"]
        if not isinstance(probabilities, dict) or list(probabilities) != labels:
            raise ValueError("reference report probability labels differ")
        for value in probabilities.values():
            number(value, 0, 1)
        if not math.isclose(sum(probabilities.values()), 1, rel_tol=0, abs_tol=1e-12):
            raise ValueError("reference report probabilities do not sum to one")
        number(answer["confidence"], 0, 1)
        if kind == "choice":
            if answer["choice"] != max(probabilities, key=probabilities.get):
                raise ValueError("reference report choice does not match probabilities")
        else:
            number(answer["score"], 0, len(labels) - 1)
            score = sum(
                index * probabilities[label] for index, label in enumerate(labels)
            )
            if not math.isclose(answer["score"], score, rel_tol=0, abs_tol=1e-12):
                raise ValueError("reference report score does not match probabilities")
            if answer["legend"] != dict(zip(labels, question["criteria"])):
                raise ValueError("reference report score legend differs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, help="Product source root, not package/checkpoint."
    )
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sdk-source", type=Path, required=True)
    parser.add_argument("--hf-cache", type=Path)
    parser.add_argument("--reference-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--reference-only", action="store_true")
    parser.add_argument(
        "--reference-report",
        type=Path,
        help="Reuse a completed, matching original-reference.json instead of rerunning it.",
    )
    parser.add_argument(
        "--reference-child", action="store_true", help=argparse.SUPPRESS
    )
    parser.add_argument("--reference-workspace", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--case", action="append", choices=[name for name, _ in CASES])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--port", type=int, default=18798)
    parser.add_argument("--startup-timeout", type=float, default=600)
    parser.add_argument("--request-timeout", type=float, default=300)
    parser.add_argument("--reference-timeout", type=float, default=1800)
    args = parser.parse_args()
    if args.reference_report and (args.reference_only or args.reference_child):
        parser.error(
            "--reference-report cannot be combined with --reference-only or --reference-child"
        )
    if (
        args.reference_report
        and args.reference_report.resolve() == args.output.resolve()
    ):
        parser.error("--output must not overwrite the reused reference report")
    if args.dtype == "float32" and not (args.reference_only or args.reference_child):
        parser.error(
            "float32 is only supported with --reference-only; the product serves BF16"
        )
    if args.source is None and args.workspace is None:
        parser.error("provide --source or --workspace")
    if args.reference_child and args.reference_workspace is None:
        parser.error("reference child requires a parent-owned --reference-workspace")
    if args.threads < 1 or not 1 <= args.port <= 65535:
        parser.error("threads must be positive and port must be in 1..65535")
    if min(args.startup_timeout, args.request_timeout, args.reference_timeout) <= 0:
        parser.error("timeouts must be positive")
    args.source = (args.source or args.workspace / "checkpoint" / MODEL).resolve()
    args.workspace = (args.workspace or args.source.parents[2]).resolve()
    args.hf_cache = (args.hf_cache or args.workspace.parent / "hf-cache").resolve()
    args.sdk_source, args.output = args.sdk_source.resolve(), args.output.resolve()
    args.workspace.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.reference_child:
        reference(args)
        return
    report = {
        "status": "running",
        "model": MODEL,
        "revision": REVISION,
        "environment": environment(),
        "dtype": args.dtype,
        "measured": False,
        "reference_only": args.reference_only,
        "service_runs": [],
        "answer_absolute_tolerance": TOLERANCE,
        "scope": "Selected synthetic text/JSON requests, Choice/Noul/Score; same-dtype author parity, not task quality or performance benchmarking.",
        "selected_case_ids": [name for name, _ in selected_cases(args)],
    }
    if sys.platform != "win32":
        report.update(status="skipped", reason="Native Windows is required.")
        save(args.output, report)
        return
    child_environment = os.environ.copy()
    for key in list(child_environment):
        if key.startswith("JEV_") or key == "VLLM_JEV_DECISION_TEMPLATE":
            child_environment.pop(key)
    child_environment.update(
        PYTHONPATH="",
        PYTHONDONTWRITEBYTECODE="1",
        USE_TF="0",
        USE_FLAX="0",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1",
        HF_HUB_CACHE=str(args.hf_cache),
        JEV_TORCH_DTYPE=args.dtype,
        TOKENIZERS_PARALLELISM="false",
    )
    reference_output = args.output.with_name(
        args.output.stem + "-original-reference.json"
    )
    log = args.output.with_name(args.output.stem + "-original-reference.log")
    started = time.perf_counter()
    before = None
    try:
        if not args.reference_only:
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", args.port))
        before = verify_source(args.source)
        command = [
            str(args.reference_python.resolve()),
            "-B",
            str(Path(__file__).resolve()),
            "--reference-child",
            "--source",
            str(args.source),
            "--workspace",
            str(args.workspace),
            "--output",
            str(reference_output),
            "--sdk-source",
            str(args.sdk_source),
            "--hf-cache",
            str(args.hf_cache),
            "--threads",
            str(args.threads),
            "--dtype",
            args.dtype,
        ]
        for case in args.case or []:
            command.extend(["--case", case])
        if args.reference_report:
            reference_output = args.reference_report.resolve()
            expected, provenance = validate_reference_report(
                args, reference_output, before
            )
        else:
            run_reference_process(args, command, child_environment, log, report)
            expected = json.loads(reference_output.read_text(encoding="utf-8"))
            if expected["status"] != "passed" or len(expected["cases"]) != len(
                selected_cases(args)
            ):
                raise AssertionError(
                    "author reference did not finish every selected case"
                )
            provenance = {
                "path": str(reference_output),
                "sha256": digest(reference_output),
                "reused": False,
                "rerun_this_invocation": True,
            }
        report.update(
            original_reference=str(reference_output),
            reference_provenance=provenance,
            measured=True,
            reference_peak_wset_bytes=expected["peak_wset_bytes"],
        )
        if not args.reference_only:
            product_environment = {**child_environment, "PYTHONPATH": str(ROOT)}
            for round_number in (1, 2):
                service_run(args, expected, product_environment, report, round_number)
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        try:
            if before is not None:
                report["source_unchanged"] = verify_source(args.source) == before
                if not report["source_unchanged"]:
                    report["status"] = "failed"
        except (OSError, ValueError) as error:
            report.update(
                status="failed", source_unchanged=False, source_error=str(error)
            )
        report["wall_seconds"] = time.perf_counter() - started
        save(args.output, report)
        print(json.dumps({"status": report["status"], "output": str(args.output)}))
        if report["status"] == "failed" and sys.exc_info()[0] is None:
            raise RuntimeError("Validation failed; inspect the report.")


if __name__ == "__main__":
    main()
