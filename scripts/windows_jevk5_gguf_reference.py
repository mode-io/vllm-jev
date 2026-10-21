"""Execute the unchanged pinned JevK5GGUF author SDK in a separate process.

Only source verification, process lifecycle and passive observations live here;
the original SDK renders, tokenizes, infers, applies the top-k floor and calibrates.
"""

import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from windows_mica_reference import SOURCE_HASHES as MICA_SOURCE_HASHES

MODEL_ID = "alibiserikbay/JevK5"
MODEL_ID_9B = "alibiserikbay/JevK5-9B"
REVISION = "ec67b0bfce5119a8b11a2cdb430bb43e3fa3e82a"
SDK_COMMIT = "f26426d16f59e8bbe1470e5b162cc89329e29b29"
GGUF_FILENAME = "jevk5-4b-v0.3-Q5_K_M.gguf"
SOURCE_HASHES = {
    **{
        name: digest
        for name, digest in MICA_SOURCE_HASHES.items()
        if name.startswith("runtime/")
    },
    GGUF_FILENAME: "b1df9869dd14c0f5b243fa956dd985f0141c3d23ec4847da4c503c73c1050601",
    "README.md": "5c75f91401539afe2ce722e87f8e81f1bbd23cb3cea16758180be4a316389fbe",
    "SHA256SUMS": "ad21b6c0b7a21fb6cc6980495a91bb2a1c97078beacfc065d4f19e934d6d9422",
    "runtime/ggml-rpc.dll": "fd0f79c26c25fdd342928f0404f9e12c05ca4bdc6592cff8a36374a4e24915f3",
    "runtime/llama-common.dll": "99834085a3ee03a8e903f5f3b6100a6c49bd1b0b64e6dfd52170a0760ce7897a",
    "runtime/llama-server-impl.dll": "b7b05aef4636b30357cc1fc1e62b832d09b0f608f1d40d78c4ff99b52d97f96d",
    "runtime/llama-server.exe": "f156c6fcfb61c7dd259e2c577ccbab214185083650f18a9cdfdfb9c38d6cf3b3",
    "runtime/mtmd.dll": "6606247614f88d665693aae9b178f9bd1c704268d13997d0961112690ceb3027",
}
# Keep these pins independent of the product adapter. The unchanged author SDK
# owns the prompt, next-token readout, missing-letter floor and calibration.
MODELS = {
    MODEL_ID: {
        "filename": GGUF_FILENAME,
        "temperature": 1.22,
        "knockout_temperature": 0.93,
        "source_hashes": SOURCE_HASHES,
    },
    MODEL_ID_9B: {
        "filename": "jevk5-9b-v0.3.3-Q5_K_M.gguf",
        "temperature": 1.316,
        "knockout_temperature": 1.05,
        "source_hashes": {
            **{
                name: value
                for name, value in SOURCE_HASHES.items()
                if name != GGUF_FILENAME
            },
            "jevk5-9b-v0.3.3-Q5_K_M.gguf": "da4a4becd04c3beabd4cefbd09b546ce2648740d923adc8d14ae1d0cb921486e",
        },
    },
}
SDK_HASHES = {
    "jevk5/__init__.py": "3ec8170fe224e7b9b2ce0333a58ba1cb8066effd4e9d825fc913ef033b907e2b",
    "jevk5/prompt.py": "322d69ba6abea83602f27b7a5aa0d3e6ab00847fd03ddccd35a28ae5b23acc1a",
    "jevk5/gguf.py": "6505de497ce115f0bf85f110cf708fc8535a774f1361b811e3415b3e91a10d03",
}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(root, expected):
    for name, checksum in expected.items():
        path = root / name
        if path.is_symlink() or not path.is_file() or digest(path) != checksum:
            raise ValueError(f"Pinned author input checksum mismatch: {name}")
    return dict(expected)


def source_hashes(model_id=MODEL_ID):
    try:
        return dict(MODELS[model_id]["source_hashes"])
    except KeyError:
        raise ValueError(f"Unsupported author JevK5 GGUF model: {model_id}") from None


def verify_source(root, model_id=MODEL_ID):
    root = Path(root)
    expected = source_hashes(model_id)
    if not root.is_dir() or root.is_symlink() or root.is_junction():
        raise ValueError("Author JevK5 source directory is missing or linked")
    for item in root.rglob("*"):
        name = item.relative_to(root).as_posix()
        if item.is_symlink() or item.is_junction():
            raise ValueError(f"Linked author source input: {name}")
        if name == ".cache" or name.startswith(".cache/"):
            continue
        if (item.is_file() and name not in expected) or (
            item.is_dir() and name != "runtime"
        ):
            raise ValueError(f"Unexpected author source input: {name}")
    return verify(root, expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(MODELS), default=MODEL_ID)
    for name in ("source", "sdk-source", "workspace", "cases", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    specification = MODELS[args.model]
    if sys.platform != "win32" or not 1 <= args.threads <= 4:
        parser.error("native Windows and 1 to 4 threads required")
    for field in ("source", "sdk_source", "workspace", "cases", "output"):
        setattr(args, field, getattr(args, field).resolve())
    if any(
        args.workspace.is_relative_to(root) or args.output.is_relative_to(root)
        for root in (args.source, args.sdk_source)
    ):
        parser.error("workspace/output must be outside original inputs")
    report = {
        "status": "running",
        "model": args.model,
        "revision": REVISION,
        "sdk_commit": SDK_COMMIT,
        "source_hashes": verify_source(args.source, args.model),
        "sdk_hashes": verify(args.sdk_source, SDK_HASHES),
        "runner_sha256": digest(Path(__file__)),
        "cases_sha256": digest(args.cases),
        "python": sys.executable,
        "threads": args.threads,
        "temperature": specification["temperature"],
        "knockout_temperature": specification["knockout_temperature"],
        "cases": [],
    }
    args.workspace.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    private = args.workspace / "author-sdk"
    if private.exists():
        raise ValueError("reference requires a fresh private workspace")
    for name in SDK_HASHES:
        target = private / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.sdk_source / name, target)
    verify(private, SDK_HASHES)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(private))
    from jevk5 import JevK5GGUF

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    command = [
        str(args.source / "runtime/llama-server.exe"),
        "--model",
        str(args.source / specification["filename"]),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--device",
        "none",
        "--gpu-layers",
        "0",
        "--ctx-size",
        "2048",
        "--parallel",
        "1",
        "--threads",
        str(args.threads),
        "--threads-batch",
        str(args.threads),
        "--threads-http",
        "2",
        "--batch-size",
        "1024",
        "--ubatch-size",
        "512",
        "--predict",
        "1",
        "--cache-ram",
        "0",
        "--no-cache-prompt",
        "--no-cont-batching",
        "--no-context-shift",
        "--no-warmup",
        "--no-webui",
        "--no-agent",
        "--offline",
        "--fit",
        "off",
    ]
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.upper().startswith(("LLAMA_", "GGML_", "MTMD_"))
    }
    report["runtime_command"] = command
    process = None
    try:
        with args.output.with_suffix(".server.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                cwd=args.source / "runtime",
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            report["server_pid"] = process.pid
            deadline = time.monotonic() + 180
            while True:
                if process.poll() is not None:
                    raise RuntimeError("author llama-server exited during loading")
                try:
                    with urllib.request.urlopen(url + "/health", timeout=1) as response:
                        if response.status == 200:
                            break
                except (urllib.error.URLError, OSError):
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError("author llama-server startup timed out")
                time.sleep(0.1)
            author = JevK5GGUF(
                url=url,
                temperature=specification["temperature"],
                knockout_temperature=specification["knockout_temperature"],
                top_k=40,
            )
            for case in json.loads(args.cases.read_text(encoding="utf-8")):
                observations, probabilities = {}, {}
                started = time.perf_counter()
                for name, question in case["questions"].items():
                    events = []

                    def observe(frame, event, value):
                        if (
                            event == "return"
                            and frame.f_code is JevK5GGUF._logprobs.__code__
                        ):
                            local = frame.f_locals
                            events.append(
                                {
                                    "prompt": local["prompt"],
                                    "tokens": list(local["tokens"]),
                                    "output": local["out"],
                                }
                            )

                    sys.setprofile(observe)
                    try:
                        probs, count = author.probabilities(case["state"], question)
                    finally:
                        sys.setprofile(None)
                    if (
                        len(events) != 1
                        or count != len(events[0]["tokens"])
                        or not 1 <= count <= 1024
                    ):
                        raise RuntimeError(
                            "reference must observe one complete bounded uncached forward"
                        )
                    probabilities[name] = list(probs.values())
                    observations[name] = events[0]
                report["cases"].append(
                    {
                        "case": case,
                        "probabilities": probabilities,
                        "observations": observations,
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                )
                args.output.write_text(
                    json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            report.update(status="passed", missing_decisions=author.missing)
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if process is not None:
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=15)
            report["server_stopped"] = process.poll() is not None
        report["source_unchanged"] = (
            verify_source(args.source, args.model) == report["source_hashes"]
        )
        report["sdk_unchanged"] = (
            verify(args.sdk_source, SDK_HASHES) == report["sdk_hashes"]
        )
        report["torch_imported"] = "torch" in sys.modules
        report["product_imported"] = any(
            name == "vllm_jev" or name.startswith("vllm_jev.") for name in sys.modules
        )
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
