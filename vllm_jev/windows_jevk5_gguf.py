"""Official JevK5 4B/9B Q5_K_M with the author's calibrated top-40 readout.

Bounded single-pass profile: at most 16 choices and 1024 full prompt tokens.
No downloaded Python is imported. The original SDK is evaluated separately.
"""

import json
import math
import os
import shutil
import sys
import tempfile
import threading
import time
import urllib.request
import zipfile
from pathlib import Path

from filelock import FileLock
from huggingface_hub import snapshot_download

from .windows_jevk5 import _questions
from .windows_mica import RUNTIME_HASHES as MICA_RUNTIME_HASHES
from .windows_mica import RUNTIME_URL, RUNTIME_ZIP_SHA256, _answer, _sha256

MODEL_ID = "alibiserikbay/JevK5"
MODEL_ID_9B = "alibiserikbay/JevK5-9B"
ARTIFACT_ID = "alibiserikbay/JevK5-GGUF"
REVISION = "ec67b0bfce5119a8b11a2cdb430bb43e3fa3e82a"
SDK_COMMIT = "f26426d16f59e8bbe1470e5b162cc89329e29b29"
PROTOCOL = "jevk5_gguf_v1"
BACKEND = "jevk5-llama-cpp-q5-b11010"
GGUF_FILENAME = "jevk5-4b-v0.3-Q5_K_M.gguf"
MAX_SEQUENCE_TOKENS = 1024
TEMPERATURE = 1.22
HF_SOURCE_HASHES = {
    GGUF_FILENAME: "b1df9869dd14c0f5b243fa956dd985f0141c3d23ec4847da4c503c73c1050601",
    "README.md": "5c75f91401539afe2ce722e87f8e81f1bbd23cb3cea16758180be4a316389fbe",
    "SHA256SUMS": "ad21b6c0b7a21fb6cc6980495a91bb2a1c97078beacfc065d4f19e934d6d9422",
}
MODELS = {
    MODEL_ID: {
        "revision": REVISION,
        "filename": GGUF_FILENAME,
        "temperature": TEMPERATURE,
        "files": HF_SOURCE_HASHES,
    },
    MODEL_ID_9B: {
        "revision": REVISION,
        "filename": "jevk5-9b-v0.3.3-Q5_K_M.gguf",
        "temperature": 1.316,
        "files": {
            "jevk5-9b-v0.3.3-Q5_K_M.gguf": "da4a4becd04c3beabd4cefbd09b546ce2648740d923adc8d14ae1d0cb921486e",
            "README.md": HF_SOURCE_HASHES["README.md"],
            "SHA256SUMS": HF_SOURCE_HASHES["SHA256SUMS"],
        },
    },
}
RUNTIME_HASHES = {
    **MICA_RUNTIME_HASHES,
    "ggml-rpc.dll": "fd0f79c26c25fdd342928f0404f9e12c05ca4bdc6592cff8a36374a4e24915f3",
    "llama-common.dll": "99834085a3ee03a8e903f5f3b6100a6c49bd1b0b64e6dfd52170a0760ce7897a",
    "llama-server-impl.dll": "b7b05aef4636b30357cc1fc1e62b832d09b0f608f1d40d78c4ff99b52d97f96d",
    "llama-server.exe": "f156c6fcfb61c7dd259e2c577ccbab214185083650f18a9cdfdfb9c38d6cf3b3",
    "mtmd.dll": "6606247614f88d665693aae9b178f9bd1c704268d13997d0961112690ceb3027",
}
SOURCE_HASHES = {
    **HF_SOURCE_HASHES,
    **{f"runtime/{name}": value for name, value in RUNTIME_HASHES.items()},
}


def _require_model(model_id):
    try:
        return MODELS[model_id]
    except KeyError:
        raise ValueError(f"unsupported Windows JevK5 GGUF model: {model_id}") from None


def source_hashes(model_id=MODEL_ID):
    return {
        **_require_model(model_id)["files"],
        **{f"runtime/{name}": value for name, value in RUNTIME_HASHES.items()},
    }


def validate_template(template=None):
    if template is None:
        from .decision_template import current_template

        template = current_template()
    if template.config not in ({}, {"layout": "native"}):
        raise ValueError(
            "Windows JevK5 GGUF supports only the default/native decision template"
        )


def verify_source(checkpoint, model_id=MODEL_ID):
    specification = _require_model(model_id)
    checkpoint = Path(checkpoint)
    for folder, names, directories in (
        (checkpoint, specification["files"], {"runtime", ".cache"}),
        (checkpoint / "runtime", RUNTIME_HASHES, set()),
    ):
        if not folder.is_dir() or folder.is_symlink() or folder.is_junction():
            raise ValueError("JevK5 GGUF source/runtime directory is missing or linked")
        for entry in folder.iterdir():
            if entry.is_symlink() or entry.is_junction():
                raise ValueError(f"linked JevK5 GGUF source entry: {entry.name}")
            if entry.name in directories and entry.is_dir():
                continue
            if entry.name not in names or not entry.is_file() or entry.is_symlink():
                raise ValueError(f"unexpected JevK5 GGUF source entry: {entry.name}")
    expected_hashes = source_hashes(model_id)
    for name, expected in expected_hashes.items():
        path = checkpoint / name
        if not path.is_file() or path.is_symlink() or _sha256(path) != expected:
            raise ValueError(
                f"JevK5 GGUF source checksum mismatch or missing file: {name}"
            )
    return expected_hashes


def identify_source(checkpoint):
    candidates = [
        model_id
        for model_id, specification in MODELS.items()
        if (Path(checkpoint) / specification["filename"]).is_file()
    ]
    if len(candidates) != 1:
        raise ValueError("JevK5 GGUF source must contain exactly one pinned release")
    verify_source(checkpoint, candidates[0])
    return candidates[0]


def prepare(model_id, workspace):
    specification = _require_model(model_id)
    validate_template()
    output = Path(workspace).resolve() / "checkpoint" / model_id
    output.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output) + ".lock"):
        if output.exists():
            verify_source(output, model_id)
            return output
        with tempfile.TemporaryDirectory(
            prefix=".jevk5-gguf-", dir=output.parent
        ) as directory:
            directory = Path(directory)
            source = directory / "source"
            snapshot_download(
                repo_id=ARTIFACT_ID,
                revision=REVISION,
                token=False,
                local_dir=source,
                allow_patterns=list(specification["files"]),
            )
            archive = directory / "runtime.zip"
            with (
                urllib.request.urlopen(RUNTIME_URL, timeout=60) as incoming,
                archive.open("wb") as outgoing,
            ):
                shutil.copyfileobj(incoming, outgoing)
            if _sha256(archive) != RUNTIME_ZIP_SHA256:
                raise ValueError("JevK5 GGUF CPU archive checksum mismatch")
            runtime = source / "runtime"
            runtime.mkdir()
            with zipfile.ZipFile(archive) as bundle:
                for name in RUNTIME_HASHES:
                    entries = [
                        entry for entry in bundle.infolist() if entry.filename == name
                    ]
                    if len(entries) != 1 or entries[0].is_dir():
                        raise ValueError(f"CPU archive requires exactly one {name}")
                    with (
                        bundle.open(entries[0]) as incoming,
                        (runtime / name).open("wb") as outgoing,
                    ):
                        shutil.copyfileobj(incoming, outgoing)
            verify_source(source, model_id)
            source.replace(output)
    return output


def rows_for(state, questions):
    if not isinstance(state, (str, dict, list)):
        raise ValueError("decision state must be text or JSON")
    text = (
        state
        if isinstance(state, str)
        else json.dumps(state, ensure_ascii=False, allow_nan=False)
    )
    if len(text) > 50_000:
        raise ValueError("decision state exceeds 50,000 characters")
    rows = []
    for identifier, question in _questions(questions).items():
        kind, criteria = question["type"], question.get("criteria")
        if kind == "choice":
            pairs = [(key, value or key) for key, value in criteria.items()]
        elif kind == "noul":
            pairs = [
                (key, (criteria or {}).get(key) or f"The proposition is {key}.")
                for key in ("true", "false")
            ]
        else:
            pairs = [(str(i), value) for i, value in enumerate(criteria)]
        payload = {
            "evidence": state,
            "criterion": question["instructions"],
            "options": [
                {"letter": chr(65 + i), "description": f"{key}: {value}"}
                for i, (key, value) in enumerate(pairs)
            ],
        }
        system = (
            "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
            "Respond with only its uppercase letter, with no explanation or reasoning."
        )
        prompt = (
            f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n"
            + json.dumps(payload, ensure_ascii=False, allow_nan=False)
            + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )
        rows.append(
            {
                "id": identifier,
                "kind": kind,
                "prompt": prompt,
                "candidates": [{"id": key, "text": value} for key, value in pairs],
            }
        )
    if sum(len(row["candidates"]) for row in rows) > 256:
        raise ValueError("System One exceeds 256 candidates")
    return rows


def read_probabilities(output, count, model_id=MODEL_ID):
    """Preserve the published SDK's top-k floor, including absent letters."""
    temperature = _require_model(model_id)["temperature"]
    top = output["completion_probabilities"][0]["top_logprobs"]
    if not top or any(
        not isinstance(item["token"], str) or not math.isfinite(item["logprob"])
        for item in top
    ):
        raise RuntimeError("JevK5 returned invalid top-token log probabilities")
    seen = {item["token"]: item["logprob"] for item in top}
    floor = min(seen.values()) - 2.0
    logits = [seen.get(chr(65 + i), floor) for i in range(count)]
    weights = [math.exp((value - max(logits)) / temperature) for value in logits]
    return [value / sum(weights) for value in weights], sum(
        chr(65 + i) not in seen for i in range(count)
    )


class JevK5GGUFAgent:
    device = "cpu"

    def __init__(self, server, model_id=MODEL_ID):
        self.specification = _require_model(model_id)
        self.server = server
        self.model_id, self.revision = model_id, REVISION
        self._lock = threading.RLock()

    def system_one(self, state, questions):
        with self._lock:
            validate_template()
            if self.server is None:
                raise RuntimeError("JevK5 GGUF CPU model is closed")
            rows = rows_for(state, questions)
            for row in rows:
                tokens = self.server.post(
                    "/tokenize",
                    {
                        "content": row["prompt"],
                        "add_special": False,
                        "parse_special": True,
                    },
                )["tokens"]
                if not tokens or len(tokens) > MAX_SEQUENCE_TOKENS:
                    raise ValueError(
                        "JevK5 GGUF complete prompt exceeds 1024 tokens; evidence was not truncated"
                    )
                row["tokens"] = tokens
            started = time.perf_counter()
            answers, missing = {}, 0
            for row in rows:
                output = self.server.post(
                    "/completion",
                    {
                        "prompt": row["tokens"],
                        "n_predict": 1,
                        "n_probs": 40,
                        "temperature": 0,
                        "cache_prompt": False,
                    },
                )
                if output.get("truncated") or output.get("tokens_evaluated") != len(
                    row["tokens"]
                ):
                    raise RuntimeError(
                        "JevK5 GGUF did not evaluate the complete uncached prompt"
                    )
                probabilities, absent = read_probabilities(
                    output, len(row["candidates"]), self.model_id
                )
                missing += absent
                answers[row["id"]] = (
                    {"type": "noul", "noul": probabilities[0]}
                    if row["kind"] == "noul"
                    else _answer(row, probabilities)
                )
            count = sum(len(row["tokens"]) for row in rows)
            return {
                "model": self.model_id,
                "answers": answers,
                "usage": {"input_tokens": count, "output_tokens": 0},
                "internal_usage": {"compute_tokens": count},
                "metadata": {
                    "backend": BACKEND,
                    "device": "cpu",
                    "experimental": True,
                    "model": self.model_id,
                    "revision": REVISION,
                    "artifact_repository": ARTIFACT_ID,
                    "artifact_filename": self.specification["filename"],
                    "weight_format": "GGUF",
                    "quantization": "Q5_K_M",
                    "runtime_revision": "b11010",
                    "threads": self.server.threads,
                    "n_gpu_layers": 0,
                    "row_batch_size": 1,
                    "max_sequence_tokens": MAX_SEQUENCE_TOKENS,
                    "context_tokens": 2048,
                    "temperature": self.specification["temperature"],
                    "top_k": 40,
                    "missing_letter_margin": 2.0,
                    "missing_letters": missing,
                    "cache_reuse": False,
                    "runtime_generated_tokens": len(rows),
                    "torch_imported": "torch" in sys.modules,
                    "inference_seconds": time.perf_counter() - started,
                },
            }

    def close(self):
        with self._lock:
            if self.server is not None:
                self.server.close()
                self.server = None


def load_agent(checkpoint, model_id=MODEL_ID, threads=None):
    from .windows_llama_server import CPUServer

    specification = _require_model(model_id)
    validate_template()
    threads = min(4, os.cpu_count() or 1) if threads is None else threads
    if type(threads) is not int or not 1 <= threads <= 4:
        raise ValueError("JevK5 GGUF CPU threads must be between 1 and 4")
    verify_source(checkpoint, model_id)
    checkpoint = Path(checkpoint).resolve()
    server = CPUServer(
        checkpoint / "runtime/llama-server.exe",
        checkpoint / specification["filename"],
        threads,
    )
    return JevK5GGUFAgent(server, model_id)
