"""Pinned Decider 4B Q4_K_M on the official Windows CPU llama.cpp wheel.

The product compiler and owned C API readout preserve the author's independent
rows and FP32 calibrated probabilities. Downloaded Python is never executed.
"""

import ctypes
import hashlib
import importlib.metadata
import json
import os
import tempfile
import threading
import time
from pathlib import Path

from filelock import FileLock
from huggingface_hub import snapshot_download

from .windows_decider import _questions

MODEL_ID = "Mapika/decider-4b-GGUF"
REVISION = "b79f09d9ba7837f1b744295ea267b55d08e958ec"
PROTOCOL = "decider_gguf_v1"
BACKEND = "decider-llama-cpp-q4-0.3.35"
GGUF_FILENAME = "decider-4b-v2.1-Q4_K_M.gguf"
MAX_SEQUENCE_TOKENS = 1024
CONTEXT_TOKENS = 2048
WHEEL_URL = "https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.35/llama_cpp_python-0.3.35-py3-none-win_amd64.whl"
WHEEL_SHA256 = "31590ea000d5aff6f05f1e428048e72318a83709288159a5bd4dabec530080bb"
SOURCE_HASHES = {
    GGUF_FILENAME: "c7083fcfc93f650cd66caeade4a3840ef9eeeeb1e80920907c554f70619d7c56",
    "tokenizer.json": "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
    "tokenizer_config.json": "171ecbe7ddae98d11840698f7df2b8d5b4722139db0f0620d3bbf429bd656250",
    "chat_template.jinja": "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715",
    "decider_config.json": "fc83293b6f707172e20176d603ea88dd1a3be2abd3596fdb1ccbec9404626baf",
    "README.md": "7fa5815c0b78d6fee9febdd8d734dbe95bd41138f7421ea28db0622d12cac7c2",
}
# Python bindings and all DLLs loaded by the official CPU wheel are verified
# before import. Other model environments need not install this wheel.
RUNTIME_HASHES = {
    "llama_cpp/__init__.py": "49bc119b3aa3ccaf6f90e6ced6a573293a492026e12e4c6709f11358a2174512",
    "llama_cpp/_ctypes_extensions.py": "3c421b78bc80159d2cf360a00bb5af223fbe4495df2a394a37f0bc5b114146a9",
    "llama_cpp/llama_cpp.py": "d8088bf0f05f9e49ae57d555813b3a6fa1dca6ae72e38fa84c152b4aa0e13d73",
    "llama_cpp/lib/ggml-base.dll": "246df296941ae3a0dcf48aff08516afc857ad74c1ed893ce0f889dc3b4d0581e",
    "llama_cpp/lib/ggml-cpu.dll": "cd91f4ed375998da4da57fedaab1b0638fba8b2af88e74a2632bc046e7fa4850",
    "llama_cpp/lib/ggml.dll": "eeb2b9614d32aa8212866e4736c4bb2b5cf00523faee387f9e354aeec38e4e52",
    "llama_cpp/lib/llama.dll": "fadb7f4ffa452cf0702a9a27938562afcddc2538bcd71977e3ff584b071746ad",
    "llama_cpp/lib/mtmd.dll": "88bd93ebd209909171408f540621a9bb243197b3d11e132535f7092896a270b0",
}
_QUIET = None


def _sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _require_model(model_id):
    if model_id != MODEL_ID:
        raise ValueError(f"unsupported Windows Decider GGUF model: {model_id}")


def validate_template(template=None):
    if template is None:
        from .decision_template import current_template

        template = current_template()
    if template.config not in ({}, {"layout": "native"}):
        raise ValueError("Decider GGUF supports only the default/native template")


def verify_runtime():
    try:
        dist = importlib.metadata.distribution("llama-cpp-python")
    except importlib.metadata.PackageNotFoundError:
        raise ValueError(
            "Decider GGUF needs the official Windows CPU wheel: " + WHEEL_URL
        ) from None
    if dist.version != "0.3.35":
        raise ValueError("Decider GGUF requires llama-cpp-python 0.3.35")
    if "LLAMA_CPP_LIB_PATH" in os.environ:
        raise ValueError("Decider GGUF does not support LLAMA_CPP_LIB_PATH overrides")
    for name, expected in RUNTIME_HASHES.items():
        path = Path(dist.locate_file(name))
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"Decider GGUF official CPU runtime mismatch: {name}")
    dlls = {p.name for p in Path(dist.locate_file("llama_cpp/lib")).glob("*.dll")}
    if dlls != {Path(n).name for n in RUNTIME_HASHES if n.endswith(".dll")}:
        raise ValueError("Decider GGUF runtime contains unexpected backend DLLs")


def verify_source(checkpoint, model_id=MODEL_ID):
    _require_model(model_id)
    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_dir():
        raise ValueError("Decider GGUF source directory is missing")
    for entry in checkpoint.iterdir():
        if entry.name == ".cache" and entry.is_dir():
            continue
        if entry.name not in SOURCE_HASHES or not entry.is_file() or entry.is_symlink():
            raise ValueError(f"unexpected Decider GGUF source entry: {entry.name}")
    for name, expected in SOURCE_HASHES.items():
        path = checkpoint / name
        if not path.is_file() or path.is_symlink() or _sha256(path) != expected:
            raise ValueError(f"Decider GGUF source checksum mismatch: {name}")
    return json.loads((checkpoint / "decider_config.json").read_text(encoding="utf-8"))


def identify_source(checkpoint):
    verify_source(checkpoint)
    return MODEL_ID


def prepare(model_id, workspace):
    _require_model(model_id)
    validate_template()
    verify_runtime()
    output = Path(workspace).resolve() / "checkpoint" / model_id
    output.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output) + ".lock"):
        if output.exists():
            verify_source(output)
            return output
        with tempfile.TemporaryDirectory(
            prefix=".decider-gguf-", dir=output.parent
        ) as d:
            source = Path(d) / "source"
            snapshot_download(
                repo_id=MODEL_ID,
                revision=REVISION,
                token=False,
                local_dir=source,
                allow_patterns=list(SOURCE_HASHES),
            )
            verify_source(source)
            source.replace(output)
    return output


class NativeReadout:
    """Own one CPU context; clear its hybrid memory before every prompt row."""

    def __init__(self, path, label_ids, threads):
        import llama_cpp as L
        import numpy as np

        global _QUIET
        self.L, self.model, self.context, self.batch = L, None, None, None
        self.threads = threads
        self.labels = np.asarray(label_ids)
        if _QUIET is None:
            _QUIET = L.llama_log_callback(lambda level, message, data: None)
            L.llama_log_set(_QUIET, ctypes.c_void_p(0))
        try:
            L.llama_backend_init()
            mp = L.llama_model_default_params()
            mp.n_gpu_layers = 0
            self.model = L.llama_model_load_from_file(os.fsencode(path), mp)
            if not self.model:
                raise RuntimeError("Decider GGUF CPU model could not be loaded")
            cp = L.llama_context_default_params()
            cp.n_ctx = cp.n_batch = cp.n_ubatch = CONTEXT_TOKENS
            cp.n_seq_max = 1
            cp.kv_unified = True
            cp.n_threads = cp.n_threads_batch = threads
            self.context = L.llama_init_from_model(self.model, cp)
            if not self.context:
                raise RuntimeError("Decider GGUF CPU context could not be created")
            self.n_vocab = L.llama_vocab_n_tokens(L.llama_model_get_vocab(self.model))
            if (
                len(self.labels) != 255
                or min(label_ids) < 0
                or max(label_ids) >= self.n_vocab
            ):
                raise ValueError(
                    "Decider tokenizer labels do not match GGUF vocabulary"
                )
            self.batch = L.llama_batch_init(CONTEXT_TOKENS, 0, 1)
            if not self.batch.token:
                raise RuntimeError("Decider GGUF CPU batch allocation failed")
        except BaseException:
            self.close()
            raise

    def read(self, ids):
        import numpy as np
        import torch

        if not self.context:
            raise RuntimeError("Decider GGUF model is closed")
        if not 1 <= len(ids) <= MAX_SEQUENCE_TOKENS or any(
            type(t) is not int or not 0 <= t < self.n_vocab for t in ids
        ):
            raise ValueError("invalid Decider GGUF prompt tokens")
        L, batch = self.L, self.batch
        L.llama_memory_clear(L.llama_get_memory(self.context), True)
        for i, token in enumerate(ids):
            batch.token[i], batch.pos[i] = token, i
            batch.n_seq_id[i], batch.seq_id[i][0] = 1, 0
            batch.logits[i] = i == len(ids) - 1
        batch.n_tokens = len(ids)
        rc = L.llama_decode(self.context, batch)
        if rc:
            raise RuntimeError(f"Decider GGUF llama_decode returned {rc}")
        pointer = L.llama_get_logits_ith(self.context, len(ids) - 1)
        if not pointer:
            raise RuntimeError("Decider GGUF answer logits are missing")
        logits = np.ctypeslib.as_array(pointer, shape=(self.n_vocab,))[
            self.labels
        ].copy()
        if not np.isfinite(logits).all():
            raise RuntimeError("Decider GGUF returned nonfinite logits")
        return torch.from_numpy(logits)

    def close(self):
        # Idempotent even when construction failed between native allocations.
        if self.batch is not None:
            self.L.llama_batch_free(self.batch)
            self.batch = None
        if self.context:
            self.L.llama_free(self.context)
            self.context = None
        if self.model:
            self.L.llama_model_free(self.model)
            self.model = None


class DeciderGGUFAgent:
    device = "cpu"

    def __init__(self, tokenizer, labels, readout):
        from .decision import DecisionCompiler

        self.readout = readout
        self.lock = threading.RLock()
        self.compiler = DecisionCompiler.__new__(DecisionCompiler)
        self.compiler.manifest = {
            "max_length": MAX_SEQUENCE_TOKENS,
            "max_options": 255,
            "temperature": 1.099,
            "temperatures": {"choice": 1.110, "noul": 1.560, "score": 1.287},
            "isolated_levels": True,
            "label_token_ids": [token for _, token in labels],
        }
        self.compiler.tokenizer = tokenizer
        self.compiler.model_names, self.compiler.model_id = [MODEL_ID], MODEL_ID
        self.compiler.protocol = "decider_slot_v1"
        self.compiler.max_length = MAX_SEQUENCE_TOKENS
        self.compiler.labels = [name for name, _ in labels]
        self.compiler.special = []

    def system_one(self, state, questions):
        with self.lock:
            return self._system_one(state, questions)

    def _system_one(self, state, questions):
        import torch

        from .decision import _answer
        from .request_models import SystemOneRequest

        validate_template()
        if self.readout is None:
            raise RuntimeError("Decider GGUF model is closed")
        if not isinstance(state, (str, dict, list)):
            raise ValueError("decision state must be text or JSON")
        text = (
            state
            if isinstance(state, str)
            else json.dumps(state, ensure_ascii=False, allow_nan=False)
        )
        if len(text) > 50_000:
            raise ValueError("decision state exceeds 50,000 characters")
        payload = SystemOneRequest(
            state=state, questions=_questions(questions), model=MODEL_ID
        )
        # Compile the complete request first: never truncate or execute earlier
        # rows before discovering a later row that exceeds the context limit.
        prepared, state_tokens = self.compiler.compile(payload)
        started = time.perf_counter()
        encoded, answers = {}, {}
        input_tokens = state_tokens
        for identifier, kind, keys, descriptions, isolated, rows in prepared:
            probabilities = []
            temperature = self.compiler.manifest["temperatures"][kind]
            for ids, positions, count in rows:
                if positions != [len(ids) - 1]:
                    raise RuntimeError("unsupported Decider GGUF readout positions")
                key = tuple(ids)
                if key not in encoded:
                    encoded[key] = self.readout.read(ids)
                logits = encoded[key].clone()
                logits[count:] = float("-inf")
                probabilities.append(
                    torch.softmax(logits / temperature, -1)[:count].tolist()
                )
            if isolated:
                mass = [p[1] for p in probabilities]
                total = sum(mass)
                # Match the author's zero-mass result, without inventing a
                # uniform probability distribution after numerical underflow.
                probability = [p / total for p in mass] if total else [0.0] * len(mass)
            else:
                probability = probabilities[0]
            answers[identifier] = _answer(kind, keys, descriptions, probability)
            if isolated and total == 0:
                answers[identifier]["confidence"] = 0.0
            input_tokens += sum(len(row[0]) - state_tokens for row in rows)
        return {
            "model": MODEL_ID,
            "answers": answers,
            "usage": {"input_tokens": input_tokens, "output_tokens": 0},
            "internal_usage": {"compute_tokens": sum(len(ids) for ids in encoded)},
            "metadata": {
                "quantization": "Q4_K_M",
                "logits_dtype": "float32",
                "row_batch_size": 1,
                "max_sequence_tokens": MAX_SEQUENCE_TOKENS,
                "context_tokens": CONTEXT_TOKENS,
                "n_gpu_layers": 0,
                "threads": self.readout.threads,
                "inference_seconds": time.perf_counter() - started,
            },
        }

    def close(self):
        with self.lock:
            if self.readout is not None:
                self.readout.close()
                self.readout = None
            self.compiler = None


def load_agent(checkpoint, model_id=MODEL_ID, threads=None):
    _require_model(model_id)
    validate_template()
    threads = min(4, os.cpu_count() or 1) if threads is None else threads
    if type(threads) is not int or not 1 <= threads <= min(4, os.cpu_count() or 1):
        raise ValueError("Decider GGUF CPU threads must be between 1 and 4")
    verify_runtime()
    checkpoint = Path(checkpoint).resolve()
    verify_source(checkpoint)
    import torch
    from transformers import AutoTokenizer

    from .decision_export import _labels

    torch.set_num_threads(threads)
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint, local_files_only=True, trust_remote_code=False
    )
    labels = _labels(tokenizer)
    readout = NativeReadout(
        checkpoint / GGUF_FILENAME, [token for _, token in labels], threads
    )
    try:
        return DeciderGGUFAgent(tokenizer, labels, readout)
    except BaseException:
        readout.close()
        raise
