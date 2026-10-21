"""Pinned Mica Q5_K_M decisions with the b11010 Windows CPU runtime.

The upstream prompt, codebook, native prefill and calibrated softmax are
vendored with attribution. The downloaded checkpoint contains data and a
hash-verified official CPU runtime; no downloaded Python is executed.
"""

import hashlib
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

MODEL_ID = "sky7350/Mica-v0.1-4B"
REVISION = "9b6428b272f4708ef54a4702b52bb453a4059363"
SDK_COMMIT = "f6bccd76e9e42b9193d1cdf3549b971e7b200d59"
PROTOCOL = "mica_gguf_v1"
BACKEND = "mica-llama-cpp-q5-b11010"
GGUF_FILENAME = "mica-v0.1-4b-Q5_K_M.gguf"
RUNTIME_REVISION = "b11010"
MAX_SEQUENCE_TOKENS = 1024
RUNTIME_URL = (
    "https://github.com/ggml-org/llama.cpp/releases/download/b11010/"
    "llama-b11010-bin-win-cpu-x64.zip"
)
RUNTIME_ZIP_SHA256 = "3011251607251eca68ae67e35512364d9943d862ff598599ff9a4888c1753008"
HF_SOURCE_HASHES = {
    "calibration.json": "90076b77fd398dc42123dd970933d5af17ebca8cc9d8de6e2225bc2d0c76fbca",
    "chat_template.jinja": "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715",
    "config.json": "a9e9564540d0ab364e5aa08eb2a63d3f9ca3ba3216734c4c14f0ea629328378b",
    "tokenizer.json": "87a7830d63fcf43bf241c3c5242e96e62dd3fdc29224ca26fed8ea333db72de4",
    "tokenizer_config.json": "83febe739ecefd99f984f546dfdc10149cedf293b9ee0291b99894341188b613",
    "LICENSE": "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30",
    "NOTICE": "ce0b725c0171063827818bb5c914f7884e20eb2f105c7cd06bcc1e567edca30f",
    GGUF_FILENAME: "7fbd1be2293ba158157bbbdcdc4ea57569bd9ed53901ed1766f81264d9da8292",
}
RUNTIME_HASHES = {
    "ggml-base.dll": "cd2603e0ba72b57f830455293185748a998be147b380f16965a973f484ada9e5",
    "ggml-cpu-alderlake.dll": "2bee7689149fcf3f0112002a85eca55c279c5c088a584dd83cd2433d9312bc15",
    "ggml-cpu-cannonlake.dll": "5124c61d027dfcb82a0ed988dfbf4c0ea7ec95f692d4574a824fb77c2e18eeec",
    "ggml-cpu-cascadelake.dll": "c6a79ad792b7e18002889c8b4d0c8fde7cd7b9f24344f08000a381ef9af23dcf",
    "ggml-cpu-cooperlake.dll": "9c708128b5a58005144d74f7224d5d1e56b4719151b63d70f9d88943e3c07bd1",
    "ggml-cpu-haswell.dll": "355dbeacd74b59a49129db49f9abfbf611acd7a9db827c8b4d7922b75070a34e",
    "ggml-cpu-icelake.dll": "bb96cf1a3324a87cbe43c44f3c30e2bacc4e0e1f0f11f4745eee5847353bcef1",
    "ggml-cpu-ivybridge.dll": "dae774e6f52b977e357bc3d97b93df05a00d0b397d71755962e237498beb3a7b",
    "ggml-cpu-piledriver.dll": "9ef3d5ee166de057bcc1b9276f42efeb54c0b58e6772f80c8243e750d7109c35",
    "ggml-cpu-sandybridge.dll": "9fdedb44733dbbec53c7d1087ccc7270a0e248bd89ae5e71c14f78206ef256a0",
    "ggml-cpu-sapphirerapids.dll": "f2246c1c47a2a8a3749cd74d71e5dd2a78577a5c68cbb0403175a7b08495cbda",
    "ggml-cpu-skylakex.dll": "4d57d36144f42d7c29b2a67d88ad7251c9dd1776bb6ef954bcf142d5b3c08f73",
    "ggml-cpu-sse42.dll": "597e4a12ef2b28ae1f5d5a0af1faa5dcc53212bdc4b15f966abac71f47239306",
    "ggml-cpu-x64.dll": "5ed04a90b6193377ba14e8c1dca88dfeb0544dae196d610d7a92b268e20543d8",
    "ggml-cpu-zen4.dll": "e051b474a192064f03b831cb223087f7df0daa25861658790f2ae88dd243805f",
    "ggml.dll": "52c375a006ca3aa802f47c50b883b29c9ba41c7c6dfc2a96a718beb4ed944b40",
    "libomp.dll": "a12116ba72d1d6820407cf30be23da04ce79d6bb8a71a5ee71759c5a1faa6f1c",
    "LICENSE-LLVM-OpenMP": "fdad1758a9e1f9d5a81e18879b3406772115edc92c24bfa36b70c654f325e8e4",
    "llama.dll": "6bb4edebf33785cf808dd7fc3964096317df5df63cc44c8918967903edabe633",
}
SOURCE_HASHES = {
    **HF_SOURCE_HASHES,
    **{f"runtime/{name}": digest for name, digest in RUNTIME_HASHES.items()},
}
SOURCE_FILES = tuple(SOURCE_HASHES)


def _require_model(model_id):
    if model_id != MODEL_ID:
        raise ValueError(f"unsupported Windows Mica model: {model_id}")


def validate_template(template=None):
    """The first GGUF profile preserves the exact published native prompt."""
    if template is None:
        from .decision_template import current_template

        template = current_template()
    if template.config not in ({}, {"layout": "native"}):
        raise ValueError(
            "Windows Mica supports only the default/native decision template"
        )


def _sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_source(checkpoint: Path, model_id: str = MODEL_ID) -> dict:
    """Verify model, tokenizer, calibration and every loadable CPU library."""
    _require_model(model_id)
    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_dir():
        raise ValueError("Mica source directory is missing")
    for entry in checkpoint.iterdir():
        if entry.name == ".cache" and entry.is_dir() and not entry.is_symlink():
            continue
        if entry.name == "runtime" and entry.is_dir() and not entry.is_symlink():
            continue
        if (
            entry.name not in HF_SOURCE_HASHES
            or not entry.is_file()
            or entry.is_symlink()
        ):
            raise ValueError(f"unexpected Mica source entry: {entry.name}")
    runtime = checkpoint / "runtime"
    if not runtime.is_dir():
        raise ValueError("Mica CPU runtime directory is missing")
    for entry in runtime.iterdir():
        if (
            entry.name not in RUNTIME_HASHES
            or not entry.is_file()
            or entry.is_symlink()
        ):
            raise ValueError(f"unexpected Mica CPU runtime entry: {entry.name}")
    for name, expected in SOURCE_HASHES.items():
        path = checkpoint / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Mica source file is missing or linked: {name}")
        if _sha256(path) != expected:
            raise ValueError(f"Mica source checksum mismatch: {name}")
    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    if (
        config.get("model_type") != "qwen3_5_text"
        or config.get("architectures") != ["Qwen3_5ForCausalLM"]
        or config.get("hidden_size") != 2560
        or config.get("num_hidden_layers") != 32
        or config.get("auto_map")
    ):
        raise ValueError("unsupported Mica source identity")
    return config


def identify_source(checkpoint: Path) -> str:
    verify_source(checkpoint)
    return MODEL_ID


def _install_runtime(temporary: Path, source: Path):
    archive = temporary / "runtime.zip"
    with (
        urllib.request.urlopen(RUNTIME_URL, timeout=60) as response,
        archive.open("wb") as output,
    ):
        shutil.copyfileobj(response, output)
    if _sha256(archive) != RUNTIME_ZIP_SHA256:
        raise ValueError("Mica b11010 CPU archive checksum mismatch")
    runtime = source / "runtime"
    runtime.mkdir()
    with zipfile.ZipFile(archive) as bundle:
        for name, expected in RUNTIME_HASHES.items():
            # Copy only fixed basenames, never archive-supplied paths. The CPU
            # release also contains RPC and CLI libraries that are not needed.
            matches = [
                item for item in bundle.infolist() if Path(item.filename).name == name
            ]
            if len(matches) != 1 or matches[0].is_dir():
                raise ValueError(f"Mica CPU archive requires exactly one {name}")
            target = runtime / name
            with bundle.open(matches[0]) as incoming, target.open("wb") as outgoing:
                shutil.copyfileobj(incoming, outgoing)
            if _sha256(target) != expected:
                raise ValueError(f"Mica CPU library checksum mismatch: {name}")


def prepare(model_id: str, workspace: Path) -> Path:
    _require_model(model_id)
    validate_template()
    output = Path(workspace).resolve() / "checkpoint" / model_id
    output.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output) + ".lock"):
        if output.exists():
            verify_source(output, model_id)
            return output
        with tempfile.TemporaryDirectory(
            prefix=".mica-download-", dir=output.parent
        ) as temporary:
            temporary = Path(temporary)
            source = temporary / "source"
            snapshot_download(
                repo_id=model_id,
                revision=REVISION,
                token=False,
                local_dir=source,
                allow_patterns=list(HF_SOURCE_HASHES),
            )
            _install_runtime(temporary, source)
            verify_source(source, model_id)
            source.replace(output)
    return output


class MicaTokenizer:
    """The fixed tokenizer and two-message chat template, without PyTorch.

    Transformers 5.x can import Torch while resolving AutoTokenizer, even when
    USE_TORCH=0. Torch's Intel OpenMP conflicts with this llama.cpp release's
    LLVM OpenMP. Use the same Rust tokenizer artifact and sandboxed Jinja
    settings directly, limited to the exact calls made by Mica's native prompt.
    """

    runtime = "tokenizers-jinja2"

    def __init__(self, checkpoint: Path):
        from importlib.metadata import version

        from jinja2 import TemplateError
        from jinja2.sandbox import ImmutableSandboxedEnvironment
        from tokenizers import Tokenizer

        checkpoint = Path(checkpoint)
        for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
            if _sha256(checkpoint / name) != HF_SOURCE_HASHES[name]:
                raise ValueError(f"Mica tokenizer checksum mismatch: {name}")
        self._tokenizer = Tokenizer.from_file(str(checkpoint / "tokenizer.json"))
        self._tokenizer.no_truncation()
        self._tokenizer.no_padding()
        self.tokenizers_version = version("tokenizers")
        self.jinja2_version = version("jinja2")
        self.chat_template = (checkpoint / "chat_template.jinja").read_text(
            encoding="utf-8"
        )
        environment = ImmutableSandboxedEnvironment(
            trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"]
        )

        def raise_exception(message):
            raise TemplateError(message)

        # Match Transformers' non-HTML-escaping JSON filter. The fixed two
        # text-message path does not enter the template's tool/media branches.
        def tojson(
            value, ensure_ascii=False, indent=None, separators=None, sort_keys=False
        ):
            return json.dumps(
                value,
                ensure_ascii=ensure_ascii,
                indent=indent,
                separators=separators,
                sort_keys=sort_keys,
            )

        environment.globals["raise_exception"] = raise_exception
        environment.filters["tojson"] = tojson
        self._template = environment.from_string(self.chat_template)

    def encode(self, text, *, add_special_tokens=False):
        if not isinstance(text, str) or add_special_tokens is not False:
            raise ValueError(
                "Mica tokenizer supports text without added special tokens"
            )
        return self._tokenizer.encode(text, add_special_tokens=False).ids

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    ):
        if (
            tokenize is not False
            or add_generation_prompt is not True
            or enable_thinking is not False
            or not isinstance(messages, list)
            or len(messages) != 2
            or [
                message.get("role") for message in messages if isinstance(message, dict)
            ]
            != ["system", "user"]
            or any(
                set(message) != {"role", "content"}
                or not isinstance(message["content"], str)
                for message in messages
            )
        ):
            raise ValueError("Mica supports only its native system/user text prompt")
        return self._template.render(
            messages=messages,
            tools=None,
            documents=None,
            add_generation_prompt=True,
            enable_thinking=False,
        )


def _rows(state, questions):
    """Strict System One subset of the author's typesafe_server conversion."""
    if not isinstance(state, (str, dict, list)):
        raise ValueError("decision state must be text or JSON")
    try:
        text = (
            state
            if isinstance(state, str)
            else json.dumps(state, ensure_ascii=False, indent=1, allow_nan=False)
        )
        question_bytes = json.dumps(
            questions, ensure_ascii=False, allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as error:
        raise ValueError("decision data must be finite JSON") from error
    if len(text) > 50_000:
        raise ValueError("decision state exceeds 50,000 characters")
    if not isinstance(questions, dict) or not 1 <= len(questions) <= 64:
        raise ValueError("System One requires 1 to 64 questions")
    if len(question_bytes) > 2_000_000:
        raise ValueError("decision questions exceed 2 MiB")
    rows, candidate_count = [], 0
    for identifier, question in questions.items():
        if (
            not isinstance(identifier, str)
            or not identifier
            or not isinstance(question, dict)
        ):
            raise ValueError("question IDs and definitions must be nonempty")
        instructions = question.get("instructions")
        if not isinstance(instructions, str) or not instructions.strip():
            raise ValueError("Mica instructions must be nonempty text")
        kind, criteria = question.get("type"), question.get("criteria")
        if kind == "choice":
            if (
                not isinstance(criteria, dict)
                or not 2 <= len(criteria) <= 255
                or any(not isinstance(key, str) or not key for key in criteria)
                or any(not isinstance(value, str) for value in criteria.values())
            ):
                raise ValueError(
                    "Mica Choice requires 2 to 255 string labels and descriptions"
                )
            candidates = [{"id": key, "text": value} for key, value in criteria.items()]
        elif kind == "score":
            if (
                not isinstance(criteria, list)
                or not 2 <= len(criteria) <= 10
                or any(not isinstance(value, str) for value in criteria)
            ):
                raise ValueError("Mica Score requires 2 to 10 string levels")
            if question.get("isolated", False) is not False:
                raise ValueError("Mica Score does not use isolated level rows")
            candidates = [
                {"id": str(i), "text": value} for i, value in enumerate(criteria)
            ]
        elif kind == "noul":
            if criteria is not None and (
                not isinstance(criteria, dict)
                or not set(criteria) <= {"false", "true"}
                or any(
                    value is not None and not isinstance(value, str)
                    for value in criteria.values()
                )
            ):
                raise ValueError("Mica Noul criteria require optional false/true text")
            criteria = criteria or {}
            candidates = [
                {"id": key, "text": criteria.get(key) or key}
                for key in ("false", "true")
            ]
        else:
            raise ValueError("unsupported Mica question type")
        candidate_count += len(candidates)
        rows.append(
            {
                "id": identifier,
                "kind": kind,
                "state": text,
                "question": instructions,
                "candidates": candidates,
            }
        )
    if candidate_count > 256:
        raise ValueError("System One exceeds 256 candidates")
    return rows


def _answer(row, probabilities):
    """Project wire semantics; raw calibrated author probabilities stay intact."""
    n = len(row["candidates"])
    if (
        len(probabilities) != n
        or any(
            not isinstance(p, (int, float)) or not math.isfinite(p) or p < 0 or p > 1
            for p in probabilities
        )
        or not math.isclose(sum(probabilities), 1.0, rel_tol=1e-7, abs_tol=1e-7)
    ):
        raise RuntimeError("Mica returned an invalid probability distribution")
    kind = row["kind"]
    if kind == "noul":
        return {"type": kind, "noul": probabilities[1]}
    keys = [candidate["id"] for candidate in row["candidates"]]
    selected = max(range(n), key=probabilities.__getitem__)
    result = {"type": kind, "probabilities": dict(zip(keys, probabilities))}
    if kind == "choice":
        result.update(
            choice=keys[selected],
            confidence=max(0.0, (n * max(probabilities) - 1) / (n - 1)),
        )
    else:
        distance = sum(abs(i - (n - 1) / 2) for i in range(n)) / n
        result.update(
            score=sum(i * p for i, p in enumerate(probabilities)),
            confidence=max(
                0.0,
                1
                - sum(p * abs(i - selected) for i, p in enumerate(probabilities))
                / distance,
            ),
            legend={
                candidate["id"]: candidate["text"] for candidate in row["candidates"]
            },
        )
    return result


class MicaAgent:
    device = "cpu"

    def __init__(self, judge, model_id: str = MODEL_ID):
        _require_model(model_id)
        self.model_id, self.revision = model_id, REVISION
        self.judge = judge
        self._lock = threading.RLock()
        self._runtime_metadata()

    def _runtime_metadata(self):
        judge = self.judge
        if judge is None:
            raise RuntimeError("Mica CPU model is closed")
        runtime = dict(judge.runtime_configuration)
        if (
            type(runtime.get("threads")) is not int
            or not 1 <= runtime["threads"] <= 4
            or runtime.get("batch_threads") != runtime["threads"]
            or runtime.get("n_ubatch") != 512
            or runtime.get("flash_attn") != -1
            or runtime.get("n_gpu_layers") != 0
            or judge.n_ctx_seq != MAX_SEQUENCE_TOKENS
            or judge.max_length != MAX_SEQUENCE_TOKENS
            or judge.n_seq_max != 1
        ):
            raise ValueError("Mica runtime does not match the fixed CPU profile")
        return {
            **runtime,
            "backend": BACKEND,
            "device": "cpu",
            "experimental": True,
            "model": self.model_id,
            "revision": self.revision,
            "runtime": "llama.cpp",
            "runtime_revision": RUNTIME_REVISION,
            "weight_format": "GGUF",
            "quantization": "Q5_K_M",
            "logits_dtype": "float32",
            "softmax_dtype": "float64",
            "tokenizer_runtime": getattr(judge.tokenizer, "runtime", "injected"),
            "tokenizers_version": getattr(judge.tokenizer, "tokenizers_version", None),
            "jinja2_version": getattr(judge.tokenizer, "jinja2_version", None),
            "torch_imported": "torch" in sys.modules,
            "transformers_imported": "transformers" in sys.modules,
            "context_tokens": judge.n_ctx_seq,
            "n_seq_max": judge.n_seq_max,
            "n_batch": judge.n_batch,
            "row_batch_size": 1,
            "max_sequence_tokens": MAX_SEQUENCE_TOKENS,
            "prefix": False,
            "session": None,
            "cache_reuse": False,
        }

    def system_one(self, state, questions):
        from ._vendor.mica.native import prompt_for

        with self._lock:
            validate_template()
            metadata = self._runtime_metadata()
            rows = _rows(state, questions)
            token_counts = []
            # Validate the complete request before the first native forward.
            for row in rows:
                tokens = self.judge.tokenize(prompt_for(row, self.judge.tokenizer))
                if not tokens or len(tokens) > MAX_SEQUENCE_TOKENS:
                    raise ValueError(
                        "Mica complete prompt exceeds 1024 tokens; evidence was not truncated"
                    )
                labels = self.judge.label_ids(row)
                if len(labels) != len(row["candidates"]):
                    raise ValueError(
                        "Mica candidate label count does not match its codebook"
                    )
                token_counts.append(len(tokens))
            started = time.perf_counter()
            answers = {}
            for row in rows:
                output = self.judge.judge([row], share_prefix=False, session_id=None)
                if len(output) != 1:
                    raise RuntimeError("Mica returned an incomplete question readout")
                answers[row["id"]] = _answer(row, output[0])
            return {
                "model": self.model_id,
                "answers": answers,
                "usage": {"input_tokens": sum(token_counts), "output_tokens": 0},
                "internal_usage": {"compute_tokens": sum(token_counts)},
                "metadata": {
                    **metadata,
                    "inference_seconds": time.perf_counter() - started,
                },
            }

    def close(self):
        with self._lock:
            if self.judge is not None:
                self.judge.close()
                self.judge = None


def _verify_loaded_runtime(runtime: Path):
    """Reject Windows' reuse of a different, already loaded native library."""
    import ctypes
    from ctypes import wintypes

    if "torch" in sys.modules or "transformers" in sys.modules:
        raise ValueError(
            "Mica requires a fresh tokenizer-only process: PyTorch/Transformers "
            "is already imported and can load Intel OpenMP, which conflicts "
            "with llama.cpp LLVM OpenMP"
        )
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.GetCurrentProcess.argtypes = []
    kernel.GetModuleFileNameW.restype = wintypes.DWORD
    kernel.GetModuleFileNameW.argtypes = [
        wintypes.HMODULE,
        wintypes.LPWSTR,
        wintypes.DWORD,
    ]
    psapi.EnumProcessModules.restype = wintypes.BOOL
    psapi.EnumProcessModules.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.HMODULE),
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    ]
    modules = (wintypes.HMODULE * 2048)()
    needed = wintypes.DWORD()
    if not psapi.EnumProcessModules(
        kernel.GetCurrentProcess(),
        modules,
        ctypes.sizeof(modules),
        ctypes.byref(needed),
    ):
        raise OSError(
            ctypes.get_last_error(), "cannot inspect loaded Mica runtime libraries"
        )
    if needed.value > ctypes.sizeof(modules):
        raise RuntimeError("too many loaded libraries to verify the Mica runtime")
    for module in modules[: needed.value // ctypes.sizeof(wintypes.HMODULE)]:
        buffer = ctypes.create_unicode_buffer(32768)
        size = kernel.GetModuleFileNameW(module, buffer, len(buffer))
        if not size or size >= len(buffer):
            raise OSError(
                ctypes.get_last_error(), "cannot identify a loaded native library"
            )
        path = Path(buffer.value)
        name = path.name.lower()
        if name.startswith("libiomp"):
            raise ValueError(
                "Mica cannot start with Intel OpenMP already loaded "
                f"({path.name}); use a fresh tokenizer-only process"
            )
        if not (name.startswith("ggml") or name in {"llama.dll", "libomp.dll"}):
            continue
        if name not in RUNTIME_HASHES or path.resolve() != (runtime / name).resolve():
            raise ValueError(
                f"a different Mica runtime library is already loaded: {path.name}"
            )
        if _sha256(path) != RUNTIME_HASHES[name]:
            raise ValueError(f"loaded Mica runtime checksum mismatch: {path.name}")


def load_agent(checkpoint: Path, model_id: str = MODEL_ID, threads: int | None = None):
    _require_model(model_id)
    validate_template()
    threads = min(4, os.cpu_count() or 1) if threads is None else threads
    if type(threads) is not int or not 1 <= threads <= 4:
        raise ValueError("Mica CPU threads must be an integer from 1 to 4")
    if sys.platform != "win32":
        raise ValueError("this Mica profile requires native Windows x64")
    checkpoint = Path(checkpoint).resolve()
    verify_source(checkpoint, model_id)
    runtime = checkpoint / "runtime"
    _verify_loaded_runtime(runtime)

    from ._vendor.mica import codebook
    from ._vendor.mica.direct_backend import DirectJudge

    tokenizer = MicaTokenizer(checkpoint)
    codebook.verify(tokenizer)
    # Check again before Windows resolves dependencies by already-loaded name.
    _verify_loaded_runtime(runtime)
    judge = DirectJudge.from_calibration(
        checkpoint / GGUF_FILENAME,
        tokenizer,
        checkpoint / "calibration.json",
        model_identity=model_id,
        runtime_dir=runtime,
        n_ctx=MAX_SEQUENCE_TOKENS,
        n_seq_max=1,
        n_gpu_layers=0,
        n_threads=threads,
        max_length=MAX_SEQUENCE_TOKENS,
        flash_attn=-1,
        n_ubatch=512,
    )
    try:
        _verify_loaded_runtime(runtime)
        # Check every label against both HF and GGUF vocabularies at startup.
        judge.label_ids({"kind": "choice", "candidates": [{}] * 255})
        judge.label_ids({"kind": "noul", "candidates": [{}, {}]})
        return MicaAgent(judge, model_id)
    except BaseException:
        judge.close()
        raise
