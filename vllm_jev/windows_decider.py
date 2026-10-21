"""Pinned Decider text checkpoints, with native BF16 readouts on Windows CPU.

The published source stays read-only. The adapter uses the project compiler,
not downloaded Python, and never materializes full-vocabulary token logits.
"""

import hashlib
import json
import math
import os
import tempfile
import time
from pathlib import Path

from filelock import FileLock
from huggingface_hub import snapshot_download

MODEL_ID = "Mapika/decider-0.8b"
REVISION = "a0a01d6f8135298f400a8c856b355793012ae971"
MODEL_ID_2B = "Mapika/decider-2b"
REVISION_2B = "ff99089b6810691bd79134ec6febb5d9afa6e619"
PROTOCOL = "decider_slot_v1"
MAX_SEQUENCE_TOKENS = 1024
SOURCE_HASHES = {
    "config.json": "3f56b6210db3c52e0bafb50da005eb52b6d340ab4ad980091879b397743675fc",
    "decider_config.json": "a817fa20524b2c062753f59fd5fae02207f1eff4808b729db6e0aa8d1bc3cce2",
    "generation_config.json": "62153eb6c69f2e1f426beaa8002b7186437e949c7588167085df14e10e9c0a73",
    "tokenizer_config.json": "8671bed7c852ce9e661be94f179a7b4ffd091c2a65aea0363e5501c20318ee45",
    "chat_template.jinja": "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80",
    # Expected upstream LFS digests; downloaded bytes are checked before loading.
    "model.safetensors": "6926f82ef7e9ea408ad881555204daa6bb694ca82509c4d1514b7be2e713563d",
    "tokenizer.json": "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
}
SOURCE_FILES = tuple(SOURCE_HASHES)
SOURCE_HASHES_2B = {
    "config.json": "6cb8daca9fb653c61485ff7452fc068bacd5c27cbee659ecd24b47186b0d1b52",
    "decider_config.json": "6e4891f2754a1c18a10f8dadb0c04e439e7f79fab0333d56641491bd4a05e722",
    "generation_config.json": "62153eb6c69f2e1f426beaa8002b7186437e949c7588167085df14e10e9c0a73",
    "tokenizer_config.json": "171ecbe7ddae98d11840698f7df2b8d5b4722139db0f0620d3bbf429bd656250",
    "chat_template.jinja": "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80",
    # Expected upstream LFS digests; downloaded bytes are checked before loading.
    "model.safetensors": "acaef2228b134dcdc20cad4ee79219482c927ec819aa3687b9b8a575c338817f",
    "tokenizer.json": "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
}
SOURCE_FILES_2B = tuple(SOURCE_HASHES_2B)
MODELS = {
    MODEL_ID: {
        "revision": REVISION,
        "hidden_size": 1024,
        "files": SOURCE_HASHES,
        "version": "0.8b-v1",
        "temperature": 1.03,
        "temperatures": {},
    },
    MODEL_ID_2B: {
        "revision": REVISION_2B,
        "hidden_size": 2048,
        "files": SOURCE_HASHES_2B,
        "version": "2b-v11",
        "temperature": 1.145,
        "temperatures": {"choice": 1.164, "noul": 1.624, "score": 1.124},
    },
}


def _specification(model_id):
    try:
        return MODELS[model_id]
    except KeyError:
        raise ValueError(f"unsupported Windows Decider model: {model_id}") from None


def verify_source(checkpoint: Path, model_id: str = MODEL_ID) -> dict:
    """Check every allowed source byte before tokenizer or weight loading."""
    specification = _specification(model_id)
    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_dir():
        raise ValueError("Decider source directory is missing")
    for entry in checkpoint.iterdir():
        if entry.name == ".cache" and entry.is_dir():
            continue
        if entry.name not in specification["files"] or not entry.is_file():
            raise ValueError(f"unexpected Decider source entry: {entry.name}")
    for name, expected in specification["files"].items():
        path = checkpoint / name
        if not path.is_file():
            raise ValueError(f"Decider source file is missing: {name}")
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"Decider source checksum mismatch: {name}")
    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    calibration = json.loads(
        (checkpoint / "decider_config.json").read_text(encoding="utf-8")
    )
    if (
        config.get("model_type") != "qwen3_5_text"
        or config.get("architectures") != ["Qwen3_5ForCausalLM"]
        or config.get("hidden_size") != specification["hidden_size"]
        or config.get("tie_word_embeddings") is not True
        or "text_config" in config
        or config.get("auto_map")
        or calibration.get("layout", "plain") != "plain"
        or calibration.get("neutralize_none") is not False
        or calibration.get("schema_first") is not False
        or calibration.get("max_options") != 255
        or calibration.get("isolated_levels") is not True
        or calibration.get("version") != specification["version"]
        or calibration.get("temperature") != specification["temperature"]
        or calibration.get("temperature_by_type", {}) != specification["temperatures"]
    ):
        raise ValueError("unsupported Decider source identity or protocol")
    return calibration


def identify_source(checkpoint: Path) -> str:
    # The small architecture config selects a release before reading any large
    # weight file; never attempt a full weight hash for each possible model.
    checkpoint = Path(checkpoint).resolve()
    config = checkpoint / "config.json"
    if not config.is_file():
        raise ValueError("Decider source config is missing")
    with config.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    for model_id, specification in MODELS.items():
        if digest == specification["files"]["config.json"]:
            verify_source(checkpoint, model_id)
            return model_id
    raise ValueError("unrecognized pinned Decider source config")


def prepare(model_id: str, workspace: Path) -> Path:
    specification = _specification(model_id)
    output = Path(workspace).resolve() / "checkpoint" / model_id
    output.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output) + ".lock"):
        if output.exists():
            verify_source(output, model_id)
            return output
        with tempfile.TemporaryDirectory(
            prefix=".decider-download-", dir=output.parent
        ) as temporary:
            source = Path(temporary) / "source"
            snapshot_download(
                repo_id=model_id,
                revision=specification["revision"],
                token=False,
                local_dir=source,
                allow_patterns=list(specification["files"]),
            )
            verify_source(source, model_id)
            source.replace(output)
    return output


def _questions(questions):
    """Normalize accepted wire inputs before compiling any model rows."""
    if not isinstance(questions, dict) or not 1 <= len(questions) <= 64:
        raise ValueError("System One requires 1 to 64 questions")
    if (
        len(json.dumps(questions, ensure_ascii=False, allow_nan=False).encode())
        > 2_000_000
    ):
        raise ValueError("decision questions exceed 2 MiB")
    result = {}
    for identifier, source in questions.items():
        if (
            not isinstance(identifier, str)
            or not identifier
            or not isinstance(source, dict)
        ):
            raise ValueError("question IDs and definitions must be nonempty")
        question = dict(source)
        kind, criteria = question.get("type"), question.get("criteria")
        if kind == "choice" and isinstance(criteria, list):
            if any(not isinstance(key, str) or not key for key in criteria):
                raise ValueError("Choice labels must be nonempty strings")
            if len(set(criteria)) != len(criteria):
                raise ValueError("Choice labels must be distinct")
            question["criteria"] = dict.fromkeys(criteria)
        elif kind == "noul" and criteria is not None:
            if not isinstance(criteria, dict) or not set(criteria) <= {"true", "false"}:
                raise ValueError(
                    "Noul criteria must contain optional true/false descriptions"
                )
        elif kind == "score":
            if not isinstance(criteria, list) or not 2 <= len(criteria) <= 10:
                raise ValueError("Score requires 2 to 10 levels")
            if "isolated" in question and type(question["isolated"]) is not bool:
                raise ValueError("Score isolated must be a boolean")
        result[identifier] = question
    return result


class DeciderAgent:
    device = "cpu"

    def __init__(self, model, tokenizer, weight, manifest, model_id=MODEL_ID):
        from .decision import DecisionCompiler

        _specification(model_id)
        self.model_id = model_id
        self.model = model.eval().requires_grad_(False)
        self.weight = weight.detach().contiguous()
        # The shared compiler needs an in-memory manifest and tokenizer, not a
        # rewritten config or exported checkpoint on the user's source path.
        self.compiler = DecisionCompiler.__new__(DecisionCompiler)
        self.compiler.manifest = dict(manifest)
        self.compiler.tokenizer = tokenizer
        self.compiler.model_names = [model_id]
        self.compiler.model_id = model_id
        self.compiler.protocol = PROTOCOL
        self.compiler.max_length = min(MAX_SEQUENCE_TOKENS, manifest["max_length"])
        self.compiler.labels = manifest["labels"]
        self.compiler.special = []
        if not isinstance(tokenizer.pad_token_id, int):
            raise ValueError("Decider tokenizer needs its published padding token")
        for temperature in [
            manifest["temperature"],
            *manifest.get("temperatures", {}).values(),
        ]:
            if (
                isinstance(temperature, bool)
                or not isinstance(temperature, (float, int))
                or not math.isfinite(temperature)
                or temperature <= 0
            ):
                raise ValueError("Decider temperature must be finite and positive")
        self._precision()

    def _precision(self):
        import torch

        if self.model is None:
            raise RuntimeError("Decider CPU model is closed")
        tensors = [*self.model.parameters(), *self.model.buffers(), self.weight]
        if any(t.is_meta or t.device.type != "cpu" for t in tensors):
            raise ValueError("Decider parameters must be materialized on CPU")
        embedding = self.model.get_input_embeddings().weight
        if (
            any(p.dtype != torch.bfloat16 for p in self.model.parameters())
            or embedding.dtype != torch.bfloat16
            or self.weight.dtype != torch.bfloat16
        ):
            raise ValueError("Decider requires BF16 backbone and vocabulary readout")
        if self.weight.shape != (len(self.compiler.labels), embedding.shape[1]):
            raise ValueError("invalid Decider vocabulary readout shape")
        if not torch.isfinite(self.weight).all():
            raise ValueError("nonfinite Decider vocabulary readout")
        return {
            "backbone_dtype": str(embedding.dtype).removeprefix("torch."),
            "head_dtype": str(self.weight.dtype).removeprefix("torch."),
            "readout_dtype": str(self.weight.dtype).removeprefix("torch."),
            "logits_dtype": "float32",
            "row_batch_size": 1,
            "padding_multiple": 64,
            "max_sequence_tokens": self.compiler.max_length,
        }

    def _readout(self, ids):
        import torch
        import torch.nn.functional as F

        with torch.inference_mode():
            width = ((len(ids) + 63) // 64) * 64
            if width > self.compiler.max_length:
                raise ValueError("Decider padded prompt exceeds model context")
            tokens = torch.full(
                (1, width),
                self.compiler.tokenizer.pad_token_id,
                dtype=torch.long,
                device="cpu",
            )
            tokens[0, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention_mask = torch.zeros_like(tokens)
            attention_mask[0, : len(ids)] = 1
            hidden = self.model(
                input_ids=tokens,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            ).last_hidden_state
            if hidden.shape != (1, width, self.weight.shape[1]):
                raise RuntimeError("Decider returned incomplete hidden states")
            if hidden.dtype != self.weight.dtype or hidden.device.type != "cpu":
                raise RuntimeError(
                    "Decider returned an unexpected hidden-state precision"
                )
            # The author projects all 255 BF16 rows before converting to FP32.
            # Narrowing the GEMM or converting its operands changes rounding.
            logits = F.linear(hidden[:, len(ids) - 1], self.weight).float()[0]
            if not torch.isfinite(logits).all():
                raise RuntimeError("Decider returned nonfinite logits")
            return logits

    def system_one(self, state, questions):
        import torch

        from .decision import _answer
        from .request_models import SystemOneRequest

        precision = self._precision()
        if not isinstance(state, (str, dict, list)):
            raise ValueError("decision state must be text or JSON")
        state_text = (
            state
            if isinstance(state, str)
            else json.dumps(state, ensure_ascii=False, allow_nan=False)
        )
        if len(state_text) > 50_000:
            raise ValueError("decision state exceeds 50,000 characters")
        payload = SystemOneRequest(
            state=state, questions=_questions(questions), model=self.model_id
        )
        prepared, state_tokens = self.compiler.compile(payload)
        started = time.perf_counter()
        encoded, answers = {}, {}
        input_tokens = state_tokens
        for identifier, kind, keys, descriptions, isolated, rows in prepared:
            probabilities = []
            temperature = self.compiler.manifest.get("temperatures", {}).get(
                kind, self.compiler.manifest["temperature"]
            )
            for ids, positions, count in rows:
                if positions != [len(ids) - 1] or not ids:
                    raise RuntimeError("unsupported Decider readout positions")
                key = tuple(ids)
                if key not in encoded:
                    encoded[key] = self._readout(ids)
                # Match the full author projection/masked softmax shape.
                logits = encoded[key].clone()
                logits[count:] = float("-inf")
                probabilities.append(
                    torch.softmax(logits / temperature, -1)[:count].tolist()
                )
            if isolated:
                mass = [p[1] for p in probabilities]
                total = sum(mass)
                probability = (
                    [p / total for p in mass] if total else [1 / len(mass)] * len(mass)
                )
            else:
                probability = probabilities[0]
            answers[identifier] = _answer(kind, keys, descriptions, probability)
            input_tokens += sum(len(row[0]) - state_tokens for row in rows)
        return {
            "model": self.model_id,
            "answers": answers,
            "usage": {"input_tokens": input_tokens, "output_tokens": 0},
            "internal_usage": {"compute_tokens": sum(len(ids) for ids in encoded)},
            "metadata": {
                **precision,
                "inference_seconds": time.perf_counter() - started,
            },
        }

    def close(self):
        self.model = None
        self.weight = None
        self.compiler = None


def load_agent(checkpoint: Path, model_id: str = MODEL_ID, threads=None):
    _specification(model_id)
    if threads is None:
        threads = min(4, os.cpu_count() or 1)
    if type(threads) is not int or not 1 <= threads <= (os.cpu_count() or 1):
        raise ValueError("threads must be between 1 and the logical CPU count")
    checkpoint = Path(checkpoint).resolve()
    calibration = verify_source(checkpoint, model_id)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .decision_export import _labels

    torch.set_num_threads(threads)
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint, local_files_only=True, trust_remote_code=False
    )
    labels = _labels(tokenizer)
    full, loading = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map={"": "cpu"},
        local_files_only=True,
        trust_remote_code=False,
        output_loading_info=True,
    )
    if any(
        loading.get(key)
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    ):
        raise ValueError("Decider weights were not loaded completely")
    weight = (
        full.get_output_embeddings()
        .weight[[token for _, token in labels]]
        .detach()
        .clone()
    )
    if weight.shape != (255, _specification(model_id)["hidden_size"]):
        raise ValueError("Decider readout does not match the selected release")
    backbone = full.model
    del full
    backbone.config.use_cache = False
    manifest = {
        "prompt_protocol": PROTOCOL,
        "max_length": MAX_SEQUENCE_TOKENS,
        "max_options": calibration["max_options"],
        "temperature": calibration["temperature"],
        "temperatures": calibration.get("temperature_by_type", {}),
        "isolated_levels": calibration["isolated_levels"],
        "labels": [name for name, _ in labels],
        "label_token_ids": [token for _, token in labels],
    }
    return DeciderAgent(backbone, tokenizer, weight, manifest, model_id)
