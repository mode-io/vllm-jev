"""Pinned This-That releases, packed multi-question inference on Windows CPU.

The author CPU backend uses FP32, a complete 255-label projection, NumPy FP32
softmax and a final Python-float normalization. No downloaded Python executes.
"""

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

from filelock import FileLock
from huggingface_hub import snapshot_download

MODEL_ID = "flock-io/this-that-model-1.2"
REVISION = "c4d1c30b8d512d278726de439b8cc81fccc70c8f"
PROTOCOL = "thisthat_slot_v1"
MAX_SEQUENCE_TOKENS = 1024
SOURCE_HASHES = {
    "config.json": "6cb8daca9fb653c61485ff7452fc068bacd5c27cbee659ecd24b47186b0d1b52",
    "generation_config.json": "62153eb6c69f2e1f426beaa8002b7186437e949c7588167085df14e10e9c0a73",
    "tokenizer_config.json": "171ecbe7ddae98d11840698f7df2b8d5b4722139db0f0620d3bbf429bd656250",
    "chat_template.jinja": "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80",
    "tokenizer.json": "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
    "model.safetensors": "585295458a6b6e01162d2a72d33ba630c286b51c844869afc35272700c23f199",
}
SOURCE_FILES = tuple(SOURCE_HASHES)
MODEL_ID_1_1 = "flock-io/this-that-model-1.1"
MODEL_ID_1_0 = "flock-io/this-that-model-1.0"
MODELS = {
    MODEL_ID: {"revision": REVISION, "source_hashes": SOURCE_HASHES},
    MODEL_ID_1_1: {
        "revision": "fc083c82bd5872abfeaf0d44cf7482324efce87d",
        "source_hashes": {
            **SOURCE_HASHES,
            "model.safetensors": "63e1179554bc56654eb7aa1ccc5dc1edbaf99a2b32e1cd19f77fcf65a98be04d",
        },
    },
    MODEL_ID_1_0: {
        "revision": "3d927195c4f9845efe66c5715883a7a0f42b1239",
        "source_hashes": {
            **SOURCE_HASHES,
            "model.safetensors": "11bab4bbbce0214dcb4d70a88e74b3e4bde6fe8a95f239e3d0c355946e0000e0",
            "tokenizer_config.json": "353c083805d863719e54973335a38e252a92e1ac64d32b83bb0b333cc1644ef2",
        },
    },
}


def _require_model(model_id):
    if not isinstance(model_id, str) or model_id not in MODELS:
        raise ValueError(f"unsupported Windows This-That model: {model_id}")
    return MODELS[model_id]


def _read_source(checkpoint: Path) -> tuple[dict, dict]:
    """Read every allowlisted file once, including the release-specific weights."""
    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_dir():
        raise ValueError("This-That source directory is missing")
    for entry in checkpoint.iterdir():
        if entry.name == ".cache" and entry.is_dir():
            continue
        if entry.name not in SOURCE_FILES or not entry.is_file():
            raise ValueError(f"unexpected This-That source entry: {entry.name}")
    hashes = {}
    for name in SOURCE_FILES:
        path = checkpoint / name
        if not path.is_file():
            raise ValueError(f"This-That source file is missing: {name}")
        with path.open("rb") as stream:
            hashes[name] = hashlib.file_digest(stream, "sha256").hexdigest()
    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    if (
        config.get("model_type") != "qwen3_5_text"
        or config.get("architectures") != ["Qwen3_5ForCausalLM"]
        or config.get("hidden_size") != 2048
        or config.get("num_hidden_layers") != 24
        or config.get("tie_word_embeddings") is not True
        or config.get("auto_map")
        or "text_config" in config
    ):
        raise ValueError("unsupported This-That source identity")
    return config, hashes


def _verify_hashes(actual: dict, expected: dict) -> None:
    if set(actual) != set(expected):
        raise ValueError("incomplete This-That release identity")
    for name, digest in expected.items():
        if actual[name] != digest:
            raise ValueError(f"This-That source checksum mismatch: {name}")


def verify_source(checkpoint: Path, model_id: str = MODEL_ID) -> dict:
    """Verify the entire selected release before importing or loading a backend."""
    profile = _require_model(model_id)
    config, hashes = _read_source(checkpoint)
    _verify_hashes(hashes, profile["source_hashes"])
    return config


def identify_source(checkpoint: Path) -> str:
    # Shared config/tokenizer bytes cannot distinguish releases or Decider 2B.
    # Hash the large weight file only once, then check all six files for its ID.
    _, hashes = _read_source(checkpoint)
    matches = [
        model_id
        for model_id, profile in MODELS.items()
        if profile["source_hashes"]["model.safetensors"] == hashes["model.safetensors"]
    ]
    if len(matches) != 1:
        raise ValueError("This-That source checksum mismatch: model.safetensors")
    model_id = matches[0]
    _verify_hashes(hashes, MODELS[model_id]["source_hashes"])
    return model_id


def prepare(model_id: str, workspace: Path) -> Path:
    profile = _require_model(model_id)
    output = Path(workspace).resolve() / "checkpoint" / model_id
    output.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output) + ".lock"):
        if output.exists():
            verify_source(output, model_id)
            return output
        with tempfile.TemporaryDirectory(
            prefix=".thisthat-download-", dir=output.parent
        ) as temporary:
            source = Path(temporary) / "source"
            snapshot_download(
                repo_id=model_id,
                revision=profile["revision"],
                token=False,
                local_dir=source,
                allow_patterns=list(profile["source_hashes"]),
            )
            verify_source(source, model_id)
            source.replace(output)
    return output


def _questions(questions):
    if not isinstance(questions, dict) or not 1 <= len(questions) <= 64:
        raise ValueError("System One requires 1 to 64 questions")
    if (
        len(json.dumps(questions, ensure_ascii=False, allow_nan=False).encode())
        > 2_000_000
    ):
        raise ValueError("decision questions exceed 2 MiB")
    for identifier, question in questions.items():
        if (
            not isinstance(identifier, str)
            or not identifier
            or not isinstance(question, dict)
        ):
            raise ValueError("question IDs and definitions must be nonempty")
        if (
            not isinstance(question.get("instructions"), str)
            or not question["instructions"].strip()
        ):
            raise ValueError("This-That instructions must be nonempty text")
        kind, criteria = question.get("type"), question.get("criteria")
        if kind == "choice":
            labels = list(criteria) if isinstance(criteria, (dict, list)) else []
            if (
                not 2 <= len(labels) <= 255
                or any(not isinstance(k, str) or not k for k in labels)
                or len(set(labels)) != len(labels)
            ):
                raise ValueError(
                    "Choice requires 2 to 255 distinct nonempty string labels"
                )
        elif kind == "score":
            if not isinstance(criteria, list) or not 2 <= len(criteria) <= 255:
                raise ValueError("Score requires 2 to 255 levels")
            if question.get("isolated", False) is not False:
                raise ValueError("This-That Score does not use isolated level rows")
        elif kind == "noul":
            if criteria is not None and (
                not isinstance(criteria, dict) or not set(criteria) <= {"false", "true"}
            ):
                raise ValueError(
                    "Noul criteria must contain optional true/false descriptions"
                )
        else:
            raise ValueError("unsupported This-That question type")
    return questions


class ThisThatAgent:
    device = "cpu"

    def __init__(
        self, model, tokenizer, weight, labels, token_ids, model_id: str = MODEL_ID
    ):
        from .decision import DecisionCompiler

        profile = _require_model(model_id)
        self.model_id = model_id
        self.revision = profile["revision"]
        self.model = model.eval().requires_grad_(False)
        self.weight = weight.detach().contiguous()
        self.compiler = DecisionCompiler.__new__(DecisionCompiler)
        self.compiler.manifest = {
            "prompt_protocol": PROTOCOL,
            "max_length": MAX_SEQUENCE_TOKENS,
            "max_options": 255,
            "temperature": 1.0,
            "isolated_levels": False,
            "labels": list(labels),
            "label_token_ids": list(token_ids),
        }
        self.compiler.tokenizer = tokenizer
        self.compiler.model_names = [model_id]
        self.compiler.model_id = model_id
        self.compiler.protocol = PROTOCOL
        self.compiler.max_length = MAX_SEQUENCE_TOKENS
        self.compiler.labels = list(labels)
        self.compiler.special = []
        self.pad_id = tokenizer.pad_token_id
        if self.pad_id is None:
            self.pad_id = tokenizer.eos_token_id or 0
        if type(self.pad_id) is not int or self.pad_id < 0:
            raise ValueError("This-That padding token is invalid")
        if len(labels) != 255 or len(set(token_ids)) != 255:
            raise ValueError("This-That requires 255 distinct single-token labels")
        self._precision()

    def _precision(self):
        import torch

        if self.model is None:
            raise RuntimeError("This-That CPU model is closed")
        tensors = [*self.model.parameters(), *self.model.buffers(), self.weight]
        if any(t.is_meta or t.device.type != "cpu" for t in tensors):
            raise ValueError("This-That parameters must be materialized on CPU")
        embedding = self.model.get_input_embeddings().weight
        if (
            any(p.dtype != torch.float32 for p in self.model.parameters())
            or embedding.dtype != torch.float32
            or self.weight.dtype != torch.float32
        ):
            raise ValueError("This-That requires FP32 backbone and vocabulary readout")
        if (
            self.weight.shape != (255, embedding.shape[1])
            or not torch.isfinite(self.weight).all()
        ):
            raise ValueError("invalid This-That vocabulary readout")
        return {
            "backbone_dtype": str(embedding.dtype).removeprefix("torch."),
            "head_dtype": str(self.weight.dtype).removeprefix("torch."),
            "readout_dtype": str(self.weight.dtype).removeprefix("torch."),
            "logits_dtype": "float32",
            "softmax_dtype": "float32",
            "softmax_backend": "numpy",
            "row_batch_size": 1,
            "padding_multiple": 64,
            "max_sequence_tokens": MAX_SEQUENCE_TOKENS,
            "packed_questions": True,
            "temperature": 1.0,
        }

    def _probabilities(self, ids, positions, counts):
        import numpy as np
        import torch
        import torch.nn.functional as F

        width = (len(ids) + 63) // 64 * 64
        if not ids or width > MAX_SEQUENCE_TOKENS:
            raise ValueError("This-That complete padded prompt exceeds model context")
        if (
            not positions
            or len(positions) != len(counts)
            or any(not 0 <= p < len(ids) for p in positions)
        ):
            raise ValueError("invalid This-That readout positions")
        with torch.inference_mode():
            tokens = torch.full((1, width), self.pad_id, dtype=torch.long)
            tokens[0, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            mask = torch.zeros_like(tokens)
            mask[0, : len(ids)] = 1
            hidden = self.model(
                input_ids=tokens, attention_mask=mask, use_cache=False, return_dict=True
            ).last_hidden_state
            if (
                hidden.shape != (1, width, self.weight.shape[1])
                or hidden.dtype != torch.float32
                or hidden.device.type != "cpu"
            ):
                raise RuntimeError(
                    "This-That returned incomplete or non-FP32 hidden states"
                )
            # Keep the author's N-question by 255-label matrix shape. Splitting
            # it into narrow projections changes reduction order and rounding.
            logits = F.linear(hidden[0, positions], self.weight).float().cpu().numpy()
        if not np.isfinite(logits).all():
            raise RuntimeError("This-That returned nonfinite logits")
        past = np.arange(255)[None, :] >= np.asarray(counts)[:, None]
        z = np.where(past, -np.inf, logits).astype(np.float32) / 1.0
        z = z - z.max(axis=-1, keepdims=True)
        exponential = np.exp(z)
        probabilities = exponential / exponential.sum(axis=-1, keepdims=True)
        result = []
        for values, count in zip(probabilities, counts):
            row = [float(x) for x in values[:count]]
            total = sum(row) or 1.0
            result.append([x / total for x in row])
        return result

    def system_one(self, state, questions):
        from .decision import _answer
        from .request_models import SystemOneRequest

        precision = self._precision()
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
            state=state, questions=_questions(questions), model=self.model_id
        )
        prepared, _ = self.compiler.compile(payload)
        ids, _, _ = prepared[0][-1][0]
        positions, counts = [], []
        for _, _, _, _, isolated, rows in prepared:
            if isolated or len(rows) != 1 or rows[0][0] != ids or len(rows[0][1]) != 1:
                raise RuntimeError(
                    "This-That requires one packed sequence for all questions"
                )
            positions.extend(rows[0][1])
            counts.append(rows[0][2])
        started = time.perf_counter()
        probabilities = self._probabilities(ids, positions, counts)
        answers = {
            identifier: _answer(kind, keys, descriptions, row)
            for (identifier, kind, keys, descriptions, _, _), row in zip(
                prepared, probabilities
            )
        }
        return {
            "model": self.model_id,
            "answers": answers,
            "usage": {"input_tokens": len(ids), "output_tokens": 0},
            "internal_usage": {"compute_tokens": len(ids)},
            "metadata": {
                **precision,
                "model": self.model_id,
                "revision": self.revision,
                "inference_seconds": time.perf_counter() - started,
            },
        }

    def close(self):
        self.model = self.weight = self.compiler = None


def load_agent(checkpoint: Path, model_id: str = MODEL_ID, threads=None):
    _require_model(model_id)
    if threads is None:
        threads = min(4, os.cpu_count() or 1)
    if type(threads) is not int or not 1 <= threads <= (os.cpu_count() or 1):
        raise ValueError("threads must be between 1 and the logical CPU count")
    checkpoint = Path(checkpoint).resolve()
    verify_source(checkpoint, model_id)
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
        dtype=torch.float32,
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
        raise ValueError("This-That weights were not loaded completely")
    weight = (
        full.get_output_embeddings()
        .weight[[token for _, token in labels]]
        .detach()
        .clone()
    )
    if weight.shape != (255, 2048):
        raise ValueError("This-That readout does not match the selected release")
    backbone = full.model
    del full
    backbone.config.use_cache = False
    return ThisThatAgent(
        backbone,
        tokenizer,
        weight,
        [name for name, _ in labels],
        [token for _, token in labels],
        model_id=model_id,
    )
