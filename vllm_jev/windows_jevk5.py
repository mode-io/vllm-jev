"""Pinned JevK5-2B, using the author's BF16 single-pass readout on CPU.

Source files remain read-only. No downloaded Python executes in the product.
"""

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

from filelock import FileLock
from huggingface_hub import snapshot_download

MODEL_ID = "alibiserikbay/JevK5-2B"
REVISION = "7922d1f55df137b72ef763fced56fd09efc5e99d"
PROTOCOL = "jevk5_letters_v1"
MAX_SEQUENCE_TOKENS = 1024
TEMPERATURE = 1.42
LABELS = "ABCDEFGHIJKLMNOP"
SOURCE_HASHES = {
    "config.json": "6cb8daca9fb653c61485ff7452fc068bacd5c27cbee659ecd24b47186b0d1b52",
    "generation_config.json": "62153eb6c69f2e1f426beaa8002b7186437e949c7588167085df14e10e9c0a73",
    "tokenizer_config.json": "9cf04fffe3d8c3b85e439fb35c7acad0761ab51c422a8c4256d9f887c3a0be7d",
    "chat_template.jinja": "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80",
    "jevk5_config.json": "c3ebf048dd55a95c3a45e368efc7534dc5ca5a5fd4276b3bc8ead068b38097d4",
    # Fixed release LFS digests; actual downloaded bytes must match before loading.
    "model.safetensors": "847f8b2e6016f1c779cd01ff75428333ad2e9ab491853c36e9521529caf39892",
    "tokenizer.json": "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
}
SOURCE_FILES = tuple(SOURCE_HASHES)


def _require_model(model_id):
    if model_id != MODEL_ID:
        raise ValueError(f"unsupported Windows JevK5 model: {model_id}")


def verify_source(checkpoint: Path, model_id: str = MODEL_ID) -> dict:
    _require_model(model_id)
    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_dir():
        raise ValueError("JevK5 source directory is missing")
    for entry in checkpoint.iterdir():
        if entry.name == ".cache" and entry.is_dir():
            continue
        if entry.name not in SOURCE_HASHES or not entry.is_file():
            raise ValueError(f"unexpected JevK5 source entry: {entry.name}")
    for name, expected in SOURCE_HASHES.items():
        path = checkpoint / name
        if not path.is_file():
            raise ValueError(f"JevK5 source file is missing: {name}")
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"JevK5 source checksum mismatch: {name}")
    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    calibration = json.loads(
        (checkpoint / "jevk5_config.json").read_text(encoding="utf-8")
    )
    if (
        config.get("model_type") != "qwen3_5_text"
        or config.get("architectures") != ["Qwen3_5ForCausalLM"]
        or config.get("hidden_size") != 2048
        or config.get("num_hidden_layers") != 24
        or config.get("tie_word_embeddings") is not True
        or config.get("dtype") != "bfloat16"
        or "text_config" in config
        or config.get("auto_map")
        or calibration != {"temperature": TEMPERATURE}
    ):
        raise ValueError("unsupported JevK5 source identity or protocol")
    return calibration


def identify_source(checkpoint: Path) -> str:
    # The backbone config equals Decider-2B's; use the distinct calibration file.
    checkpoint = Path(checkpoint).resolve()
    calibration = checkpoint / "jevk5_config.json"
    if not calibration.is_file():
        raise ValueError("JevK5 source calibration is missing")
    with calibration.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != SOURCE_HASHES["jevk5_config.json"]:
        raise ValueError("unrecognized pinned JevK5 source calibration")
    verify_source(checkpoint)
    return MODEL_ID


def prepare(model_id: str, workspace: Path) -> Path:
    _require_model(model_id)
    output = Path(workspace).resolve() / "checkpoint" / model_id
    output.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output) + ".lock"):
        if output.exists():
            verify_source(output, model_id)
            return output
        with tempfile.TemporaryDirectory(
            prefix=".jevk5-download-", dir=output.parent
        ) as temporary:
            source = Path(temporary) / "source"
            snapshot_download(
                repo_id=model_id,
                revision=REVISION,
                token=False,
                local_dir=source,
                allow_patterns=list(SOURCE_HASHES),
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
    result = {}
    for identifier, source in questions.items():
        if (
            not isinstance(identifier, str)
            or not identifier
            or not isinstance(source, dict)
        ):
            raise ValueError("question IDs and definitions must be nonempty")
        question = dict(source)
        if (
            not isinstance(question.get("instructions"), str)
            or not question["instructions"].strip()
        ):
            raise ValueError("question instructions must be nonempty text")
        kind, criteria = question.get("type"), question.get("criteria")
        if kind == "choice":
            if isinstance(criteria, list):
                if any(not isinstance(key, str) or not key for key in criteria):
                    raise ValueError("Choice labels must be nonempty strings")
                if len(set(criteria)) != len(criteria):
                    raise ValueError("Choice labels must be distinct")
                criteria = question["criteria"] = dict.fromkeys(criteria)
            if not isinstance(criteria, dict) or not 2 <= len(criteria) <= 16:
                raise ValueError("JevK5 Choice requires 2 to 16 options")
            if any(not isinstance(key, str) or not key for key in criteria):
                raise ValueError("Choice labels must be nonempty strings")
            if any(
                value is not None and not isinstance(value, str)
                for value in criteria.values()
            ):
                raise ValueError("Choice descriptions must be text or null")
        elif kind == "noul":
            if criteria is not None and (
                not isinstance(criteria, dict)
                or not set(criteria) <= {"true", "false"}
                or any(
                    value is not None and not isinstance(value, str)
                    for value in criteria.values()
                )
            ):
                raise ValueError(
                    "Noul criteria must contain optional true/false descriptions"
                )
        elif kind == "score":
            if (
                not isinstance(criteria, list)
                or not 2 <= len(criteria) <= 10
                or any(not isinstance(level, str) for level in criteria)
            ):
                raise ValueError("Score requires 2 to 10 text levels")
            if question.get("isolated", False) is not False:
                raise ValueError("JevK5 supports listwise Score only")
        else:
            raise ValueError("unsupported JevK5 question type")
        result[identifier] = question
    return result


class JevK5Agent:
    device = "cpu"

    def __init__(self, model, tokenizer, weight, model_id=MODEL_ID):
        from .decision import DecisionCompiler

        _require_model(model_id)
        self.model_id = model_id
        self.model = model.eval().requires_grad_(False)
        self.weight = weight.detach().contiguous()
        self.compiler = DecisionCompiler.__new__(DecisionCompiler)
        self.compiler.manifest = {
            "prompt_protocol": PROTOCOL,
            "max_length": MAX_SEQUENCE_TOKENS,
            "max_options": 16,
            "temperature": TEMPERATURE,
            "isolated_levels": False,
            "labels": list(LABELS),
        }
        self.compiler.tokenizer = tokenizer
        self.compiler.model_names = [model_id]
        self.compiler.model_id = model_id
        self.compiler.protocol = PROTOCOL
        self.compiler.max_length = MAX_SEQUENCE_TOKENS
        self.compiler.labels = list(LABELS)
        self.compiler.special = []
        self._precision()

    def _precision(self):
        import torch

        if self.model is None:
            raise RuntimeError("JevK5 CPU model is closed")
        tensors = [*self.model.parameters(), *self.model.buffers(), self.weight]
        if any(t.is_meta or t.device.type != "cpu" for t in tensors):
            raise ValueError("JevK5 parameters must be materialized on CPU")
        embedding = self.model.get_input_embeddings().weight
        if (
            any(p.dtype != torch.bfloat16 for p in self.model.parameters())
            or self.weight.dtype != torch.bfloat16
        ):
            raise ValueError("JevK5 requires BF16 backbone and vocabulary readout")
        if (
            self.weight.shape != (16, embedding.shape[1])
            or not torch.isfinite(self.weight).all()
        ):
            raise ValueError("invalid JevK5 vocabulary readout")
        return {
            "backbone_dtype": "bfloat16",
            "head_dtype": "bfloat16",
            "readout_dtype": "bfloat16",
            "logits_dtype": "float32",
            "row_batch_size": 1,
            "padding_multiple": 1,
            "max_sequence_tokens": MAX_SEQUENCE_TOKENS,
            "temperature": TEMPERATURE,
        }

    def _readout(self, ids):
        import torch

        with torch.inference_mode():
            tokens = torch.tensor([ids], dtype=torch.long, device="cpu")
            hidden = self.model(input_ids=tokens, use_cache=False).last_hidden_state
            if (
                hidden.shape != (1, len(ids), self.weight.shape[1])
                or hidden.dtype != torch.bfloat16
                or hidden.device.type != "cpu"
            ):
                raise RuntimeError("JevK5 returned unexpected hidden states")
            # Preserve the author's 1 x 16 BF16 matrix product before slicing.
            logits = (hidden[:, -1] @ self.weight.T)[0].float()
            if not torch.isfinite(logits).all():
                raise RuntimeError("JevK5 returned nonfinite logits")
            return logits.numpy()

    def system_one(self, state, questions):
        import numpy as np

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
        prepared, _ = self.compiler.compile(payload)
        started = time.perf_counter()
        encoded, answers = {}, {}
        input_tokens = 0
        for identifier, kind, keys, descriptions, isolated, rows in prepared:
            if isolated or len(rows) != 1:
                raise RuntimeError("JevK5 requires one listwise row per question")
            ids, positions, count = rows[0]
            if not ids or positions != [len(ids) - 1] or not 2 <= count <= 16:
                raise RuntimeError("unsupported JevK5 readout positions")
            key = tuple(ids)
            if key not in encoded:
                encoded[key] = self._readout(ids)
            logits = encoded[key][:count] / TEMPERATURE
            probability = np.exp(logits - logits.max())
            probability /= probability.sum()
            answers[identifier] = _answer(
                kind, keys, descriptions, [float(p) for p in probability]
            )
            input_tokens += len(ids)
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
    _require_model(model_id)
    if threads is None:
        threads = min(4, os.cpu_count() or 1)
    if type(threads) is not int or not 1 <= threads <= (os.cpu_count() or 1):
        raise ValueError("threads must be between 1 and the logical CPU count")
    checkpoint = Path(checkpoint).resolve()
    verify_source(checkpoint, model_id)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(threads)
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint, local_files_only=True, trust_remote_code=False
    )
    labels = [tokenizer.encode(letter, add_special_tokens=False) for letter in LABELS]
    if any(len(row) != 1 for row in labels) or len({row[0] for row in labels}) != 16:
        raise ValueError("JevK5 answer letters must be distinct single tokens")
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
        raise ValueError("JevK5 weights were not loaded completely")
    weight = (
        full.get_output_embeddings().weight[[row[0] for row in labels]].detach().clone()
    )
    if weight.shape != (16, 2048):
        raise ValueError("JevK5 readout does not match the selected release")
    backbone = full.model
    del full
    backbone.config.use_cache = False
    return JevK5Agent(backbone, tokenizer, weight, model_id)
