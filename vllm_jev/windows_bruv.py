"""Pinned Bruv direct-options scoring with FP32 on native Windows CPU.

The published BF16 weights are evaluated in an explicit FP32 profile. This is
the author's hidden-state/label-row scorer, not full-vocabulary generation.
Choice/Noul/Score are this project's documented mapping to the author protocol.
"""

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

from filelock import FileLock
from huggingface_hub import snapshot_download

MODEL_ID = "bvolpato/bruv1-0.8b"
REVISION = "7d3660db482cae2afb1fe1ee091a659444791c13"
PROTOCOL = "bruv_direct_options_v1"
BACKEND = "bruv-pytorch"
MAX_SEQUENCE_TOKENS = 1024
LETTERS = "ABCDEFGHIJKLMNOP"
DIRECT_SYSTEM = (
    "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
    "Respond with only its uppercase letter, with no explanation or reasoning."
)
SOURCE_HASHES = {
    "config.json": "3f56b6210db3c52e0bafb50da005eb52b6d340ab4ad980091879b397743675fc",
    "generation_config.json": "62153eb6c69f2e1f426beaa8002b7186437e949c7588167085df14e10e9c0a73",
    "tokenizer.json": "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
    "tokenizer_config.json": "9cf04fffe3d8c3b85e439fb35c7acad0761ab51c422a8c4256d9f887c3a0be7d",
    "chat_template.jinja": "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80",
    "model.safetensors": "0da0a6084adc748d61952774f21ddc4e55cfa9f760d948588ecad6096f4b4589",
    "LICENSE": "bbedc3fda3305820b977265f01b8619d87570a6739de3a5582c3464840f1e57a",
    "training_run.json": "0b9dbb7e944ff0254a60f423c6590a3cf552504debfc73333ff1b812fbe33d57",
}
SOURCE_FILES = tuple(SOURCE_HASHES)


def validate_template(template=None):
    from .decision_template import current_template

    if template is None:
        template = current_template()
    if template.config not in ({}, {"layout": "native"}):
        raise ValueError("Bruv supports only the unchanged native decision template")


def verify_source(checkpoint: Path, model_id: str = MODEL_ID) -> dict:
    if model_id != MODEL_ID:
        raise ValueError("unsupported Windows Bruv model")
    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_dir():
        raise ValueError("Bruv source directory is missing")
    for entry in checkpoint.iterdir():
        if entry.name == ".cache" and entry.is_dir():
            continue
        if entry.name not in SOURCE_HASHES or not entry.is_file():
            raise ValueError(f"unexpected Bruv source entry: {entry.name}")
    for name, expected in SOURCE_HASHES.items():
        path = checkpoint / name
        if not path.is_file():
            raise ValueError(f"Bruv source file is missing: {name}")
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"Bruv source checksum mismatch: {name}")
    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    if (
        config.get("model_type") != "qwen3_5_text"
        or config.get("architectures") != ["Qwen3_5ForCausalLM"]
        or config.get("hidden_size") != 1024
        or config.get("num_hidden_layers") != 24
        or config.get("vocab_size") != 248320
        or config.get("tie_word_embeddings") is not True
        or config.get("auto_map")
        or "text_config" in config
    ):
        raise ValueError("unsupported Bruv source identity")
    return config


def identify_source(checkpoint: Path) -> str:
    verify_source(checkpoint)
    return MODEL_ID


def prepare(model_id: str, workspace: Path) -> Path:
    validate_template()
    if model_id != MODEL_ID:
        raise ValueError("unsupported Windows Bruv model")
    output = Path(workspace).resolve() / "checkpoint" / model_id
    output.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output) + ".lock"):
        if output.exists():
            verify_source(output)
            return output
        with tempfile.TemporaryDirectory(
            prefix=".bruv-download-", dir=output.parent
        ) as tmp:
            source = Path(tmp) / "source"
            snapshot_download(
                repo_id=model_id,
                revision=REVISION,
                token=False,
                local_dir=source,
                allow_patterns=list(SOURCE_FILES),
            )
            verify_source(source)
            source.replace(output)
    return output


def _question(question):
    if not isinstance(question, dict):
        raise ValueError("Bruv questions must be objects")
    kind = question.get("type")
    instructions = question.get("instructions")
    criteria = question.get("criteria")
    if not isinstance(instructions, str) or not instructions.strip():
        raise ValueError("question instructions must be nonempty text")
    if "isolated" in question:
        raise ValueError(
            "Bruv uses joint option probabilities; isolated is unsupported"
        )
    if kind == "choice" and isinstance(criteria, (dict, list)):
        keys = list(criteria)
        if any(not isinstance(key, str) or not key for key in keys) or len(
            set(keys)
        ) != len(keys):
            raise ValueError("Choice labels must be distinct nonempty strings")
        descriptions = (
            list(criteria.values())
            if isinstance(criteria, dict)
            else [None] * len(keys)
        )
        if any(
            value is not None and not isinstance(value, str) for value in descriptions
        ):
            raise ValueError("Choice descriptions must be text or null")
        options = [
            f"{key}: {value}" if value else key
            for key, value in zip(keys, descriptions)
        ]
        if not 2 <= len(keys) <= 16:
            raise ValueError("Bruv Choice requires 2 to 16 options")
    elif kind == "noul":
        if criteria is not None and (
            not isinstance(criteria, dict) or not set(criteria) <= {"false", "true"}
        ):
            raise ValueError(
                "Noul criteria must contain optional false/true descriptions"
            )
        keys = ["false", "true"]
        descriptions = [(criteria or {}).get(key) for key in keys]
        if any(
            value is not None and not isinstance(value, str) for value in descriptions
        ):
            raise ValueError("Noul descriptions must be text or null")
        options = [
            f"{label}: {value}" if value else label
            for label, value in zip(("No", "Yes"), descriptions)
        ]
    elif kind == "score" and isinstance(criteria, list):
        if not 2 <= len(criteria) <= 10 or any(
            not isinstance(value, str) or not value for value in criteria
        ):
            raise ValueError("Bruv Score requires 2 to 10 nonempty text levels")
        keys = [str(i) for i in range(len(criteria))]
        descriptions = list(criteria)
        options = list(criteria)
    else:
        raise ValueError("unsupported Bruv question")
    return kind, keys, descriptions, instructions, options


class BruvAgent:
    device = "cpu"
    model_id = MODEL_ID
    revision = REVISION

    def __init__(self, model, tokenizer, threads=4):
        import torch

        self.model = model.eval().requires_grad_(False)
        self.tokenizer = tokenizer
        self.threads = threads
        ids = []
        for letter in LETTERS:
            encoded = tokenizer.encode(letter, add_special_tokens=False)
            if len(encoded) != 1 or tokenizer.decode(encoded) != letter:
                raise ValueError("Bruv option letters must be round-trip single tokens")
            ids.append(encoded[0])
        if len(set(ids)) != 16:
            raise ValueError("Bruv option letter tokens are not distinct")
        self.label_ids = torch.tensor(ids, dtype=torch.long, device="cpu")
        self._precision()

    def _precision(self):
        import torch

        if self.model is None:
            raise RuntimeError("Bruv CPU model is closed")
        if any(
            t.is_meta or t.device.type != "cpu"
            for t in (*self.model.parameters(), *self.model.buffers())
        ):
            raise ValueError("Bruv parameters must be materialized on CPU")
        if any(p.dtype != torch.float32 for p in self.model.parameters()):
            raise ValueError("Bruv Windows profile requires FP32 parameters")
        return {
            "backbone_dtype": "float32",
            "head_dtype": "float32",
            "logits_dtype": "float32",
            "softmax_dtype": "float32",
            "temperature": 1.0,
            "attention_implementation": "eager",
            "use_cache": False,
            "cache_reuse": False,
            "row_batch_size": 1,
            "max_sequence_tokens": MAX_SEQUENCE_TOKENS,
            "threads": self.threads,
            "prompt_format": "kevala-direct-options-v1",
            "systemone_mapping": "bruv_joint_v1",
        }

    def _encode(self, state, instructions, options):
        payload = {
            "evidence": state,
            "criterion": instructions,
            "options": [
                {"letter": LETTERS[i], "description": value}
                for i, value in enumerate(options)
            ],
        }
        prompt = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": DIRECT_SYSTEM},
                {
                    "role": "user",
                    "content": json.dumps(payload, ensure_ascii=False, allow_nan=False),
                },
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if not ids or len(ids) > MAX_SEQUENCE_TOKENS:
            raise ValueError("Bruv complete prompt exceeds 1024 tokens")
        for letter, token in zip(LETTERS[: len(options)], self.label_ids.tolist()):
            if self.tokenizer.encode(
                prompt + letter, add_special_tokens=False
            ) != ids + [token]:
                raise ValueError("Bruv answer token changes at prompt boundary")
        return ids

    def _probabilities(self, ids, count):
        import torch
        import torch.nn.functional as F

        with torch.inference_mode():
            tokens = torch.tensor([ids], dtype=torch.long, device="cpu")
            hidden = self.model.model(
                input_ids=tokens,
                attention_mask=torch.ones_like(tokens),
                use_cache=False,
            ).last_hidden_state
            if (
                hidden.dtype != torch.float32
                or hidden.device.type != "cpu"
                or hidden.shape[:2] != (1, len(ids))
            ):
                raise RuntimeError("Bruv returned unexpected hidden states")
            last = hidden[:, -1].float()
            # Keep all 16 author label rows before masking, preserving FP32 GEMM.
            rows = self.model.lm_head.weight.index_select(0, self.label_ids).float()
            logits = F.linear(last, rows)[0]
            if not torch.isfinite(logits).all():
                raise RuntimeError("Bruv returned nonfinite option logits")
            logits = logits.masked_fill(torch.arange(16) >= count, float("-inf"))
            return torch.softmax(logits, dim=-1)[:count].tolist()

    def system_one(self, state, questions):
        from .decision import _answer

        validate_template()
        precision = self._precision()
        if not isinstance(state, (str, dict, list)):
            raise ValueError("decision state must be text or JSON")
        state_json = json.dumps(state, ensure_ascii=False, allow_nan=False)
        if len(state if isinstance(state, str) else state_json) > 50_000:
            raise ValueError("decision state exceeds 50,000 characters")
        if not isinstance(questions, dict) or not 1 <= len(questions) <= 64:
            raise ValueError("System One requires 1 to 64 questions")
        if (
            len(json.dumps(questions, ensure_ascii=False, allow_nan=False).encode())
            > 2_000_000
        ):
            raise ValueError("decision questions exceed 2 MiB")
        prepared, candidates = [], 0
        for identifier, spec in questions.items():
            if not isinstance(identifier, str) or not identifier:
                raise ValueError("question IDs must be nonempty strings")
            kind, keys, descriptions, instructions, options = _question(spec)
            candidates += len(options)
            if candidates > 256:
                raise ValueError("System One exceeds 256 candidates")
            prepared.append(
                (
                    identifier,
                    kind,
                    keys,
                    descriptions,
                    self._encode(state, instructions, options),
                )
            )
        # Validate every full prompt before the first model forward.
        started = time.perf_counter()
        encoded, answers = {}, {}
        for identifier, kind, keys, descriptions, ids in prepared:
            key = (tuple(ids), len(keys))
            if key not in encoded:
                encoded[key] = self._probabilities(ids, len(keys))
            answers[identifier] = _answer(kind, keys, descriptions, encoded[key])
        return {
            "model": MODEL_ID,
            "answers": answers,
            "usage": {
                "input_tokens": sum(len(row[-1]) for row in prepared),
                "output_tokens": 0,
            },
            "internal_usage": {"compute_tokens": sum(len(key[0]) for key in encoded)},
            "metadata": {
                **precision,
                "cached_tokens": 0,
                "inference_seconds": time.perf_counter() - started,
            },
        }

    def close(self):
        self.model = self.tokenizer = self.label_ids = None


def load_agent(checkpoint: Path, model_id: str = MODEL_ID, threads=None):
    validate_template()
    if threads is None:
        threads = min(4, os.cpu_count() or 1)
    if type(threads) is not int or not 1 <= threads <= min(4, os.cpu_count() or 1):
        raise ValueError("Bruv threads must be between 1 and 4 and within CPU count")
    verify_source(checkpoint, model_id)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(threads)
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint, local_files_only=True, trust_remote_code=False
    )
    model, loading = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        dtype=torch.float32,
        attn_implementation="eager",
        local_files_only=True,
        trust_remote_code=False,
        output_loading_info=True,
    )
    if any(
        loading.get(key)
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    ):
        raise ValueError("Bruv weights did not load completely")
    if model.config.model_type != "qwen3_5_text" or model.config.hidden_size != 1024:
        raise ValueError("Bruv loaded an unexpected backbone")
    return BruvAgent(model, tokenizer, threads)
