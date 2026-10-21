"""Pinned Tiny-Jev source and a native PyTorch CPU decision adapter.

The published Python file is retained and hashed for reference verification,
but serving never imports it or enables ``trust_remote_code``. The backbone
and scalar head run in FP32 without generation or a persistent KV cache.
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

from .decision_template import current_template
from .prompt import choice_result, render_value, tiny_token_ids

MODEL_ID = "lostargon/Tiny-Jev"
REVISION = "62449fb3272d68fb8c8fe8a33082566643a76d0b"
PROTOCOL = "tiny_jev_marker"
SOURCE_HASHES = {
    "model.safetensors": "513411b17cb0eca54839ca3948b4353e7f65710723ffcde8d13e3cb04328633e",
    "config.json": "3897b8c2f7ee8e566449b170f2f8a9cc2dde447fb474be50a155485d69859937",
    "tokenizer.json": "e3fccc77adda606e2a461ffd17ca5a55a5a75ac40a5aa0a91c5307fb31b1efac",
    "tokenizer_config.json": "5e44295ef654157e02ca593d0be8c07ed11265d4553e7920005d721ff5a9feab",
    "chat_template.jinja": "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8",
    "modeling_tiny_jev.py": "5c2d146e2f92ffcb7bb8a0927b180a3d9458b5fc3c4f5d99835302910802e05d",
}
SOURCE_FILES = tuple(SOURCE_HASHES)
MODEL_ID_17B = "lostargon/Tiny-Jev-1.7B"
MODELS = {
    MODEL_ID: {
        "revision": REVISION,
        "hidden_size": 1024,
        "temperature": 0.95,
        "source_hashes": SOURCE_HASHES,
    },
    MODEL_ID_17B: {
        "revision": "c41bcf3f191407c779bd2ebce1b992dd1469437f",
        "hidden_size": 2048,
        "temperature": 1.15,
        "source_hashes": {
            **SOURCE_HASHES,
            "model.safetensors": "49ff7a60951abe35631b9f0632b8250acdb9142f79f39a57f56aa42befac7a3b",
            "config.json": "48cd095ffe1863dd003895a38a3359be84d76ef79ec95e8e76ecf92604e28839",
        },
    },
}


def _require_model(model_id: str) -> None:
    if model_id not in MODELS:
        raise ValueError("this Windows CPU adapter supports only " + ", ".join(MODELS))


def verify_source(checkpoint: Path, model_id: str = MODEL_ID) -> dict:
    """Require the exact original release, including its tokenizer and head."""
    _require_model(model_id)
    checkpoint = Path(checkpoint).resolve()
    spec = MODELS[model_id]
    for name, expected in spec["source_hashes"].items():
        source = checkpoint / name
        if not source.is_file():
            raise ValueError(f"Tiny-Jev source file is missing: {name}")
        with source.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"Tiny-Jev source checksum mismatch: {name}")
    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    if (
        config.get("model_type") != "tiny_jev"
        or config.get("architectures") != ["TinyJevModel"]
        or config.get("hidden_size") != spec["hidden_size"]
        or config.get("opt_token") != "<opt>"
        or config.get("temperature") != spec["temperature"]
    ):
        raise ValueError("unsupported Tiny-Jev source architecture")
    return config


def identify_source(checkpoint: Path) -> str:
    checkpoint = Path(checkpoint)
    weight = checkpoint / "model.safetensors"
    if weight.is_file():
        with weight.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        for model_id, spec in MODELS.items():
            if digest == spec["source_hashes"][weight.name]:
                verify_source(checkpoint, model_id)
                return model_id
    raise ValueError("unrecognized Tiny-Jev source checkpoint")


def prepare(model_id: str, workspace: Path) -> Path:
    """Publish a pinned download only after all source files pass verification."""
    _require_model(model_id)
    output = Path(workspace).resolve() / "checkpoint" / model_id
    output.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output) + ".lock"):
        if output.exists():
            verify_source(output, model_id)
            return output
        with tempfile.TemporaryDirectory(
            prefix=".tiny-download-", dir=output.parent
        ) as directory:
            source = Path(directory) / "source"
            snapshot_download(
                repo_id=model_id,
                revision=MODELS[model_id]["revision"],
                token=False,
                local_dir=source,
                allow_patterns=list(MODELS[model_id]["source_hashes"]),
            )
            verify_source(source, model_id)
            source.replace(output)
    return output


def _describe(value) -> str:
    if not isinstance(value, (str, dict, list)):
        raise ValueError("instructions and descriptions must be text or JSON")
    return render_value(value)


def _questions(questions: dict) -> tuple[list, int]:
    """Mirror the endpoint's Tiny-Jev labels, Noul criteria, and Score levels."""
    if not isinstance(questions, dict) or not 1 <= len(questions) <= 64:
        raise ValueError("System One requires 1 to 64 questions")
    prepared = []
    candidate_count = 0
    for identifier, definition in questions.items():
        if (
            not isinstance(identifier, str)
            or not identifier
            or not isinstance(definition, dict)
        ):
            raise ValueError("question IDs and definitions must be nonempty")
        kind = definition.get("type")
        question = _describe(definition.get("instructions"))
        if not question.strip():
            raise ValueError("question must be nonempty text")
        criteria = definition.get("criteria")
        if kind == "choice":
            if not isinstance(criteria, dict) or not 1 <= len(criteria) <= 255:
                raise ValueError("Choice requires 1 to 255 candidates")
            keys = list(criteria)
            if any(not isinstance(key, str) or not key.strip() for key in keys):
                raise ValueError("Choice candidate names must be nonempty strings")
            options = [
                key if value is None else f"{key}: {_describe(value)}"
                for key, value in criteria.items()
            ]
        elif kind == "score":
            if not isinstance(criteria, list) or not 2 <= len(criteria) <= 10:
                raise ValueError("Score requires 2 to 10 descriptive levels")
            keys = options = [_describe(value) for value in criteria]
        elif kind == "noul":
            if criteria is not None:
                if not isinstance(criteria, dict) or set(criteria) != {"true", "false"}:
                    raise ValueError("Noul criteria must define true and false")
                question += (
                    f"\nYes means: {_describe(criteria['true'])}"
                    f"\nNo means: {_describe(criteria['false'])}"
                )
            keys, options = [], ["no", "yes"]
        else:
            raise ValueError("question type must be choice, score or noul")
        candidate_count += len(options)
        prepared.append((identifier, kind, question, criteria, keys, options))
    if candidate_count > 256:
        raise ValueError("System One exceeds 256 candidate sequences")
    return prepared, candidate_count


class TinyAgent:
    """Synchronous adapter; the HTTP service owns serialization and cancellation."""

    device = "cpu"
    protocol = PROTOCOL

    def __init__(
        self,
        model,
        tokenizer,
        weight,
        bias,
        *,
        temperature=0.95,
        cue=" - correct?",
        opt_token="<opt>",
        model_id=MODEL_ID,
    ):
        _require_model(model_id)
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        self.model = model
        self.tokenizer = tokenizer
        self.weight = weight
        self.bias = bias
        self.temperature = float(temperature)
        self.cue = cue
        self.opt_token = opt_token
        self.model_id = model_id
        self.revision = MODELS[model_id]["revision"]

    def _scores(self, ids: list[int], positions: list[int]) -> list[float]:
        import torch
        from torch.nn import functional as F

        with torch.inference_mode():
            input_ids = torch.tensor([ids], dtype=torch.long, device="cpu")
            hidden = self.model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                use_cache=False,
                return_dict=True,
            ).last_hidden_state[0]
            # The release reads the cue's final token, just BEFORE <opt>.
            selected = hidden[positions].float()
            normalized = F.layer_norm(selected, (hidden.shape[-1],), eps=1e-5)
            return F.linear(normalized, self.weight, self.bias)[:, 0].tolist()

    def system_one(self, state, questions) -> dict:
        if self.model is None:
            raise RuntimeError("Tiny-Jev service is closed")
        if not isinstance(state, (str, dict, list)):
            raise ValueError("state must be text or JSON")
        # Catch malformed or nonfinite JSON before a model forward.
        try:
            json.dumps([state, questions], ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError, RecursionError) as error:
            raise ValueError("decision data must be finite JSON") from error
        current_template().validate_protocol(PROTOCOL)
        prepared, candidate_count = _questions(questions)
        encoded = [
            tiny_token_ids(
                self.tokenizer,
                state,
                question,
                options,
                cue=self.cue,
                opt_token=self.opt_token,
            )
            if len(options) > 1
            else ([], [])
            for _, _, question, _, _, options in prepared
        ]
        answers = {}
        started = time.perf_counter()
        for item, (ids, positions) in zip(prepared, encoded):
            identifier, kind, _, criteria, keys, options = item
            if len(options) == 1:
                answers[identifier] = {
                    "type": "choice",
                    "choice": keys[0],
                    "probabilities": {keys[0]: 1.0},
                    "confidence": 1.0,
                }
                continue
            result = choice_result(
                options, self._scores(ids, positions), self.temperature
            )
            if kind == "noul":
                answers[identifier] = {
                    "type": kind,
                    "noul": result["probabilities"]["yes"],
                }
                continue
            probabilities = {
                key: result["probabilities"][option]
                for key, option in zip(keys, options)
            }
            answer = {
                "type": kind,
                "probabilities": probabilities,
                "confidence": result["selected_probability"],
            }
            if kind == "choice":
                answer["choice"] = keys[options.index(result["choice"])]
            else:
                answer["score"] = sum(
                    i * probabilities[key] for i, key in enumerate(keys)
                )
                answer["legend"] = dict(zip(keys, criteria))
            answers[identifier] = answer
        return {
            "answers": answers,
            "model": self.model_id,
            "usage": {
                "input_tokens": sum(len(ids) for ids, _ in encoded),
                "output_tokens": 0,
            },
            "metadata": {
                "method": "marker_token_head",
                "temperature": self.temperature,
                "candidate_sequences": candidate_count,
                "cached_tokens": 0,
                "inference_seconds": time.perf_counter() - started,
            },
        }

    def close(self) -> None:
        self.model = self.tokenizer = self.weight = self.bias = None


def load_agent(
    checkpoint: Path, model_id: str = MODEL_ID, threads: int | None = None
) -> TinyAgent:
    """Load the original release without modifying source files or importing code."""
    _require_model(model_id)
    if threads is None:
        threads = min(4, os.cpu_count() or 1)
    if type(threads) is not int or not 1 <= threads <= (os.cpu_count() or 1):
        raise ValueError("threads must be between 1 and the logical CPU count")
    checkpoint = Path(checkpoint).resolve()
    source_config = verify_source(checkpoint, model_id)
    import torch
    from safetensors.torch import load_file
    from transformers import Qwen2TokenizerFast, Qwen3Config, Qwen3Model

    torch.set_num_threads(threads)
    config = dict(source_config)
    for field in ("auto_map", "architectures", "model_type", "dtype"):
        config.pop(field, None)
    # This release was saved by Transformers 5; 4.x uses separate RoPE fields.
    rope = config.pop("rope_parameters")
    if rope.get("rope_type") != "default":
        raise ValueError("unsupported Tiny-Jev rotary embedding configuration")
    config.update(rope_theta=rope["rope_theta"], use_cache=False)
    config = Qwen3Config.from_dict(config)
    config._attn_implementation = "sdpa"
    if model_id == MODEL_ID_17B:
        from accelerate import init_empty_weights
        from safetensors import safe_open

        # Leave nonpersistent rotary buffers on CPU, while avoiding an initial
        # full FP32 parameter allocation alongside the converted release.
        with init_empty_weights(include_buffers=False):
            model = Qwen3Model(config)
        weight_path = checkpoint / "model.safetensors"
        with safe_open(weight_path, framework="pt", device="cpu") as source:
            names = list(source.keys())
        weights = {}
        for name in names:
            # Release each source mapping before converting the next tensor.
            # The release's FP32 head also needs its own storage: a no-op
            # dtype conversion would retain the entire BF16 checkpoint mapping.
            with safe_open(weight_path, framework="pt", device="cpu") as source:
                weights[name] = source.get_tensor(name).to(
                    dtype=torch.float32, copy=True
                )
    else:
        # Preserve the already-validated 0.6B loading path.
        model = Qwen3Model(config).to(device="cpu", dtype=torch.float32)
        weights = load_file(str(checkpoint / "model.safetensors"), device="cpu")
    weight, bias = weights.pop("head.weight"), weights.pop("head.bias")
    if (
        weight.shape != (1, config.hidden_size)
        or bias.shape != (1,)
        or not torch.isfinite(weight).all()
        or not torch.isfinite(bias).all()
        or any(not name.startswith("model.") for name in weights)
    ):
        raise ValueError("invalid Tiny-Jev backbone or decision head")
    model.load_state_dict(
        {name.removeprefix("model."): value for name, value in weights.items()},
        strict=True,
        assign=model_id == MODEL_ID_17B,
    )
    del weights
    model.eval().requires_grad_(False)
    if any(
        tensor.device.type != "cpu"
        for tensor in (*model.parameters(), *model.buffers())
    ):
        raise ValueError("Tiny-Jev parameters and buffers must be materialized on CPU")
    tokenizer_config = json.loads(
        (checkpoint / "tokenizer_config.json").read_text(encoding="utf-8")
    )
    for field in ("tokenizer_class", "backend", "is_local", "local_files_only"):
        tokenizer_config.pop(field, None)
    tokenizer_config["additional_special_tokens"] = tokenizer_config.pop(
        "extra_special_tokens"
    )
    tokenizer = Qwen2TokenizerFast(
        tokenizer_file=str(checkpoint / "tokenizer.json"), **tokenizer_config
    )
    if (
        len(tokenizer) != config.vocab_size
        or tokenizer.encode("\n", add_special_tokens=False) != [198]
        or tokenizer.convert_tokens_to_ids(source_config["opt_token"]) is None
    ):
        raise ValueError("Tiny-Jev tokenizer does not match the marker protocol")
    return TinyAgent(
        model,
        tokenizer,
        weight.float().contiguous(),
        bias.float().contiguous(),
        temperature=source_config["temperature"],
        cue=source_config["cue"],
        opt_token=source_config["opt_token"],
        model_id=model_id,
    )
