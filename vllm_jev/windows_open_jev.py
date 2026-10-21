"""Pinned Open-Jev-2B, with its BF16 text backbone and FP32 head on CPU.

The release is read-only. No downloaded Python, generation, persistent cache,
quantization, padding, or JEV_TORCH_DTYPE override is used by this adapter.
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
from .prompt import noul_prompt, render_value
from .prompt import probabilities as candidate_probabilities

MODEL_ID = "ZefanCai/Open-Jev-2B"
REVISION = "0c7aa498b1627be8da4acf34c863ff0ee0a92785"
BASE_MODEL_ID = "Qwen/Qwen3.5-2B"
BASE_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
PROTOCOL = "open_jev_choice"
MAX_SEQUENCE_TOKENS = 1024
SOURCE_HASHES = {
    "LICENSE": "bbedc3fda3305820b977265f01b8619d87570a6739de3a5582c3464840f1e57a",
    "LICENSE-CODE": "ef2f6f6db605a2f611bc2d037c59276f615cdea76ac6e7551a28f71db890c9d2",
    "package/checkpoint/adapter/adapter_config.json": "d27624bdbbe984cf175e10c7fa3d9beaea9876231ce943a248ad40e12742fe74",
    "package/checkpoint/adapter/adapter_model.safetensors": "2d23935b1a7380db444abac572c04646918ba794e59002d1588236182a3ca18f",
    "package/checkpoint/head.pt": "3532cd576c58d5ad5bf17c3e9f2df4be8c70e08c07fa6bb7fa673dcd7b401f2a",
    "package/checkpoint/model.json": "d985eafa4d635113ee8cc912e0512a95aac0bed3426e9256931235b75cb3d6c5",
    "package/checkpoint/temperature.json": "090fb330a616338210a924d1c370c934e934f30f53f92c8a52a5cc7852cadd99",
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
    # Upstream LFS expected digest; verify_source checks the downloaded bytes.
    "base/model.safetensors-00001-of-00001.safetensors": "aa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1",
}
SOURCE_FILES = tuple(SOURCE_HASHES)


def _require_model(model_id):
    if model_id != MODEL_ID:
        raise ValueError(f"this Windows CPU adapter supports only {MODEL_ID}")


def verify_source(checkpoint: Path, model_id: str = MODEL_ID) -> dict:
    """Verify source identity and reject files that could override local loaders."""
    _require_model(model_id)
    checkpoint = Path(checkpoint).resolve()
    # Reject unexpected entries before hashing large weights. Hub bookkeeping
    # is allowed only at the two download roots, never inside an adapter.
    for directory in ("base", "package/checkpoint", "package/checkpoint/adapter"):
        path = checkpoint / directory
        if not path.is_dir():
            raise ValueError(f"Open-Jev source directory is missing: {directory}")
        allowed = {
            name.removeprefix(directory + "/").split("/")[0]
            for name in SOURCE_HASHES
            if name.startswith(directory + "/")
        }
        for entry in path.iterdir():
            if directory == "base" and entry.name == ".cache" and entry.is_dir():
                continue
            relative = entry.relative_to(checkpoint).as_posix()
            if entry.name not in allowed or (
                relative in SOURCE_HASHES and not entry.is_file()
            ):
                raise ValueError(f"unexpected Open-Jev source entry: {relative}")
    for name, expected in SOURCE_HASHES.items():
        path = checkpoint / name
        if not path.is_file():
            raise ValueError(f"Open-Jev source file is missing: {name}")
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"Open-Jev source checksum mismatch: {name}")
    package = checkpoint / "package/checkpoint"
    specification = json.loads((package / "model.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        (checkpoint / "release-manifest.json").read_text(encoding="utf-8")
    )
    config = json.loads((checkpoint / "base/config.json").read_text(encoding="utf-8"))
    if (
        specification.get("model_id") != BASE_MODEL_ID
        or specification.get("revision") != BASE_REVISION
        or specification.get("max_length") != 4096
        or specification.get("lora_rank") != 8
        or manifest.get("repo_id") != MODEL_ID
        or manifest.get("base_model") != BASE_MODEL_ID
        or manifest.get("base_revision") != BASE_REVISION
        or config.get("model_type") != "qwen3_5"
        or config.get("text_config", {}).get("model_type") != "qwen3_5_text"
    ):
        raise ValueError("unsupported Open-Jev source identity or architecture")
    return specification


def identify_source(checkpoint: Path) -> str:
    verify_source(checkpoint)
    return MODEL_ID


def prepare(model_id: str, workspace: Path) -> Path:
    """Publish the original package and base only after their fixed hashes match."""
    _require_model(model_id)
    output = Path(workspace).resolve() / "checkpoint" / model_id
    output.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output) + ".lock"):
        if output.exists():
            verify_source(output, model_id)
            return output
        with tempfile.TemporaryDirectory(
            prefix=".open-jev-download-", dir=output.parent
        ) as temporary:
            source = Path(temporary) / "source"
            snapshot_download(
                repo_id=model_id,
                revision=REVISION,
                token=False,
                local_dir=source,
                allow_patterns=[
                    name for name in SOURCE_HASHES if not name.startswith("base/")
                ],
            )
            snapshot_download(
                repo_id=BASE_MODEL_ID,
                revision=BASE_REVISION,
                token=False,
                local_dir=source / "base",
                allow_patterns=[
                    name[5:] for name in SOURCE_HASHES if name.startswith("base/")
                ],
            )
            verify_source(source, model_id)
            source.replace(output)
    return output


def _describe(value):
    if not isinstance(value, (str, dict, list)):
        raise ValueError("instructions and descriptions must be text or JSON")
    return render_value(value)


def _questions(questions):
    if not isinstance(questions, dict) or not 1 <= len(questions) <= 64:
        raise ValueError("System One requires 1 to 64 questions")
    prepared, count = [], 0
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
            if any(not isinstance(key, str) or not key for key in keys):
                raise ValueError("Choice candidate names must be nonempty strings")
            options = [
                key if value is None else f"{key}: {_describe(value)}"
                for key, value in criteria.items()
            ]
            count += len(options)
        elif kind == "score":
            if not isinstance(criteria, list) or not 2 <= len(criteria) <= 10:
                raise ValueError("Score requires 2 to 10 descriptive levels")
            keys = [str(i) for i in range(len(criteria))]
            options = [_describe(value) for value in criteria]
            count += len(options)
        elif kind == "noul":
            if criteria is not None:
                if not isinstance(criteria, dict) or not set(criteria) <= {
                    "true",
                    "false",
                }:
                    raise ValueError(
                        "Noul criteria must contain only true and false descriptions"
                    )
                for key, label in (("true", "Yes"), ("false", "No")):
                    if key in criteria:
                        question += f"\n{label} means: {_describe(criteria[key])}"
            keys, options = [], []
            count += 1
        else:
            raise ValueError("question type must be choice, score or noul")
        prepared.append((identifier, kind, question, criteria, keys, options))
    if count > 256:
        raise ValueError("System One exceeds 256 candidate sequences")
    return prepared, count


def _candidate_prompts(state, question, options):
    """Keep every candidate position, including identical rendered text."""
    if any(not option.strip() for option in options):
        raise ValueError("every candidate must be nonempty text")
    template = current_template()
    options_text = (
        render_value(list(options))
        if template.context_order is not None
        or "options" in template.fields.get("choice_prompt", set())
        else ""
    )
    return [
        template.choice(render_value(state), question, option, options_text)
        for option in options
    ]


class OpenJev2BAgent:
    """Batch-one uncached scorer; the shared service serializes model access."""

    device = "cpu"
    protocol = PROTOCOL

    def __init__(self, model, tokenizer, weight, bias, *, temperature):
        import torch

        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        self.model, self.tokenizer = model, tokenizer
        self.weight, self.bias = weight, bias
        self.temperature = float(temperature)
        for name, value in (*model.named_parameters(), *model.named_buffers()):
            if value.device.type != "cpu" or value.is_meta:
                raise ValueError(
                    f"Open-Jev parameter is not materialized on CPU: {name}"
                )
        self._precision()
        hidden_size = model.get_input_embeddings().weight.shape[-1]
        if (
            weight.shape != (1, hidden_size)
            or bias.shape != (1,)
            or not torch.isfinite(weight).all()
            or not torch.isfinite(bias).all()
        ):
            raise ValueError("invalid Open-Jev scalar head")

    def _precision(self):
        import torch

        embedding = self.model.get_input_embeddings().weight
        if embedding.device.type != "cpu" or embedding.dtype != torch.bfloat16:
            raise ValueError("Open-Jev requires a materialized BF16 CPU backbone")
        if any(
            value.device.type != "cpu" or value.dtype != torch.float32
            for value in (self.weight, self.bias)
        ):
            raise ValueError("Open-Jev requires a materialized FP32 CPU head")
        return {
            "backbone_dtype": str(embedding.dtype).removeprefix("torch."),
            "head_dtype": str(self.weight.dtype).removeprefix("torch."),
            "candidate_batch_size": 1,
            "max_sequence_tokens": MAX_SEQUENCE_TOKENS,
        }

    def _encode(self, raw):
        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": raw}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if not 1 <= len(ids) <= MAX_SEQUENCE_TOKENS:
            raise ValueError(
                f"Open-Jev CPU prompt must contain 1 to {MAX_SEQUENCE_TOKENS} tokens; no truncation"
            )
        return ids

    def _scores(self, branches):
        import torch
        from torch.nn import functional as F

        scores = []
        with torch.inference_mode():
            for ids in branches:
                tokens = torch.tensor([ids], dtype=torch.long, device="cpu")
                hidden = (
                    self.model(
                        input_ids=tokens,
                        attention_mask=torch.ones_like(tokens),
                        use_cache=False,
                        return_dict=True,
                    )
                    .last_hidden_state[:, -1]
                    .float()
                )
                scores.append(float(F.linear(hidden, self.weight, self.bias).item()))
        if any(not math.isfinite(value) for value in scores):
            raise RuntimeError("Open-Jev returned nonfinite candidate scores")
        return scores

    def system_one(self, state, questions):
        if self.model is None:
            raise RuntimeError("Open-Jev service is closed")
        precision = self._precision()
        if not isinstance(state, (str, dict, list)):
            raise ValueError("state must be text or JSON")
        try:
            json.dumps([state, questions], ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError, RecursionError) as error:
            raise ValueError("decision data must be finite JSON") from error
        current_template().validate_protocol(PROTOCOL)
        prepared, count = _questions(questions)
        encoded = []
        for _, kind, question, _, _, options in prepared:
            prompts = (
                [noul_prompt(state, question)]
                if kind == "noul"
                else _candidate_prompts(state, question, options)
                if len(options) > 1
                else []
            )
            encoded.append([self._encode(prompt) for prompt in prompts])
        answers = {}
        started = time.perf_counter()
        for item, branches in zip(prepared, encoded):
            identifier, kind, _, criteria, keys, options = item
            if kind == "choice" and len(options) == 1:
                answers[identifier] = {
                    "type": kind,
                    "choice": keys[0],
                    "probabilities": {keys[0]: 1.0},
                    "confidence": 1.0,
                }
                continue
            scores = self._scores(branches)
            if kind == "noul":
                scaled = scores[0] / self.temperature
                probability = (
                    1 / (1 + math.exp(-scaled))
                    if scaled >= 0
                    else math.exp(scaled) / (1 + math.exp(scaled))
                )
                answers[identifier] = {"type": kind, "noul": probability}
                continue
            # Descriptions are model input, not probability identities. Distinct
            # Choice keys or Score levels may have the same rendered text.
            values = candidate_probabilities(scores, self.temperature)
            probabilities = dict(zip(keys, values))
            answer = {"type": kind, "probabilities": probabilities}
            if kind == "choice":
                selected = max(range(len(values)), key=values.__getitem__)
                uniform = 1.0 / len(values)
                answer.update(
                    choice=keys[selected],
                    confidence=(values[selected] - uniform) / (1.0 - uniform),
                )
            else:
                mode = max(range(len(keys)), key=lambda i: probabilities[keys[i]])
                distance = sum(
                    probabilities[key] * abs(i - mode) for i, key in enumerate(keys)
                )
                center = (len(keys) - 1) / 2
                uniform = sum(abs(i - center) for i in range(len(keys))) / len(keys)
                answer.update(
                    score=sum(i * probabilities[key] for i, key in enumerate(keys)),
                    confidence=max(0.0, 1.0 - distance / uniform),
                    legend=dict(zip(keys, criteria)),
                )
            answers[identifier] = answer
        return {
            "answers": answers,
            "model": MODEL_ID,
            "usage": {
                "input_tokens": sum(len(ids) for group in encoded for ids in group),
                "output_tokens": 0,
            },
            "metadata": {
                "method": "lora_decision_head",
                "temperature": self.temperature,
                "candidate_sequences": count,
                "cached_tokens": 0,
                "inference_seconds": time.perf_counter() - started,
                **precision,
            },
        }

    def close(self):
        self.model = self.tokenizer = self.weight = self.bias = None


def load_agent(checkpoint: Path, model_id: str = MODEL_ID, threads=None):
    """Load the author's complete wrapper, retain only its adapted text model."""
    _require_model(model_id)
    if threads is None:
        threads = min(4, os.cpu_count() or 1)
    if type(threads) is not int or not 1 <= threads <= (os.cpu_count() or 1):
        raise ValueError("threads must be between 1 and the logical CPU count")
    checkpoint = Path(checkpoint).resolve()
    verify_source(checkpoint, model_id)
    import torch
    from peft import PeftModel, get_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    torch.set_num_threads(threads)
    base, package = checkpoint / "base", checkpoint / "package/checkpoint"
    tokenizer = AutoTokenizer.from_pretrained(
        base,
        local_files_only=True,
        trust_remote_code=False,
    )
    if not tokenizer.chat_template:
        raise ValueError("Open-Jev requires the pinned chat template")
    full, loading = AutoModelForImageTextToText.from_pretrained(
        base,
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
        raise ValueError("Open-Jev base weights were not loaded completely")
    backbone = full.model.language_model
    del full
    backbone.config.use_cache = False
    backbone = PeftModel.from_pretrained(
        backbone,
        package / "adapter",
        is_trainable=False,
        local_files_only=True,
    )
    # A PEFT load can warn and continue with missing adapters. Compare the
    # actual retained tensors with every published adapter key before serving.
    expected = load_file(
        str(package / "adapter/adapter_model.safetensors"), device="cpu"
    )
    actual = get_peft_model_state_dict(backbone, save_embedding_layers=False)
    if (
        not expected
        or set(actual) != set(expected)
        or any(
            value.is_meta
            or value.device.type != "cpu"
            or not torch.isfinite(value).all()
            or not torch.equal(value, expected[name].to(value.dtype))
            for name, value in actual.items()
        )
    ):
        raise ValueError("Open-Jev LoRA parameters did not load completely")
    del expected, actual
    backbone.eval().requires_grad_(False)
    head = torch.load(package / "head.pt", map_location="cpu", weights_only=True)
    if not isinstance(head, dict) or set(head) != {"weight", "bias"}:
        raise ValueError("invalid Open-Jev scalar head keys")
    temperature = json.loads(
        (package / "temperature.json").read_text(encoding="utf-8")
    )["temperature"]
    return OpenJev2BAgent(
        backbone,
        tokenizer,
        head["weight"],
        head["bias"],
        temperature=temperature,
    )
