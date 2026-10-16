"""Pinned JevK5-Lite encoder and label-conditioned scorer on Windows CPU.

The product uses the author's single-head FP32 calculation and JevBench label
mapping. Complete rows must fit 512 tokens; input is never silently truncated.
Downloaded model directories are read-only and contain no executable code.
"""

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

from filelock import FileLock
from huggingface_hub import snapshot_download

MODEL_ID = "alibiserikbay/JevK5-Lite"
REVISION = "315ee211f828a899161477c534cea23e84ba3568"
PROTOCOL = "jevk5_lite_v1"
BACKEND = "jevk5-lite-pytorch"
MAX_SEQUENCE_TOKENS = 512
TEMPERATURE = 1.8660659830736146
SOURCE_HASHES = {
    "config.json": "4b143dff2309fc312d0413c1d42b4c1f75cc5f33e70b593f4d4568877e56783e",
    "lite_config.json": "e80a53d67af90d2d8237cfb6a9dfd365501b0da3b8019ad2ed7827fee3b35777",
    "tokenizer_config.json": "d008608e43ac0a01895fb8a6a241316c2869ae5af65724195b4a28d4f600a8b1",
    "tokenizer.json": "5eebf550a388329ccc733e5e9b73f75c83e4e50ff0fe99a9885217cf20d274fd",
    "NOTICE": "ced5d9c7234168de5b3b057895305cd1c1377a3556b2afc09e74d3c063fa92ba",
    "model.safetensors": "152b56535b93874f3404ce421c86c88e0e064b4f321a7e41c709357843f714de",
    "scorer.safetensors": "1ca3c4f5f2e88b87a9951055f567fd9e64258e027d0ecd5d9b375e49399ee98d",
}
SOURCE_FILES = tuple(SOURCE_HASHES)


def _require_model(model_id):
    if model_id != MODEL_ID:
        raise ValueError(f"unsupported Windows JevK5-Lite model: {model_id}")


def validate_template(template=None):
    if template is None:
        from .decision_template import current_template

        template = current_template()
    if template.config not in ({}, {"layout": "native"}):
        raise ValueError(
            "Windows JevK5-Lite supports only the default/native decision template"
        )


def verify_source(checkpoint: Path, model_id: str = MODEL_ID) -> dict:
    _require_model(model_id)
    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_dir():
        raise ValueError("JevK5-Lite source directory is missing")
    for entry in checkpoint.iterdir():
        if entry.name == ".cache" and entry.is_dir() and not entry.is_symlink():
            continue
        if entry.name not in SOURCE_HASHES or not entry.is_file() or entry.is_symlink():
            raise ValueError(f"unexpected JevK5-Lite source entry: {entry.name}")
    for name, expected in SOURCE_HASHES.items():
        path = checkpoint / name
        if not path.is_file():
            raise ValueError(f"JevK5-Lite source file is missing: {name}")
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"JevK5-Lite source checksum mismatch: {name}")
    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    calibration = json.loads(
        (checkpoint / "lite_config.json").read_text(encoding="utf-8")
    )
    tokenizer = json.loads(
        (checkpoint / "tokenizer_config.json").read_text(encoding="utf-8")
    )
    if (
        config.get("model_type") != "deberta-v2"
        or config.get("architectures") != ["DebertaV2Model"]
        or config.get("hidden_size") != 1024
        or config.get("num_hidden_layers") != 24
        or config.get("vocab_size") != 128003
        or config.get("dtype") != "float32"
        or config.get("auto_map")
        or tokenizer.get("tokenizer_class") != "DebertaV2Tokenizer"
        or tokenizer.get("backend") != "tokenizers"
        or tokenizer.get("auto_map")
        or calibration.get("format") != "jevk5-lite/1"
        or calibration.get("base") != "microsoft/deberta-v3-large"
        or calibration.get("max_len") != MAX_SEQUENCE_TOKENS
        or calibration.get("temperature_single") != TEMPERATURE
        or calibration.get("temperature_multi") != 1.319507910772894
    ):
        raise ValueError("unsupported JevK5-Lite source identity or calibration")
    return calibration


def identify_source(checkpoint: Path) -> str:
    verify_source(checkpoint)
    return MODEL_ID


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
            prefix=".jevk5-lite-download-", dir=output.parent
        ) as temporary:
            source = Path(temporary) / "source"
            snapshot_download(
                repo_id=model_id,
                revision=REVISION,
                token=False,
                local_dir=source,
                allow_patterns=list(SOURCE_FILES),
            )
            verify_source(source, model_id)
            source.replace(output)
    return output


def _finite_json(value):
    try:
        return json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError, RecursionError) as error:
        raise ValueError("decision data must be finite JSON") from error


def _questions(questions):
    """The published bench/lite_head mapping, limited to supported wire inputs."""
    if not isinstance(questions, dict) or not 1 <= len(questions) <= 64:
        raise ValueError("System One requires 1 to 64 questions")
    if len(_finite_json(questions).encode("utf-8")) > 2_000_000:
        raise ValueError("decision questions exceed 2 MiB")
    result, candidates = [], 0
    for identifier, question in questions.items():
        if (
            not isinstance(identifier, str)
            or not identifier
            or not isinstance(question, dict)
        ):
            raise ValueError("question IDs and definitions must be nonempty")
        instructions = question.get("instructions")
        if not isinstance(instructions, str) or not instructions.strip():
            raise ValueError("question instructions must be nonempty text")
        kind, criteria = question.get("type"), question.get("criteria")
        if kind == "choice":
            if isinstance(criteria, list):
                if any(not isinstance(key, str) or not key for key in criteria) or len(
                    set(criteria)
                ) != len(criteria):
                    raise ValueError("Choice labels must be distinct nonempty strings")
                criteria = dict.fromkeys(criteria)
            if not isinstance(criteria, dict) or not 2 <= len(criteria) <= 255:
                raise ValueError("JevK5-Lite Choice requires 2 to 255 options")
            if any(not isinstance(key, str) or not key for key in criteria) or any(
                value is not None and not isinstance(value, str)
                for value in criteria.values()
            ):
                raise ValueError(
                    "Choice labels must be text; descriptions text or null"
                )
            pairs = [(key, value or key) for key, value in criteria.items()]
        elif kind == "noul":
            if criteria is not None and criteria != {}:
                raise ValueError("JevK5-Lite Noul does not support custom criteria")
            pairs = [("true", "true"), ("false", "false")]
        elif kind == "score":
            if (
                not isinstance(criteria, list)
                or not 2 <= len(criteria) <= 10
                or any(not isinstance(level, str) for level in criteria)
            ):
                raise ValueError("Score requires 2 to 10 text levels")
            if question.get("isolated", False) is not False:
                raise ValueError("JevK5-Lite supports listwise Score only")
            pairs = [(str(index), value) for index, value in enumerate(criteria)]
        else:
            raise ValueError("unsupported JevK5-Lite question type")
        keys, labels = [p[0] for p in pairs], [p[1] for p in pairs]
        if len(set(labels)) != len(labels):
            labels = [f"{key}: {label}" for key, label in pairs]
        # This can still collide for adversarial key/description combinations;
        # the original classifier rejects them rather than merging probability.
        if len(set(labels)) != len(labels):
            raise ValueError("JevK5-Lite rendered labels must be distinct")
        candidates += len(labels)
        result.append((identifier, kind, instructions, keys, labels))
    if candidates > 256:
        raise ValueError("System One exceeds 256 candidates")
    return result


def make_scorer(hidden_size):
    """Create only the author's small MLP; Torch remains a lazy dependency."""
    from torch import nn

    return nn.Sequential(
        nn.Linear(3 * hidden_size, hidden_size), nn.GELU(), nn.Linear(hidden_size, 1)
    )


class LiteAgent:
    device = "cpu"

    def __init__(self, model, tokenizer, scorer, model_id=MODEL_ID):
        _require_model(model_id)
        validate_template()
        self.model_id = model_id
        self.revision = REVISION
        self.model = model.eval().requires_grad_(False)
        self.scorer = scorer.eval().requires_grad_(False)
        self.tokenizer = tokenizer
        self.task_id = tokenizer.convert_tokens_to_ids("[TASK]")
        self.label_id = tokenizer.convert_tokens_to_ids("[LABEL]")
        tokens = [
            self.task_id,
            self.label_id,
            tokenizer.cls_token_id,
            tokenizer.sep_token_id,
        ]
        vocabulary = self.model.get_input_embeddings().weight.shape[0]
        if (
            any(
                type(token) is not int or not 0 <= token < vocabulary
                for token in tokens
            )
            or len(set(tokens)) != 4
            or tokenizer.unk_token_id in tokens
        ):
            raise ValueError("JevK5-Lite requires distinct valid schema tokens")
        self._precision()
        import torch

        if any(not torch.isfinite(p).all() for p in self.scorer.parameters()):
            raise ValueError("JevK5-Lite scorer contains nonfinite weights")

    def _precision(self):
        import torch

        if self.model is None:
            raise RuntimeError("JevK5-Lite CPU model is closed")
        parameters = [*self.model.parameters(), *self.scorer.parameters()]
        tensors = [*parameters, *self.model.buffers(), *self.scorer.buffers()]
        if any(tensor.is_meta or tensor.device.type != "cpu" for tensor in tensors):
            raise ValueError("JevK5-Lite tensors must be materialized on CPU")
        if any(tensor.dtype != torch.float32 for tensor in parameters) or any(
            tensor.is_floating_point() and tensor.dtype != torch.float32
            for tensor in tensors
        ):
            raise ValueError("JevK5-Lite requires FP32 encoder and scorer")
        hidden = self.model.get_input_embeddings().weight.shape[1]
        expected = {
            "0.weight": (hidden, 3 * hidden),
            "0.bias": (hidden,),
            "2.weight": (1, hidden),
            "2.bias": (1,),
        }
        if {
            key: tuple(t.shape) for key, t in self.scorer.state_dict().items()
        } != expected:
            raise ValueError("JevK5-Lite scorer shape is invalid")
        return {
            "model": self.model_id,
            "revision": self.revision,
            "backbone_dtype": str(parameters[0].dtype).removeprefix("torch."),
            "head_dtype": str(self.scorer[0].weight.dtype).removeprefix("torch."),
            "readout_dtype": "float32",
            "logits_dtype": "float32",
            "softmax_dtype": "float32",
            "softmax_backend": "torch",
            "temperature": TEMPERATURE,
            "threads": torch.get_num_threads(),
            "row_batch_size": 1,
            "padding_multiple": 1,
            "padding": "none",
            "packed_questions": False,
            "max_sequence_tokens": MAX_SEQUENCE_TOKENS,
            "truncated_state_tokens": 0,
            "cache_reuse": False,
            "graphs": False,
            "parameter_devices": sorted({p.device.type for p in parameters}),
            "parameter_dtypes": sorted({str(p.dtype) for p in parameters}),
        }

    def _piece(self, text):
        result = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        vocabulary = self.model.get_input_embeddings().weight.shape[0]
        if not isinstance(result, list) or any(
            type(token) is not int or not 0 <= token < vocabulary for token in result
        ):
            raise ValueError("JevK5-Lite tokenizer returned invalid token IDs")
        return result

    def _encode(self, state, instructions, labels):
        schema = [self.task_id] + self._piece(f"{instructions.replace('_', ' ')} (one)")
        positions = []
        for label in labels:
            positions.append(1 + len(schema))
            schema += [self.label_id] + self._piece(label)
        body = self._piece(state)
        ids = (
            [self.tokenizer.cls_token_id]
            + schema
            + [self.tokenizer.sep_token_id]
            + body
            + [self.tokenizer.sep_token_id]
        )
        if len(ids) > MAX_SEQUENCE_TOKENS:
            raise ValueError("JevK5-Lite complete prompt exceeds 512 tokens")
        start = 2 + len(schema)
        return ids, positions, (start, start + len(body))

    def _probabilities(self, row):
        import torch

        ids, positions, (start, end) = row
        with torch.inference_mode():
            tokens = torch.tensor([ids], dtype=torch.long, device="cpu")
            hidden = self.model(
                input_ids=tokens, attention_mask=torch.ones_like(tokens)
            ).last_hidden_state
            width = self.model.get_input_embeddings().weight.shape[1]
            if (
                hidden.shape != (1, len(ids), width)
                or hidden.dtype != torch.float32
                or hidden.device.type != "cpu"
            ):
                raise RuntimeError("JevK5-Lite returned unexpected hidden states")
            hidden = hidden[0]
            text = hidden[start : max(end, start + 1)].mean(0, keepdim=True)
            labels = hidden[positions]
            text = text.expand_as(labels)
            logits = (
                self.scorer(torch.cat([labels, text, labels * text], -1))
                .squeeze(-1)
                .float()
            )
            if logits.shape != (len(positions),) or not torch.isfinite(logits).all():
                raise RuntimeError("JevK5-Lite returned invalid label logits")
            # Preserve the author's Torch FP32 softmax, without a second
            # normalization in Python or a conversion through NumPy.
            return torch.softmax(logits / TEMPERATURE, -1).tolist()

    def system_one(self, state, questions):
        from .decision import _answer

        validate_template()
        precision = self._precision()
        if not isinstance(state, (str, dict, list)):
            raise ValueError("decision state must be text or JSON")
        text = state if isinstance(state, str) else _finite_json(state)
        if len(text) > 50_000:
            raise ValueError("decision state exceeds 50,000 characters")
        # Validate and tokenize all rows before any encoder call. Late failures
        # must not turn an invalid multi-question request into partial work.
        prepared = [
            (identifier, kind, keys, labels, self._encode(text, instructions, labels))
            for identifier, kind, instructions, keys, labels in _questions(questions)
        ]
        started = time.perf_counter()
        answers = {
            identifier: _answer(kind, keys, labels, self._probabilities(row))
            for identifier, kind, keys, labels, row in prepared
        }
        input_tokens = sum(len(row[0]) for *_, row in prepared)
        return {
            "model": self.model_id,
            "answers": answers,
            "usage": {"input_tokens": input_tokens, "output_tokens": 0},
            "internal_usage": {"compute_tokens": input_tokens},
            "metadata": {
                **precision,
                "inference_seconds": time.perf_counter() - started,
            },
        }

    def close(self):
        self.model = self.scorer = self.tokenizer = None


def load_agent(checkpoint: Path, model_id: str = MODEL_ID, threads=None):
    _require_model(model_id)
    validate_template()
    maximum = min(4, os.cpu_count() or 1)
    if threads is None:
        threads = maximum
    if type(threads) is not int or not 1 <= threads <= maximum:
        raise ValueError("JevK5-Lite threads must be between 1 and min(4, CPU count)")
    checkpoint = Path(checkpoint).resolve()
    verify_source(checkpoint, model_id)
    import torch
    from safetensors.torch import load_file
    from transformers import AutoModel, AutoTokenizer

    torch.set_num_threads(threads)
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint, local_files_only=True, trust_remote_code=False
    )
    model, loading = AutoModel.from_pretrained(
        checkpoint,
        dtype=torch.float32,
        attn_implementation="eager",
        device_map={"": "cpu"},
        local_files_only=True,
        trust_remote_code=False,
        output_loading_info=True,
    )
    if any(
        loading.get(key)
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    ):
        raise ValueError("JevK5-Lite encoder weights were not loaded completely")
    if (
        model.config.model_type != "deberta-v2"
        or model.config.hidden_size != 1024
        or model.config.num_hidden_layers != 24
        or model.get_input_embeddings().weight.shape != (128003, 1024)
    ):
        raise ValueError("JevK5-Lite encoder does not match the selected release")
    head = load_file(checkpoint / "scorer.safetensors", device="cpu")
    if any(value.dtype != torch.float32 for value in head.values()):
        raise ValueError("JevK5-Lite released scorer must be FP32")
    scorer = make_scorer(model.config.hidden_size)
    scorer.load_state_dict(head, strict=True)
    return LiteAgent(model, tokenizer, scorer, model_id)
