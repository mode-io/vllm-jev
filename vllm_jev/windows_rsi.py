"""Pinned RSI-Jev v1.0 option cross-attention on native Windows CPU.

The state-first layout and mean-pool/scorer equations follow Shanghua Gao's
MIT-licensed RSI-Jev implementation. The checkpoint model card lists Apache-2.0.
Serving never executes downloaded Python. Original release code is run
independently by the reference.
"""

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

from filelock import FileLock
from huggingface_hub import snapshot_download

MODEL_ID = "shgao/rsi-jev-v1.0-qwen3.5-0.8b"
REVISION = "f9248caceb89caf2e6c968ea33bf0d6eb7f957b0"
BASE_ID = "Qwen/Qwen3.5-0.8B-Base"
BASE_REVISION = "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"
PROTOCOL = "rsi_option_xattn_v1"
MAX_SEQUENCE_TOKENS = 1024
MAX_OPTIONS = 80
SOURCE_HASHES = {
    "meta.json": "2dbae889af349197429e924865abd79a98017771ef6a017e5c7fd2479b956141",
    "tower.safetensors": "60f8ea114a6dc3070be268a1f95973b503bd300b2965555f8acad2d95cfcff67",
    "scorer.safetensors": "22cb924e0eb9432db467539e18df48ba837a09eee1106b94f04a21b430c11a06",
    "base/config.json": "b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204",
    "base/LICENSE": "50cbab8a892c5f2993b8c7351a99182507472def3b1374558308605d99b86b32",
    "base/model.safetensors.index.json": "ce9a885efdf27d3664fdef5d512ad365216f1074051ef840c7cd8e5431495d0a",
    "base/model.safetensors-00001-of-00001.safetensors": "c2b1e5a17d9c1e27685d92ed9b382911ebb99955ecd89052d1721241adfbab6c",
    "base/tokenizer.json": "fe000e3ed39ed12b8d2481d527d44f93c65d37e87645d2dcc80d1bf9d50d2927",
    "base/tokenizer_config.json": "e611fbccc7c29ef3b1cafb1cb7ea548d189968632901d678fd62be68c47885de",
}


def validate_template(template=None):
    if template is None:
        from .decision_template import current_template

        template = current_template()
    if template.config not in ({}, {"layout": "native"}):
        raise ValueError("RSI-Jev supports only the default/native template")


def verify_source(checkpoint: Path, model_id: str = MODEL_ID):
    if model_id != MODEL_ID:
        raise ValueError("unsupported Windows RSI-Jev release")
    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_dir():
        raise ValueError("RSI-Jev source directory is missing")
    for directory, prefix in ((checkpoint, ""), (checkpoint / "base", "base/")):
        if not directory.is_dir():
            raise ValueError("RSI-Jev base directory is missing")
        for entry in directory.iterdir():
            if entry.name == ".cache" and entry.is_dir():
                continue
            if not prefix and entry.name == "base" and entry.is_dir():
                continue
            if (
                prefix + entry.name not in SOURCE_HASHES
                or not entry.is_file()
                or entry.is_symlink()
            ):
                raise ValueError(
                    f"unexpected RSI-Jev source entry: {prefix}{entry.name}"
                )
    for name, expected in SOURCE_HASHES.items():
        path = checkpoint / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"RSI-Jev source file is missing: {name}")
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                raise ValueError(f"RSI-Jev source checksum mismatch: {name}")
    meta = json.loads((checkpoint / "meta.json").read_text(encoding="utf-8"))
    recipe = meta["spec"]
    if (
        meta.get("base_model") != BASE_ID
        or recipe.get("readout") != "option_xattn"
        or recipe.get("readout_layer") != -1
        or recipe.get("layout") != "state_first"
        or recipe.get("option_pool") != "mean"
        or recipe.get("max_options") != MAX_OPTIONS
        or recipe.get("residual") is not False
        or recipe.get("logit_cap") is not None
        or recipe.get("head_input_norm") is not False
        or recipe.get("arch_extra") != {}
    ):
        raise ValueError("unsupported RSI-Jev scorer recipe")
    return meta


def identify_source(checkpoint: Path) -> str:
    verify_source(checkpoint)
    return MODEL_ID


def prepare(model_id: str, workspace: Path) -> Path:
    validate_template()
    if model_id != MODEL_ID:
        raise ValueError("unsupported Windows RSI-Jev release")
    output = Path(workspace).resolve() / "checkpoint" / model_id
    output.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output) + ".lock"):
        if output.exists():
            verify_source(output)
            return output
        with tempfile.TemporaryDirectory(prefix=".rsi-", dir=output.parent) as tmp:
            source = Path(tmp) / "source"
            snapshot_download(
                repo_id=MODEL_ID,
                revision=REVISION,
                token=False,
                local_dir=source,
                allow_patterns=[
                    name for name in SOURCE_HASHES if not name.startswith("base/")
                ],
            )
            snapshot_download(
                repo_id=BASE_ID,
                revision=BASE_REVISION,
                token=False,
                local_dir=source / "base",
                allow_patterns=[
                    name.removeprefix("base/")
                    for name in SOURCE_HASHES
                    if name.startswith("base/")
                ],
            )
            verify_source(source)
            source.replace(output)
    return output


def _text(value):
    return (
        value
        if isinstance(value, str)
        else json.dumps(
            value, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        )
    )


def parse_questions(questions):
    if not isinstance(questions, dict) or not 1 <= len(questions) <= 64:
        raise ValueError("System One requires 1 to 64 questions")
    if len(_text(questions).encode("utf-8")) > 2_000_000:
        raise ValueError("decision questions exceed 2 MiB")
    parsed, count = [], 0
    for key, spec in questions.items():
        if not isinstance(key, str) or not key or not isinstance(spec, dict):
            raise ValueError("question IDs and definitions must be nonempty")
        if "isolated" in spec:
            raise ValueError(
                "RSI-Jev uses joint option probabilities; isolated is unsupported"
            )
        kind, instructions, criteria = (
            spec.get("type"),
            spec.get("instructions"),
            spec.get("criteria"),
        )
        if instructions is None:
            raise ValueError("RSI-Jev instructions are required")
        instructions = _text(instructions)
        if kind == "noul":
            if criteria is not None and (
                not isinstance(criteria, dict) or set(criteria) - {"false", "true"}
            ):
                raise ValueError("Noul criteria may only set false and true")
            given = criteria or {}
            options = ["false", "true"]
            descriptions = [
                _text(given.get(k, default))
                for k, default in zip(options, ("No", "Yes"))
            ]
        elif kind == "choice":
            if isinstance(criteria, list):
                if any(not isinstance(k, str) or not k for k in criteria) or len(
                    set(criteria)
                ) != len(criteria):
                    raise ValueError("Choice labels must be distinct nonempty strings")
                criteria = dict.fromkeys(criteria)
            if not isinstance(criteria, dict) or not 2 <= len(criteria) <= MAX_OPTIONS:
                raise ValueError("RSI-Jev Choice requires 2 to 80 options")
            if any(not isinstance(k, str) or not k for k in criteria):
                raise ValueError("Choice labels must be nonempty strings")
            options = list(criteria)
            descriptions = ["" if v is None else _text(v) for v in criteria.values()]
        elif kind == "score":
            if not isinstance(criteria, list) or not 2 <= len(criteria) <= 10:
                raise ValueError("RSI-Jev Score requires 2 to 10 levels")
            options = [str(i) for i in range(len(criteria))]
            descriptions = [_text(v) for v in criteria]
        else:
            raise ValueError("question type must be choice, noul or score")
        count += len(options)
        parsed.append((key, kind, instructions, options, descriptions))
    if count > 256:
        raise ValueError("System One exceeds 256 candidates")
    return parsed


def encode_question(tokenizer, state, instructions, options, descriptions):
    """Preserve the author's per-block tokenization; reject without truncation."""
    head = (
        f"{state}\n\n{instructions}\nOptions:\n"
        if state.strip()
        else f"{instructions}\nOptions:\n"
    )
    ids = list(tokenizer(head, add_special_tokens=False)["input_ids"])
    spans = []
    for option, description in zip(options, descriptions):
        block = f"- {option}: {description}" if description else f"- {option}"
        start = len(ids)
        ids += tokenizer("\n" + block, add_special_tokens=False)["input_ids"]
        spans.append((start, len(ids)))
    ids += tokenizer("\n\nAnswer:", add_special_tokens=False)["input_ids"]
    if not ids or len(ids) > MAX_SEQUENCE_TOKENS:
        raise ValueError(
            "RSI-Jev complete prompt exceeds 1024 tokens; truncation is disabled"
        )
    if any(a >= b for a, b in spans):
        raise ValueError("RSI-Jev option block has no tokens")
    return ids, spans, len(ids) - 1


def make_scorer(hidden=1024):
    """The released four-head option_xattn modules, without training features."""
    import torch
    from torch import nn

    class OptionScorer(nn.Module):
        def __init__(self):
            super().__init__()
            self.q, self.k, self.v = (nn.Linear(hidden, hidden) for _ in range(3))
            self.attn = nn.MultiheadAttention(hidden, 4, batch_first=True)
            self.proj = nn.Linear(hidden, 1)

        def forward(self, decision, options, mask):
            with torch.autocast("cpu", enabled=False):
                q = self.q(decision).unsqueeze(1)
                k, v = self.k(options), self.v(options)
                ctx, _ = self.attn(q, k, v, key_padding_mask=~mask, need_weights=False)
                return self.proj(v + ctx).squeeze(-1).masked_fill(~mask, float("-inf"))

    return OptionScorer().float()


class RsiAgent:
    device = "cpu"
    model_id = MODEL_ID
    revision = REVISION

    def __init__(self, tower, tokenizer, scorer):
        self.tower = tower.eval().requires_grad_(False)
        self.tokenizer = tokenizer
        self.scorer = scorer.eval().requires_grad_(False)
        self._precision()

    def _precision(self):
        import torch

        if self.tower is None:
            raise RuntimeError("RSI-Jev CPU model is closed")
        for module in (self.tower, self.scorer):
            if any(
                t.is_meta
                or t.device.type != "cpu"
                or (t.is_floating_point() and t.dtype != torch.float32)
                for t in (*module.parameters(), *module.buffers())
            ):
                raise ValueError(
                    "RSI-Jev requires materialized FP32 CPU tower and scorer"
                )
        return {
            "backbone_dtype": "float32",
            "head_dtype": "float32",
            "logits_dtype": "float32",
            "temperature": 1.0,
            "row_batch_size": 1,
            "max_sequence_tokens": MAX_SEQUENCE_TOKENS,
            "attention_implementation": "sdpa",
            "use_cache": True,
            "cache_reuse": False,
            "option_pool": "mean",
            "readout": "option_xattn",
            "scorer_slots": MAX_OPTIONS,
            "threads": torch.get_num_threads(),
        }

    def _probabilities(self, ids, spans, decision):
        import torch

        with torch.inference_mode():
            tokens = torch.tensor([ids], dtype=torch.long)
            h = self.tower(
                input_ids=tokens,
                attention_mask=torch.ones_like(tokens),
                use_cache=True,
                return_dict=True,
            ).last_hidden_state
            # Preserve the author's padded 80-slot einsum and attention shapes.
            start = torch.zeros((1, MAX_OPTIONS), dtype=torch.long)
            end = torch.zeros_like(start)
            mask = torch.zeros((1, MAX_OPTIONS), dtype=torch.bool)
            for i, (a, b) in enumerate(spans):
                start[0, i], end[0, i], mask[0, i] = a, b, True
            t = torch.arange(h.shape[1]).view(1, 1, -1)
            inside = ((t >= start.unsqueeze(-1)) & (t < end.unsqueeze(-1))).to(h.dtype)
            options = torch.einsum("bkt,bth->bkh", inside, h) / inside.sum(
                -1, keepdim=True
            ).clamp_min(1.0)
            logits = self.scorer(h[:, decision], options, mask)
            if not torch.isfinite(logits[mask]).all():
                raise RuntimeError("RSI-Jev produced nonfinite option logits")
            return torch.softmax(logits, -1)[0, : len(spans)].tolist()

    def system_one(self, state, questions):
        from .decision import _answer

        validate_template()
        precision = self._precision()
        if not isinstance(state, (str, dict, list)):
            raise ValueError("decision state must be text or JSON")
        state = _text(state)
        if len(state) > 50_000:
            raise ValueError("decision state exceeds 50,000 characters")
        parsed = parse_questions(questions)
        # Compile EVERY row before the first forward pass, never a partial answer.
        rows = [
            encode_question(self.tokenizer, state, q[2], q[3], q[4]) for q in parsed
        ]
        answers, cached = {}, {}
        started = time.perf_counter()
        for (identifier, kind, _, keys, descriptions), (ids, spans, decision) in zip(
            parsed, rows
        ):
            key = (tuple(ids), tuple(spans), decision)
            if key not in cached:
                cached[key] = self._probabilities(ids, spans, decision)
            answers[identifier] = _answer(kind, keys, descriptions, cached[key])
        return {
            "model": MODEL_ID,
            "answers": answers,
            "usage": {
                "input_tokens": sum(len(row[0]) for row in rows),
                "output_tokens": 0,
            },
            "internal_usage": {"compute_tokens": sum(len(k[0]) for k in cached)},
            "metadata": {
                **precision,
                "cached_tokens": 0,
                "inference_seconds": time.perf_counter() - started,
            },
        }

    def close(self):
        self.tower = self.scorer = self.tokenizer = None


def load_agent(checkpoint: Path, model_id: str = MODEL_ID, threads=None):
    validate_template()
    if threads is None:
        threads = min(4, os.cpu_count() or 1)
    if type(threads) is not int or not 1 <= threads <= min(4, os.cpu_count() or 1):
        raise ValueError("RSI-Jev CPU threads must be between 1 and 4")
    checkpoint = Path(checkpoint).resolve()
    verify_source(checkpoint, model_id)
    import torch
    from accelerate import init_empty_weights
    from safetensors import safe_open
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    torch.set_num_threads(threads)
    config = AutoConfig.from_pretrained(
        checkpoint / "base", local_files_only=True, trust_remote_code=False
    ).text_config
    config.dtype = torch.float32
    config._attn_implementation = "sdpa"
    config.use_cache = True
    if (
        config.model_type != "qwen3_5_text"
        or config.hidden_size != 1024
        or config.num_hidden_layers != 24
    ):
        raise ValueError("unexpected RSI-Jev text configuration")
    # Only parameter slots are empty; rotary buffers must remain real CPU FP32.
    with init_empty_weights(include_buffers=False):
        tower = AutoModel.from_config(config, trust_remote_code=False)
    weights = load_file(str(checkpoint / "tower.safetensors"), device="cpu")
    expected = tower.state_dict()
    if set(weights) != set(expected) - {"embed_tokens.weight"}:
        raise ValueError("RSI-Jev release tower keys differ from the base architecture")
    for name, tensor in weights.items():
        if (
            tensor.dtype != torch.float32
            or tensor.shape != expected[name].shape
            or not torch.isfinite(tensor).all()
        ):
            raise ValueError(f"invalid RSI-Jev tower tensor: {name}")
    shard = checkpoint / "base/model.safetensors-00001-of-00001.safetensors"
    with safe_open(shard, framework="pt", device="cpu") as source:
        embedding = source.get_tensor("model.language_model.embed_tokens.weight")
        weights["embed_tokens.weight"] = embedding.float()
    # assign avoids a second 2 GiB tower and never constructs the unused vision/LM head.
    tower.load_state_dict(weights, strict=True, assign=True)
    del weights, expected, embedding
    scorer = make_scorer()
    scorer.load_state_dict(
        load_file(str(checkpoint / "scorer.safetensors")), strict=True
    )
    if any(not torch.isfinite(p).all() for p in scorer.parameters()):
        raise ValueError("RSI-Jev scorer contains nonfinite parameters")
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint / "base", local_files_only=True, trust_remote_code=False
    )
    return RsiAgent(tower, tokenizer, scorer)
