"""RSI-Jev decisions over native Qwen3.5 token states.

The tower runs in vLLM. Each question is one sequence: the state, the
instructions, one ``- key: description`` block per option, and ``Answer:``.
The released option cross-attention scorer reads the final normed state at the
last token and the mean state of each option block; the released calibration
then divides each question's logits by one positive temperature. Prompt and
readout follow the published ``rsijev.encode`` and ``rsijev.arch`` code.
"""

import asyncio
import json
import math
import os
import secrets
import time
import uuid
from contextlib import suppress
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import nn

from .media import load_image, validate_text

PROTOCOL = "rsijev_xattn_v1"
MODES = ("choice", "noul", "score")
NOUL_OPTIONS = ("false", "true")
NOUL_DEFAULTS = {"true": "Yes", "false": "No"}
CHAT_ROLES = {"system", "user", "assistant", "tool"}
MAX_QUESTIONS = 64
MAX_IMAGES = 4
IMAGE_MARKER = "<image>"
VISION_START, IMAGE_PAD, VISION_END = (
    "<|vision_start|>",
    "<|image_pad|>",
    "<|vision_end|>",
)
CAL_PCA_DIM = 16
CAL_LOGT_CLAMP = 3.0


def serialize(value) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def state_parts(state):
    """Return ``(text, images)``. Images come from ``image_url`` message parts.

    Without images, a chat transcript is rendered as compact JSON, as RSI-Jev's
    own server does. With images, text parts and images are concatenated in
    order with nothing added between them: each image stands where RSI-Jev's
    ``<image>`` marker would, so ``[image, "\\nQuestion"]`` is the trained
    ``"<image>\\nQuestion"`` state.
    """
    if not isinstance(state, (str, dict, list)):
        raise ValueError("RSI-Jev state must be text, JSON, or messages")
    candidate = (
        state["messages"]
        if isinstance(state, dict) and set(state) == {"messages"}
        else state
    )
    if not (
        isinstance(candidate, list)
        and candidate
        and all(isinstance(item, dict) and "role" in item for item in candidate)
    ):
        return validate_text(serialize(state)), []
    contents = []
    for item in candidate:
        if item["role"] not in CHAT_ROLES:
            raise ValueError("chat state has an unsupported message role")
        content = item.get("content")
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        if content is None:
            content = []
        if not isinstance(content, list) or not all(
            isinstance(part, dict)
            and (
                (part.get("type") == "text" and isinstance(part.get("text"), str))
                or part.get("type") == "image_url"
            )
            for part in content
        ):
            raise ValueError("RSI-Jev message content must be text and image parts")
        contents += content
    if not any(part["type"] == "image_url" for part in contents):
        return validate_text(serialize(candidate)), []
    parts, images = [], []
    for part in contents:
        if part["type"] == "text":
            if IMAGE_MARKER in validate_text(part["text"]):
                raise ValueError(f"text parts must not contain {IMAGE_MARKER}")
            parts.append(part["text"])
            continue
        if len(images) == MAX_IMAGES:
            raise ValueError(f"RSI-Jev accepts at most {MAX_IMAGES} images")
        url = part.get("image_url")
        if not isinstance(url, dict):
            raise ValueError("image_url must contain a data URL")
        images.append(load_image(url.get("url")))
        parts.append(IMAGE_MARKER)
    return "".join(parts), images


def to_question(key: str, spec, max_options: int):
    """One System One question as ``(key, mode, instructions, options, criteria)``."""
    if not key or not isinstance(spec, dict):
        raise ValueError("question IDs and definitions must be nonempty")
    kind, instructions = spec.get("type"), spec.get("instructions")
    if instructions is None:
        raise ValueError(f"questions.{key}: instructions is required")
    text = validate_text(serialize(instructions))
    criteria = spec.get("criteria")
    if kind == "noul":
        given = criteria or {}
        if not isinstance(given, dict) or set(given) - set(NOUL_OPTIONS):
            raise ValueError(f"questions.{key}: Noul criteria may set true and false")
        described = {k: serialize(given.get(k, NOUL_DEFAULTS[k])) for k in NOUL_OPTIONS}
        options = NOUL_OPTIONS
    elif kind == "choice":
        if not isinstance(criteria, dict) or not 2 <= len(criteria) <= max_options:
            raise ValueError(
                f"questions.{key}: Choice needs 2 to {max_options} options"
            )
        if any(not isinstance(option, str) or not option for option in criteria):
            raise ValueError(f"questions.{key}: Choice labels must be nonempty text")
        options = tuple(criteria)
        described = {k: "" if v is None else serialize(v) for k, v in criteria.items()}
    elif kind == "score":
        if not isinstance(criteria, list) or not 2 <= len(criteria) <= max_options:
            raise ValueError(f"questions.{key}: Score needs 2 to {max_options} levels")
        options = tuple(str(i) for i in range(len(criteria)))
        described = {str(i): serialize(v) for i, v in enumerate(criteria)}
    else:
        raise ValueError("question type must be choice, noul or score")
    for value in (*options, *described.values()):
        validate_text(value)
    return key, kind, text, options, described


def render(state: str, instructions: str, options, criteria, answer_cue="Answer:"):
    """The ``state_first`` layout: a head, one block per option, and a tail."""
    blocks = [
        f"- {option}: {criteria[option]}" if criteria.get(option) else f"- {option}"
        for option in options
    ]
    head = (
        f"{state}\n\n{instructions}\nOptions:\n"
        if state.strip()
        else f"{instructions}\nOptions:\n"
    )
    return head, blocks, f"\n\n{answer_cue}"


def encode_question(tokenizer, state, instructions, options, criteria, max_length):
    """Token ids, option block spans, and the decision position.

    A sequence over ``max_length`` loses tokens from the left of the head, so
    options and the answer cue are never cut.
    """
    head, blocks, tail = render(state, instructions, options, criteria)

    def tokens(text):
        return tokenizer(text, add_special_tokens=False)["input_ids"]

    head_ids = tokens(head)
    ids = list(head_ids)
    spans = []
    for block in blocks:
        start = len(ids)
        ids += tokens("\n" + block)
        spans.append((start, len(ids)))
    ids += tokens(tail)
    overflow = len(ids) - max_length
    if overflow > 0:
        if overflow >= len(head_ids):
            raise ValueError("the options alone exceed the RSI-Jev context")
        ids = ids[overflow:]
        spans = [(a - overflow, b - overflow) for a, b in spans]
    return ids, spans, len(ids) - 1


def expand_state(state: str, tokens_per_image) -> str:
    runs = [VISION_START + IMAGE_PAD * n + VISION_END for n in tokens_per_image]
    count = state.count(IMAGE_MARKER)
    if count == 0:
        return "\n".join(runs) + ("\n" + state if state.strip() else "")
    if count != len(runs):
        raise ValueError(
            f"state has {count} {IMAGE_MARKER} markers for {len(runs)} images"
        )
    pieces = state.split(IMAGE_MARKER)
    return "".join(
        piece + (runs[i] if i < len(runs) else "") for i, piece in enumerate(pieces)
    )


def image_size_limits(n_images: int, budget: int, floor: int, unit: int = 32):
    """Per-image token cap and the processor size bounds RSI-Jev trained with."""
    per_image = max(floor, budget // max(1, n_images))
    return per_image, {
        "shortest_edge": min(floor, per_image) * unit * unit,
        "longest_edge": per_image * unit * unit,
    }


def image_tokens(image, size: dict, unit: int = 32) -> int:
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

    height, width = smart_resize(
        image.height,
        image.width,
        factor=unit,
        min_pixels=size["shortest_edge"],
        max_pixels=size["longest_edge"],
    )
    return (height // unit) * (width // unit)


class RsiJevHead(nn.Module):
    """The released ``option_xattn`` scorer and its fitted calibration."""

    def __init__(self, hidden: int, config: dict):
        super().__init__()
        self.combine = config["xattn_combine"]
        dim = config.get("xattn_dim") or hidden
        self.q = nn.Linear(hidden, dim)
        self.k = nn.Linear(hidden, dim)
        self.v = nn.Linear(hidden, dim)
        self.attn = nn.MultiheadAttention(dim, config["xattn_heads"], batch_first=True)
        self.proj = nn.Linear(dim, 1)
        if self.combine == "mlp":
            width = config["xattn_mlp_hidden"]
            self.comb = nn.Sequential(
                nn.Linear(3 * dim, width), nn.GELU(), nn.Linear(width, 1)
            )
        elif self.combine != "sum":
            raise ValueError(f"unsupported RSI-Jev scorer combine: {self.combine}")
        self.cal_mode = config["cal_mode"]
        width = CAL_PCA_DIM + 4 + len(MODES)
        for name, shape in (
            ("cal_logT", ()),
            ("cal_logT_mode", (len(MODES),)),
            ("cal_pca_mean", (hidden,)),
            ("cal_pca_W", (hidden, CAL_PCA_DIM)),
            ("cal_feat_mu", (width,)),
            ("cal_feat_sd", (width,)),
            ("cal_w", (width,)),
            ("cal_b", ()),
        ):
            self.register_buffer(name, torch.zeros(shape))

    def load(self, scorer: dict, calibration: dict | None) -> None:
        if calibration is None:
            if self.cal_mode != "none":
                raise ValueError("RSI-Jev calibration tensors are missing")
            calibration = {
                name: value
                for name, value in self.state_dict().items()
                if name.startswith("cal_")
            }
        self.load_state_dict({**scorer, **calibration}, strict=True)

    def scores(self, decision, options, mask):
        query = self.q(decision).unsqueeze(1)
        keys, values = self.k(options), self.v(options)
        context, _ = self.attn(
            query, keys, values, key_padding_mask=~mask, need_weights=False
        )
        if self.combine == "sum":
            logits = self.proj(values + context).squeeze(-1)
        else:
            context = context.expand_as(values)
            logits = self.comb(
                torch.cat([values, values * context, values - context], -1)
            ).squeeze(-1)
        return logits.masked_fill(~mask, float("-inf"))

    def log_temperature(self, logits, decision, mode):
        if self.cal_mode == "none":
            return torch.zeros(len(logits), device=logits.device)
        if self.cal_mode == "temp":
            return self.cal_logT.expand(len(logits))
        if self.cal_mode == "temp_mode":
            return self.cal_logT_mode[mode]
        finite = torch.isfinite(logits)
        count = finite.sum(-1).clamp_min(2).float()
        probabilities = torch.softmax(logits, -1)
        top = torch.topk(logits.masked_fill(~finite, -1e9), 2, -1).values
        gap = (top[:, 0] - top[:, 1]).clamp(0, 30)
        entropy = -(probabilities * torch.log(probabilities.clamp_min(1e-12)))
        entropy = entropy.masked_fill(~finite, 0).sum(-1) / torch.log(count)
        features = torch.cat(
            [
                (decision - self.cal_pca_mean) @ self.cal_pca_W,
                probabilities.max(-1).values[:, None],
                gap[:, None],
                entropy[:, None],
                torch.log(count)[:, None],
                F.one_hot(mode, len(MODES)).float(),
            ],
            -1,
        )
        features = (features - self.cal_feat_mu) / self.cal_feat_sd
        value = (features @ self.cal_w + self.cal_b).clamp(
            -CAL_LOGT_CLAMP, CAL_LOGT_CLAMP
        )
        if self.cal_mode == "oof_head_scorefloor":
            value = torch.where(mode == MODES.index("score"), value.clamp_min(0), value)
        elif self.cal_mode != "oof_head":
            raise ValueError(f"unsupported RSI-Jev calibration: {self.cal_mode}")
        return value

    @torch.inference_mode()
    def forward(self, rows, hidden_dtype=torch.bfloat16):
        """``rows`` holds ``(states, spans, decision, mode)``; positions index states.

        Option blocks are mean-pooled in the tower's dtype, as the reference
        serving path does; the scorer and the calibration run in float32.
        """
        device = self.q.weight.device
        width = max(len(spans) for _, spans, _, _ in rows)
        decision = torch.zeros((len(rows), self.q.in_features), device=device)
        options = torch.zeros((len(rows), width, self.q.in_features), device=device)
        mask = torch.zeros((len(rows), width), dtype=torch.bool, device=device)
        mode = torch.zeros(len(rows), dtype=torch.long, device=device)
        for row, (states, spans, index, kind) in enumerate(rows):
            states = states.to(device=device, dtype=hidden_dtype)
            decision[row] = states[index].float()
            for column, (start, end) in enumerate(spans):
                # bf16 GEMM semantics: fp32 sum, then a rounded sum and count.
                total = states[start:end].float().sum(0).to(hidden_dtype)
                count = torch.tensor(float(end - start), dtype=hidden_dtype)
                options[row, column] = (total / count.to(device)).float()
            mask[row, : len(spans)] = True
            mode[row] = MODES.index(kind)
        logits = self.scores(decision, options, mask)
        logits = (
            logits / torch.exp(self.log_temperature(logits, decision, mode))[:, None]
        )
        return [
            torch.softmax(logits[row, : len(spans)], -1).tolist()
            for row, (_, spans, _, _) in enumerate(rows)
        ]


def answer(kind: str, options, criteria, probabilities):
    if kind == "noul":
        return {"type": "noul", "noul": probabilities[1]}
    count = len(probabilities)
    confidence = min(1.0, max(0.0, (count * max(probabilities) - 1) / (count - 1)))
    distribution = dict(zip(options, probabilities))
    if kind == "score":
        return {
            "type": "score",
            "score": sum(i * p for i, p in enumerate(probabilities)),
            "legend": dict(criteria),
            "probabilities": distribution,
            "confidence": confidence,
        }
    selected = max(range(count), key=probabilities.__getitem__)
    return {
        "type": "choice",
        "choice": options[selected],
        "probabilities": distribution,
        "confidence": confidence,
    }


class RsiJevService:
    def __init__(
        self,
        engine_client,
        model_path: Path,
        model_id: str | list[str],
        max_length: int,
        max_inflight: int = 128,
    ):
        from transformers import AutoTokenizer

        self.engine_client = engine_client
        self.config = json.loads((model_path / "rsijev_config.json").read_text())
        model_config = json.loads((model_path / "config.json").read_text())
        hidden = model_config["text_config"]["hidden_size"]
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=True
        )
        self.image_pad_id = model_config["image_token_id"]
        device = os.environ.get("VLLM_JEV_RSIJEV_DEVICE") or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.head = RsiJevHead(hidden, self.config)
        calibration = model_path / "rsijev_calibration.safetensors"
        self.head.load(
            load_file(str(model_path / "rsijev_scorer.safetensors")),
            load_file(str(calibration)) if calibration.is_file() else None,
        )
        self.head.to(device).float().eval()
        self.max_options = self.config["max_options"]
        self.text_length = min(max_length, self.config["max_length_text"])
        self.vision = self.config.get("vision")
        self.image_length = (
            min(max_length, self.config["max_length_image"]) if self.vision else 0
        )
        self.read_prefix_cache = os.environ.get("VLLM_JEV_RSIJEV_PREFIX_CACHE") != "0"
        self.model_names = [model_id] if isinstance(model_id, str) else model_id
        self.model_id = self.model_names[0]
        self.semaphore = asyncio.Semaphore(max_inflight)
        self.compile_lock = asyncio.Lock()

    def _compile(self, payload):
        if payload.model not in (None, "vllm-jev", "rsi-jev", *self.model_names):
            raise ValueError("requested model is not loaded")
        if not isinstance(payload.questions, dict) or not (
            1 <= len(payload.questions) <= MAX_QUESTIONS
        ):
            raise ValueError(f"System One requires 1 to {MAX_QUESTIONS} questions")
        questions = [
            to_question(key, spec, self.max_options)
            for key, spec in payload.questions.items()
        ]
        state, images = state_parts(payload.state)
        mm = None
        max_length = self.text_length
        if images:
            if not self.vision:
                raise ValueError("this RSI-Jev checkpoint is text-only")
            per_image, size = image_size_limits(
                len(images),
                self.vision["image_token_budget"],
                self.vision["min_tokens_per_image"],
            )
            counts = [image_tokens(image, size) for image in images]
            state = expand_state(state, counts)
            max_length = self.image_length
            mm = (images, {"size": size}, sum(counts))
        rows = []
        for key, kind, instructions, options, criteria in questions:
            ids, spans, decision = encode_question(
                self.tokenizer, state, instructions, options, criteria, max_length
            )
            if mm and sum(token == self.image_pad_id for token in ids) != mm[2]:
                raise ValueError("the state is too long: truncation would cut an image")
            rows.append((key, kind, options, criteria, ids, spans, decision))
        return rows, mm

    async def _states(self, ids, need_from, mm, salt):
        """Final hidden states for ``ids[need_from:]``, read from vLLM.

        With prefix caching, vLLM returns states only for the tokens it
        computed. Every row the readout needs follows the shared state, so a
        hit that stops before ``need_from`` is enough; otherwise the sequence
        is computed again without reading the cache.
        """
        prompt_ids = []
        for token in ids:
            if token != self.image_pad_id or not prompt_ids or prompt_ids[-1] != token:
                prompt_ids.append(token)
        prompt = {"prompt_token_ids": prompt_ids, "cache_salt": salt}
        if mm:
            images, kwargs, _ = mm
            prompt["multi_modal_data"] = {
                "image": images[0] if len(images) == 1 else images
            }
            prompt["mm_processor_kwargs"] = kwargs
        reruns = 0
        for read_cache in (self.read_prefix_cache, False):
            request_id = f"rsijev-{uuid.uuid4().hex}"
            complete = False
            async with self.semaphore:
                try:
                    final = None
                    async for output in self.engine_client.encode(
                        prompt=prompt,
                        pooling_params=_pooling_params(read_cache),
                        request_id=request_id,
                    ):
                        final = output
                    if (
                        final is None
                        or not final.finished
                        or list(final.prompt_token_ids) != ids
                    ):
                        raise RuntimeError(
                            "vLLM did not return the expected RSI-Jev tokens"
                        )
                    hidden = torch.as_tensor(final.outputs.data)
                    offset = len(ids) - hidden.shape[0]
                    if hidden.ndim != 2 or offset < 0:
                        raise RuntimeError("vLLM returned invalid RSI-Jev token states")
                    complete = True
                finally:
                    if not complete:
                        with suppress(Exception):
                            await self.engine_client.abort(request_id)
            if offset <= need_from:
                # Copy the readout rows so the full response can be released.
                return (
                    hidden[need_from - offset :].clone(),
                    int(final.num_cached_tokens),
                    reruns,
                )
            if not read_cache:
                raise RuntimeError("vLLM returned partial states without a cache read")
            reruns += 1
        raise AssertionError("unreachable")

    async def systemone(self, payload):
        from .endpoint import _compile_serialized, _gather_cancel_on_error

        started = time.perf_counter()
        rows, mm = await _compile_serialized(self.compile_lock, self._compile, payload)
        salt = secrets.token_hex(16)
        needs = [spans[0][0] for _, _, _, _, _, spans, _ in rows]
        results = await _gather_cancel_on_error(
            self._states(ids, need, mm, salt)
            for (_, _, _, _, ids, _, _), need in zip(rows, needs)
        )
        readout = [
            (
                states,
                [(a - need, b - need) for a, b in spans],
                decision - need,
                kind,
            )
            for (_, kind, _, _, _, spans, decision), need, (states, _, _) in zip(
                rows, needs, results
            )
        ]
        probabilities = self.head(readout)
        if any(not math.isfinite(p) for row in probabilities for p in row):
            raise RuntimeError("RSI-Jev returned nonfinite probabilities")
        answers = {
            key: answer(kind, options, criteria, p)
            for (key, kind, options, criteria, _, _, _), p in zip(rows, probabilities)
        }
        return {
            "model": self.model_id,
            "answers": answers,
            "usage": {
                "input_tokens": sum(len(row[4]) for row in rows),
                "output_tokens": 0,
            },
            "metadata": {
                "cached_tokens": sum(cached for _, cached, _ in results),
                "uncached_reruns": sum(reruns for _, _, reruns in results),
                "inference_seconds": time.perf_counter() - started,
            },
        }


def _pooling_params(read_cache: bool):
    from vllm import PoolingParams

    return PoolingParams(
        task="token_embed",
        use_activation=False,
        skip_reading_prefix_cache=not read_cache,
    )
