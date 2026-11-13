"""Valen-Decision-1.5 decisions over native Qwen3.5 token states.

The backbone runs in vLLM and returns the final state of every token. A
request is one sequence: the state, a user turn with every question and its
candidates, and an assistant turn with one readout per question. The released
Mixer head scores each candidate from four pooled roles (state, instruction,
candidate, readout); the grounding head answers ``bbox`` questions over the
tokens of one image. Prompt, heads, and answers follow the release's
``compiler.py``, ``heads.py``, ``grounding.py`` and ``responses.py``; those
files are not executed.

This adapter is separate from the Valen preview adapter in ``valen.py``.
"""

import asyncio
import json
import math
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as functional
from safetensors.torch import load_file

from .decision_template import current_template
from .media import MAX_IMAGES_PER_REQUEST, load_image, validate_text
from .valen_decision_blocks import GLUE, PREFIX, block_segments, token_segments

PROTOCOL = "valen_decision_v1"
HEAD_FILE = "valen_decision_head.safetensors"
MANIFEST = "valen_decision_manifest.json"
NOUL_CANDIDATES = (
    ("true", "True / 是：满足问题中的条件。"),
    ("false", "False / 否：不满足问题中的条件。"),
)
MAX_QUESTIONS = 64
MAX_CHOICE_OPTIONS = 255
MAX_SCORE_LEVELS = 255


def question_options(question: dict) -> list[tuple[str, str]]:
    """Candidate keys and descriptions; a bbox question has none."""
    kind = question["type"]
    if kind == "bbox":
        return []
    if kind == "noul":
        return list(NOUL_CANDIDATES)
    if kind == "choice":
        return list(question["criteria"].items())
    return [(str(index), text) for index, text in enumerate(question["criteria"])]


def question_instructions(question: dict) -> str:
    """A Noul rubric is part of the instructions, as in training."""
    criteria = question.get("criteria")
    if question["type"] != "noul" or not criteria:
        return question["instructions"]
    return (
        question["instructions"]
        + "\nTrue criterion: "
        + str(criteria["true"])
        + "\nFalse criterion: "
        + str(criteria["false"])
    )


def validate_question(identifier, question, *, grounding: bool) -> None:
    if not isinstance(identifier, str) or not identifier:
        raise ValueError("question IDs must be nonempty text")
    if not isinstance(question, dict):
        raise ValueError(f"questions.{identifier}: definition must be an object")
    kind = question.get("type")
    if kind not in ("choice", "noul", "score", "bbox"):
        raise ValueError(
            f"questions.{identifier}: type must be choice, noul, score or bbox"
        )
    instructions = question.get("instructions")
    if not isinstance(instructions, str) or not instructions.strip():
        raise ValueError(f"questions.{identifier}: instructions must be nonempty text")
    criteria = question.get("criteria")
    if kind == "bbox":
        if not grounding:
            raise ValueError("this checkpoint has no grounding head for bbox questions")
        if criteria is not None:
            raise ValueError(f"questions.{identifier}: bbox takes no criteria")
        image = question.get("image", 0)
        if type(image) is not int or image < 0:
            raise ValueError(
                f"questions.{identifier}: image must be a nonnegative image index"
            )
        return
    if "image" in question:
        raise ValueError(f"questions.{identifier}: only bbox questions name an image")
    if kind == "noul":
        if criteria is not None and (
            not isinstance(criteria, dict)
            or set(criteria) != {"true", "false"}
            or any(not isinstance(v, str) or not v.strip() for v in criteria.values())
        ):
            raise ValueError(
                f"questions.{identifier}: Noul criteria must describe true and false"
            )
        return
    if kind == "choice":
        if not isinstance(criteria, dict) or not 1 <= len(criteria) <= (
            MAX_CHOICE_OPTIONS
        ):
            raise ValueError(
                f"questions.{identifier}: Choice needs 1 to {MAX_CHOICE_OPTIONS} options"
            )
    elif not isinstance(criteria, list) or not 2 <= len(criteria) <= MAX_SCORE_LEVELS:
        raise ValueError(
            f"questions.{identifier}: Score needs 2 to {MAX_SCORE_LEVELS} levels"
        )
    if any(
        not isinstance(key, str)
        or not key
        or not isinstance(description, str)
        or not description.strip()
        for key, description in question_options(question)
    ):
        raise ValueError(
            f"questions.{identifier}: candidate names and descriptions must be "
            "nonempty text"
        )


@dataclass(frozen=True)
class ImageTokens:
    """One image's contiguous tokens and merged (rows, cols) grid."""

    start: int
    rows: int
    cols: int


@dataclass(frozen=True)
class EncodedQuestion:
    question_id: str
    kind: str
    instruction: tuple[int, int]
    candidates: tuple[tuple[int, int], ...]
    keys: tuple[str, ...]
    descriptions: tuple[str, ...]
    decision: int
    image: ImageTokens | None = None


def encode(
    tokenizer,
    base_ids: list[int],
    questions: dict,
    images: list[ImageTokens],
    max_length: int,
):
    """Token ids, role spans, and block ids, as the release's shared compiler.

    ``base_ids`` is the processor's rendering of the state with every image
    expanded. Returns the complete ids, one ``EncodedQuestion`` per question,
    and the compiler's block id of every appended token: turn markers 0,
    question i as i + 1.
    """
    ids = list(base_ids)
    segments: list[int] = []

    def append(text, span=None, segment=0):
        start = len(ids)
        if span is None:
            ids.extend(tokenizer.encode(text, add_special_tokens=False))
            segments.extend([segment] * (len(ids) - start))
            return None
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        ids.extend(encoded["input_ids"])
        segments.extend([segment] * (len(ids) - start))
        indices = [
            index
            for index, (a, b) in enumerate(encoded["offset_mapping"])
            if a < span[1] and b > span[0]
        ]
        if not indices:
            raise ValueError("empty token span for a Valen decision role")
        return start + indices[0], start + indices[-1] + 1

    append(GLUE[0])
    pending = []
    for index, (identifier, question) in enumerate(questions.items()):
        block = index + 1
        kind = question["type"]
        instructions = question_instructions(question)
        options = question_options(question)
        prefix = f"Question {index + 1}\nTask: {kind}\nQuestion: "
        span = (len(prefix), len(prefix) + len(instructions))
        if kind == "bbox":
            image = question.get("image", 0)
            if image >= len(images):
                raise ValueError(
                    f"questions.{identifier}: bbox refers to image {image}, "
                    f"the state has {len(images)}"
                )
            instruction = append(
                prefix + instructions + f"\nImage: {image + 1}\n", span, block
            )
            pending.append((identifier, kind, instruction, (), options, images[image]))
            continue
        instruction = append(prefix + instructions + "\nCandidates:\n", span, block)
        spans = []
        for key, description in options:
            lead = key + ": " if kind != "score" else ""
            spans.append(
                append(
                    lead + description, (len(lead), len(lead) + len(description)), block
                )
            )
            append("\n", segment=block)
        pending.append((identifier, kind, instruction, tuple(spans), options, None))
    append(GLUE[1])
    encoded = []
    for index, (identifier, kind, instruction, spans, options, image) in enumerate(
        pending
    ):
        append(f"Question {index + 1} Decision:", segment=index + 1)
        encoded.append(
            EncodedQuestion(
                identifier,
                kind,
                instruction,
                spans,
                tuple(key for key, _ in options),
                tuple(description for _, description in options),
                len(ids) - 1,
                image,
            )
        )
        append("\n", segment=index + 1)
    if len(ids) > max_length:
        raise ValueError(
            f"Valen prompt has {len(ids)} tokens; the maximum is {max_length} "
            "(no truncation)"
        )
    return ids, encoded, segments


class MixerBlock(torch.nn.Module):
    def __init__(self, width: int, token_hidden: int, channel_hidden: int):
        super().__init__()
        self.token_norm = torch.nn.LayerNorm(width)
        self.token_mlp = torch.nn.Sequential(
            torch.nn.Linear(4, token_hidden),
            torch.nn.GELU(),
            torch.nn.Linear(token_hidden, 4),
        )
        self.channel_norm = torch.nn.LayerNorm(width)
        self.channel_mlp = torch.nn.Sequential(
            torch.nn.Linear(width, channel_hidden),
            torch.nn.GELU(),
            torch.nn.Linear(channel_hidden, width),
        )

    def forward(self, x):
        # Mix the four roles of one candidate, never different candidates.
        x = x + self.token_mlp(self.token_norm(x).transpose(1, 2)).transpose(1, 2)
        return x + self.channel_mlp(self.channel_norm(x))


class MixerHead(torch.nn.Module):
    """The released FP32 role Mixer; parameter names match its weights."""

    def __init__(
        self,
        hidden_size: int,
        head_width: int = 256,
        head_layers: int = 2,
        head_token_hidden_dim: int = 16,
        head_channel_hidden_dim: int = 1024,
    ):
        super().__init__()
        self.input_norm = torch.nn.LayerNorm(hidden_size)
        self.projection = torch.nn.Linear(hidden_size, head_width)
        self.roles = torch.nn.Parameter(torch.zeros(1, 4, head_width))
        self.blocks = torch.nn.Sequential(
            *(
                MixerBlock(head_width, head_token_hidden_dim, head_channel_hidden_dim)
                for _ in range(head_layers)
            )
        )
        self.output = torch.nn.Sequential(
            torch.nn.LayerNorm(head_width), torch.nn.Linear(head_width, 1)
        )

    def forward(self, features):
        """``features`` is [candidates, 4 roles, hidden]; returns one logit each."""
        x = self.projection(self.input_norm(features.float())) + self.roles
        # The third row is the candidate role.
        return self.output(self.blocks(x)[:, 2]).squeeze(-1)


def role_features(hidden, context_end: int, question: EncodedQuestion):
    """Pooled state, instruction, candidate and readout states per candidate."""

    def pool(span):
        start, end = span
        if not 0 <= start < end <= hidden.shape[0]:
            raise ValueError(f"invalid Valen role span: {span}")
        return hidden[start:end].float().mean(0)

    context, instruction = pool((0, context_end)), pool(question.instruction)
    decision = hidden[question.decision].float()
    return torch.stack(
        [
            torch.stack((context, instruction, pool(span), decision))
            for span in question.candidates
        ]
    )


def cell_centres(grid, device):
    rows, cols = grid
    y = (torch.arange(rows, device=device, dtype=torch.float32) + 0.5) / rows
    x = (torch.arange(cols, device=device, dtype=torch.float32) + 0.5) / cols
    cy, cx = torch.meshgrid(y, x, indexing="ij")
    return cx.reshape(-1), cy.reshape(-1)


class GroundingHead(torch.nn.Module):
    """The released FP32 grounding head; parameter names match its weights.

    The readout scores every token of the image, and every token regresses
    the distances from its cell centre to the four box edges.
    """

    def __init__(self, hidden_size: int, width: int = 512, hidden: int = 1024):
        super().__init__()
        self.decision_norm = torch.nn.LayerNorm(hidden_size)
        self.token_norm = torch.nn.LayerNorm(hidden_size)
        self.query = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden),
            torch.nn.GELU(),
            torch.nn.Linear(hidden, width),
        )
        self.key = torch.nn.Linear(hidden_size, width)
        self.gate = torch.nn.Linear(hidden_size, width)
        self.regress = torch.nn.Sequential(
            torch.nn.LayerNorm(width),
            torch.nn.Linear(width, hidden),
            torch.nn.GELU(),
            torch.nn.Linear(hidden, 4),
        )
        self.scale = math.sqrt(width)

    def forward(self, decision, tokens, grid):
        """Token logits [N] and normalized x1, y1, x2, y2 boxes [N, 4]."""
        rows, cols = grid
        if tokens.shape[0] != rows * cols:
            raise ValueError("image tokens do not match their grid")
        d = self.decision_norm(decision.float())
        k = self.key(self.token_norm(tokens.float()))
        logits = (k @ self.query(d)) / self.scale
        # Query-conditioned regression of edge distances in cell units.
        distances = functional.softplus(self.regress(k * torch.sigmoid(self.gate(d))))
        distances = distances + 0.5
        cx, cy = cell_centres(grid, logits.device)
        d = distances * logits.new_tensor([1 / cols, 1 / rows, 1 / cols, 1 / rows])
        boxes = torch.stack(
            (cx - d[:, 0], cy - d[:, 1], cx + d[:, 2], cy + d[:, 3]), -1
        ).clamp(0, 1)
        return logits, boxes


def decode_box(logits, boxes, grid) -> dict:
    """The box of the most probable token, its centre, and its confidence."""
    probabilities = logits.float().softmax(-1)
    index = int(probabilities.argmax())
    box = boxes[index].float()
    cx, cy = cell_centres(grid, box.device)
    inside = (cx >= box[0]) & (cx <= box[2]) & (cy >= box[1]) & (cy <= box[3])
    # Confidence is the probability mass of tokens centred inside the box.
    confidence = (
        float(probabilities[inside].sum())
        if bool(inside.any())
        else float(probabilities[index])
    )
    x1, y1, x2, y2 = (round(float(value), 4) for value in box)
    return {
        "type": "bbox",
        "bbox": [x1, y1, x2, y2],
        "point": [round((x1 + x2) / 2, 4), round((y1 + y2) / 2, 4)],
        "confidence": round(min(max(confidence, 0.0), 1.0), 4),
    }


def answer(question: EncodedQuestion, logits: list[float]) -> dict:
    """Probabilities and summary values, as the release's ``responses.answer``."""
    if len(logits) != len(question.keys) or not all(map(math.isfinite, logits)):
        raise RuntimeError("Valen returned invalid decision scores")
    scores = torch.tensor(logits, dtype=torch.float64)
    probabilities = (scores - scores.max()).softmax(-1).tolist()
    total = sum(probabilities)
    probabilities = [p / total for p in probabilities]
    result = {"type": question.kind}
    if question.kind == "noul":
        result["noul"] = probabilities[question.keys.index("true")]
        return result
    result["probabilities"] = dict(zip(question.keys, probabilities))
    count = len(probabilities)
    mode = max(range(count), key=probabilities.__getitem__)
    if question.kind == "choice":
        result["choice"] = question.keys[mode]
        result["confidence"] = (
            1.0
            if count == 1
            else max(0.0, (probabilities[mode] - 1 / count) / (1 - 1 / count))
        )
        return result
    result["score"] = sum(index * p for index, p in enumerate(probabilities))
    result["legend"] = dict(zip(question.keys, question.descriptions))
    distance = sum(p * abs(index - mode) for index, p in enumerate(probabilities))
    uniform = sum(abs(index - (count - 1) / 2) for index in range(count)) / count
    result["confidence"] = max(0.0, 1.0 - distance / uniform)
    return result


class _Heads(torch.nn.Module):
    """Holds the heads under the names of the released state dict."""

    def __init__(self, hidden_size: int, head: dict, grounding: dict | None):
        super().__init__()
        self.head = MixerHead(hidden_size, **head)
        if grounding:
            self.grounding = GroundingHead(hidden_size, **grounding)


class ValenDecisionService:
    def __init__(
        self,
        engine_client,
        model_path: Path,
        model_id: str | list[str],
        max_length: int,
        max_inflight: int = 128,
    ):
        from transformers import AutoProcessor

        from .backend import readout_device

        manifest = json.loads((model_path / MANIFEST).read_text())
        self.engine_client = engine_client
        self.processor = AutoProcessor.from_pretrained(
            model_path, local_files_only=True
        )
        self.tokenizer = self.processor.tokenizer
        self.image_pad_id = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        self.merge_size = self.processor.image_processor.merge_size
        self.special_tokens = set(self.tokenizer.all_special_tokens) | {
            str(token)
            for token in self.tokenizer.added_tokens_decoder.values()
            if token.special
        }
        self.glue = [self._encode_text(text) for text in GLUE]
        self.mask = manifest["full_attention_mask"]
        self.media_kwargs = manifest["media_kwargs"]
        self.max_length = min(max_length, manifest["max_length"])
        self.grounding = bool(manifest["grounding"])
        self.device = readout_device()
        self.heads = _Heads(
            manifest["hidden_size"], manifest["head"], manifest["grounding"]
        )
        self.heads.load_state_dict(load_file(str(model_path / HEAD_FILE)), strict=True)
        self.heads.to(self.device, torch.float32).eval()
        self.model_names = [model_id] if isinstance(model_id, str) else model_id
        self.model_id = self.model_names[0]
        self.semaphore = asyncio.Semaphore(max_inflight)
        self.compile_lock = asyncio.Lock()
        # One thread runs the heads so requests do not block the event loop.
        self.head_executor = ThreadPoolExecutor(
            1, thread_name_prefix="valen-decision-head"
        )

    def _encode_text(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _text(self, value: str) -> str:
        # User text must not inject Qwen control tokens.
        validate_text(value)
        if any(token in value for token in self.special_tokens):
            raise ValueError("text must not contain reserved tokenizer control tokens")
        return value

    def _state(self, state):
        """Rendered state ids with images expanded, the images, and their tokens."""
        if isinstance(state, str):
            messages = [{"role": "user", "content": [{"type": "text", "text": state}]}]
        elif isinstance(state, dict) and isinstance(state.get("messages"), list):
            messages = state["messages"]
            if set(state) != {"messages"} or not messages:
                raise ValueError("Valen state.messages must be a nonempty list")
        else:
            raise ValueError("Valen state must be text or a messages object")
        rendered, images = [], []
        for message in messages:
            if not isinstance(message, dict) or message.get("role") not in (
                "user",
                "assistant",
                "system",
            ):
                raise ValueError("Valen messages need a user, assistant or system role")
            content = message.get("content")
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            if not isinstance(content, list) or not content:
                raise ValueError(
                    "Valen message content must be text or a nonempty list"
                )
            parts = []
            for item in content:
                if not isinstance(item, dict):
                    raise ValueError("Valen content items must be objects")
                if item.get("type") == "text":
                    if not isinstance(item.get("text"), str):
                        raise ValueError("Valen text content must be a string")
                    parts.append({"type": "text", "text": self._text(item["text"])})
                elif item.get("type") == "image_url":
                    if len(images) >= MAX_IMAGES_PER_REQUEST:
                        raise ValueError(
                            f"Valen accepts at most {MAX_IMAGES_PER_REQUEST} images"
                        )
                    source = item.get("image_url")
                    if not isinstance(source, dict):
                        raise ValueError("Valen image_url must contain a data URL")
                    images.append(load_image(source.get("url")))
                    parts.append({"type": "image", "image": images[-1]})
                else:
                    raise ValueError(
                        "Valen-Decision-1.5 supports text and image_url content"
                    )
            rendered.append({"role": message["role"], "content": parts})
        base = self.processor.apply_chat_template(
            rendered,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs=dict(self.media_kwargs, return_metadata=True),
        )
        ids = base["input_ids"].flatten().tolist()
        tokens, cursor = [], 0
        grids = base["image_grid_thw"].tolist() if images else []
        positions = [
            index for index, token in enumerate(ids) if token == self.image_pad_id
        ]
        for frames, height, width in grids:
            rows, cols = height // self.merge_size, width // self.merge_size
            count = frames * rows * cols
            run = positions[cursor : cursor + count]
            if len(run) != count or run != list(range(run[0], run[0] + count)):
                raise ValueError("image tokens do not match the processor grid")
            tokens.append(ImageTokens(run[0], rows, cols))
            cursor += count
        if cursor != len(positions) or len(tokens) != len(images):
            raise ValueError("image token count does not match the images")
        return ids, images, tokens

    def _compile(self, payload):
        if payload.model not in (None, "vllm-jev", *self.model_names):
            raise ValueError("requested model is not loaded")
        if current_template().config not in ({}, {"layout": "native"}):
            raise ValueError("Valen-Decision-1.5 supports only its native template")
        questions = payload.questions
        if not isinstance(questions, dict) or not 1 <= len(questions) <= MAX_QUESTIONS:
            raise ValueError(f"System One requires 1 to {MAX_QUESTIONS} questions")
        for identifier, question in questions.items():
            validate_question(identifier, question, grounding=self.grounding)
            self._text(question_instructions(question))
            for key, description in question_options(question):
                self._text(key)
                self._text(description)
        base_ids, images, image_tokens = self._state(payload.state)
        ids, encoded, segments = encode(
            self.tokenizer, base_ids, questions, image_tokens, self.max_length
        )
        # vLLM reads the blocks back from the prompt's tokens; both must agree.
        bbox = {index + 1 for index, q in enumerate(encoded) if q.kind == "bbox"}
        blocks = [PREFIX] * len(base_ids) + block_segments(self.mask, segments, bbox)
        if (
            blocks
            != token_segments(
                torch.tensor(ids),
                self.glue,
                self.mask == "block_bidirectional",
                self._encode_text,
            ).tolist()
        ):
            raise ValueError("question text must not repeat a Valen question header")
        # vLLM expands each image placeholder again.
        collapsed = [
            token
            for index, token in enumerate(ids)
            if token != self.image_pad_id or index == 0 or ids[index - 1] != token
        ]
        prompt = {"prompt_token_ids": collapsed}
        if images:
            prompt["multi_modal_data"] = {
                "image": images[0] if len(images) == 1 else images
            }
        return ids, len(base_ids), encoded, prompt

    async def _states(self, ids, prompt):
        request_id = f"valen-decision-{uuid.uuid4().hex}"
        complete = False
        async with self.semaphore:
            try:
                final = None
                async for output in self.engine_client.encode(
                    prompt=prompt,
                    pooling_params=_pooling_params(),
                    request_id=request_id,
                ):
                    final = output
                if final is None or not final.finished:
                    raise RuntimeError("vLLM returned no final pooling result")
                # The heads read states at the compiler's token positions, so
                # vLLM must have run exactly this sequence in one prefill.
                if list(final.prompt_token_ids) != ids:
                    raise RuntimeError(
                        "vLLM tokenization differs from Valen's encoding"
                    )
                hidden = torch.as_tensor(final.outputs.data)
                if hidden.ndim != 2 or hidden.shape[0] != len(ids):
                    raise RuntimeError("vLLM returned incomplete Valen token states")
                complete = True
                return hidden
            finally:
                if not complete:
                    with suppress(Exception):
                        await self.engine_client.abort(request_id)

    @torch.no_grad()
    def _readout(self, hidden, context_end, encoded):
        hidden = hidden.to(self.device)
        answers = {}
        for question in encoded:
            if question.kind == "bbox":
                image = question.image
                grid = (image.rows, image.cols)
                logits, boxes = self.heads.grounding(
                    hidden[question.decision],
                    hidden[image.start : image.start + image.rows * image.cols],
                    grid,
                )
                if not torch.isfinite(logits).all() or not torch.isfinite(boxes).all():
                    raise RuntimeError("Valen returned a nonfinite bbox readout")
                answers[question.question_id] = decode_box(logits, boxes, grid)
                continue
            logits = self.heads.head(role_features(hidden, context_end, question))
            answers[question.question_id] = answer(question, logits.tolist())
        return answers

    async def systemone(self, payload):
        from .endpoint import _compile_serialized

        started = time.perf_counter()
        ids, context_end, encoded, prompt = await _compile_serialized(
            self.compile_lock, self._compile, payload
        )
        hidden = await self._states(ids, prompt)
        answers = await asyncio.get_running_loop().run_in_executor(
            self.head_executor, self._readout, hidden, context_end, encoded
        )
        return {
            "model": self.model_id,
            "answers": answers,
            "usage": {"input_tokens": len(ids), "output_tokens": 0},
            "metadata": {"inference_seconds": time.perf_counter() - started},
        }


def _pooling_params():
    from vllm import PoolingParams

    return PoolingParams(task="token_embed", use_activation=False)
