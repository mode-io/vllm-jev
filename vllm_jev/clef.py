"""Clef decisions over native Qwen3.5 token states.

The backbone runs in vLLM and returns the final state of every token. A
request is one sequence: the system prompt, the media, the state, one block
per question with its allowed options, and the decision cue. The released
joint schema head reads all of those states and scores every option of every
question at once. Encoding, head, and answers follow the published
``joint_schema_model.py`` of the Clef release; that file is not executed.
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
from safetensors import safe_open
from safetensors.torch import load_file

from .decision_template import current_template
from .media import MAX_IMAGES_PER_REQUEST, load_image, validate_text
from .video import load_video

PROTOCOL = "clef_joint_v1"
SYSTEM_PROMPT = (
    "Read the complete state and schema. Decide every field jointly. Each answer "
    "must be exactly one of that field's allowed options."
)
IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"
VIDEO_PLACEHOLDER = "<|vision_start|><|video_pad|><|vision_end|>"
QUESTION_TYPES = {"noul": 0, "choice": 1, "score": 2}
NOUL_CRITERIA = {
    "true": "The proposition is true or the answer is yes.",
    "false": "The proposition is false or the answer is no.",
}
MAX_QUESTIONS = 64
MAX_CHOICE_OPTIONS = 255
MAX_SCORE_LEVELS = 10
MAX_TOTAL_OPTIONS = 2048
MAX_SCHEMA_BYTES = 1024 * 1024
MAX_VIDEOS = 1


def render(value) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def question_options(question: dict) -> list[tuple[str, object]]:
    kind = question["type"]
    if kind == "noul":
        criteria = dict(NOUL_CRITERIA)
        criteria.update(question.get("criteria") or {})
        return [(key, criteria[key]) for key in ("true", "false")]
    if kind == "choice":
        return sorted((str(key), value) for key, value in question["criteria"].items())
    return [(str(index), value) for index, value in enumerate(question["criteria"])]


def validate_question(identifier: str, question) -> None:
    if not identifier or not isinstance(question, dict):
        raise ValueError("question IDs and definitions must be nonempty")
    kind = question.get("type")
    if not isinstance(kind, str) or kind not in QUESTION_TYPES:
        raise ValueError(f"questions.{identifier}: type must be noul, choice or score")
    instructions = question.get("instructions")
    if instructions is not None and not isinstance(instructions, str):
        raise ValueError(f"questions.{identifier}: instructions must be text")
    criteria = question.get("criteria")
    if kind == "choice":
        if not isinstance(criteria, dict) or not 1 <= len(criteria) <= (
            MAX_CHOICE_OPTIONS
        ):
            raise ValueError(
                f"questions.{identifier}: Choice needs 1 to {MAX_CHOICE_OPTIONS} options"
            )
        if any(not option for option in criteria):
            raise ValueError(f"questions.{identifier}: Choice labels must be nonempty")
    elif kind == "score":
        if not isinstance(criteria, list) or not 2 <= len(criteria) <= (
            MAX_SCORE_LEVELS
        ):
            raise ValueError(
                f"questions.{identifier}: Score needs 2 to {MAX_SCORE_LEVELS} levels"
            )
    elif criteria is not None and (
        not isinstance(criteria, dict) or set(criteria) - set(NOUL_CRITERIA)
    ):
        raise ValueError(
            f"questions.{identifier}: Noul criteria may set true and false"
        )


@dataclass(frozen=True)
class EncodedQuestion:
    question_id: str
    question_type: int
    question_span: tuple[int, int]
    option_spans: tuple[tuple[int, int], ...]
    option_ids: tuple[str, ...]


def encode(tokenizer, questions: dict, state, media_ids: list[int], max_length: int):
    """Token ids and spans, exactly as the release's ``encode_record``.

    ``media_ids`` is the processor's expansion of the media placeholders. A
    state that does not fit loses tokens from its end; the schema is kept.
    """

    def tokens(text: str) -> list[int]:
        return tokenizer(text, add_special_tokens=False).input_ids

    schema = tokens("\n\nSCHEMA FIELDS:\n")
    encoded = []
    for index, (identifier, question) in enumerate(questions.items()):
        schema += tokens(
            f"\nFIELD {index + 1}\nID: {identifier}\nTYPE: {question['type']}"
            "\nINSTRUCTION: "
        )
        start = len(schema)
        instructions = question.get("instructions")
        schema += tokens(render(instructions or str(identifier)))
        question_span = (start, len(schema))
        schema += tokens("\nALLOWED OPTIONS:\n")
        spans, option_ids = [], []
        for number, (option, description) in enumerate(question_options(question)):
            schema += tokens(f"OPTION {number + 1}: ")
            start = len(schema)
            semantics = {"option_id": option}
            if description is not None:
                semantics["description"] = description
            schema += tokens(render(semantics))
            spans.append((start, len(schema)))
            option_ids.append(option)
            schema += tokens("\n")
        schema += tokens("END FIELD\n")
        encoded.append(
            EncodedQuestion(
                str(identifier),
                QUESTION_TYPES[question["type"]],
                question_span,
                tuple(spans),
                tuple(option_ids),
            )
        )
    prefix = tokens(
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nSTATE:\n"
    )
    suffix = tokens(
        "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        "JOINT SCHEMA DECISIONS:"
    )
    if current_template().instructions_first:
        # The head was trained on state-conditioned schema representations.
        # Keep that schema after the evidence, even with an earlier preamble.
        instructions = "\n".join(
            f"FIELD {index + 1}: {render(question.get('instructions') or str(identifier))}"
            for index, (identifier, question) in enumerate(questions.items())
        )
        prefix = tokens(
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n"
            f"INSTRUCTIONS:\n{instructions}\n\nSTATE:\n"
        )
    prefix += media_ids
    fixed = len(prefix) + len(schema) + len(suffix)
    if fixed > max_length:
        raise ValueError(
            f"schema and media require {fixed} tokens; the maximum is {max_length}"
        )
    state_ids = tokens(render(state))[: max_length - fixed]
    offset = len(prefix) + len(state_ids)
    shifted = tuple(
        EncodedQuestion(
            question.question_id,
            question.question_type,
            (question.question_span[0] + offset, question.question_span[1] + offset),
            tuple((a + offset, b + offset) for a, b in question.option_spans),
            question.option_ids,
        )
        for question in encoded
    )
    return prefix + state_ids + schema + suffix, shifted


@dataclass(frozen=True)
class VisionTokens:
    image_pad: int
    video_pad: int
    vision_start: int
    vision_end: int


def collapse_media(ids: list[int], vision: VisionTokens) -> list[int]:
    """Undo the processor's placeholder expansion so vLLM can expand it again.

    An image keeps one ``<|image_pad|>``. A Qwen3-VL video expands to nested
    ``<|vision_start|>`` groups with timestamps; it collapses back to
    ``<|vision_start|><|video_pad|><|vision_end|>``.
    """
    collapsed = []
    index, length = 0, len(ids)
    while index < length:
        token = ids[index]
        collapsed.append(token)
        index += 1
        if token == vision.image_pad:
            while index < length and ids[index] == vision.image_pad:
                index += 1
        elif (
            token == vision.vision_start
            and index < length
            and ids[index] != vision.image_pad
        ):
            depth = 1
            while index < length and depth:
                if ids[index] == vision.vision_start:
                    depth += 1
                elif ids[index] == vision.vision_end:
                    depth -= 1
                index += 1
            collapsed += [vision.video_pad, vision.vision_end]
    return collapsed


class EvidenceRoutingLayer(torch.nn.Module):
    def __init__(self, width: int, heads: int, feedforward: int):
        super().__init__()
        self.query_norm = torch.nn.LayerNorm(width)
        self.memory_norm = torch.nn.LayerNorm(width)
        self.attention = torch.nn.MultiheadAttention(width, heads, batch_first=True)
        self.feedforward_norm = torch.nn.LayerNorm(width)
        self.feedforward = torch.nn.Sequential(
            torch.nn.Linear(width, feedforward),
            torch.nn.GELU(),
            torch.nn.Dropout(0.0),
            torch.nn.Linear(feedforward, width),
            torch.nn.Dropout(0.0),
        )

    def forward(self, queries, memory):
        memory = self.memory_norm(memory)
        routed, _ = self.attention(
            self.query_norm(queries), memory, memory, need_weights=False
        )
        queries = queries + routed
        return queries + self.feedforward(self.feedforward_norm(queries))


class ClefHead(torch.nn.Module):
    """The released joint schema head; parameter names match its weights."""

    def __init__(
        self,
        hidden_size: int,
        width: int,
        routing_layers: int,
        layers: int,
        heads: int,
        feedforward: int,
    ):
        super().__init__()
        self.hidden_norm = torch.nn.LayerNorm(hidden_size)
        for name in (
            "memory",
            "question",
            "option_question",
            "global",
            "option_context",
            "option_lexical",
        ):
            setattr(
                self,
                f"{name}_projection",
                torch.nn.Linear(hidden_size, width, bias=False),
            )
        self.type_embedding = torch.nn.Embedding(3, width)
        self.evidence_layers = torch.nn.ModuleList(
            EvidenceRoutingLayer(width, heads, feedforward)
            for _ in range(routing_layers)
        )
        self.option_summary_norm = torch.nn.LayerNorm(width)
        self.layers = torch.nn.ModuleList(
            torch.nn.TransformerDecoderLayer(
                d_model=width,
                nhead=heads,
                dim_feedforward=feedforward,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(layers)
        )
        self.field_norm = torch.nn.LayerNorm(width)
        self.option_norm = torch.nn.LayerNorm(width)
        self.residual_scorer = torch.nn.Sequential(
            torch.nn.Linear(width * 4, width),
            torch.nn.GELU(),
            torch.nn.Dropout(0.0),
            torch.nn.Linear(width, 1),
        )
        self.prior_logit_scale = torch.nn.Parameter(torch.zeros(()))
        self.joint_logit_scale = torch.nn.Parameter(torch.zeros(()))
        self.residual_gate = torch.nn.Parameter(torch.zeros(()))

    @torch.inference_mode()
    def forward(self, hidden, questions, lexical):
        """Logits per question for one sequence.

        ``hidden`` holds the backbone's final states; ``lexical`` holds, per
        question, the mean output embedding of each option's tokens.
        """
        hidden = self.hidden_norm(hidden)
        memory = self.memory_projection(hidden).unsqueeze(0)
        global_vector = hidden[-1]
        question_vectors = torch.stack(
            [hidden[slice(*question.question_span)].mean(0) for question in questions]
        )
        types = torch.tensor(
            [question.question_type for question in questions], device=hidden.device
        )
        queries = []
        for index, question in enumerate(questions):
            context = torch.stack(
                [hidden[start:end].mean(0) for start, end in question.option_spans]
            )
            queries.append(
                self.option_context_projection(context)
                + self.option_lexical_projection(lexical[index])
                + self.option_question_projection(question_vectors[index]).unsqueeze(0)
            )
        routed = torch.cat(queries).unsqueeze(0)
        for layer in self.evidence_layers:
            routed = layer(routed, memory)
        options_per_question = torch.split(
            routed[0], [len(question.option_spans) for question in questions]
        )
        fields = self.question_projection(question_vectors)
        summaries = []
        for field, options in zip(fields, options_per_question):
            weights = torch.softmax(options @ field / math.sqrt(options.shape[-1]), 0)
            summaries.append((weights.unsqueeze(-1) * options).sum(0))
        fields = (
            fields
            + self.option_summary_norm(torch.stack(summaries))
            + self.global_projection(global_vector).unsqueeze(0)
            + self.type_embedding(types)
        ).unsqueeze(0)
        for layer in self.layers:
            fields = layer(fields, memory)
        fields = self.field_norm(fields[0])
        prior_scale = self.prior_logit_scale.clamp(max=math.log(100.0)).exp()
        joint_scale = self.joint_logit_scale.clamp(max=math.log(100.0)).exp()
        gate = torch.sigmoid(self.residual_gate)
        logits = []
        for index, (field, options) in enumerate(zip(fields, options_per_question)):
            anchor = functional.normalize(
                question_vectors[index] + global_vector, dim=-1
            )
            prior = prior_scale * (
                functional.normalize(lexical[index], dim=-1) @ anchor
            )
            options = self.option_norm(options)
            field = field.unsqueeze(0).expand_as(options)
            cosine = functional.cosine_similarity(field, options, dim=-1)
            features = torch.cat(
                [field, options, field * options, (field - options).abs()], -1
            )
            residual = self.residual_scorer(features).squeeze(-1)
            logits.append(prior + gate * (joint_scale * cosine + residual))
        return logits


def answer(question: dict, probabilities: dict[str, float]) -> dict:
    """The release's System One answer for one question."""
    if question["type"] == "noul":
        return {"type": "noul", "noul": round(probabilities["true"], 4)}
    if question["type"] == "choice":
        options = [str(option) for option in question["criteria"]]
        choice = max(options, key=probabilities.__getitem__)
        return {
            "type": "choice",
            "choice": choice,
            "confidence": round(probabilities[choice], 4),
            "probabilities": {
                option: round(probabilities[option], 4) for option in options
            },
        }
    levels = [str(index) for index in range(len(question["criteria"]))]
    return {
        "type": "score",
        "score": round(
            sum(index * probabilities[level] for index, level in enumerate(levels)), 4
        ),
        "confidence": round(max(probabilities[level] for level in levels), 4),
        "legend": dict(zip(levels, question["criteria"])),
        "probabilities": {level: round(probabilities[level], 4) for level in levels},
    }


def _video(item):
    if isinstance(item, str):
        return load_video(item)
    if isinstance(item, dict) and set(item) <= {"url", "num_frames"}:
        return load_video(item.get("url"), item.get("num_frames", 8))
    raise ValueError("videos must be MP4 data URLs or {url, num_frames} objects")


class ClefService:
    def __init__(
        self,
        engine_client,
        model_path: Path,
        model_id: str | list[str],
        max_length: int,
        max_inflight: int = 128,
    ):
        from transformers import AutoProcessor

        self.engine_client = engine_client
        self.processor = AutoProcessor.from_pretrained(
            model_path, local_files_only=True
        )
        self.tokenizer = self.processor.tokenizer
        token = self.tokenizer.convert_tokens_to_ids
        self.vision = VisionTokens(
            token("<|image_pad|>"),
            token("<|video_pad|>"),
            token("<|vision_start|>"),
            token("<|vision_end|>"),
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config = json.loads((model_path / "joint_head_config.json").read_text())
        self.head = ClefHead(**config)
        self.head.load_state_dict(
            load_file(str(model_path / "joint_head.safetensors")), strict=True
        )
        self.head.to(self.device, torch.bfloat16).eval()
        # Option tokens read rows of the output embedding, which vLLM's pooling
        # model does not load. Only the rows a request uses move to the GPU.
        index = json.loads((model_path / "model.safetensors.index.json").read_text())
        with safe_open(
            model_path / index["weight_map"]["lm_head.weight"],
            framework="pt",
            device="cpu",
        ) as file:
            self.output_embedding = file.get_tensor("lm_head.weight")
        self.max_length = max_length
        self.model_names = [model_id] if isinstance(model_id, str) else model_id
        self.model_id = self.model_names[0]
        self.semaphore = asyncio.Semaphore(max_inflight)
        self.compile_lock = asyncio.Lock()
        # One thread runs the head so requests do not block the event loop.
        self.head_executor = ThreadPoolExecutor(1, thread_name_prefix="clef-head")

    def _media(self, payload):
        images = payload.images or []
        videos = payload.videos or []
        if not isinstance(images, list) or not isinstance(videos, list):
            raise ValueError("images and videos must be lists")
        if len(images) > MAX_IMAGES_PER_REQUEST:
            raise ValueError(f"Clef accepts at most {MAX_IMAGES_PER_REQUEST} images")
        if len(videos) > MAX_VIDEOS:
            raise ValueError(f"Clef accepts at most {MAX_VIDEOS} video")
        images = [load_image(image) for image in images]
        videos = [_video(video) for video in videos]
        if not images and not videos:
            return [], None
        kwargs = {}
        if videos:
            # Frames are already sampled; the processor would otherwise assume
            # 24 fps and resample each clip again.
            kwargs = {
                "do_sample_frames": False,
                "video_metadata": [dict(video.metadata) for video in videos],
            }
        encoded = self.processor(
            text=[
                IMAGE_PLACEHOLDER * len(images) + VIDEO_PLACEHOLDER * len(videos) + "\n"
            ],
            images=images or None,
            videos=[video.frames for video in videos] or None,
            return_tensors="pt",
            **kwargs,
        )
        data = {}
        if images:
            data["image"] = images[0] if len(images) == 1 else images
        if videos:
            data["video"] = (
                videos[0].frames,
                dict(videos[0].metadata, do_sample_frames=False),
            )
        return encoded["input_ids"][0].tolist(), data

    def _compile(self, payload):
        if payload.model not in (None, "vllm-jev", *self.model_names):
            raise ValueError("requested model is not loaded")
        questions = payload.questions
        if not isinstance(questions, dict) or not 1 <= len(questions) <= MAX_QUESTIONS:
            raise ValueError(f"System One requires 1 to {MAX_QUESTIONS} questions")
        total_options = 0
        schema_bytes = 0
        for identifier, question in questions.items():
            validate_question(identifier, question)
            total_options += (
                2 if question["type"] == "noul" else len(question["criteria"])
            )
            if total_options > MAX_TOTAL_OPTIONS:
                raise ValueError(
                    f"System One accepts at most {MAX_TOTAL_OPTIONS} options per request"
                )
            rendered = render(question)
            schema_bytes += len(identifier.encode("utf-8")) + len(
                rendered.encode("utf-8")
            )
            if schema_bytes > MAX_SCHEMA_BYTES:
                raise ValueError(
                    "question IDs and definitions exceed the 1 MiB schema limit"
                )
            validate_text(identifier)
            validate_text(rendered)
        validate_text(render(payload.state))
        media_ids, data = self._media(payload)
        ids, encoded = encode(
            self.tokenizer, questions, payload.state, media_ids, self.max_length
        )
        rows = sorted(
            {
                token
                for question in encoded
                for start, end in question.option_spans
                for token in ids[start:end]
            }
        )
        prompt = {"prompt_token_ids": collapse_media(ids, self.vision)}
        if data:
            prompt["multi_modal_data"] = data
        return ids, encoded, prompt, rows

    async def _states(self, ids, prompt):
        request_id = f"clef-{uuid.uuid4().hex}"
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
                # The head reads states at Clef's token positions, so vLLM must
                # have run exactly Clef's sequence.
                if list(final.prompt_token_ids) != ids:
                    raise RuntimeError("vLLM tokenization differs from Clef's encoding")
                hidden = torch.as_tensor(final.outputs.data)
                if hidden.ndim != 2 or hidden.shape[0] != len(ids):
                    raise RuntimeError("vLLM returned incomplete Clef token states")
                complete = True
                return hidden
            finally:
                if not complete:
                    with suppress(Exception):
                        await self.engine_client.abort(request_id)

    def _readout(self, hidden, ids, encoded, rows):
        hidden = hidden.to(self.device, torch.bfloat16)
        embedding = self.output_embedding[rows].to(self.device)
        position = {token: index for index, token in enumerate(rows)}
        lexical = [
            torch.stack(
                [
                    embedding[[position[token] for token in ids[start:end]]].mean(0)
                    for start, end in question.option_spans
                ]
            )
            for question in encoded
        ]
        logits = self.head(hidden, encoded, lexical)
        return [row.float().softmax(-1).tolist() for row in logits]

    async def systemone(self, payload):
        from .endpoint import _compile_serialized

        started = time.perf_counter()
        ids, encoded, prompt, rows = await _compile_serialized(
            self.compile_lock, self._compile, payload
        )
        hidden = await self._states(ids, prompt)
        probabilities = await asyncio.get_running_loop().run_in_executor(
            self.head_executor, self._readout, hidden, ids, encoded, rows
        )
        if any(not math.isfinite(p) for row in probabilities for p in row):
            raise RuntimeError("Clef returned nonfinite probabilities")
        return {
            "model": self.model_id,
            "answers": {
                question.question_id: answer(
                    payload.questions[question.question_id],
                    dict(zip(question.option_ids, p)),
                )
                for question, p in zip(encoded, probabilities)
            },
            "usage": {"input_tokens": len(ids), "output_tokens": 0},
            "metadata": {"inference_seconds": time.perf_counter() - started},
        }


def _pooling_params():
    from vllm import PoolingParams

    return PoolingParams(task="token_embed", use_activation=False)
