"""Typed decisions from native pooling states and released readout heads."""

import asyncio
import json
import math
import re
import secrets
import time
import uuid
from contextlib import suppress
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoTokenizer
from vllm import PoolingParams

from .decision_protocols import jevk5_prompt, mica_prompt, packed_slots, task_prompt


def _json(value):
    return (
        value
        if isinstance(value, str)
        else json.dumps(value, ensure_ascii=False, allow_nan=False)
    )


def _render_kev(value, indent=0):
    if indent > 64:
        raise ValueError("decision data is too deeply nested")
    pad = "  " * indent
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("decision data must be finite JSON")
        return str(value)
    if isinstance(value, list):
        return "\n".join(f"{pad}- {_render_kev(v, indent + 1).lstrip()}" for v in value)
    if not isinstance(value, dict):
        raise ValueError("decision state must be JSON-compatible")
    return "\n".join(
        f"{pad}{key}:\n{_render_kev(v, indent + 1)}"
        if isinstance(v, (dict, list))
        else f"{pad}{key}: {_render_kev(v)}"
        for key, v in value.items()
    )


def _indexed(value):
    if isinstance(value, list):
        return (
            [
                {"_index": i, **_indexed(v)}
                if isinstance(v, dict)
                else {"_index": i, "value": _indexed(v)}
                for i, v in enumerate(value)
            ]
            if len(value) >= 8
            else [_indexed(v) for v in value]
        )
    if isinstance(value, dict):
        return {k: _indexed(v) for k, v in value.items()}
    return value


def _answer(kind, keys, descriptions, probabilities):
    if kind == "noul":
        return {"type": kind, "noul": probabilities[keys.index("true")]}
    n = len(probabilities)
    selected = max(range(n), key=probabilities.__getitem__)
    confidence = 1.0 if n == 1 else max(0.0, (n * max(probabilities) - 1) / (n - 1))
    result = {"type": kind, "probabilities": dict(zip(keys, probabilities))}
    if kind == "choice":
        result.update(choice=keys[selected], confidence=confidence)
    else:
        distance = sum(abs(i - (n - 1) / 2) for i in range(n)) / n
        confidence = (
            1.0
            if n == 1
            else max(
                0.0,
                1
                - sum(p * abs(i - selected) for i, p in enumerate(probabilities))
                / distance,
            )
        )
        result.update(
            score=sum(i * p for i, p in enumerate(probabilities)),
            confidence=confidence,
            legend=dict(zip(keys, descriptions)),
        )
    return result


def _readout_positions(prepared):
    """Collect just the states needed by all questions sharing each prompt."""
    readouts = {}
    for *_, rows in prepared:
        for ids, positions, _ in rows:
            slots = readouts.setdefault(tuple(ids), {})
            for position in positions:
                slots.setdefault(position, len(slots))
    return readouts


def _thisthat_question(spec):
    """Match the author's System One labels and description legends."""
    kind, instructions, criteria = (
        spec.get("type"),
        spec.get("instructions"),
        spec.get("criteria"),
    )
    if not isinstance(instructions, str) or not instructions.strip():
        raise ValueError("question instructions must be nonempty")
    if kind == "noul":
        criteria = criteria if isinstance(criteria, dict) else {}
        keys, options = ["false", "true"], ["no", "yes"]
        descriptions = [criteria.get(key) for key in keys]
        heading = "Options:" if any(descriptions) else None
    elif kind == "choice" and isinstance(criteria, (dict, list)):
        keys = [str(key) for key in criteria]
        options = keys
        descriptions = (
            list(criteria.values())
            if isinstance(criteria, dict)
            else [None] * len(keys)
        )
        heading = "Options:"
    elif kind == "score" and isinstance(criteria, list):
        keys = [str(i) for i in range(len(criteria))]
        options = keys
        descriptions = [str(value) for value in criteria]
        heading = "Levels:"
    else:
        raise ValueError("unsupported This-That question")
    if not 2 <= len(options) <= 255 or len(set(options)) != len(options):
        raise ValueError("This-That requires 2 to 255 distinct options")
    if heading:
        legend = "\n".join(
            f"- {label}: {description}" if description else f"- {label}"
            for label, description in zip(options, descriptions)
        )
        instructions += f"\n\n{heading}\n{legend}"
    return kind, keys, descriptions, instructions, options


class DecisionCompiler:
    def __init__(self, checkpoint: Path, model_id: str, max_length: int):
        self.manifest = json.loads((checkpoint / "decision_manifest.json").read_text())
        self.tokenizer = AutoTokenizer.from_pretrained(
            checkpoint, local_files_only=True
        )
        self.model_names = [model_id] if isinstance(model_id, str) else model_id
        self.model_id = self.model_names[0]
        self.protocol = self.manifest["prompt_protocol"]
        self.max_length = min(max_length, self.manifest["max_length"])
        self.labels = self.manifest.get("labels", [])
        self.special = self.manifest.get("markers", [])

    def _question(self, spec):
        if self.protocol == "thisthat_slot_v1":
            return _thisthat_question(spec)
        if self.protocol == "jevk5_letters_v1":
            kind, criteria = spec.get("type"), spec.get("criteria")
            if kind == "noul":
                criteria = criteria or {}
                if not isinstance(criteria, dict):
                    raise ValueError("Noul criteria must be an object")
                pairs = [
                    (key, criteria.get(key) or f"The proposition is {key}.")
                    for key in ("true", "false")
                ]
            elif kind == "choice" and isinstance(criteria, dict):
                pairs = [(key, value or key) for key, value in criteria.items()]
            elif kind == "score" and isinstance(criteria, list):
                pairs = [(str(i), value) for i, value in enumerate(criteria)]
            else:
                raise ValueError("unsupported JevK5 question")
            if not 2 <= len(pairs) <= 16:
                raise ValueError("JevK5 single-pass requests require 2 to 16 options")
            keys = [key for key, _ in pairs]
            descriptions = [f"{key}: {value}" for key, value in pairs]
            return kind, keys, descriptions, spec.get("instructions", ""), descriptions
        if self.protocol == "mica_labels_v1":
            kind, criteria = spec.get("type"), spec.get("criteria")
            if kind == "noul":
                criteria = criteria if isinstance(criteria, dict) else {}
                keys = ["false", "true"]
                descriptions = [
                    str(criteria.get("false") or criteria.get("no") or "false"),
                    str(criteria.get("true") or criteria.get("yes") or "true"),
                ]
            elif (
                kind == "choice"
                and isinstance(criteria, dict)
                and 2 <= len(criteria) <= 255
            ):
                keys, descriptions = (
                    [str(k) for k in criteria],
                    [str(v) for v in criteria.values()],
                )
            elif (
                kind == "score"
                and isinstance(criteria, list)
                and 2 <= len(criteria) <= 10
            ):
                keys, descriptions = (
                    [str(i) for i in range(len(criteria))],
                    [str(v) for v in criteria],
                )
            else:
                raise ValueError("unsupported Mica question or option count")
            return (
                kind,
                keys,
                descriptions,
                str(spec.get("instructions", "")),
                descriptions,
            )
        kind = spec.get("type")
        instructions = spec.get("instructions")
        criteria = spec.get("criteria")
        render = _render_kev if self.protocol == "kev_pointer_v1" else _json
        if (
            self.protocol != "kev_pointer_v1"
            and kind == "noul"
            and instructions in (None, "")
        ):
            described = isinstance(criteria, dict) and any(
                criteria.get(k) not in (None, "") for k in ("true", "false")
            )
            if not described:
                raise ValueError(
                    "Noul without instructions needs criteria descriptions"
                )
            instructions = "Which answer fits the context?"
        instructions = render(instructions)
        if kind == "noul":
            if criteria is not None and not isinstance(criteria, dict):
                raise ValueError(
                    "Noul criteria must contain optional true/false descriptions"
                )
            criteria = criteria or {}
            keys = ["false", "true"]
            options = [
                name
                if criteria.get(key) in (None, "")
                else f"{name}: {render(criteria[key])}"
                for key, name in zip(keys, ["no", "yes"])
            ]
            descriptions = options
        elif kind == "choice" and isinstance(criteria, dict):
            keys = list(criteria)
            if not all(isinstance(k, str) and k for k in keys):
                raise ValueError("Choice keys must be nonempty strings")
            options = [
                (key if value in (None, "") else render(value))
                if self.protocol in ("task_json_v1", "thisthat_slot_v1")
                else (key if value in (None, "") else f"{key}: {render(value)}")
                for key, value in criteria.items()
            ]
            descriptions = options
        elif kind == "score" and isinstance(criteria, list):
            keys = [str(i) for i in range(len(criteria))]
            descriptions = [render(v) for v in criteria]
            options = (
                descriptions
                if self.protocol
                in ("kev_pointer_v1", "task_json_v1", "thisthat_slot_v1")
                else [f"{i}: {v}" for i, v in enumerate(descriptions)]
            )
        else:
            raise ValueError("unsupported decision question")
        minimum = 1 if self.protocol == "kev_pointer_v1" else 2
        if not minimum <= len(options) <= self.manifest["max_options"]:
            raise ValueError("candidate count exceeds checkpoint limits")
        if not instructions and self.protocol != "kev_pointer_v1":
            raise ValueError("question instructions must be nonempty")
        return kind, keys, descriptions, instructions, options

    def _base(self, state):
        if self.protocol == "kev_pointer_v1":
            text = re.sub(r"<\|([A-Za-z0-9_]+)\|>", r"<¦\1¦>", state)
            return [self.special[0]] + self.tokenizer.encode(
                text, add_special_tokens=False
            )
        return self.tokenizer.encode("Context:\n" + state, add_special_tokens=False)

    def _row(self, base, instructions, options, keys=None, kind="choice"):
        tok = self.tokenizer

        def encode(text):
            return tok.encode(text, add_special_tokens=False)

        if self.protocol == "jevk5_letters_v1":
            ids = encode(jevk5_prompt(tok, base, instructions, options))
            positions = [len(ids) - 1]
        elif self.protocol == "mica_labels_v1":
            text = mica_prompt(
                tok, base, instructions, keys, options, kind, self.labels
            )
            ids = encode(text)
            positions = [len(ids) - 1]
        elif self.protocol == "task_json_v1":
            text = task_prompt(tok, base, instructions, keys, options)
            ids = encode(text)
            for i in range(len(options)):
                full = encode(text + self.labels[i])
                if full[:-1] != ids or full[-1] != self.manifest["label_token_ids"][i]:
                    raise ValueError(
                        "decision answer labels do not align with the prompt"
                    )
            positions = [len(ids) - 1]
        elif self.protocol == "kev_pointer_v1":

            def user(text):
                return encode(re.sub(r"<\|([A-Za-z0-9_]+)\|>", r"<¦\1¦>", text))

            ids = list(base) + [self.special[1]] + user(instructions)
            positions = []
            for option in options:
                ids += [self.special[2]] + user(option) + [self.special[3]]
                positions.append(len(ids) - 1)
            ids.append(self.special[4])
            positions.append(len(ids) - 1)
        else:
            ids = list(base)
            piece = "\n\nQuestion: " + instructions + "\nOptions:"
            if len(options) <= 10:
                piece += "".join(
                    f"\n({self.labels[i]}) {o}" for i, o in enumerate(options)
                )
                ids += encode(piece + "\nAnswer: (")
            else:
                ids += encode(piece)
                for i, option in enumerate(options):
                    ids += (
                        encode("\n(")
                        + [self.manifest["label_token_ids"][i]]
                        + encode(") " + option)
                    )
                ids += encode("\nAnswer: (")
            positions = [len(ids) - 1]
        if len(ids) > self.max_length:
            raise ValueError("decision prompt exceeds model context")
        return ids, positions, len(options)

    def compile(self, payload):
        if payload.model not in (
            None,
            "vllm-jev",
            *getattr(self, "model_names", [self.model_id]),
        ):
            raise ValueError("requested model is not loaded")
        if not 1 <= len(payload.questions) <= 64:
            raise ValueError("System One requires 1 to 64 questions")
        if isinstance(payload.state, dict) and isinstance(
            payload.state.get("messages"), list
        ):
            for message in payload.state["messages"]:
                content = message.get("content") if isinstance(message, dict) else None
                if isinstance(content, list) and any(
                    isinstance(item, dict)
                    and item.get("type") in ("image_url", "video_url")
                    for item in content
                ):
                    raise ValueError(
                        "this checkpoint accepts text, not image/video inputs"
                    )
        state = (
            _render_kev(payload.state)
            if self.protocol == "kev_pointer_v1"
            else _json(payload.state)
            if self.protocol == "thisthat_slot_v1"
            else _json(_indexed(payload.state))
        )
        if self.protocol == "thisthat_slot_v1" and payload.state is None:
            raise ValueError("This-That requires a state")
        if self.protocol in ("task_json_v1", "mica_labels_v1", "jevk5_letters_v1"):
            base = payload.state
            state_tokens = len(
                self.tokenizer.encode(_json(payload.state), add_special_tokens=False)
            )
        else:
            base = self._base(state)
            if self.protocol == "thisthat_slot_v1":
                base = base[:1539]
            state_tokens = len(base)
        if state_tokens >= self.max_length:
            raise ValueError("decision state exceeds model context")
        prepared = []
        candidate_count = 0
        for identifier, spec in payload.questions.items():
            if not identifier or not isinstance(spec, dict):
                raise ValueError("question IDs and definitions must be nonempty")
            kind, keys, descriptions, instructions, options = self._question(spec)
            candidate_count += len(options)
            isolated = (
                kind == "score"
                and self.manifest.get("isolated_levels", False)
                and spec.get("isolated", True)
            )
            if self.protocol == "thisthat_slot_v1":
                rows = []
            elif isolated:
                rows = [
                    self._row(
                        base,
                        f"{instructions}\nProposed answer: {re.sub(r'^\s*-?\d+\s*:\s*', '', level)}\nDoes the proposed answer fit?",
                        ["no", "yes"],
                    )
                    for level in descriptions
                ]
            else:
                rows = [self._row(base, instructions, options, keys, kind)]
            prepared.append((identifier, kind, keys, descriptions, isolated, rows))
        if candidate_count > 256:
            raise ValueError("System One exceeds 256 candidates")
        if self.protocol == "thisthat_slot_v1":
            requested = [self._question(spec) for spec in payload.questions.values()]
            ids, slots = packed_slots(
                self.tokenizer,
                state,
                [(q[3], q[4]) for q in requested],
                self.labels,
                self.manifest["label_token_ids"],
            )
            if len(ids) > self.max_length:
                raise ValueError("packed decision prompt exceeds model context")
            prepared = [
                (identifier, q[0], q[1], q[2], False, [(ids, [slot], len(q[4]))])
                for identifier, q, slot in zip(payload.questions, requested, slots)
            ]
        if self.protocol != "kev_pointer_v1":
            rows = [row[0] for item in prepared for row in item[-1]]
            shared = min(map(len, rows))
            for i in range(shared):
                if any(ids[i] != rows[0][i] for ids in rows[1:]):
                    shared = i
                    break
            state_tokens = shared if len(rows) > 1 else 0
        return prepared, state_tokens


class DecisionService:
    def __init__(
        self,
        engine_client,
        checkpoint: Path,
        model_id,
        max_length: int,
        max_inflight: int = 128,
    ):
        self.engine_client = engine_client
        self.model_names = [model_id] if isinstance(model_id, str) else model_id
        self.model_id = self.model_names[0]
        self.compiler = DecisionCompiler(checkpoint, self.model_names, max_length)
        self.head = load_file(str(checkpoint / "decision_head.safetensors"))
        self.semaphore = asyncio.Semaphore(max_inflight)
        self.compile_lock = asyncio.Lock()

    def _probabilities(self, hidden, positions, count, kind):
        head = self.head
        hidden = hidden[positions].float()
        if self.compiler.protocol == "kev_pointer_v1":
            query = F.linear(hidden[-1], head["q.weight"], head["q.bias"])
            keys = F.linear(hidden[:-1], head["k.weight"], head["k.bias"])
            logits = keys @ query / math.sqrt(query.numel())
        else:
            weight = head["weight"]
            if self.compiler.protocol == "mica_labels_v1":
                weight = weight[:2] if kind == "noul" else weight[2 : 2 + count]
            else:
                weight = weight[:count]
            if self.compiler.protocol in ("thisthat_slot_v1", "jevk5_letters_v1"):
                logits = F.linear(hidden[0].bfloat16(), weight.bfloat16()).float()
            else:
                logits = F.linear(hidden[0], weight)
        temperatures = self.compiler.manifest.get("temperatures", {})
        temperature = temperatures.get(kind, self.compiler.manifest["temperature"])
        if not torch.isfinite(logits).all():
            raise RuntimeError("decision head returned nonfinite logits")
        return torch.softmax(logits / temperature, -1).tolist()

    async def _encode(self, ids, salt, positions):
        request_id = f"decision-{uuid.uuid4().hex}"
        complete = False
        async with self.semaphore:
            try:
                final = None
                async for output in self.engine_client.encode(
                    prompt={"prompt_token_ids": ids, "cache_salt": salt},
                    pooling_params=PoolingParams(
                        task="token_embed", use_activation=False
                    ),
                    request_id=request_id,
                ):
                    final = output
                if final is None or not final.finished or final.prompt_token_ids != ids:
                    raise RuntimeError("vLLM did not return expected decision tokens")
                hidden = torch.as_tensor(final.outputs.data)
                if hidden.shape != (len(ids), self.compiler.manifest["hidden_size"]):
                    raise RuntimeError("vLLM returned incomplete decision states")
                # Advanced indexing owns its storage; the completed task must not
                # retain the full token-embedding response until all questions finish.
                hidden = hidden[positions]
                complete = True
                return hidden
            finally:
                if not complete:
                    with suppress(Exception):
                        await self.engine_client.abort(request_id)

    async def _score(self, row, kind, salt, encoded, readouts):
        ids, positions, count = row
        key = tuple(ids)
        slots = readouts[key]
        if key not in encoded:
            encoded[key] = asyncio.create_task(self._encode(ids, salt, list(slots)))
        hidden = await encoded[key]
        return self._probabilities(hidden, [slots[p] for p in positions], count, kind)

    async def systemone(self, payload):
        from .endpoint import _compile_serialized, _gather_cancel_on_error

        started = time.perf_counter()
        prepared, state_tokens = await _compile_serialized(
            self.compile_lock, self.compiler.compile, payload
        )
        answers = {}
        salt = secrets.token_hex(16)
        input_tokens = state_tokens
        compute_tokens = 0
        encoded = {}
        readouts = _readout_positions(prepared)
        try:
            for identifier, kind, keys, descriptions, isolated, rows in prepared:
                probabilities = await _gather_cancel_on_error(
                    self._score(row, kind, salt, encoded, readouts) for row in rows
                )
                input_tokens += sum(len(row[0]) - state_tokens for row in rows)
                compute_tokens = sum(len(ids) for ids in encoded)
                if isolated:
                    mass = [p[1] for p in probabilities]
                    total = sum(mass)
                    p = (
                        [x / total for x in mass]
                        if total
                        else [1 / len(mass)] * len(mass)
                    )
                else:
                    p = probabilities[0]
                answers[identifier] = _answer(kind, keys, descriptions, p)
        finally:
            for task in encoded.values():
                if not task.done():
                    task.cancel()
            await asyncio.gather(*encoded.values(), return_exceptions=True)
        return {
            "model": self.model_id,
            "answers": answers,
            "usage": {"input_tokens": input_tokens, "output_tokens": 0},
            "internal_usage": {"compute_tokens": compute_tokens},
            "metadata": {"inference_seconds": time.perf_counter() - started},
        }
