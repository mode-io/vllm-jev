"""vLLM endpoint plugin for typed Jev Choice decisions."""

import asyncio
import json
import math
import secrets
import time
import uuid
from contextlib import suppress
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from vllm import PoolingParams

from .prompt import candidate_prompts, choice_result, render_value


class ChoiceRequest(BaseModel):
    state: Any
    question: str
    options: list[str] = Field(min_length=2, max_length=255)
    temperature: float | None = Field(default=None, gt=0.0)
    cache_salt: str | None = None
    use_prefix_cache: bool = True
    priority: int = 0


class BatchChoiceRequest(BaseModel):
    requests: list[ChoiceRequest] = Field(min_length=1, max_length=64)


class SystemOneRequest(BaseModel):
    state: Any
    questions: dict[str, dict[str, Any]]
    model: str | None = None


async def _gather_cancel_on_error(coroutines):
    tasks = [asyncio.create_task(coroutine) for coroutine in coroutines]
    try:
        return await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


class _JevService:
    def __init__(self, engine_client, tokenizer, max_inflight: int = 128,
                 default_temperature: float = 1.0):
        self.engine_client = engine_client
        self.tokenizer = tokenizer
        self.semaphore = asyncio.Semaphore(max_inflight)
        self.default_temperature = default_temperature

    async def _one_score(
        self, prompt: str, cache_salt: str, use_prefix_cache: bool, priority: int
    ) -> tuple[float, int, int]:
        request_id = f"jev-{uuid.uuid4().hex}"
        completed = False
        async with self.semaphore:
            try:
                iterator = self.engine_client.encode(
                    prompt={"prompt": prompt, "cache_salt": cache_salt},
                    pooling_params=PoolingParams(
                        task="classify", use_activation=False,
                        skip_reading_prefix_cache=not use_prefix_cache,
                    ),
                    request_id=request_id,
                    priority=priority,
                )
                final = None
                async for output in iterator:
                    final = output
                if final is None or not final.finished:
                    raise RuntimeError("vLLM returned no final pooling result")
                completed = True
                scalar = float(final.outputs.data.reshape(-1)[0])
                return scalar, int(final.num_cached_tokens), len(final.prompt_token_ids)
            finally:
                if not completed:
                    with suppress(Exception):
                        await self.engine_client.abort(request_id)

    async def choice(self, payload: ChoiceRequest) -> dict:
        started = time.perf_counter()
        raw_prompts = candidate_prompts(
            payload.state, payload.question, payload.options
        )
        prompts = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for text in raw_prompts
        ]
        # A fresh salt isolates unrelated users while preserving reuse among
        # the branches of this single Choice request.
        salt = payload.cache_salt or secrets.token_hex(16)
        rendered = time.perf_counter()
        entries = await _gather_cancel_on_error(
            self._one_score(prompt, salt, payload.use_prefix_cache, payload.priority)
            for prompt in prompts
        )
        scored = time.perf_counter()
        temperature = payload.temperature if payload.temperature is not None else self.default_temperature
        result = choice_result(
            payload.options, [entry[0] for entry in entries],
            temperature,
        )
        finished = time.perf_counter()
        result.update(
            candidate_count=len(entries),
            cached_tokens=[entry[1] for entry in entries],
            prompt_tokens=[entry[2] for entry in entries],
            latency_seconds=finished - started,
            timing={
                "render_seconds": rendered - started,
                "engine_seconds": scored - rendered,
                "postprocess_seconds": finished - scored,
            },
            prefix_cache_requested=payload.use_prefix_cache,
            temperature=temperature,
        )
        return result

    async def noul(self, state: Any, question: str, salt: str) -> tuple[float, int, int]:
        raw = (f"Context:\n{render_value(state)}\n\nQuestion: {render_value(question)}\n"
               "Is the answer to this question yes? Answer Yes or No.")
        prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": raw}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )
        score, cached, tokens = await self._one_score(prompt, salt, True, 0)
        scaled = score / self.default_temperature
        probability = (1 / (1 + math.exp(-scaled)) if scaled >= 0 else
                       math.exp(scaled) / (1 + math.exp(scaled)))
        return probability, cached, tokens


async def _system_one(payload: SystemOneRequest, service: _JevService) -> dict:
    def describe(value):
        if not isinstance(value, (str, dict, list)):
            raise ValueError("instructions and descriptions must be text or JSON")
        return render_value(value)

    if payload.model not in (None, "open-jev", "jev-latest", "vllm-jev"):
        raise ValueError("requested model is not loaded")
    if not isinstance(payload.state, (str, dict, list)):
        raise ValueError("state must be text or JSON")
    if not 1 <= len(payload.questions) <= 64:
        raise ValueError("System One requires 1 to 64 questions")
    salt = secrets.token_hex(16)
    prepared = []
    candidate_count = 0
    for identifier, definition in payload.questions.items():
        if not identifier or not isinstance(definition, dict):
            raise ValueError("question IDs and definitions must be nonempty")
        kind = definition.get("type")
        instruction = definition.get("instructions")
        question = describe(instruction)
        criteria = definition.get("criteria")
        if kind == "choice":
            if not isinstance(criteria, dict) or not 1 <= len(criteria) <= 255:
                raise ValueError("Choice requires 1 to 255 candidates")
            keys = list(criteria)
            if any(not isinstance(key, str) or not key for key in keys):
                raise ValueError("Choice candidate names must be nonempty strings")
            options = [key if value is None else f"{key}: {describe(value)}"
                       for key, value in criteria.items()]
            candidate_count += len(options)
        elif kind == "score":
            if not isinstance(criteria, list) or not 2 <= len(criteria) <= 10:
                raise ValueError("Score requires 2 to 10 descriptive levels")
            keys = [str(index) for index in range(len(criteria))]
            options = [describe(value) for value in criteria]
            candidate_count += len(options)
        elif kind == "noul":
            if criteria is not None:
                if not isinstance(criteria, dict) or set(criteria) != {"true", "false"}:
                    raise ValueError("Noul criteria must define true and false")
                question += (f"\nYes means: {describe(criteria['true'])}"
                             f"\nNo means: {describe(criteria['false'])}")
            keys, options = [], []
            candidate_count += 1
        else:
            raise ValueError("question type must be choice, score or noul")
        prepared.append((identifier, kind, question, criteria, keys, options))
    if candidate_count > 256:
        raise ValueError("System One exceeds 256 candidate sequences")

    async def answer(item):
        identifier, kind, question, criteria, keys, options = item
        if kind == "noul":
            probability, cached, tokens = await service.noul(payload.state, question, salt)
            return identifier, {"type": "noul", "noul": probability}, cached, tokens
        if len(options) == 1:
            return identifier, {"type": "choice", "choice": keys[0],
                                "probabilities": {keys[0]: 1.0}, "confidence": 1.0}, 0, 0
        result = await service.choice(ChoiceRequest(
            state=payload.state, question=question, options=options,
            cache_salt=salt,
        ))
        probabilities = {key: result["probabilities"][option]
                         for key, option in zip(keys, options)}
        if kind == "choice":
            selected = keys[options.index(result["choice"])]
            answer_value = {"type": kind, "choice": selected,
                            "probabilities": probabilities,
                            "confidence": result["confidence"]}
        else:
            mode = max(range(len(keys)), key=lambda i: probabilities[keys[i]])
            distance = sum(probabilities[key] * abs(i - mode)
                           for i, key in enumerate(keys))
            center = (len(keys) - 1) / 2
            uniform_deviation = sum(abs(i - center) for i in range(len(keys))) / len(keys)
            answer_value = {"type": kind,
                            "score": sum(i * probabilities[key]
                                         for i, key in enumerate(keys)),
                            "probabilities": probabilities,
                            "confidence": max(0.0, 1.0 - distance / uniform_deviation),
                            "legend": {key: value for key, value in zip(keys, criteria)}}
        return identifier, answer_value, sum(result["cached_tokens"]), sum(result["prompt_tokens"])

    started = time.perf_counter()
    values = await _gather_cancel_on_error(answer(item) for item in prepared)
    return {"answers": {identifier: value for identifier, value, _, _ in values},
            "model": "vllm-jev",
            "usage": {"input_tokens": sum(tokens for _, _, _, tokens in values),
                      "output_tokens": 0},
            "metadata": {"method": "lora_decision_head",
                         "temperature": service.default_temperature,
                         "candidate_sequences": candidate_count,
                         "cached_tokens": sum(cached for _, _, cached, _ in values),
                         "inference_seconds": time.perf_counter() - started}}


class JevEndpointPlugin:
    name = "vllm_jev_endpoint"
    required_tasks = ("classify",)

    def attach_router(self, app: FastAPI) -> None:
        @app.post("/plugins/vllm-jev/choice")
        async def choice(payload: ChoiceRequest, raw_request: Request):
            service = getattr(raw_request.app.state, "vllm_jev_service", None)
            if service is None:
                raise HTTPException(status_code=503, detail="Jev engine unavailable")
            try:
                return await service.choice(payload)
            except ValueError as error:
                raise HTTPException(status_code=422, detail=str(error)) from error

        @app.post("/plugins/vllm-jev/batch")
        async def batch(payload: BatchChoiceRequest, raw_request: Request):
            service = getattr(raw_request.app.state, "vllm_jev_service", None)
            if service is None:
                raise HTTPException(status_code=503, detail="Jev engine unavailable")
            total_candidates = sum(len(item.options) for item in payload.requests)
            if total_candidates > 256:
                raise HTTPException(
                    status_code=422, detail="batch exceeds 256 candidates"
                )
            started = time.perf_counter()
            try:
                results = await _gather_cancel_on_error(
                    service.choice(item) for item in payload.requests
                )
            except ValueError as error:
                raise HTTPException(status_code=422, detail=str(error)) from error
            return {"results": results,
                    "batch_wall_seconds": time.perf_counter() - started,
                    "decision_count": len(results),
                    "candidate_count": total_candidates}

        @app.post("/v1/systemone")
        async def system_one(payload: SystemOneRequest, raw_request: Request):
            service = getattr(raw_request.app.state, "vllm_jev_service", None)
            if service is None:
                raise HTTPException(status_code=503, detail="Jev engine unavailable")
            try:
                return await _system_one(payload, service)
            except ValueError as error:
                raise HTTPException(status_code=422, detail=str(error)) from error

    async def init_state(self, engine_client, state, args) -> None:
        if engine_client is None:
            state.vllm_jev_service = None
            return
        import os
        max_inflight = int(os.environ.get("VLLM_JEV_MAX_INFLIGHT", "128"))
        if max_inflight < 1:
            raise ValueError("VLLM_JEV_MAX_INFLIGHT must be positive")
        manifest = Path(args.model) / "jev_manifest.json"
        default_temperature = 1.0
        if manifest.is_file():
            default_temperature = json.loads(manifest.read_text()).get(
                "calibration_temperature", 1.0)
        if not isinstance(default_temperature, (int, float)) or not math.isfinite(
            default_temperature
        ) or default_temperature <= 0:
            raise ValueError("invalid Jev calibration temperature")
        state.vllm_jev_service = _JevService(
            engine_client, engine_client.get_tokenizer(), max_inflight,
            float(default_temperature),
        )
