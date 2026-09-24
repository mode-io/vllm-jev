"""vLLM endpoint plugin for typed Jev Choice decisions."""

import asyncio
import secrets
import time
import uuid
from contextlib import suppress
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from vllm import PoolingParams

from .prompt import candidate_prompts, choice_result


class ChoiceRequest(BaseModel):
    state: Any
    question: str
    options: list[str] = Field(min_length=2, max_length=255)
    temperature: float = Field(default=1.0, gt=0.0)
    cache_salt: str | None = None
    use_prefix_cache: bool = True
    priority: int = 0


class BatchChoiceRequest(BaseModel):
    requests: list[ChoiceRequest] = Field(min_length=1, max_length=64)


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
    def __init__(self, engine_client, tokenizer, max_inflight: int = 128):
        self.engine_client = engine_client
        self.tokenizer = tokenizer
        self.semaphore = asyncio.Semaphore(max_inflight)

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
        result = choice_result(
            payload.options, [entry[0] for entry in entries],
            payload.temperature,
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
        )
        return result


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

    async def init_state(self, engine_client, state, args) -> None:
        if engine_client is None:
            state.vllm_jev_service = None
            return
        import os
        max_inflight = int(os.environ.get("VLLM_JEV_MAX_INFLIGHT", "128"))
        if max_inflight < 1:
            raise ValueError("VLLM_JEV_MAX_INFLIGHT must be positive")
        state.vllm_jev_service = _JevService(
            engine_client, engine_client.get_tokenizer(), max_inflight
        )
