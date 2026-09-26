"""vLLM endpoint plugin for typed Jev Choice decisions."""

import asyncio
import json
import math
import os
import secrets
import time
import uuid
from contextlib import suppress
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from vllm import PoolingParams
from vllm.entrypoints.serve.utils.api_utils import load_aware_call, with_cancellation

from . import QWEN3_PROJECTED_ARCHITECTURE
from .prompt import (
    branch_token_ids,
    candidate_prompts,
    choice_result,
    render_value,
    tiny_token_ids,
)


class ChoiceRequest(BaseModel):
    state: Any
    question: str
    options: list[str] = Field(min_length=2, max_length=255)
    temperature: float | None = Field(default=None, gt=0.0, allow_inf_nan=False)
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
    def __init__(
        self,
        engine_client,
        tokenizer,
        max_inflight: int = 128,
        default_temperature: float = 1.0,
        model_id: str | list[str] = "vllm-jev",
        protocol: str = "open_jev_choice",
        temperatures: dict[str, float] | None = None,
        head: tuple[Any, Any] | None = None,
        marker: tuple[str, str] = (" - correct?", "<opt>"),
        projected_token_scores: bool = False,
    ):
        self.engine_client = engine_client
        self.tokenizer = tokenizer
        self.semaphore = asyncio.Semaphore(max_inflight)
        self.default_temperature = default_temperature
        self.model_names = [model_id] if isinstance(model_id, str) else model_id
        self.model_id = self.model_names[0]
        self.protocol = protocol
        self.temperatures = temperatures or {
            kind: default_temperature for kind in ("choice", "noul", "score")
        }
        self.head = head
        self.marker = marker
        self.projected_token_scores = projected_token_scores

    async def _encode(
        self,
        prompt: str | list[int],
        cache_salt: str,
        use_prefix_cache: bool,
        priority: int,
        task: str = "classify",
    ):
        request_id = f"jev-{uuid.uuid4().hex}"
        completed = False
        async with self.semaphore:
            try:
                iterator = self.engine_client.encode(
                    prompt=(
                        {"prompt_token_ids": prompt, "cache_salt": cache_salt}
                        if isinstance(prompt, list)
                        else {"prompt": prompt, "cache_salt": cache_salt}
                    ),
                    pooling_params=PoolingParams(
                        task=task,
                        use_activation=False,
                        skip_reading_prefix_cache=True
                        if not use_prefix_cache
                        else None,
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
                return final
            finally:
                if not completed:
                    with suppress(Exception):
                        await self.engine_client.abort(request_id)

    async def _one_score(
        self,
        prompt: str | list[int],
        cache_salt: str,
        use_prefix_cache: bool,
        priority: int,
    ) -> tuple[float, int, int]:
        final = await self._encode(prompt, cache_salt, use_prefix_cache, priority)
        values = final.outputs.data.reshape(-1)
        if len(values) != 1:
            raise RuntimeError("vLLM returned an invalid scalar decision head")
        scalar = float(values[0])
        return scalar, int(final.num_cached_tokens), len(final.prompt_token_ids)

    async def _token_scores(
        self,
        ids: list[int],
        positions: list[int],
        salt: str,
        priority: int,
        use_prefix_cache: bool,
    ) -> tuple[list[float], int, int]:
        import torch
        import torch.nn.functional as functional

        # vLLM skips cache reads for token_embed to return every token's state.
        final = await self._encode(ids, salt, use_prefix_cache, priority, "token_embed")
        if self.projected_token_scores:
            scores = torch.as_tensor(final.outputs.data).reshape(-1)
            if scores.numel() == len(ids):
                return (
                    scores[positions].float().tolist(),
                    int(final.num_cached_tokens),
                    len(final.prompt_token_ids),
                )
        weight, bias = self.head
        hidden = torch.as_tensor(final.outputs.data).reshape(-1, weight.shape[1])
        if hidden.shape[0] != len(ids):
            raise RuntimeError("vLLM returned an incomplete token embedding sequence")
        selected = hidden[positions].float()
        scores = (
            functional.linear(
                functional.layer_norm(selected, (weight.shape[1],)).to(weight.dtype),
                weight,
                bias,
            )
            .float()
            .flatten()
            .tolist()
        )
        return scores, int(final.num_cached_tokens), len(final.prompt_token_ids)

    async def choice(self, payload: ChoiceRequest, kind: str = "choice") -> dict:
        started = time.perf_counter()
        if self.protocol == "tiny_jev_marker":
            ids, positions = tiny_token_ids(
                self.tokenizer,
                payload.state,
                payload.question,
                payload.options,
                cue=self.marker[0],
                opt_token=self.marker[1],
            )
            prompts = []
        elif self.protocol == "openjev_branch_v03":
            prompts = branch_token_ids(
                self.tokenizer, payload.state, payload.question, payload.options
            )
        else:
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
        if self.protocol == "tiny_jev_marker":
            scores, cached, tokens = await self._token_scores(
                ids, positions, salt, payload.priority, payload.use_prefix_cache
            )
            entries = [
                (score, cached if index == 0 else 0, tokens if index == 0 else 0)
                for index, score in enumerate(scores)
            ]
        else:
            entries = await _gather_cancel_on_error(
                self._one_score(
                    prompt, salt, payload.use_prefix_cache, payload.priority
                )
                for prompt in prompts
            )
        scored = time.perf_counter()
        temperature = (
            payload.temperature
            if payload.temperature is not None
            else self.temperatures[kind]
        )
        if any(not math.isfinite(entry[0]) for entry in entries):
            raise RuntimeError("vLLM returned nonfinite decision scores")
        result = choice_result(
            payload.options,
            [entry[0] for entry in entries],
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

    async def noul(
        self, state: Any, question: str, salt: str
    ) -> tuple[float, int, int]:
        raw = (
            f"Context:\n{render_value(state)}\n\nQuestion: {render_value(question)}\n"
            "Is the answer to this question yes? Answer Yes or No."
        )
        prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": raw}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        score, cached, tokens = await self._one_score(prompt, salt, True, 0)
        if not math.isfinite(score):
            raise RuntimeError("vLLM returned a nonfinite decision score")
        scaled = score / self.temperatures["noul"]
        probability = (
            1 / (1 + math.exp(-scaled))
            if scaled >= 0
            else math.exp(scaled) / (1 + math.exp(scaled))
        )
        return probability, cached, tokens


async def _system_one(payload: SystemOneRequest, service: _JevService) -> dict:
    def describe(value):
        if not isinstance(value, (str, dict, list)):
            raise ValueError("instructions and descriptions must be text or JSON")
        return render_value(value)

    if payload.model not in (
        None,
        "open-jev",
        "jev-latest",
        "vllm-jev",
        *service.model_names,
    ):
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
            minimum = 2 if service.protocol == "openjev_branch_v03" else 1
            if not isinstance(criteria, dict) or not minimum <= len(criteria) <= 255:
                raise ValueError(f"Choice requires {minimum} to 255 candidates")
            keys = list(criteria)
            if any(not isinstance(key, str) or not key for key in keys):
                raise ValueError("Choice candidate names must be nonempty strings")
            options = (
                [describe(value) for value in criteria.values()]
                if service.protocol == "openjev_branch_v03"
                else [
                    key if value is None else f"{key}: {describe(value)}"
                    for key, value in criteria.items()
                ]
            )
            candidate_count += len(options)
        elif kind == "score":
            if not isinstance(criteria, list) or not 2 <= len(criteria) <= 10:
                raise ValueError("Score requires 2 to 10 descriptive levels")
            keys = (
                [describe(value) for value in criteria]
                if service.protocol == "tiny_jev_marker"
                else [str(index) for index in range(len(criteria))]
            )
            options = [describe(value) for value in criteria]
            candidate_count += len(options)
        elif kind == "noul":
            if criteria is not None:
                if not isinstance(criteria, dict) or set(criteria) != {"true", "false"}:
                    raise ValueError("Noul criteria must define true and false")
                if service.protocol in ("open_jev_choice", "tiny_jev_marker"):
                    question += (
                        f"\nYes means: {describe(criteria['true'])}"
                        f"\nNo means: {describe(criteria['false'])}"
                    )
            keys = []
            if service.protocol == "openjev_branch_v03":
                options = (
                    [describe(criteria["false"]), describe(criteria["true"])]
                    if criteria
                    else [
                        f"The following statement is false: {question}",
                        f"The following statement is true: {question}",
                    ]
                )
                candidate_count += 2
            elif service.protocol == "tiny_jev_marker":
                options = ["no", "yes"]
                candidate_count += 2
            else:
                options = []
                candidate_count += 1
        else:
            raise ValueError("question type must be choice, score or noul")
        prepared.append((identifier, kind, question, criteria, keys, options))
    if candidate_count > 256:
        raise ValueError("System One exceeds 256 candidate sequences")

    async def answer(item):
        identifier, kind, question, criteria, keys, options = item
        if kind == "noul":
            if service.protocol in ("openjev_branch_v03", "tiny_jev_marker"):
                result = await service.choice(
                    ChoiceRequest(
                        state=payload.state,
                        question=question,
                        options=options,
                        cache_salt=salt,
                    ),
                    kind="noul",
                )
                return (
                    identifier,
                    {"type": "noul", "noul": result["probabilities"][options[1]]},
                    sum(result["cached_tokens"]),
                    sum(result["prompt_tokens"]),
                )
            probability, cached, tokens = await service.noul(
                payload.state, question, salt
            )
            return identifier, {"type": "noul", "noul": probability}, cached, tokens
        if len(options) == 1:
            return (
                identifier,
                {
                    "type": "choice",
                    "choice": keys[0],
                    "probabilities": {keys[0]: 1.0},
                    "confidence": 1.0,
                },
                0,
                0,
            )
        result = await service.choice(
            ChoiceRequest(
                state=payload.state,
                question=question,
                options=options,
                cache_salt=salt,
            ),
            kind=kind,
        )
        probabilities = {
            key: result["probabilities"][option] for key, option in zip(keys, options)
        }
        if kind == "choice":
            selected = keys[options.index(result["choice"])]
            answer_value = {
                "type": kind,
                "choice": selected,
                "probabilities": probabilities,
                "confidence": (
                    result["selected_probability"]
                    if service.protocol in ("openjev_branch_v03", "tiny_jev_marker")
                    else result["confidence"]
                ),
            }
        else:
            mode = max(range(len(keys)), key=lambda i: probabilities[keys[i]])
            distance = sum(
                probabilities[key] * abs(i - mode) for i, key in enumerate(keys)
            )
            center = (len(keys) - 1) / 2
            uniform_deviation = sum(abs(i - center) for i in range(len(keys))) / len(
                keys
            )
            answer_value = {
                "type": kind,
                "score": sum(i * probabilities[key] for i, key in enumerate(keys)),
                "probabilities": probabilities,
                "confidence": (
                    max(probabilities.values())
                    if service.protocol in ("openjev_branch_v03", "tiny_jev_marker")
                    else max(0.0, 1.0 - distance / uniform_deviation)
                ),
                "legend": {key: value for key, value in zip(keys, criteria)},
            }
        return (
            identifier,
            answer_value,
            sum(result["cached_tokens"]),
            sum(result["prompt_tokens"]),
        )

    started = time.perf_counter()
    values = await _gather_cancel_on_error(answer(item) for item in prepared)
    return {
        "answers": {identifier: value for identifier, value, _, _ in values},
        "model": service.model_id,
        "usage": {
            "input_tokens": sum(tokens for _, _, _, tokens in values),
            "output_tokens": 0,
        },
        "metadata": {
            "method": (
                "marker_token_head"
                if service.protocol == "tiny_jev_marker"
                else "lora_decision_head"
            ),
            "temperature": (
                service.temperatures
                if service.protocol == "openjev_branch_v03"
                else service.default_temperature
            ),
            "candidate_sequences": candidate_count,
            "cached_tokens": sum(cached for _, _, cached, _ in values),
            "inference_seconds": time.perf_counter() - started,
        },
    }


class JevEndpointPlugin:
    name = "vllm_jev_endpoint"
    required_tasks = ("classify", "token_embed")

    def attach_router(self, app: FastAPI) -> None:
        @app.post("/plugins/vllm-jev/choice")
        @with_cancellation
        @load_aware_call
        async def choice(payload: ChoiceRequest, raw_request: Request):
            service = getattr(raw_request.app.state, "vllm_jev_service", None)
            if service is None:
                raise HTTPException(status_code=503, detail="Jev engine unavailable")
            try:
                return await service.choice(payload)
            except ValueError as error:
                raise HTTPException(status_code=422, detail=str(error)) from error

        @app.post("/plugins/vllm-jev/batch")
        @with_cancellation
        @load_aware_call
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
            return {
                "results": results,
                "batch_wall_seconds": time.perf_counter() - started,
                "decision_count": len(results),
                "candidate_count": total_candidates,
            }

        @app.post("/v1/systemone")
        @with_cancellation
        @load_aware_call
        async def system_one(payload: SystemOneRequest, raw_request: Request):
            vjev = getattr(raw_request.app.state, "vllm_vjev_service", None)
            if vjev is not None:
                try:
                    return await vjev.systemone(payload)
                except ValueError as error:
                    raise HTTPException(status_code=422, detail=str(error)) from error
            valen = getattr(raw_request.app.state, "vllm_valen_service", None)
            if valen is not None:
                try:
                    return await valen.systemone(payload)
                except ValueError as error:
                    raise HTTPException(status_code=422, detail=str(error)) from error
            service = getattr(raw_request.app.state, "vllm_jev_service", None)
            if service is None:
                raise HTTPException(status_code=503, detail="Jev engine unavailable")
            try:
                return await _system_one(payload, service)
            except ValueError as error:
                raise HTTPException(status_code=422, detail=str(error)) from error

    async def init_state(self, engine_client, state, args) -> None:
        state.vllm_jev_service = None
        state.vllm_valen_service = None
        state.vllm_vjev_service = None
        manifest = Path(args.model) / "jev_manifest.json"
        valen_manifest = Path(args.model) / "valen_manifest.json"
        vjev_manifest = Path(args.model) / "vjev_manifest.json"
        if engine_client is None or not (
            manifest.is_file() or valen_manifest.is_file() or vjev_manifest.is_file()
        ):
            return
        max_inflight = int(os.environ.get("VLLM_JEV_MAX_INFLIGHT", "128"))
        if max_inflight < 1:
            raise ValueError("VLLM_JEV_MAX_INFLIGHT must be positive")
        if vjev_manifest.is_file():
            from .vjev import VjevService

            state.vllm_vjev_service = VjevService(
                engine_client,
                Path(args.model),
                getattr(args, "served_model_name", None) or "vjev",
                engine_client.model_config.max_model_len,
                max_inflight=max_inflight,
            )
            return
        if valen_manifest.is_file():
            from .valen import ValenService

            model_id = getattr(args, "served_model_name", None) or "Valen"
            state.vllm_valen_service = ValenService(
                engine_client,
                Path(args.model),
                model_id,
                engine_client.model_config.max_model_len,
                max_inflight=max_inflight,
            )
            return
        data = json.loads(manifest.read_text())
        default_temperature = data.get("calibration_temperature", 1.0)
        protocol = data.get("prompt_protocol", "open_jev_choice")
        temperatures = data.get("calibration_temperatures")
        head = None
        marker = (" - correct?", "<opt>")
        projected_token_scores = (
            protocol == "tiny_jev_marker"
            and data.get("architecture") == QWEN3_PROJECTED_ARCHITECTURE
        )
        if (
            not isinstance(default_temperature, (int, float))
            or not math.isfinite(default_temperature)
            or default_temperature <= 0
        ):
            raise ValueError("invalid Jev calibration temperature")
        if protocol not in ("open_jev_choice", "openjev_branch_v03", "tiny_jev_marker"):
            raise ValueError("unsupported Jev prompt protocol")
        if temperatures is not None and (
            set(temperatures) != {"choice", "noul", "score"}
            or any(
                not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
                for value in temperatures.values()
            )
        ):
            raise ValueError("invalid Jev calibration temperatures")
        if protocol == "tiny_jev_marker":
            from safetensors.torch import load_file

            config = json.loads((Path(args.model) / "config.json").read_text())
            tensors = load_file(str(Path(args.model) / "score.safetensors"))
            prefix = "score." if projected_token_scores else ""
            head = (tensors[prefix + "weight"], tensors[prefix + "bias"])
            marker = (config.get("cue", marker[0]), config.get("opt_token", marker[1]))
        state.vllm_jev_service = _JevService(
            engine_client,
            engine_client.get_tokenizer(),
            max_inflight,
            float(default_temperature),
            getattr(args, "served_model_name", None) or "vllm-jev",
            protocol,
            temperatures,
            head,
            marker,
            projected_token_scores,
        )
