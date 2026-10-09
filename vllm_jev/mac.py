"""Apple Silicon serving for Jev decision models."""

import asyncio
import json
import math
import time
from concurrent.futures import CancelledError, ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Event

import mlx.core as mx
import uvicorn
from fastapi import FastAPI
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.models.cache import ArraysCache, ConcatenateKVCache

from .endpoint import ChoiceRequest, JevEndpointPlugin
from .prompt import (
    branch_token_ids,
    candidate_prompts,
    choice_result,
    noul_prompt,
    tiny_token_ids,
)


class _MacService:
    def __init__(self, executor: ThreadPoolExecutor):
        self.executor = executor
        self.lock = asyncio.Lock()

    async def _run(self, call, *args) -> dict:
        async with self.lock:
            cancelled = Event()
            try:
                return await asyncio.get_running_loop().run_in_executor(
                    self.executor, call, *args, cancelled
                )
            except asyncio.CancelledError:
                cancelled.set()
                raise

    def _close(self) -> None:
        mx.synchronize()
        try:
            self.close()
        finally:
            # Release MLX's thread-local compilation cache before worker exit.
            mx.clear_streams()


class MacJevService(_MacService):
    # Larger branch batches showed measurable BF16 probability drift on MLX.
    _batch_choice_max_branches = 4
    _batch_choice_max_tokens = 192
    _prefix_choice_min_tokens = 128
    _branch_prefix_min_reused_tokens = 300

    def __init__(self, checkpoint: Path, model_id: str, executor: ThreadPoolExecutor):
        super().__init__(executor)
        manifest = json.loads((checkpoint / "jev_manifest.json").read_text())
        if manifest["prompt_protocol"] not in (
            "openjev_branch_v03",
            "open_jev_choice",
            "tiny_jev_marker",
        ):
            raise ValueError("unsupported Mac text decision protocol")
        config = json.loads((checkpoint / "config.json").read_text())
        if config.get("model_type") not in ("qwen3", "qwen3_5_text"):
            raise ValueError("macOS serving requires a supported Qwen text checkpoint")
        self.max_length = config["max_position_embeddings"]
        if (
            config["model_type"] == "qwen3_5_text"
            or manifest["prompt_protocol"] == "openjev_branch_v03"
        ):
            mx.set_cache_limit(512 * 1024**2)
        self._hybrid_cache = config["model_type"] == "qwen3_5_text"

        overrides = {}
        if config["model_type"] == "qwen3":
            overrides["rope_theta"] = (
                config.get("rope_theta") or config["rope_parameters"]["rope_theta"]
            )
        else:
            overrides["model_type"] = "qwen3_5"
        self.model, self.tokenizer = load(str(checkpoint), model_config=overrides)
        head = mx.load(str(checkpoint / "score.safetensors"))
        self.weight, self.bias = head["score.weight"], head["score.bias"]
        self.protocol = manifest["prompt_protocol"]
        self.temperatures = manifest.get("calibration_temperatures") or {
            kind: manifest["calibration_temperature"]
            for kind in ("choice", "noul", "score")
        }
        self.default_temperature = self.temperatures["choice"]
        self.model_id = model_id
        self.model_names = [model_id]

    def _score(self, ids: list[int], cancelled: Event, cache=None) -> float:
        if cancelled.is_set():
            raise CancelledError()
        hidden = self.model.model(mx.array([ids]), cache=cache)[0, -1].astype(
            mx.float32
        )
        value = mx.matmul(hidden[None], self.weight.T)[0, 0] + self.bias[0]
        mx.eval(value)
        return float(value.item())

    def _batch_scores(self, branches: list[list[int]], cancelled: Event) -> list[float]:
        if cancelled.is_set():
            raise CancelledError()
        lengths = [len(branch) for branch in branches]
        longest = max(lengths)
        pad = self.tokenizer.pad_token_id or 0
        token_ids = mx.array(
            [branch + [pad] * (longest - len(branch)) for branch in branches],
            dtype=mx.int32,
        )
        hidden = self.model.model(token_ids).astype(mx.float32)
        selected = hidden[mx.arange(len(branches)), mx.array(lengths) - 1]
        values = (selected @ self.weight.T)[:, 0] + self.bias[0]
        mx.eval(values)
        return values.tolist()

    def _prefix_scores(
        self, branches: list[list[int]], prefix: int, cancelled: Event
    ) -> list[float]:
        if cancelled.is_set():
            raise CancelledError()
        cache = self.model.make_cache()
        output = self.model.model(mx.array([branches[0][:prefix]]), cache=cache)
        states = [
            value
            for _, value in tree_flatten([item.state for item in cache])
            if isinstance(value, mx.array)
        ]
        mx.eval(output, *states)
        scores = []
        for branch in branches:
            fork = []
            for item in cache:
                if isinstance(item, ArraysCache):
                    arrays, left_padding, lengths = item.state
                    fork.append(
                        ArraysCache.from_state(
                            (list(arrays), left_padding, lengths), item.meta_state
                        )
                    )
                else:
                    fork.append(
                        ConcatenateKVCache.from_state(item.state, item.meta_state)
                    )
            scores.append(self._score(branch[prefix:], cancelled, fork))
        return scores

    def _choice(self, payload: ChoiceRequest, kind: str, cancelled: Event) -> dict:
        started = time.perf_counter()
        if self.protocol == "tiny_jev_marker":
            ids, positions = tiny_token_ids(
                self.tokenizer, payload.state, payload.question, payload.options
            )
            rendered = time.perf_counter()
            if cancelled.is_set():
                raise CancelledError()
            hidden = self.model.model(mx.array([ids]))[0]
            selected = hidden[mx.array(positions)].astype(mx.float32)
            mean = mx.mean(selected, axis=-1, keepdims=True)
            variance = mx.mean(mx.square(selected - mean), axis=-1, keepdims=True)
            normalized = (selected - mean) * mx.rsqrt(variance + 1e-5)
            values = mx.matmul(normalized, self.weight.T)[:, 0] + self.bias[0]
            mx.eval(values)
            scores = values.tolist()
            scored = time.perf_counter()
            temperature = payload.temperature or self.temperatures[kind]
            result = choice_result(payload.options, scores, temperature)
            finished = time.perf_counter()
            result.update(
                candidate_count=len(positions),
                cached_tokens=[0] * len(positions),
                prompt_tokens=[len(ids), *([0] * (len(positions) - 1))],
                latency_seconds=finished - started,
                timing={
                    "render_seconds": rendered - started,
                    "engine_seconds": scored - rendered,
                    "postprocess_seconds": finished - scored,
                },
                prefix_cache_requested=False,
                temperature=temperature,
            )
            return result
        if self.protocol == "open_jev_choice":
            prompts = candidate_prompts(
                payload.state, payload.question, payload.options
            )
            branches = [
                self.tokenizer.encode(
                    self.tokenizer.apply_chat_template(
                        [{"role": "user", "content": prompt}],
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    ),
                    add_special_tokens=False,
                )
                for prompt in prompts
            ]
            if any(len(branch) > self.max_length for branch in branches):
                raise ValueError(
                    f"prompt exceeds model context ({self.max_length} tokens)"
                )
            rendered = time.perf_counter()
            cached_tokens = [0] * len(branches)
            scores = None
            if (
                len(branches) <= self._batch_choice_max_branches
                and max(map(len, branches)) <= self._batch_choice_max_tokens
            ):
                scores = self._batch_scores(branches, cancelled)
            elif payload.use_prefix_cache and self._hybrid_cache and len(branches) >= 4:
                prefix = next(
                    (
                        index
                        for index, values in enumerate(zip(*branches))
                        if len(set(values)) > 1
                    ),
                    min(map(len, branches)),
                )
                if (
                    prefix >= self._prefix_choice_min_tokens
                    and prefix * 2 >= max(map(len, branches))
                    and all(len(branch) > prefix for branch in branches)
                ):
                    scores = self._prefix_scores(branches, prefix, cancelled)
                    cached_tokens[1:] = [prefix] * (len(branches) - 1)
            if scores is None:
                scores = [self._score(branch, cancelled) for branch in branches]
            scored = time.perf_counter()
            temperature = payload.temperature or self.temperatures[kind]
            result = choice_result(payload.options, scores, temperature)
            finished = time.perf_counter()
            result.update(
                candidate_count=len(branches),
                cached_tokens=cached_tokens,
                prompt_tokens=[len(branch) for branch in branches],
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
        branches = branch_token_ids(
            self.tokenizer, payload.state, payload.question, payload.options
        )
        prefix = next(
            (
                index
                for index, values in enumerate(zip(*branches))
                if len(set(values)) > 1
            ),
            min(map(len, branches)),
        )
        rendered = time.perf_counter()
        cached_tokens = [0] * len(branches)
        if cancelled.is_set():
            raise CancelledError()

        if (
            payload.use_prefix_cache
            and prefix > 0
            and len(branches) > 1
            and prefix * (len(branches) - 1) >= self._branch_prefix_min_reused_tokens
            and all(len(branch) > prefix for branch in branches)
        ):
            cache = [ConcatenateKVCache() for _ in self.model.layers]
            prefix_output = self.model.model(mx.array([branches[0][:prefix]]), cache)
            mx.eval(prefix_output, *[value for item in cache for value in item.state])
            scores = []
            for index, branch in enumerate(branches):
                fork = [
                    ConcatenateKVCache.from_state(item.state, item.meta_state)
                    for item in cache
                ]
                scores.append(self._score(branch[prefix:], cancelled, fork))
                if index:
                    cached_tokens[index] = prefix
        else:
            scores = [self._score(branch, cancelled) for branch in branches]

        scored = time.perf_counter()
        temperature = payload.temperature or self.temperatures[kind]
        result = choice_result(payload.options, scores, temperature)
        finished = time.perf_counter()
        result.update(
            candidate_count=len(branches),
            cached_tokens=cached_tokens,
            prompt_tokens=[len(branch) for branch in branches],
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

    async def choice(self, payload: ChoiceRequest, kind: str = "choice") -> dict:
        return await self._run(self._choice, payload, kind)

    def _noul(self, state, question: str, cancelled: Event):
        raw = noul_prompt(state, question)
        prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": raw}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(ids) > self.max_length:
            raise ValueError(f"prompt exceeds model context ({self.max_length} tokens)")
        score = self._score(ids, cancelled)
        scaled = score / self.temperatures["noul"]
        probability = (
            1 / (1 + math.exp(-scaled))
            if scaled >= 0
            else math.exp(scaled) / (1 + math.exp(scaled))
        )
        return probability, 0, len(ids)

    async def noul(self, state, question: str, salt: str):
        return await self._run(self._noul, state, question)

    def close(self) -> None:
        del self.model, self.weight, self.bias
        mx.clear_cache()


def serve(checkpoint: Path, model_id: str, *, host: str, port: int) -> None:
    is_valen = (checkpoint / "valen_manifest.json").is_file()
    is_decision = (checkpoint / "decision_manifest.json").is_file()
    from .laya_export import MODELS as LAYA_MODELS

    is_laya = model_id in LAYA_MODELS

    def load_service(executor):
        if is_decision:
            from .mac_decision import MacDecisionService

            return MacDecisionService(checkpoint, model_id, executor)
        if is_laya:
            from .mac_laya import MacLayaService

            return MacLayaService(checkpoint, model_id, executor)
        if is_valen:
            from .mac_valen import MacValenService

            return MacValenService(checkpoint, model_id, executor)
        return MacJevService(checkpoint, model_id, executor)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # MLX arrays and their streams must be created and used on one thread.
        with ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="jev-mlx"
        ) as executor:
            loop = asyncio.get_running_loop()
            service = await loop.run_in_executor(executor, load_service, executor)
            name = (
                "vllm_decision_service"
                if is_decision
                else "vllm_laya_service"
                if is_laya
                else "vllm_valen_service"
                if is_valen
                else "vllm_jev_service"
            )
            setattr(app.state, name, service)
            try:
                yield
            finally:
                setattr(app.state, name, None)
                await loop.run_in_executor(executor, service._close)

    app = FastAPI(title="vLLM Jev", lifespan=lifespan)
    JevEndpointPlugin().attach_router(app)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": model_id, "object": "model"}]}

    uvicorn.run(app, host=host, port=port, workers=1, access_log=False)
