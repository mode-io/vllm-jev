"""Apple Silicon serving for Jev decision models."""

import asyncio
import json
import time
from concurrent.futures import CancelledError, ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Event

import mlx.core as mx
import uvicorn
from fastapi import FastAPI
from mlx_lm import load
from mlx_lm.models.cache import ConcatenateKVCache

from .endpoint import ChoiceRequest, JevEndpointPlugin
from .prompt import branch_token_ids, choice_result


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
    def __init__(self, checkpoint: Path, model_id: str, executor: ThreadPoolExecutor):
        super().__init__(executor)
        manifest = json.loads((checkpoint / "jev_manifest.json").read_text())
        if manifest["prompt_protocol"] != "openjev_branch_v03":
            raise ValueError("macOS serving currently supports OpenJev-0.6B")
        config = json.loads((checkpoint / "config.json").read_text())
        if config.get("model_type") != "qwen3":
            raise ValueError("macOS serving requires a Qwen3 text checkpoint")

        rope_theta = config.get("rope_theta") or config["rope_parameters"]["rope_theta"]
        self.model, self.tokenizer = load(
            str(checkpoint), model_config={"rope_theta": rope_theta}
        )
        head = mx.load(str(checkpoint / "score.safetensors"))
        self.weight, self.bias = head["score.weight"], head["score.bias"]
        self.protocol = manifest["prompt_protocol"]
        self.temperatures = manifest["calibration_temperatures"]
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

    def _choice(self, payload: ChoiceRequest, kind: str, cancelled: Event) -> dict:
        started = time.perf_counter()
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

    def close(self) -> None:
        del self.model, self.weight, self.bias
        mx.clear_cache()


def serve(checkpoint: Path, model_id: str, *, host: str, port: int) -> None:
    is_valen = (checkpoint / "valen_manifest.json").is_file()

    def load_service(executor):
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
            name = "vllm_valen_service" if is_valen else "vllm_jev_service"
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
