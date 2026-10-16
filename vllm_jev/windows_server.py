"""Shared HTTP lifecycle and bounded execution for Windows CPU decision models."""

import asyncio
import json
import time
from collections.abc import Callable
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from contextlib import asynccontextmanager, suppress
from threading import Event
from typing import Any

from fastapi import FastAPI, HTTPException, Request

from .decision_template import current_template
from .request_models import SystemOneRequest


class ServiceBusyError(RuntimeError):
    """The bounded CPU request queue has no available slot."""


def _finite_json(value) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError, RecursionError) as error:
        raise ValueError("decision data must be finite JSON") from error


def prepare_request(
    payload: SystemOneRequest,
    model_id: str,
    protocol: str,
    aliases: tuple[str, ...] = (),
) -> SystemOneRequest:
    if payload.model not in (None, "vllm-jev", model_id, *aliases):
        raise ValueError("requested model is not loaded")
    if payload.cache_salt is not None:
        raise ValueError("cache_salt is not supported by the Windows CPU backend")
    if payload.images or payload.videos or payload.media:
        raise ValueError("the Windows CPU backend accepts text, not image/video inputs")
    if not isinstance(payload.state, (str, dict, list)):
        raise ValueError("decision state must be text or JSON")
    if isinstance(payload.state, dict):
        messages = payload.state.get("messages")
        if isinstance(messages, list):
            for message in messages:
                content = message.get("content") if isinstance(message, dict) else None
                if isinstance(content, list) and any(
                    isinstance(item, dict)
                    and item.get("type")
                    in ("image_url", "video_url", "image", "video", "input_image")
                    for item in content
                ):
                    raise ValueError("the Windows CPU backend accepts text only")
    state_text = (
        payload.state if isinstance(payload.state, str) else _finite_json(payload.state)
    )
    if len(state_text) > 50_000:
        raise ValueError("decision state exceeds 50,000 characters")
    if len(payload.questions) > 64:
        raise ValueError("System One exceeds 64 questions")
    if len(_finite_json(payload.questions).encode("utf-8")) > 2_000_000:
        raise ValueError("decision questions exceed 2 MiB")
    candidates = 0
    for identifier, question in payload.questions.items():
        if not identifier:
            raise ValueError("question IDs must be nonempty")
        kind = question.get("type")
        if kind not in ("choice", "noul", "score"):
            raise ValueError("question type must be choice, noul or score")
        if "instructions" not in question:
            raise ValueError(f"question {identifier!r} requires instructions")
        criteria = question.get("criteria")
        if kind == "choice":
            if not isinstance(criteria, (dict, list)) or not criteria:
                raise ValueError("Choice requires a nonempty criteria object or list")
            if isinstance(criteria, list) and any(
                not isinstance(value, str) for value in criteria
            ):
                raise ValueError("choice labels must be strings")
        elif kind == "score":
            if not isinstance(criteria, list) or not criteria:
                raise ValueError("Score requires a nonempty list of levels")
        elif criteria is not None and (
            not isinstance(criteria, dict)
            or not {str(key).lower() for key in criteria} <= {"true", "false"}
        ):
            raise ValueError(
                "Noul criteria must contain optional true/false descriptions"
            )
        # Open-Jev reads one scalar for Noul; the other protocols score two labels.
        candidates += (
            (1 if protocol == "open_jev_choice" else 2)
            if kind == "noul"
            else len(criteria)
        )
    if candidates > 256:
        raise ValueError("System One exceeds 256 candidates")
    template = current_template()
    template.validate_protocol(protocol)
    questions = {
        key: {**spec, "instructions": template.instructions(spec["instructions"])}
        for key, spec in payload.questions.items()
    }
    state = template.state(payload.state)
    if len(state if isinstance(state, str) else _finite_json(state)) > 50_000:
        raise ValueError("decision state exceeds 50,000 characters after templating")
    if len(_finite_json(questions).encode("utf-8")) > 2_000_000:
        raise ValueError("decision questions exceed 2 MiB after templating")
    return payload.model_copy(update={"state": state, "questions": questions})


async def _drain(future) -> None:
    """Cleanup cannot release the model while its synchronous call is running."""
    while not future.done():
        with suppress(Exception, asyncio.CancelledError):
            await asyncio.shield(future)
    with suppress(Exception, asyncio.CancelledError):
        future.result()


def _consume(future) -> None:
    # A disconnected caller may no longer be awaiting a worker's exception.
    with suppress(Exception, asyncio.CancelledError):
        future.exception()


class WindowsCPUService:
    def __init__(
        self, agent, model_id, revision, executor, *, backend, prepare, max_pending=16
    ):
        self.agent = agent
        self.model_id = model_id
        self.revision = revision
        self.executor = executor
        self.backend = backend
        self.prepare = prepare
        self.max_pending = max_pending
        self.pending: dict[Future, Event] = {}
        self.closed = False

    def identity(self) -> dict:
        return {
            "backend": self.backend,
            "device": "cpu",
            "experimental": True,
            "model": self.model_id,
            "revision": self.revision,
        }

    def _systemone(self, payload, cancelled):
        if cancelled.is_set():
            raise CancelledError()
        started = time.perf_counter()
        result = self.agent.system_one(payload.state, payload.questions)
        if cancelled.is_set():
            raise CancelledError()
        result["model"] = self.model_id
        result["metadata"] = {
            **result.get("metadata", {}),
            **self.identity(),
            "inference_seconds": time.perf_counter() - started,
        }
        try:
            _finite_json(result)
        except ValueError as error:
            raise RuntimeError("CPU model returned nonfinite decision data") from error
        return result

    async def systemone(self, payload):
        if self.closed:
            raise ServiceBusyError("CPU service is shutting down")
        payload = self.prepare(payload)
        if len(self.pending) >= self.max_pending:
            raise ServiceBusyError("CPU request queue is full")
        loop = asyncio.get_running_loop()
        cancelled = Event()
        worker = self.executor.submit(self._systemone, payload, cancelled)
        self.pending[worker] = cancelled
        worker.add_done_callback(
            lambda completed: loop.call_soon_threadsafe(
                self.pending.pop, completed, None
            )
        )
        future = asyncio.wrap_future(worker)
        future.add_done_callback(_consume)
        try:
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            cancelled.set()
            # Future.cancel() marks queued work done without removing its
            # ThreadPoolExecutor work item. Keep the capacity reserved until
            # the worker consumes it, otherwise repeated disconnects can fill
            # the executor's unbounded queue despite max_pending.
            raise

    def _close_agent(self):
        agent, self.agent = self.agent, None
        close = getattr(agent, "close", None)
        if close is not None:
            close()

    async def close(self):
        self.closed = True
        waiting = list(self.pending.items())
        for worker, cancelled in waiting:
            cancelled.set()
            worker.cancel()
        if waiting:
            await _drain(
                asyncio.gather(
                    *(asyncio.wrap_future(worker) for worker, _ in waiting),
                    return_exceptions=True,
                )
            )
        future = asyncio.wrap_future(self.executor.submit(self._close_agent))
        await _drain(future)
        # Surface runtime cleanup errors once model access has safely ended.
        future.result()


def create_app(
    model_id: str,
    revision: str,
    *,
    backend: str,
    load: Callable[[], Any],
    prepare: Callable[[SystemOneRequest], SystemOneRequest],
    max_pending: int = 16,
    state_name: str = "cpu_service",
) -> FastAPI:
    """Host one model adapter; keep model-specific loading and validation outside."""
    if max_pending < 1:
        raise ValueError("max_pending must be positive")

    @asynccontextmanager
    async def lifespan(app):
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="jev-cpu")
        loading = asyncio.wrap_future(executor.submit(load))
        service = None
        try:
            agent = await asyncio.shield(loading)
            service = WindowsCPUService(
                agent,
                model_id,
                revision,
                executor,
                backend=backend,
                prepare=prepare,
                max_pending=max_pending,
            )
            setattr(app.state, state_name, service)
            yield
        finally:
            setattr(app.state, state_name, None)
            try:
                if service is None:
                    await _drain(loading)
                    if not loading.cancelled() and loading.exception() is None:
                        service = WindowsCPUService(
                            loading.result(),
                            model_id,
                            revision,
                            executor,
                            backend=backend,
                            prepare=prepare,
                        )
                if service is not None:
                    await service.close()
            finally:
                executor.shutdown(wait=True, cancel_futures=True)

    app = FastAPI(title="vLLM Jev — Windows CPU (experimental)", lifespan=lifespan)
    setattr(app.state, state_name, None)

    @app.get("/health")
    async def health():
        service = getattr(app.state, state_name)
        if service is None or service.closed:
            raise HTTPException(status_code=503, detail="Jev engine unavailable")
        return {"status": "ok", **service.identity()}

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": model_id, "object": "model"}]}

    @app.post("/v1/systemone")
    async def system_one(payload: SystemOneRequest, request: Request):
        service = getattr(app.state, state_name)
        if service is None:
            raise HTTPException(status_code=503, detail="Jev engine unavailable")

        stop_watching = asyncio.Event()

        async def disconnected():
            while not stop_watching.is_set():
                if await request.is_disconnected():
                    return
                try:
                    await asyncio.wait_for(stop_watching.wait(), timeout=0.1)
                except TimeoutError:
                    pass

        work = asyncio.create_task(service.systemone(payload))
        disconnect = asyncio.create_task(disconnected())
        try:
            done, _ = await asyncio.wait(
                (work, disconnect), return_when=asyncio.FIRST_COMPLETED
            )
            if work in done:
                return await work
            raise HTTPException(status_code=499, detail="client disconnected")
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except ServiceBusyError as error:
            raise HTTPException(
                status_code=503, detail=str(error), headers={"Retry-After": "1"}
            ) from error
        finally:
            # Starlette probes receive inside an AnyIO cancellation scope, which
            # can consume Task.cancel(). Signal our watcher to stop explicitly.
            stop_watching.set()
            if not work.done():
                work.cancel()
            await asyncio.gather(work, disconnect, return_exceptions=True)

    return app
