"""Windows Laya bindings for the shared native CPU HTTP service."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

from .windows_server import create_app as create_cpu_app
from .windows_server import prepare_request


def create_app(
    checkpoint: Path,
    model_id: str,
    *,
    threads: int | None = None,
    agent_factory: Callable[[], Any] | None = None,
    max_pending: int = 16,
):
    from .laya_export import MODELS

    if model_id not in MODELS:
        raise ValueError(
            "the Laya CPU backend supports only the three pinned Laya releases"
        )
    if threads is not None and threads < 1:
        raise ValueError("threads must be positive")

    def load():
        if agent_factory is not None:
            return agent_factory()
        from .windows_models import load_agent

        return load_agent(checkpoint, model_id, threads)

    return create_cpu_app(
        model_id,
        MODELS[model_id][0],
        backend="laya-pytorch",
        load=load,
        prepare=lambda payload: prepare_request(
            payload, model_id, "laya_markers_v1", ("laya",)
        ),
        max_pending=max_pending,
        state_name="laya_service",
    )


def serve(
    checkpoint: Path,
    model_id: str,
    *,
    host: str = "127.0.0.1",
    port: int = 8795,
    threads: int | None = None,
) -> None:
    import uvicorn

    uvicorn.run(
        create_app(checkpoint, model_id, threads=threads),
        host=host,
        port=port,
        workers=1,
        access_log=False,
    )
