"""Windows CPU decision adapters hosted through the shared HTTP service."""

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
    from . import windows_tiny

    if model_id in windows_tiny.MODELS:
        adapter, backend = windows_tiny, "tiny-jev-pytorch"
    else:
        from . import windows_decider, windows_open_jev, windows_openjev

        if model_id == windows_openjev.MODEL_ID:
            adapter, backend = windows_openjev, "openjev-pytorch"
        elif model_id == windows_open_jev.MODEL_ID:
            adapter, backend = windows_open_jev, "open-jev-pytorch"
        elif model_id in windows_decider.MODELS:
            adapter, backend = windows_decider, "decider-pytorch"
        else:
            from .windows_models import _adapter_model_ids, _additional_adapters

            adapters = {
                model: item
                for item in _additional_adapters()
                for model in _adapter_model_ids(item)
            }
            if model_id not in adapters:
                raise ValueError("unsupported Windows CPU model")
            adapter = adapters[model_id]
            backend = {
                "decision_gguf_v1": "decision-llama-cpp-0.3.35",
                "bruv_direct_options_v1": "bruv-pytorch",
                "rsi_option_xattn_v1": "rsi-pytorch",
                "decider_gguf_v1": "decider-llama-cpp-q4-0.3.35",
                "jevk5_letters_v1": "jevk5-pytorch",
                "jevk5_gguf_v1": "jevk5-llama-cpp-q5-b11010",
                "jevk5_lite_v1": "jevk5-lite-pytorch",
                "kev_pointer_v1": "kev-pytorch",
                "thisthat_slot_v1": "thisthat-pytorch",
                "mica_gguf_v1": "mica-llama-cpp-q5-b11010",
            }[adapter.PROTOCOL]
            if adapter.PROTOCOL in (
                "mica_gguf_v1",
                "jevk5_lite_v1",
                "jevk5_gguf_v1",
                "decider_gguf_v1",
                "decision_gguf_v1",
                "bruv_direct_options_v1",
                "rsi_option_xattn_v1",
            ):
                adapter.validate_template()
    if threads is not None and threads < 1:
        raise ValueError("threads must be positive")

    def load():
        if agent_factory is not None:
            return agent_factory()
        return adapter.load_agent(checkpoint, model_id, threads)

    revision = (
        adapter.MODELS[model_id]["revision"]
        if hasattr(adapter, "MODELS")
        else adapter.REVISION
    )
    return create_cpu_app(
        model_id,
        revision,
        backend=backend,
        load=load,
        prepare=lambda payload: prepare_request(
            payload, model_id, adapter.PROTOCOL, ("open-jev", "jev-latest")
        ),
        max_pending=max_pending,
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
