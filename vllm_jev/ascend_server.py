"""Endpoint bridge for vLLM 0.22.1, before native endpoint plugins existed.

The bridge is local to this dedicated server process. It preserves vLLM's app,
authentication, cancellation, engine startup and shutdown implementations.
"""

import functools


def install_endpoint_bridge(server, plugin):
    if getattr(server, "_jev_endpoint_bridge", False):
        return
    original_build = server.build_app
    original_init = server.init_app_state

    @functools.wraps(original_build)
    def build(*args, **kwargs):
        app = original_build(*args, **kwargs)
        plugin.attach_router(app)
        return app

    @functools.wraps(original_init)
    async def init(engine_client, state, args, *rest, **kwargs):
        await original_init(engine_client, state, args, *rest, **kwargs)
        if not hasattr(engine_client, "model_config"):
            engine_client = EngineView(engine_client)
        await plugin.init_state(engine_client, state, args)

    server.build_app = build
    server.init_app_state = init
    server._jev_endpoint_bridge = True


class EngineView:
    def __init__(self, engine):
        self._engine = engine
        self.model_config = engine.vllm_config.model_config

    def __getattr__(self, name):
        return getattr(self._engine, name)


def validate_api_count(arguments):
    for i, value in enumerate(arguments):
        if value == "--api-server-count":
            if i + 1 >= len(arguments) or arguments[i + 1] != "1":
                raise ValueError(
                    "Legacy Ascend endpoint bridge requires one API-server process"
                )
        if value.startswith("--api-server-count=") and value != "--api-server-count=1":
            raise ValueError(
                "Legacy Ascend endpoint bridge requires one API-server process"
            )


def main():
    import importlib.metadata

    if importlib.metadata.version("vllm").split("+", 1)[0] != "0.22.1":
        raise RuntimeError("Legacy Ascend endpoint bridge requires vLLM 0.22.1")
    import sys

    validate_api_count(sys.argv[1:])
    from vllm.entrypoints.openai import api_server

    from .endpoint import JevEndpointPlugin

    install_endpoint_bridge(api_server, JevEndpointPlugin())
    from vllm.entrypoints.cli.main import main as native_main

    native_main()


if __name__ == "__main__":
    main()
