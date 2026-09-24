"""Loopback load balancer for independent vLLM Jev replicas.

Use one model replica per GPU. Explicit cache salts select a stable replica;
ordinary independent decisions are distributed round-robin.
"""

import argparse
import hashlib
import itertools
from contextlib import asynccontextmanager

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response


def select_replica(payload, count, sequence):
    if not isinstance(payload, dict):
        payload = {}
    salt = payload.get("cache_salt")
    if isinstance(salt, str) and salt:
        return int.from_bytes(hashlib.sha256(salt.encode()).digest()[:8], "big") % count
    return next(sequence) % count


def create_app(upstreams):
    if not upstreams:
        raise ValueError("at least one upstream is required")
    upstreams = [url.rstrip("/") for url in upstreams]
    sequence = itertools.count()

    @asynccontextmanager
    async def lifespan(app):
        async with httpx.AsyncClient(timeout=120, trust_env=False) as client:
            app.state.client = client
            yield

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    async def health(request: Request):
        statuses = []
        for url in upstreams:
            try:
                response = await request.app.state.client.get(url + "/health")
                statuses.append(response.status_code == 200)
            except httpx.RequestError:
                statuses.append(False)
        if not any(statuses):
            raise HTTPException(503, "all Jev replicas unavailable")
        return {"upstreams": upstreams, "healthy": statuses}

    async def proxy(request: Request, payload, path):
        first = select_replica(payload, len(upstreams), sequence)
        body = await request.body()
        for offset in range(len(upstreams)):
            index = (first + offset) % len(upstreams)
            try:
                result = await request.app.state.client.post(
                    upstreams[index] + path, content=body,
                    headers={"content-type": "application/json"},
                )
            except httpx.RequestError:
                continue
            # Preserve application errors rather than retrying malformed requests.
            return Response(content=result.content, status_code=result.status_code,
                            media_type="application/json",
                            headers={"x-jev-replica": str(index)})
        raise HTTPException(503, "all Jev replicas unavailable")

    @app.post("/plugins/vllm-jev/choice")
    async def choice(request: Request):
        payload = await request.json()
        return await proxy(request, payload, "/plugins/vllm-jev/choice")

    @app.post("/plugins/vllm-jev/batch")
    async def batch(request: Request):
        payload = await request.json()
        return await proxy(request, payload, "/plugins/vllm-jev/batch")

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", action="append", required=True,
                        help="loopback URL of one Jev server; repeat per GPU")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8793)
    args = parser.parse_args()
    uvicorn.run(create_app(args.upstream), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
