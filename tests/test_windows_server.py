"""Shared CPU queue behavior, without loading model runtimes or weights."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace

import pytest

from vllm_jev.windows_server import ServiceBusyError, WindowsCPUService


def test_cancelled_queued_request_holds_capacity_until_consumed():
    async def exercise():
        started, release = Event(), Event()
        calls = []

        class Agent:
            def system_one(self, state, questions):
                calls.append(state)
                if state == "running":
                    started.set()
                    assert release.wait(5), "test did not release the first request"
                return {"answers": {"state": state}}

            def close(self):
                pass

        executor = ThreadPoolExecutor(max_workers=1)
        service = WindowsCPUService(
            Agent(),
            "test",
            "test",
            executor,
            backend="test",
            prepare=lambda payload: payload,
            max_pending=2,
        )
        tasks = []

        def request(state):
            return service.systemone(SimpleNamespace(state=state, questions={}))

        try:
            running = asyncio.create_task(request("running"))
            tasks.append(running)
            assert await asyncio.to_thread(started.wait, 2)
            queued = asyncio.create_task(request("cancelled"))
            tasks.append(queued)
            await asyncio.sleep(0)
            queued.cancel()
            with pytest.raises(asyncio.CancelledError):
                await queued
            # Allow any Future completion callbacks to run before testing the
            # limit: cancellation must not return capacity prematurely.
            await asyncio.sleep(0)
            for _ in range(5):
                with pytest.raises(ServiceBusyError, match="queue is full"):
                    await asyncio.wait_for(request("must not be queued"), 0.5)
            assert calls == ["running"]

            release.set()
            assert (await running)["answers"] == {"state": "running"}

            async def wait_until_consumed():
                while service.pending:
                    await asyncio.sleep(0)

            await asyncio.wait_for(wait_until_consumed(), 2)
            result = await asyncio.wait_for(request("recovered"), 2)
            assert result["answers"] == {"state": "recovered"}
            assert calls == ["running", "recovered"]
        finally:
            release.set()
            await asyncio.gather(*tasks, return_exceptions=True)
            await service.close()
            executor.shutdown(wait=True, cancel_futures=True)

    asyncio.run(exercise())
