"""Actual global executor-call telemetry; host intervals are not GPU kernel time."""

import functools
import json
import os
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace


class Trace:
    def __init__(self, directory=None):
        self.file = None
        if directory:
            root = Path(directory)
            root.mkdir(parents=True, exist_ok=True)
            self.file = (root / f"radixflow-{os.getpid()}.jsonl").open("a", buffering=1)

    def emit(self, kind, payload):
        if self.file:
            self.file.write(
                json.dumps(
                    dict(kind=kind, pid=os.getpid(), **payload), separators=(",", ":")
                )
                + "\n"
            )

    def close(self):
        if self.file:
            self.file.close()
            self.file = None


_installed = False


def install_executor_metrics():
    """Enabled only by the RadixFlow launcher; attach to each private EngineCore."""
    global _installed
    if _installed:
        return
    from vllm.v1.engine.core import EngineCore

    old_step = EngineCore.step

    def attach(core):
        scheduler = core.scheduler
        if getattr(core, "_radixflow_executor_wrapped", False):
            return
        native_directory = os.getenv("VLLM_JEV_NATIVE_TRACE_DIR")
        if not hasattr(scheduler, "rf_config") and not native_directory:
            return
        if not hasattr(scheduler, "rf_config"):
            # Read-only native baseline: no index, admission, state or copy changes.
            scheduler.rf_state = SimpleNamespace(seq=0)
            scheduler.rf_counters = Counter()
            scheduler.rf_trace = Trace(native_directory)
            native_schedule = scheduler.schedule

            @functools.wraps(native_schedule)
            def observed_schedule(*a, **k):
                scheduler.rf_state.seq += 1
                out = native_schedule(*a, **k)
                scheduler.rf_trace.emit(
                    "round",
                    dict(
                        seq=scheduler.rf_state.seq,
                        selected=dict(out.num_scheduled_tokens),
                        tokens=out.total_num_scheduled_tokens,
                        events=[],
                        observer_only=True,
                    ),
                )
                return out

            scheduler.schedule = observed_schedule
            native_free = scheduler._free_request

            @functools.wraps(native_free)
            def observed_free(request, *a, **k):
                value = native_free(request, *a, **k)
                scheduler.rf_trace.emit(
                    "request_end",
                    dict(
                        id=request.request_id,
                        status=str(request.status),
                        observer_only=True,
                    ),
                )
                return value

            scheduler._free_request = observed_free
            scheduler.rf_trace.emit(
                "capabilities",
                dict(
                    policy="native_vllm",
                    scheduler=type(scheduler).__module__
                    + "."
                    + type(scheduler).__name__,
                    observer_only=True,
                ),
            )
        if core.batch_queue is not None:
            raise ValueError("RadixFlow requires a synchronous single-batch EngineCore")
        old_execute = core.model_executor.execute_model

        @functools.wraps(old_execute)
        def execute(output, *args, **kwargs):
            rows = []
            for rid, n in output.num_scheduled_tokens.items():
                req = scheduler.requests[rid]
                rows.append(
                    dict(
                        id=rid,
                        n=n,
                        start=req.num_computed_tokens - n,
                        prompt=req.num_prompt_tokens,
                        namespace=getattr(req, "cache_salt", None),
                    )
                )
            seq = scheduler.rf_state.seq
            started = time.perf_counter_ns()
            result = old_execute(output, *args, **kwargs)
            elapsed = time.perf_counter_ns() - started
            nonempty = bool(output.total_num_scheduled_tokens)
            scheduler.rf_counters[
                "global_nonempty_calls" if nonempty else "global_empty_calls"
            ] += 1
            scheduler.rf_counters["host_execute_ns"] += elapsed if nonempty else 0
            prefill = any(r["start"] < r["prompt"] for r in rows)
            decode = any(r["start"] + r["n"] > r["prompt"] for r in rows)
            phase = (
                "mixed" if prefill and decode else "prefill" if prefill else "decode"
            )
            scheduler.rf_trace.emit(
                "execute",
                dict(
                    seq=seq,
                    tokens=output.total_num_scheduled_tokens,
                    rows=rows,
                    host_ns=elapsed,
                    nonempty=nonempty,
                    phase=phase,
                ),
            )
            measured = False

            def completed():
                nonlocal measured
                if measured:
                    return
                measured = True
                total = time.perf_counter_ns() - started
                scheduler.rf_counters["host_completion_ns"] += total if nonempty else 0
                scheduler.rf_trace.emit(
                    "execute_complete",
                    dict(
                        seq=seq,
                        nonempty=nonempty,
                        host_enqueue_ns=elapsed,
                        host_completion_ns=total,
                    ),
                )

            if callable(getattr(result, "result", None)):
                native_result = result.result

                @functools.wraps(native_result)
                def wait_result(*a, **k):
                    value = native_result(*a, **k)
                    completed()
                    return value

                result.result = wait_result
            else:
                completed()
            return result

        core.model_executor.execute_model = execute
        core._radixflow_executor_wrapped = True

    @functools.wraps(old_step)
    def step(core, *args, **kwargs):
        # Plugins may load inside an already-running constructor. Hook the first
        # actual step rather than relying on replacing that constructor.
        attach(core)
        scheduler = core.scheduler
        if not hasattr(scheduler, "rf_state"):
            return old_step(core, *args, **kwargs)
        previous = scheduler.rf_state.seq
        started = time.perf_counter_ns()
        result = old_step(core, *args, **kwargs)
        if scheduler.rf_state.seq != previous:
            scheduler.rf_trace.emit(
                "core_step",
                dict(
                    seq=scheduler.rf_state.seq,
                    host_ns=time.perf_counter_ns() - started,
                    scope="schedule, executor completion, sampling, abort processing, output update",
                ),
            )
        return result

    EngineCore.step = step
    _installed = True


def install_copy_guard():
    # Use the same complete-state copy rule for every custom comparison mode.
    from vllm.v1.worker.gpu import model_runner

    from .extent_guard import guarded_copy

    old = model_runner.copy_kv_cache_blocks_inplace
    if getattr(old, "_radixflow_guard", False):
        return

    @functools.wraps(old)
    def copy(views, num_blocks, pairs):
        guarded_copy(views, num_blocks, pairs, old)

    copy._radixflow_guard = True
    model_runner.copy_kv_cache_blocks_inplace = copy
