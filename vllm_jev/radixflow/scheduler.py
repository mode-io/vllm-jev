"""Explicit custom Scheduler for synchronous vLLM 0.29; no global scheduler patch."""

import functools
import importlib.metadata
import os
import time
from collections import Counter
from types import MethodType

from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import MambaManager

from .adapter import adapt_schedule
from .index import RustIndex
from .policy import Admission, Config, RoundState
from .telemetry import Trace


class RadixFlowScheduler(Scheduler):
    def __init__(self, *args, **kwargs):
        if importlib.metadata.version("vllm") != "0.29.0":
            raise RuntimeError("RadixFlow requires the validated vLLM 0.29.0 adapter")
        if os.getenv("VLLM_JEV_ONLINE_PREFIX_CACHE") == "1":
            raise ValueError(
                "RadixFlow cannot be combined with online checkpoint creation"
            )
        super().__init__(*args, **kwargs)
        if self.scheduler_config.async_scheduling:
            raise ValueError("RadixFlow currently requires --no-async-scheduling")
        if (
            self.connector is not None
            or getattr(self, "ec_connector", None) is not None
        ):
            raise ValueError("RadixFlow does not yet support remote cache connectors")
        if getattr(self, "use_pp", False):
            raise ValueError("RadixFlow requires pipeline_parallel_size=1")
        if not self.vllm_config.cache_config.enable_prefix_caching:
            raise ValueError("RadixFlow requires native prefix caching")
        self.rf_config = Config.environment()
        self.rf_state = RoundState(self.rf_config)
        self.rf_admission = Admission(self.rf_config)
        self.rf_index = RustIndex(
            self.rf_config.library,
            self.rf_config.max_namespaces,
            self.rf_config.max_tokens,
        )
        self.rf_trace = Trace(self.rf_config.trace_dir)
        self.rf_counters = Counter()
        self.rf_held = set()
        self.rf_last = {}
        self.rf_native_schedule, adapter = adapt_schedule(Scheduler.schedule)
        if (
            self.rf_config.policy == "radixflow_frag"
            and not adapter["zero_token_sites"]
        ):
            raise RuntimeError("fragment policy unsupported by this native scheduler")
        self._instrument_managers()
        if self.vllm_config.model_config.runner_type == "pooling":
            self.num_sampled_tokens_per_step = 0
        self.rf_trace.emit(
            "capabilities",
            dict(
                policy=self.rf_config.policy,
                adapter=adapter,
                mamba_groups=self.rf_mamba_groups,
                index="exact_rust",
                async_scheduling=False,
                state_storage="native",
                copy_guard="complete_extent",
            ),
        )

    def _instrument_managers(self):
        self.rf_mamba_groups = 0
        for manager in self.kv_cache_manager.coordinator.single_type_managers:
            if not isinstance(manager, MambaManager):
                continue
            self.rf_mamba_groups += 1
            old_cache = manager.cache_blocks
            old_number = manager.get_num_blocks_to_allocate
            if not hasattr(manager, "cached_blocks_this_step"):
                raise RuntimeError("native current-round state guard is missing")

            def cache(m, request, *a, _old=old_cache, **k):
                value = _old(request, *a, **k)
                for block in m.req_to_blocks.get(request.request_id, ()):
                    h = block.block_hash
                    if (
                        not block.is_null
                        and h is not None
                        and h in m.cached_blocks_this_step
                    ):
                        self.rf_state.producers.setdefault(
                            (m.kv_cache_group_id, h), request.request_id
                        )
                return value

            def number(m, *a, _old=old_number, **k):
                rid = k.get("request_id", a[0] if a else None)
                blocks = k.get("new_computed_blocks", a[2] if len(a) > 2 else ())
                guarded = bool(
                    blocks and blocks[-1].block_hash in m.cached_blocks_this_step
                )
                value = _old(*a, **k)
                if guarded and value == m.block_pool.num_gpu_blocks + 1:
                    block = blocks[-1]
                    self.rf_state.guards.setdefault(rid, []).append(
                        dict(
                            group=m.kv_cache_group_id,
                            block=block.block_id,
                            producer=self.rf_state.producers.get(
                                (m.kv_cache_group_id, block.block_hash)
                            ),
                        )
                    )
                return value

            manager.cache_blocks = MethodType(
                functools.wraps(old_cache)(cache), manager
            )
            manager.get_num_blocks_to_allocate = MethodType(
                functools.wraps(old_number)(number), manager
            )

    def add_request(self, request):
        if request.resumable:
            raise ValueError("RadixFlow streaming input sessions are not yet supported")
        super().add_request(
            request
        )  # Registered before it can be deferred or cancelled.
        self.rf_admission.add(request.request_id, time.monotonic())
        ids = request.prompt_token_ids or ()
        namespace = (
            request.cache_salt,
            getattr(request.lora_request, "lora_int_id", 0),
        )
        try:
            matched = self.rf_index.observe(namespace, ids)
        except (ValueError, RuntimeError, MemoryError):
            # Advisory metadata failures never authorize reuse or strand a request.
            matched = 0
            self.rf_counters["index_errors"] += 1
        self.rf_counters["index_observations"] += 1
        self.rf_counters["index_exact_match_tokens"] += matched or 0

    def _residual(self, request):
        computed = request.num_computed_tokens
        if computed == 0 and self.kv_cache_manager.prefix_cache_lookup_enabled(request):
            _, computed, _ = self.kv_cache_manager.coordinator.find_longest_cache_hit(
                request.block_hashes, request.num_tokens - 1
            )
        return max(
            1, min(self.max_num_scheduled_tokens, request.num_prompt_tokens - computed)
        )

    def _radixflow_hold(self, request):
        return request.request_id in self.rf_held

    def _radixflow_reject(self, request, site, selected):
        return self.rf_state.reject(request.request_id, selected, site)

    def schedule(self, *args, **kwargs):
        self.rf_state.begin()
        waiting = list(self.waiting)
        if hasattr(self, "skipped_waiting"):
            waiting += list(self.skipped_waiting)
        self.rf_held = self.rf_admission.snapshot(
            waiting,
            bool(self.running),
            self.max_num_running_reqs,
            self.max_num_scheduled_tokens,
            self._residual,
            time.monotonic(),
        )
        if waiting and not self.running and len(self.rf_held) == len(waiting):
            # Bound idle polling; control returns to EngineCore for aborts and arrivals.
            time.sleep(0.0002)
        started = time.perf_counter_ns()
        try:
            output = self.rf_native_schedule(self, *args, **kwargs)
        finally:
            self.rf_held.clear()
        ns = time.perf_counter_ns() - started
        self.rf_last = self.rf_state.finish(output.num_scheduled_tokens)
        self.rf_last["scheduler_ns"] = ns
        self.rf_last["tokens"] = output.total_num_scheduled_tokens
        self.rf_last["rows"] = len(output.num_scheduled_tokens)
        self.rf_counters["schedule_calls"] += 1
        self.rf_counters["scheduled_tokens"] += output.total_num_scheduled_tokens
        self.rf_trace.emit("round", self.rf_last)
        return output

    def _mamba_block_aligned_split(self, request, n, local=0, external=0, **kwargs):
        local = kwargs.get("num_new_local_computed_tokens", local)
        external = kwargs.get("num_external_computed_tokens", external)
        result = super()._mamba_block_aligned_split(request, n, local, external)
        if self.rf_config.policy != "radixflow_frag":
            return result
        status = getattr(request.status, "name", str(request.status))
        remaining = request.num_tokens - request.num_computed_tokens - local - external
        if (
            status not in ("WAITING", "PREEMPTED")
            or not 0 < remaining <= self.max_num_scheduled_tokens
        ):
            return result
        ideal = super()._mamba_block_aligned_split(request, remaining, local, external)
        if 0 < result < ideal:
            self.rf_state.fragment_blocked.add(request.request_id)
            return 0
        return result

    def _free_request(self, request, *args, **kwargs):
        value = super()._free_request(request, *args, **kwargs)
        rid = request.request_id
        self.rf_admission.remove(rid)
        self.rf_trace.emit(
            "request_end",
            dict(
                id=rid,
                defers=self.rf_state.deferrals.pop(rid, 0),
                status=str(request.status),
            ),
        )
        return value

    def radixflow_stats(self):
        return dict(
            policy=self.rf_config.policy,
            counters=dict(self.rf_counters),
            dependencies=dict(self.rf_state.totals),
            index=self.rf_index.stats(),
            active_requests=len(self.rf_admission.arrivals),
            admission_releases=dict(self.rf_admission.counts),
        )

    def shutdown(self):
        try:
            return super().shutdown()
        finally:
            self.rf_index.close()
            self.rf_trace.close()
