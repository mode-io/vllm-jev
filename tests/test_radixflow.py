"""CPU regression tests for scheduling safety and bounded lifecycle."""

import importlib
import sys
import tempfile
import types
import unittest
from collections import deque
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

from vllm_jev.radixflow.adapter import adapt_schedule
from vllm_jev.radixflow.launch import configure
from vllm_jev.radixflow.policy import Admission, Config, RoundState


class PolicyTests(unittest.TestCase):
    def test_admission_early_release_and_deadline(self):
        requests = [NS(request_id=f"r{i}") for i in range(8)]
        for n in (1, 2, 4, 8):
            a = Admission(Config(policy="flex"))
            for r in requests[:n]:
                a.add(r.request_id, 0)
            held = a.snapshot(requests[:n], False, 16, 8192, lambda r: 16, 0.001)
            self.assertEqual(len(held), n)
            self.assertEqual(
                a.snapshot(requests[:n], True, 16, 8192, lambda r: 16, 0.001), set()
            )
        a = Admission(Config(policy="timer10"))
        a.add("r0", 0)
        self.assertEqual(
            a.snapshot(requests[:1], False, 16, 8192, lambda r: 16, 0.009), {"r0"}
        )
        self.assertEqual(
            a.snapshot(requests[:1], False, 16, 8192, lambda r: 16, 0.010), set()
        )

    def test_token_budget_release_and_cancellation(self):
        a = Admission(Config())
        r = NS(request_id="r")
        a.add("r", 0)
        self.assertEqual(a.snapshot([r], False, 64, 100, lambda r: 100, 0.001), set())
        a.remove("r")
        self.assertFalse(a.arrivals)
        self.assertFalse(a.released)

    def test_unknown_producer_falls_back_and_defer_is_bounded(self):
        s = RoundState(Config(max_deferrals=2))
        s.guards["A"] = [dict(producer=None)]
        self.assertFalse(s.reject("A", {"P": 10}, "allocation"))
        for expect in (True, True, False):
            s.begin()
            s.guards["A"] = [dict(producer="P")]
            self.assertEqual(s.reject("A", {"P": 10}, "allocation"), expect)
        self.assertEqual(s.deferrals["A"], 2)

    def test_recovery_requires_same_cut_and_no_prior_fragment(self):
        s = RoundState(Config())
        s.guards["A"] = [dict(producer="P")]
        self.assertTrue(s.reject("A", {"P": 10}, "allocation"))
        self.assertEqual(s.finish({"P": 10, "B": 10})["within_round_recovered"], ["B"])
        self.assertEqual(s.finish({"B": 10})["within_round_recovered"], [])
        s.begin()
        s.fragment_blocked.add("X")
        s.reject("X", {"P": 10}, "zero_tokens")
        s.guards["A"] = [dict(producer="P")]
        s.reject("A", {"P": 10}, "allocation")
        self.assertEqual(s.finish({"P": 10, "B": 10})["within_round_recovered"], [])

    def test_bad_native_structure_rejected(self):
        def wrong(self):
            return None

        with self.assertRaises(RuntimeError):
            adapt_schedule(wrong)

    def test_launcher_default_and_conflicts(self):
        with tempfile.NamedTemporaryFile() as f:
            args = NS(
                radixflow_policy="radixflow",
                radixflow_library=Path(f.name),
                radixflow_trace_dir=None,
                radixflow_wait_ms=5,
            )
            env = {"VLLM_PLUGINS": "other"}
            flags = configure(args, [], env)
            self.assertIn("--no-async-scheduling", flags)
            self.assertEqual(env["VLLM_PLUGINS"], "other,vllm_jev_model")
            with self.assertRaises(ValueError):
                configure(args, ["--async-scheduling"], {})
            args.radixflow_policy = "off"
            self.assertEqual(configure(args, [], env), [])
            self.assertNotIn("VLLM_JEV_RADIXFLOW_POLICY", env)


class Queue:
    def __init__(self):
        self.q = deque()

    def __bool__(self):
        return bool(self.q)

    def __len__(self):
        return len(self.q)

    def __iter__(self):
        return iter(self.q)

    def peek_request(self):
        return self.q[0]

    def pop_request(self):
        return self.q.popleft()

    def prepend_request(self, r):
        self.q.appendleft(r)

    def prepend_requests(self, q):
        self.q.extendleft(reversed(list(q)))


class Manager:
    def __init__(self):
        self.cached_blocks_this_step = set()
        self.req_to_blocks = {}
        self.kv_cache_group_id = 1
        self.block_pool = NS(num_gpu_blocks=32)

    def get_num_blocks_to_allocate(self, request_id, n, new_computed_blocks):
        if (
            new_computed_blocks
            and new_computed_blocks[-1].block_hash in self.cached_blocks_this_step
        ):
            return 33
        return 1

    def cache_blocks(self, request, n):
        block = NS(is_null=False, block_hash=request.prefix, block_id=3)
        self.req_to_blocks[request.request_id] = [block]
        self.cached_blocks_this_step.add(request.prefix)


class NativeScheduler:
    def __init__(self, config):
        self.vllm_config = config
        self.scheduler_config = config.scheduler_config
        self.connector = self.ec_connector = None
        self.running = []
        self.waiting = Queue()
        self.requests = {}
        self.max_num_running_reqs = 16
        self.max_num_scheduled_tokens = 8192
        self.manager = Manager()
        self.kv_cache_manager = NS(coordinator=NS(single_type_managers=[self.manager]))
        self.encoder_cache_manager = NS(free=lambda r: None)

    def add_request(self, request):
        self.requests[request.request_id] = request
        self.waiting.q.append(request)

    def schedule(self):
        self.manager.cached_blocks_this_step.clear()
        step_skipped_waiting = Queue()
        num_scheduled_tokens = {}
        while self.waiting:
            request_queue = self.waiting
            request = request_queue.peek_request()
            new_computed_blocks = [NS(block_hash=request.prefix, block_id=3)]
            new_blocks = (
                None
                if self.manager.get_num_blocks_to_allocate(
                    request.request_id, 10, new_computed_blocks
                )
                == 33
                else [1]
            )
            if new_blocks is None:
                if request.has_encoder_inputs:
                    self.encoder_cache_manager.free(request)
                break
            num_scheduled_tokens[request.request_id] = 10
            self.manager.cache_blocks(request, 10)
            request_queue.pop_request()
        self.waiting.prepend_requests(step_skipped_waiting)
        return NS(
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=sum(num_scheduled_tokens.values()),
        )

    def _free_request(self, request):
        self.requests.pop(request.request_id, None)

    def _mamba_block_aligned_split(self, request, n, local=0, external=0):
        return n

    def shutdown(self):
        pass


class FakeIndex:
    def __init__(self, *a):
        pass

    def observe(self, *a):
        return 0

    def close(self):
        pass

    def stats(self):
        return {}


class SchedulerTests(unittest.TestCase):
    def setUp(self):
        modules = {}
        for name in [
            "vllm",
            "vllm.v1",
            "vllm.v1.core",
            "vllm.v1.core.sched",
            "vllm.v1.core.sched.scheduler",
            "vllm.v1.core.single_type_kv_cache_manager",
        ]:
            modules[name] = types.ModuleType(name)
        modules["vllm.v1.core.sched.scheduler"].Scheduler = NativeScheduler
        modules["vllm.v1.core.single_type_kv_cache_manager"].MambaManager = Manager
        self.module_patch = patch.dict(sys.modules, modules)
        self.module_patch.start()
        sys.modules.pop("vllm_jev.radixflow.scheduler", None)
        self.mod = importlib.import_module("vllm_jev.radixflow.scheduler")
        self.index_patch = patch.object(self.mod, "RustIndex", FakeIndex)
        self.index_patch.start()
        self.version_patch = patch.object(
            self.mod.importlib.metadata, "version", return_value="0.29.0"
        )
        self.version_patch.start()

    def tearDown(self):
        self.version_patch.stop()
        self.index_patch.stop()
        self.module_patch.stop()
        sys.modules.pop("vllm_jev.radixflow.scheduler", None)

    def make(self, policy):
        config = NS(
            scheduler_config=NS(async_scheduling=False),
            cache_config=NS(enable_prefix_caching=True),
            model_config=NS(runner_type="generate"),
        )
        with patch.dict("os.environ", {"VLLM_JEV_RADIXFLOW_POLICY": policy}):
            return self.mod.RadixFlowScheduler(config)

    def add(self, s, rid, prefix):
        r = NS(
            request_id=rid,
            prefix=prefix,
            resumable=False,
            prompt_token_ids=[1, 2],
            cache_salt="tenant",
            lora_request=None,
            has_encoder_inputs=False,
            status="FINISHED",
            num_tokens=10,
            num_prompt_tokens=10,
            num_computed_tokens=0,
        )
        s.add_request(r)
        s.rf_admission.released.add(rid)
        return r

    def test_real_guard_producer_binding_and_next_round_progress(self):
        s = self.make("radixflow")
        self.add(s, "P", "shared")
        consumer = self.add(s, "A", "shared")
        self.add(s, "B", "independent")
        out = s.schedule()
        self.assertEqual(list(out.num_scheduled_tokens), ["P", "B"])
        self.assertEqual(s.rf_last["within_round_recovered"], ["B"])
        self.assertEqual([r.request_id for r in s.waiting], ["A"])
        self.assertEqual(list(s.schedule().num_scheduled_tokens), ["A"])
        s._free_request(consumer)
        self.assertNotIn("A", s.rf_admission.arrivals)
        self.assertNotIn("A", s.rf_state.deferrals)

    def test_radix_preserves_native_stop_without_global_patch(self):
        original = NativeScheduler.schedule
        s = self.make("radix")
        for rid, prefix in [("P", "shared"), ("A", "shared"), ("B", "other")]:
            self.add(s, rid, prefix)
        self.assertEqual(list(s.schedule().num_scheduled_tokens), ["P"])
        self.assertIs(NativeScheduler.schedule, original)
        self.assertEqual([r.request_id for r in s.waiting], ["A", "B"])

    def test_unknown_state_guard_does_not_skip(self):
        s = self.make("radixflow")
        s.rf_state.guards["A"] = [dict(producer=None)]
        self.assertFalse(s._radixflow_reject(NS(request_id="A"), "allocation", {}))


class RustIndexTests(unittest.TestCase):
    @unittest.skipUnless(
        __import__("os").getenv("RADIXFLOW_TEST_LIBRARY"), "Rust library required"
    )
    def test_namespace_bounds_and_exact_matches(self):
        import os

        from vllm_jev.radixflow.index import RustIndex

        index = RustIndex(os.environ["RADIXFLOW_TEST_LIBRARY"], 2, 20)
        self.assertEqual(index.observe("one", [1, 2, 3]), 0)
        self.assertEqual(index.observe("one", [1, 2, 4]), 2)
        self.assertEqual(index.observe("two", [1, 2, 4]), 0)
        index.observe("three", [1, 2])
        self.assertEqual(index.stats()["namespaces"], 2)
        self.assertEqual(index.observe("one", [1, 2]), 0)
        self.assertIsNone(index.observe("one", list(range(21))))
        with self.assertRaises(ValueError):
            index.observe("bad", [-1])
        index.close()
        self.assertEqual(index.stats()["namespaces"], 0)


class CopyGuardTests(unittest.TestCase):
    def test_dense_alias_coverage_and_bad_stride(self):
        try:
            import torch
        except ImportError:
            self.skipTest("CPU torch required")
        from vllm_jev.radixflow.extent_guard import UnsupportedGeometry, covering_views

        full = torch.arange(3 * 64).reshape(3, 64)
        chosen, proof = covering_views([full[:, :8], full], 3)
        self.assertEqual(len(chosen), 1)
        self.assertEqual(chosen[0].shape, full.shape)
        self.assertEqual(proof[0]["bytes"], 64 * full.element_size())
        with self.assertRaises(UnsupportedGeometry):
            covering_views([full[:, ::2]], 3)


class TelemetryTests(unittest.TestCase):
    def test_actual_executor_boundary_and_empty_cleanup(self):
        from collections import Counter

        from vllm_jev.radixflow import telemetry

        class Engine:
            def __init__(self, scheduler):
                self.scheduler = scheduler
                self.batch_queue = None
                self.model_executor = NS(execute_model=lambda output: "done")

            def step(self, output):
                return self.model_executor.execute_model(output)

        modules = {
            name: types.ModuleType(name)
            for name in ["vllm", "vllm.v1", "vllm.v1.engine", "vllm.v1.engine.core"]
        }
        modules["vllm.v1.engine.core"].EngineCore = Engine
        s = NS(
            rf_config=Config(),
            rf_state=NS(seq=7),
            rf_trace=telemetry.Trace(),
            rf_counters=Counter(),
            requests={"r": NS(num_computed_tokens=10, num_prompt_tokens=10)},
        )
        with (
            patch.dict(sys.modules, modules),
            patch.object(telemetry, "_installed", False),
        ):
            engine = Engine(s)  # Constructor already exists when plugins load.
            telemetry.install_executor_metrics()
            execute = engine.step
            self.assertEqual(
                execute(
                    NS(num_scheduled_tokens={"r": 10}, total_num_scheduled_tokens=10)
                ),
                "done",
            )
            execute(NS(num_scheduled_tokens={}, total_num_scheduled_tokens=0))
            self.assertEqual(s.rf_counters["global_nonempty_calls"], 1)
            self.assertEqual(s.rf_counters["global_empty_calls"], 1)
            self.assertGreater(s.rf_counters["host_execute_ns"], 0)


class FutureTelemetryTests(unittest.TestCase):
    def test_future_completion_retains_result_and_records_once(self):
        from collections import Counter
        from concurrent.futures import Future

        from vllm_jev.radixflow import telemetry

        events = []
        future = Future()
        future.set_result("native-result")

        class Engine:
            def __init__(self, s):
                self.scheduler = s
                self.batch_queue = None
                self.model_executor = NS(execute_model=lambda out: future)

            def step(self, out):
                result = self.model_executor.execute_model(out)
                self.asserted_result = result
                return result.result()

        modules = {
            name: types.ModuleType(name)
            for name in ["vllm", "vllm.v1", "vllm.v1.engine", "vllm.v1.engine.core"]
        }
        modules["vllm.v1.engine.core"].EngineCore = Engine
        s = NS(
            rf_config=Config(),
            rf_state=NS(seq=3),
            rf_trace=NS(emit=lambda kind, value: events.append((kind, value))),
            rf_counters=Counter(),
            requests={"r": NS(num_computed_tokens=10, num_prompt_tokens=10)},
        )
        with (
            patch.dict(sys.modules, modules),
            patch.object(telemetry, "_installed", False),
        ):
            engine = Engine(s)
            telemetry.install_executor_metrics()
            self.assertEqual(
                engine.step(
                    NS(num_scheduled_tokens={"r": 10}, total_num_scheduled_tokens=10)
                ),
                "native-result",
            )
            self.assertIs(engine.asserted_result, future)
            self.assertEqual(future.result(), "native-result")
            done = [v for k, v in events if k == "execute_complete"]
            self.assertEqual(len(done), 1)
            self.assertGreaterEqual(
                done[0]["host_completion_ns"], done[0]["host_enqueue_ns"]
            )


class NativeObserverTests(unittest.TestCase):
    def test_native_scan_unchanged_by_read_only_observer(self):
        import json

        from vllm_jev.radixflow import telemetry

        class Engine:
            def __init__(self, scheduler):
                self.scheduler = scheduler
                self.batch_queue = None
                self.model_executor = NS(execute_model=lambda out: "done")

            def step(self):
                out = self.scheduler.schedule()
                for rid, n in out.num_scheduled_tokens.items():
                    self.scheduler.requests[rid].num_computed_tokens += n
                self.model_executor.execute_model(out)
                return out

        modules = {
            name: types.ModuleType(name)
            for name in ["vllm", "vllm.v1", "vllm.v1.engine", "vllm.v1.engine.core"]
        }
        modules["vllm.v1.engine.core"].EngineCore = Engine
        scheduler = NativeScheduler(NS(scheduler_config=NS(async_scheduling=False)))
        for rid, prefix in [("P", "shared"), ("A", "shared"), ("B", "other")]:
            scheduler.add_request(
                NS(
                    request_id=rid,
                    prefix=prefix,
                    has_encoder_inputs=False,
                    num_computed_tokens=0,
                    num_prompt_tokens=10,
                    cache_salt="namespace",
                )
            )
        original = NativeScheduler.schedule
        with (
            tempfile.TemporaryDirectory() as d,
            patch.dict(sys.modules, modules),
            patch.dict("os.environ", {"VLLM_JEV_NATIVE_TRACE_DIR": d}),
            patch.object(telemetry, "_installed", False),
        ):
            telemetry.install_executor_metrics()
            out = Engine(scheduler).step()
            self.assertEqual(list(out.num_scheduled_tokens), ["P"])
            self.assertIs(NativeScheduler.schedule, original)
            self.assertFalse(hasattr(scheduler, "rf_config"))
            records = [
                json.loads(line)
                for f in Path(d).glob("*.jsonl")
                for line in f.read_text().split("\n")
                if line
            ]
            self.assertEqual(
                [x for x in records if x["kind"] == "execute"][0]["rows"][0][
                    "namespace"
                ],
                "namespace",
            )
            self.assertTrue(
                [x for x in records if x["kind"] == "capabilities"][0]["observer_only"]
            )
            scheduler.rf_trace.close()


if __name__ == "__main__":
    unittest.main()
