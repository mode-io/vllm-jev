# RadixFlow engineering integration (experimental)

See [validation record](radixflow-validation.md) for the exercised checks and
[pressure comparisons](radixflow-benchmarks.md) for matched serving measurements.

**Status: experimental, opt-in, disabled by default.**

RadixFlow uses the native vLLM cache allocator and complete hybrid state. It
changes admission and the response to a confirmed current-round dependency.
Select the custom scheduler explicitly; ordinary serving remains unchanged.

## Scope

The adapter targets synchronous vLLM **0.29.0**, pipeline parallelism 1, and
native prefix caching. It works with ordinary generative models and native Jev
pooling models. Tensor parallelism uses the same EngineCore global call boundary.
Full-attention models keep the admission rule while the recurrent guard is
inactive. Remote cache connectors, streaming-input sessions, asynchronous
scheduling, and the older online-checkpoint experiment are rejected explicitly.
An unrecognized native scheduler structure is rejected at startup.

This integration has CPU regression and native source-contract checks. A small
Qwen3.5-0.8B-Base GPU canary completed cold and hot requests over independent
namespaces and exercised actual dependency deferral. This is functional
validation; deployment coverage and speed comparisons use separate matched runs.

## Build and enable

```bash
python scripts/build_radixflow.py --output /your/runtime/libjev_radixflow.so
pip install -e . --no-deps
vllm-jev-radixflow /your/model \
  --radixflow true \
  --radixflow-library /your/runtime/libjev_radixflow.so \
  --radixflow-trace-dir /your/runtime/radixflow-traces \
  --max-num-seqs 64 --max-num-batched-tokens 8192
```

The public switch accepts `--radixflow true`, `--radixflow false`,
`--radixflow` and `--no-radixflow`. It defaults to disabled. The old
`--radixflow-policy` option remains available for experiments; contradictory
switch/policy settings are rejected.

Build the Rust library once, then set its path for subsequent launches:

```bash
export VLLM_JEV_RADIXFLOW_LIBRARY=/your/runtime/libjev_radixflow.so
# Ordinary generative model, using its native vLLM implementation:
vllm-jev-radixflow /your/model --radixflow true --max-num-seqs 64
# Jev checkpoint:
vllm-jev serve ZefanCai/Open-Jev-2B --radixflow true
# Disable; retain ordinary serving behavior:
vllm-jev serve ZefanCai/Open-Jev-2B --radixflow false
```

The integration does not use a model-name allowlist. It supports native
prefix-cache-compatible models through the same scheduler interface. This is
not a claim that every architecture/configuration has been validated or gains
speed. All-attention models use admission; compatible hybrid state managers
also use dependency deferral. Models whose all-token readouts require prefix
caching to be disabled retain that constraint and reject explicit enablement.
The generic launcher forwards model, quantization and tensor-parallel arguments
to vLLM without exporting or replacing the model.

The generative launcher uses the original model, tokenizer, forward pass and
sampling. It does not export a Jev head. Existing Jev serving accepts the same
`--radixflow-*` flags. A model profile that disables prefix caching is rejected
instead of being silently changed. The admission bound is 5 ms; Flex releases
arrived work earlier for running work, row capacity or native residual demand.
No complete-group or fixed-four-branch assumption is used.

| Mode | Admission | Waiting scan |
| --- | --- | --- |
| `off` | Stock configured vLLM | Stock configured vLLM |
| `radix` | Native | Native stop |
| `timer5` | 5 ms / row capacity | Native stop |
| `timer10` | 10 ms / row capacity | Native stop |
| `flex` | Bounded, resource-aware | Native stop |
| `radixflow` | Flex | Defer confirmed dependencies |
| `radixflow_frag` | Flex | Also defer shortened aligned fragments |

All enabled modes share the same exact Rust observation index, complete-state
copy guard, native cache primitives and telemetry. The Rust index is namespace
local and bounded; observing or splitting a token node never creates GPU state
or counts as a native cache hit. The native allocator validates every selected
branch. Unknown producer identity preserves the native stop action. Requests
are registered before deferral and lifetime counters are removed on native
completion or cancellation. A dependency may be deferred at most 64 times;
the next visit follows native handling. Native request priorities and skipped
queue handling are retained.

Timer eligibility is snapshotted once per pass. This removes the old experimental
hook's mid-scan timer-release artifact; engineering timer results must be labeled
separately from the paper's frozen controls.

## Observability

With a trace directory, each EngineCore writes a separate JSONL file. Records:

* `capabilities`: policy, native source hash and adapter sites.
* `round`: selected branches, actual PENDING/STOP/DEFER events, producer IDs,
  cut preservation, post-pending rows, and qualified local recovered rows.
* `execute`: **actual global executor call**, scheduled tokens and request
  intervals, prefill/decode/mixed classification, empty cleanup calls, and
  host enqueue interval.
* `execute_complete`: the same call through native Future result retrieval.
* `core_step`: scheduling, executor completion, sampling, abort processing and
  output update. These are host intervals, not pure GPU kernel time.
* `request_end`: completion/cancellation status and lifetime deferral count.

The within-round recovered rows compare the retained partial batch and actual
selection on the same state. They are not a counterfactual whole replay. Requests
selected after a prior fragment deferral are not credited to dependency recovery.
Actual cache-hit tokens come from native HTTP usage, never the advisory index.
No prompt text or credentials are written to these event records.

## Speed demo preparation

Use JSONL rows `{"id":"q1","prompts":["shared prefix ... A","shared prefix ... B"]}`.
Run the same locked file on fresh, matched baseline and RadixFlow services:

```bash
python scripts/radixflow_demo.py --url http://127.0.0.1:8000 \
  --model your-served-name --label radix --inputs /your/inputs.jsonl \
  --output /your/demo/radix --traces /your/runtime/radixflow-traces
```

Restart using the other policy and repeat into a separate output directory.
Keep model, GPU, S/T budgets, input file, output length, concurrency, cache
isolation and instrumentation equal. Check GPU ownership before starting either
service. The demo reports wall time, branch latency, throughput and actual reuse.
With `--traces`, the summary also reports actual global worker steps and mean
host time per step. Empty cleanup calls are reported for the whole trace directory.
Use a fresh trace directory for each service. Raw failures
remain `FAILED.json`; an incomplete run does not produce a success summary.
The demo is closed-loop and does not claim a saturated serving throughput.

## CPU checks

```bash
python -m unittest discover -s tests -p test_radixflow.py -v
RADIXFLOW_TEST_LIBRARY=/your/runtime/libjev_radixflow.so \
  python -m unittest discover -s tests -p test_radixflow.py -v
python scripts/check_radixflow_adapter.py /path/to/vllm/v1/core/sched/scheduler.py
rustc --edition=2021 --test rust/radixflow/src/lib.rs -o /your/runtime/radix-tests
/your/runtime/radix-tests
```

To render the matched result as a comparison page:

```bash
python scripts/radixflow_compare.py /your/demo/radix/SUMMARY.json \
  /your/demo/radixflow/SUMMARY.json --html /your/demo/comparison.html
```

The comparator rejects mismatched model names, input hashes, output lengths and
concurrency. A missing worker measurement remains `unmeasured`.

## Read-only native baseline tracing

`VLLM_JEV_NATIVE_TRACE_DIR=/your/fresh/trace-directory` enables executor and
request-lifecycle observation on the native Scheduler. It adds no Rust index,
admission gate, state deferral, or copy guard. Leave the variable unset in
ordinary serving. This observer provides a native-scheduling comparison with
the same host enqueue/completion and complete-step timing boundaries.
