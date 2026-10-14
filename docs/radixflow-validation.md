# RadixFlow integration validation

The integration builds on public vLLM Jev commit `57226d4`.
The scheduling integration is opt-in and does not change ordinary serving
settings.

## Checked

* The public synchronization passes 14 local Python tests; two additional tests
  require the Linux tensor runtime or compiled Rust library and are skipped
  locally. The boolean switch, legacy policies and forwarded vLLM flags are
  covered. Ruff and whitespace checks pass.

* 13 Python regression tests passed on the Linux runtime, including actual
  native-state-guard binding, queue continuation, next-round consumer progress,
  metadata cleanup, unknown-producer fallback and the deferral bound.
* 2 Rust tests passed. The exact-prefix test compares 2000 observations against
  a longest-prefix oracle. Python FFI checks cover namespace isolation and
  bounded metadata retention.
* The actual vLLM 0.29 scheduler source passed the structural adapter check:
  one waiting-allocation site, one admission visit, and two zero-token sites.
* CPU tensor tests confirm that a short alias cannot hide a larger state extent;
  non-dense geometry is rejected.
* The real-model canary used Qwen3.5-0.8B-Base, bf16, S=16, T=4096, context=4096,
  eager synchronous execution and two independent cache namespaces. Sixteen
  one-token HTTP requests completed. Each hot request reported 1984 native
  cache-hit tokens. No model weights were modified.
* The final canary trace contains 5 actual nonempty global executor calls and
  4 actual empty cleanup calls. Thirteen native PENDING events were observed.
  One independently selected row passes the retained-cut / producer-in-cut
  check for local recovery. This is direct functional evidence, not a measured
  replay speedup.
* All nine executor records join to their scheduler rounds, native Future
  completion intervals and complete core-step intervals. Token totals and
  row budgets reconcile, and all 16 requests have distinct terminal events.
* A local HTTP fixture checks the demo's logical-latency grouping, real usage
  parsing and HTML comparator. Missing worker measurements stay unmeasured.

The service was shut down after the canary and its GPU was released. Earlier
setup failures and raw traces remain outside the repository. The first setup
attempt revealed a missing module entry call; later attempts identified a
text-only model startup setting and an incompatible host FlashInfer build.
The native telemetry test also exposed plugin loading during an existing
EngineCore constructor; metrics now attach at the first real step. A regression
covers this late-loading case and native Future result preservation.

## Interpretation

The canary validates one hybrid generative model and its exercised cache layout.
It is not a six-policy timing study, task-quality evaluation, or comprehensive
production deployment matrix. The speed demo must use matched model, GPU,
budgets, input hashes, output lengths and concurrency. Asynchronous scheduling,
pipeline parallelism, remote connectors and streaming input remain explicitly
unsupported by this synchronous adapter.
