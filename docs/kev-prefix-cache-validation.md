# Kev prefix-cache integration validation

This internal integration imports [PR #13](https://github.com/mode-io/vllm-jev/pull/13) and its reproducible cache-check/A/B scripts. Native CUDA pooling can reuse a shared state prefix when a trusted caller supplies the same `cache_salt`. Without a supplied salt, requests remain isolated; questions within one request share a fresh namespace. Set `VLLM_JEV_KEV_PREFIX_CACHE=0` to disable reads and shared salts.

## Readout safety and compatibility

The adapter aligns partial token states using the returned suffix length. If a cache hit covers a required readout, it recomputes that row with reads disabled. An uncached call must return every token's state; unexpected truncation is rejected instead of silently accepted.

Cache capability is checked against the current Kev protocol, backend and pooling API. Ascend retains its validated full-state path. APIs without `skip_reading_prefix_cache` are called without that keyword and cannot advertise shared-cache support. Other decision protocols and Drex do not opt into this feature.

`metadata.cached_tokens` sums cache counts from the passes whose states were used; a discarded cached attempt followed by a full rerun is reported through `metadata.uncached_reruns`. `internal_usage.compute_tokens` retains its historical count of unique compiled prompt rows' tokens; it is not a measurement of executed GPU tokens after caching or retries.

## Local validation on 2026-11-24

The full local suite passed **1,576 tests with 24 skips**. It includes token-state engine doubles, endpoint routing, cache-salt isolation, readout-overlap fallback, Drex and other protocol regressions, and end-to-end benchmark-script tests against local fake HTTP servers. These tests do not establish GPU throughput or model accuracy.

Eleven additional regression cases fail against the original PR and pass with this integration. They cover legacy pooling API compatibility, the Ascend and protocol capability gates, unexpected uncached truncation, and invalid numerical outputs in the A/B checker. In particular, NaN drift must not pass simply because `NaN > tolerance` is false. Probability vectors must be finite, in range and normalized; a reported cache hit must have a positive, finite counter increment.

All eight GPUs on the available Tencent server were occupied when checked. No existing process was interrupted and no new GPU service was launched. Independent real-weight cache A/B validation remains outstanding for this integration. The previous Kev-4B/27B CUDA acceptance in [the native validation report](kev-native-validation.md) predates this cache change and is not cache-validation evidence.

## Contributor measurements

The contributor reports H20 measurements for Kev-27B in PR #13, including cache reuse, block-alignment effects, latency and probability comparisons. Those figures are attributed to the contributor and were not independently reproduced here. The reported uncached baseline used an upstream revision rather than this same revision with the feature disabled, so its speedup ratios do not isolate only the cache toggle.

Before promoting these measurements to an independently validated result, run the provided scripts on the same pinned weights and code revision with reads enabled versus disabled, using isolated services and the same requests, scheduling settings and budgets. Check actual cache hits, fallback counts, answer differences and concurrency behavior. A short warm-cache result is not a sustained throughput limit or a decision-quality benchmark.
