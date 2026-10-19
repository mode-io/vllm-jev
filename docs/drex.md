# Drex v2.0 adapter prototype

Experimental Linux/native-vLLM integration for `nace-ai/drex-v2.0`. This is an
adapter with bounded released-weight functional validation, not a speed claim. The current
scope is text/JSON Choice, Noul and Score with explicit `state_bidir=false`.

```bash
vllm-jev serve nace-ai/drex-v2.0 --workspace /your/large-data-disk \
  --tensor-parallel-size 2 --max-model-len 8193
```

The pinned release is `3e6d164cb5c8cf264eec167905cc2448d89ea97d`. Automatic preparation
verifies the pinned file sizes/digests, ignores downloaded Python, and loads
`head.pt` with `weights_only=True`. Original safetensors shards are hardlinked
unchanged into an export with a separate config and FP32 pointer head. A
cross-filesystem copy needs enough free space. The downloader retains 32 GiB
headroom and fails before downloading if space is insufficient.

For an already downloaded original merged checkpoint:

```bash
python -m vllm_jev.drex_export --source /weights/drex-v2.0 \
  --output /exports/drex-v2.0
vllm-jev serve /exports/drex-v2.0 --tensor-parallel-size 2 --max-model-len 8193
```

Local exports are operator-supplied compatible checkpoints. The release digest
check applies to automatic model-ID downloads; a local export does not establish
that its weights are the official release. Keep model and export on the same
filesystem to avoid duplicating the roughly 70 GB backbone. Exports preserve the
original weight names, including the native MoE layout. There is no LoRA merge.

Default input length is 16,384 tokens, with one additional native scheduler slot.
Each complete question row must fit; input is rejected rather than truncated.
The structural upper bound is 131,072, but long-context resource behavior has
not been validated. The prototype returns full token states for its pointer
readout, so native prefix caching and `cache_salt` are disabled; RadixFlow is
not compatible with this initial path. Cached readout selection needs separate
implementation/validation. MoE active parameter count does not describe total
weight memory.

The unchanged native template is required. Media, option-isolated scoring,
bidirectional-state checkpoints and live LoRA adapters are rejected. Pointer
projection and calibration remain FP32. Score confidence, legends and four-place
serialization follow the author runtime; `usage.output_tokens=0` follows Jev's
no-generation usage convention, rather than the author's serialized-response
billing count. The service allows up to four simultaneous native encode calls.

## Integration and request format

This adapter uses the shared model preparation lock, decision manifest verifier,
model registry, native vLLM CLI forwarding and `/v1/systemone` endpoint. Its
compiler defines Drex control tokens and causal question rows; the pointer head
runs in FP32 after the shared native encoding path. Hostnames, GPU IDs, ports and
workspace paths are supplied by the operator, not embedded in the adapter.

```bash
curl http://127.0.0.1:8000/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"state":"A customer requests a refund.","questions":{"team":{"type":"choice","criteria":{"billing":"Payments and refunds","technical":"Software issues"}}}}'
```

The example uses the native Linux vLLM default port; use `--port` to change it.
For large downloads, the standard Hugging Face `HF_HUB_DOWNLOAD_TIMEOUT` setting
can be increased. An already downloaded checkpoint can use the local export
command above, avoiding a second network download. Keep HF, Torch and temporary
caches on the data disk as well as the `--workspace` directory.

## Validation boundary

`tests/test_drex.py` uses synthetic shards and heads and checks source immutability,
unsafe/configuration rejection, context limits, disk guards and pointer math.
Separate local diagnostics compare the compiler/readout with the pinned upstream
reference and exercise a tiny random hybrid MoE. These checks do not establish
35B quality or 131K serving. The released-weight checks below cover one native
TP=2 configuration; they do not establish production support.

### Current checks (2026-10-19)

- 89 Python checks pass with one environment-dependent Rust test skipped across
  the Drex, template, RadixFlow, JevAny and precision test selections.
- Six synthetic request cases have exact token/readout/answer parity against the
  pinned author runtime using FP32 CPU inference. They include Chinese JSON,
  optional instructions, control-looking text, repeated levels and one-option
  questions.
- Two fresh native vLLM 0.29.0 A800 runs load the random 4-layer hybrid MoE and
  complete six golden requests plus 24 requests each at concurrency 1/4/8.
  Three invalid requests are rejected in each run. Maximum observed option
  probability difference against the CPU reference is 0.0035.
- Wheel/sdist packaging checks cover all three Drex modules. Model/source files
  and raw test logs remain outside the repository.
- An additional native A800 run completes 320 requests at concurrency 1/4/8/16/32,
  five 1,416-token inputs with a 1,024-token scheduling budget, and recovery after
  eight client disconnects. Three invalid requests are rejected. This uses the
  tiny random checkpoint, not the released 35B weights.
- Adapter-only stress with a synthetic async engine completes 4,096 requests at
  concurrency up to 256 and 25 cancellation/failure recovery cycles. These are
  lifecycle checks, not GPU throughput measurements.
- The official hash-pinned 35B BF16 release passes a native vLLM 0.29.0 test on
  two A800 80 GB GPUs (TP=2). Full exported-checkpoint verification checks 28 files;
  native logs show all 16 weight shards loading and 31.95 GiB model memory per rank.
  Configuration: model length 8,193, eight sequences, 1,024 scheduled tokens,
  memory utilization 0.65, eager execution, synchronous scheduling.
- The trained-weight run completes six functional examples, 320 requests across
  concurrency 1/4/8/16/32, five 1,416-token inputs, and one recovery request after
  eight client disconnects: 332 successful HTTP responses in total. Three invalid
  requests are rejected. Returned probability vectors are finite and normalized
  within serialization tolerance. Mixed-test client p50/p95 are 0.567/2.373 s;
  these include different request types and concurrency levels, not a benchmark
  against the author implementation.
- Both test GPUs were verified free after shutdown. The native runtime reports
  semaphore/shared-memory cleanup warnings at exit; this test does not establish
  absence of lifecycle leaks. Author-runtime output parity, quality, other TP
  configurations and 131K context remain unvalidated. The response metadata
  `released_weight_validation=false` remains a conservative static prototype flag,
  not an attestation of the model files loaded in an individual deployment.
  Prefix caching and RadixFlow remain disabled. All remote model files, caches,
  logs and temporary paths use data disks rather than the home disk.

### Independent higher-concurrency retest

A second fresh-service run of the same official release completes 512 stress
requests plus six functional examples, five long inputs and one post-disconnect
probe: 524 successful HTTP responses. It also rejects three invalid requests and
recovers after eight disconnects. Both GPUs are free after exit. The Drex module
hashes used remotely match the checked-in source.

| Concurrent clients | Successful requests | Completed req/s | Client p50 (s) | Client p95 (s) |
|---|---:|---:|---:|---:|
| 1 | 128 | 7.19 | 0.117 | 0.231 |
| 8 | 128 | 12.00 | 0.642 | 1.127 |
| 32 | 128 | 13.61 | 2.281 | 2.432 |
| 64 | 128 | 13.73 | 4.539 | 4.776 |

Each row replays 128 short requests cycling through the same six synthetic
functional inputs on trained weights. Client concurrency includes queued work;
the adapter admits at most four native encode calls simultaneously. These rates
apply to this input mix and hardware, not a saturation benchmark or comparison
with the author runtime. The long-input cases contain 1,416 tokens and exceed the
1,024-token scheduling budget; they do not validate the full context limit.
