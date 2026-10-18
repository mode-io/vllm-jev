# Drex v2.0 adapter prototype

Experimental Linux/native-vLLM integration for `nace-ai/drex-v2.0`. This is an
adapter prototype, not released-weight acceptance or a speed claim. The current
scope is text/JSON Choice, Noul and Score with explicit `state_bidir=false`.

```bash
vllm-jev serve nace-ai/drex-v2.0 --workspace /your/large-data-disk \
  --tensor-parallel-size 2 --max-model-len 16385
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
vllm-jev serve /exports/drex-v2.0 --tensor-parallel-size 2
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

## Validation boundary

`tests/test_drex.py` uses synthetic shards and heads and checks source immutability,
unsafe/configuration rejection, context limits, disk guards and pointer math.
Separate local diagnostics compare the compiler/readout with the pinned upstream
reference and exercise a tiny random hybrid MoE. These checks do not establish
35B quality, latency, memory capacity or 131K serving. Full released-weight
acceptance is still required before advertising production support.

### Current checks (2026-10-18)

- 81 Python checks pass with one environment-dependent Rust test skipped across
  the Drex, template, RadixFlow and JevAny test selections.
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
- This exercises TP=1 and short inputs only. A follow-up chunked-prefill/long-input
  boundary run did not start because the server SSH connection became unavailable.
  TP>1, released 35B weights, prefix-cache optimization and 131K context remain
  unvalidated. The first two test services exited and GPU4 was verified free.
