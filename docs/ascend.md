# Ascend port (0.4 development branch)

This branch prepares the shared vllm-jev model adapters for Ascend. **It is not an
all-model NPU acceptance release.** CPU tests validate the portable code paths;
trained-model NPU serving, media, long contexts and performance are still pending.
Do not infer NPU compatibility from the existing CUDA results.

## Runtime and installation

The original CUDA environment remains `vllm==0.29.0`; install with
`uv pip install '.[cuda]'`. The core package no longer forces a CUDA vLLM wheel
into an NPU runtime. Python 3.11 is accepted for platform-managed CANNLab images.

Select a complete vendor-supported runtime row first:

| Native vLLM | vLLM Ascend | CANN | PyTorch / torch-npu | Entry point |
|---|---|---|---|---|
| 0.22.1 | 0.22.1rc1 | 9.0.0 | 2.10.0 / 2.10.0 | Process-local endpoint bridge |
| 0.30.0 | 0.30.0rc1 | 9.2.0 | 2.10.0 / 2.10.0.post4 | Native endpoint plugins |

These are **integration targets**, not tested model configurations. Follow the
[upstream compatibility matrix](https://github.com/vllm-project/vllm-ascend/blob/main/docs/source/community/versioning_policy.md)
for the complete NNAL, Triton, firmware and driver requirements. A matching
vLLM/Ascend version pair alone is insufficient. Do not install an arbitrary
PyPI CUDA vLLM wheel or upgrade a platform-managed driver.

The observed CANNLab host is ARM64 Ubuntu 20.04, 910B3 with 64 GB HBM,
driver 25.5.0, CANN 9.0.0 and torch/torch-npu 2.7.1/2.7.1.post4. It initially
has no vLLM, vLLM Ascend or Transformers. Its existing Torch pair does **not**
match either target above. An isolated matched runtime is needed for serving.
The physical NPU reports healthy with no user process; that is not a model test.

Once a matching runtime and the ordinary Jev dependencies are present:

```bash
source /home/developer/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:${LD_LIBRARY_PATH:-}
# Load NNAL ATB too, using the ABI of the matched Torch runtime.
source /home/developer/Ascend/nnal/atb/set_env.sh --cxx_abi=1
bash scripts/install_ascend.sh /mnt/workspace/jev /path/to/runtime/bin/python
vllm-jev doctor --smoke
vllm-jev serve IamBusy/OpenJev-0.6B --backend ascend \
  --workspace /mnt/workspace/jev --max-model-len 2048
```

`install_ascend.sh` only installs this plugin with `--no-deps` into a supplied
runtime; it does not provision or certify the external engine. The launcher
retains the `ascend` platform plugin as well as the Jev plugins. It starts in
eager, synchronous mode until graph/asynchronous combinations are validated.
For native vLLM 0.22.1, the dedicated server process attaches the same Jev routes
and state initializer to the native app; it preserves native authentication,
cancellation and engine lifecycle. The bridge does not patch an installed file.

## Port coverage and remaining acceptance work

| Family / feature | Engineering changes | NPU acceptance |
|---|---|---|
| OpenJev 0.6B | CPU-safe HF/PEFT preparation; native scalar pooling | 910B3: 128/8, 1024/32 and 4096/64 client-concurrency requests passed; see record below |
| Open-Jev 2B | Shared device selection; CPU-safe HF/PEFT preparation; native scalar pooling | S16/S64 NPU sweeps passed; best43.69 req/s atS64,64clients |
| Open-Jev 9B | Shared device selection; CPU-safe HF/PEFT preparation; native pooling | Pending |
| Tiny-Jev | FP32 tensor score projection on non-CUDA devices; CUDA Triton kernel lazily loaded | Pending |
| Laya English | Portable head placement; eager head instead of CUDA Graph | Six warmed NPU sweep arms passed (256 requests each); best10.40 req/s at16 clients, nativeS16 |
| Laya multilingual / typed decisions | Shared portable head and protocol/compiler | Both six-arm NPU sweeps passed; measured rates below |
| Valen | CPU-safe weight preparation; native token pooling and existing head | Pending |
| vjev / vjev pilot | Native token pooling; existing listwise head | Pending |
| RSI-Jev text / VL | Device-aware FP32 cross-attention head | Pending |
| Clef / Clef-Flash | Device-aware joint head; existing text/image/video schema | Pending |
| JevAny dense / MoE / VL-MoE | Existing shard-wise preparation and FP32 pointer heads; native backbone | Pending |
| Drex 35B MoE | Existing verified merged export and causal pointer protocol | Pending; single 64 GB cannot hold its ~65.4 GiB BF16 weights |
| Decider 0.8B | Pure-text Qwen3.5 pooling registration and strict weight-name alias loading | Six warmed NPU arms passed; best15.94 req/s@32clients |
| Kev 0.8B | Verified merged adapter and CPU pointer head over native NPU pooling | Six warmed NPU arms passed; best25.82 req/s@32clients |
| Decider 2B | Shared pure-text pooling and alias loading | Six warmed NPU arms passed; best15.64 req/s@32clients |
| Other Task, ThisThat, Mica, JevK5 | Existing decision compilers and portable heads | Pending separate NPU tests |
| Choice / Noul / Score, templates | Shared endpoint and compilers retained | Three typed operations passed on OpenJev 0.6B; other families/templates pending |
| Media / video | Shared input limits and processors retained | Native NPU vision acceptance pending |
| Native prefix cache | Still owned by the engine; protocol-specific cache restrictions retained | NPU cache/token conservation pending |
| Rust index / flex / state-ready / fragmentation | Host algorithms retained | NPU scheduler/state-copy adapter pending; launch explicitly rejected |
| Online hybrid checkpoints | Original CUDA adapter retained | NPU publication/copy/eviction checks pending; launch rejected |
| CUDA Graph / cuBLAS split-K controls | CUDA implementation retained | CUDA-only controls rejected on Ascend |

The device-independent layer is shared engineering, not a separate implementation
of each model's prompt semantics. Model exporters preserve exact head/tokenizer
rules. A CUDA-specific optimization has an eager tensor alternative where that
alternative preserves semantics. Cache state-copy and scheduling hooks require
backend-specific validation and cannot safely be enabled by replacing a device
string.

## Diagnostics and data location

`vllm-jev doctor` reports OS/architecture, dependency versions, device nodes,
cache paths and `npu-smi` status. `--smoke` additionally runs a small NPU matmul;
it is not a serving or numerical-parity acceptance test.

```bash
npu-smi info
watch -n 1 npu-smi info
```

Use `/mnt/workspace` for model weights, virtual environments, source, caches,
logs and temporary files. SSH may require the platform IDE's local forwarding
helper to stay running. A working cloud environment does not guarantee that its
localhost SSH tunnel is currently available.

## Development validation record (2026-10-20–21)

The local selection passes 229 tests, with 16 reference/runtime-dependent tests
skipped. It includes device selection, CPU/NPU separation, Tiny-Jev projection
against FP32 layer normalization, the legacy endpoint bridge lifecycle, CLI
forwarding, and existing protocol/template/RadixFlow/precision regressions.
Static checks, shell syntax and wheel/sdist construction pass. None of these
results certifies native NPU model serving. The authenticated SSH forward has been recovered. A real 910B3 run using the
platform's original torch/torch-npu 2.7.1 pair passes 192 portable projection
checks at 1/4/8 host threads against CPU FP32 layer normalization, with maximum
absolute error 1.91e-5. A second isolated torch/torch-npu 2.10.0 run passes the
same 192 checks, also with maximum absolute error 1.91e-5. These exercise the
numerical path, not a trained model or
native vLLM serving. The matched runtime was built from source because the host glibc is 2.31
and the available Ascend wheel requires 2.34.

After each model is served on the matched NPU runtime, record an API probe:

```bash
python scripts/ascend_acceptance.py --url http://127.0.0.1:8795 \
  --requests 128 --concurrency 8 \
  --output /mnt/workspace/jev/results/model-family.json
```

This script never downloads or starts a model. Preserve the launch command,
weight digests, runtime/driver versions, NPU occupancy and reference-output
checks alongside its results. A probe cannot identify the backend by itself.

### Robustness fixes

- The doctor and installer check the complete declared runtime row, including
  TorchNPU, Transformers, Triton and selected CANN version. Package presence or
  a matching vLLM version alone cannot mark the environment ready.
- Local version suffixes such as `0.22.1+empty` are accepted for matching source
  builds; incompatible base versions are still rejected.
- Monitoring timeouts and smoke failures produce structured diagnostics. A
  non-ready doctor returns a nonzero status.
- API pressure validates required answer fields and types, finite normalized
  probabilities, selected options and score ranges. Validation remains active
  under Python `-O`; failures are counted and saved instead of losing the report.
- The legacy endpoint bridge enforces a single API-server process before model
  preparation. Native multi-process endpoint plugin support is a separate path.

Source builds must include upstream Git submodules. The PyPI source archive did
not include the Catlass dependency required by the NPU kernel build. The current
trial uses upstream tag `v0.22.1rc1`, commit
`5f6faa0cb8830f667266f3b8121cd1383606f2a1`, with its declared Catlass commit
`41bf90da655bba3c66d0acd7e00abe33960ecfd6`. Source build artifacts, temporary homes
and Git configuration are confined to the data-disk workspace. Compilation is
not itself model acceptance.

The CANNLab source build loaded the trained OpenJev-0.6B weights on 910B3.
The first request exposed a pure-attention cache-mode incompatibility: Ascend
0.22 enters recurrent-state copying whenever `mamba_cache_mode=align`. The
launcher now omits recurrent cache defaults for known full-attention backbones;
hybrid backbones retain them. The repaired launcher passed the functional acceptance below.
The private runtime also records unresolved OpenCV/NumPy and optional CANN
profiler dependency declarations; a successful import is not a clean dependency audit.

### Trained OpenJev-0.6B functional acceptance

Native vLLM 0.22.1 + vllm-ascend 0.22.1rc1 loaded the BF16 merged
OpenJev-0.6B checkpoint on the allocated 910B3. RadixFlow was off. Tests used
loopback HTTP, eager synchronous execution, `max_model_len=1024`, 16 sequences
and 50% device-memory utilization. The fixed adapter revision was
`b4bf78e7c053407b5d54a71bcc682ef5bbe59e72`; its Qwen3-0.6B foundation was
`c1899de289a04d12100db370d81485cdf75e47ca`.

| Requests | Client concurrency | Successful | Failed | Wall time (s) |
|---:|---:|---:|---:|---:|
| 128 | 8 | 128 | 0 | 8.75 |
| 1024 | 32 | 1024 | 0 | 42.77 |
| 4096 | 64 | 4096 | 0 | 168.78 |

Each request checks Choice, Noul and Score answer IDs, types and finite,
normalized probabilities. A separate edge probe passed 16 long-input requests,
four invalid-input rejections, and 16 subsequent requests after eight clients
sent a request and closed the connection without reading the response. This
checks recovery, not proof that every disconnected request was aborted before
execution. These are functional results for one family, not a hardware speedup
comparison or acceptance of every listed model.

A small BF16 numerical audit compared the original HF+PEFT CPU model against
native NPU scoring on four prompts (ten candidate branches). Top-1 choices
agreed on all four. The mean absolute score difference was 0.183 and the maximum
was 0.496; this is not bitwise equivalence or a downstream accuracy evaluation.

An additional 256-request mixed probe passed at 16-way client concurrency. It
varied the state text and its length, used 2/3/4/8 choice candidates, and included
Noul and Score in every request (14.27 s wall time, 0.975 s client p95).
All measurements here are client/API timings, not isolated GPU-kernel time.

### Sequence-budget sensitivity on OpenJev-0.6B

A separate warmed sweep fixed `max_num_batched_tokens=8192` and varied the
client concurrency at native sequence budgets 16 and 64. At budget 16 the
best 256-request arm reached 23.75 req/s. At budget 64, the best short arm
reached 76.68 req/s. A fresh-service 4,096-request check at budget 64 and
32-way client concurrency completed without failure in 51.69 s: **79.24 API
requests/s**, client p50 0.402 s and p95 0.477 s. Every request contains three
typed questions (237.73 questions/s). These observed rates depend on the stated
workload and budgets; they are not a hardware-theoretical limit or a CUDA comparison.

Laya English also completed the six-arm warmed concurrency sweep with no failed
requests. At native sequence budget16 its highest observed throughput was
10.40 API requests/s at16 clients (256 requests, p50 1.521 s, p95 1.712 s).
The API payload was the same as OpenJev; architecture-specific tokenization and
head execution still differ. This is a short sweep result, not a sustained check.

### Measured text API throughput by model

| Model | Native sequence budget | Client concurrency | Observed req/s | Client p95 (s) | Measurement |
|---|---:|---:|---:|---:|---|
| OpenJev-0.6B | 64 | 32 | 79.24 | 0.477 | 4096-request check |
| Laya English | 16 | 16 | 10.40 | 1.712 | Best of six 256-request arms |
| Laya multilingual | 16 | 16 | 14.59 | 1.279 | Best of six 256-request arms |
| Laya typed decisions | 16 | 16 | 10.26 | 1.849 | Best of six 256-request arms |
| Decider-0.8B | 16 | 32 | 15.94 | 2.006 | Best of six 256-request arms |
| Kev-0.8B | 16 | 32 | 25.82 | 1.509 | Best of six 256-request arms |
| Kev-4B | 16 | 64 | 20.49 | 3.314 | Best of six 256-request arms |
| Decider-2B | 16 | 32 | 15.64 | 2.056 | Best of six 256-request arms |
| Open-Jev-2B | 64 | 64 | 43.69 | 1.457 | Best of six 256-request arms |
| Open-Jev-9B | 16 | 32 | 10.48 | 3.160 | Best of six 256-request arms |

All listed arms had zero failures and used the same three-question text API
payload. The three Laya sweeps each completed all six concurrency settings
(1/4/8/16/32/64). Budgets and measurement lengths differ as shown; these rows
report tested configurations, not a controlled ranking of architectures.
Remaining families keep their pending status until their own NPU tests pass.

Decider-0.8B completed all six concurrency arms with zero failed requests after
registering the standalone Qwen3.5 text pooling wrapper and mapping checkpoint
weight aliases at load time. The mapper retains tensor values and the native
loader still requires every parameter to be initialized. Earlier failed starts
remain in the development audit; they are not included in throughput results.

The common payload contains three logical questions and seven alternative
candidate scores. Models may pack those alternatives differently; seven
alternatives do not imply seven physical engine requests for every protocol.
Throughput is reported at the typed API boundary, rather than inferred engine
branches/s. Kev-0.8B completed all six arms with zero failed requests after
source and foundation verification and a fresh, isolated service launch.

Open-Jev-2B completed six warmed arms at each sequence budget16 and64, with
zero failures. The highest observed short-arm throughput increased from
13.40 req/s at budget16 to43.69 req/s at budget64. Its pinned foundation is
retained separately for exact-version reuse by subsequent adapters.
