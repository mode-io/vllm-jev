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
| Open-Jev 2B/9B; OpenJev 0.6B | Shared device selection; CPU-safe HF/PEFT preparation; native pooling | Pending |
| Tiny-Jev | FP32 tensor score projection on non-CUDA devices; CUDA Triton kernel lazily loaded | Pending |
| Laya / multilingual / typed decisions | Portable head placement; same protocol/compiler; eager head instead of CUDA Graph | Pending |
| Valen | CPU-safe weight preparation; native token pooling and existing head | Pending |
| vjev / vjev pilot | Native token pooling; existing listwise head | Pending |
| RSI-Jev text / VL | Device-aware FP32 cross-attention head | Pending |
| Clef / Clef-Flash | Device-aware joint head; existing text/image/video schema | Pending |
| JevAny dense / MoE / VL-MoE | Existing shard-wise preparation and FP32 pointer heads; native backbone | Pending |
| Drex 35B MoE | Existing verified merged export and causal pointer protocol | Pending; single 64 GB cannot hold its ~65.4 GiB BF16 weights |
| Kev / Decider / Task / ThisThat / Mica / JevK5 | Existing decision compilers and CPU heads over native pooling | Pending |
| Choice / Noul / Score, templates | Shared endpoint and compilers retained | CPU regressions only |
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

## Development validation record (2026-10-20)

The local selection passes 220 tests, with 16 reference/runtime-dependent tests
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
native vLLM serving. A separate matched runtime is still being built from source
because the host glibc is 2.31 and the available Ascend wheel requires 2.34.

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
