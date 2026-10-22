# Kev native CUDA validation

This page records the acceptance scope for [PR #11](https://github.com/mode-io/vllm-jev/pull/11). Kev-4B has passed the checks below. **Kev-27B v2 is still downloading and has not yet passed real-weight GPU acceptance.** Its new support must not be described as validated until that run completes.

## Checkpoints and implementation

- Kev-4B: `jaredpalmer/kev-4b` at `6cfce5c2fa4b4bd64026336ab649c5ca78857d52`, merged with its exact declared `Qwen/Qwen3.5-4B-Base` revision `1001bb4d826a52d1f399e183466143f4da7b741b`. Preparation merges ordinary LoRA pairs by shard and removes the unused vision/MTP modules.
- Kev-27B v2: `jaredpalmer/kev-27b` at `af0e6d551bdc2cc724f3e9d7a8bee1cd4fb8f7bf`. This is the full-weight release, not the `v1-lora` adapter. Its head declares calibration temperature approximately 1.319508. The exporter adds the native `model.` key prefix and ties the unused LM head to the embeddings, without changing backbone tensor values or dtypes. No separate foundation-weight download is needed. Allow approximately 110 GB for the source and rewritten export.

Downloads use fixed revisions and must match the official Hugging Face Git/LFS hashes, including when served through ModelScope. Checkpoint Python files are neither required nor executed. These are text/JSON pointer-head adapters; the new export path does not add Windows, Ascend-27B, or media validation.

## Measured workload

The Kev-4B run used one NVIDIA A800-SXM4-80GB, vLLM 0.29.0, Torch 2.13.0 and Transformers 5.10.4. Native BF16 pooling and an FP32 decision head were served with eager execution, synchronous scheduling, prefix caching enabled, 32 sequence slots, an 8,192-token batch budget and a 4,096-token complete-prompt limit. No quantization was used.

Each API request contained three logical questions: a two-option Choice, a Noul and a three-level listwise Score. Request IDs varied in the state. Six client-concurrency arms each measured 256 completed requests after 16 separate warmup requests. API rates include host processing and queueing; they are not GPU-kernel time, token throughput, or original-author speed comparisons.

| Client concurrency | Completed | Failures | API requests/s | P50 (s) | P95 (s) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 256 | 0 | 5.88 | 0.170 | 0.175 |
| 4 | 256 | 0 | 12.18 | 0.331 | 0.336 |
| 8 | 256 | 0 | 24.49 | 0.328 | 0.334 |
| 16 | 256 | 0 | 47.15 | 0.338 | 0.358 |
| 32 | 256 | 0 | 78.31 | 0.404 | 0.438 |
| 64 | 256 | 0 | 123.82 | 0.500 | 0.555 |

These are single short sweeps, including a roughly two-second highest-concurrency arm. They do not establish sustained saturation, uncertainty intervals, or model quality. Logical questions/s are three times API requests/s for this workload; this does not imply three or seven physical engine branches per API request.

All 1,536 measured requests completed with finite, normalized outputs and zero decode tokens. Three longer-input requests, three invalid-input rejections, eight connection closures followed by a successful recovery request also passed. The service loaded its real exported weights, no foreign GPU owner was observed during acceptance, and GPU memory returned to zero after cleanup. Raw logs and results were archived with SHA-256 verification.

## Bugs and regression scope

The PR's composite LoRA merger could silently use the wrong `alpha / r` scale if the declared rank disagreed with the actual A/B tensors. It also accepted nonfinite or boolean alpha values. Preparation now checks the actual rank, tensor dimensions and finite floating values, and rejects invalid scale metadata.

Five new regression cases fail against the original PR and pass with the fix. The complete local CPU/synthetic suite passes 1,525 tests with 24 skips; that suite is separate from real-weight GPU acceptance and does not rerun every supported model on every platform.

The first GPU harness attempt used an invalid dictionary for Score criteria and correctly received HTTP 422. Its partial log is retained as a setup failure and excluded from the table. The corrected full attempt uses list-valued Score criteria and a fresh service. Shutdown messages caused by intentional service termination are retained in the raw log.

No original-author output-equivalence or decision-accuracy claim is made by these stress results.
