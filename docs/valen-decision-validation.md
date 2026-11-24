# Valen-Decision-1.5 internal integration

This integration brings [PR #12](https://github.com/mode-io/vllm-jev/pull/12) into the internal repository while retaining the Kev cache changes from PR #13. The public PR remains separate. The adapter targets Linux CUDA and supports text/image Choice, Noul, Score and image `bbox` questions; the Valen-Preview adapter remains unchanged. Windows, Mac, Ascend, video, prefix caching, chunked prefill and RadixFlow are outside this adapter's scope.

## Repairs

- **Bind inference semantics to the release.** Export retains `valen_release_config.json`, checked against the pinned original config hash. The manifest's attention mask, maximum length, head dimensions and media settings, plus the rewritten native model config, must match that original configuration. Editing those settings, even while recalculating a local config checksum, must not pass verification. Older PR exports without this file must be re-exported; verification does not silently relax the check.
- **Keep the mandatory architecture.** Unsupported backends fail before downloading weights. A user-supplied `--hf-overrides` is rejected so it cannot replace the block-attention architecture with an ordinary causal model.
- **Validate complete token states.** Pooling results must match both the complete prompt length and the trained hidden width. Malformed results abort the engine request rather than reaching the heads.
- **Bound pending readouts.** The request limit covers compilation, backbone execution and the head executor together. Completed token-state tensors cannot accumulate in an unbounded head queue simply because the backbone semaphore has been released. A 20-request synthetic test with a limit of two checks this behavior.
- **Do not accept malformed comparison output.** The reference driver rejects nonfinite/out-of-range probabilities, malformed or empty boxes, and invalid tolerance values. It compares `point` coordinates as well as box edges and confidence. Without an explicit tolerance, valid numerical differences and selection disagreements are still recorded as data, as the contributor designed; this is not an equality gate.

## Local verification

The full local CPU/synthetic suite passed **1,626 tests with 34 skips**. Ten skips in the new adapter require `VALEN_DECISION_RELEASE` and the release's local reference code; those reference comparisons were not run here. The remaining skips are existing optional environment/platform tests. The suite covers ordinary decision heads, block-mask math, export integrity, CLI rejection, preview routing, service routing, and the additional regressions above.

New regression cases are also run against the original PR to demonstrate that the guards detect the original problems. Tests use tiny generated tensors and local fixtures; no model weights were downloaded and no checkpoint Python files were executed during this integration.

## Hardware evidence and limits

The contributor reports real-weight H200 runs for 0.8B, 2B and 4B in PR #12. The guide preserves those measurements with attribution, including probability differences, selected-answer differences and box IoU. They are not independent results from this integration. The contributor did not run 9B/27B GPU acceptance, concurrency, near-limit inputs or throughput tests.

At the server check for this integration, all eight available Tencent GPUs were occupied. No existing job was interrupted and no GPU service was started. Independent real-weight acceptance and pressure testing remain outstanding, including the effect of these integration repairs. Neither this local suite nor the contributor's drawn-image comparisons establish production reliability or decision accuracy.
