# Upstream PR map

The private repository is organized so its verified implementation can be lifted into vLLM with a small, reviewable patch. It uses vLLM's model runner, pooling adapter, scheduler and hybrid prefix cache. There is no independent model execution loop.

| This repository | Equivalent in-tree location | Change |
|---|---|---|
| `vllm_jev.model` | `vllm/model_executor/models/qwen3_5.py`, `registry.py` | Register a text-only Qwen3.5 sequence classifier, using existing `as_seq_cls_model` and hybrid Mamba/M-RoPE hooks. |
| `vllm_jev.endpoint` | `vllm/entrypoints/serve/jev/api_router.py` plus router attachment | Add opt-in typed Choice and bounded Batch Choice routes using `EngineClient.encode`. |
| `vllm_jev.prompt` | Jev serving protocol module | Render candidate-independent prompts and normalize scalar scores per decision. |
| `vllm_jev.export` | Model conversion tooling | Merge PEFT LoRA, retain the fp32 scalar head, remap HF Qwen3.5 text keys and write a provenance manifest. |
| Local-only validation scripts | vLLM model/endpoint tests and benchmark recipes | Validate prompts, cache-salt isolation, candidate limits, parity and performance without publishing workload data. |

The minimum new engine behavior is the scalar pooling model registration. Candidate fan-out uses existing pooling requests; prefix reuse uses the existing vLLM hybrid cache. An upstream PR should preserve the route's opt-in nature and request-local cache salt, and should include a GPU parity fixture because Qwen3.5 hybrid cache currently produces small numerical differences. The optional `use_prefix_cache=false` request flag provides an uncached execution path for callers who need to audit that difference.

For native `--data-parallel-size > 1`, an additional small upstream change would be valuable: let `EngineClient.encode` accept a stable DP-rank/affinity key, as generation requests can already pass `data_parallel_rank`. The Jev endpoint would choose one rank per decision and send all its candidate pooling calls there. In the current vLLM 0.29 API, candidate calls are routed independently; a two-GPU validation observed less prefix reuse and lower throughput than the separate-replica decision router. Detailed test records remain local. This repository keeps the workaround outside vLLM rather than monkey-patching private engine internals.

This is not yet an upstream pull request. The repo is private by user request. Full 0.29 eager/compiled/async MMLU audits now support the compiled, async-scheduled cached path as the default, with explicit uncached, synchronous and eager modes for numerical review.
