# vLLM 0.29 feature map for Jev

Jev is a `classify` **pooling** model. Its small extension registers the model and two typed HTTP routes; it calls vLLM's `EngineClient.encode` for each candidate. Standard `vllm serve` arguments remain the control surface. `scripts/serve.sh` supplies the checkpoint and tested Jev defaults, then forwards every additional argument unchanged (`"$@"`). For example:

```bash
CUDA_VISIBLE_DEVICES=4 bash scripts/serve.sh \
  --max-num-seqs 64 --max-num-batched-tokens 8192
```

The [vLLM pooling guide](https://docs.vllm.ai/en/v0.29.0/models/pooling_models/) distinguishes pooling APIs from generation APIs. “All vLLM features” therefore means retaining applicable native engine and serving features, not inventing generation-token behavior for a scalar classifier.

| Capability | Jev status in this repository | Control / limit |
|---|---|---|
| Model registry and endpoint plugins | Implemented using official plugin hooks | `VLLM_PLUGINS=vllm_jev_model,vllm_jev_endpoint`; [endpoint plugin contract](https://docs.vllm.ai/en/v0.29.0/design/endpoint_plugins/) |
| Continuous scheduling, candidate batching and chunked prefill | Inherited from native vLLM; live HTTP concurrency tested | `--max-num-seqs`, `--max-num-batched-tokens`, `--max-num-queued-reqs` passed to `vllm serve`; [engine arguments](https://docs.vllm.ai/en/v0.29.0/configuration/engine_args/) |
| Priority scheduling | Jev Choice and Batch Choice pass each decision's optional `priority` to all its candidate requests | Start with `--scheduling-policy priority`; lower numeric values run earlier. The default FCFS policy ignores request priority; [scheduler arguments](https://docs.vllm.ai/en/v0.29.0/configuration/engine_args/) |
| GPU memory and KV capacity | 90% startup and optional fixed KV size tested | `--gpu-memory-utilization` or `--kv-cache-memory-bytes`, never both for an effective utilization setting; [cache config](https://docs.vllm.ai/en/v0.29.0/api/vllm/config/cache/) |
| Prefix caching and Qwen3.5 hybrid state | Tested; request-local salt isolates decisions while candidates share it | `--enable-prefix-caching`, `--mamba-cache-mode align`; `use_prefix_cache=false` on Jev Choice is a diagnostic; [automatic prefix caching](https://docs.vllm.ai/en/v0.29.0/features/automatic_prefix_caching/) |
| Compilation, piecewise CUDA graphs, async scheduling | Tested on the full 14,042-item MMLU run | Enabled by current Jev launch defaults; `VLLM_JEV_EAGER=1` and `VLLM_JEV_ASYNC=0` compare alternatives |
| Native pooling HTTP APIs | `/v1/models`, `/classify` and `/pooling` returned 200 alongside Jev Choice/Batch Choice | `/classify` returns per-input model classification, not Jev's cross-option probability distribution; [pooling APIs](https://docs.vllm.ai/en/v0.29.0/models/pooling_models/) |
| Observability | `/metrics` gauges/counters and per-choice timing/cache counts tested | Consume the native `/metrics` endpoint; detailed load-test scripts are local-only; [vLLM metrics](https://docs.vllm.ai/en/v0.29.0/design/metrics/) |
| Two-GPU scale-out | Native DP=2 starts and serves Choice, but underperformed decision-level routing on long questions | Native `--data-parallel-size 2` routes candidates independently; use separate replicas plus `scripts/router.py` when branch prefix reuse matters; [data parallel deployment](https://docs.vllm.ai/en/v0.29.0/serving/data_parallel_deployment/) |
| Tensor/pipeline parallelism | vLLM CLI available; Jev/Qwen3.5 checkpoint not validated in those modes | The 4B model fits one A800, so replicate for independent decisions; [parallelism guidance](https://docs.vllm.ai/en/v0.29.0/serving/parallelism_scaling/) |
| FP8 KV cache, KV/weight offloading, quantization | Native switches exist but Jev quality/performance not validated | Pass relevant `vllm serve` flags only after an accuracy and latency audit; [engine arguments](https://docs.vllm.ai/en/v0.29.0/configuration/engine_args/) |
| Runtime LoRA switching | Not implemented for this exported checkpoint | The trained LoRA is merged into the backbone and the scalar head is exported with it; serving another head needs a new export or additional adapter-aware model work |
| Streaming tokens, sampling, speculative decoding, structured generation and tool calling | Not applicable to this scalar pooling task | Those features operate on generated output tokens; Jev instead returns a typed probability object after candidate scoring |
| Multimodal inputs | Not supported by this text-only Qwen3.5 Jev adapter | The model plugin rejects multimodal M-RoPE features |

The official data-parallel API server balances individual engine requests using queue state. Jev fans one decision out into multiple pooling requests, which need **decision affinity** for best shared-prefix reuse. In a fixed long-question comparison, native DP=2 measured 12.27 decisions/s and 42.2% cached tokens at concurrency 16; two separate replicas behind the decision-level router measured 19.80 decisions/s and 64.4% cached tokens. An upstream-quality fix would add stable rank affinity to the pooling encode API, then choose one rank per Jev decision without an external router. The current public `EngineClient.encode` signature has no DP-rank parameter, so this repository does not patch private vLLM internals to simulate it. Detailed workload records are kept outside the source repository.

The plugin routes and vLLM pooling routes are bound to loopback in the provided launch commands. vLLM's [security documentation](https://docs.vllm.ai/en/v0.29.0/usage/security/) lists several pooling endpoints as outside its API-key protection; the Jev plugin does not add authentication. Put an authenticating proxy in front before exposing these routes to a network.
