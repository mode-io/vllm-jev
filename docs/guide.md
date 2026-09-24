# vLLM Jev integration guide

This guide documents the **vLLM 0.29.0** plugin and its verified behavior. It follows vLLM's [plugin lifecycle](https://docs.vllm.ai/en/v0.29.0/design/plugin_system/), [endpoint plugin interface](https://docs.vllm.ai/en/v0.29.0/design/endpoint_plugins/) and [pooling model APIs](https://docs.vllm.ai/en/v0.29.0/models/pooling_models/). The [README](../README.md) is the short installation and request example.

## Design

```text
Choice request (state, question, options)
  └─ one Open-Jev prompt per option, all with one request-local cache salt
      └─ vLLM EngineClient.encode(task="classify", use_activation=False)
          └─ Qwen3.5 text backbone → final hidden state → scalar score head
              └─ per-decision softmax → choice, probabilities and timing
```

The package registers two entry points in `pyproject.toml`:

| Entry point | Implementation | Responsibility |
|---|---|---|
| `vllm.general_plugins` | `vllm_jev.model` | Register the text-only Qwen3.5 sequence-classification architecture with vLLM's model registry. |
| `vllm.endpoint_plugins` | `vllm_jev.endpoint` | Attach typed Choice and Batch Choice routes to the standard vLLM server. |

`vllm_jev.export` merges the trained LoRA into the backbone and writes the scalar head into a safetensors checkpoint. `vllm_jev.checkpoint` verifies its manifest and checksums. Inference loads this checkpoint; no training occurs in the serving process. The endpoint contains no independent GPU execution loop: vLLM supplies batching, scheduling, compilation and hybrid KV-cache management. The plugin formats prompts, issues pooling calls and normalizes their scalar results.

## HTTP API

`POST /plugins/vllm-jev/choice` takes:

| Field | Type | Meaning |
|---|---|---|
| `state` | nonempty text or JSON object/array | Context shared by every candidate. |
| `question` | nonempty string | The question to decide. |
| `options` | 2–255 distinct nonempty strings | Candidate answers, in returned order. |
| `temperature` | positive float, default `1.0` | Scale applied to candidate scores before softmax. |
| `cache_salt` | optional string | Stable cross-request prefix reuse when explicitly supplied; otherwise a fresh salt isolates each decision. |
| `use_prefix_cache` | boolean, default `true` | Set `false` for a diagnostic uncached scoring pass. |
| `priority` | integer, default `0` | Shared by all candidates; effective when the server uses `--scheduling-policy priority` (smaller values run earlier). |

The result includes `choice`, `scores`, `probabilities`, `selected_probability`, `confidence`, `cached_tokens`, `prompt_tokens`, and an elapsed-time breakdown. Probabilities sum to one **across this request's options**; they are not generated digit strings. `confidence = (selected_probability − 1/K) / (1 − 1/K)` for `K` options and is not calibrated correctness probability. An invalid Choice is rejected with a client error. vLLM may return HTTP 400 for request-schema failures and the plugin returns 422 for its semantic validation failures.

`POST /plugins/vllm-jev/batch` takes `{"requests":[<Choice request>, ...]}` with at most 64 decisions and 256 total candidates. It returns ordered `results`, `decision_count`, `candidate_count` and `batch_wall_seconds`. Batch is a client convenience; vLLM already batches candidate engine requests. Native `/classify` and `/pooling` expose per-input classification, while the Jev route adds the cross-option normalization. See vLLM's [classification guide](https://docs.vllm.ai/en/v0.29.0/models/pooling_models/classify/) for the native API distinction.

```bash
curl -sS http://127.0.0.1:8795/plugins/vllm-jev/batch \
  -H 'Content-Type: application/json' \
  -d '{"requests":[{"state":"A short context","question":"Pick one","options":["A","B"]},{"state":"Another context","question":"Pick one","options":["C","D"]}]}'
```

## Serve and tune

Use `vllm serve` as shown in the README. `scripts/serve.sh` is a thin optional launcher: it verifies the checkpoint, supplies tested Jev defaults and forwards any additional vLLM arguments unchanged. The most relevant settings are:

| Setting | Guidance |
|---|---|
| `--gpu-memory-utilization 0.90` | Lets vLLM profile and reserve memory for model execution and KV cache. Idle KV utilization can be zero even while memory is allocated. |
| `--kv-cache-memory-bytes N` | Optional fixed KV reservation; when set, it overrides GPU memory utilization for KV sizing. |
| `--enable-prefix-caching --mamba-cache-mode align` | Enables supported Qwen3.5 hybrid prefix reuse. Cache hits depend on prompt length and block boundaries. |
| `--max-num-seqs`, `--max-num-batched-tokens` | Native concurrency and per-step token budgets; tune with matching workload measurements. |
| `--scheduling-policy priority` | Makes per-Choice `priority` effective. The default FCFS scheduler does not prioritize it. |
| `--async-scheduling` | Tested together with vLLM compilation and piecewise CUDA graphs. |

The corresponding definitions are in the official [engine arguments](https://docs.vllm.ai/en/v0.29.0/configuration/engine_args/) and [cache configuration](https://docs.vllm.ai/en/v0.29.0/api/vllm/config/cache/). Use `/health` for readiness and `/metrics` for running/waiting candidate requests, KV use, prefix hits and preemptions. A high `nvidia-smi` allocation indicates capacity reserved by the process, not high useful-cache occupancy.

## Multiple GPUs

The 4B model fits on one A800. For independent decisions, one model replica per GPU avoids sharding this small model. Start each replica on a different GPU and port, then point `scripts/router.py` at both ports. The router keeps the **whole Choice request** on one replica, so its candidate branches can share prefix cache. It is optional for a single GPU.

Native `vllm serve --data-parallel-size 2` also starts and serves this plugin. Its internal load balancing routes individual pooling calls rather than whole Jev decisions; candidate branches may land on different ranks. On a fixed 128-question long-context test at concurrency 16, native DP measured **12.27 decisions/s and 42.2% cached tokens**; two independent replicas with decision-level routing measured **19.80 decisions/s and 64.4% cached tokens**. These were separate runs, so the observed cache difference is consistent with the routing mechanism but does not prove it explains the entire throughput gap. The official [data-parallel guide](https://docs.vllm.ai/en/v0.29.0/serving/data_parallel_deployment/) describes vLLM's internal balancing.

## Validation and boundaries

In an earlier full 14,042-item MMLU audit, the compiled+async Jev vLLM path answered **10,793/14,042 (76.862%)** correctly, compared with **10,794/14,042 (76.869%)** for the original scorer. Loaded-model wall time was 1,060 s versus 2,873 s in separate A800 runs, an observed 2.71× rate ratio rather than a controlled simultaneous service comparison. Per-choice scores are not bitwise identical across implementations or scheduler modes.

Local-only load tests covered 40 combinations of prompt length, 2–32 options and 1–32 client concurrency: **2,560/2,560** requests passed transport and probability-structure checks. A five-minute mixed-option soak completed **1,829** decisions with zero errors and zero KV preemptions; it does not establish multi-day reliability. Details and raw workload IDs are intentionally excluded from this source repository.

Candidate counts above four in the load tests used synthetic distractors; those runs measure throughput and response integrity, not benchmark accuracy.

Generation-token features such as streaming text, speculative decoding, tool calls and constrained generation do not apply to this scalar pooling task. The exported LoRA is merged rather than hot-swapped, and this adapter is text-only. vLLM's optional gRPC server can start with the checkpoint, but its current `Embed` RPC requests the `embed` pooling task; this Jev model supports `classify`, so the native gRPC interface does not provide a Jev Choice RPC. An experimental gRPC-to-HTTP bridge worked locally but showed no speed advantage and is not part of this repository.

## Upstream boundary and security

A reviewable upstream change would retain the model registry entry, scalar pooling adapter, prompt/normalization code and opt-in endpoint plugin. vLLM's public pooling `EngineClient.encode` in this pinned version does not expose a DP-rank affinity parameter. Adding that option would let all candidates of a Jev decision share one native DP rank without an external router; this repository does not patch private engine internals.

The example server binds to `127.0.0.1`. Neither the custom route nor all native pooling routes should be assumed protected by `--api-key`; place an authenticating, allowlisted reverse proxy in front of any network-exposed deployment. See vLLM's [security guidance](https://docs.vllm.ai/en/v0.29.0/usage/security/). Model weights, adapters, raw benchmarks, stress scripts, credentials and machine-specific paths stay outside the repository.
