# Integration guide

This guide covers public model setup, the HTTP API, and vLLM serving options. Start with the [README](../README.md) if you want the shortest path to a working server. The plugin targets **vLLM 0.29.0**.

## Public models

Choose one of the released Open-Jev profiles:

| Profile | Release | Base | Approximate exported size |
|---|---|---|---:|
| `2b` | [Open-Jev-2B](https://huggingface.co/ZefanCai/Open-Jev-2B) | Qwen3.5-2B | 3.6 GB |
| `9b` | [Open-Jev-9B](https://huggingface.co/ZefanCai/Open-Jev-9B) | Qwen3.5-9B | 15 GB |

```bash
bash scripts/quickstart.sh       # 2B
bash scripts/quickstart.sh 9b    # 9B
```

The script installs a pinned runtime, downloads the base model and released LoRA/score head, exports a vLLM checkpoint, then runs `vllm serve`. It verifies the release hashes before export. Files are saved under `.local/` by default. Set `VLLM_JEV_HOME` to use another directory.

The released calibration temperature is loaded with each profile. A direct Choice request can override it. To restrict GPU selection, set `JEV_GPU_POOL=0,1`; to select one device, set `CUDA_VISIBLE_DEVICES=0`.

### Export another Open-Jev checkpoint

For a compatible Qwen3.5 text backbone, LoRA adapter, and scalar `head.pt`:

```bash
.local/runtime/bin/python -m vllm_jev.export \
  --base /path/to/qwen3.5-base \
  --adapter /path/to/adapter \
  --head /path/to/head.pt \
  --output /path/to/jev-checkpoint
```

Place `temperature.json` beside `head.pt` to use its temperature by default. Otherwise the default is `1.0`. The exporter merges the LoRA and writes a safetensors classifier. Serving loads the exported model; it performs no training.

## How serving works

```text
Choice request
  → format one prompt per candidate
  → score candidates through vLLM pooling
  → apply temperature and softmax
  → return the chosen label and probabilities
```

The package uses two official [vLLM plugin entry points](https://docs.vllm.ai/en/v0.29.0/design/plugin_system/):

| Entry point | Role |
|---|---|
| `vllm.general_plugins` | Register the Qwen3.5 scalar classifier. |
| `vllm.endpoint_plugins` | Add the Choice and System One HTTP routes. |

vLLM handles scheduling, batching, compilation, and KV-cache management. The endpoint sends each candidate to `EngineClient.encode` with `task="classify"`. Candidates within a request share a cache salt, which allows common-prefix reuse.

## HTTP API

### System One

`POST /v1/systemone` accepts a shared `state` and a map of `questions`. Each question has `type`, `instructions`, and `criteria`. See the [README example](../README.md#make-a-decision).

| Type | `criteria` | Result |
|---|---|---|
| `choice` | Map of candidate names to descriptions | Selected name and candidate probabilities |
| `noul` | Optional `true` and `false` descriptions | Probability of “yes” |
| `score` | Array of 2–10 ordered descriptions | Expected score and level probabilities |

A request may contain up to 64 questions and 256 candidate sequences. `usage.output_tokens` is zero because the model returns pooled scores rather than generating answer text.

### Choice and batch

`POST /plugins/vllm-jev/choice` takes `state`, `question`, and `options` (2–255 distinct strings). Optional fields are:

| Field | Default | Purpose |
|---|---|---|
| `temperature` | Checkpoint temperature | Scale scores before softmax. |
| `cache_salt` | Fresh value per decision | Reuse a prefix across requests when you supply the same salt. |
| `use_prefix_cache` | `true` | Disable cache reads for a diagnostic comparison. |
| `priority` | `0` | Set scheduling priority when vLLM uses priority scheduling. |

The response includes `choice`, `scores`, `probabilities`, token/cache counts, and timing. Probabilities sum to one over the supplied options. `confidence` measures how far the selected probability is above a uniform choice; it is not an estimate of answer correctness.

`POST /plugins/vllm-jev/batch` accepts `{"requests": [<Choice request>, ...]}`. It supports up to 64 decisions and 256 candidates in total. vLLM batches the underlying candidate requests.

```bash
curl -sS http://127.0.0.1:8795/plugins/vllm-jev/choice \
  -H 'Content-Type: application/json' \
  -d '{"state":"The sky is blue.","question":"Choose the true statement.","options":["The sky is blue","The sky is green"]}'
```

Native `/classify` and `/pooling` remain available for per-input scores. The Jev endpoints add cross-candidate normalization.

## Serve with vLLM directly

For an exported checkpoint, use the standard CLI:

```bash
export CUDA_VISIBLE_DEVICES=0
export VLLM_PLUGINS=vllm_jev_model,vllm_jev_endpoint
export VLLM_WORKER_MULTIPROC_METHOD=spawn
.local/runtime/bin/vllm serve /path/to/jev-checkpoint \
  --runner pooling --convert none --max-model-len 4096 \
  --enable-prefix-caching --mamba-cache-mode align \
  --mamba-ssm-cache-dtype float32 --async-scheduling \
  --gpu-memory-utilization 0.90 \
  --served-model-name vllm-jev --host 127.0.0.1 --port 8795
```

`scripts/serve.sh` provides these defaults, verifies the checkpoint, and forwards extra vLLM flags. Set `VLLM_JEV_WORKSPACE=.local` and `VLLM_JEV_CHECKPOINT` to the exported directory before using it directly. If you chose another workspace with `VLLM_JEV_HOME`, use its `runtime/bin` in these commands.

Use native vLLM options to tune the server:

| Option | Use |
|---|---|
| `--gpu-memory-utilization` | Budget GPU memory for model execution and KV cache. |
| `--kv-cache-memory-bytes` | Set a fixed KV-cache size instead. |
| `--max-num-seqs`, `--max-num-batched-tokens` | Control concurrency and tokens per step. |
| `--scheduling-policy priority` | Enable the Choice `priority` field. |
| `--data-parallel-size` | Run independent replicas on multiple GPUs. |

See the official [engine arguments](https://docs.vllm.ai/en/v0.29.0/configuration/engine_args/) and [data-parallel guide](https://docs.vllm.ai/en/v0.29.0/serving/data_parallel_deployment/). `/health` reports readiness; `/metrics` exposes request and cache metrics. Data parallelism may place candidates from one decision on different ranks, so shared-prefix hits can vary.

## Compatibility and validation

The one-command profiles load the released **Qwen3.5 scalar-head Open-Jev 2B/9B** models. Other Jev-style projects, including [Laya](https://github.com/NandhaKishorM/laya) and [SemIf](https://github.com/TheoLeeCJ/SemIf-OpenJev), use different readouts. Their weights need their own model adapters. The `/v1/systemone` route provides a familiar request shape; it does not make those weights interchangeable.

Both public profiles started with the pinned runtime and passed Choice, Noul, and Score response checks. One prompt per profile selected the same candidate as the original HF/PEFT path. Maximum candidate-probability differences on those prompts were **0.00195 (2B)** and **0.00078 (9B)**. These are spot checks, not full accuracy benchmarks. Load tests exercised varied prompt lengths, option counts, and concurrency without response errors.

This is a text-only pooling model. It returns decision scores, so generation features such as text streaming and tool calls are outside this API. The example binds to `127.0.0.1`; follow vLLM's [security guidance](https://docs.vllm.ai/en/v0.29.0/usage/security/) before exposing the service on a network.
