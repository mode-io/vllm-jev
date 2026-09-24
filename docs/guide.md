# Integration guide

This guide covers installation, public model setup, the HTTP API, and vLLM serving options. Start with [Quickstart](#quickstart) to launch your first server. The plugin targets **vLLM 0.29.0**.

## Installation

Use Linux with an NVIDIA GPU and Python 3.12 or later. Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run these commands from a clone of this repository:

```bash
uv venv --python 3.12 .local/runtime
source .local/runtime/bin/activate
uv pip install .
```

For development, replace the last command with `uv pip install -e .` so source edits take effect without reinstalling the package.

## Quickstart

Download a supported model and start its server:

```bash
vllm-jev serve ZefanCai/Open-Jev-2B
```

The command downloads and prepares the model, then starts native vLLM at `http://127.0.0.1:8795`. Later runs verify and reuse the prepared checkpoint. Check the server from another terminal:

```bash
curl -f http://127.0.0.1:8795/health
```

Send your first request using the [README example](../README.md#example). To serve another model, replace the Hugging Face ID with one from the table below.

### Serving options

Choose a GPU and pass standard vLLM flags after the model ID:

```bash
CUDA_VISIBLE_DEVICES=0 vllm-jev serve IamBusy/OpenJev-0.6B \
  --gpu-memory-utilization 0.90 --port 9000
```

The launcher defaults to pooling, a 4,096-token context, prefix caching, async scheduling, and a 90% GPU-memory budget. Extra vLLM flags override these defaults. `vllm-jev serve --help` lists launcher options; `vllm-jev serve --help=all` shows native vLLM options. See [Serve with vLLM directly](#serve-with-vllm-directly) for the equivalent native command.

Use `--workspace /path/to/storage` or `VLLM_JEV_HOME` to choose where prepared models are stored. The default is `.local/` in the current directory. Source downloads use `HF_HOME` when set, or `<workspace>/.cache` otherwise. `--protocol auto` detects the model format; an explicit protocol checks that the model matches it.

<details>
<summary>Setup scripts and manual preparation</summary>

From a repository checkout, the bootstrap script can install the runtime and start the same command in one step:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/quickstart.sh ZefanCai/Open-Jev-2B
```

### Prepare and start separately

After installation, you can run each step yourself:

```bash
export CUDA_VISIBLE_DEVICES=0

HF_HOME="$PWD/.local/.cache" \
  .local/runtime/bin/python -m vllm_jev.native_models ZefanCai/Open-Jev-2B --workspace .local

vllm-jev serve ZefanCai/Open-Jev-2B
```

</details>

## Supported models

Choose a model and run its command. Each command downloads, prepares, and serves the model. The default `--protocol auto` selects the matching native vLLM scoring protocol.

| Model | Start server |
|---|---|
| [ZefanCai/Open-Jev-2B](https://huggingface.co/ZefanCai/Open-Jev-2B) | `vllm-jev serve ZefanCai/Open-Jev-2B` |
| [ZefanCai/Open-Jev-9B](https://huggingface.co/ZefanCai/Open-Jev-9B) | `vllm-jev serve ZefanCai/Open-Jev-9B` |
| [IamBusy/OpenJev-0.6B](https://huggingface.co/IamBusy/OpenJev-0.6B) | `vllm-jev serve IamBusy/OpenJev-0.6B` |
| [lostargon/Tiny-Jev](https://huggingface.co/lostargon/Tiny-Jev) | `vllm-jev serve lostargon/Tiny-Jev` |

<details>
<summary>Model protocols and readouts</summary>

| Model ID | Native protocol | Readout |
|---|---|---|
| [`ZefanCai/Open-Jev-2B`](https://huggingface.co/ZefanCai/Open-Jev-2B) | `open_jev_choice` | Qwen3.5 candidate branches + scalar head |
| [`ZefanCai/Open-Jev-9B`](https://huggingface.co/ZefanCai/Open-Jev-9B) | `open_jev_choice` | Qwen3.5 candidate branches + scalar head |
| [`IamBusy/OpenJev-0.6B`](https://huggingface.co/IamBusy/OpenJev-0.6B) | `openjev_branch_v03` | Qwen3 candidate branches + scalar head |
| [`lostargon/Tiny-Jev`](https://huggingface.co/lostargon/Tiny-Jev) | `tiny_jev_marker` | Qwen3 token embeddings + marker-position head |

</details>

The exported model is stored at `<workspace>/checkpoint/<Hugging Face ID>`. The adapter releases are checked against their published file hashes; the Tiny-Jev importer pins the Hub revision and records the source-weight hash.

Choose the GPU explicitly with `CUDA_VISIBLE_DEVICES`. The released calibration temperature is loaded for each model; a direct Choice request can override it. To require a specific format, pass `--protocol open_jev_choice`, `--protocol openjev_branch_v03`, or `--protocol tiny_jev_marker` after the model ID. The override still checks the repository layout. The default is `auto`.

### Export another Open-Jev checkpoint

For a compatible Qwen3.5 text backbone with `head.pt`, or a Qwen3 backbone with `head.safetensors`:

```bash
.local/runtime/bin/python -m vllm_jev.export \
  --base /path/to/qwen3.5-base \
  --adapter /path/to/adapter \
  --head /path/to/head.pt \
  --output /path/to/jev-checkpoint
```

Place `temperature.json` beside `head.pt` for the Qwen3.5 format, or `calibration-v03.json` beside `head.safetensors` for the Qwen3 branch format. The exporter merges the LoRA and writes a safetensors classifier. Serving loads the exported model; it performs no training.

## How serving works

```text
Choice request
  → select a prompt/readout protocol from the checkpoint
  → run the Qwen backbone through vLLM pooling
  → apply temperature and softmax
  → return the chosen label and probabilities
```

The package uses two official [vLLM plugin entry points](https://docs.vllm.ai/en/v0.29.0/design/plugin_system/):

| Entry point | Role |
|---|---|
| `vllm.general_plugins` | Register the Qwen3/Qwen3.5 scalar and token-embedding models. |
| `vllm.endpoint_plugins` | Add the Choice and System One HTTP routes. |

vLLM handles scheduling, batching, compilation, and KV-cache management. The two branch protocols call `EngineClient.encode` with `task="classify"` for each candidate and share a request-local cache salt. Tiny-Jev sends one sequence with `task="token_embed"` and scores its option positions in the endpoint. It skips prefix-cache reads because the readout needs token hidden states that a KV cache does not store.

## HTTP API

### System One

`POST /v1/systemone` accepts a shared `state` and a map of `questions`. Each question has `type`, `instructions`, and `criteria`. See the [README example](../README.md#example).

| Type | `criteria` | Result |
|---|---|---|
| `choice` | Map of candidate names to descriptions | Selected name and candidate probabilities |
| `noul` | Optional `true` and `false` descriptions | Probability of “yes” |
| `score` | Array of 2–10 ordered descriptions | Expected score and level probabilities |

A request may contain up to 64 questions and 256 candidate sequences. `usage.output_tokens` is zero because the model returns pooled scores rather than generating answer text.
For `score`, Tiny-Jev uses the level descriptions as probability keys; the scalar-head protocols use zero-based level indices.

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

Native `/classify` remains available for the scalar-head models, and `/pooling` exposes the selected native task. The Jev endpoints add model-specific prompts and cross-candidate normalization.

## Serve with vLLM directly

For an exported Qwen3.5 scalar-head checkpoint, use the standard CLI:

```bash
export CUDA_VISIBLE_DEVICES=0
export VLLM_PLUGINS=vllm_jev_model,vllm_jev_endpoint
export VLLM_WORKER_MULTIPROC_METHOD=spawn
.local/runtime/bin/vllm serve /path/to/jev-checkpoint \
  --runner pooling --convert none --max-model-len 4096 \
  --enable-prefix-caching --mamba-cache-mode align \
  --mamba-ssm-cache-dtype float32 --async-scheduling \
  --gpu-memory-utilization 0.90 \
  --served-model-name ZefanCai/Open-Jev-2B --host 127.0.0.1 --port 8795
```

`vllm-jev serve` applies these defaults automatically. Pass native vLLM options after the model ID to tune the server:

| Option | Use |
|---|---|
| `--gpu-memory-utilization` | Budget GPU memory for model execution and KV cache. |
| `--kv-cache-memory-bytes` | Set a fixed KV-cache size instead. |
| `--max-num-seqs`, `--max-num-batched-tokens` | Control concurrency and tokens per step. |
| `--scheduling-policy priority` | Enable the Choice `priority` field. |
| `--data-parallel-size` | Run independent replicas on multiple GPUs. |

See the official [engine arguments](https://docs.vllm.ai/en/v0.29.0/configuration/engine_args/) and [data-parallel guide](https://docs.vllm.ai/en/v0.29.0/serving/data_parallel_deployment/). `/health` reports readiness; `/metrics` exposes request and cache metrics. Data parallelism may place candidates from one decision on different ranks, so shared-prefix hits can vary.

## Compatibility and validation

Automatic detection now covers the four models and three protocols in the [supported-model table](#supported-models). It reads the repository layout and validates the selected protocol before export. A Hugging Face name by itself does not imply that a new architecture can use an existing readout.

| Repository | Published format | Current plugin |
|---|---|---|
| [com-kotobalabs/open-jev-deberta-v3-large](https://huggingface.co/com-kotobalabs/open-jev-deberta-v3-large) | DeBERTa encoder and multi-option head | Needs a different pooling model. |
| [apus-ailab/APUS-OpenJev-v1-4B](https://huggingface.co/apus-ailab/APUS-OpenJev-v1-4B) | Qwen3.5 generative model with vocabulary-based candidate scoring | Native generation weights do not supply this plugin's scalar head. |
| [ZefanCai/Open-Jev-27B-v1.1](https://huggingface.co/ZefanCai/Open-Jev-27B-v1.1) | Similar LoRA/head package on Qwen3.8-27B; its pinned config declares `qwen3_5_text` | Auto-detection accepts the architecture preflight, but full weight export and GPU serving have not been run. |

The DeBERTa and APUS repositories were rejected before weight download; neither has passed a native inference test here. The 27B model passed the pinned-config preflight without downloading its roughly 55.6 GB base weights. The `/v1/systemone` route provides a common request shape without treating different weights as interchangeable. Other Jev-style projects, including [Laya](https://github.com/NandhaKishorM/laya) and [SemIf](https://github.com/TheoLeeCJ/SemIf-OpenJev), also use different readouts.

Both public profiles started with the pinned runtime and passed Choice, Noul, and Score response checks. One prompt per profile selected the same candidate as the original HF/PEFT path. Maximum candidate-probability differences on those prompts were **0.00195 (2B)** and **0.00078 (9B)**. These are spot checks, not full accuracy benchmarks. Load tests exercised varied prompt lengths, option counts, and concurrency without response errors.

On seven local Choice/Noul/Score cases, `IamBusy/OpenJev-0.6B` matched the original model's selections. The largest absolute probability difference was **0.0201** with vLLM BF16 and **0.00749** with vLLM FP32 against the author's CPU FP32 loader. `lostargon/Tiny-Jev` matched the original selections on seven cases; its largest probability difference was **0.00261**. Both models completed 64 short, three-question requests at client concurrency 8 with zero response errors. Client p95 was about **93 ms** for the 0.6B branch model and **74 ms** for Tiny-Jev in separate single-GPU tests. These are small parity and stability checks, not benchmark-wide accuracy measurements.

This is a text-only pooling model. It returns decision scores, so generation features such as text streaming and tool calls are outside this API. The example binds to `127.0.0.1`; follow vLLM's [security guidance](https://docs.vllm.ai/en/v0.29.0/usage/security/) before exposing the service on a network.
