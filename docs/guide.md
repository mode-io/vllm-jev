# vLLM Jev integration guide

This guide documents the **vLLM 0.29.0** plugin and its verified behavior. It follows vLLM's [plugin lifecycle](https://docs.vllm.ai/en/v0.29.0/design/plugin_system/), [endpoint plugin interface](https://docs.vllm.ai/en/v0.29.0/design/endpoint_plugins/) and [pooling model APIs](https://docs.vllm.ai/en/v0.29.0/models/pooling_models/). The [README](../README.md) provides the one-command public-model start.

## Public checkpoints and setup

`bash scripts/quickstart.sh` selects the public Open-Jev 2B profile; `bash scripts/quickstart.sh 9b` selects 9B. Both profiles pin the released adapter repository and exact upstream Qwen revision in [`vllm_jev.public_models`](../vllm_jev/public_models.py). The script bootstraps `uv`, installs the isolated runtime and plugin, downloads the public LoRA/head and base model, exports a merged checkpoint, verifies its checksums and executes `vllm serve`. Downloads and exported weights stay outside Git under `.local/` or `VLLM_JEV_HOME`.

Set `JEV_GPU_POOL=0,1` to limit automatic GPU selection, or use `CUDA_VISIBLE_DEVICES` to choose one explicitly. The downloader uses public Hugging Face repositories without sending an implicit access token. The repository revisions and released file hashes are pinned and checked before export.

| Profile | Public release | Base model | Saved temperature | Approximate exported size |
|---|---|---|---:|---:|
| `2b` | [Open-Jev-2B](https://huggingface.co/ZefanCai/Open-Jev-2B) | Qwen3.5-2B | 1.518796342858676 | 3.6 GB |
| `9b` | [Open-Jev-9B](https://huggingface.co/ZefanCai/Open-Jev-9B) | Qwen3.5-9B | 1.8969118766347646 | 15 GB |

Both public releases contain a LoRA adapter, `head.pt`, `model.json` and `temperature.json`; they do not contain the base-model weights. Their cards license the released adapter/head under Apache-2.0. The importer checks each package's pinned model identity and published adapter/head/temperature hashes before merging. An incomplete base snapshot without the tokenizer chat template fails export explicitly rather than starting a model that cannot format requests.

This is an independently implemented vLLM loader for the public artifacts, not the upstream Open-Jev loader. The small HF/PEFT comparisons below check numerical proximity on one prompt per profile; they do not establish exact parity across the full public evaluation.

To export an independently trained adapter/head instead, install as described in the README and run:

```bash
runtime/bin/python -m vllm_jev.export \
  --base /path/to/qwen3.5-base \
  --adapter /path/to/adapter \
  --head /path/to/head.pt \
  --output /path/to/jev-checkpoint
```

If `temperature.json` is alongside `head.pt`, its positive finite temperature is saved in the export manifest and used as the service default; otherwise the default is `1.0`. A request may explicitly override Choice temperature.

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

`vllm_jev.export` merges the trained LoRA into the backbone and writes the scalar head into a safetensors checkpoint. It handles single-file and sharded Qwen3.5 exports and omits an unused generative `lm_head` when present. `vllm_jev.checkpoint` verifies the manifest and checksums. Inference loads this checkpoint; no training occurs in the serving process. The endpoint contains no independent GPU execution loop: vLLM supplies batching, scheduling, compilation and hybrid KV-cache management. The plugin formats prompts, issues pooling calls and normalizes their scalar results.

## HTTP API

### Public System One shape

`POST /v1/systemone` accepts a shared `state` and a `questions` mapping. Each question has `instructions`, `type` and `criteria`:

| Type | `criteria` | Returned answer |
|---|---|---|
| `choice` | Map of 1–255 candidate names to optional descriptions | Selected name, probabilities and choice confidence. |
| `noul` | Optional `{ "true": description, "false": description }` | Probability that the answer is yes. |
| `score` | Array of 2–10 ordered descriptions | Expected ordinal score, probabilities, confidence and legend. |

The route accepts up to 64 questions and 256 candidate scoring sequences per request. It uses one cache salt for the request and the checkpoint's saved calibration temperature. `usage.output_tokens` is zero because the decision does not generate answer text. The response is shaped like the released Open-Jev System One API; limits and metadata reflect this vLLM implementation.

### Direct Choice and Batch Choice

`POST /plugins/vllm-jev/choice` takes:

| Field | Type | Meaning |
|---|---|---|
| `state` | text or JSON object/array | Context shared by every candidate; it may be empty. |
| `question` | nonempty string | The question to decide. |
| `options` | 2–255 distinct nonempty strings | Candidate answers, in returned order. |
| `temperature` | optional positive float | Override the checkpoint's saved calibration temperature (or `1.0` if none was saved). |
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

The one-command script invokes the ordinary vLLM CLI. For a previously exported checkpoint, the equivalent direct launch is:

```bash
export CUDA_VISIBLE_DEVICES=0
export VLLM_PLUGINS=vllm_jev_model,vllm_jev_endpoint
export VLLM_WORKER_MULTIPROC_METHOD=spawn
runtime/bin/vllm serve /path/to/jev-checkpoint \
  --runner pooling --convert none --max-model-len 4096 \
  --enable-prefix-caching --mamba-cache-mode align \
  --mamba-ssm-cache-dtype float32 --async-scheduling \
  --gpu-memory-utilization 0.90 \
  --served-model-name vllm-jev --host 127.0.0.1 --port 8795
```

`scripts/serve.sh` is a thin optional launcher: it verifies the checkpoint, supplies tested Jev defaults and forwards any additional vLLM arguments unchanged. The most relevant settings are:

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

Use vLLM's native [data parallel deployment](https://docs.vllm.ai/en/v0.29.0/serving/data_parallel_deployment/) for multiple independent replicas, or tensor parallelism when a model does not fit on one GPU. The endpoint submits one pooling request per candidate, so native data parallelism may send candidates from one decision to different ranks; prefix-cache reuse across those ranks is not guaranteed.

## Validation and boundaries

Both public profiles have started with the pinned runtime and passed Choice/Noul/Score response checks. A single prompt per profile matched the original HF/PEFT implementation's selected candidate; the largest candidate-probability difference was **0.00195 for 2B** and **0.00078 for 9B**. These spot checks do not establish full numerical or benchmark parity. Load tests exercised varied prompt lengths, option counts and concurrency without response errors, but they do not establish long-running reliability. Raw workload scripts and machine-specific measurements are kept outside the source repository.

The one-command profiles support the publicly released 2B/9B scalar-head Open-Jev family. Other popular Jev-style projects require different adapters:

| Project | Why its weights are not loaded by this plugin |
|---|---|
| [Laya](https://github.com/NandhaKishorM/laya) | Separate encoder and typed decision-head architecture. |
| [SemIf](https://github.com/TheoLeeCJ/SemIf-OpenJev) | Reads option-token logits from a frozen model; no trained scalar LoRA/head package. |
| [razorback16/openjev](https://github.com/razorback16/openjev) | Packages a separate DiffusionGemma/vLLM server and API; its weights and readout are not this Qwen3.5 scalar classifier. |
| [openjev/openjev](https://huggingface.co/openjev/openjev) | A 27B generative logprob model with its own helper, not an independent-candidate scalar head. Its weights have a noncommercial license. |
| [autotrust/JEV](https://huggingface.co/autotrust/JEV) | Uses a different 24-slot decision head and request template. |

The related [Open-Jev 27B v1.1](https://huggingface.co/ZefanCai/Open-Jev-27B-v1.1) uses a scalar head but a different Qwen3.8-27B backbone; it has not been validated with this Qwen3.5 model plugin. These projects are not advertised as interchangeable checkpoints.

The `/v1/systemone` route gives clients a common Choice/Noul/Score request shape across several Jev-style services. That is **API-level compatibility**, not permission to load their different model weights into this plugin or a claim of identical probabilities.

Generation-token features such as streaming text, speculative decoding, tool calls and constrained generation do not apply to this scalar pooling task. The exported LoRA is merged rather than hot-swapped, and this adapter is text-only. vLLM's optional gRPC server can start with the checkpoint, but its current `Embed` RPC requests the `embed` pooling task; this Jev model supports `classify`, so the native gRPC interface does not provide a Jev Choice RPC. An experimental gRPC-to-HTTP bridge worked locally but showed no speed advantage and is not part of this repository.

## Upstream boundary and security

A reviewable upstream change would retain the model registry entry, scalar pooling adapter, prompt/normalization code and opt-in endpoint plugin. vLLM's public pooling `EngineClient.encode` in this pinned version does not expose a DP-rank affinity parameter. Adding that option could keep candidates of one decision on a common data-parallel rank; this repository does not patch private engine internals.

The example server binds to `127.0.0.1`. Neither the custom route nor all native pooling routes should be assumed protected by `--api-key`; place an authenticating, allowlisted reverse proxy in front of any network-exposed deployment. See vLLM's [security guidance](https://docs.vllm.ai/en/v0.29.0/usage/security/). Model weights, adapters, raw benchmarks, stress scripts, credentials and machine-specific paths stay outside the repository.
