# vLLM Jev

**Multiple-choice scoring on vLLM's native pooling runner.** Give the service a question and a list of candidate answers; it returns a selected answer and a probability for each candidate.

**vLLM 0.29.0** · **Python 3.12+** · **Qwen3.5-4B** · **Apache-2.0**

[Quick start](#quick-start) · [Request example](#try-it) · [How it works](#how-it-works) · [Integration guide](docs/guide.md)

Jev adds a small model plugin and a typed HTTP endpoint. vLLM continues to handle model execution, batching, scheduling, compilation and KV caching. This repository contains no model weights, training data or load-test scripts.

## Quick start

You need `uv`, a local Qwen3.5-4B base-model snapshot, a trained Jev adapter and its scalar `head.pt`.

**1. Install the pinned runtime and plugin.**

```bash
bash scripts/install_runtime.sh "$PWD"
uv pip install --python "$PWD/runtime/bin/python" --no-deps --no-build-isolation -e .
```

**2. Export the classifier checkpoint.** Keep the source weights and output outside the Git repository.

```bash
export BASE_MODEL=/path/to/qwen3.5-4b
export JEV_ADAPTER=/path/to/adapter
export JEV_HEAD=/path/to/head.pt
export JEV_CHECKPOINT=/path/to/jev-checkpoint

runtime/bin/python -m vllm_jev.export \
  --base "$BASE_MODEL" --adapter "$JEV_ADAPTER" \
  --head "$JEV_HEAD" --output "$JEV_CHECKPOINT"
```

**3. Serve with the official vLLM CLI.** Select a free GPU first.

```bash
export CUDA_VISIBLE_DEVICES=0
export VLLM_PLUGINS=vllm_jev_model,vllm_jev_endpoint
export VLLM_WORKER_MULTIPROC_METHOD=spawn

runtime/bin/vllm serve "$JEV_CHECKPOINT" \
  --runner pooling --convert none --max-model-len 4096 \
  --enable-prefix-caching --mamba-cache-mode align \
  --mamba-ssm-cache-dtype float32 --async-scheduling \
  --gpu-memory-utilization 0.90 \
  --served-model-name vllm-jev --host 127.0.0.1 --port 8795
```

Wait for `curl -f http://127.0.0.1:8795/health` to succeed. The optional `scripts/serve.sh` verifies the exported checkpoint and runs the same server defaults:

```bash
VLLM_JEV_CHECKPOINT="$JEV_CHECKPOINT" bash scripts/serve.sh
```

Additional vLLM flags pass through the script unchanged, for example `--max-num-seqs 64`.

## Try it

```bash
curl -sS http://127.0.0.1:8795/plugins/vllm-jev/choice \
  -H 'Content-Type: application/json' \
  -d '{"state":"The sky is blue.","question":"What color is the sky?","options":["A. Blue","B. Green","C. Red"]}'
```

An abridged response from the evaluated checkpoint:

```json
{
  "choice": "A. Blue",
  "probabilities": {
    "A. Blue": 0.9996636611863505,
    "B. Green": 0.00014645269890626773,
    "C. Red": 0.0001898861147431309
  }
}
```

The full response also includes raw `scores`, `selected_probability`, `confidence`, cached/prompt token counts and timing. Probabilities are normalized **across the options in this request**. `confidence` is a score adjusted above a uniform-choice baseline; it is not a calibrated probability of correctness.

## How it works

```text
question + options
   → one prompt and one scalar classifier score per option
   → softmax across those scores
   → choice + probabilities
```

Candidates in one decision share a cache salt so vLLM can reuse their common prefix; separate decisions receive fresh salts by default. `/plugins/vllm-jev/batch` accepts multiple decisions. Native vLLM `/classify`, `/pooling`, `/v1/models` and `/metrics` remain available.

For API fields, multi-GPU behavior, tuning and measured limitations, see the [integration guide](docs/guide.md). The example server binds to loopback; add an authenticating proxy before exposing it to a network.

## License

Apache-2.0. The Choice prompt contract follows MIT-licensed [Open-Jev](https://github.com/Zefan-Cai/Open-Jev). See [third-party notices](THIRD_PARTY_NOTICES.md).
