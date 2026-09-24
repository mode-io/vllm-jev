# vLLM Jev

Jev-style multiple-choice scoring as a small [vLLM](https://github.com/vllm-project/vllm) pooling plugin. Each answer candidate receives one scalar classifier score; the plugin normalizes those scores into a choice and a probability distribution. vLLM remains responsible for model execution, batching, scheduling, compilation and KV caching.

This repository targets **vLLM 0.29.0** and an exported **Qwen3.5-4B** classifier checkpoint. It is a plugin, not a vLLM fork. Model weights, training data and load-test scripts are not included.

## Quick start

Install the pinned runtime and this package in an isolated environment:

```bash
bash scripts/install_runtime.sh "$PWD"
uv pip install --python "$PWD/runtime/bin/python" --no-deps --no-build-isolation -e .
```

Export a checkpoint from an existing base-model snapshot, trained Jev adapter and scalar head:

```bash
export BASE_MODEL=/path/to/qwen3.5-4b
export JEV_ADAPTER=/path/to/adapter
export JEV_HEAD=/path/to/head.pt
export JEV_CHECKPOINT=/path/to/jev-checkpoint

runtime/bin/python -m vllm_jev.export \
  --base "$BASE_MODEL" --adapter "$JEV_ADAPTER" \
  --head "$JEV_HEAD" --output "$JEV_CHECKPOINT"
```

Serve it with the standard vLLM CLI:

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

`scripts/serve.sh` provides the same tested defaults and verifies the checkpoint before startup. Set `VLLM_JEV_CHECKPOINT="$JEV_CHECKPOINT"` when using a checkpoint outside the repository. It accepts additional native vLLM options, such as `VLLM_JEV_CHECKPOINT="$JEV_CHECKPOINT" bash scripts/serve.sh --max-num-seqs 64`.

## Make a Choice request

```bash
curl -sS http://127.0.0.1:8795/plugins/vllm-jev/choice \
  -H 'Content-Type: application/json' \
  -d '{"state":"The sky is blue.","question":"What color is the sky?", "options":["A. Blue","B. Green","C. Red"]}'
```

An abridged response from the evaluated checkpoint:

```json
{
  "choice": "A. Blue",
  "probabilities": {
    "A. Blue": 0.9996636611863505,
    "B. Green": 0.00014645269890626773,
    "C. Red": 0.0001898861147431309
  },
  "selected_probability": 0.9996636611863505,
  "confidence": 0.9994954917795258
}
```

The response contains the selected `choice`, raw candidate `scores`, conditional `probabilities`, `selected_probability`, `confidence`, token-cache counts and timing. `/plugins/vllm-jev/batch` accepts multiple decisions. The native vLLM `/classify`, `/pooling`, `/v1/models` and `/metrics` endpoints remain available.

`confidence` is a score adjusted above a uniform-choice baseline; it is **not** a calibrated probability of correctness. The default cache salt is fresh per decision, while candidates of that decision share it for prefix reuse. [The guide](docs/guide.md) explains the API contract, vLLM integration, multi-GPU serving, measured limits and feature compatibility.

## License

Apache-2.0. The Choice prompt contract follows [Open-Jev](https://github.com/Zefan-Cai/Open-Jev), which is MIT licensed. See [third-party notices](THIRD_PARTY_NOTICES.md). Keep the example server on loopback or put authentication in front of it before network exposure.
