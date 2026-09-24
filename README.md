# vLLM Jev

A small vLLM 0.29 plugin for Open-Jev-style candidate scoring. It registers a Qwen3.5 pooling classifier and a typed Choice endpoint. Each candidate receives one scalar score from the merged backbone and readout head; the endpoint returns a softmax across candidates. vLLM owns scheduling, batching, compilation and KV-cache management.

## Install and export

Use an isolated runtime and keep model weights outside the source repository:

```bash
bash scripts/install_runtime.sh "$PWD"
uv pip install --python "$PWD/runtime/bin/python" --no-deps --no-build-isolation -e .

runtime/bin/python -m vllm_jev.export \
  --base "$BASE_MODEL" --adapter "$JEV_ADAPTER" \
  --head "$JEV_HEAD" --output "$JEV_CHECKPOINT"
```

The exporter merges the adapter, stores the scalar head as `score.weight` and `score.bias`, and writes a manifest with file checksums. The model weights, trained adapter and datasets are not part of this repository.

## Serve

Run the official CLI with the installed plugins enabled:

```bash
export VLLM_PLUGINS=vllm_jev_model,vllm_jev_endpoint
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_VISIBLE_DEVICES=0

runtime/bin/vllm serve "$JEV_CHECKPOINT" \
  --runner pooling --convert none --max-model-len 4096 \
  --enable-prefix-caching --mamba-cache-mode align \
  --mamba-ssm-cache-dtype float32 --async-scheduling \
  --gpu-memory-utilization 0.90 \
  --served-model-name vllm-jev --host 127.0.0.1 --port 8795
```

`scripts/serve.sh` verifies the checkpoint and invokes the same CLI. Its defaults can be changed with `VLLM_JEV_CHECKPOINT`, `VLLM_JEV_PORT`, `VLLM_JEV_GPU_MEMORY_UTILIZATION` and the other documented environment variables. Extra CLI arguments pass through unchanged, for example `bash scripts/serve.sh --max-num-seqs 64`.

POST `{"state":"...","question":"...","options":["A","B","C","D"]}` to `/plugins/vllm-jev/choice`. The response includes `choice`, candidate `scores`, conditional `probabilities`, token-cache counts and timing. `/plugins/vllm-jev/batch` accepts multiple decisions. Native vLLM `/classify`, `/pooling`, `/v1/models` and `/metrics` remain available.

The [feature map](docs/vllm_features.md) distinguishes tested pooling features from generation-only features. Two independent replicas can be balanced by `scripts/router.py` at the **decision** level so all candidates of one question stay on the same GPU. Native vLLM data parallelism works but routes individual candidate requests independently, which can reduce shared-prefix reuse for Jev.

Keep these inference routes on loopback or behind an authenticating proxy. The repository intentionally excludes stress scripts, raw benchmark outputs, training data, machine paths and credentials. See [operations](docs/operations.md) and the [upstream PR map](docs/upstreaming.md) for the integration boundary.

This project integrates with [vLLM](https://github.com/vllm-project/vllm) under Apache-2.0 and follows the [Open-Jev](https://github.com/Zefan-Cai/Open-Jev) Choice prompt contract under MIT; see [third-party notices](THIRD_PARTY_NOTICES.md).
