<h1 align="center">vLLM Jev</h1>

<h3 align="center">Candidate decisions and probabilities, served by vLLM</h3>

<p align="center">
  <a href="docs/guide.md"><b>Documentation</b></a> ·
  <a href="#getting-started"><b>Getting Started</b></a> ·
  <a href="#example"><b>Example</b></a>
</p>

---

## About

vLLM Jev serves compatible Jev-style checkpoints through [vLLM](https://github.com/vllm-project/vllm). Give it a question and candidate answers; it returns a label and a probability for each candidate.

- **Native vLLM serving:** scheduling, batching, compilation, KV cache, and metrics.
- **Native decision readouts:** scalar candidate branches or marker-token scores, selected from the model format.
- **Structured decisions:** Choice, Noul (yes/no), and Score (ordered levels) over HTTP.
- **Automatic setup:** give the launcher a supported Hugging Face model ID; it selects and verifies the native protocol. Serving does not train.

## Getting Started

On Linux with an NVIDIA GPU, install [uv](https://docs.astral.sh/uv/getting-started/installation/) and run these commands from the repository directory:

```bash
# Use the same GPU to prepare and serve the model.
export CUDA_VISIBLE_DEVICES=0

# Install vLLM and the plugin in .local/runtime.
bash scripts/install_runtime.sh .local

# Download and prepare the public Open-Jev-2B model.
HF_HOME="$PWD/.local/.cache" \
  .local/runtime/bin/python -m vllm_jev.native_models ZefanCai/Open-Jev-2B --workspace .local

# Start the server on GPU 0.
VLLM_JEV_MODEL_ID=ZefanCai/Open-Jev-2B \
  bash scripts/serve.sh .local/checkpoint/ZefanCai/Open-Jev-2B
```

The server listens at `http://127.0.0.1:8795`. Check it with `curl -f http://127.0.0.1:8795/health`.

For another supported model, replace the Hugging Face ID in the download command, `VLLM_JEV_MODEL_ID`, and checkpoint path. The automatic launcher combines all three steps:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/quickstart.sh IamBusy/OpenJev-0.6B
```

It also supports `ZefanCai/Open-Jev-2B`, `ZefanCai/Open-Jev-9B`, and `lostargon/Tiny-Jev`. Downloads are reused on later runs. See the [integration guide](docs/guide.md) for the protocol table and optional `--protocol` override.

## Example

With the server running, send a Choice request:

```bash
curl -sS http://127.0.0.1:8795/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"state":"I was charged twice and want a refund.","questions":{"intent":{"type":"choice","instructions":"Choose the customer intent.","criteria":{"billing":"A payment or refund issue","technical":"A malfunction or setup issue","other":"Another request"}}}}'
```

The response contains `answers.intent.choice` and `answers.intent.probabilities` for the supplied labels.

See the [integration guide](docs/guide.md) for API fields, model export, storage paths, vLLM options, and multi-GPU serving. The plugin targets **vLLM 0.29.0** and **Python 3.12+**.

## License

Apache-2.0. The Choice prompt follows MIT-licensed [Open-Jev](https://github.com/Zefan-Cai/Open-Jev); see [third-party notices](THIRD_PARTY_NOTICES.md).
