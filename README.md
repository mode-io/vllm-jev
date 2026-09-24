<h1 align="center">vLLM Jev</h1>

<h3 align="center">Candidate decisions and probabilities, served by vLLM</h3>

<p align="center">
  <a href="docs/guide.md"><b>Documentation</b></a> ·
  <a href="#getting-started"><b>Getting Started</b></a> ·
  <a href="#example"><b>Example</b></a>
</p>

---

## About

vLLM Jev serves the public Open-Jev 2B and 9B checkpoints through [vLLM](https://github.com/vllm-project/vllm). Give it a question and candidate answers; it returns a label and a probability for each candidate.

- **Native vLLM serving:** scheduling, batching, compilation, KV cache, and metrics.
- **Parallel candidate scoring:** one pooling score per option, with shared-prefix caching within a decision.
- **Structured decisions:** Choice, Noul (yes/no), and Score (ordered levels) over HTTP.
- **Public checkpoints:** downloads the released adapter and score head, then prepares a vLLM model. Serving does not train.

## Getting Started

On Linux with an NVIDIA GPU, install [uv](https://docs.astral.sh/uv/getting-started/installation/) and run these commands from the repository directory:

```bash
# Use the same GPU to prepare and serve the model.
export CUDA_VISIBLE_DEVICES=0

# Install vLLM and the plugin in .local/runtime.
bash scripts/install_runtime.sh .local

# Download and prepare the public Open-Jev-2B model.
HF_HOME="$PWD/.local/.cache" \
  .local/runtime/bin/python -m vllm_jev.public_models ZefanCai/Open-Jev-2B --workspace .local

# Start the server on GPU 0.
VLLM_JEV_MODEL_ID=ZefanCai/Open-Jev-2B \
  bash scripts/serve.sh .local/checkpoint/ZefanCai/Open-Jev-2B
```

The server listens at `http://127.0.0.1:8795`. Check it with `curl -f http://127.0.0.1:8795/health`.

For [Open-Jev-9B](https://huggingface.co/ZefanCai/Open-Jev-9B), replace `ZefanCai/Open-Jev-2B` with `ZefanCai/Open-Jev-9B`. To combine all three steps, run `CUDA_VISIBLE_DEVICES=0 bash scripts/quickstart.sh ZefanCai/Open-Jev-2B` (or `ZefanCai/Open-Jev-9B`). Downloads are reused on later runs.

## Example

With the server running, send a Choice request:

```bash
curl -sS http://127.0.0.1:8795/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"state":"I was charged twice and want a refund.","questions":{"intent":{"type":"choice","instructions":"Choose the customer intent.","criteria":{"billing":"A payment or refund issue","technical":"A malfunction or setup issue","other":"Another request"}}}}'
```

The response contains `answers.intent.choice` and `answers.intent.probabilities` for the supplied labels.

See the [integration guide](docs/guide.md) for API fields, model export, storage paths, vLLM options, and multi-GPU serving. This plugin targets **vLLM 0.29.0**, **Python 3.12+**, and the Qwen3.5 scalar-head [Open-Jev-2B](https://huggingface.co/ZefanCai/Open-Jev-2B) / [Open-Jev-9B](https://huggingface.co/ZefanCai/Open-Jev-9B) releases.

## License

Apache-2.0. The Choice prompt follows MIT-licensed [Open-Jev](https://github.com/Zefan-Cai/Open-Jev); see [third-party notices](THIRD_PARTY_NOTICES.md).
