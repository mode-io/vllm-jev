<h1 align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/vllm-jev-dark.svg">
    <img src="docs/assets/vllm-jev-light.svg" alt="vLLM Jev" width="430">
  </picture>
</h1>

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

Install vLLM Jev with [`uv`](https://docs.astral.sh/uv/) from the repository directory:

```bash
uv pip install .
```

For development, use an [editable installation](docs/guide.md#installation).

Start a server with a supported Hugging Face model ID:

```bash
vllm-jev serve ZefanCai/Open-Jev-2B
```

The command downloads and prepares the model, then starts native vLLM at `http://127.0.0.1:8795`. Later runs reuse the prepared checkpoint.

<details>
<summary>GPU, port, and serving options</summary>

```bash
CUDA_VISIBLE_DEVICES=0 vllm-jev serve ZefanCai/Open-Jev-2B \
  --gpu-memory-utilization 0.90 --port 8795
```

Add standard `vllm serve` flags to override the defaults. See [serving options](docs/guide.md#serving-options) for cache paths, protocol selection, and the equivalent native command.

</details>

Visit our [documentation](docs/guide.md) to learn more.

- [Installation](docs/guide.md#installation)
- [Quickstart](docs/guide.md#quickstart)
- [List of Supported Models](docs/guide.md#supported-models)

## Supported Models

Choose a model and run its command:

| Model | Start server |
|---|---|
| [ZefanCai/Open-Jev-2B](https://huggingface.co/ZefanCai/Open-Jev-2B) | `vllm-jev serve ZefanCai/Open-Jev-2B` |
| [ZefanCai/Open-Jev-9B](https://huggingface.co/ZefanCai/Open-Jev-9B) | `vllm-jev serve ZefanCai/Open-Jev-9B` |
| [IamBusy/OpenJev-0.6B](https://huggingface.co/IamBusy/OpenJev-0.6B) | `vllm-jev serve IamBusy/OpenJev-0.6B` |
| [lostargon/Tiny-Jev](https://huggingface.co/lostargon/Tiny-Jev) | `vllm-jev serve lostargon/Tiny-Jev` |

## Example

With the server running, send a Choice request:

```bash
curl -sS http://127.0.0.1:8795/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"state":"I was charged twice and want a refund.","questions":{"intent":{"type":"choice","instructions":"Choose the customer intent.","criteria":{"billing":"A payment or refund issue","technical":"A malfunction or setup issue","other":"Another request"}}}}'
```

The response contains `answers.intent.choice` and `answers.intent.probabilities` for the supplied labels.

See the [integration guide](docs/guide.md) for API fields, model export, storage paths, vLLM options, and multi-GPU serving. The plugin targets **vLLM 0.29.0** and **Python 3.12+**.

## Hosted inference performance

Each row uses the same model, input, and GPU for both endpoints. Lower latency and higher throughput are better.

Paired values: **Without vLLM Jev → With vLLM Jev**.

| Model | Input | Concurrency | Latency (ms) | P95 (ms) | Throughput (req/s) | Speedup | Throughput gain |
|---|---|---:|---:|---:|---:|---:|---:|
| [Open-Jev-2B](https://huggingface.co/ZefanCai/Open-Jev-2B) | Short | 1 | 546.6 → 73.0 | 557.6 → 74.3 | 1.83 → 13.66 | 7.5× | 7.5× |
| [Open-Jev-2B](https://huggingface.co/ZefanCai/Open-Jev-2B) | Short | 8 | 4360.5 → 100.5 | 4451.8 → 105.2 | 1.83 → 78.83 | 43.4× | 43.1× |
| [Open-Jev-2B](https://huggingface.co/ZefanCai/Open-Jev-2B) | Long | 1 | 582.3 → 81.9 | 588.1 → 83.8 | 1.72 → 12.20 | 7.1× | 7.1× |
| [Open-Jev-2B](https://huggingface.co/ZefanCai/Open-Jev-2B) | Long | 8 | 4674.5 → 248.9 | 4745.4 → 272.2 | 1.71 → 31.65 | 18.8× | 18.5× |
| [Open-Jev-9B](https://huggingface.co/ZefanCai/Open-Jev-9B) | Short | 1 | 674.9 → 79.2 | 678.3 → 81.1 | 1.48 → 12.58 | 8.5× | 8.5× |
| [Open-Jev-9B](https://huggingface.co/ZefanCai/Open-Jev-9B) | Short | 8 | 5337.6 → 205.2 | 5393.2 → 207.4 | 1.50 → 38.87 | 26.0× | 26.0× |
| [Open-Jev-9B](https://huggingface.co/ZefanCai/Open-Jev-9B) | Long | 1 | 716.9 → 149.7 | 728.1 → 152.5 | 1.39 → 6.67 | 4.8× | 4.8× |
| [Open-Jev-9B](https://huggingface.co/ZefanCai/Open-Jev-9B) | Long | 8 | 5726.5 → 1016.6 | 5749.3 → 1019.3 | 1.40 → 7.87 | 5.6× | 5.6× |
| [OpenJev-0.6B](https://huggingface.co/IamBusy/OpenJev-0.6B)¹ | Short | 1 | 74.6 → 16.4 | 76.5 → 17.0 | 13.35 → 63.03 | 4.5× | 4.7× |
| [OpenJev-0.6B](https://huggingface.co/IamBusy/OpenJev-0.6B)¹ | Short | 8 | 583.2 → 37.5 | 601.6 → 49.3 | 13.62 → 198.70 | 15.5× | 14.6× |
| [OpenJev-0.6B](https://huggingface.co/IamBusy/OpenJev-0.6B)¹ | Long | 1 | 74.2 → 23.8 | 75.5 → 24.2 | 13.46 → 42.55 | 3.1× | 3.2× |
| [OpenJev-0.6B](https://huggingface.co/IamBusy/OpenJev-0.6B)¹ | Long | 8 | 584.5 → 67.9 | 598.0 → 70.6 | 13.63 → 117.24 | 8.6× | 8.6× |
| [Tiny-Jev](https://huggingface.co/lostargon/Tiny-Jev)² | Short | 1 | 28.5 → 10.2 | 29.0 → 17.8 | 35.03 → 88.45 | 2.8× | 2.5× |
| [Tiny-Jev](https://huggingface.co/lostargon/Tiny-Jev)² | Short | 8 | 216.9 → 28.8 | 222.0 → 45.7 | 36.61 → 254.99 | 7.5× | 7.0× |
| [Tiny-Jev](https://huggingface.co/lostargon/Tiny-Jev)² | Long | 1 | 28.4 → 11.5 | 30.1 → 12.4 | 34.59 → 85.70 | 2.5× | 2.5× |
| [Tiny-Jev](https://huggingface.co/lostargon/Tiny-Jev)² | Long | 8 | 223.0 → 43.3 | 224.3 → 59.7 | 35.75 → 179.94 | 5.1× | 5.0× |

Measured on one A800 GPU, with the author and vLLM Jev services run **sequentially**. Each row is one four-option Choice question, 8 warm-up requests, then 64 measured HTTP requests; short and long states contain 67 and 2,058 characters. Latency is the median client latency; p95 means 95% of requests finished within that time. Throughput is completed requests divided by wall time. Every run succeeded **64/64** and both endpoints selected the same option on this repeated test input. Model loading and compilation are excluded.

¹ The author’s published Linux server runs this model on **CPU**. For a same-GPU comparison, a benchmark-only HTTP service moved its original branch scorer to CUDA and cast the backbone to BF16; the scoring code was unchanged. The published CPU server, with its declared Transformers 4.57.6 / PEFT 0.18.0 versions, was also measured: p50 was 667 ms (short) and 1,880 ms (long) at concurrency 1. At concurrency 8, its CPU throughput was 1.48 req/s (short) and 0.53 req/s (long). CPU values are excluded from the speedup columns.

² Tiny-Jev publishes a Python inference API, not an HTTP server. The baseline uses that API behind a minimal benchmark-only HTTP service on the same GPU. The shim serializes calls, as do the other author services.

The Open-Jev 2B/9B author servers used their `--prefix-cache` option and the projects' pinned Transformers 5.10.2 / PEFT 0.19.1 versions. Their optional flash-linear-attention kernels were unavailable, so Transformers used its torch reference path. vLLM Jev used its documented default compiled, async, 90% GPU-memory launch. At concurrency 8, gains include queueing in the author HTTP services and vLLM batching; they are serving-path gains, not isolated kernel speedups. The repeated requests do not measure model accuracy or cross-model quality. [Integration details](docs/guide.md) explain the three readout protocols.

## License

Apache-2.0. The Choice prompt follows MIT-licensed [Open-Jev](https://github.com/Zefan-Cai/Open-Jev); see [third-party notices](THIRD_PARTY_NOTICES.md).
