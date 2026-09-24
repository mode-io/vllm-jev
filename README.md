# vLLM Jev

**Serve public Open-Jev checkpoints with vLLM's native pooling engine.** Send a question and candidate answers; receive a typed choice and probabilities without generating probability digits.

**vLLM 0.29.0** · **Python 3.12+** · **Qwen3.5 2B / 9B** · **Apache-2.0**

[One-command start](#one-command-start) · [Request example](#try-it) · [How it works](#how-it-works) · [Integration guide](docs/guide.md)

## One-command start

From this repository on a machine with an NVIDIA GPU, Python 3 and network access:

```bash
bash scripts/quickstart.sh
```

The default profile is the public [Open-Jev-2B](https://huggingface.co/ZefanCai/Open-Jev-2B). For the public [Open-Jev-9B](https://huggingface.co/ZefanCai/Open-Jev-9B), run `bash scripts/quickstart.sh 9b`. The script finds a free GPU, bootstraps `uv` if needed, installs the pinned vLLM runtime, downloads the released adapter/head and exact Qwen base revision from Hugging Face, exports a merged classifier, then starts `vllm serve` on `127.0.0.1:8795`. A fresh install downloads several gigabytes of CUDA/runtime packages and model weights and can take tens of minutes on a slower connection; later runs reuse them.

Fresh installs stay under `.local/` and are ignored by Git; an existing `runtime/` beside the script is reused. Put downloads on a larger data volume with `VLLM_JEV_HOME=/path/to/data bash scripts/quickstart.sh`. Allow about **30 GB free disk for 2B** or **60 GB for 9B** on a fresh install, including the downloaded base, exported model, runtime and installer cache. Choose a GPU manually with `CUDA_VISIBLE_DEVICES=0`, or restrict automatic selection with `JEV_GPU_POOL=0,1`. Run it inside tmux or a service manager if it should survive a terminal disconnect, then check `curl -f http://127.0.0.1:8795/health`.

## Try it

The public Open-Jev API shape is available at `/v1/systemone`:

```bash
curl -sS http://127.0.0.1:8795/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"state":"I was charged twice and want a refund.","questions":{"intent":{"type":"choice","instructions":"Choose the customer intent.","criteria":{"billing":"A payment or refund issue","technical":"A malfunction or setup issue","other":"Another request"}}}}'
```

An abridged response from the released 2B checkpoint:

```json
{
  "answers": {
    "intent": {
      "type": "choice",
      "choice": "billing",
      "probabilities": {
        "billing": 0.9518019562719459,
        "technical": 0.004683143491413408,
        "other": 0.043514900236640744
      }
    }
  }
}
```

The same endpoint accepts `choice`, `noul` (yes/no) and `score` (ordinal levels) questions in one request. The simpler `/plugins/vllm-jev/choice` and `/plugins/vllm-jev/batch` routes remain available. Probabilities are normalized over the supplied options; `confidence` is not a calibrated probability of correctness. The public profiles apply their published calibration temperatures by default.

## How it works

```text
state + question + candidates
  → one Qwen3.5 pooling score per candidate
  → per-question calibrated softmax
  → typed answer and probabilities
```

The small model and endpoint plugins use vLLM for scheduling, batching, compilation and KV-cache management. Candidates in one decision share a cache salt for prefix reuse. The trained LoRA and scalar head are merged once at export; serving does not train. Native vLLM `/classify`, `/pooling`, `/v1/models` and `/metrics` remain available.

Public [Laya](https://github.com/NandhaKishorM/laya) and [SemIf](https://github.com/TheoLeeCJ/SemIf-OpenJev) are active Jev-style projects, but use different model/readout designs; their checkpoints cannot be loaded as this Qwen3.5 scalar-head plugin. The one-command profiles intentionally target the released Open-Jev 2B/9B artifacts that match this architecture. For manual checkpoint export, CLI options, multi-GPU behavior and validation limits, see the [integration guide](docs/guide.md).

## License

Code: Apache-2.0. The Choice prompt contract follows MIT-licensed [Open-Jev](https://github.com/Zefan-Cai/Open-Jev); its released 2B/9B adapter weights are Apache-2.0. See [third-party notices](THIRD_PARTY_NOTICES.md). The example server binds to loopback; add authentication before network exposure.
