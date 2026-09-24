# vLLM Jev

**Run Open-Jev decisions on vLLM.** Give the model a question and candidate answers. It scores every candidate and returns a choice with probabilities, without generating probability digits.

`vLLM 0.29.0` · `Python 3.12+` · `Qwen3.5 2B / 9B` · `Apache-2.0`

## Quick start

On a machine with an NVIDIA GPU, Python 3, and access to Hugging Face:

```bash
cd vllm-jev
bash scripts/quickstart.sh
```

The command downloads the public **Open-Jev-2B** release, prepares its vLLM checkpoint, and starts `vllm serve` at `http://127.0.0.1:8795`. Run `bash scripts/quickstart.sh 9b` for **Open-Jev-9B**. Later starts reuse the downloaded files.

Check that the server is ready:

```bash
curl -f http://127.0.0.1:8795/health
```

| To… | Use… |
|---|---|
| Select a GPU | `CUDA_VISIBLE_DEVICES=0 bash scripts/quickstart.sh` |
| Limit automatic GPU selection | `JEV_GPU_POOL=0,1 bash scripts/quickstart.sh` |
| Store downloads elsewhere | `VLLM_JEV_HOME=/path/to/data bash scripts/quickstart.sh` |
| Change the port | `VLLM_JEV_PORT=9000 bash scripts/quickstart.sh` |

A fresh setup needs roughly **30 GB** of disk for 2B or **60 GB** for 9B, including the runtime, download cache, and exported model. The default storage directory is `.local/`. Keep the command running in tmux or a service manager to serve after disconnecting.

## Make a decision

Ask the server to classify a customer request:

```bash
curl -sS http://127.0.0.1:8795/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "state": "I was charged twice and want a refund.",
    "questions": {
      "intent": {
        "type": "choice",
        "instructions": "Choose the customer intent.",
        "criteria": {
          "billing": "A payment or refund issue",
          "technical": "A malfunction or setup issue",
          "other": "Another request"
        }
      }
    }
  }'
```

The response includes the selected label and a probability for each candidate:

```json
{
  "answers": {
    "intent": {
      "type": "choice",
      "choice": "billing",
      "probabilities": {
        "billing": 0.9527,
        "technical": 0.0046,
        "other": 0.0427
      }
    }
  }
}
```

The example shows an abbreviated response from Open-Jev-2B. `/v1/systemone` also accepts **Noul** (yes/no) and **Score** (ordered levels), including several questions that share one state. For a smaller request format, use `/plugins/vllm-jev/choice` or `/plugins/vllm-jev/batch`.

## What you get

```text
state + question + candidates
              │
              ▼
   one vLLM pooling score per candidate
              │
              ▼
       choice + probabilities
```

- **Native vLLM serving:** scheduling, batching, compilation, KV cache, `/metrics`, `/classify`, and `/pooling` come from vLLM.
- **Shared-prefix scoring:** candidates in one decision use the same cache salt, so vLLM can reuse their common prefix.
- **Ready-to-serve releases:** the quick start downloads and prepares the public [Open-Jev-2B](https://huggingface.co/ZefanCai/Open-Jev-2B) or [Open-Jev-9B](https://huggingface.co/ZefanCai/Open-Jev-9B) adapter and score head. Serving does not train the model.

This plugin loads the Qwen3.5 scalar-head Open-Jev releases. Other Jev-style projects use different model architectures. See the [integration guide](docs/guide.md) for checkpoint export, API fields, vLLM flags, multi-GPU setup, and compatibility.

## License

The code is Apache-2.0. The Choice prompt follows MIT-licensed [Open-Jev](https://github.com/Zefan-Cai/Open-Jev); see [third-party notices](THIRD_PARTY_NOTICES.md). The server binds to loopback by default.
