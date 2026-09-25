# User guide

vLLM Jev serves supported Jev models through vLLM. Send a state and a question to get a label, a probability distribution, a yes/no probability, or an ordered score. Serving does not train the model.

## Installation

On Linux with an NVIDIA GPU, [install uv](https://docs.astral.sh/uv/getting-started/installation/). From a clone of this repository:

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install .
```

## Quickstart

Choose a supported Hugging Face model ID:

```bash
vllm-jev serve ZefanCai/Open-Jev-2B
```

The command downloads and prepares the model, then starts the server at `http://127.0.0.1:8795`. Check it from another terminal:

```bash
curl -f http://127.0.0.1:8795/health
```

## Supported models

Input support depends on the model:

| Model | Input | Start server |
|---|---|---|
| [ZefanCai/Open-Jev-2B](https://huggingface.co/ZefanCai/Open-Jev-2B) | Text | `vllm-jev serve ZefanCai/Open-Jev-2B` |
| [ZefanCai/Open-Jev-9B](https://huggingface.co/ZefanCai/Open-Jev-9B) | Text | `vllm-jev serve ZefanCai/Open-Jev-9B` |
| [IamBusy/OpenJev-0.6B](https://huggingface.co/IamBusy/OpenJev-0.6B) | Text | `vllm-jev serve IamBusy/OpenJev-0.6B` |
| [lostargon/Tiny-Jev](https://huggingface.co/lostargon/Tiny-Jev) | Text | `vllm-jev serve lostargon/Tiny-Jev` |
| [Valen-Team/Valen-Preview-0923](https://huggingface.co/Valen-Team/Valen-Preview-0923) | Text + images | `vllm-jev serve Valen-Team/Valen-Preview-0923` |
| [yah01/vjev-vision](https://huggingface.co/yah01/vjev-vision) | Text + images | `vllm-jev serve yah01/vjev-vision` |
| [yah01/vjev-vision-pilot](https://huggingface.co/yah01/vjev-vision-pilot) | Text + images | `vllm-jev serve yah01/vjev-vision-pilot` |

Image requests use `/v1/systemone`. The vjev pilot is an earlier vision checkpoint; use `yah01/vjev-vision` by default. Video input is not supported by these adapters yet.

## Serving options

Choose a GPU or pass regular vLLM options after the model ID:

```bash
CUDA_VISIBLE_DEVICES=0 vllm-jev serve ZefanCai/Open-Jev-2B --port 9000
```

The default port is 8795 and the GPU memory budget is 90%. The default maximum sequence length is 4,096 tokens for the text models and 8,192 for Valen. Run `vllm-jev serve --help=all` to see additional vLLM options.

Set `HF_HOME` for the Hugging Face base-model cache. Set `VLLM_JEV_HOME` for the downloaded Jev files and prepared model; its default is `.local/` in your current directory.

## HTTP API

| Route | What it does |
|---|---|
| `POST /plugins/vllm-jev/choice` | Ask one multiple-choice question; returns a label and probabilities. |
| `POST /plugins/vllm-jev/batch` | Ask several multiple-choice questions together. |
| `POST /v1/systemone` | Ask Choice, Noul (yes/no), or Score questions about a shared state. |

Valen and vjev use `/v1/systemone` for text and image requests. The two `/plugins/vllm-jev/*` routes are for the four text models.

Noul and Score use `/v1/systemone` with a question `type`. They do not have separate URLs. The native vLLM `/classify` and `/pooling` routes remain available for raw model outputs.

One System One request supports up to 64 questions and 256 candidate sequences.

### Choice

Send a `state`, `question`, and 2–255 `options`:

```bash
curl -sS http://127.0.0.1:8795/plugins/vllm-jev/choice \
  -H 'Content-Type: application/json' \
  -d '{"state":"A customer was charged twice and requests a refund.","question":"What is the issue?","options":["billing","technical","shipping"]}'
```

Read the selected label from `choice` and the distribution from `probabilities`. You can set an optional `temperature` to scale the probabilities.

### Noul (yes/no)

Put a `noul` question under `questions`:

```bash
curl -sS http://127.0.0.1:8795/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"state":"A customer was charged twice and requests a refund.","questions":{"urgent":{"type":"noul","instructions":"Does this require urgent action?"}}}'
```

`answers.urgent.noul` is the probability of yes, between 0 and 1.

### Score

Provide 2–10 ordered levels from lowest to highest:

```bash
curl -sS http://127.0.0.1:8795/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"state":"A customer was charged twice and requests a refund.","questions":{"severity":{"type":"score","instructions":"How severe is it?","criteria":["low","medium","high"]}}}'
```

`answers.severity.score` is the expected level. Levels start at zero, so this example returns a score between 0 and 2. The level distribution is in `answers.severity.probabilities`.

To ask several types together, put multiple entries in the same `questions` map. For a Choice question on this route, use `"type":"choice"` and a `criteria` map from labels to descriptions, as in the [README example](../README.md#example).

### Batch Choice

The batch route accepts up to 64 Choice questions and 256 candidates total:

```bash
curl -sS http://127.0.0.1:8795/plugins/vllm-jev/batch \
  -H 'Content-Type: application/json' \
  -d '{"requests":[{"state":"The sky is blue.","question":"Choose the true statement.","options":["The sky is blue","The sky is green"]},{"state":"The grass is green.","question":"Choose the true statement.","options":["The grass is green","The grass is blue"]}]}'
```

The reply has one Choice result per request under `results`, in the same order.

### Text and image with Valen

Start `vllm-jev serve Valen-Team/Valen-Preview-0923`. Supply PNG or JPEG images as data URLs in `state.messages`; use a text-only `state` string when no image is needed. This example asks about a local image:

```python
import base64
import json
import urllib.request

image = base64.b64encode(open("image.png", "rb").read()).decode()
request = {
    "state": {"messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image}"}},
        {"type": "text", "text": "Look at this image."},
    ]}]},
    "questions": {"color": {"type": "choice", "instructions": "What color is the object?", "criteria": {"red": "Red", "blue": "Blue"}}},
}
body = json.dumps(request).encode()
url = "http://127.0.0.1:8795/v1/systemone"
response = urllib.request.urlopen(urllib.request.Request(url, body, {"Content-Type": "application/json"}))
print(json.load(response)["answers"]["color"])
```

The answer contains `choice`, `confidence`, and `probabilities`. Valen also accepts Noul and Score questions through this route. Its current adapter supports text and images; video is not yet supported.

The same request body works with `vllm-jev serve yah01/vjev-vision` or `vllm-jev serve yah01/vjev-vision-pilot`. Both vjev checkpoints use a trained vision readout and support text-only requests too. They accept multiple images; the author trained these checkpoints on single images, so evaluate multi-image answers for your task. Probabilities may differ slightly from the author's inference runtime.

## Updates

### 2026-09-25

This release adds Valen text/image inference and a fused Tiny-Jev decision head. The Valen adapter runs its Qwen3.5 backbone through native vLLM token pooling and applies the released decision projections at the candidate positions. `vllm-jev serve Valen-Team/Valen-Preview-0923` downloads and prepares the pinned inference weights; it does not train the model.

For 500 image questions from the Valen evaluation game on one A800, the author's sequential inference and vLLM Jev used the same model weights and inputs:

| Measurement | No vLLM Jev | vLLM Jev |
|---|---:|---:|
| Median latency | 231.9 ms | 77.3 ms |
| P95 latency | 240.2 ms | 82.0 ms |
| Throughput | 4.30 req/s | 12.90 req/s |
| Target-support accuracy | 86.8% | 86.8% |

The selected choice matched on 494/500 questions (98.8%). Probability outputs have numerical differences: median maximum per-question probability difference was small, but the worst was about 0.295. Accuracy on this set therefore does not imply exact probability equivalence. An HTTP run at eight concurrent requests completed 500/500 requests with 40.6 req/s and 193.4 ms median latency. The fused Tiny-Jev head improved the measured 16-option, concurrency-16 throughput by 12.1%; four-option requests were roughly flat.

“No vLLM Jev” in the table uses Valen's author runtime.
