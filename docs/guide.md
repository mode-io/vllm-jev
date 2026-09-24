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

The same HTTP requests work with all four models.

| Model | Start server |
|---|---|
| [ZefanCai/Open-Jev-2B](https://huggingface.co/ZefanCai/Open-Jev-2B) | `vllm-jev serve ZefanCai/Open-Jev-2B` |
| [ZefanCai/Open-Jev-9B](https://huggingface.co/ZefanCai/Open-Jev-9B) | `vllm-jev serve ZefanCai/Open-Jev-9B` |
| [IamBusy/OpenJev-0.6B](https://huggingface.co/IamBusy/OpenJev-0.6B) | `vllm-jev serve IamBusy/OpenJev-0.6B` |
| [lostargon/Tiny-Jev](https://huggingface.co/lostargon/Tiny-Jev) | `vllm-jev serve lostargon/Tiny-Jev` |

## Serving options

Choose a GPU or pass regular vLLM options after the model ID:

```bash
CUDA_VISIBLE_DEVICES=0 vllm-jev serve ZefanCai/Open-Jev-2B --port 9000
```

The default port is 8795, the GPU memory budget is 90%, and the maximum sequence length is 4,096 tokens. Run `vllm-jev serve --help=all` to see additional vLLM options.

Set `HF_HOME` for the Hugging Face base-model cache. Set `VLLM_JEV_HOME` for the downloaded Jev files and prepared model; its default is `.local/` in your current directory.

## HTTP API

| Route | What it does |
|---|---|
| `POST /plugins/vllm-jev/choice` | Ask one multiple-choice question; returns a label and probabilities. |
| `POST /plugins/vllm-jev/batch` | Ask several multiple-choice questions together. |
| `POST /v1/systemone` | Ask Choice, Noul (yes/no), or Score questions about a shared state. |

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
