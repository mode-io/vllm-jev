# User guide

Serve Jev models with vLLM. Send text, images, or supported videos with questions; receive labels, probabilities, yes/no answers, or scores.

## Installation

On Linux with an NVIDIA GPU, [install uv](https://docs.astral.sh/uv/getting-started/installation/). From a clone of this repository:

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install .
```

For development, use `uv pip install -e .` to install from your working tree.

On Apple Silicon with macOS 15 or newer, run the setup script from the repository directory. Sourcing it leaves the Python 3.12 environment active:

```bash
source scripts/install_mac.sh
vllm-jev serve Valen-Team/Valen-Preview-0923
```

Valen accepts text, images, and short videos through `/v1/systemone`. For text-only Choice and batch requests, start OpenJev-0.6B or Tiny-Jev. Laya handles Choice, Noul, and Score through the same `/v1/systemone` route on Mac.

OpenJev-0.6B allows 768 tokens for the shared state prefix and 192 tokens for each question-and-candidate branch, including prompt formatting.

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

| Model | Input | Platform | Start server |
|---|---|---|---|
| [ZefanCai/Open-Jev-2B](https://huggingface.co/ZefanCai/Open-Jev-2B) | Text | Linux, Mac | `vllm-jev serve ZefanCai/Open-Jev-2B` |
| [ZefanCai/Open-Jev-9B](https://huggingface.co/ZefanCai/Open-Jev-9B) | Text | Linux | `vllm-jev serve ZefanCai/Open-Jev-9B` |
| [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya) | Text | Linux, Mac | `vllm-jev serve convaiinnovations/laya` |
| [convaiinnovations/laya-multilingual](https://huggingface.co/convaiinnovations/laya-multilingual) | Text | Linux, Mac | `vllm-jev serve convaiinnovations/laya-multilingual` |
| [convaiinnovations/laya-typed-decisions](https://huggingface.co/convaiinnovations/laya-typed-decisions) | Text | Linux, Mac | `vllm-jev serve convaiinnovations/laya-typed-decisions` |
| [IamBusy/OpenJev-0.6B](https://huggingface.co/IamBusy/OpenJev-0.6B) | Text | Linux, Mac | `vllm-jev serve IamBusy/OpenJev-0.6B` |
| [lostargon/Tiny-Jev](https://huggingface.co/lostargon/Tiny-Jev) | Text | Linux, Mac | `vllm-jev serve lostargon/Tiny-Jev` |
| [Valen-Team/Valen-Preview-0923](https://huggingface.co/Valen-Team/Valen-Preview-0923) | Text + images + video | Linux, Mac | `vllm-jev serve Valen-Team/Valen-Preview-0923` |
| [yah01/vjev-vision](https://huggingface.co/yah01/vjev-vision) | Text + images | Linux | `vllm-jev serve yah01/vjev-vision` |
| [yah01/vjev-vision-pilot](https://huggingface.co/yah01/vjev-vision-pilot) | Text + images | Linux | `vllm-jev serve yah01/vjev-vision-pilot` |

Use `yah01/vjev-vision` for the current vjev release; the pilot is an earlier checkpoint. Both were trained on single images.

Image requests use `/v1/systemone`: up to 8 PNG/JPEG images, 8 MiB per image. Valen also accepts one short MP4 video per request; see [Video input](#video-input).

### Additional decision checkpoints (experimental)

These text-only adapters use the same `vllm-jev serve` command and `/v1/systemone` requests for Choice, Noul, and Score. The platforms exercised so far are listed below; other checkpoints in each family remain under validation.

| Model | Tested platform | Start server |
|---|---|---|
| [jaredpalmer/kev-0.8b](https://huggingface.co/jaredpalmer/kev-0.8b) | Linux | `vllm-jev serve jaredpalmer/kev-0.8b` |
| [jaredpalmer/kev-4b](https://huggingface.co/jaredpalmer/kev-4b) | Linux | `vllm-jev serve jaredpalmer/kev-4b` |
| [Mapika/decider-0.8b](https://huggingface.co/Mapika/decider-0.8b) | Linux, Mac | `vllm-jev serve Mapika/decider-0.8b` |
| [Mapika/decider-2b](https://huggingface.co/Mapika/decider-2b) | Linux | `vllm-jev serve Mapika/decider-2b` |
| [sky7350/Mica-v0.1-4B](https://huggingface.co/sky7350/Mica-v0.1-4B) | Linux | `vllm-jev serve sky7350/Mica-v0.1-4B` |
| [flock-io/this-that-model-1.0](https://huggingface.co/flock-io/this-that-model-1.0) | Linux | `vllm-jev serve flock-io/this-that-model-1.0` |
| [flock-io/this-that-model-1.1](https://huggingface.co/flock-io/this-that-model-1.1) | Linux | `vllm-jev serve flock-io/this-that-model-1.1` |
| [flock-io/this-that-model-1.2](https://huggingface.co/flock-io/this-that-model-1.2) | Linux, Mac | `vllm-jev serve flock-io/this-that-model-1.2` |
| [alibiserikbay/JevK5](https://huggingface.co/alibiserikbay/JevK5) | Linux | `vllm-jev serve alibiserikbay/JevK5` |
| [alibiserikbay/JevK5-2B](https://huggingface.co/alibiserikbay/JevK5-2B) | Mac | `vllm-jev serve alibiserikbay/JevK5-2B` |

Linux uses native vLLM pooling; Mac uses MLX. These adapters use only `/v1/systemone`. JevK5 accepts 2–16 options per question; larger option sets and JevK5-Lite are not supported yet. This-That keeps the first 1,536 state tokens before adding the questions.

Tev protocol code is experimental and is not listed as a supported checkpoint; its published weight license is still being clarified.

### RSI-Jev (experimental)

| Model | Input | Platform | Start server |
|---|---|---|---|
| [shgao/rsi-jev-v4.0-vl-qwen3.5-2b](https://huggingface.co/shgao/rsi-jev-v4.0-vl-qwen3.5-2b) | Text + images | Linux | `vllm-jev serve shgao/rsi-jev-v4.0-vl-qwen3.5-2b` |
| [shgao/rsi-jev-v3.0-qwen3.5-2b](https://huggingface.co/shgao/rsi-jev-v3.0-qwen3.5-2b) | Text | Linux | `vllm-jev serve shgao/rsi-jev-v3.0-qwen3.5-2b` |

Both use `/v1/systemone` for Choice, Noul, and Score. Each question is one sequence: the state, the instructions, one `- label: description` line per option, and `Answer:`. A trained head reads the option lines and the last token, and a fitted calibration sets one temperature per question. Option labels are part of the prompt, so renaming a label can change the answer.

- Up to 64 questions, with 2–160 options or levels each.
- A state longer than 2,048 tokens loses tokens from its start; options and the question are kept.
- JSON states and text-only `state.messages` are sent as compact JSON.
- v4.0-VL accepts up to 4 PNG/JPEG images as `image_url` parts in `state.messages`. Text parts and images are joined in order with nothing between them, so start the text after an image with a newline: `[image, "\nWhat is shown?"]`. The images share a budget of 1,024 image tokens. A request whose state would cut into an image is rejected.

The export combines the release's fine-tuned text tower, cast to bf16, with the embedding and vision tower of the pinned `Qwen/Qwen3.5-2B-Base`. Questions that share a state reuse vLLM's prefix cache only in whole cache blocks, so several questions about a short state are each computed in full. Set `VLLM_JEV_RSIJEV_PREFIX_CACHE=0` to stop reading the prefix cache. The head runs on the server's GPU; set `VLLM_JEV_RSIJEV_DEVICE=cpu` to move it.

## Serving options

Choose a GPU or pass regular vLLM options after the model ID:

```bash
CUDA_VISIBLE_DEVICES=0 vllm-jev serve ZefanCai/Open-Jev-2B --port 9000
```

On Linux, the default port is 8795 and the GPU memory budget is 90%. The default maximum sequence length is 4,096 tokens, 8,192 for Valen, 3,072 for RSI-Jev v4.0-VL, 2,048 for RSI-Jev v3.0, 512 for Laya English, and 1,024 for Laya multilingual and typed decisions. Run `vllm-jev serve --help=all` to see additional vLLM options.

On macOS, `vllm-jev serve` supports `--host` and `--port`. Open-Jev-2B, OpenJev-0.6B, Tiny-Jev, and Valen use MLX; Laya uses the published PyTorch MPS runtime. Mac scores and speed may differ from Linux CUDA results.

Cancelling a Mac request stops its HTTP wait. A Laya MPS inference already in progress may finish before the next queued request runs.

Set `HF_HOME` for the Hugging Face base-model cache. Set `VLLM_JEV_HOME` for the downloaded Jev files and prepared model; its default is `.local/` in your current directory.

### Optional acceleration

On Mac, Open-Jev-2B batches eligible short candidate branches. Both Open-Jev-2B and OpenJev-0.6B reuse shared prefixes when beneficial. Their MLX allocator keeps about 512 MiB of reusable freed buffers; model weights and active tensors use additional memory. Send `"use_prefix_cache": false` to the Choice endpoint to disable prefix reuse.

For Laya multilingual on Linux, enable CUDA Graph readouts and batch-invariant execution:

```bash
VLLM_JEV_LAYA_CUDA_GRAPH=1 VLLM_BATCH_INVARIANT=1 \
  vllm-jev serve convaiinnovations/laya-multilingual
```

The graph path handles a single four-option Choice question with at most 320 input tokens. Other requests use the regular readout. Creating a new graph adds a small first-request delay and uses extra GPU memory. Batch-invariant execution stabilizes repeated outputs, but its probabilities can differ slightly from the default mode.

For repeated images on Mac, enable Valen's image cache:

```bash
VLLM_JEV_MAC_IMAGE_CACHE=1 vllm-jev serve Valen-Team/Valen-Preview-0923
```

This opt-in cache retains up to two identical decoded images and their visual features in process memory until eviction or shutdown. It helps repeated single-image requests; new images use the normal path. Multi-image requests and large entries bypass the cache.

## HTTP API

| Route | What it does |
|---|---|
| `POST /v1/systemone` | Choice, Noul (yes/no), and Score for all supported models. |
| `POST /plugins/vllm-jev/choice` | One multiple-choice question for Open-Jev and Tiny-Jev text models. |
| `POST /plugins/vllm-jev/batch` | A batch of questions for Open-Jev and Tiny-Jev text models. |

Noul and Score are question types on `/v1/systemone`. One request supports up to 64 questions and 256 candidates.

All three Laya checkpoints use `/v1/systemone` for Choice, Noul, and Score. Their trained decision heads read the options supplied in each request. The English checkpoint allows 192 tokens for question text and options together; multilingual and typed decisions allow 256.

The English and typed-decision releases contain an out-of-range calibration value for Choice requests with 11 or more options. The pinned Laya runtime clamps it; check confidence on your own data for that case.

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

<a id="text-and-image-with-valen"></a>

### Text and images

Start Valen or either vjev model from the table above. Put image data URLs in `state.messages`; use a `state` string for text-only requests. For a local PNG:

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

`answers.color` contains `choice`, `confidence`, and `probabilities`. The same image request can include Noul and Score questions.

### Video input

Start `vllm-jev serve Valen-Team/Valen-Preview-0923` on Linux or Mac. Video uses the same `/v1/systemone` route and Choice, Noul, and Score questions.

In the Python example above, replace `request["state"]` before sending it:

```python
video = base64.b64encode(open("clip.mp4", "rb").read()).decode()
request["state"] = {"messages": [{"role": "user", "content": [
    {"type": "video_url", "video_url": {
        "url": f"data:video/mp4;base64,{video}", "num_frames": 8,
    }},
    {"type": "text", "text": "Watch the entire video."},
]}]}
```

Use one MP4, up to 16 MiB, 30 seconds, 60 fps, and 1080p. `num_frames` defaults to 8 and accepts even values from 2 to 16. Frames are sampled across the clip. Variable-frame-rate clips use an evenly spaced observation timeline. More frames use more context and memory. Video requests can include text; mixed image/video requests and audio are not supported.

Video input is experimental. The Valen preview was trained for image-based Sokoban; check its decisions on your own video tasks.

## Updates

### 2026-09-29

- Added experimental text decision adapters for Kev, Decider, Mica, This-That, and JevK5. See the [platform table](#additional-decision-checkpoints-experimental).

### 2026-09-28

- Added experimental Valen video input on Linux and Apple Silicon.

### 2026-09-27

- Improved Mac candidate scoring and memory caching.
- Added optional Laya multilingual CUDA Graph readouts and Mac Valen image reuse.

### 2026-09-26

- Added Laya English, multilingual, and typed-decision serving on Linux and Apple Silicon.
- Added Tiny-Jev marker decisions on Apple Silicon.
- Added Open-Jev-2B scalar decisions on Apple Silicon.
- Added Apple Silicon text serving with OpenJev-0.6B and text/image serving with Valen.
- Added one-step setup and the same model-name-based serving command on Mac.
- Improved responsiveness under load and cancellation of pending work.

### 2026-09-25

- Added Valen and vjev text/image serving.
- Improved Tiny-Jev inference speed.
- Improved model validation and request handling.

See the [demos](../README.md#demos) and [performance results](../README.md#inference-performance).
