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

Valen and vjev image requests use `/v1/systemone`: up to 8 PNG/JPEG images, 8 MiB per image. Valen also accepts one short MP4 video per request. The experimental [RSI-Jev](#rsi-jev-experimental) and [Clef](#clef-experimental) adapters have their own limits and request formats; see [Text and images](#text-and-images) and [Video input](#video-input).

### Additional decision checkpoints (experimental)

These text-only adapters use `vllm-jev serve` and `/v1/systemone` for Choice, Noul, and Score on the listed platforms.

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

### RSI-Jev (experimental)

| Model | Input | Platform | Start server |
|---|---|---|---|
| [shgao/rsi-jev-v4.0-vl-qwen3.5-2b](https://huggingface.co/shgao/rsi-jev-v4.0-vl-qwen3.5-2b) | Text + images | Linux | `vllm-jev serve shgao/rsi-jev-v4.0-vl-qwen3.5-2b` |
| [shgao/rsi-jev-v3.0-qwen3.5-2b](https://huggingface.co/shgao/rsi-jev-v3.0-qwen3.5-2b) | Text | Linux | `vllm-jev serve shgao/rsi-jev-v3.0-qwen3.5-2b` |

Both use `/v1/systemone` for Choice, Noul, and Score. Option labels are part of the prompt, so renaming a label can change the answer.

The Linux adapter was contributed by [Shanghua Gao in PR #2](https://github.com/mode-io/vllm-jev/pull/2). Probabilities and close decisions can differ from the author's runtime.

- Up to 64 questions, with 2–160 options or levels each.
- The complete per-question text sequence is limited to 2,048 tokens, including instructions and options. If it exceeds the limit, tokens are removed from the start of the state/instruction prefix; option lines and the answer cue are kept. v4.0-VL image requests have a 3,072-token sequence limit.
- JSON states and text-only `state.messages` are sent as compact JSON.
- v4.0-VL accepts up to 4 PNG/JPEG images as `image_url` parts in `state.messages`. Text parts and images are joined in order with nothing between them, so start the text after an image with a newline: `[image, "\nWhat is shown?"]`. The images share a budget of 1,024 image tokens. A request whose state would cut into an image is rejected.

Questions sharing a state reuse complete prefix-cache blocks. The remaining tail is recomputed per question, so long-state, multi-question requests can be slower than the author's server. Set `VLLM_JEV_RSIJEV_PREFIX_CACHE=0` to disable prefix reuse.

Shared-state tokenization is **disabled by default**. To enable it for repeated text or image state across multiple questions:

```bash
VLLM_JEV_RSIJEV_FAST_ENCODE=1 \
  vllm-jev serve shgao/rsi-jev-v4.0-vl-qwen3.5-2b
```

This follows Shanghua Gao's encoding strategy: tokenize the shared state once to reduce CPU preparation time. Requests with unsupported tokenizers or truncated prefixes use the regular path. GPU prefix-cache behavior is unchanged.

### Clef (experimental)

| Model | Input | Platform | Start server |
|---|---|---|---|
| [Cloudflare/clef-flash](https://huggingface.co/Cloudflare/clef-flash) (9B) | Text, images, video | Linux | `vllm-jev serve Cloudflare/clef-flash` |
| [Cloudflare/clef](https://huggingface.co/Cloudflare/clef) (27B) | Text, images, video | Linux | `vllm-jev serve Cloudflare/clef` |

Clef answers all questions about a state in one forward pass, returning a probability for each option. Text, media, and questions share one input sequence.

The adapter was contributed by [Arcobalneo in PR #4](https://github.com/mode-io/vllm-jev/pull/4). It supports Cloudflare's original bf16 checkpoints on Linux. Quantized GGUF and MLX variants are not supported. Probabilities and close decisions can differ from the author's runtime.

- Up to 64 questions per request, with up to 255 options for Choice, up to 10 levels for Score, and at most 2,048 options across the entire request (Noul counts as two).
- Question IDs and serialized question definitions together must fit within 1 MiB of UTF-8 text.
- The default 16,384-token context includes the state, media, schema, and prompt formatting. State tokens that do not fit are removed from its end; the schema is kept intact. A schema and media that exceed the context limit are rejected.
- Images: up to 8 PNG or JPEG data URLs, 8 MiB each. Pass them in the `images` field as `data:image/png;base64,...` or `data:image/jpeg;base64,...` strings.
- Video: one MP4 data URL per request in the `videos` field, as `data:video/mp4;base64,...`, up to 16 MiB, 30 seconds, 60 fps, and 1080p. By default 8 frames are sampled; pass `{"url": "data:video/mp4;base64,...", "num_frames": N}` (N even, 2–16) to control sampling. Images and one video may be combined within the same request.
- Prefix caching is disabled; every request computes the full sequence so the head can read every token's state.

To prepare a checkpoint from a local release instead of downloading it:

```bash
python -m vllm_jev.clef_export --source /path/to/release \
  --output /path/to/checkpoint --model-id Cloudflare/clef
vllm-jev serve /path/to/checkpoint
```

Use `Cloudflare/clef-flash` for the 9B release.

## Serving options

Choose a GPU or pass regular vLLM options after the model ID:

```bash
CUDA_VISIBLE_DEVICES=0 vllm-jev serve ZefanCai/Open-Jev-2B --port 9000
```

On Linux, the default port is 8795 and the GPU memory budget is 90%. The default input limit is 4,096 tokens, 8,192 for Valen, 3,072 for RSI-Jev v4.0-VL, 2,048 for RSI-Jev v3.0, 16,384 for Clef, 512 for Laya English, and 1,024 for Laya multilingual and typed decisions. Run `vllm-jev serve --help=all` to see additional vLLM options.

On macOS, `vllm-jev serve` supports `--host` and `--port`. Open-Jev-2B, OpenJev-0.6B, Tiny-Jev, and Valen use MLX; Laya uses the published PyTorch MPS runtime. Mac scores and speed may differ from Linux CUDA results.

Cancelling a Mac request stops its HTTP wait. A Laya MPS inference already in progress may finish before the next queued request runs.

Set `HF_HOME` for the Hugging Face base-model cache. Set `VLLM_JEV_HOME` for the downloaded Jev files and prepared model; its default is `.local/` in your current directory.

### Optional acceleration

On Mac, Open-Jev-2B batches short candidate branches. Both Open-Jev-2B and OpenJev-0.6B reuse shared prefixes when beneficial. Send `"use_prefix_cache": false` to the Choice endpoint to disable prefix reuse.

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

Noul and Score are question types on `/v1/systemone`. Question, option, and context budgets depend on the model; consult its [supported-model section](#supported-models). The separate batch Choice route allows up to 64 questions and 256 candidates total.

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

Start Valen, either vjev model, or RSI-Jev v4.0-VL from the tables above. Put image data URLs in `state.messages`; use a `state` string for text-only requests. RSI-Jev v4.0-VL accepts at most four images and requires its own [context limits](#rsi-jev-experimental). For a local PNG:

```python
import base64
import json
import urllib.request

image = base64.b64encode(open("image.png", "rb").read()).decode()
request = {
    "state": {"messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image}"}},
        {"type": "text", "text": "\nLook at this image."},
    ]}]},
    "questions": {"color": {"type": "choice", "instructions": "What color is the object?", "criteria": {"red": "Red", "blue": "Blue"}}},
}
body = json.dumps(request).encode()
url = "http://127.0.0.1:8795/v1/systemone"
response = urllib.request.urlopen(urllib.request.Request(url, body, {"Content-Type": "application/json"}))
print(json.load(response)["answers"]["color"])
```

`answers.color` contains `choice`, `confidence`, and `probabilities`. The same image request can include Noul and Score questions.

For either Clef checkpoint, start `vllm-jev serve Cloudflare/clef` or `vllm-jev serve Cloudflare/clef-flash` on Linux and instead put the image data URL in the top-level `images` list. In the example above, replace the state and add the media before sending:

```python
request["state"] = "Look at this image."
request["images"] = [f"data:image/png;base64,{image}"]
```

### Video input

Valen (Linux and Mac) and Clef (Linux only) accept video through `/v1/systemone` with Choice, Noul, and Score questions. Their media fields differ.

For Valen, start `vllm-jev serve Valen-Team/Valen-Preview-0923` on Linux or Mac.

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

Use one MP4, up to 16 MiB, 30 seconds, 60 fps, and 1080p. `num_frames` defaults to 8 and accepts even values from 2 to 16. Frames are sampled across the clip. Variable-frame-rate clips use an evenly spaced observation timeline. More frames use more context and memory. Valen video requests can include text, but cannot mix images and video.

For either Clef checkpoint, start `vllm-jev serve Cloudflare/clef` or `vllm-jev serve Cloudflare/clef-flash` on Linux. Use the top-level `videos` list instead of a `state.messages` video part:

```python
video = base64.b64encode(open("clip.mp4", "rb").read()).decode()
request["state"] = "Watch the entire video."
request["videos"] = [{
    "url": f"data:video/mp4;base64,{video}", "num_frames": 8,
}]
```

Clef uses the same clip and frame-count limits and may combine `images` and `videos` in one request, within its context limit. Audio input is not supported. Valen video input is experimental; its preview checkpoint was trained for image-based Sokoban.

## Updates

### 2026-10-04 · v0.3.1

- Fixed RSI-Jev requests at the context limit, cached model selection, and invalid media handling.
- Fixed the Linux helper installer and simplified the setup and model guides.

### 2026-10-04 · v0.3.0

- Added [Clef 27B](#clef-experimental) alongside Clef-Flash 9B, with text, image, and video decisions on Linux.

### 2026-10-04 · v0.2.0

- Added experimental Linux serving for [RSI-Jev](#rsi-jev-experimental) v3.0/v4.0-VL by [Shanghua Gao (PR #2)](https://github.com/mode-io/vllm-jev/pull/2) and [Clef-Flash](#clef-experimental) by [Arcobalneo (PR #4)](https://github.com/mode-io/vllm-jev/pull/4).
- Added aggregate request limits for Clef and fixed platform and reference-test issues.
- Added optional [RSI-Jev shared-state tokenization](#rsi-jev-experimental) to reduce preparation time for multi-question requests.

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
