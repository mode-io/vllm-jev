# User guide

Serve Jev models with vLLM. Send text, images, or supported videos with questions; receive labels, probabilities, yes/no answers, or scores.

## Installation

On Linux with an NVIDIA GPU, [install uv](https://docs.astral.sh/uv/getting-started/installation/). From a clone of this repository:

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install '.[cuda]'
```

For development, use `uv pip install -e '.[cuda]'` to install from your working tree.

On Apple Silicon with macOS 15 or newer, run the setup script from the repository directory. Sourcing it leaves the Python 3.12 environment active:

```bash
source scripts/install_mac.sh
vllm-jev serve Valen-Team/Valen-Preview-0923
```

Valen accepts text, images, and short videos through `/v1/systemone`. For text-only Choice and batch requests, start OpenJev-0.6B or Tiny-Jev. Laya handles Choice, Noul, and Score through the same `/v1/systemone` route on Mac.

OpenJev-0.6B allows 768 tokens for the shared state prefix and 192 tokens for each question-and-candidate branch, including prompt formatting.

### Windows CPU (experimental)

The Windows adaptation closes at **24 validated text/JSON checkpoints** on 2026-10-16: nine in `windows-legacy`, thirteen in `windows-modern`, and one each in `windows-bruv` and `windows-rsi`. The [onboarding guide](project-onboarding.md) provides the complete model-to-profile map and the main development entry points. Acceptance means real released-weight inference through this project's unified CPU entrypoint; it is not a claim of published-package availability or testing on another operating system.

Use the per-model sections below for preparation, source pins, request limits and independent validation. The [current regression scope](#current-windows-regression-scope) separates the final two-model acceptance from regression of the previous 22 models. The [Tiny 1.7B / JevK5 9B guide](windows-tiny17-jev9.md) and [Decision / Bruv / RSI guide](windows-next-four.md) contain their additional environment and native-runtime requirements.

Laya uses its published PyTorch runtime in eager FP32 mode. Tiny-Jev and OpenJev-0.6B use Qwen3 backbones and scalar heads in FP32. Open-Jev-2B uses a Qwen3.5 text backbone in BF16 with an FP32 head. Both OpenJev adapters keep their LoRA adapters unmerged and evaluate complete candidates without caching; Open-Jev-2B uses a candidate batch size of one. Decider-0.8B/2B use complete BF16 Qwen3.5 text checkpoints and BF16 answer-label readouts, one prompt row at a time; Decider-4B uses its separately validated Q4_K_M artifact. Windows CPU serving accepts text and JSON; images/video, GPU acceleration, RadixFlow, and `cache_salt` are not supported. The JevK5, Kev and This-That 1.0/1.1/1.2 CPU adapters are described below; Open-Jev-9B remains outside this Windows scope.

Use separate Python 3.12 environments for the dependency profiles. From this repository in PowerShell, install CPU PyTorch before installing the local checkout. For the eight legacy models listed in the launch examples below:

```powershell
uv venv --python 3.12 .venv-windows-legacy
uv pip install --python .venv-windows-legacy\Scripts\python.exe torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv-windows-legacy\Scripts\python.exe -e ".[windows-legacy]"
```

For JevK5-2B, JevK5-Lite, JevK5 4B Q5_K_M, Kev-0.6B/0.8B, This-That 1.0/1.1/1.2, or Mica Q5_K_M, create another environment:

```powershell
uv venv --python 3.12 .venv-windows-modern
uv pip install --python .venv-windows-modern\Scripts\python.exe torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv-windows-modern\Scripts\python.exe -e ".[windows-modern]"
```

| Dependency profile | CPU PyTorch | Transformers | PEFT | Accelerate | Intended Windows models |
|---|---|---|---|---|---|
| `windows-legacy` | `2.8.0` | `5.10.2` | `0.19.1` | `1.13.0` | Laya (three releases), Tiny-Jev 0.6B/1.7B, OpenJev-0.6B, Open-Jev-2B, Decider-0.8B/2B; Tiny 1.7B retains its separate environment |
| `windows-modern` | `2.8.0` | `5.17.0` | `0.21.0` | `1.15.0` | JevK5-2B/Lite/4B Q5_K_M/9B Q5_K_M, Kev-0.6B/0.8B, This-That 1.0/1.1/1.2, Mica Q5_K_M, Decider-4B Q4_K_M and Decision-0.8B Q8_0/4B Q4_K_M; Decider Q4 and Decision require the fixed official CPU wheel |
| `windows-bruv` | `2.9.1` | `5.17.0` | `0.21.0` | `1.15.0` | Bruv1-0.8B HF, explicit FP32 |
| `windows-rsi` | `2.13.0` | `5.17.0` | `0.21.0` | `1.15.0` | RSI v1.0-0.8B, explicit FP32 |

The retained legacy and modern profiles install `torch==2.8.0`, `numpy==2.5.4` and `laya==0.3.20`; neither installs vLLM or MLX. The modern extra also pins `tokenizers==0.23.3` and `jinja2==3.1.6`; Mica checks these versions and uses them without importing Torch or Transformers. Earlier models retain their existing dependency guards. Choose exactly one extra per environment, install the CPU PyTorch wheel first, and keep existing environments intact when trying new models. The executable has the same name and request API in every profile. Use the executable inside the environment for the model you want to run. These instructions install this local checkout and do not imply a new PyPI release; existing Laya or Transformers 4.57.6 reference environments should also remain separate. The JevK5/Kev instructions below use a fresh `.venv-windows-next` environment with the same modern pins; Decider Q4 uses a separate environment with an additional fixed official CPU wheel. See the [Decision, Bruv and RSI guide](windows-next-four.md) for their separate installation commands, exact artifacts and acceptance status. Bruv and RSI require their own PyTorch versions; do not install their extras into a retained environment.

For Tiny-Jev-1.7B and JevK5-9B, follow their [separate environment commands](windows-tiny17-jev9.md#isolated-installations) instead of modifying a retained environment. Tiny 1.7B uses 10 GiB available RAM and 20 GiB commit headroom as acceptance launch guards; JevK5-9B uses 10 GiB and 12 GiB respectively. Both retain a 1.5 GiB available-RAM watchdog. These are conservative validation budgets, not minimum hardware requirements.

Start one model at a time; choose one of these legacy commands:

```powershell
.\.venv-windows-legacy\Scripts\vllm-jev.exe serve convaiinnovations/laya --cpu-threads 4
.\.venv-windows-legacy\Scripts\vllm-jev.exe serve convaiinnovations/laya-multilingual --cpu-threads 4
.\.venv-windows-legacy\Scripts\vllm-jev.exe serve convaiinnovations/laya-typed-decisions --cpu-threads 4
.\.venv-windows-legacy\Scripts\vllm-jev.exe serve lostargon/Tiny-Jev --cpu-threads 4
.\.venv-windows-legacy\Scripts\vllm-jev.exe serve IamBusy/OpenJev-0.6B --cpu-threads 4
.\.venv-windows-legacy\Scripts\vllm-jev.exe serve ZefanCai/Open-Jev-2B --cpu-threads 4
.\.venv-windows-legacy\Scripts\vllm-jev.exe serve Mapika/decider-0.8b --cpu-threads 4
.\.venv-windows-legacy\Scripts\vllm-jev.exe serve Mapika/decider-2b --cpu-threads 4
```

For a modern adapter, use the same command from its separate environment, selecting one model:

```powershell
.\.venv-windows-modern\Scripts\vllm-jev.exe serve alibiserikbay/JevK5-2B --cpu-threads 4
.\.venv-windows-modern\Scripts\vllm-jev.exe serve alibiserikbay/JevK5-Lite --cpu-threads 4
.\.venv-windows-modern\Scripts\vllm-jev.exe serve alibiserikbay/JevK5 --cpu-threads 4
.\.venv-windows-modern\Scripts\vllm-jev.exe serve jaredpalmer/kev-0.8b --cpu-threads 4
.\.venv-windows-modern\Scripts\vllm-jev.exe serve jaredpalmer/kev-0.6b --cpu-threads 4
.\.venv-windows-modern\Scripts\vllm-jev.exe serve flock-io/this-that-model-1.2 --cpu-threads 4
.\.venv-windows-modern\Scripts\vllm-jev.exe serve flock-io/this-that-model-1.1 --cpu-threads 4
.\.venv-windows-modern\Scripts\vllm-jev.exe serve flock-io/this-that-model-1.0 --cpu-threads 4
.\.venv-windows-modern\Scripts\vllm-jev.exe serve sky7350/Mica-v0.1-4B --cpu-threads 4
```

Windows selects the CPU backend automatically. The first run downloads and verifies the pinned original release; later runs reuse it. An existing local Laya model must contain `encoder/`, `tokenizer/`, `rl_agent_config.json`, and `model.safetensors`. Tiny-Jev requires its original `config.json`, `model.safetensors`, `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`, and `modeling_tiny_jev.py`, all matching the pinned hashes. Linux-exported vLLM checkpoints are not accepted. For example:

```powershell
.\.venv-windows-legacy\Scripts\vllm-jev.exe serve .\models\laya --cpu-threads 4
```

For OpenJev-0.6B, the checkpoint root contains the complete pinned Hugging Face bundle: `MANIFEST.json`, `openjev_config.json`, `adapter_config.json`, `adapter_model.safetensors`, `head.safetensors`, `calibration-v03.json`, and the release's example, README, results, license, and third-party notice files. Its `base/` directory contains the separately pinned Qwen3-0.6B `model.safetensors`, `config.json`, `generation_config.json`, `tokenizer.json`, `tokenizer_config.json`, `vocab.json`, and `merges.txt`. The automatic download builds this layout; `serve .\models\openjev` accepts the same complete layout locally. All required files are hash-checked, and unrecognized model or tokenizer overrides in `base/` are rejected.

For Open-Jev-2B, the source root contains the original five files under `package/checkpoint/`, the release manifest and license/attribution files, and a separately pinned Qwen3.5-2B `base/` directory. All 18 files are checked against fixed SHA256 values. The complete base download includes a 4,548,221,488-byte weight file; serving retains only the text backbone and decision head after loading. `serve .\models\open-jev-2b` accepts the same layout. Complete tokenized prompts, including formatting and each candidate, are limited to 1,024 tokens and rejected when oversized; they are never silently truncated. The Windows 2B adapter uses BF16 and does not change precision in response to `JEV_TORCH_DTYPE`.

The default address is `http://127.0.0.1:8795`; `--host` and `--port` override it. `--cpu-threads` is optional and defaults to the smaller of four and the logical CPU count. Requests share one model worker, with at most 16 requests running or waiting. A full queue returns HTTP 503 with `Retry-After: 1`; `/health` remains responsive. Cancelled requests keep their slot until the worker consumes or finishes their work, preventing repeated cancellations from bypassing the limit. Cancelling an HTTP request does not interrupt CPU inference already in progress; shutdown waits for that work before releasing the model.

From another PowerShell window, check the backend identity and send text with all three question types:

```powershell
Invoke-RestMethod http://127.0.0.1:8795/health
$payload = @'
{
  "state": "I was charged twice. Please refund the duplicate payment.",
  "questions": {
    "department": {"type": "choice", "instructions": "Choose the responsible department.", "criteria": {"billing": "Payments and refunds", "technical": "Software issues"}},
    "refund": {"type": "noul", "instructions": "Does the customer request a refund?"},
    "urgency": {"type": "score", "instructions": "Rate the urgency.", "criteria": ["Can wait", "Soon", "Today"]}
  }
}
'@
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8795/v1/systemone -ContentType 'application/json; charset=utf-8' -Body $payload
```

`/health` identifies the backend, `device: cpu`, `experimental: true`, and the pinned model/revision. Examples include `laya-pytorch`, `tiny-jev-pytorch` for Tiny 0.6B/1.7B, `openjev-pytorch` for 0.6B, `open-jev-pytorch` for Open-Jev-2B, `decider-pytorch` for the BF16 Decider releases, `kev-pytorch` for both Kev releases, `jevk5-llama-cpp-q5-b11010` for JevK5 4B/9B Q5, and `decider-llama-cpp-q4-0.3.35` for Decider Q4. Decision responses include the same identity in `metadata`. Open-Jev-2B also reports `backbone_dtype: bfloat16`, `head_dtype: float32`, `candidate_batch_size: 1`, and `max_sequence_tokens: 1024` in decision metadata. The BF16 Decider releases report BF16 backbone/readout, FP32 logits and row batch size one. The `state` may also be a JSON object or list. Use `/v1/systemone`; the separate Choice and batch routes are not exposed by this Windows service.

The additional Windows backends report `decision-llama-cpp-0.3.35`, `bruv-pytorch` or `rsi-pytorch`. Their [model-specific limits and reference boundaries](windows-next-four.md#fixed-artifacts-and-scoring-contracts) differ from the older models below.

OpenJev-0.6B uses the following input constraints. All questions are validated before any inference:

| Field | OpenJev-0.6B constraint |
|---|---|
| `instructions` | Nonempty plain text, at most 4,096 characters; JSON instruction objects are not accepted. |
| Choice `criteria` | An object with 2–255 nonempty names and distinct, nonempty string descriptions; at most 4,096 characters per description. |
| Score `criteria` | A list of 2–10 distinct, nonempty string descriptions, ordered from the lowest to highest level; at most 4,096 characters per description. |
| Noul `criteria` | Omit it, or provide exactly `false` and `true` with distinct, nonempty string descriptions; at most 4,096 characters per description. |
| Request budget | 1–64 questions, at most 256 candidates in total, and the shared service's 50,000-character state limit. |

The native OpenJev prompt permits at most 768 tokens for the state prefix and 192 for each question/candidate suffix, including formatting. Oversized inputs are rejected rather than truncated. Customized context layouts retain these checks and additionally enforce 960 tokens per complete branch. Choice returns the named candidate; Noul returns the probability of true; Score returns the expected zero-based level. Score probabilities and `legend` use string indices such as `"0"` and `"1"`. The official SDK returns its legend as a list; the HTTP API maps it to this index-keyed object. The SDK's near-tie Choice rule is preserved: candidates within `1e-7` of the highest probability are ordered by case-folded description, then original description, while confidence remains the highest probability.

#### Check a Windows installation

The boundary tests exercise request validation, CLI dispatch, queue/cancellation behavior, and source-file integrity with substitute agents. Qwen tests also use small randomly initialized models to check readouts and adapter loading. These tests do not download released weights or establish model accuracy:

```powershell
uv pip install --python .venv-windows-legacy\Scripts\python.exe pytest httpx psutil
.\.venv-windows-legacy\Scripts\python.exe -m pytest tests\test_windows_laya.py tests\test_windows_server.py tests\test_windows_qwen.py tests\test_windows_openjev.py tests\test_windows_open_jev.py tests\test_windows_open_jev_driver.py tests\test_windows_open_jev_integration.py tests\test_windows_decider.py tests\test_windows_decider_driver.py tests\test_windows_decider_integration.py tests\test_windows_decider_reference.py tests\test_decision_template.py -q
```

On 2026-10-14, the three Laya checkpoints, Tiny-Jev, and OpenJev-0.6B passed a separate model-ID regression run on Windows 11 x64 with Python 3.12.5, PyTorch 2.8.0+cpu, Transformers 5.10.2, PEFT 0.19.1, and Laya 0.3.20. Each driver checked two fresh CLI starts and preserved the original source hashes. The Laya and Tiny-Jev HTTP answers matched the prior product environment exactly; OpenJev-0.6B's largest absolute difference was `4.69e-6`, within `1e-5`. These five-model results do not establish Open-Jev-2B validation, task accuracy, or performance. The earlier per-model checks below retain their original environment and scope.

For real Laya inference, run its validation driver separately for each of the three Laya model IDs:

```powershell
.\.venv-windows-legacy\Scripts\python.exe scripts\check_windows_laya.py --model convaiinnovations/laya --workspace .\windows-check --output .\windows-check\laya.json
```

The driver downloads the pinned files when needed. Add `--source .\models\laya` to use an existing original checkpoint. It runs the published SDK and the HTTP service in separate processes, compares Choice/Noul/Score outputs on a small fixed text/JSON workload, and records package versions, source hashes, and service checks in JSON. It checks consistency with the SDK, not decision accuracy, Linux/CUDA equivalence, or a throughput claim. It stops only the server process it starts; its process cleanup does not establish graceful shutdown behavior.

All three Laya checkpoints passed these checks on Windows 11 x64 with Python 3.12.5, PyTorch 2.6.0+cpu, Transformers 4.48.2, and Laya 0.3.20 on 2026-10-14. They subsequently passed again in a separate Transformers 4.57.6 environment, with two fresh CLI starts per model and answers matching the earlier SDK baseline. The checks include unsupported-input rejection, concurrent requests, and unchanged source hashes. Original SDK calibration warnings remain visible; matching its outputs does not establish calibrated confidence for other candidate counts.

#### Validate JevK5, Kev and This-That on Windows

Five additional pinned text checkpoints share the existing Windows CLI and HTTP lifecycle. Each checkpoint has a distinct pinned source identity; reference runners execute the original author implementations independently of the product adapters:

| Model | Release revision | CPU calculation | Complete input limit |
|---|---|---|---|
| `alibiserikbay/JevK5-2B` | `7922d1f55df137b72ef763fced56fd09efc5e99d` | BF16 backbone and all 16 answer-letter rows; FP32 NumPy softmax, temperature 1.42; no padding | 1,024 tokens per question, Choice 2–16, Score 2–10 |
| `jaredpalmer/kev-0.8b` | `bf75a6a8848ea6960ff2ed108d9ed44c2941174f` | FP32 text backbone, merged LoRA and pointer head; eager attention, one question per forward | 1,024 tokens per question, Choice/Score 1–255 options |
| `flock-io/this-that-model-1.2` | `c4d1c30b8d512d278726de439b8cc81fccc70c8f` | FP32 backbone and vocabulary projection; packed questions, padding to 64, NumPy softmax | 1,024 tokens for the complete packed request, including padding |
| `flock-io/this-that-model-1.1` | `fc083c82bd5872abfeaf0d44cf7482324efce87d` | FP32 backbone and vocabulary projection; packed questions, padding to 64, NumPy softmax | 1,024 tokens for the complete packed request, including padding |
| `flock-io/this-that-model-1.0` | `3d927195c4f9845efe66c5715883a7a0f42b1239` | FP32 backbone and vocabulary projection; packed questions, padding to 64, NumPy softmax | 1,024 tokens for the complete packed request, including padding |

The shared text-only, 64-question and 256-candidate limits apply. Oversized complete prompts are rejected before inference. JevK5's Noul readout is true-first; the others are false-first. Score remains the project's probability-weighted expected level and confidence formula. JevK5's Score legend retains its existing `index: description` presentation. Kev creates a fresh model cache for each forward to match the original CPU calculation; no cache survives between questions or requests. These paths do not enable the authors' optional CUDA graphs, fused kernels, tree search, or prefix caching.

This-That 1.2 passed released-weight Windows CPU validation on 2026-10-15 using the modern environment, Windows 11, Python 3.12.5, Core Ultra 5 125H and four CPU threads. An independently executed original-author reference and two fresh CLI starts (model ID and a local path containing Chinese characters and spaces) covered English and Chinese JSON requests with all three question types, a wider Choice, and concurrent requests. All ten product responses matched the original FP32 probabilities with zero observed difference at a fixed `1e-5` tolerance. Fourteen invalid requests returned HTTP 422, health stayed responsive while inference was pending, source and SDK hashes stayed unchanged, and owned processes and private workspaces were cleaned up. A separate 136-test lightweight suite passed. The ten previously accepted models were not rerun for this increment.

The six sequential This-That HTTP requests took 5.74–6.60 seconds, with a median of 6.24 seconds, excluding model loading. The largest observed product-process peak working set was 7.53 GiB and peak commit was 9.95 GiB. These are observations from fixed functional cases while other applications were active, not a throughput benchmark or minimum-memory requirement. FP32 parameters alone occupy approximately 7.01 GiB, before runtime and loading overhead.

This acceptance run required at least 9 GiB of available physical memory and 12 GiB of remaining system commit before each model process started. Its watchdog threshold was 1.5 GiB of available physical memory; crossing it would stop only processes owned by the validation driver. These experiment guards are not guarantees of sufficient RAM for arbitrary inputs. The complete packed request, including padding to a multiple of 64, must fit within 1,024 tokens.

The subsequent release-family increment validated This-That 1.1 and 1.0 in that order on the same host and unchanged modern dependencies. Each release used its own pinned six-file source, a newly executed original-author FP32 reference, and two fresh installed `vllm-jev.exe` starts (model ID and Unicode local path, with no `PYTHONPATH` override). Each passed ten valid requests and fourteen HTTP 422 checks with zero observed probability difference at the unchanged `1e-5` tolerance. Health stayed responsive during concurrent inference, returned model/revision identities matched the selected release, sources and SDK files stayed unchanged, and owned processes/workspaces were cleaned up. This-That 1.2 passed the same fresh reference and installed-entrypoint regression after the shared adapter was extended.

The legacy Windows/template suite passed 661 tests; 329 modern-adapter/shared-boundary tests passed in the modern environment. An exploratory full-suite run in the modern environment hit fourteen legacy-model dependency guards; those tests passed in the required legacy environment. Neither the guards nor dependencies were relaxed. The other ten previously accepted models were not rerun with real weights in this increment; their inference implementations and the legacy environment were unchanged.

| This-That release | Six sequential request times | Median | Observed peak product working set | Observed peak product commit |
|---|---|---|---|---|
| 1.1 | 2.45–2.93 s | 2.74 s | 7.54 GiB | 9.95 GiB |
| 1.0 | 2.01–2.36 s | 2.14 s | 7.53 GiB | 9.94 GiB |
| 1.2 regression | 5.80–6.45 s | 6.17 s | 7.53 GiB | 9.96 GiB |

These are fixed-case functional observations with other applications active, not a speed comparison between releases or a minimum-memory promise. All three use the same FP32, packed-question, 255-label-head calculation and 1,024-token padded-request limit. The 1.0 tokenizer configuration is pinned separately; sharing the backbone configuration does not identify a release.

JevK5-2B and Kev-0.8B each passed two fresh CLI starts (model ID and a local path with Chinese characters and spaces) on Windows 11, Python 3.12.5, Core Ultra 5 125H and four CPU threads on 2026-10-15. Each start checked three synthetic English/Chinese JSON/wide-Choice cases and two concurrent requests: all ten responses matched independently executed author probabilities after the project's response mapping, with zero observed difference and a fixed `1e-5` tolerance. Each model rejected fourteen invalid requests with HTTP 422; health checks stayed responsive while decision requests were pending. Source hashes stayed unchanged, and owned processes and private workspaces were cleaned up. These checks establish local integration and numerical parity, not decision accuracy, Linux/GPU equivalence or graceful shutdown.

JevK5's first trial under the older Transformers 5.10.2 runtime differed from the 5.17.0 author reference by approximately `0.004149`, exceeding tolerance. Its complete product run passed in the separate 5.17.0 environment using that same provenance-checked author report; the tolerance was not relaxed. Transformers changed the CPU Qwen3.5 fallback's normalization precision and triangular calculation between these versions. The experiment establishes agreement in the matched environment, without assigning the entire difference to one kernel change or upgrading earlier models in place. Kev's reference and product runs used separate environments with the same modern dependency versions.

The six sequential requests per model took 35.03–83.78 seconds for JevK5 and 0.86–1.72 seconds for Kev, excluding loading. Some requests contained three questions and the wide-Choice request contained one. Other applications were active; these timings are local functional-test observations, not a performance benchmark. JevK5's BF16 CPU path is particularly slow on this host.

The retained legacy environment then passed a focused real-weight regression of all eight earlier models: one fresh CLI start, two mixed text/JSON requests, and two HTTP 422 checks per model. These sixteen product requests reused hash-checked historical author baselines rather than rerunning the authors. The largest absolute answer difference was `4.66e-6`, within `1e-5`; checkpoint hashes were unchanged and owned processes/workspaces were cleaned up. All 52 installed dependency versions matched the pre-change environment, and the existing model implementations were unchanged. The final Windows/template suite passed 590 tests; 169 adapter/dependency tests also passed in the separate modern environment.

Use `.venv-windows-modern` for the JevK5, Kev and This-That 1.0/1.1/1.2 product runs and keep `.venv-windows-legacy` for the eight earlier models. The original-author reference uses a third Python 3.12 environment, without installing this project:

```powershell
uv venv --python 3.12 .venv-candidate-reference
uv pip install --python .venv-candidate-reference\Scripts\python.exe torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv-candidate-reference\Scripts\python.exe transformers==5.17.0 peft==0.21.0 accelerate==1.15.0 numpy==2.5.4 tokenizers==0.23.3 safetensors==0.8.0 psutil pydantic fastapi uvicorn
```

Obtain the author SDK at the following fixed commits, then copy exactly the `SDK_HASHES` files from the corresponding reference script into a separate SDK directory, preserving original LF bytes. The runner verifies those bytes before executing a private copy:

- JevK5: `allebee/jevk5@f26426d16f59e8bbe1470e5b162cc89329e29b29`, `scripts/windows_jevk5_reference.py`.
- Kev: `jaredpalmer/kev@dd375f27fd61486c52204205e10a0d2e10f3b34a`, `scripts/windows_kev_reference.py`. The runner calls the original `Checkpoint.load`, request conversion, encoder and forward code directly; the author's higher-level predictor imports Unix-only `fcntl` through a research module.
- This-That: `FLock-io/this-that-model@6af11c63c8bb260aa4bed2b7717f33f25decd16e`, `scripts/windows_thisthat_reference.py`.

Prepare model data separately using the selected adapter's `prepare` function. For example, run `.\.venv-windows-modern\Scripts\python.exe -c "from pathlib import Path; from vllm_jev.windows_jevk5 import prepare, MODEL_ID; print(prepare(MODEL_ID, Path('windows-candidate-data')))"`. JevK5 uses seven pinned files at the checkpoint root. Kev uses seven release files and six fixed `Qwen/Qwen3.5-0.8B-Base` files under `base/`. This-That uses six pinned files; select `vllm_jev.windows_thisthat` in the preparation command and pass the explicit model ID for 1.1 or 1.0 (`MODEL_ID` retains the 1.2 default). Each release has a distinct weight hash; 1.0 also has a distinct tokenizer-configuration hash. Extra loader overrides are rejected; original sources are never rewritten.

Once the source and SDK are prepared, the same acceptance driver handles each profile:

```powershell
.\.venv-windows-modern\Scripts\python.exe -B scripts\check_windows_candidates.py --profile jevk5 --source .\windows-candidate-data\checkpoint\alibiserikbay\JevK5-2B --sdk-source .\reference-source\jevk5 --reference-python .\.venv-candidate-reference\Scripts\python.exe --workspace .\windows-candidate-check --output .\windows-candidate-check\jevk5.json --threads 4

.\.venv-windows-modern\Scripts\python.exe -B scripts\check_windows_candidates.py --profile thisthat --source .\windows-candidate-data\checkpoint\flock-io\this-that-model-1.2 --sdk-source .\reference-source\thisthat --reference-python .\.venv-candidate-reference\Scripts\python.exe --workspace .\windows-candidate-check --output .\windows-candidate-check\thisthat.json --threads 4 --abort-below-available-gib 1.5 --console-entrypoint .\.venv-windows-modern\Scripts\vllm-jev.exe
```

Select `--profile kev` with its source and SDK paths for Kev. For This-That, `--profile thisthat` retains the 1.2 default; use `--profile thisthat11` or `--profile thisthat10` with the matching checkpoint path and a distinct output filename for 1.1 or 1.0. All three use `windows_thisthat_reference.py --model <full-model-id>` and the same fixed author SDK. Keep the 1.5 GiB watchdog threshold and pass `--console-entrypoint` to validate the installed command; omitting it retains the older Python-module launch. The installed-entrypoint mode requires the executable beside the selected product Python, verifies its distribution/module identity, and clears `PYTHONPATH`. `--reference-only` establishes an author baseline; `--reference-report` explicitly reuses a completed matching author JSON. The driver verifies model/revision, ordered request content, source/SDK/runner/helper hashes, reference dependencies and precision before reuse. This-That also checks the reference interpreter identity. A report from a different This-That release is rejected; the new releases require their own original-author run. It runs the reference and product models sequentially, uses model-ID and Chinese/space-containing local-path starts, checks three mixed/wide cases, queues concurrent requests, and checks invalid inputs. Memory guards stop only its own processes. This is fixed-case numerical and integration validation, not task accuracy, throughput, or graceful-shutdown validation.

#### Validate JevK5 4B Q5_K_M on Windows

The Windows command `vllm-jev serve alibiserikbay/JevK5` selects the author's official **4B Q5_K_M GGUF**. It passed real-weight acceptance on 2026-10-15. This profile is distinct from JevK5-2B's BF16 implementation and JevK5-Lite's FP32 encoder. Create a fresh product environment so existing validated environments stay unchanged:

```powershell
uv venv --python 3.12 .venv-windows-next
uv pip install --python .venv-windows-next\Scripts\python.exe torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv-windows-next\Scripts\python.exe -e ".[windows-modern]" psutil requests
.\.venv-windows-next\Scripts\vllm-jev.exe serve alibiserikbay/JevK5 --cpu-threads 4
```

Preparation downloads `alibiserikbay/JevK5-GGUF` at revision `ec67b0bfce5119a8b11a2cdb430bb43e3fa3e82a`. The selected `jevk5-4b-v0.3-Q5_K_M.gguf` is 3,074,986,400 bytes, with SHA256 `b1df9869dd14c0f5b243fa956dd985f0141c3d23ec4847da4c503c73c1050601`. Its README and SHA256SUMS are also pinned. The official [llama.cpp b11010 Windows x64 CPU archive](https://github.com/ggml-org/llama.cpp/releases/download/b11010/llama-b11010-bin-win-cpu-x64.zip) has SHA256 `3011251607251eca68ae67e35512364d9943d862ff598599ff9a4888c1753008`; each selected executable, DLL and license is verified separately. A complete prepared directory, including its fixed `runtime/` subdirectory, can be passed as a local path. A lone GGUF file, another quantization or an arbitrary runtime is not accepted.

The backend owns a private, authenticated loopback `llama-server` process with CPU-only execution, 1–4 threads, one sequence, a 2,048-token context, and no cross-request cache reuse. Its Windows job object is responsible for the child process lifetime. The public service remains `/v1/systemone`; its internal llama.cpp endpoint is not a configurable external service. The product serving process does not import Torch for this profile, and native inference runs in the separate child.

The original author's prompt and readout are preserved: tokenize with `parse_special: true` and `add_special: false`, request one generated position and its top 40 token log probabilities, take candidate letters A–P, assign an absent letter the smallest returned log probability minus 2, and normalize with temperature **1.22**. This is the published Q5 calculation; it does not establish equivalence to BF16 JevK5. Metadata identifies the artifact repository/file, quantization, runtime, missing-letter count and internally generated position count; API `output_tokens` remains zero for the decision response.

Choice accepts 2–16 distinct named options, Noul accepts optional true/false descriptions with true-first probability order, and Score accepts 2–10 ordered text levels with listwise scoring. Only the default/native template is supported. All complete question prompts must fit within **1,024 tokens**; the entire request is checked before any completion call, and oversized inputs are rejected without truncation. The common 64-question/256-candidate and finite text/JSON limits apply. Images, video, isolated Score, other templates, GPU options and `cache_salt` are unsupported. The original author's greater-than-16 knockout/tree paths are outside this bounded profile.

For independent acceptance, obtain [the original JevK5 SDK](https://github.com/allebee/jevk5/tree/f26426d16f59e8bbe1470e5b162cc89329e29b29) at the fixed commit with unchanged LF bytes. The runner privately copies and verifies `jevk5/__init__.py`, `jevk5/prompt.py`, and `jevk5/gguf.py`, then calls the unmodified `JevK5GGUF.probabilities`. Its separate reference environment needs only Python's standard library; it does not install the product or Torch:

```powershell
uv venv --python 3.12 .venv-jevk5-gguf-reference
git -c core.autocrlf=false clone https://github.com/allebee/jevk5 .\reference-source\jevk5-gguf
git -C .\reference-source\jevk5-gguf -c core.autocrlf=false checkout f26426d16f59e8bbe1470e5b162cc89329e29b29
.\.venv-windows-next\Scripts\python.exe -c "from pathlib import Path; from vllm_jev.windows_jevk5_gguf import prepare, MODEL_ID; print(prepare(MODEL_ID, Path('windows-jevk5-gguf-data')))"
.\.venv-windows-next\Scripts\python.exe -B scripts\check_windows_jevk5_gguf.py --source .\windows-jevk5-gguf-data\checkpoint\alibiserikbay\JevK5 --sdk-source .\reference-source\jevk5-gguf --reference-python .\.venv-jevk5-gguf-reference\Scripts\python.exe --workspace .\windows-jevk5-gguf-check --output .\windows-jevk5-gguf-check\acceptance.json --console-entrypoint .\.venv-windows-next\Scripts\vllm-jev.exe --threads 4
```

The completed run used Windows 11 x64, Python 3.12.5, a Core Ultra 5 125H and four CPU threads. A fresh independent author run and two fresh installed CLI starts, using the model ID and a Chinese/space-containing local path, passed **twelve product requests** and **eighteen HTTP 422 checks**. Fixed cases covered mixed Choice/Noul/Score, Chinese JSON, a 16-option boundary, duplicate Score descriptions and concurrent requests. The maximum observed answer difference was **0** at an unchanged `1e-5` tolerance. Health remained responsive; original source and SDK hashes stayed unchanged; no owned processes or private service directories remained. The eight sequential HTTP requests took **11.53–24.15 seconds**, excluding loading. The highest observed native inference-process peak working set was **3.19 GiB**; this is not the combined process-tree peak or a minimum-memory requirement.

The driver checks 6 GiB of available physical RAM and 8 GiB of remaining system commit before each model startup, and stops only its owned processes if available RAM falls below 1.5 GiB. These are experiment guards. `--reference-only` creates a baseline, while `--reference-report` reuses a matching completed author report after identity, input and hash checks. Acceptance establishes fixed-case local numerical parity, not task accuracy, BF16 quality equivalence, throughput or graceful shutdown. Cleanup exercised the validation driver's owned-process termination; it did not test every parent-process termination scenario.

#### Validate Kev-0.6B on Windows

`jaredpalmer/kev-0.6b` passed real-weight acceptance on 2026-10-15. Use the fresh `.venv-windows-next` modern-profile environment above and start one model at a time:

```powershell
.\.venv-windows-next\Scripts\vllm-jev.exe serve jaredpalmer/kev-0.6b --cpu-threads 4
```

The adapter pins release `dece6dba8d43f0f7ded45e9f5b9df12474d90843` and `Qwen/Qwen3-0.6B-Base` at `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`. Seven original adapter/head/tokenizer/provenance files and five base files are hash-verified. Unlike Kev-0.8B's Qwen3.5 hybrid backbone, this checkpoint uses the 28-layer Qwen3 attention backbone. Both retain their own release hashes, base identity, LoRA targets and calibration. The 0.6B checkpoint's original metadata omits temperature and weight-precision fields; the adapter follows the original loader's defaults, **temperature 1.0 and FP32**, only for this fixed release.

Product inference loads the complete FP32 backbone, merges its rank-16/alpha-32 LoRA and applies the FP32 pointer head with eager attention, one question per forward. Choice and Score accept 1–255 options subject to the common 256-candidate request budget; Noul uses false/true ordering and optional descriptions. Every complete question prompt must fit within **1,024 tokens** and all questions are compiled before inference. Score is joint/listwise; the `isolated` field is rejected. A fresh model cache is created for each forward, with no question- or request-level cache reuse. The common text/JSON-only service limits apply.

The independent reference is [jaredpalmer/kev at `dd375f27fd61486c52204205e10a0d2e10f3b34a`](https://github.com/jaredpalmer/kev/tree/dd375f27fd61486c52204205e10a0d2e10f3b34a). `scripts/windows_kev06_reference.py` verifies the original SDK files and runs `Checkpoint.load`, `to_record`, `encode` and `forward` directly in a private process. It observes the author's original packed-question block-causal mask and position IDs; it does not replace that computation with the product's separate question rows. The resulting raw probabilities are mapped independently to project response conventions. Keep the reference separate from the product and earlier model environments:

```powershell
uv venv --python 3.12 .venv-kev06-reference
uv pip install --python .venv-kev06-reference\Scripts\python.exe torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv-kev06-reference\Scripts\python.exe transformers==5.17.0 peft==0.21.0 accelerate==1.15.0 numpy==2.5.4 tokenizers==0.23.3 huggingface_hub==1.33.0 safetensors==0.8.0 psutil pydantic fastapi uvicorn
git -c core.autocrlf=false clone https://github.com/jaredpalmer/kev .\reference-source\kev06
git -C .\reference-source\kev06 -c core.autocrlf=false checkout dd375f27fd61486c52204205e10a0d2e10f3b34a
.\.venv-windows-next\Scripts\python.exe -c "from pathlib import Path; from vllm_jev.windows_kev import prepare; print(prepare('jaredpalmer/kev-0.6b', Path('windows-kev06-data')))"
.\.venv-windows-next\Scripts\python.exe -B scripts\check_windows_candidates.py --profile kev06 --source .\windows-kev06-data\checkpoint\jaredpalmer\kev-0.6b --sdk-source .\reference-source\kev06 --reference-python .\.venv-kev06-reference\Scripts\python.exe --workspace .\windows-kev06-check --output .\windows-kev06-check\acceptance.json --console-entrypoint .\.venv-windows-next\Scripts\vllm-jev.exe --threads 4 --abort-below-available-gib 1.5
```

On the same Windows host with four CPU threads, a fresh author reference and two fresh installed CLI starts passed **ten product requests** and **fourteen HTTP 422 checks**. Model-ID and Unicode local-path routes covered English/Chinese JSON mixed questions, a wider Choice, repeated concurrent requests and responsive health checks. The largest absolute answer difference was approximately **`1.74e-6`**, within the fixed `1e-5` tolerance; source and SDK hashes remained unchanged and owned processes/private workspaces were removed. The six sequential HTTP requests took **0.82–2.10 seconds**, excluding loading. These results establish local fixed-case integration and numerical parity, not task accuracy or a performance guarantee. The driver's 5 GiB available-RAM/7 GiB commit-headroom startup checks and 1.5 GiB watchdog are experiment guards; cleanup is not a graceful-shutdown test.

#### Validate Decider-4B Q4_K_M on Windows

`Mapika/decider-4b-GGUF` passed real-weight Windows CPU acceptance on 2026-10-15, using protocol `decider_gguf_v1` and backend `decider-llama-cpp-q4-0.3.35`. The selected author artifact is `decider-4b-v2.1-Q4_K_M.gguf` from revision `b79f09d9ba7837f1b744295ea267b55d08e958ec`, with 2,708,804,640 bytes and SHA256 `c7083fcfc93f650cd66caeade4a3840ef9eeeeb1e80920907c554f70619d7c56`. Preparation verifies the six original GGUF/tokenizer/template/calibration/README files. A complete prepared directory can also be passed to `serve`; a lone GGUF, another quantization or extra loader overrides are rejected.

This bounded profile uses 1–4 CPU threads, no GPU layers, one prompt row at a time, a 2,048-token runtime context and at most **1,024 tokens per complete prompt**. Every question is compiled before inference; oversized inputs are rejected without truncation. Choice supports 2–255 options, Noul uses false/true order and optional descriptions, and Score supports 2–10 ordered levels. Score evaluates levels independently by default, using each binary yes-probability and normalizing across levels; explicit `isolated: false` selects joint scoring. Answer-label logits and softmax use FP32, with temperatures **Choice 1.110, Noul 1.560 and Score 1.287**. Isolated Score rows retain the Score temperature. The model does not generate answer tokens or retain a cache across requests.

Only the default/native template is supported. The common finite text/JSON, 64-question, 256-total-candidate and 50,000-character state limits apply; option limits do not guarantee a prompt fits within 1,024 tokens. Images, video, GPU options, custom templates and `cache_salt` are unsupported. This profile is separate from BF16 Decider-0.8B/2B and does not establish Q4-to-BF16 quality equivalence.

Use a **new** modern-profile environment plus the official `llama-cpp-python==0.3.35` Windows x64 CPU wheel; no new project extra is needed. The fixed wheel is [published in the upstream v0.3.35 release](https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.35/llama_cpp_python-0.3.35-py3-none-win_amd64.whl), with SHA256 `31590ea000d5aff6f05f1e428048e72318a83709288159a5bd4dabec530080bb`:

```powershell
uv venv --python 3.12 .venv-windows-decider4
uv pip install --python .venv-windows-decider4\Scripts\python.exe torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv-windows-decider4\Scripts\python.exe -e ".[windows-modern]" psutil requests
New-Item -ItemType Directory -Force .\windows-decider4-packages | Out-Null
$deciderWheel = Join-Path $PWD 'windows-decider4-packages\llama_cpp_python-0.3.35-py3-none-win_amd64.whl'
Invoke-WebRequest 'https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.35/llama_cpp_python-0.3.35-py3-none-win_amd64.whl' -OutFile $deciderWheel
if ((Get-FileHash -LiteralPath $deciderWheel -Algorithm SHA256).Hash.ToLowerInvariant() -ne '31590ea000d5aff6f05f1e428048e72318a83709288159a5bd4dabec530080bb') { throw 'Decider CPU wheel checksum mismatch' }
uv pip install --python .venv-windows-decider4\Scripts\python.exe $deciderWheel
.\.venv-windows-decider4\Scripts\vllm-jev.exe serve Mapika/decider-4b-GGUF --cpu-threads 4
```

The CPU library uses Windows VCOMP140 and worked alongside Torch 2.8 CPU in the tested environment without `KMP_DUPLICATE_LIB_OK`. Serving verifies the fixed binding/DLL hashes before import and rejects `LLAMA_CPP_LIB_PATH` overrides and unexpected backend DLLs. Other model environments do not need this wheel.

For an independent baseline, obtain the original [Decider SDK at `d6fc522aed7fc392576a9b66a330409bc6c382bd`](https://github.com/Mapika/decider/tree/d6fc522aed7fc392576a9b66a330409bc6c382bd). The reference runner privately copies seven verified original files, blocks product imports and calls the unmodified author `Decider`/`system_one` GGUF path. It observes original return frames without replacing author functions. The acceptance driver independently recomputes each masked, typed-temperature FP32 distribution from all 255 recorded logits, checks the author's rounded wire response, and then compares unrounded probabilities and project response conventions. The same fixed Q4 artifact and official CPU wheel are used on both sides.

Keep reference dependencies separate from the product and earlier model environments. After the installation above, stop the manually started service before running acceptance:

```powershell
uv venv --python 3.12 .venv-decider4-reference
uv pip install --python .venv-decider4-reference\Scripts\python.exe torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv-decider4-reference\Scripts\python.exe transformers==5.17.0 numpy==2.5.4 tokenizers==0.23.3 huggingface_hub==1.33.0 safetensors==0.8.0 jinja2==3.1.6 psutil==7.2.2 diskcache==5.6.3 $deciderWheel
git -c core.autocrlf=false clone https://github.com/Mapika/decider .\reference-source\decider4
git -C .\reference-source\decider4 -c core.autocrlf=false checkout d6fc522aed7fc392576a9b66a330409bc6c382bd
.\.venv-windows-decider4\Scripts\python.exe -c "from pathlib import Path; from vllm_jev.windows_decider_gguf import prepare, MODEL_ID; print(prepare(MODEL_ID, Path('windows-decider4-data')))"
.\.venv-windows-decider4\Scripts\python.exe -B scripts\check_windows_decider_gguf.py --source .\windows-decider4-data\checkpoint\Mapika\decider-4b-GGUF --sdk-source .\reference-source\decider4 --reference-python .\.venv-decider4-reference\Scripts\python.exe --llama-wheel $deciderWheel --workspace .\windows-decider4-check --output .\windows-decider4-check\acceptance.json --console-entrypoint .\.venv-windows-decider4\Scripts\vllm-jev.exe --threads 4
```

The completed run used Windows 11 x64, Python 3.12.5, a Core Ultra 5 125H and four CPU threads. A fresh original-author run covered five fixed cases: mixed questions, Chinese JSON, wider Choice, explicit joint Score and indexed JSON state. Two fresh installed CLI starts, via model ID and a local path containing Chinese characters and spaces, passed **fourteen real HTTP requests** including concurrent requests, plus **twenty-eight HTTP 422 checks**. The maximum observed answer difference was **0** at the unchanged `1e-5` tolerance; independent raw-logit calibration checks used `1e-6`. Health remained responsive. Checkpoint and SDK hashes stayed unchanged; the author's runtime bytes were unchanged. A separate product-runtime postcheck verified the entire installed package against the fixed official wheel and its initial hashes. No owned processes or private workspaces remained.

The ten sequential requests took **1.43–31.04 seconds**, excluding loading; indexed JSON was the longest case. The highest observed product-process peak working set was **4.82 GiB** and peak commit was **6.35 GiB**. The lowest sampled available system RAM during product execution was **5.23 GiB**, with other applications active. These are fixed-case observations, not minimum-memory requirements, an accuracy assessment or a throughput benchmark. Startup requires 6 GiB available RAM and 8 GiB commit headroom; the watchdog stops only owned processes below 1.5 GiB available RAM. The driver's default port is 18829. `--reference-only` creates an author baseline; `--reference-report` reuses a matching baseline after provenance and numerical checks. Cleanup exercises driver-owned process termination and does not establish graceful shutdown.

#### Current Windows regression scope

The Tiny-Jev-1.7B/JevK5-9B increment brings Windows acceptance to **24 models**. Its two new real-weight validations and **4 installed CLI starts** covered **28 successful product requests and 60 invalid-input rejections**. Tiny's fresh unchanged-author reference was provenance-checked and explicitly reused for the final product rerun; JevK5-9B ran its original SDK freshly in the final acceptance invocation. Maximum absolute answer differences were `7.753524231546294e-7` and `0`, respectively, at the unchanged `1e-5` tolerance. See the [individual results, memory measurements and retained attempts](windows-tiny17-jev9.md#acceptance-results).

All **22 earlier models** passed fresh real product inference in retained environments: **36 CLI starts, 184 successful requests and 304 HTTP 422 rejections**. Twenty-one reused verified historical reference reports. JevK5 4B ran a fresh original-author SDK reference because its reference runner changed; an old report was not accepted in place of that run. The eight legacy models each used one module CLI start and two real requests; the other fourteen models each used both installed CLI routes. Legacy regressions used fixed-hash historical author/product reports, pinned sources, request schemas and retained runtime versions; older Laya/Tiny reports do not contain every provenance field present in newer drivers. All required source-preservation checks passed.

Owned-process cleanup was confirmed using driver records and live PID/creation-time checks; no user process was stopped. Before/after distribution metadata matched in all **19 retained environments**. This metadata audit does not hash every installed library byte. New product/reference environments remained separate. Released-weight runs establish fixed-case numerical and integration evidence; earlier sections and the [Decision/Bruv/RSI acceptance record](windows-next-four.md) retain historical totals that should not be added to this increment.

The **final closeout on 2026-10-16** reran the complete Windows lightweight selection after the Tiny loader refinement, using the checked-in [`check_windows_suite.py`](../scripts/check_windows_suite.py) runner. It passed **1,186 unique tests**, with zero failures, errors, skips or duplicated test IDs:

| Test group | Passed |
|---|---:|
| Legacy / Tiny | 60 |
| Modern | 1,061 |
| Bruv | 37 |
| RSI | 28 |

These tests use small synthetic models, substitute runtimes and pinned SDK source checks; the closeout suite did not rerun the released-weight acceptances above. See the [copyable four-profile command](project-onboarding.md#test-and-verify). Wheel/sdist build and isolated installed-wheel CLI/resource checks also passed; they establish packaging integrity, not model execution or Linux/Mac runtime validation. The [closeout retrospective](project-retrospective.md#closeout-checks) records their scope.

#### Validate Tiny-Jev-1.7B and JevK5-9B on Windows

Use the [dedicated setup and acceptance guide](windows-tiny17-jev9.md) for the pinned artifacts, isolated reference/product environments and reproducible commands. Both use the installed `vllm-jev serve` entrypoint and `/v1/systemone`; model IDs and verified local paths containing Chinese characters and spaces passed.

Tiny 1.7B converts its published BF16 backbone and FP32 head to owned FP32 tensors, releasing each source mapping during loading. Its final listening-process peak working set was **6.822 GiB**, separate from **16.724 GiB** lifetime peak commit; minimum sampled host available RAM was **4.207 GiB**. It retains the author's 1.15 temperature, marker readout and 4,096-token state-truncation policy. The original Tiny 0.6B loading path is preserved.

JevK5-9B selects the official `jevk5-9b-v0.3.3-Q5_K_M.gguf` and pinned llama.cpp b11010 CPU runtime, with temperature **1.316**, top-40 label readout and at most 16 choices. Complete prompts over **1,024 tokens** are rejected. Its native llama-server peak working set was **5.719 GiB**; the Python service's approximately 0.0715 GiB is a separate process and does not represent model memory. These are local observations, not minimum memory or performance guarantees. This checkpoint is separate from `ZefanCai/Open-Jev-9B`, which remains outside the Windows scope.

#### Validate JevK5-Lite on Windows

`alibiserikbay/JevK5-Lite` is a separate DeBERTa encoder and scoring-head model, not a smaller Qwen checkpoint or a quantization of JevK5. Its Windows adapter passed **released-weight acceptance on 2026-10-15**. Use the isolated `windows-modern` installation above and the same installed entrypoint:

```powershell
.\.venv-windows-modern\Scripts\vllm-jev.exe serve alibiserikbay/JevK5-Lite --cpu-threads 4
```

Preparation pins release `315ee211f828a899161477c534cea23e84ba3568` and verifies all seven original files: `config.json`, `lite_config.json`, `tokenizer.json`, `tokenizer_config.json`, `model.safetensors`, `scorer.safetensors`, and `NOTICE`. The SHA256 values are recorded in [the adapter](../vllm_jev/windows_jevk5_lite.py) and independently in [the author reference runner](../scripts/windows_jevk5_lite_reference.py). The source directory remains unchanged; extra loader overrides are rejected. A complete local directory can also be passed to `serve`. Product serving uses installed Transformers classes with `trust_remote_code=False`, without executing downloaded author Python.

The encoder, scoring head, readout and Torch softmax run in FP32 with the release's single-decision temperature `1.8660659830736146`. Each question uses one complete forward with no padding, packed-question execution or cache reuse. The service accepts text or finite JSON state, 1–64 questions and at most 256 total candidates. Choice accepts 2–255 options; empty object descriptions fall back to their keys, and repeated rendered descriptions receive key prefixes. Score accepts 2–10 ordered string levels and uses listwise scoring (`isolated: true` is rejected); repeated levels receive index prefixes in the returned legend. Noul accepts only omitted/null criteria or an empty object, with raw probabilities ordered true then false. Score remains the project's probability-weighted expected level and confidence rather than the author's selected-level display.

Only the default/native template is supported. **Every complete question sequence must fit within 512 tokens**, including instructions, schema and state. Unlike the author's long-state truncation path, this adapter rejects oversized inputs before any question is evaluated; it does not silently drop state tokens. The option-count limits do not guarantee that a large schema fits. `--cpu-threads` accepts 1–4, bounded by the logical CPU count, and defaults to the smaller of four and that count. Images, video, custom Noul criteria, alternative templates and `cache_salt` are unsupported.

For an independent reference, obtain [the original JevK5 SDK](https://github.com/allebee/jevk5/tree/f26426d16f59e8bbe1470e5b162cc89329e29b29) at the fixed commit, preserving its LF bytes. The runner checks and privately copies exactly `jevk5/__init__.py`, `jevk5/prompt.py`, `jevk5/lite.py`, and `bench/jevk5_lite.py`. It calls the original bench adapter and classifier; the only compatibility shim is the bench's result data container. It neither imports the product inference adapter nor rewrites the author's numerical implementation. Keep its Python environment separate from the product and the earlier model environments:

```powershell
uv venv --python 3.12 .venv-lite-reference
uv pip install --python .venv-lite-reference\Scripts\python.exe torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv-lite-reference\Scripts\python.exe transformers==5.17.0 accelerate==1.15.0 numpy==2.5.4 tokenizers==0.23.3 huggingface_hub==1.33.0 safetensors==0.8.0 psutil
uv pip install --python .venv-windows-modern\Scripts\python.exe psutil requests
git -c core.autocrlf=false clone https://github.com/allebee/jevk5 .\reference-source\jevk5-lite
git -C .\reference-source\jevk5-lite -c core.autocrlf=false checkout f26426d16f59e8bbe1470e5b162cc89329e29b29
.\.venv-windows-modern\Scripts\python.exe -c "from pathlib import Path; from vllm_jev.windows_jevk5_lite import prepare, MODEL_ID; print(prepare(MODEL_ID, Path('windows-lite-data')))"
```

Run the original-author stage first, then validate the installed product command against that completed baseline. The second invocation reuses the first stage's `reference-gate-author.json` only after checking its release, ordered cases, source/SDK/runner hashes, dependency versions, interpreter identity and actual FP32 runtime settings:

```powershell
.\.venv-windows-modern\Scripts\python.exe -B scripts\check_windows_candidates.py --profile jevk5_lite --source .\windows-lite-data\checkpoint\alibiserikbay\JevK5-Lite --sdk-source .\reference-source\jevk5-lite --reference-python .\.venv-lite-reference\Scripts\python.exe --console-entrypoint .\.venv-windows-modern\Scripts\vllm-jev.exe --workspace .\windows-lite-check --output .\windows-lite-check\reference-gate.json --reference-only --threads 4 --abort-below-available-gib 1.5

.\.venv-windows-modern\Scripts\python.exe -B scripts\check_windows_candidates.py --profile jevk5_lite --source .\windows-lite-data\checkpoint\alibiserikbay\JevK5-Lite --sdk-source .\reference-source\jevk5-lite --reference-python .\.venv-lite-reference\Scripts\python.exe --console-entrypoint .\.venv-windows-modern\Scripts\vllm-jev.exe --workspace .\windows-lite-check --output .\windows-lite-check\acceptance.json --reference-report .\windows-lite-check\reference-gate-author.json --threads 4 --abort-below-available-gib 1.5
```

The completed acceptance ran on Windows 11 x64, Python 3.12.5, Core Ultra 5 125H and four CPU threads, using CPU PyTorch 2.8.0 and Transformers 5.17.0. A fresh original-author run established the baseline, which was then provenance-checked and reused for two fresh installed CLI starts: a model ID and a local path containing Chinese characters and spaces, with `PYTHONPATH` cleared. Each start passed four English/Chinese JSON/mixed/wide/duplicate-description cases plus two concurrent requests, for **twelve successful product responses**. Empty Choice descriptions and repeated Choice/Score descriptions were exercised. The maximum absolute difference from the author's raw probabilities and independently mapped project answers was **0**, within the unchanged `1e-5` tolerance. Sixteen invalid requests returned HTTP 422, including overlength and custom-Noul-criteria requests. Health stayed responsive during pending inference, repeated answers agreed, source and SDK hashes stayed unchanged, and owned processes and private workspaces were cleaned up.

The eight sequential HTTP requests took **0.41–1.28 seconds**, with a median of **0.84 seconds**, excluding model loading. Each contained one, two or three questions, whose complete input sequences were only **30–68 tokens** in the author report. The 512-token limit and oversized-input rejection do not establish successful inference or latency at exactly 512 tokens. The highest observed product working set was **2.09 GiB** and peak commit was **2.22 GiB**; the lowest sampled system available RAM during product execution was **10.79 GiB**. These are fixed-case observations with other applications active, not minimum-RAM requirements, a throughput benchmark or a latency guarantee.

The reference and product models loaded sequentially. Startup required 6 GiB available RAM and 8 GiB system commit headroom; the commands above used a 1.5 GiB available-RAM watchdog that stops only owned processes. These are experiment guards, not measured hardware requirements. Acceptance establishes local numerical and integration parity for the fixed cases, not task accuracy, other-platform equivalence or graceful shutdown.

All fourteen earlier models then passed fresh, sequentially scheduled product regression in their retained environments. The eight legacy models each passed one CLI start, two real requests and two HTTP 422 checks. JevK5-2B, Kev-0.8B, This-That 1.0/1.1/1.2 and Mica each passed two CLI starts, ten real requests and fourteen HTTP 422 checks. This totals **20 CLI starts, 76 successful requests and 100 rejections** for the earlier models. Their original-author baselines were hash-verified and reused, not rerun; the product inference was executed again. The maximum absolute difference from those historical author baselines was `4.66e-6`, within the existing per-model tolerances. All source-integrity and owned-process/private-workspace cleanup checks passed. Including Lite, the increment covered **15 models, 22 CLI starts, 88 successful requests and 116 rejections**.

Package inventories before and after the increment matched exactly in all three retained runtime environments: legacy, modern and Mica. No installed dependency was added, removed or changed there; Lite used new isolated product and reference environments. The modern lightweight suite passed **614 distinct test identities** after deduplicating overlapping reports and applying the final reference/harness rerun; the legacy suite separately passed **435 tests**. Both final sets had no failures or skips. These are separate environment-specific test counts, not a combined unique-test total, and do not replace the real-weight checks above.

#### Validate Mica Q5_K_M on Windows

The Windows adapter for `sky7350/Mica-v0.1-4B` uses the published **Q5_K_M GGUF** and the official **llama.cpp b11010 Windows x64 CPU** runtime. Use the `windows-modern` environment and start it through the same entrypoint:

```powershell
.\.venv-windows-modern\Scripts\vllm-jev.exe serve sky7350/Mica-v0.1-4B --cpu-threads 4
```

Preparation pins checkpoint revision `9b6428b272f4708ef54a4702b52bb453a4059363`, the 3,512,029,568-byte `mica-v0.1-4b-Q5_K_M.gguf` (SHA256 `7fbd1be2293ba158157bbbdcdc4ea57569bd9ed53901ed1766f81264d9da8292`), and the CPU runtime archive (SHA256 `3011251607251eca68ae67e35512364d9943d862ff598599ff9a4888c1753008`). The prepared directory contains eight checkpoint/tokenizer/license files and a `runtime/` directory with eighteen verified CPU DLLs and the OpenMP license. No native libraries are bundled in the Python wheel. Other quantizations, BF16 weights, added runtime DLLs, and arbitrary loader overrides are rejected. A prepared local directory, including a path containing Chinese characters or spaces, can also be passed to `serve`.

Mica uses its original prompt, 255-label codebook, last-position candidate-logit readout and calibration temperature `1.1244734010661372`. The adapter evaluates one question at a time with four or fewer CPU threads, no GPU layers, no shared-prefix or session cache, and a 1,024-token complete-prompt limit per question. Oversized requests are rejected before any question is evaluated. Choice accepts an object with 2–255 named string candidates; Score accepts 2–10 ordered string levels; Noul accepts its default false/true criteria or named false/true descriptions. Requests accept text or JSON state, with at most 64 questions and 256 total candidates. Only the default/native decision template is supported. Score is mapped to the project's expected zero-based level; the original calibrated probabilities are retained. These are quantized CPU decisions, not a claim of BF16 quality equivalence.

The product uses the pinned Rust tokenizer artifact and sandboxed Jinja template directly, with `tokenizers==0.23.3` and `jinja2==3.1.6`. It does not import PyTorch or Transformers into the Mica serving process. A process that has already imported either library, or loaded a conflicting native runtime, is rejected; use a fresh `vllm-jev` process. A separate tokenizer check matched the original AutoTokenizer on 25 complete prompts and their token IDs, plus all 257 answer-label IDs, covering whitespace, CRLF, Chinese/emoji, JSON, escaped special markers and a 255-option prompt. The last case checks tokenizer compatibility and does not imply that every 255-option request fits the serving context limit.

Released-weight acceptance passed on 2026-10-15 on Windows 11 x64, Python 3.12.5, Core Ultra 5 125H and four CPU threads. Two fresh installed `vllm-jev.exe` starts (model ID and a Unicode/space-containing local directory) returned ten successful responses. English, Chinese JSON and wider Choice cases covered Choice/Noul/Score; queued repeat requests and the second service start were identical to the initial responses. The maximum absolute answer-value difference against the separately executed original-author Q5 reference was **0**, within the fixed `1e-5` tolerance. Fourteen invalid requests returned HTTP 422, `/health` stayed responsive during pending inference, source/SDK hashes stayed unchanged, and all owned processes and private workspaces were cleaned up. Metadata identified the fixed GGUF/runtime, effective batch size 1,024, tokenizer versions, and the absence of imported Torch/Transformers. The author requests a batch size of 2,048 internally, which b11010 clamps to the 1,024-token context; acceptance checks the observed value.

Across the six sequential HTTP requests, elapsed time ranged from **9.46 to 21.54 seconds** (median **16.60 seconds**), excluding startup; each request contained either one or three questions. The highest observed product working set was **3.20 GiB**, and the lowest sampled system available RAM during the product runs was **3.61 GiB**. The independent author process reached a 3.21 GiB working set. These observations are neither minimum-RAM requirements nor performance guarantees. The adapter/entrypoint/dependency/shared-service suite passed 264 tests, and the independent reference/acceptance harness passed 53 tests. The new runtime environment was isolated from the existing legacy and modern environments; the earlier thirteen model implementations were unchanged.

After the shared CLI import changes, the three Laya releases passed six fresh real-model requests in the retained legacy environment, and This-That 1.2 passed ten fresh requests across two installed-entrypoint starts in the retained modern environment. These focused regressions reused verified historical author baselines; they did not rerun those authors or all thirteen previous models. Invalid-input checks, source integrity and owned-process/workspace cleanup also passed.

To establish an independent reference, obtain `akivet/Mica` at commit `f6bccd76e9e42b9193d1cdf3549b971e7b200d59` outside this checkout and copy exactly the seven `SDK_HASHES` files listed in `scripts/windows_mica_reference.py` to a separate SDK directory, preserving the original bytes. The reference executes the unmodified author implementation with its own Hugging Face tokenizer. Its Python 3.12 environment deliberately omits PyTorch: loading PyTorch's OpenMP library together with this llama.cpp CPU build caused a native-library conflict on Windows. Install only the tokenizer and reference dependencies:

```powershell
uv venv --python 3.12 .venv-mica-reference
uv pip install --python .venv-mica-reference\Scripts\python.exe transformers==5.5.0 tokenizers==0.22.2 numpy==2.5.4 huggingface_hub==1.33.0 safetensors==0.8.0 psutil requests
uv pip install --python .venv-windows-modern\Scripts\python.exe psutil requests
.\.venv-windows-modern\Scripts\python.exe -c "from pathlib import Path; from vllm_jev.windows_mica import prepare, MODEL_ID; print(prepare(MODEL_ID, Path('windows-mica-data')))"
.\.venv-windows-modern\Scripts\python.exe -B scripts\check_windows_mica.py --source .\windows-mica-data\checkpoint\sky7350\Mica-v0.1-4B --sdk-source .\reference-source\mica --reference-python .\.venv-mica-reference\Scripts\python.exe --workspace .\windows-mica-check --output .\windows-mica-check\mica.json --console-entrypoint .\.venv-windows-modern\Scripts\vllm-jev.exe --threads 4
```

The acceptance driver runs the reference and product serially, compares the same pinned Q5 model/runtime at an absolute tolerance of `1e-5`, checks model-ID and Unicode local-path starts, concurrent requests, responsive health, invalid-input rejection, unchanged source files, and owned-process cleanup. `--reference-only` creates an author baseline; `--reference-report` reuses a completed baseline only after verifying its model, inputs, source/SDK/runner hashes, environment and runtime settings. Its 6 GiB available-RAM and 8 GiB commit-headroom startup gates, plus the 1.5 GiB available-RAM watchdog, are experiment guards rather than measured hardware requirements. The driver does not establish task accuracy, BF16 equivalence, throughput, or graceful shutdown.

#### Validate Tiny-Jev on Windows

Tiny-Jev passed validation with its released weights on 2026-10-14: two fresh CLI starts using the model ID on Windows 11 x64, Python 3.12.5, PyTorch 2.6.0+cpu, and Transformers 4.57.6 matched the original implementation running separately with Transformers 5.17.0. All four fixed text/JSON cases per start passed, with a maximum absolute answer-value difference of `6.27e-6`, below the `1e-5` tolerance. Concurrent requests and unsupported-input checks passed, and source hashes remained unchanged. Two additional starts using a local checkpoint path containing Chinese characters and spaces also passed with unchanged source hashes.

The adapter uses the pinned `lostargon/Tiny-Jev` release `62449fb3272d68fb8c8fe8a33082566643a76d0b`, checks all six source files, and loads the Qwen3 backbone and scalar head directly in FP32. It preserves the author's readout immediately before each `<opt>` marker and the 0.95 temperature. The source files remain unchanged: RoPE and tokenizer metadata compatibility adjustments happen in memory. Serving does not execute the downloaded `modeling_tiny_jev.py` or enable `trust_remote_code`.

The reference driver deliberately uses a separate environment with the release's native Transformers 5.17.0. After verifying its fixed SHA256, it executes the original `modeling_tiny_jev.py` and calls the author's `decide` implementation. It does not reuse the project's prompt builder, compatibility adjustments, model loader, or scoring implementation. Prepare that reference environment separately from the installed checkout:

```powershell
uv venv --python 3.12 .venv-tiny-reference
uv pip install --python .venv-tiny-reference\Scripts\python.exe torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv-tiny-reference\Scripts\python.exe transformers==5.17.0 safetensors
.\.venv-windows-legacy\Scripts\python.exe scripts\check_windows_qwen.py --model lostargon/Tiny-Jev --workspace .\windows-check --output .\windows-check\tiny.json --reference-python .\.venv-tiny-reference\Scripts\python.exe
```

The driver downloads the pinned files if necessary; add `--source .\models\tiny-jev` to use an existing original release. It runs reference and service processes sequentially, then checks two fresh CLI starts, text/JSON and Chinese examples, Choice/Noul/Score agreement, concurrent requests, rejection of unsupported inputs, and unchanged source hashes. Its JSON report uses an absolute numerical tolerance of `1e-5`. The check measures agreement on these fixed cases, not task accuracy or throughput; it terminates only its own server process and does not validate graceful shutdown.

The model-ID validation run observed a median HTTP time of 1.09 seconds across eight requests, each containing one Choice, one Noul, and one Score question, with four CPU threads. The separate local-path checks are not included in this observation. It excludes model startup and is not a performance benchmark or a latency guarantee for longer inputs.

#### Validate OpenJev-0.6B on Windows

OpenJev-0.6B passed validation on 2026-10-14 with Python 3.12.5, PyTorch 2.6.0+cpu, Transformers 4.57.6, and PEFT 0.18.1. Two fresh CLI starts using the model ID matched the unmodified official v0.3.2 SDK using `predict(cached=False)` on four text/JSON states with all three question types. No absolute answer-value difference was observed on those cases after mapping the SDK's Score legend list to the HTTP object. Concurrent requests, unsupported-input rejection, and source-integrity checks also passed. Two additional starts using a local checkpoint path containing Chinese characters and spaces passed separately, with reference and serving source files unchanged. These checks establish agreement on the fixed cases, not model accuracy, cache equivalence, or throughput.

The model bundle is pinned to `IamBusy/OpenJev-0.6B` revision `b4bf78e7c053407b5d54a71bcc682ef5bbe59e72`, with the `Qwen/Qwen3-0.6B` base at `c1899de289a04d12100db370d81485cdf75e47ca`. The serving adapter uses installed PyTorch, Transformers, and PEFT; it neither imports the downloaded SDK nor enables `trust_remote_code`.

For an independent reference, obtain the official [IamBusy/OpenJev repository](https://github.com/IamBusy/OpenJev) at commit `5a827c549035cc4231ee77c5dd2bfcf7e42543ac` outside this checkout. Keep this SDK's reference in a separate PyTorch 2.6.0 / Transformers 4.57.6 / PEFT 0.18.1 environment and pass it explicitly with `--reference-python`; the product uses the newer Windows versions above. The driver checks fixed source hashes, copies only those verified files into a temporary package, and executes the original SDK in a separate process. The reference model also uses a private copy containing only the allowlisted bundle and base files, preserving the original checkpoint. No SDK installation or vendored source is required:

```powershell
uv venv --python 3.12 .venv-openjev-reference
uv pip install --python .venv-openjev-reference\Scripts\python.exe torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv-openjev-reference\Scripts\python.exe transformers==4.57.6 peft==0.18.1 safetensors "pydantic>=2,<3"
git clone --config core.autocrlf=false https://github.com/IamBusy/OpenJev.git ..\OpenJev-v0.3.2-reference
git -C ..\OpenJev-v0.3.2-reference checkout --detach 5a827c549035cc4231ee77c5dd2bfcf7e42543ac
.\.venv-windows-legacy\Scripts\python.exe scripts\check_windows_openjev.py --model IamBusy/OpenJev-0.6B --workspace .\windows-check --output .\windows-check\openjev.json --sdk-source ..\OpenJev-v0.3.2-reference --reference-python .\.venv-openjev-reference\Scripts\python.exe
```

Add `--source .\models\openjev` to use an existing complete bundle with `base/`. Both reference and service use FP32 CPU inference without caching, with an absolute comparison tolerance of `1e-5`. The driver records two fresh service starts, process ownership, source hashes, and observations in JSON. It terminates only its own server process; its cleanup does not validate graceful shutdown.

#### Validate Open-Jev-2B on Windows

The Windows adapter and its original BF16 CPU reference passed real-weight validation on 2026-10-14. The fixed model revision is `0c7aa498b1627be8da4acf34c863ff0ee0a92785`, with `Qwen/Qwen3.5-2B` at `15852e8c16360a2fea060d615a32b45270f8a8fc`. It serves all three question types through the text backbone and scalar head without generation, image processing, or a prefix cache. The following procedure checks agreement with the original author implementation; it is not an accuracy or throughput benchmark.

The author-reference run on 2026-10-14 used Windows 11, a Core Ultra 5 125H, 32 GB RAM, and four CPU threads. Loading took 6.989 seconds; its four requests, each containing Choice, Noul, and Score, took 67.953, 70.050, 59.848, and 59.888 seconds. Windows peak working set was 3,246,247,936 bytes (3.023 GiB), with private bytes around 5.63 GB. Source, HF snapshot, and private checkpoint hashes remained unchanged. These observations describe this reference run only: they are not minimum memory requirements, a benchmark, or product-service measurements.

The product CLI then passed two fresh starts, using the model ID and a local path containing Chinese characters and spaces. Each start ran all four reference cases and two concurrent requests: all 12 answers matched the saved, provenance-checked author reference within `1e-5`, with a maximum absolute difference of approximately `2.22e-16`. The reference was reused explicitly, not rerun in this product-validation invocation. All 14 invalid requests returned HTTP 422, and both health checks returned HTTP 200 while two decision requests were unfinished. Source hashes remained unchanged; all owned server processes exited and their private workspaces were removed. Terminating those processes does not establish graceful-shutdown behavior.

Across the eight sequential product requests, HTTP latency ranged from 48.069 to 90.763 seconds, with a median of 53.131 seconds. Each request contained Choice, Noul, and Score, requiring six candidate forwards. The highest observed service working set was 3,267,612,672 bytes (3.043 GiB). These are local observations on the same machine with four CPU threads, not a throughput benchmark, memory requirement, or Linux/GPU parity claim. This CPU path provides a functional baseline; it remains experimental and slow for interactive use. The Windows test suite passed 217 tests, and the five earlier models passed real-weight regression checks in the isolated dependency environment.

**1. Prepare the pinned reference and a shared source.** Use the Python 3.12 Windows environment from the installation section, with CPU PyTorch 2.8.0, Transformers 5.10.2, PEFT 0.19.1, and Accelerate 1.13.0. Add the memory-reporting dependency and obtain the original code outside this checkout. Disabling Git line-ending conversion preserves the bytes checked by the driver:

```powershell
uv pip install --python .venv-windows-legacy\Scripts\python.exe psutil
git clone --config core.autocrlf=false --filter=blob:none --no-checkout https://github.com/Zefan-Cai/Open-Jev.git ..\Open-Jev-2B-reference
git -C ..\Open-Jev-2B-reference sparse-checkout set jev
git -C ..\Open-Jev-2B-reference checkout --detach bd4118882f733574a3250a4b65fe4d884130c08b
```

The author code resolves its base by Hub ID, so it needs a standard Hugging Face cache. A product-only `serve` download uses `local_dir` and does not create that cache. For a fresh validation directory on one NTFS volume, the snippet below downloads the base once into the standard cache and hard-links its files into the product layout. It downloads only the nine pinned base files and nine release files; no duplicate weight copy or helper script is written. Preparation uses the product's lightweight constants; the validation driver independently checks its own pinned source and SDK hashes before loading anything:

```powershell
@'
import os
from pathlib import Path
from huggingface_hub import snapshot_download
from vllm_jev.windows_open_jev import (
    BASE_MODEL_ID, BASE_REVISION, MODEL_ID, REVISION, SOURCE_HASHES, verify_source,
)

root = Path("windows-check-2b").resolve()
source = root / "workspace/checkpoint" / MODEL_ID
base_files = [name[5:] for name in SOURCE_HASHES if name.startswith("base/")]
base = Path(snapshot_download(
    BASE_MODEL_ID, revision=BASE_REVISION, token=False,
    cache_dir=root / "hf-cache", allow_patterns=base_files,
))
snapshot_download(
    MODEL_ID, revision=REVISION, token=False, local_dir=source,
    allow_patterns=[name for name in SOURCE_HASHES if not name.startswith("base/")],
)
(source / "base").mkdir(parents=True, exist_ok=True)
for name in base_files:
    original, target = (base / name).resolve(), source / "base" / name
    if target.exists():
        if not os.path.samefile(original, target):
            raise RuntimeError(f"Existing file is not the expected hard link: {target}")
    else:
        os.link(original, target)
verify_source(source)
print(source)
'@ | .\.venv-windows-legacy\Scripts\python.exe -B -
```

**2. Run the author reference alone.** The driver copies five hash-verified author modules into a private package and calls the original `jev.serving.load_predictor().predict()` in a separate process. It runs offline, with CPU BF16, an FP32 head, candidate batch size one, and prefix caching disabled. It does not import product inference modules or modify the original checkpoint:

```powershell
$checkArgs = @(
  'scripts\check_windows_open_jev.py',
  '--source', '.\windows-check-2b\workspace\checkpoint\ZefanCai\Open-Jev-2B',
  '--workspace', '.\windows-check-2b\workspace',
  '--hf-cache', '.\windows-check-2b\hf-cache',
  '--sdk-source', '..\Open-Jev-2B-reference',
  '--reference-python', '.\.venv-windows-legacy\Scripts\python.exe',
  '--dtype', 'bfloat16', '--threads', '4'
)
.\.venv-windows-legacy\Scripts\python.exe -B @checkArgs --reference-only --output .\windows-check-2b\reference-bf16.json
```

**3. Compare with two fresh product CLI starts.** After the reference-only report passes, run the full driver. It reruns the reference first and starts each service only after the preceding model process has exited:

```powershell
.\.venv-windows-legacy\Scripts\python.exe -B @checkArgs --output .\windows-check-2b\cli-parity-bf16.json
```

To reuse step 2's completed author report instead of rerunning those same reference cases, supply its `-original-reference.json` file (not the outer summary):

```powershell
.\.venv-windows-legacy\Scripts\python.exe -B @checkArgs --reference-report .\windows-check-2b\reference-bf16-original-reference.json --output .\windows-check-2b\cli-parity-bf16.json
```

Reuse requires matching completed cases, question/state content, source and SDK hashes, package versions, threads, precision, and uncached inference settings. The driver checks these before starting any CLI and records the reused report's path, SHA256, and that the reference was not rerun. An incomplete or incompatible report is rejected. This option cannot be combined with `--reference-only`; all product cases and concurrency checks still run. Omitting it retains the full reference-first procedure.

The full check uses the model-ID route and then a local path containing Chinese characters and spaces. It compares four fixed text/JSON states with Choice, Noul, and Score, using a fixed absolute tolerance of `1e-5`, and checks unsupported inputs, concurrent requests, responsive health checks, and unchanged source hashes. Reports include actual parameter precision, load/request wall time, and Windows `peak_wset` for the reference and actual listening service process. Cleanup stops only each run's recorded process tree; if exit cannot be confirmed, its temporary directory and diagnostic path are retained. This does not validate graceful shutdown behavior.

Use `--case` to select a smaller subset; the report records the selected case IDs. `--dtype float32` is allowed only with `--reference-only` and produces a separate baseline; it cannot be compared with this BF16 service. The driver does not download weights or install dependencies.

#### Validate Decider on Windows

The experimental Windows CPU adapter handles `Mapika/decider-0.8b` at revision `a0a01d6f8135298f400a8c856b355793012ae971` and `Mapika/decider-2b` at revision `ff99089b6810691bd79134ec6febb5d9afa6e619`, using the same pinned dependency versions above. It loads the complete BF16 text checkpoint and projects the answer-slot state onto all 255 BF16 answer-label rows before converting logits to FP32. It does not generate tokens or use a persistent cache. Score levels are judged independently by default and their yes-probabilities are normalized across levels.

The 0.8B release uses temperature `1.03`. The 2B v11 release uses Choice `1.164`, Noul `1.624`, and Score `1.124`, with global fallback `1.145`. Its independent Score no/yes rows retain the Score temperature; they do not use the Noul temperature. Each release has its own seven-file identity and calibration checks.

The release is kept read-only. The source directory must contain exactly the seven hash-verified inference files: `config.json`, `decider_config.json`, `generation_config.json`, `tokenizer_config.json`, `tokenizer.json`, `chat_template.jinja`, and `model.safetensors`. The automatic model-ID preparation downloads this exact layout. No author Python is executed by the product server. All extra entries, including README or license files copied from a full Hub checkout, are rejected; only Hugging Face's `.cache/` bookkeeping directory is allowed. Keep the separately downloaded author reference in its own directory.

Rows run one at a time on CPU, with the author's attention mask and right padding to a multiple of 64. Each complete row is limited to 1,024 tokens, including padding; oversized inputs fail before any model forward and are never silently truncated. Choice accepts 2–255 distinct labels, and Score accepts 2–10 levels. The shared 64-question/256-candidate and text-only limits also apply. `/health` identifies `backend: decider-pytorch` and the exact model revision. Decision metadata includes BF16 backbone/readout, FP32 logits, row batch size one, and the padding/context limits.

On 2026-10-14, each Decider release passed ten real product requests across two fresh starts (model ID and a Chinese/space-containing local path): six sequential requests covering English mixed Choice/Noul/Score, Chinese JSON mixed questions, and a 12-option Choice, plus four queued concurrent requests. All answer values matched the independently executed author CPU reference after mapping to project response conventions, with maximum absolute difference **0** for both models at a tolerance of `1e-5`. Each model rejected fourteen unsupported requests with 422; health remained responsive during queued decisions. Original checkpoint hashes remained unchanged and all owned model processes and temporary service directories were cleaned up. The completed author references were explicitly reused during the corresponding CLI checks, not rerun in those invocations.

These runs used Windows 11, Core Ultra 5 125H, 32 GB RAM, four CPU threads, PyTorch 2.8.0+cpu and Transformers 5.10.2:

| Model | Median sequential HTTP time | Sequential range | Peak service working set | Peak private commit |
|---|---:|---:|---:|---:|
| Decider-0.8B | 39.159 s | 10.328–45.342 s | 1,531,166,720 bytes (1.426 GiB) | 4,051,992,576 bytes (3.774 GiB) |
| Decider-2B | 48.246 s | 18.442–76.265 s | 3,284,185,088 bytes (3.059 GiB) | 8,578,715,648 bytes (7.990 GiB) |

Each median covers six sequential requests; the single-row wide Choice is cheaper than the mixed five-row requests. Commit is not physical RAM usage; neither observation is a minimum-memory requirement. These fixed cases check integration and numerical parity, not task accuracy, GPU parity or throughput. Both models remain slow CPU baselines for interactive use.

The same fresh environment passed 358 Windows/template tests and real-weight regression for all six earlier models. The three Laya releases and Tiny-Jev matched their prior product answers exactly; OpenJev-0.6B's largest difference was `4.69e-6` within `1e-5`. The Open-Jev-2B regression reused its complete, provenance-checked historical author report and ran six focused requests across two fresh routes (English and JSON, each once sequentially and twice queued), with maximum difference `2.22e-16`. It did not rerun the historical twelve-request acceptance suite. Existing model implementations, dependency pins and prior environments remained unchanged. The repository reference runner also passed a fresh 0.8B run with raw probabilities identical to the earlier external reference runner.

The author APIs round probabilities to four decimals and Score values to two. The 0.8B API uses maximum-probability confidence; the 2B v11 API uses the project's confidence formulas and exposes maximum probability separately as `x_p_max`. This project retains unrounded probabilities and its existing chance-normalized Choice/distance-based Score confidence. Validation observes the original, unmodified author's raw probabilities, checks them against each release's displayed answers, and independently maps them into the project response format. Custom templates and stricter input validation are project features; the real parity cases use the original native template.

To reproduce, first install `psutil` in the isolated Windows environment. Obtain the fixed inference data and the matching author Python files in separate directories. The commands below download data only; the validation step checks independent pins before executing the author reference:

```powershell
uv pip install --python .venv-windows-legacy\Scripts\python.exe psutil
$deciderModel = 'Mapika/decider-0.8b' # Or Mapika/decider-2b
$deciderRoot = Join-Path $PWD ('windows-check-' + $deciderModel.Replace('/', '-'))
@'
import sys
from pathlib import Path
from huggingface_hub import snapshot_download
from vllm_jev.windows_decider import prepare, MODELS

model, root = sys.argv[1], Path(sys.argv[2]).resolve()
prepare(model, root / "workspace")
snapshot_download(model, revision=MODELS[model]["revision"], token=False,
                  local_dir=root / "reference-source", allow_patterns=["decider/*.py"])
'@ | .\.venv-windows-legacy\Scripts\python.exe -B - $deciderModel $deciderRoot

$deciderArgs = @(
  'scripts\check_windows_decider.py', '--model', $deciderModel,
  '--source', (Join-Path $deciderRoot ('workspace\checkpoint\' + $deciderModel)),
  '--workspace', (Join-Path $deciderRoot 'workspace'),
  '--sdk-source', (Join-Path $deciderRoot 'reference-source'), '--threads', '4'
)
.\.venv-windows-legacy\Scripts\python.exe -B @deciderArgs --reference-only --output (Join-Path $deciderRoot 'reference.json')
.\.venv-windows-legacy\Scripts\python.exe -B @deciderArgs --reference-report (Join-Path $deciderRoot 'reference-original-reference.json') --output (Join-Path $deciderRoot 'cli-parity.json')
```

Changing `$deciderModel` also selects a separate data/reference directory. Do not mix the two releases' author packages: their files and hashes differ. The repository's `scripts/windows_decider_reference.py` is the default independent reference runner; it copies the pinned author package into a private directory and never imports the product inference modules. `--reference-report` reuses a completed matching raw report and records that reuse; omitting it runs a fresh reference first. The driver runs each model process sequentially, tests both CLI routes, records peak working set/commit and request times, and terminates only its own recorded process tree. This cleanup is not a graceful-shutdown test. It does not download model data or install dependencies during validation.

The experiment precheck requires at least 4 GiB available physical memory and 6 GiB remaining system commit capacity for 0.8B; selecting 2B raises these defaults to 6 GiB and 8 GiB. A watchdog stops only the test's own model process if available RAM falls below 0.75 GiB. These are conservative guardrails for short validation cases, not established production requirements. Save work and close unused applications first; the validator never closes unrelated applications. Check that each saved report says `status: passed` before continuing; insufficient startup memory produces a `skipped` report without starting a model.

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

Input support depends on the model. Platform entries distinguish CUDA and Ascend on Linux.
Ascend entries have passed text API pressure tests on 910B3; Valen media on Ascend remain unverified; JevAny also passed generated-image functional checks. Use `--backend ascend` and the matched [Ascend runtime](ascend.md).

| Model | Input | Platform | Start server |
|---|---|---|---|
| [ZefanCai/Open-Jev-2B](https://huggingface.co/ZefanCai/Open-Jev-2B) | Text | Linux (CUDA, Ascend), Mac, Windows | `vllm-jev serve ZefanCai/Open-Jev-2B` |
| [ZefanCai/Open-Jev-9B](https://huggingface.co/ZefanCai/Open-Jev-9B) | Text | Linux (CUDA, Ascend) | `vllm-jev serve ZefanCai/Open-Jev-9B` |
| [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya) | Text | Linux (CUDA, Ascend), Mac, Windows | `vllm-jev serve convaiinnovations/laya` |
| [convaiinnovations/laya-multilingual](https://huggingface.co/convaiinnovations/laya-multilingual) | Text | Linux (CUDA, Ascend), Mac, Windows | `vllm-jev serve convaiinnovations/laya-multilingual` |
| [convaiinnovations/laya-typed-decisions](https://huggingface.co/convaiinnovations/laya-typed-decisions) | Text | Linux (CUDA, Ascend), Mac, Windows | `vllm-jev serve convaiinnovations/laya-typed-decisions` |
| [IamBusy/OpenJev-0.6B](https://huggingface.co/IamBusy/OpenJev-0.6B) | Text | Linux (CUDA, Ascend), Mac, Windows | `vllm-jev serve IamBusy/OpenJev-0.6B` |
| [lostargon/Tiny-Jev](https://huggingface.co/lostargon/Tiny-Jev) | Text | Linux (CUDA, Ascend), Mac, Windows | `vllm-jev serve lostargon/Tiny-Jev` |
| [lostargon/Tiny-Jev-1.7B](https://huggingface.co/lostargon/Tiny-Jev-1.7B) | Text | Windows CPU FP32 (experimental) | `vllm-jev serve lostargon/Tiny-Jev-1.7B` |
| [Valen-Team/Valen-Preview-0923](https://huggingface.co/Valen-Team/Valen-Preview-0923) | Text + images + video | Linux (CUDA; Ascend text only), Mac | `vllm-jev serve Valen-Team/Valen-Preview-0923` |
| [yah01/vjev-vision](https://huggingface.co/yah01/vjev-vision) | Text + images | Linux (CUDA; Ascend text only) | `vllm-jev serve yah01/vjev-vision` |
| [yah01/vjev-vision-pilot](https://huggingface.co/yah01/vjev-vision-pilot) | Text + images | Linux (CUDA; Ascend text only) | `vllm-jev serve yah01/vjev-vision-pilot` |
| [shgao/rsi-jev-v4.0-vl-qwen3.5-2b](https://huggingface.co/shgao/rsi-jev-v4.0-vl-qwen3.5-2b) | Text + images | Linux (CUDA, Ascend) | `vllm-jev serve shgao/rsi-jev-v4.0-vl-qwen3.5-2b` |
| [shgao/rsi-jev-v3.0-qwen3.5-2b](https://huggingface.co/shgao/rsi-jev-v3.0-qwen3.5-2b) | Text | Linux (CUDA, Ascend) | `vllm-jev serve shgao/rsi-jev-v3.0-qwen3.5-2b` |
| [SimpleJev/JevAny-Qwen3.5-4B-LoRA](https://huggingface.co/SimpleJev/JevAny-Qwen3.5-4B-LoRA) | Text, images | Linux (CUDA, Ascend) | `vllm-jev serve SimpleJev/JevAny-Qwen3.5-4B-LoRA` |
| [SimpleJev/JevAny-Qwen3.8-27B-LoRA](https://huggingface.co/SimpleJev/JevAny-Qwen3.8-27B-LoRA) | Text, images | Linux (CUDA) | `vllm-jev serve SimpleJev/JevAny-Qwen3.8-27B-LoRA` |
| [Cloudflare/clef-flash](https://huggingface.co/Cloudflare/clef-flash) (9B) | Text, images, video | Linux (CUDA) | `vllm-jev serve Cloudflare/clef-flash` |
| [Cloudflare/clef](https://huggingface.co/Cloudflare/clef) (27B) | Text, images, video | Linux (CUDA) | `vllm-jev serve Cloudflare/clef` |
| [nace-ai/drex-v2.0](https://huggingface.co/nace-ai/drex-v2.0) (35B, experimental) | Text / JSON | Linux (CUDA) | `vllm-jev serve nace-ai/drex-v2.0 --workspace /data/models/drex --tensor-parallel-size 2 --max-model-len 8193` |

Use `yah01/vjev-vision` for the current vjev release; the pilot is an earlier checkpoint. Both were trained on single images.

Valen and vjev image requests use `/v1/systemone`: up to 8 PNG/JPEG images, 8 MiB per image. Valen also accepts one short MP4 video per request. The [RSI-Jev](#rsi-jev) and [Clef](#clef) adapters have their own limits and request formats; see [Text and images](#text-and-images) and [Video input](#video-input).

### Drex v2.0 (experimental)

Drex uses the same `vllm-jev serve` CLI and `/v1/systemone` API for text/JSON
Choice, Noul and Score. The official 35B BF16 release has been exercised with
TP=2 on two A800 80 GB GPUs. Use an explicit data-disk workspace. Native causal
rows are required; media, non-native templates, prefix caching and RadixFlow
are rejected. See the [Drex guide](drex.md) for setup and validation scope.

### Additional decision checkpoints (experimental)

These text-only adapters use `vllm-jev serve` and `/v1/systemone` for Choice, Noul, and Score on the listed platforms.

| Model | Tested platform | Start server |
|---|---|---|
| [jaredpalmer/kev-0.6b](https://huggingface.co/jaredpalmer/kev-0.6b) | Windows CPU (experimental) | `vllm-jev serve jaredpalmer/kev-0.6b` |
| [jaredpalmer/kev-0.8b](https://huggingface.co/jaredpalmer/kev-0.8b) | Linux (CUDA, Ascend), Windows | `vllm-jev serve jaredpalmer/kev-0.8b` |
| [jaredpalmer/kev-4b](https://huggingface.co/jaredpalmer/kev-4b) | Linux (CUDA, Ascend) | `vllm-jev serve jaredpalmer/kev-4b` |
| [Mapika/decider-0.8b](https://huggingface.co/Mapika/decider-0.8b) | Linux (CUDA, Ascend), Mac, Windows | `vllm-jev serve Mapika/decider-0.8b` |
| [Mapika/decider-2b](https://huggingface.co/Mapika/decider-2b) | Linux (CUDA, Ascend), Windows | `vllm-jev serve Mapika/decider-2b` |
| [Mapika/decider-4b-GGUF](https://huggingface.co/Mapika/decider-4b-GGUF) | Windows CPU Q4_K_M (experimental) | `vllm-jev serve Mapika/decider-4b-GGUF` |
| [sky7350/Mica-v0.1-4B](https://huggingface.co/sky7350/Mica-v0.1-4B) | Linux (CUDA, Ascend), Windows | `vllm-jev serve sky7350/Mica-v0.1-4B` |
| [flock-io/this-that-model-1.0](https://huggingface.co/flock-io/this-that-model-1.0) | Linux (CUDA, Ascend), Windows | `vllm-jev serve flock-io/this-that-model-1.0` |
| [flock-io/this-that-model-1.1](https://huggingface.co/flock-io/this-that-model-1.1) | Linux (CUDA, Ascend), Windows | `vllm-jev serve flock-io/this-that-model-1.1` |
| [flock-io/this-that-model-1.2](https://huggingface.co/flock-io/this-that-model-1.2) | Linux (CUDA, Ascend), Mac, Windows | `vllm-jev serve flock-io/this-that-model-1.2` |
| [alibiserikbay/JevK5](https://huggingface.co/alibiserikbay/JevK5) | Linux (CUDA, Ascend), Windows | `vllm-jev serve alibiserikbay/JevK5` |
| [alibiserikbay/JevK5-9B](https://huggingface.co/alibiserikbay/JevK5-9B) | Windows CPU official Q5_K_M (experimental) | `vllm-jev serve alibiserikbay/JevK5-9B` |
| [alibiserikbay/JevK5-2B](https://huggingface.co/alibiserikbay/JevK5-2B) | Linux (Ascend), Mac, Windows | `vllm-jev serve alibiserikbay/JevK5-2B` |
| [alibiserikbay/JevK5-Lite](https://huggingface.co/alibiserikbay/JevK5-Lite) | Windows CPU only (experimental) | `vllm-jev serve alibiserikbay/JevK5-Lite` |
| [evalengine/decision-0.8b-gguf](https://huggingface.co/evalengine/decision-0.8b-gguf) | Windows CPU Q8_0 (experimental) | `vllm-jev serve evalengine/decision-0.8b-gguf` |
| [bvolpato/bruv1-0.8b](https://huggingface.co/bvolpato/bruv1-0.8b) | Windows CPU FP32 (experimental) | `vllm-jev serve bvolpato/bruv1-0.8b` |
| [evalengine/decision-4b-gguf](https://huggingface.co/evalengine/decision-4b-gguf) | Windows CPU Q4_K_M (experimental) | `vllm-jev serve evalengine/decision-4b-gguf` |
| [shgao/rsi-jev-v1.0-qwen3.5-0.8b](https://huggingface.co/shgao/rsi-jev-v1.0-qwen3.5-0.8b) | Windows CPU FP32 (experimental) | `vllm-jev serve shgao/rsi-jev-v1.0-qwen3.5-0.8b` |

Linux uses native vLLM pooling; Mac uses MLX; the listed Windows adapters use the experimental CPU backend. These adapters use only `/v1/systemone`. JevK5, JevK5-2B and the Windows JevK5-9B adapter accept 2–16 options per question; Windows JevK5 4B/9B select their fixed official Q5_K_M artifacts and support only the native template. The separate Windows-only JevK5-Lite adapter accepts 2–255 Choice options and 2–10 listwise Score levels, within a 512-token complete-question limit. Kev-0.6B/0.8B use separate pinned Qwen3/Qwen3.5 bases with a 1,024-token limit per complete question. This-That keeps at most 1,536 state tokens before adding the questions; the Windows 1.0/1.1/1.2 adapter additionally rejects a complete packed request exceeding 1,024 tokens after padding. [Decider-4B Q4_K_M](#validate-decider-4b-q4_k_m-on-windows) requires the fixed official CPU wheel, accepts only the native template, and limits each complete prompt to 1,024 tokens.

The additional [Decision, Bruv and RSI Windows adapters](windows-next-four.md) have their own native-only templates, option limits and isolated environments. Eval Engine Decision is separate from Mapika Decider; RSI v1.0 is a text release separate from the v3.0/v4.0-VL adapters below.

<a id="rsi-jev-experimental"></a>

### JevAny Pointer

[JevAny](https://github.com/SimpleJev/JevAny) Pointer models run on Linux and answer Choice, Noul and Score through `/v1/systemone`. The adapter was contributed by [heshien / AsirAsir02 in PR #9](https://github.com/mode-io/vllm-jev/pull/9).

```bash
vllm-jev serve SimpleJev/JevAny-Qwen3.5-4B-LoRA
# Or: SimpleJev/JevAny-Qwen3.8-27B-LoRA
```

First use downloads the pinned adapter and base, merges LoRA, and prepares a reusable checkpoint. An exported checkpoint directory can also be served directly. The optional protocol name is `jevany_pointer_v1`.

- One question per request; 1–255 Choice options or 2–255 Score levels.
- Up to two PNG/JPEG images, 8 MiB each, in top-level `media`. Video and Mac serving are not supported.
- State and the complete prompt must each fit the checkpoint's 8,192-token limits. Overlong requests are rejected rather than truncated.
- Prefix caching is disabled because the Pointer head reads multiple token states from the complete sequence.
- `context_order`, `instructions-first`, and state/instruction wrappers are supported. Custom context ordering keeps the image prefix and trailing Pointer readouts intact, repeating the instruction/options where needed; changing the order can change answers.

```json
{
  "state": "Describe the visible object.",
  "media": [{"type": "image", "uri": "data:image/png;base64,..."}],
  "questions": {
    "object": {
      "type": "choice",
      "instructions": "Which object is shown?",
      "criteria": {"cat": "A cat", "dog": "A dog"}
    }
  }
}
```

### RSI-Jev

RSI-Jev v3.0 (text) and v4.0-VL (text and images) run on Linux and use `/v1/systemone` for Choice, Noul, and Score. Option labels are part of the prompt, so renaming a label can change the answer.

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

<a id="clef-experimental"></a>

### Clef

Clef-Flash (9B) and Clef (27B) run on Linux and answer all questions about a state in one forward pass, returning a probability for each option. Text, media, and questions share one input sequence.

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

On Windows, the models listed in the [experimental CPU guide](#windows-cpu-experimental) use the same CPU service. It accepts `--host`, `--port`, and `--cpu-threads`; vLLM and RadixFlow options are rejected.

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

### Decision templates

`--decision-template` controls decision text before the model's chat formatting. Omitting it, or using `native`, preserves the original prompts.

Save the logical block order in a JSON file, for example `routing.json`:

```json
{
  "context_order": ["instructions", "criteria", "state"]
}
```

```bash
vllm-jev serve ZefanCai/Open-Jev-2B --decision-template routing.json
```

`instructions` contains question rules; `criteria` contains candidate descriptions or score levels; `state` contains the changing evidence, including supported media. List all three names exactly once, in any order. For example, `["criteria", "state", "instructions"]` puts the candidate context first. Each adapter renders these logical blocks in its own text, JSON, token or chat format. Leading system messages and the model's native readout structure are retained. Some adapters repeat question/option text in the protected readout after the ordered context.

You can combine `context_order` with `state_template` and `instruction_template`. It is a separate configuration from `layout`, `choice_prompt` and `noul_prompt`; combining them is rejected at startup. Model context limits still apply, including any repeated readout text.

Matching fixed blocks can share eligible prefix-cache entries on backends supporting reuse. Cross-request reuse also needs the same trusted [`cache_salt`](#shared-prefixes-across-system-one-requests). The order option does not enable suffix caching or add cache support to a backend.

The existing `native` and `instructions-first` presets remain compatible on the paths that support both. The Windows JevK5-Lite, JevK5 4B/9B Q5_K_M, Mica Q5_K_M, Decider-4B Q4_K_M, Decision, Bruv and RSI v1.0 adapters accept only the default/native template and reject other presets or context overrides. For the other supported paths, `instructions-first` preserves its original per-family behavior:

| Model family | `instructions-first` layout |
|---|---|
| Open-Jev-2B/9B, OpenJev-0.6B | Instructions, state, candidate and scoring cue |
| Tiny-Jev, Kev, Decider, Mica | Instructions, state, native option/readout blocks |
| Tev and task-JSON derivatives, JevK5 | Question and options before state inside the native JSON/chat wrapper |
| This-That | Numbered instructions, state, numbered options and answer slots |
| RSI-Jev | Instructions, state, options and final answer cue |
| Clef/Clef-Flash | Instruction preamble, state/media, native schema and final joint-decision cue |
| Valen | One numbered instruction message before state/media; each branch references its question |
| vjev | Instructions before state; Noul also retains its final question because that is its readout |
| Laya | Already instruction/options-first; the native order is unchanged |

Clef retains its schema after the state so the head reads state-conditioned question and option representations. The earlier instruction preamble does not replace those readout spans.

Laya uses bidirectional attention and Clef reads the full sequence; this option does not turn either into a decoder prefix-cache path. RSI-Jev rejects an oversized nonnative prompt rather than truncating its leading instructions. Existing platform and modality limits still apply.

All supported model adapters can apply a shared instruction template. Save a JSON file, for example `routing.json`:

```json
{
  "instruction_template": "Use the supplied evidence and our routing policy.\n{instructions}"
}
```

```bash
vllm-jev serve Cloudflare/clef-flash --decision-template routing.json
```

This wraps each explicitly supplied, nonempty text instruction on `/v1/systemone`, and the question text on the Choice/batch endpoints where available. It preserves state/media, candidate labels and descriptions, and the model's special markers and readout rules. Missing instructions retain the adapter's existing behavior.

An optional `state_template`, such as `"Business policy...\n{state}"`, can wrap text or a JSON object/list. JSON is rendered deterministically. A non-identity wrapper rejects `state.messages` so chat roles and embedded media are not flattened; use instruction formatting for those requests. Top-level Clef media fields are left intact.

For Open-Jev-2B/9B, a JSON file may also customize the complete decision text:

| JSON field | Required placeholders | Applies to |
|---|---|---|
| `context_order` | A permutation of `instructions`, `criteria`, `state` | Logical context ordering on all model families |
| `layout` | `native` or `instructions-first` | Compatibility presets, as listed above |
| `state_template` | `{state}` | Text or JSON objects/lists on all model adapters |
| `instruction_template` | `{instructions}` | All model adapters |
| `choice_prompt` | `{state}`, `{instructions}`, `{candidate}` | Open-Jev-2B/9B Choice and Score |
| `noul_prompt` | `{state}`, `{instructions}` | Open-Jev-2B/9B Noul |

A Choice/Score template may also use optional `{options}`, a JSON array of all rendered candidate descriptions. This lets fixed instructions and the full candidate list precede the state while retaining the selected candidate and scoring cue at the end:

```json
{
  "choice_prompt": "Question: {instructions}\nAllowed candidates: {options}\nContext:\n{state}\n\nProposed answer: {candidate}\nIs this proposed answer correct? Answer Yes or No."
}
```

Placeholders insert literal text; use `{{` and `}}` for literal braces. Files contain JSON, not executable code or Jinja. The configuration is loaded at startup; restart to change it. All families support state/instruction formatting and the preset above. Complete `choice_prompt`/`noul_prompt` replacement remains specific to Open-Jev-2B/9B: other families retain their trained marker, option-span and joint-readout structure.

Changing block order or text formatting can change answers, probabilities and calibration. Validate the chosen configuration on your labelled workload.

### Online prefix cache (experimental)

For repeated Open-Jev-2B text requests on Linux, an optional prefix tree learns shared token paths and retains complete Attention and GDN states at selected positions. It is **disabled by default** and has been tested with Open-Jev-2B on an A800 using vLLM 0.29.0.

Enable it when starting the server:

```bash
VLLM_JEV_ONLINE_PREFIX_CACHE=1 \
  vllm-jev serve ZefanCai/Open-Jev-2B --prefix-match-unit 16
```

Both settings are required. Send related requests to `/plugins/vllm-jev/choice` with the same `cache_salt` to reuse their shared prefix across requests:

```json
{
  "state": "The document shared by these requests...",
  "question": "Which action does the document support?",
  "options": ["Approve", "Request more information", "Decline"],
  "cache_salt": "<random secret assigned by your authenticated gateway>"
}
```

Use an unpredictable salt issued by an authenticated gateway, as described in [Shared prefixes across System One requests](#shared-prefixes-across-system-one-requests). Without an explicit salt, the Choice endpoint uses a fresh namespace for each request. A matching tree path is reusable only while all required model states remain cached. Cold paths are removed as capacity is needed, and near-limit inputs bypass checkpoint reads when a private continuation page would not fit.

The default budgets are eight namespace trees, 64 recent prompt paths per namespace, and four extra checkpoints; in-use states remain protected until they can be reclaimed. Adjust these with `VLLM_JEV_TREE_TENANTS`, `VLLM_JEV_TREE_PATHS`, and `VLLM_JEV_TREE_CHECKPOINTS`. Checkpoint admission is considered every 16 observed candidate sequences after the initial 16; `VLLM_JEV_TREE_PROMOTE_AFTER` changes that interval.

Creating checkpoints adds computation and GPU-memory cost. In the evaluated mixed-article workload, this mode did **not** improve throughput, and selected labels sometimes changed. Benchmark it on your own request pattern before enabling it for production. Other models and multimodal inputs have not been validated with this mode. Unset `VLLM_JEV_ONLINE_PREFIX_CACHE` and restart to return to ordinary vLLM caching.

The Python tree and component eviction design reference [SGLang's Unified Radix Cache](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/unified_cache/components/README.md), including its Full/Mamba state-validity and cascade-eviction rules.

## HTTP API

| Route | What it does |
|---|---|
| `POST /v1/systemone` | Choice, Noul (yes/no), and Score for all supported models. |
| `POST /plugins/vllm-jev/choice` | One multiple-choice question for Open-Jev and Tiny-Jev text models. |
| `POST /plugins/vllm-jev/batch` | A batch of questions for Open-Jev and Tiny-Jev text models. |

Noul and Score are question types on `/v1/systemone`. Question, option, and context budgets depend on the model; consult its [supported-model section](#supported-models). The separate batch Choice route allows up to 64 questions and 256 candidates total.

All three Laya checkpoints use `/v1/systemone` for Choice, Noul, and Score. Their trained decision heads read the options supplied in each request. The English checkpoint allows 192 tokens for question text and options together; multilingual and typed decisions allow 256.

The English and typed-decision releases contain an out-of-range calibration value for Choice requests with 11 or more options. The pinned Laya runtime clamps it; check confidence on your own data for that case.

### Shared prefixes across System One requests

On Linux, the native Open-Jev-2B, Open-Jev-9B, and OpenJev-0.6B sequence-classification backends accept an optional `cache_salt` on `/v1/systemone`. Reuse the same salt for related requests to let their questions reuse eligible resident prefix blocks:

```json
{
  "state": "The document shared by these requests...",
  "cache_salt": "<random secret assigned by your authenticated gateway>",
  "questions": {
    "urgent": {
      "type": "noul",
      "instructions": "Does this document require urgent action?"
    }
  }
}
```

The salt above is a placeholder. The gateway should generate an unpredictable value for each trusted caller, for example with `secrets.token_urlsafe(32)`, and reuse that value only within its intended trust group. Do not use a user name, document ID, or other guessable identifier.

The salt must be a string of 1–256 characters containing at least one non-whitespace character. Omitting it or sending `null` preserves a fresh namespace for each HTTP request; all questions within that request share the namespace. Invalid salts return HTTP 400 on the pinned vLLM 0.29 server. Other backends, including Tiny-Jev, Mac services, and the Windows CPU service, reject an explicit salt.

Use this option only for requests from the same trusted caller. In a multi-tenant deployment, have the authenticated gateway assign or validate the namespace; do not let public clients choose another tenant's salt. A salt permits sharing but does not replace vLLM's token and model-state cache identities. Different states reuse only their matching prefix, and different salts cannot share prefix blocks.

Prefix caching must be enabled on the server. Reuse depends on matching prefixes, block boundaries, model support, and available cache capacity; it is not guaranteed by a matching salt alone. `metadata.cached_tokens` reports actual cache hits. Questions are submitted independently, so another HTTP request can use a free engine slot while earlier questions are still running. Cancelling one request aborts its own engine requests; vLLM manages the lifetime of blocks still referenced by others.

Cached and fresh-prefix runs can produce slightly different probabilities. In our BF16 rerun, near-tied Score levels sometimes swapped order even though the returned expected score changed only slightly.

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

Start Valen, either vjev model, or RSI-Jev v4.0-VL from the tables above. Put image data URLs in `state.messages`; use a `state` string for text-only requests. RSI-Jev v4.0-VL accepts at most four images and requires its own [context limits](#rsi-jev). For a local PNG:

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

## CUDA BF16 matmul precision

To reduce batch-dependent BF16 GEMM rounding in Kev, opt into cuBLASLt with reduced-precision reduction and split-K disabled:

```bash
VLLM_JEV_BF16_MATMUL=no_splitk vllm-jev serve jaredpalmer/kev-0.8b
```

The default preserves existing PyTorch settings. Set the mode before starting the server; changing it requires a restart. Each mode uses a separate compile cache, so the first start may recompile. CUDA PyTorch must support cuBLASLt and both BF16 reduction controls; tested with PyTorch 2.13.0+cu130 and vLLM 0.29.0.

The setting applies to all models in the worker process and can change serial probabilities and labels. Weights and GEMM outputs remain BF16. It reduces one source of numerical drift; attention/GDN and other kernels can still vary with batching.

In A800 Kev-0.8B stress tests, all three 1,200-request runs completed without unexpected errors. Choice answer flips fell from 196 in default mode to 98 and 96 in two opt-in runs, but the number of affected requests remained similar. These synthetic tests measure repeatability, not accuracy. Decider-2B and This-That also retain differences in the [contributor's tests](https://github.com/mode-io/vllm-jev/pull/5).

RTX 5090 trials reported up to about 5% throughput loss across the tested models. The A800 runs did not establish a speed benefit. Compare latency and throughput on your workload before enabling the mode.

## Updates

### 2026-10-10

- Added declarative `context_order` for instructions, criteria and state, supporting all six permutations across the supported decision-model families. Native and `instructions-first` presets remain compatible. See the [configuration guide](#decision-templates).
- Kept Clef's readout schema after the evidence when adding an instruction preamble, avoiding the answer regression caused by moving the entire schema forward. Image and video inputs retain their native structure.
- Verified 184 regression checks and 27,300 synthetic requests across 91 model/order combinations, plus 240 video and 150 long-context requests. An independent 8,000-request retest at concurrency 32 also passed, including image, video and cancellation recovery checks. Changing order can change answers; native remains the default.

### 2026-10-09

- Added `--decision-template native|instructions-first|path.json`. Native prompts remain the default; [configuration examples](#decision-templates) show how to select a preset or load a JSON file.
- Open-Jev-2B/9B can place instructions before the state, or use custom Choice/Score and Noul text. The optional `{options}` placeholder puts the full fixed candidate list before the changing state while retaining the selected candidate and scoring cue at the end. Thanks to [xxyyy's request in #7](https://github.com/mode-io/vllm-jev/issues/7).
- All model adapters support `state_template` for text/JSON objects or lists and `instruction_template` for explicit text instructions. Chat/media envelopes keep their native structure; unsupported full-layout overrides are rejected.
- Templates are loaded at startup and apply to System One and the available Choice/batch routes. Cross-request prefix reuse still requires matching tokens and a shared, trusted `cache_salt` on supported backends. Restart to change the template; validate custom prompts on your own labelled data because outputs can change.
- Verified 128 regression tests and an independent 2,580-request HTTP stress run, including 80 image requests, concurrency up to 32, cache isolation, and cancellation recovery. Score probabilities and near-tied levels can still vary with batching/cache state.

### 2026-10-08

- Moved RSI-Jev v3.0/v4.0-VL and Clef/Clef-Flash into the main Supported Models table for Linux.
- Added optional [System One shared-prefix caching](#shared-prefixes-across-system-one-requests) for native Open-Jev, contributed by [Shelter Zhou / Sheltercosmo (PR #6)](https://github.com/mode-io/vllm-jev/pull/6).
- Clarified gateway-issued salt requirements and live request validation.
- Rechecked Open-Jev-2B shared-prefix serving with 3,320 requests at up to 32 concurrent clients, covering mixed decision types, cache isolation, and client-disconnect recovery.

### 2026-10-05

- Added the default-off [online prefix-cache experiment](#online-prefix-cache-experimental) with bounded tree growth, hybrid-state eviction, and cancellation/reset cleanup.
- Fixed full-context pooling and partial-page continuation limits in the experimental cache path.
- Added optional [CUDA BF16 precision control](#cuda-bf16-matmul-precision), contributed by [Qiao / wocqcm2 (PR #5)](https://github.com/mode-io/vllm-jev/pull/5).
- Fixed precision-mode consistency during repeated initialization; mode changes require a server restart.

### 2026-10-04 · v0.3.1

- Fixed RSI-Jev requests at the context limit, cached model selection, and invalid media handling.
- Fixed the Linux helper installer and simplified the setup and model guides.

### 2026-10-04 · v0.3.0

- Added [Clef 27B](#clef) alongside Clef-Flash 9B, with text, image, and video decisions on Linux.

### 2026-10-04 · v0.2.0

- Added experimental Linux serving for [RSI-Jev](#rsi-jev) v3.0/v4.0-VL by [Shanghua Gao (PR #2)](https://github.com/mode-io/vllm-jev/pull/2) and [Clef-Flash](#clef) by [Arcobalneo (PR #4)](https://github.com/mode-io/vllm-jev/pull/4).
- Added aggregate request limits for Clef and fixed platform and reference-test issues.
- Added optional [RSI-Jev shared-state tokenization](#rsi-jev) to reduce preparation time for multi-question requests.

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

## Ascend development port

See [Ascend setup and acceptance matrix](ascend.md). Existing model listings
above do not imply NPU validation. Use a matched native Ascend runtime and
`--backend ascend`; CUDA-specific graph/cache hooks are not silently reused.
