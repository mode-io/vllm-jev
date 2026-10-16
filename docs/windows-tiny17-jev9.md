# Tiny-Jev-1.7B and JevK5-9B Q5_K_M on Windows CPU

Both models passed independent released-weight acceptance on native Windows
CPU on **2026-10-16**, in the local `vllm-jev-wzh` branch. The final runs covered
**4 installed CLI starts, 28 valid requests, and 60 invalid-input rejections**.
Together with the previous 22 accepted models, this brings the validated
Windows total to **24**. This is local validation, not a published release.

See the [Windows installation overview](guide.md#windows-cpu-experimental)
and [current regression scope](guide.md#current-windows-regression-scope).

Both adapters use `vllm-jev serve` and `/v1/systemone`, with text/JSON state and
Choice, Noul, and Score answers. The profile is experimental native Windows x64
CPU execution, without CUDA, image/video inference, RadixFlow, or cross-request
cache reuse. Author and product processes must run sequentially.
This addition validates Windows CPU only; it does not add a tested Linux or Mac
claim for either model. `alibiserikbay/JevK5-9B` is a different checkpoint from
`ZefanCai/Open-Jev-9B`, which remains outside the Windows adaptation scope.

## Isolated installations

Use Python 3.12. Keep new product environments separate from each other and
from author-reference environments; retain all previous model environments.
Commands below are run from the repository checkout. These are setup
instructions, not a record that every command has already been executed.

| Role | CPU PyTorch | Transformers | Other fixed components |
|---|---|---|---|
| Tiny product: `windows-legacy` | 2.8.0+cpu | 5.10.2 | PEFT 0.19.1; Accelerate 1.13.0; NumPy 2.5.4; tokenizers 0.22.2; safetensors 0.8.0 |
| Tiny independent author reference | 2.6.0+cpu | 5.17.0 | tokenizers 0.23.3; safetensors 0.8.0; NumPy 2.5.4 |
| JevK5-9B product: `windows-modern` | 2.8.0+cpu installed for package compatibility; not used/imported for GGUF inference | 5.17.0 installed profile | PEFT 0.21.0; Accelerate 1.15.0; NumPy 2.5.4; tokenizers 0.23.3; Jinja2 3.1.6 |
| JevK5-9B author reference | Not installed/used | Not installed/used | Python 3.12 standard library, pinned original SDK files and Windows llama.cpp b11010 |

Both product acceptance drivers also require `psutil` for ownership tracking
and memory monitoring. The Tiny author reference uses native Windows process
counters and does not require psutil.

```powershell
uv venv --python 3.12 .venv-windows-tiny17
uv pip install --python .venv-windows-tiny17/Scripts/python.exe torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv-windows-tiny17/Scripts/python.exe -e ".[windows-legacy]" tokenizers==0.22.2 safetensors==0.8.0 psutil==7.2.2

uv venv --python 3.12 .venv-windows-tiny17-author
uv pip install --python .venv-windows-tiny17-author/Scripts/python.exe torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv-windows-tiny17-author/Scripts/python.exe transformers==5.17.0 tokenizers==0.23.3 safetensors==0.8.0 numpy==2.5.4

uv venv --python 3.12 .venv-windows-jev9
uv pip install --python .venv-windows-jev9/Scripts/python.exe torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv-windows-jev9/Scripts/python.exe -e ".[windows-modern]" psutil==7.2.2

uv venv --python 3.12 .venv-windows-jev9-author
```

The four exported requirements files in the external validation archive record
the complete isolated environments used for this run. They are local audit
evidence and are not shipped with this repository; the installation commands
and checked-in dependency profiles above are the public reproduction entry
points. An exact recreation of all transitive packages from the historical run
requires access to those exports.
The JevK5 author requirements file is empty because its retained reference uses
standard-library Python plus the verified SDK and native executable.

The exports are `tiny-runtime-requirements.txt`,
`tiny-reference-requirements.txt`, `jev9-runtime-requirements.txt`, and
`jev9-reference-requirements.txt`.

## Fixed artifacts and preparation

| Model/API identity | Artifact revision | Weight artifact |
|---|---|---|
| `lostargon/Tiny-Jev-1.7B` | `c41bcf3f191407c779bd2ebce1b992dd1469437f` | `model.safetensors`; 3,440,104,404 bytes; SHA256 `49ff7a60951abe35631b9f0632b8250acdb9142f79f39a57f56aa42befac7a3b` |
| `alibiserikbay/JevK5-9B` | `alibiserikbay/JevK5-GGUF@ec67b0bfce5119a8b11a2cdb430bb43e3fa3e82a` | `jevk5-9b-v0.3.3-Q5_K_M.gguf`; 6,467,969,504 bytes; SHA256 `da4a4becd04c3beabd4cefbd09b546ce2648740d923adc8d14ae1d0cb921486e` |

Tiny preparation verifies all six required source files, including the
unchanged author Python file, configuration, tokenizer, and template. Product
serving never imports that downloaded Python file.
The fixed releases are available in the [Tiny-Jev-1.7B repository](https://huggingface.co/lostargon/Tiny-Jev-1.7B/tree/c41bcf3f191407c779bd2ebce1b992dd1469437f)
and [official JevK5 GGUF repository](https://huggingface.co/alibiserikbay/JevK5-GGUF/tree/ec67b0bfce5119a8b11a2cdb430bb43e3fa3e82a).

JevK5 preparation verifies the selected GGUF, README, SHA256SUMS, and every
required executable/DLL from the official Windows CPU x64 llama.cpp b11010
release. The runtime archive SHA256 is
`3011251607251eca68ae67e35512364d9943d862ff598599ff9a4888c1753008`.
This is the standalone llama-server runtime, not a llama-cpp-python wheel.

The CLI automatically prepares missing artifacts when serving. For a separate
download-and-verification step without loading either model, call the existing
Python preparation API:

```powershell
.venv-windows-tiny17/Scripts/python.exe -c "from pathlib import Path; from vllm_jev.windows_models import prepare; print(prepare('lostargon/Tiny-Jev-1.7B', Path('./.local/windows-tiny17-data')))"
.venv-windows-jev9/Scripts/python.exe -c "from pathlib import Path; from vllm_jev.windows_models import prepare; print(prepare('alibiserikbay/JevK5-9B', Path('./.local/windows-jev9-data')))"
```

Serve one model at a time:

```powershell
.venv-windows-tiny17/Scripts/vllm-jev.exe serve lostargon/Tiny-Jev-1.7B --workspace ./.local/windows-tiny17-data --cpu-threads 4
.venv-windows-jev9/Scripts/vllm-jev.exe serve alibiserikbay/JevK5-9B --workspace ./.local/windows-jev9-data --cpu-threads 4
```

A verified checkpoint directory may replace the model ID, including a Unicode
path with spaces. JevK5's local directory must contain its pinned `runtime/`
files as well as the GGUF and release documents. A generic GGUF filename or
different quantization is not an accepted substitute.

## Scoring and limits

Tiny 1.7B runs the Qwen3 text backbone and scalar head in explicit FP32 using
SDPA, temperature **1.15**, and `use_cache=False`. It reads the final cue token
immediately before each option marker, applies FP32 layer normalization, and
returns normalized candidate probabilities. Choice confidence is the selected
probability; Noul returns the probability of `yes`; Score is the expected
zero-based level.

The released 1.7B checkpoint contains a BF16 backbone and an FP32 scalar head:
1,720,030,208 BF16 parameters and 2,049 FP32 parameters. Its approximately
3.204 GiB weight file therefore becomes approximately 6.408 GiB of FP32
parameters before activations and runtime overhead. This is FP32 inference,
not quantization.

The 1.7B loader constructs parameters on the meta device while retaining
nonpersistent rotary buffers on CPU. It opens the checkpoint separately for
each tensor, converts it with `to(dtype=torch.float32, copy=True)`, closes that
source mapping, and assigns the resulting owned tensors to the backbone.
Explicit copying also matters for the already-FP32 head: a no-op dtype
conversion could keep the entire checkpoint mapping alive through a tiny head
tensor. Per-tensor mapping limits simultaneous mapped source pages, while meta
initialization and assignment avoid a second complete FP32 parameter copy.
All materialized parameters and buffers must be on CPU. The previously
accepted Tiny 0.6B loading path is preserved. Measured peaks are recorded below;
parameter size alone is not a RAM budget.

Tiny allows 1–255 Choice candidates, 2–10 distinct Score descriptions, 1–64
questions, and at most 256 candidates per request. A one-candidate Choice
returns certainty without a model forward. Each nontrivial question is encoded
within 4,096 tokens. The original state-only head/tail truncation behavior is
retained: when necessary, approximately 70% of the available state-token budget
keeps the beginning and the remainder keeps the end. Oversized question/schema
content that cannot fit the complete prompt is rejected before model inference.
The shared service separately caps state text at 50,000 characters.

JevK5-9B runs official Q5_K_M on CPU with zero GPU layers. It preserves the
author's single-pass A–P label scoring: **temperature 1.316**, top-40 log
probabilities, and the original missing-letter floor. Returned probabilities
cover all supplied candidates, but they retain the author's top-k approximation
when a label is absent from the returned top-40 set. This does not establish
equivalence to full-vocabulary logits or BF16 model-card results.

JevK5 accepts 2–16 Choice candidates, 2–10 Score levels, 1–64 questions, and at
most 256 candidates per request. Its complete per-question prompt limit is
**1,024 tokens**, including state, instructions, choices, and template; oversize
prompts are rejected before inference rather than truncated. The profile uses
the default/native template and 1–4 CPU threads. The author's 9B knockout
temperature is 1.05, but knockout inference and more than 16 choices are outside
this bounded single-pass adapter.

## Independent real-weight acceptance

Both drivers require prepared artifacts; neither installs packages nor
downloads weights. They run the original reference and product sequentially,
check source/runtime identities and full probability vectors, exercise
installed CLI model-ID and Unicode-path launches, reject unsupported input,
observe health responses during concurrent requests, and terminate only owned
process trees. Owned-process cleanup is not proof of graceful shutdown.

Tiny's original reference uses the unchanged
`modeling_tiny_jev.TinyJevModel.decide` from the pinned release, one question per
forward. Passive hooks record author token IDs, option positions, calibrated
logits, and FP32 probabilities. Product tokenization is compared in a separate
process; the author process forbids product imports. Runtime `use_cache=False`
is explicit and source files remain unchanged. The comparison tolerance is
`1e-5`; it is not automatically relaxed on failure.

```powershell
.venv-windows-tiny17/Scripts/python.exe scripts/check_windows_tiny.py --source ./.local/windows-tiny17-data/checkpoint/lostargon/Tiny-Jev-1.7B --reference-python .venv-windows-tiny17-author/Scripts/python.exe --workspace ./.local/windows-tiny17-check/work --output ./.local/windows-tiny17-check/result.json --console-entrypoint .venv-windows-tiny17/Scripts/vllm-jev.exe --threads 4
```

JevK5's reference executes unchanged `jevk5/__init__.py`, `prompt.py`, and
`gguf.py` from [original SDK commit
`f26426d16f59e8bbe1470e5b162cc89329e29b29`](https://github.com/allebee/jevk5/tree/f26426d16f59e8bbe1470e5b162cc89329e29b29). Prepare that verified SDK directory
separately from both product and checkpoint directories. The script checks its
independent hashes before executing it. The original SDK creates prompts and
applies the same pinned llama-server readout and calibration.

```powershell
git -c core.autocrlf=false clone https://github.com/allebee/jevk5 ./.local/windows-jevk5-sdk
git -C ./.local/windows-jevk5-sdk -c core.autocrlf=false checkout f26426d16f59e8bbe1470e5b162cc89329e29b29
.venv-windows-jev9/Scripts/python.exe scripts/check_windows_jevk5_gguf.py --model alibiserikbay/JevK5-9B --source ./.local/windows-jev9-data/checkpoint/alibiserikbay/JevK5-9B --sdk-source ./.local/windows-jevk5-sdk --reference-python .venv-windows-jev9-author/Scripts/python.exe --workspace ./.local/windows-jev9-check/work --output ./.local/windows-jev9-check/result.json --console-entrypoint .venv-windows-jev9/Scripts/vllm-jev.exe --threads 4
```

Both drivers support `--reference-only` and `--reference-report` for staged
execution. Reused reference reports must match the current fixtures, runtime
settings, source pins, and reference-runner hash.

Both default launch gates require **10 GiB available physical RAM**. Tiny 1.7B
additionally requires **20 GiB commit headroom**, raised after observing a
16.724 GiB process commit high-water mark; JevK5-9B retains a **12 GiB commit
headroom** gate. The watchdog aborts owned inference processes if available RAM
falls below **1.5 GiB**. These numbers are conservative experiment budgets,
not measured minimum requirements or guarantees. Windows commit records the
backing obligation against RAM/pagefile capacity, not physical resident RAM;
commit headroom does not substitute for available physical RAM. Run one model
at a time.

## Acceptance results

| Model | Independent real author run | Installed CLI starts | Probability/token parity | Peak working set / commit | Final status |
|---|---|---|---|---|---|
| Tiny-Jev-1.7B FP32 | Fresh original author run, explicitly reused after the loader fix | 2 starts; 16 valid requests; 42 HTTP 422 rejections | Max absolute answer difference `7.753524231546294e-7` at `1e-5`; 10 exact token/marker sequences | Listening-process peak working set 6.822 GiB; lifetime peak commit 16.724 GiB | Passed with 10 GiB available RAM / 20 GiB commit launch gate |
| JevK5-9B Q5_K_M | Fresh original SDK run, followed by product runs in the same acceptance invocation | 2 starts; 12 valid requests; 18 HTTP 422 rejections | Max absolute answer difference `0` at `1e-5` | Native llama-server peak working set 5.719 GiB; lifetime peak commit 0.791 GiB | Passed |

The final Tiny report confirms exact equality for 10 author/product token sequences
and marker-position arrays; one singleton Choice follows the product's
deterministic shortcut. The subsequent released-weight comparison separately
checks full answer probabilities. Both CLI routes passed concurrency/health
checks, retained source hashes, and stopped their owned process trees. The
minimum sampled host available RAM was **4.207 GiB**. Its two CLI starts became
ready in **16.14 and 24.54 seconds**. Sequential nontrivial cases took
approximately **1.17–2.96 seconds** per HTTP request; the singleton shortcut took
approximately **0.02 seconds**. These timings exclude loading. Working set and committed
memory are different counters: 16.724 GiB is a process lifetime commit
high-water mark, not measured physical RAM use or a simultaneous process-tree
peak. These fixed cases establish local numerical parity, not decision quality
or a maximum-capacity benchmark.

The JevK5-9B run passed both installed CLI routes, full probability comparisons,
concurrency/health checks, unchanged source and SDK hashes, and owned-process
cleanup. Its eight sequential product requests across four fixed cases took
**21.62–46.31 seconds**, excluding loading. The largest observed Python service
peak working set was approximately **0.0715 GiB**, separately from the native
llama-server's **5.719 GiB** model working set; the Python figure is not the
model's memory use. Across the sequential author and product stages, minimum
sampled host available RAM was **6.769 GiB**. These are per-process lifetime
high-water counters and sampled host availability, not a measured simultaneous
process-tree peak, a throughput benchmark, or minimum memory requirements.

The first Tiny real-weight attempt is retained as a **failed attempt**: the
first CLI start passed its numerical checks, but the second start reached the
available-memory watchdog. The FP32 head's retained checkpoint mapping was
then removed by the loader change above. Passing part of that first attempt
was not counted as acceptance; the complete run after the fix passed and was
retained as attempt 2. The final repeat also passed with the increased
commit-headroom gate; this changed the experiment precheck, not the model
arithmetic. The totals above count only the final product runs, not the earlier
attempts.
The fresh independent author report from this adaptation was explicitly reused
after the driver's provenance checks confirmed unchanged fixtures, source,
runtime settings, and reference-runner identity. It was not rerun during the
final product invocation. No numeric tolerance or memory guard was weakened.

The existing **22 accepted models** passed a fresh real-weight regression:
**36 CLI starts, 184 valid requests, and 304 invalid-input
rejections**. Metadata snapshots for **19 retained environments** were
unchanged. These totals describe the previous models, not the two new
additions, and do not replace their independent author comparisons.
The eight legacy models each used one module CLI start; the other fourteen
models each used both installed CLI routes. Twenty-one models reused verified
historical reference reports. JevK5 4B had a fresh original-author SDK run
because this family's reference runner changed; its old report was not accepted
as a substitute for that updated runner.

The adaptation-stage lightweight run passed 1,185 tests before the final Tiny
loader refinement; focused JevK5/Tiny driver and loader checks followed. Its
aggregate report passed all nine completion gates, including both new
real-weight acceptances, previous-model regression, environment preservation
and process cleanup.

At final closeout on 2026-10-16, the complete Windows lightweight selection was
rerun after the loader refinement using the checked-in four-profile runner:
**1,186 unique tests passed**, with no failures, errors, skips or duplicated
IDs. This later suite is separate from the real-weight counts above. See the
[current regression scope](guide.md#current-windows-regression-scope) for
per-profile counts and the [onboarding test command](project-onboarding.md#test-and-verify)
for reproduction. Historical overlapping checks are not added to this total.

## Retained local evidence

The validation artifacts remain outside the repository in a local audit
archive, referred to here as `WINDOWS_VALIDATION_ROOT`. This is a documentation
label, not an application environment variable or a required checkout path.
Raw reports, downloaded weights, installed environments and historical attempts
are deliberately not repository inputs. The checked-in drivers generate new
reports in the paths supplied with `--output` and `--workspace`.

The retained archive contains:

- `validation-summary.json`: final passed status, all completion gates, source
  report hashes, 24-model count, and PID/creation-time audit.
- `tiny17/result.json` and `tiny17/result-author.json`: final product run and
  provenance-checked original-author baseline; `tiny17/attempt1/` retains the
  initial failed attempt, and `tiny17/attempt2/` retains the first successful
  loader-fix run before the stricter commit gate.
- `jev9/acceptance.json` and `jev9/acceptance-author.json`: final product and
  fresh independent original-SDK runs.
- `regressions/validation-summary.json`: all 22 previous models, 36 CLI starts,
  184 valid requests, and 304 rejections.
- `retained-environments-before.json` and `retained-environments-after.json`:
  unchanged distribution metadata in 19 old environments. This audit does not
  hash every installed library byte.

The final read-only process audit checked **192 recorded PID/creation-time
identities** and found none still running. This confirms owned-process cleanup;
it does not establish graceful shutdown or real-weight client-disconnect
behavior. No user applications were stopped.
