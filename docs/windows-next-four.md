# Decision, Bruv and RSI on Windows CPU

These experimental adapters use the unified `vllm-jev serve` command and `/v1/systemone`. They accept text and JSON. They do not provide image/video inference, CUDA execution, RadixFlow, or cross-request prefix caching. See the [general Windows service guide](guide.md#windows-cpu-experimental) for queueing and HTTP behavior.

Implementation and released-weight acceptance are separate. The current acceptance results are recorded below; a model is not added to the validated-model total until its real tests pass.

## Isolated installations

Keep these environments separate from the retained legacy and modern environments. The existing profiles still pin CPU PyTorch 2.8.0. Bruv uses 2.9.1; RSI uses 2.13.0. All commands below assume native Windows x64 and Python 3.12, run from this checkout.

| Models | Extra | CPU PyTorch | Transformers | Additional runtime |
|---|---|---|---|---|
| Decision-0.8B Q8_0 / Decision-4B Q4_K_M | `windows-modern` | 2.8.0 | 5.17.0 | Fixed official llama-cpp-python 0.3.35 CPU wheel |
| Bruv1-0.8B HF | `windows-bruv` | 2.9.1 | 5.17.0 | Eager PyTorch, FP32 |
| RSI v1.0-0.8B | `windows-rsi` | 2.13.0 | 5.17.0 | PyTorch SDPA, FP32 |

Bruv and RSI also pin PEFT 0.21.0, Accelerate 1.15.0, NumPy 2.5.4, tokenizers 0.23.3 and Jinja2 3.1.6. The runtime verifies the selected model's dependency profile before loading weights. Installing a different profile in the same environment does not preserve the earlier profile.

```powershell
uv venv --python 3.12 .venv-windows-decision
uv pip install --python .venv-windows-decision\Scripts\python.exe torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv-windows-decision\Scripts\python.exe -e ".[windows-modern]" tokenizers==0.23.3 jinja2==3.1.6

uv venv --python 3.12 .venv-windows-bruv
uv pip install --python .venv-windows-bruv\Scripts\python.exe torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv-windows-bruv\Scripts\python.exe -e ".[windows-bruv]"

uv venv --python 3.12 .venv-windows-rsi
uv pip install --python .venv-windows-rsi\Scripts\python.exe torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv-windows-rsi\Scripts\python.exe -e ".[windows-rsi]"
```

Decision additionally requires the official `llama_cpp_python-0.3.35-py3-none-win_amd64.whl` from the [0.3.35 CPU release](https://github.com/abetlen/llama-cpp-python/releases/tag/v0.3.35). Verify SHA256 `31590ea000d5aff6f05f1e428048e72318a83709288159a5bd4dabec530080bb` before installation; a wheel with the same version but different native bytes is rejected. Follow the [fixed wheel installation procedure](guide.md#validate-decider-4b-q4_k_m-on-windows), substituting `.venv-windows-decision` for the environment name. Decision and Mapika Decider are different model families.

After installing the appropriate profile and, for Decision, the verified wheel, select one model:

```powershell
.\.venv-windows-decision\Scripts\vllm-jev.exe serve evalengine/decision-0.8b-gguf --cpu-threads 4
.\.venv-windows-decision\Scripts\vllm-jev.exe serve evalengine/decision-4b-gguf --cpu-threads 4
.\.venv-windows-bruv\Scripts\vllm-jev.exe serve bvolpato/bruv1-0.8b --cpu-threads 4
.\.venv-windows-rsi\Scripts\vllm-jev.exe serve shgao/rsi-jev-v1.0-qwen3.5-0.8b --cpu-threads 4
```

The first preparation downloads fixed inference data and verifies every required SHA256. Serving does not execute downloaded Python or enable `trust_remote_code`. A fully prepared checkpoint directory can replace the model ID, including a local path containing Chinese characters and spaces. Arbitrary checkpoints and quantizations are not accepted. Use only one model at a time on memory-constrained hosts.

## Fixed artifacts and scoring contracts

| Model | Pinned release | Selected data |
|---|---|---|
| `evalengine/decision-0.8b-gguf` | `ce993792ad283053004bf0230b1709ff7351bb24` | `decision-0.8b-Q8_0.gguf`, SHA256 `efa1614fe5c7d84acb6b8acdf9fd0dda4a26acf110302732baa573d3332a4fef` |
| `evalengine/decision-4b-gguf` | `76b147cb5da4585aed9ae026dda18a3c9d78305b` | `decision-4b-Q4_K_M.gguf`, SHA256 `d67a3c72a3665f8add74a2f86c900f62d3080d3bcb554e4e39df09c6f046d0c7` |
| `bvolpato/bruv1-0.8b` | `7d3660db482cae2afb1fe1ee091a659444791c13` | Merged HF weights, SHA256 `0da0a6084adc748d61952774f21ddc4e55cfa9f760d948588ecad6096f4b4589` |
| `shgao/rsi-jev-v1.0-qwen3.5-0.8b` | `f9248caceb89caf2e6c968ea33bf0d6eb7f957b0` | Released FP32 tower/scorer plus the frozen embedding from `Qwen/Qwen3.5-0.8B-Base@dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68` |

All four accept only the default/native template, 1–4 CPU threads, 1–64 questions and at most 256 total candidates. Each complete question prompt is limited to 1,024 tokens; oversize requests are rejected before the first model forward, without truncation. The shared service limits state text to 50,000 characters. Each question uses joint probabilities at temperature 1; Score is the expected zero-based level over the supplied ordered descriptions.

| Family | Choice | Noul | Score | Additional rules |
|---|---|---|---|---|
| Decision | 2–24 A–X candidates | Ordered false/true options | 2–10 joint levels | `isolated: true` rejected; complete candidate logits, not top-k approximations |
| Bruv | 2–16 A–P candidates | No/Yes, with optional descriptions | 2–10 joint levels | Any explicit `isolated` field rejected; instructions must be nonempty text |
| RSI | 2–80 named candidates | false/true with No/Yes defaults | 2–10 joint levels | Any explicit `isolated` field rejected; JSON instructions/descriptions are serialized deterministically |

Decision follows the published JSON task protocol and the GGUF's no-thinking chat template, including its empty `<think>` section. The adapter reads every candidate letter's last-position logit and normalizes them in FP32. It clears recurrent/KV memory between rows. This same-quantization check does not establish equivalence to BF16 model-card results.

Bruv preserves the author's `kevala-direct-options-v1` encoder and restricted 16-row LM-head readout. A Choice key and nonempty description become `key: description`; absent/empty descriptions use the key. Noul descriptions are attached to No/Yes. Both product and reference explicitly use FP32, eager attention and `use_cache=False`. The author's CPU evaluator defaults to BF16; these FP32 results are not a reproduction of published BF16 performance.

RSI preserves the release's state-first block tokenization, mean pooling, 80 padded option slots and four-head `option_xattn` scorer. The product materializes only the text tower, frozen embedding and scorer. Its independent reference uses the unchanged original loader and may therefore need more peak memory. `use_cache=True` applies to a single fresh forward; no cache is reused across requests. This is the v1.0 text release, separate from the existing v3.0 text and v4.0-VL adapters.

## Independent acceptance

The checked-in drivers run reference and product processes sequentially, verify exact source/runtime identities, compare complete probability vectors, exercise model-ID and Unicode local-path starts, check invalid inputs and concurrent requests, sample memory, and clean up only their recorded process trees. Cleanup is a forced owned-process termination check, not a graceful-shutdown test. Fixed synthetic cases establish numerical/integration evidence, not task accuracy or throughput.

| Family | Driver | Independent reference |
|---|---|---|
| Decision | [`check_windows_decision_gguf.py`](../scripts/check_windows_decision_gguf.py) | [`windows_decision_gguf_reference.py`](../scripts/windows_decision_gguf_reference.py): separate implementation of the published protocol, GGUF embedded Jinja template and high-level llama-cpp evaluation; no original public scorer SDK was available |
| Bruv | [`check_windows_bruv.py`](../scripts/check_windows_bruv.py) | [`windows_bruv_reference.py`](../scripts/windows_bruv_reference.py): unchanged `data.py`/`model.py` from `bvolpato/bruv@d10751e8fd3c2a917fea81fb350593eeed3c482d`; original encode/collate/score functions |
| RSI | [`check_windows_rsi.py`](../scripts/check_windows_rsi.py) | [`windows_rsi_reference.py`](../scripts/windows_rsi_reference.py): seven unchanged Python files under the fixed HF release's `code/`; original load/encode/predict/forward |

Keep author sources and reference interpreters outside the product environment. The drivers verify independent hash pins and forbid product imports in the reference. They require prepared, verified input data; validation does not download weights. Install `psutil` and `requests` in the product validation environment and use each driver's `--help` for its source, reference, wheel and output arguments. Bruv uses a minimal inference reference; the author's full CUDA training project is not installed. RSI's official Windows CPU PyTorch 2.13.0 wheel satisfies the current version floor without editing upstream constraints.

The watchdog stops only owned processes if available system RAM falls below 1.5 GiB. Launch gates are engineering safeguards, not measured minimum requirements. Decision defaults to 4 GiB available RAM/6 GiB commit headroom for 0.8B and 6/8 GiB for 4B; Bruv defaults to 6/8 GiB. Reserve extra headroom for RSI's unchanged author loader.

## Acceptance results

All four models passed real-weight validation on the local Windows CPU host:

| Model | Installed CLI starts | Successful HTTP requests | Rejected inputs | Maximum observed answer difference | Status |
|---|---|---|---|---|---|
| Decision-0.8B Q8_0 | 2 | 16 | 28 | 0 | Passed |
| Bruv1-0.8B FP32 | 2 | 14 | 40 | 0 | Passed |
| Decision-4B Q4_K_M | 2 | 16 | 28 | 0 | Passed |
| RSI v1.0-0.8B FP32 | 2 | 14 | 32 | 0 | Passed |

Decision's reference renders the embedded template and reads actual context logits after high-level evaluation; it does not read an unpopulated `scores` array. Both independent paths explicitly agree on 25 context parameters, including disabled flash attention, 2,048-token context and batch limits, four CPU threads and no GPU layers. Initial diagnostic failures were retained, then all six independent reference cases and both CLI routes were rerun. The probability comparison tolerance remained `1e-5`; checking an FP32 probability sum allows `1e-6` rounding error without renormalizing the returned values.

Bruv's five original-author cases cover eleven encoded question rows. Every prompt token matched, and its two CLI routes passed the original `1e-5` tolerance with zero observed difference. The ten sequential requests took 3.06–6.08 seconds; the highest sampled product-process peak working set was 3.35 GiB and peak commit 4.19 GiB. Minimum sampled available system RAM across reference and product runs was 4.71 GiB. These values describe the tested short cases on a Core Ultra 5 125H with other applications active; they are not minimum memory requirements.

This increment reached **22 validated Windows CPU checkpoints** on 2026-10-15; the [current scope](project-onboarding.md#validated-model-scope) is 24 after Tiny-Jev-1.7B and JevK5-9B acceptance. All previous 18 models passed fresh real product regression: 28 CLI starts, 124 successful requests and 176 invalid-input checks, using verified historical author baselines rather than rerunning those author models. All 12 retained environments' distribution metadata remained unchanged. The final lightweight suite passed 1,141 unique tests with no failures or skips. See the [complete regression scope](guide.md#current-windows-regression-scope) for provenance and cleanup boundaries. All results are local to this checkout and do not imply a release or validation on other operating systems.

Decision-4B: the observed sequential requests took 7.22–15.18 seconds, excluding loading. The highest sampled product-process peak working set was 4.68 GiB and peak commit 6.13 GiB; minimum sampled available system RAM across recorded reference, encoding and product processes was 7.57 GiB. These are observations under the tested workload, not minimum requirements.

RSI v1.0-0.8B: the observed sequential requests took 1.75–6.37 seconds, excluding loading. The highest sampled product-process peak working set was 3.62 GiB and peak commit 6.01 GiB; minimum sampled available system RAM across recorded reference, encoding and product processes was 7.33 GiB. These are observations under the tested workload, not minimum requirements.

RSI additionally passed eleven exact encoding comparisons against the original author forward inputs, including token IDs, option spans, decision position and option order. Its released-weight run used an 8 GiB available-memory launch gate. The separate encoding probe exited before model serving, and the orchestration process remained free of Torch imports. The original reference completed before a driver hash-audit path-type error; after fixing that error, the final product attempt reused the identical verified reference output. The earlier process evidence is preserved and linked by hash, rather than recorded as a second reference run. All recorded reference, probe and service process trees exited, and their private workspaces were removed.
