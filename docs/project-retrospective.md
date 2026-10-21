# Project Retrospective: Windows CPU Adaptation

## Project Goal and Final Outcome

The goal was to make selected Jev decision checkpoints usable through this
project's unified entrypoint on a local Windows CPU computer, without changing
retained model environments or claiming support based only on code inspection.
The completed scope is **24 checkpoints** in `vllm-jev-wzh`, accepted with real
released weights by 2026-10-16. The [onboarding scope table](project-onboarding.md#validated-model-scope)
records exact identities, precisions and dependency profiles.

The implementation provides text/JSON Choice, Noul and Score through
`vllm-jev serve` and `/v1/systemone`. It includes pinned source/runtime
verification, four isolated dependency profiles, family-specific CPU scoring,
HTTP limits, independent reference drivers and regression tests. The final
two-model increment covered 4 installed CLI starts, 28 valid requests and 60
rejected inputs. The previous 22 models passed fresh product regression across
36 CLI starts, 184 valid requests and 304 rejections. See [current evidence
and reference reuse](guide.md#current-windows-regression-scope) for the exact
boundaries and lightweight-test results.

Success meant actual inference with the accepted artifact, unchanged source
hashes, full probability comparisons at the specified tolerance, rejection of
unsupported inputs and cleanup of owned validation processes. Numerical parity
does not measure downstream task accuracy. This closes this Windows adaptation
scope, not the Linux/Mac project or its release process. Eleven entries in the
original 26-model catalogue remain outside this Windows acceptance; nine
additional releases account for the total of 24.

### Closeout Checks

The final closeout reran the complete four-profile Windows lightweight suite:
**1,186 unique tests passed**, with no failures, errors, skips or duplicated IDs.
The [user guide](guide.md#current-windows-regression-scope) records its profile
counts and distinguishes these checks from the earlier real-weight acceptances.
The portable runner is checked in; its logs, XML and summary remain in an
external closeout evidence archive, outside Git.

Wheel and source-distribution builds also passed. Package inspection confirmed
all 21 Windows modules, all 8 attributed Mica vendor files byte-for-byte, the
license files and all 28 Windows validation Python scripts in the source
distribution. Windows dependency metadata excludes vLLM. A separate environment
installed the wheel and passed `vllm-jev --help`, `vllm-jev serve --help` and
resource/import checks. These are package-content and entrypoint checks, not
model inference in that build environment or a package publication.

The raw `build-run.json`, `package-audit.json` and `windows-suite/summary.json`
are retained outside the repository. The narrow Mica `.gitattributes` rule
preserves the attributed vendored bytes under Windows line-ending conversion.

An additional, non-Windows test audit is **not a passing cross-platform suite**.
Running 11 existing test files in separate processes produced 168 passes,
41 skips and 9 failures; all 9 failures require the unavailable `vllm` package.
The changed decision-template tests passed all 25 cases, and the instrumented
System One HTTP/cancellation tests passed all 34. An earlier combined attempt
also encountered missing image dependencies and an existing JevAny test stub
that affected later vLLM availability checks. The RSI native test file was not
included in the isolated rerun because it needs vLLM/vision dependencies.
These reports are retained as `shared-tests-with-pillow.*` and
`shared-isolated/summary.json`. Test-only Pillow was supplied from a separate
directory; retained model environments were not modified. Linux/GPU and Mac
execution still require their own hosts and are not established by this closeout.

## Key Technical Decisions

| Context and constraint | Decision | Alternative considered | Reason and consequences |
|---|---|---|---|
| The existing Linux service relies on vLLM/GPU behavior | Add an explicit native Windows CPU dispatch behind the same CLI and System One API | Require WSL/Linux or attempt to port the entire vLLM stack | Enables locally testable CPU inference; Windows does not inherit CUDA, RadixFlow, multimodal or vLLM performance claims. See [`cli.py`](../vllm_jev/cli.py) and [`windows_server.py`](../vllm_jev/windows_server.py). |
| Model families depend on incompatible library versions and CPU behavior | Keep four exact dependency profiles and separate product/reference environments | Upgrade all environments to the newest packages | Preserves previous baselines and exposes wrong environments before model loading. See [`windows_dependencies.py`](../vllm_jev/windows_dependencies.py). |
| Downloaded repositories may change or contain executable model code | Pin and verify source/runtime files; implement product scoring locally | Accept arbitrary checkpoints or enable remote-code execution | Defines a narrow auditable acceptance boundary; new revisions need explicit work and tests. See [`windows_models.py`](../vllm_jev/windows_models.py). |
| Larger FP32 models exceed comfortable local memory | Accept selected official/published GGUF formats with fixed CPU runtimes | Claim full-precision equivalence or accept arbitrary quantizations | Makes selected 4B/9B releases testable locally; same-quantization parity does not establish BF16 quality equivalence. |
| Independent execution can require different packages and extra memory | Run reference and product sequentially, with provenance-checked report reuse where explicitly recorded | Run both models concurrently or compare product code against itself | Keeps memory bounded and distinguishes fresh author execution from historical baseline reuse. See the [acceptance drivers](project-onboarding.md#directory-map-and-key-entry-points). |
| CPU inference is blocking and expensive | One worker with a bounded queue and cancellation accounting | Unbounded requests or releasing a slot immediately on disconnect | Keeps health responsive and prevents cancellations from bypassing capacity; an active forward is not interrupted by client disconnect. |

## What Went Well

- Family-specific adapters reused the CLI/API boundary while preserving
  different tokenization, temperatures and readout rules. Exact release and
  precision identity are visible in health/metadata and documented separately.
- Source hashes and independent reference checks made failures diagnosable.
  Changed reference-runner identity required a fresh JevK5 4B author run instead
  of silently accepting a stale report.
- Isolated environments protected completed work. Before/after distribution
  metadata matched in 19 retained environments during the final adaptation
  regression. This verifies metadata preservation, not every installed file.
- Memory guards caught a real Tiny-Jev-1.7B loader problem before completion
  was claimed. The failed attempt remains distinct from subsequent passed runs.
- Test doubles and small random models covered service and numerical boundaries
  quickly, while separate released-weight drivers supplied the real-model
  evidence those tests cannot provide.

## Problems, Causes, and Solutions

| Problem and impact | Contributing cause | Implemented solution | Verification and current status |
|---|---|---|---|
| Tiny 1.7B's first second-route startup hit the physical-memory watchdog | An already-FP32 scalar head could retain the full mapped checkpoint through a no-op dtype conversion; source mappings also overlapped during loading | Materialize parameters from meta initialization with per-tensor mappings and explicit owned FP32 copies, retaining CPU rotary buffers | Small-model state/forward checks and complete repeated real CLI acceptance passed; the failed attempt was retained. See [loader and results](windows-tiny17-jev9.md#scoring-and-limits). |
| Some GGUF serving imports could trigger a Windows OpenMP conflict | Unneeded Torch/Transformers imports in a tokenizer/native-runtime process | Keep the relevant Mica/JevK5 GGUF paths free of those inference imports and use fixed native runtime bytes | Actual GGUF process identity, source/runtime preservation and numerical checks passed; model memory is reported separately from the Python service. |
| A same-version native package is not necessarily the same executable runtime | Wheel names and version metadata do not identify every DLL byte | Check the accepted official wheel/runtime hashes and reject replacements | Decider/Decision independent acceptance and runtime postchecks passed; arbitrary same-version builds remain unsupported. |
| A previous reference result could become stale after a driver change | Historical reports do not automatically encode current runner, fixture and source identity | Check provenance before reuse and rerun references when required | The final regression records 21 historical references and one fresh JevK5 4B author run; the Tiny final run explicitly reused its verified fresh baseline after a product-only fix. |
| Documentation accumulated overlapping historical counts | Incremental adaptation introduced multiple acceptance snapshots | Keep current scope in the onboarding/user guide, shorten README and label historical increments | Current count is 24; the earlier 22-model page remains a dated record. Final checks must preserve that distinction. |

## Technical Debt and Remaining Risks

| Item | Why it remains | Impact | Mitigation or next action | Status |
|---|---|---|---|---|
| Windows native GPU/vLLM, images/video and RadixFlow | Outside the approved CPU/text scope | Existing Linux/Mac capabilities cannot be assumed on Windows | Require a separate implementation and actual platform-specific acceptance | Deferred |
| Eleven original-catalogue models without Windows acceptance | No completed supported-artifact validation in this work | Similar model names may suggest broader support than delivered | Preserve the explicit excluded-model list; assess any future addition independently | Deferred |
| Quantization and decision quality | Tests compare fixed-case numerical behavior at the same artifact/precision | Passing parity does not establish model-card accuracy or full-precision equivalence | Use representative labeled evaluation before making task-quality claims | Not evaluated |
| General host performance and maximum capacity | Acceptance ran short workloads on one local CPU host | Timing and memory observations do not predict all computers or workloads | Retain conservative guards; benchmark separately for production capacity | Not established |
| Graceful shutdown and real-model client cancellation | Acceptance verifies owned-process cleanup; unit tests cover service cancellation | Cleanup evidence alone cannot prove real-workload disconnect/shutdown behavior | Add bounded real-host lifecycle scenarios if these behaviors become release requirements | Partially covered |
| Large raw evidence is outside Git | Weights, environments and detailed process reports are unsuitable repository artifacts | Historical reports are not independently accessible from this checkout alone | Preserve a local audit archive; checked-in drivers support fresh reproduction | Documented limitation |
| Tight dependency/runtime pins | Preserving observed CPU arithmetic and DLL compatibility takes priority over broad compatibility | Package updates require revalidation; profiles may age | Upgrade in new environments and rerun affected references/products before changing pins | Intentional constraint |

## Reusable Assets

| Asset | Location | Reuse value | Usage notes |
|---|---|---|---|
| CPU service boundary | [`windows_server.py`](../vllm_jev/windows_server.py) | Shared request/queue/health contract | Preserve model-specific validation and cancellation accounting |
| Model/profile routing | [`windows_models.py`](../vllm_jev/windows_models.py), [`windows_dependencies.py`](../vllm_jev/windows_dependencies.py) | One entrypoint with explicit accepted identities | A mapping entry is not proof of real-model acceptance |
| Family implementations | `vllm_jev/windows_*.py` | Pinned preparation, tokenization, readout and calibration | Reuse only when the new release's actual protocol matches |
| Independent acceptance drivers | `scripts/check_windows_*.py`, `scripts/windows_*_reference.py` | Source/runtime verification, probability comparison, HTTP checks and cleanup evidence | Prepare the documented source data and separate reference environment first |
| Small-model and boundary tests | `tests/test_windows_*.py` | Fast regression without large model downloads | Execute in the matching profile; they do not replace real weights |
| User-facing operating procedures | [User guide](guide.md#windows-cpu-experimental), [onboarding](project-onboarding.md) | Installation, scope and reproducible entrypoints | Keep current totals separate from historical evidence |
| Final-increment evidence description | [Tiny/JevK5 acceptance](windows-tiny17-jev9.md#retained-local-evidence) | Report layout and interpretation of memory/reference evidence | Local raw reports are external audit records, not runtime dependencies |

## What to Do Differently Next Time

- **Start doing:** agree on the exact checkpoint, precision, dependency profile
  and independent reference before estimating feasibility from parameter count.
- **Stop doing:** treating weight-file size or a small Python parent process as
  the model's total memory footprint; distinguish process working set, commit
  and sampled host availability.
- **Continue doing:** retain failed attempts, keep tolerances and guards stable,
  preserve previous environments and rerun affected real models after meaningful
  arithmetic/runtime changes.
- **Pre-flight checks:** verify branch/worktree state, source/runtime pins,
  original-license notices, physical RAM, commit headroom, disk capacity and
  independent-reference requirements before loading a new model.
- **Questions to answer earlier:** whether the model uses packed questions or
  independent rows, whether its release includes a complete backbone, whether
  reference CPU precision differs from published benchmark precision, and what
  qualifies as acceptance for its supported template and option limits.
