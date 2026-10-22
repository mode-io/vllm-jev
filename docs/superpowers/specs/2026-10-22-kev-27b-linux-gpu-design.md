# Kev-27B native Linux GPU support — design

## Intent and the why

`jaredpalmer/kev-27b` is a Kev pointer-head decision model whose backbone is
the large Qwen3.8-27B release. vLLM Jev already serves the small Kev pointer
checkpoints (`kev-0.6b`, `kev-0.8b`) on Windows CPU and already serves large
composite Qwen3.5/3.8 LoRA checkpoints natively on Linux GPU for JevAny
(`SimpleJev/JevAny-Qwen3.8-27B-LoRA`). The goal is to make Kev-27B serve
through the native Linux GPU path so users can run
`vllm-jev serve jaredpalmer/kev-27b`.

The existing native `export_kev` cannot handle Kev-27B for two reasons:

1. **Whole-model load.** It calls
   `AutoModelForCausalLM.from_pretrained(base)` and `merge_and_unload()`,
   materializing the entire 27B backbone in memory. That is infeasible at 27B.
2. **Top-level `model_type`.** It reads `config["model_type"]` to pick the text
   architecture, but the 4B/27B bases are *composite* configs whose top-level
   `model_type` is `qwen3_5` (multimodal) with the text backbone nested under
   `text_config` (`qwen3_5_text`). So it raises "unsupported Kev backbone"
   before even reaching the size problem — this also means `kev-4b` is
   currently unserved.

## Facts established (from the live Hugging Face repos)

- `jaredpalmer/kev-27b` **main** (`af0e6d55`) is the v2 full-weight release:
  11 bf16 shards with bare `Qwen3_5TextModel` keys (`embed_tokens.*`,
  `layers.*`, `norm.*`; no `lm_head`, MTP or vision), a flat `qwen3_5_text`
  config (`tie_word_embeddings=false`), and a `head.pt` with
  `weights="full"`, T=1.3195. It is 0.85 x full fine-tune + 0.15 x v1
  (`interpolation.json`) and needs no base download. The v1 LoRA release lives
  on the **`v1-lora` tag** (`01b81998`) (`adapter_config.json` + `adapter_model.safetensors` +
  `head.pt`), base `Qwen/Qwen3.8-27B`, LoRA r=16/alpha=32, `target_modules`
  including the Gated-DeltaNet `in_proj_*` projections, `task_type`
  `FEATURE_EXTRACTION`, `bias` none.
- `jaredpalmer/kev-4b` **main** is already the LoRA form (adapter + head),
  base `Qwen/Qwen3.5-4B-Base` (composite config).
- Both bases are composite: top-level `model_type=qwen3_5`, with
  `text_config.model_type=qwen3_5_text`, plus a `vision_config`.
- The repo already contains `_text_backbone()` in `decision_export.py`, which
  strips `visual`/`vision_tower`/`mtp` modules and rewrites
  `model.language_model.*` → `model.*`, producing a pure `Qwen3_5ForCausalLM`
  text checkpoint. `_write()` invokes it automatically. `export_decider`
  already serves composite full-weight 27B backbones through it.
- JevAny's `export_jevany_pointer` already demonstrates **per-shard LoRA
  merge** (fp32 delta → source dtype, one shard open at a time) for exactly
  these bases.

## Approach (reuse, not rewrite)

Add a **composite/large-base branch** to `export_kev` that composes three
existing pieces rather than inventing new mechanics:

1. **Per-shard LoRA merge** (technique borrowed from JevAny): copy/hard-link
   each base shard into the output; for shards that host a LoRA target, open
   one shard, add `right @ left * (alpha/r)` in fp32, cast back to source
   dtype, and write it out. Target-key translation uses the **text backbone
   naming** (`model.language_model.<module>.weight`), matching what
   `_text_backbone` expects.
2. **Text-backbone extraction** (existing `_text_backbone` via `_write`):
   strip vision/MTP, rename to `model.*`, pick `Qwen3_5ForCausalLM`.
3. **Kev pointer protocol** (existing): same `head.pt` validation
   (`{q,k}.{weight,bias}`, `option_isolation=False`, `special_embeddings=False`)
   and the existing `kev_pointer_v1` manifest written by `_write`.

The existing small-model branch (`kev-0.6b/0.8b`, flat config, whole-model
merge) is left untouched. Selection between branches is by presence of
`text_config` in the base config.

**Full-weight releases** (`export_kev_full`): when a repo has `head.pt`, a
config and a shard index but no adapter, the head is downloaded and validated
first (`weights == "full"`, pointer-head tensors, no option isolation or
special embeddings), then the shards. Each shard is rewritten once with the
`model.` prefix vLLM's `Qwen3_5ForCausalLM` expects, values and per-tensor
dtypes unchanged. `tie_word_embeddings` is set to true because the pointer head
reads hidden states only and the checkpoint has no `lm_head`; MTP fields are
dropped. The result is the same `kev_pointer_v1` manifest as the LoRA path.

Dispatch (`native_models.py`): `KEV_PINNED_REVISIONS` pins
`jaredpalmer/kev-27b` to its v2 `main` commit, matching the author's default
`kev.serve --run jaredpalmer/kev-27b`. Downloads go to a per-revision
directory, and a cached checkpoint from a different revision is rejected with
instructions to remove it. `kev-4b` needs no pin (main is already LoRA).

## Goals

- `vllm-jev serve jaredpalmer/kev-27b` exports and serves the v2 full-weight
  release on Linux GPU via the native `kev_pointer_v1` path.
- The same new branch also enables `jaredpalmer/kev-4b`.
- Per-shard merge so peak memory stays near one shard, not the whole backbone.
- Full checksum/identity verification preserved (`verify_decision`).

## Non-goals / anti-goals

- No change to Windows CPU `windows_kev.py` (27B FP32 CPU is not viable) or to
  the small-model whole-model merge branch.
- No model-ID syntax for selecting the kev-27b `v1-lora` tag; its layout is
  handled by the LoRA export but `vllm-jev serve` serves the pinned v2 main.
- Disk: the ~51 GB download is kept, and the renamed export is a second ~51 GB
  copy (hard links are impossible once keys change).
- No new model architecture classes — reuse `Qwen3_5ForCausalLM` already
  registered via `model.py`.
- No multimodal/image support for Kev (it is a text/JSON pointer model).

## Constraints

- Cannot run true 27B end-to-end here (no GPU, no 54GB weights). Verification
  is: synthetic-tiny-model contract tests (composite base + LoRA + head),
  matching the repo's existing test style, plus code review. Real GPU
  acceptance is left to CI / the maintainers and stated in the PR.

## Left to the builder

- Exact helper factoring inside `decision_export.py` (whether to lift JevAny's
  `_adapter_pairs` into a shared helper or inline a Kev-specific one). Builder's
  call; keep it local and minimal.
