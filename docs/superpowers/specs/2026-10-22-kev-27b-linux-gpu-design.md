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

- `jaredpalmer/kev-27b` **main** is a full fine-tune (11 full shards, no
  adapter) — not servable by the LoRA merge path. The LoRA release lives on the
  **`v1-lora` tag** (`adapter_config.json` + `adapter_model.safetensors` +
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

Dispatch (`native_models.py`): add a pinned map so `jaredpalmer/kev-27b`
resolves to its **`v1-lora`** revision (the servable LoRA form), avoiding the
full-fine-tune main. `kev-4b` needs no pin (main is already LoRA). The Kev
dispatch branch already downloads the base from `head.pt`'s
`base`/`base_revision` and calls `export_kev`, so only revision selection and
the fixture flow change.

## Goals

- `vllm-jev serve jaredpalmer/kev-27b` exports and serves on Linux GPU via the
  native `kev_pointer_v1` path.
- The same new branch also enables `jaredpalmer/kev-4b`.
- Per-shard merge so peak memory stays near one shard, not the whole backbone.
- Full checksum/identity verification preserved (`verify_decision`).

## Non-goals / anti-goals

- No change to Windows CPU `windows_kev.py` (27B FP32 CPU is not viable) or to
  the small-model whole-model merge branch.
- No support for the full-fine-tune **main** form of kev-27b (no adapter). If
  desired later, that is a separate "native full-weight Kev" branch.
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
