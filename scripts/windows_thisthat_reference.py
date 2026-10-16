"""Run a pinned, unmodified This-That author SDK on CPU, independently.

No product inference module is imported. The outer acceptance process owns this
runner's working directory and lifecycle. This script downloads nothing.
"""

import argparse
import importlib.metadata
import importlib.util
import os
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

MODEL_ID = "flock-io/this-that-model-1.2"
REVISION = "c4d1c30b8d512d278726de439b8cc81fccc70c8f"
SDK_REVISION = "6af11c63c8bb260aa4bed2b7717f33f25decd16e"
SOURCE_HASHES = {
    "config.json": "6cb8daca9fb653c61485ff7452fc068bacd5c27cbee659ecd24b47186b0d1b52",
    "generation_config.json": "62153eb6c69f2e1f426beaa8002b7186437e949c7588167085df14e10e9c0a73",
    "tokenizer_config.json": "171ecbe7ddae98d11840698f7df2b8d5b4722139db0f0620d3bbf429bd656250",
    "chat_template.jinja": "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80",
    "tokenizer.json": "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
    "model.safetensors": "585295458a6b6e01162d2a72d33ba630c286b51c844869afc35272700c23f199",
}
# Release identities are maintained independently of the product adapter.
# The original 1.2 constants/default remain available to existing callers.
MODELS = {
    MODEL_ID: {"revision": REVISION, "source_hashes": SOURCE_HASHES},
    "flock-io/this-that-model-1.1": {
        "revision": "fc083c82bd5872abfeaf0d44cf7482324efce87d",
        "source_hashes": {
            **SOURCE_HASHES,
            "model.safetensors": "63e1179554bc56654eb7aa1ccc5dc1edbaf99a2b32e1cd19f77fcf65a98be04d",
        },
    },
    "flock-io/this-that-model-1.0": {
        "revision": "3d927195c4f9845efe66c5715883a7a0f42b1239",
        "source_hashes": {
            **SOURCE_HASHES,
            "model.safetensors": "11bab4bbbce0214dcb4d70a88e74b3e4bde6fe8a95f239e3d0c355946e0000e0",
            "tokenizer_config.json": "353c083805d863719e54973335a38e252a92e1ac64d32b83bb0b333cc1644ef2",
        },
    },
}
SDK_HASHES = {
    "thisthat/__init__.py": "3b16e0e93287e6af40294159057eb171f8bea4c7745f93665d4f685437bd7af9",
    "thisthat/model.py": "f5c7fa06e9f2bb1b91ffba7877bf73cb5295ac0bf9b21f084503fbcf1b8c9a65",
    "thisthat/openai_protocol.py": "118c6b03f9c0e2095707e39fe2adf3f36353dd82504a9051acadeb354a8ffb0a",
    "thisthat/prompt.py": "5c5d64eb05c24a70bcaa4b019937cc1e7704c19218bd045b5c48a41d5113606c",
    "thisthat/server.py": "3c0e361577bb336188db4932fc0ffa45d70261f95660bfd08ee1eb5ab066c08d",
    "thisthat/systemone_protocol.py": "efab3aeea4d53857c2e1a5b549a17786488fd94ff113dac8eae33a88b0a0afab",
    "thisthat/types.py": "e2021b4326ebad2379608519dd4baa6433a8f5e8c7ae94e00d165f731d307575",
    "thisthat/backends/__init__.py": "eebb6bf17b692a9cebf9658ba3b258932e2f4e88195c4de8b6f23008466a0931",
    "thisthat/backends/base.py": "6c4375f07e95982c21d99cf950f14f14ca1b505919e136b238d710fbf70c9aaf",
    "thisthat/backends/mlx_backend.py": "e3b2ee9c50c9ce57ad026c7570a0e90d71e318b026ae9e0a430be091d2e4c578",
    "thisthat/backends/torch_backend.py": "9e9a173b2fef3802a597c26132dcf7dfc1e37c2aeea7a30847d0bd5f1f22a266",
}
MAX_SEQUENCE_TOKENS = 1024
_spec = importlib.util.spec_from_file_location(
    "_thisthat_reference_utilities",
    Path(__file__).with_name("windows_decider_reference.py"),
)
common = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(common)
sha256, save = common.sha256, common.save


def verify_sdk(root):
    """Only pinned Python bytes execute, including nested backend modules."""
    for path in (root / "thisthat").rglob("*"):
        relative = path.relative_to(root).as_posix()
        if "__pycache__" in path.parts:
            continue
        if path.is_dir() and relative == "thisthat/backends":
            continue
        if path.is_file() and relative in SDK_HASHES:
            continue
        raise ValueError(f"Unexpected author package entry: {relative}")
    return common.verify(root, SDK_HASHES)


def model_profile(model_id):
    if model_id not in MODELS:
        raise ValueError(f"Unsupported independent This-That release: {model_id}")
    return MODELS[model_id]


def verify_source(root, model_id=MODEL_ID):
    return common.verify_checkpoint(root, model_profile(model_id))


def copy_inputs(source, sdk, workspace, model_id=MODEL_ID):
    """Isolate all loader inputs; hardlink only read-only large weight tensors."""
    verify_source(source, model_id)
    verify_sdk(sdk)
    checkpoint = workspace / "checkpoint"
    checkpoint.mkdir()
    for name in model_profile(model_id)["source_hashes"]:
        if name.endswith(".safetensors"):
            try:
                os.link(source / name, checkpoint / name)
                continue
            except OSError:
                pass
        shutil.copyfile(source / name, checkpoint / name)
    author_source = workspace / "author"
    for name in SDK_HASHES:
        target = author_source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(sdk / name, target)
    verify_source(checkpoint, model_id)
    verify_sdk(author_source)
    return checkpoint, author_source


def observe_forward(backend, observed):
    """Observe actual original frame locals, without replacing any computation."""

    def observe(frame, event, result):
        if event == "return" and frame.f_code is backend.slot_logits.__func__.__code__:
            local = frame.f_locals
            observed.append(
                {
                    "shape": list(local["ids"].shape),
                    "slot_indices": local["slot_idx"].tolist(),
                    "slot_batches": local["slot_batch"].tolist(),
                    "option_counts": local["n_options"].tolist(),
                    "head_shape": list(local["w"].shape),
                    "linear_input_dtype": str(local["h"].dtype),
                    "linear_weight_dtype": str(local["w"].dtype),
                    "logits_dtype": str(result.dtype),
                }
            )

    return observe


def infer(args, source, sdk, cases, report):
    if any(
        name == "thisthat" or name.startswith(("thisthat.", "vllm_jev"))
        for name in sys.modules
    ):
        raise RuntimeError("Run the independent reference in a fresh process")
    guard = common.BlockProductImports()
    sys.meta_path.insert(0, guard)
    sys.path.insert(0, str(sdk))
    try:
        import torch
        from thisthat import TypedDecider
        from thisthat.prompt import build
        from thisthat.systemone_protocol import parse_request, render_answer
        from transformers import AutoTokenizer

        if (
            Path(sys.modules["thisthat.model"].__file__).resolve()
            != sdk / "thisthat/model.py"
        ):
            raise RuntimeError(
                "Author code did not load from the pinned isolated source"
            )
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        torch.manual_seed(0)
        tokenizer = AutoTokenizer.from_pretrained(
            source, local_files_only=True, trust_remote_code=False
        )
        prepared = []
        for case in cases:
            state, typed = parse_request(case)
            built = build(
                tokenizer,
                state,
                [q.question for q in typed],
                max_state_tokens=1536,
                layout="state_first",
            )
            if (len(built["ids"]) + 63) // 64 * 64 > MAX_SEQUENCE_TOKENS:
                raise ValueError(
                    "Reference case exceeds the CPU complete-row token bound"
                )
            prepared.append((case, state, typed, built))
        started = time.perf_counter()
        author = TypedDecider.from_pretrained(
            str(source),
            device="cpu",
            dtype="float32",
            local_files_only=True,
            trust_remote_code=False,
            attn_implementation="sdpa",
        )
        report["load_seconds"] = time.perf_counter() - started
        devices = sorted({str(p.device) for p in author.model.parameters()})
        dtypes = sorted({str(p.dtype) for p in author.model.parameters()})
        if devices != ["cpu"] or dtypes != ["torch.float32"]:
            raise RuntimeError("Author reference did not load FP32 parameters on CPU")
        report.update(
            measured=True,
            memory_after_load=common.memory_sample(),
            runtime={
                "actual_parameter_devices": devices,
                "actual_parameter_dtypes": dtypes,
                "actual_embedding_dtype": str(
                    author.model.get_input_embeddings().weight.dtype
                ),
                "actual_head_dtype": str(
                    author.model.get_output_embeddings().weight.dtype
                ),
                "threads": torch.get_num_threads(),
                "interop_threads": torch.get_num_interop_threads(),
                "config_use_cache": author.model.config.use_cache,
                "attention_implementation": author.model.config._attn_implementation,
                "layout": "state_first",
                "max_state_tokens": 1536,
                "temperature": 1.0,
                "padding_multiple": 64,
                "packed_questions": True,
                "cross_request_cache": False,
            },
        )
        save(args.output, report)
        for case, state, typed, built in prepared:
            observations = []
            old = sys.getprofile()
            started = time.perf_counter()
            try:
                sys.setprofile(observe_forward(author.backend, observations))
                answers = author.decide(
                    state,
                    [q.question for q in typed],
                    temperature=1.0,
                    layout="state_first",
                    max_state_tokens=1536,
                )
            finally:
                sys.setprofile(old)
            seconds = time.perf_counter() - started
            if len(observations) != 1:
                raise RuntimeError("Reference did not use one packed forward")
            observed = observations[0]
            if (
                observed["shape"] != [1, (len(built["ids"]) + 63) // 64 * 64]
                or observed["slot_indices"] != built["slots"]
                or observed["option_counts"] != built["n_options"]
                or observed["linear_input_dtype"] != "torch.float32"
                or observed["linear_weight_dtype"] != "torch.float32"
                or observed["head_shape"] != [255, 2048]
                or observed["logits_dtype"] != "float32"
            ):
                raise RuntimeError(
                    "Unexpected actual author forward/readout configuration"
                )
            report["cases"].append(
                {
                    **case,
                    "probabilities": {
                        q.key: list(a.probabilities) for q, a in zip(typed, answers)
                    },
                    "answers": {
                        q.key: render_answer(q, a) for q, a in zip(typed, answers)
                    },
                    "prompt": built,
                    "forward": observed,
                    "seconds": seconds,
                    "memory": common.memory_sample(),
                }
            )
            save(args.output, report)
            print(f"reference case {case['id']} finished", flush=True)
        del author
    finally:
        sys.path.remove(str(sdk))
        sys.meta_path.remove(guard)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, default=MODEL_ID)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--sdk-source", type=Path, required=True)
    parser.add_argument("--cases-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    profile = model_profile(args.model)
    if not 1 <= args.threads <= (os.cpu_count() or 1):
        parser.error("invalid CPU thread count")
    for key in ("source", "sdk_source", "cases_json", "output"):
        setattr(args, key, getattr(args, key).resolve())
    if args.output == args.cases_json or any(
        args.output.is_relative_to(root) for root in (args.source, args.sdk_source)
    ):
        parser.error(
            "output must be outside the source/SDK and cannot overwrite input cases"
        )
    os.environ.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
        HF_HUB_DISABLE_TELEMETRY="1",
    )
    sys.dont_write_bytecode = True
    environment = common.environment()
    environment["packages"]["tokenizers"] = importlib.metadata.version("tokenizers")
    report = {
        "status": "running",
        "measured": False,
        "model_id": args.model,
        "revision": profile["revision"],
        "sdk_revision": SDK_REVISION,
        "sdk_commit": SDK_REVISION,
        "versions": environment["packages"],
        "environment": environment,
        "source": str(args.source),
        "sdk_source": str(args.sdk_source),
        "cases": [],
        "driver_sha256": sha256(Path(__file__)),
        "helper_sha256": sha256(Path(common.__file__)),
        "reference_kind": "original_thisthat_sdk_cpu_float32_packed",
    }
    started = time.perf_counter()
    try:
        report["source_hashes"] = verify_source(args.source, args.model)
        report["sdk_hashes"] = verify_sdk(args.sdk_source)
        cases = common.read_cases(args.cases_json)
        report["input_sha256"] = sha256(args.cases_json)
        report["cases_sha256"] = report["input_sha256"]
        if args.check_only:
            report["status"] = "source_checked"
            return
        with tempfile.TemporaryDirectory(
            prefix="thisthat-reference-", dir=Path.cwd()
        ) as temporary:
            source, sdk = copy_inputs(
                args.source, args.sdk_source, Path(temporary), args.model
            )
            report.update(source_isolated=True, sdk_isolated=True)
            infer(args, source, sdk, cases, report)
            verify_source(source, args.model)
            verify_sdk(sdk)
        report["source_unchanged"] = (
            verify_source(args.source, args.model) == report["source_hashes"]
        )
        report["sdk_unchanged"] = verify_sdk(args.sdk_source) == report["sdk_hashes"]
        report["status"] = "passed"
    except BaseException as error:
        report.update(
            status="failed",
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
        )
        raise
    finally:
        report["wall_seconds"] = time.perf_counter() - started
        save(args.output, report)


if __name__ == "__main__":
    main()
