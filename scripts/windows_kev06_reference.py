"""Pinned Kev-0.6B reference: unchanged author encoder and packed forward.

The runtime calls are the original Checkpoint, to_record, encode and forward.
Multi-question block-causal masks are observed, never replaced by product rows.
"""

import argparse
import importlib.metadata
import importlib.util
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

definition = importlib.util.spec_from_file_location(
    "_kev_reference_helpers", Path(__file__).with_name("windows_kev_reference.py")
)
common = importlib.util.module_from_spec(definition)
definition.loader.exec_module(common)
MODEL_ID = "jaredpalmer/kev-0.6b"
REVISION = "dece6dba8d43f0f7ded45e9f5b9df12474d90843"
BASE_ID = "Qwen/Qwen3-0.6B-Base"
BASE_REVISION = "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"
SDK_REVISION = "dd375f27fd61486c52204205e10a0d2e10f3b34a"
SDK_HASHES = dict(common.SDK_HASHES)
SOURCE_HASHES = {
    "adapter_config.json": "5bfcf3dd85b1383c802146cfdef179e2e772685e310ac972fb672f22d7079b4f",
    "adapter_model.safetensors": "deaab63b61f95d628503831e8b336e1c3e334b9a42d3530d979a5149b429c492",
    "head.pt": "ce6cd9ffc54db41c179b65a33d60973dc8280c29e886218eb3ec07dc28d6f28b",
    "provenance.json": "321bc9db7427b490ad943fa33576c7e01f5da4c20da1036fcbfee996f4b28f10",
    "training_config.json": "ab20e55cbb832f409de9daf42f868b69be1eb59e51720302b166bc560fd8d240",
    "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    "tokenizer_config.json": "67e5a0a11cd35f9c00ee52e0af4cdc0baa75fea0cb5fce7d1beb251b4621d15c",
    "base/config.json": "504a6b58c4271583724e66584b6b7698aea18450209df6b2f7582df0e89cee59",
    "base/LICENSE": "832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e",
    "base/tokenizer_config.json": "3c04ed3ca964ea2f6b2b5faf0dc4d31aec1cb1e8b4bcf63f402d295046b422b5",
    "base/tokenizer.json": "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
    "base/model.safetensors": "cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba",
}


def run(args, report):
    source, sdk = args.source.resolve(), args.sdk_source.resolve()
    report["source_hashes"] = common.verify(source, SOURCE_HASHES)
    report["sdk_hashes"] = common.verify(sdk, SDK_HASHES)
    cases = common.read_cases(args.cases_json)
    with tempfile.TemporaryDirectory(
        prefix="kev06-reference-", dir=Path.cwd()
    ) as directory:
        private = Path(directory)
        report["private_workspace"] = str(private)
        common._copy_verified(sdk, private / "sdk", SDK_HASHES)
        release = {
            name: value
            for name, value in SOURCE_HASHES.items()
            if not name.startswith("base/")
        }
        common._copy_verified(source, private / "checkpoint", release, large_links=True)
        hub = private / "hf/hub"
        snapshot = (
            hub
            / ("models--" + BASE_ID.replace("/", "--"))
            / "snapshots"
            / BASE_REVISION
        )
        base = {
            name.removeprefix("base/"): value
            for name, value in SOURCE_HASHES.items()
            if name.startswith("base/")
        }
        common._copy_verified(source / "base", snapshot, base, large_links=True)
        os.environ.update(
            HF_HOME=str(private / "hf"),
            HF_HUB_CACHE=str(hub),
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            HF_HUB_DISABLE_TELEMETRY="1",
            TOKENIZERS_PARALLELISM="false",
        )
        sys.dont_write_bytecode = True
        sys.meta_path.insert(0, common.BlockProductImports())
        sys.path.insert(0, str(private / "sdk"))
        import torch
        from kev.api import SystemOneRequest, to_record
        from kev.checkpoint import Checkpoint, LoadOptions

        torch.set_num_threads(args.threads)
        for name in ("torch", "transformers", "peft", "accelerate", "huggingface_hub"):
            report["versions"][name] = importlib.metadata.version(name)
        started = time.perf_counter()
        tokenizer, model = Checkpoint(str(private / "checkpoint")).load(
            "cpu",
            LoadOptions(
                dtype=torch.float32,
                merge=True,
                attn="eager",
                backend="torch",
                cuda_graphs=False,
                fused=False,
            ),
        )
        report["load_seconds"] = time.perf_counter() - started
        if (
            model.hybrid
            or model.head.temperature != 1.0
            or any(
                p.device.type != "cpu" or p.dtype != torch.float32 or p.is_meta
                for p in model.parameters()
            )
        ):
            raise RuntimeError(
                "author Kev-0.6B did not load in its pinned FP32 profile"
            )
        observed = []
        active = None

        def observe(module, positional, kwargs):
            ids, mask, positions = (
                kwargs["input_ids"],
                kwargs["attention_mask"],
                kwargs["position_ids"],
            )
            if (
                active is None
                or ids.tolist() != [active["ids"]]
                or positions.tolist() != [active["pos"]]
            ):
                raise RuntimeError(
                    "author packed tokens/positions differ from encoding"
                )
            length = len(active["ids"])
            if (
                mask.shape != (1, 1, length, length)
                or kwargs.get("past_key_values") is not None
            ):
                raise RuntimeError("author packed attention or cache shape differs")
            seg = torch.tensor(active["seg"])
            allowed = torch.tril(torch.ones(length, length, dtype=torch.bool)) & (
                (seg[None, :] == seg[:, None]) | (seg[None, :] == 0)
            )
            if not torch.equal(mask[0, 0] == 0, allowed) or not torch.all(
                mask[0, 0][~allowed] == torch.finfo(torch.float32).min
            ):
                raise RuntimeError("author mask did not isolate question branches")
            observed.append(
                {
                    "batch": 1,
                    "length": length,
                    "mask_kind": "packed_block_causal",
                    "questions": len(active["decide_idx"]),
                    "use_cache": kwargs.get("use_cache", module.config.use_cache),
                    "cache_reuse": False,
                }
            )

        hook = model.lm.register_forward_pre_hook(observe, with_kwargs=True)
        report["cases"] = []
        with torch.inference_mode():
            for case in cases:
                record, metadata = to_record(
                    SystemOneRequest(state=case["state"], questions=case["questions"])
                )
                encoded = model.encode(
                    tokenizer, record, max_state=1024, max_branch=1024, strict=True
                )
                if len(encoded["ids"]) > 4096 or model.rows_form([encoded]):
                    raise RuntimeError(
                        "reference cases must fit one bounded packed pass"
                    )
                active = encoded
                started = time.perf_counter()
                logits = model.forward(encoded)
                if len(logits) != len(metadata) or any(
                    not torch.isfinite(z).all() for z in logits
                ):
                    raise RuntimeError("invalid author packed logits")
                result = {
                    "id": case["id"],
                    "state": case["state"],
                    "questions": case["questions"],
                    "probabilities": {
                        info["id"]: torch.softmax(z, -1).tolist()
                        for info, z in zip(metadata, logits, strict=True)
                    },
                    "prompt_rows": [],
                    "packed_ids": encoded["ids"],
                    "packed_positions": encoded["pos"],
                    "inference_seconds": time.perf_counter() - started,
                }
                for question, info in zip(record["questions"], metadata, strict=True):
                    one = model.encode(
                        tokenizer,
                        {"state": record["state"], "questions": [question]},
                        max_state=1024,
                        max_branch=1024,
                        strict=True,
                    )
                    if len(one["ids"]) > 1024:
                        raise RuntimeError("author row exceeds product bound")
                    result["prompt_rows"].append(
                        {
                            "question_id": info["id"],
                            "ids": one["ids"],
                            "positions": one["opt_idx"][0] + one["decide_idx"],
                        }
                    )
                report["cases"].append(result)
                common.save(args.output, report)
        hook.remove()
        report["runtime"] = {
            "threads": torch.get_num_threads(),
            "dtype": "float32",
            "parameter_devices": sorted({p.device.type for p in model.parameters()}),
            "parameter_dtypes": sorted({str(p.dtype) for p in model.parameters()}),
            "head_dtype": str(next(model.head.parameters()).dtype),
            "row_batch_size": 1,
            "padding": "none",
            "use_cache": all(x["use_cache"] is True for x in observed),
            "cache_reuse": False,
            "graphs": model.graphs is not None,
            "max_sequence_tokens": 1024,
            "packed_questions": True,
            "attention_implementation": model.lm.config._attn_implementation,
            "temperature": model.head.temperature,
        }
        report["observed_forwards"] = observed
        del model, tokenizer
        report["source_unchanged"] = (
            common.verify(source, SOURCE_HASHES) == report["source_hashes"]
        )
        report["sdk_unchanged"] = common.verify(sdk, SDK_HASHES) == report["sdk_hashes"]
        report["reference_source_isolated"] = True
    report["private_workspace_removed"] = not Path(report["private_workspace"]).exists()
    report.update(status="passed", measured=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "sdk-source", "cases-json", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.threads <= 4:
        parser.error("1 to 4 CPU threads required")
    args.output = args.output.resolve()
    if (
        any(
            args.output.is_relative_to(root.resolve())
            for root in (args.source, args.sdk_source)
        )
        or args.output == args.cases_json.resolve()
    ):
        parser.error("output cannot replace inputs")
    report = {
        "status": "running",
        "model": MODEL_ID,
        "model_id": MODEL_ID,
        "revision": REVISION,
        "base": BASE_ID,
        "base_revision": BASE_REVISION,
        "sdk_revision": SDK_REVISION,
        "sdk_commit": SDK_REVISION,
        "driver_sha256": common.sha256(__file__),
        "helper_sha256": common.sha256(common.__file__),
        "input_sha256": common.sha256(args.cases_json),
        "measured": False,
        "reference_kind": "original_kev_checkpoint_encoder_packed_forward_fp32_cpu_eager",
        "single_question_per_forward": False,
        "row_batch_size": 1,
        "padding": "none",
        "noul_probability_order": ["false", "true"],
        "versions": {"python": sys.version},
    }
    try:
        run(args, report)
    except BaseException as error:
        report.update(
            status="failed",
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
        )
        raise
    finally:
        common.save(args.output, report)


if __name__ == "__main__":
    main()
