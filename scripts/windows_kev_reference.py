"""Independent, pinned Kev author reference on FP32 CPU.

Only the original Checkpoint, API rendering, encoder and forward are executed.
LocalPredictor additionally imports the Unix-only research suite, so this runner
uses its numerical call path directly. No product inference module is imported.
"""

import argparse
import hashlib
import importlib.abc
import importlib.metadata
import json
import os
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

MODEL_ID = "jaredpalmer/kev-0.8b"
REVISION = "bf75a6a8848ea6960ff2ed108d9ed44c2941174f"
BASE_ID = "Qwen/Qwen3.5-0.8B-Base"
BASE_REVISION = "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"
SDK_REVISION = "dd375f27fd61486c52204205e10a0d2e10f3b34a"
SDK_HASHES = {
    "kev/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "kev/api.py": "a6f53af5354f8ea17915abc78360fdf291a5108ce39a53a2d9135d48c603ae58",
    "kev/checkpoint.py": "02062399977ee0da316dd52e03a79fa5e10134aa49c0b5ed801598c52042dc32",
    "kev/model.py": "2ff3b2e67c048e3bc5a0d1a32aafe50804ed315a5564fce06bb4b95697c5271c",
}
# Independently frozen from the HF trees and downloaded original configuration
# bytes, rather than consulting the product adapter at reference runtime.
SOURCE_HASHES = {
    "adapter_config.json": "748acb2cda88454cb1ba69d745ba336f3fcb5486eac349e90960c8b8d8d3e854",
    "adapter_model.safetensors": "9b908623acb162118575f4e7a94524f9c139c335be4bfb74d6cfceca01e1885a",
    "head.pt": "f400bd12802b2b105ae45d6b03774a158a3db4fccff42413734ddca2e5c920b6",
    "provenance.json": "fce039fbda7de19451c9b5d3065aee2c5e063e1562a8bc18c72e297fbd15e9d8",
    "training_config.json": "3ec3d92146248265652d4387f8eb65ee8defaed4dd9173b049e7b5193045ece6",
    "tokenizer.json": "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
    "tokenizer_config.json": "8671bed7c852ce9e661be94f179a7b4ffd091c2a65aea0363e5501c20318ee45",
    "base/config.json": "b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204",
    "base/LICENSE": "50cbab8a892c5f2993b8c7351a99182507472def3b1374558308605d99b86b32",
    "base/model.safetensors.index.json": "ce9a885efdf27d3664fdef5d512ad365216f1074051ef840c7cd8e5431495d0a",
    "base/model.safetensors-00001-of-00001.safetensors": "c2b1e5a17d9c1e27685d92ed9b382911ebb99955ecd89052d1721241adfbab6c",
    "base/tokenizer.json": "fe000e3ed39ed12b8d2481d527d44f93c65d37e87645d2dcc80d1bf9d50d2927",
    "base/tokenizer_config.json": "e611fbccc7c29ef3b1cafb1cb7ea548d189968632901d678fd62be68c47885de",
}


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(root, expected):
    actual = {name: sha256(root / name) for name in expected}
    if actual != expected:
        raise ValueError(f"Pinned source SHA256 mismatch: {root}")
    return actual


def read_cases(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate input JSON key: {key}")
            result[key] = value
        return result

    def nonfinite(value):
        raise ValueError(f"Nonfinite input JSON: {value}")

    cases = json.loads(
        path.read_text(encoding="utf-8-sig"),
        object_pairs_hook=unique,
        parse_constant=nonfinite,
    )
    if not isinstance(cases, list) or not cases:
        raise ValueError("Cases must be a nonempty array")
    identifiers = set()
    for case in cases:
        if (
            not isinstance(case, dict)
            or not {"id", "state", "questions"} <= case.keys()
        ):
            raise ValueError("Every case requires id, state and questions")
        if (
            not isinstance(case["id"], str)
            or not case["id"]
            or case["id"] in identifiers
        ):
            raise ValueError("Case IDs must be unique nonempty strings")
        if not isinstance(case["questions"], dict) or not case["questions"]:
            raise ValueError("Cases require a nonempty questions object")
        identifiers.add(case["id"])
    return cases


class BlockProductImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] == "vllm_jev":
            raise RuntimeError(f"Reference cannot import product inference: {fullname}")
        return None


def save(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _copy_verified(source, target, expected, *, large_links=False):
    for name in expected:
        destination = target / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        original = source / name
        if large_links and original.suffix == ".safetensors":
            try:
                os.link(original, destination)
                continue
            except OSError:
                pass
        shutil.copyfile(original, destination)
    verify(target, expected)


def run(args, report):
    source, sdk = args.source.resolve(), args.sdk_source.resolve()
    # Only pinned files are placed in the private import and HF cache paths;
    # unrelated source files and pre-existing bytecode never execute.
    report["source_hashes"] = verify(source, SOURCE_HASHES)
    report["sdk_hashes"] = verify(sdk, SDK_HASHES)
    cases = read_cases(args.cases_json)
    if args.check_only:
        report["status"] = "checked"
        return
    if type(args.threads) is not int or not 1 <= args.threads <= (os.cpu_count() or 1):
        raise ValueError("threads must be between 1 and logical CPU count")
    with tempfile.TemporaryDirectory(
        prefix="kev-reference-", dir=Path.cwd()
    ) as temporary:
        private = Path(temporary)
        report["private_workspace"] = str(private)
        private_sdk = private / "sdk"
        _copy_verified(sdk, private_sdk, SDK_HASHES)
        private_source = private / "checkpoint"
        release = {
            name: value
            for name, value in SOURCE_HASHES.items()
            if not name.startswith("base/")
        }
        _copy_verified(source, private_source, release, large_links=True)
        hub = private / "hf" / "hub"
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
        _copy_verified(source / "base", snapshot, base, large_links=True)
        # Set before importing either Transformers or huggingface_hub. Original
        # load_tokenizer/from_pretrained keep the unmodified model ID+revision.
        os.environ.update(
            {
                "HF_HOME": str(private / "hf"),
                "HF_HUB_CACHE": str(hub),
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_HUB_DISABLE_TELEMETRY": "1",
                "TOKENIZERS_PARALLELISM": "false",
            }
        )
        sys.dont_write_bytecode = True
        sys.meta_path.insert(0, BlockProductImports())
        sys.path.insert(0, str(private_sdk))
        import torch
        from kev.api import SystemOneRequest, to_record
        from kev.checkpoint import Checkpoint, LoadOptions

        torch.set_num_threads(args.threads)
        for package in (
            "torch",
            "transformers",
            "peft",
            "accelerate",
            "huggingface_hub",
        ):
            report["versions"][package] = importlib.metadata.version(package)
        started = time.perf_counter()
        tokenizer, model = Checkpoint(str(private_source)).load(
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
        if any(
            p.device.type != "cpu" or p.dtype != torch.float32 or p.is_meta
            for p in model.parameters()
        ):
            raise ValueError("Original Kev reference did not load in CPU FP32")
        report["temperature"] = model.head.temperature
        report["backbone_dtype"] = str(next(model.lm.parameters()).dtype)
        report["head_dtype"] = str(next(model.head.parameters()).dtype)
        report["cases"] = []
        observed = []

        def observe_forward(module, positional, kwargs):
            ids = kwargs["input_ids"]
            mask = kwargs["attention_mask"]
            positions = kwargs["position_ids"]
            effective_cache = kwargs.get("use_cache", module.config.use_cache)
            reused = kwargs.get("past_key_values") is not None
            if ids.shape[0] != 1 or not mask.all() or ids.shape[1] > 1024 or reused:
                raise RuntimeError(
                    "Original Kev departed from the single-row reference contract"
                )
            if not torch.equal(positions, torch.arange(ids.shape[1]).unsqueeze(0)):
                raise RuntimeError("Original Kev used unexpected position IDs")
            observed.append(
                {
                    "batch": int(ids.shape[0]),
                    "length": int(ids.shape[1]),
                    "no_padding": bool(mask.all()),
                    "use_cache": effective_cache,
                    "cache_reuse": reused,
                }
            )

        hook = model.lm.register_forward_pre_hook(observe_forward, with_kwargs=True)
        with torch.inference_mode():
            for case in cases:
                request = SystemOneRequest(
                    state=case["state"], questions=case["questions"]
                )
                record, metadata = to_record(request)
                result = {
                    "id": case["id"],
                    "state": case["state"],
                    "questions": case["questions"],
                    "probabilities": {},
                    "prompt_rows": [],
                }
                started = time.perf_counter()
                for question, information in zip(record["questions"], metadata):
                    one = {"state": record["state"], "questions": [question]}
                    encoded = model.encode(
                        tokenizer, one, max_state=1024, max_branch=1024, strict=True
                    )
                    logits = model.forward(encoded)
                    if len(logits) != 1 or not torch.isfinite(logits[0]).all():
                        raise RuntimeError("Original Kev returned invalid logits")
                    result["probabilities"][information["id"]] = torch.softmax(
                        logits[0], -1
                    ).tolist()
                    result["prompt_rows"].append(
                        {
                            "question_id": information["id"],
                            "ids": encoded["ids"],
                            "positions": encoded["opt_idx"][0] + encoded["decide_idx"],
                        }
                    )
                result["inference_seconds"] = time.perf_counter() - started
                report["cases"].append(result)
                save(args.output, report)
        hook.remove()
        if not observed or not all(item["use_cache"] is True for item in observed):
            raise RuntimeError("Original Kev did not use its published cache default")
        report["runtime"] = {
            "threads": torch.get_num_threads(),
            "dtype": str(next(model.lm.parameters()).dtype).removeprefix("torch."),
            "parameter_devices": sorted({p.device.type for p in model.parameters()}),
            "parameter_dtypes": sorted({str(p.dtype) for p in model.parameters()}),
            "head_dtype": str(next(model.head.parameters()).dtype),
            "row_batch_size": max(item["batch"] for item in observed),
            "padding": "none"
            if all(item["no_padding"] for item in observed)
            else "present",
            "use_cache": all(item["use_cache"] for item in observed),
            "cache_reuse": any(item["cache_reuse"] for item in observed),
            "graphs": model.graphs is not None,
            "max_sequence_tokens": 1024,
            "attention_implementation": model.lm.config._attn_implementation,
        }
        report["observed_forwards"] = observed
        del model, tokenizer
        report["source_unchanged"] = (
            verify(source, SOURCE_HASHES) == report["source_hashes"]
        )
        report["sdk_unchanged"] = verify(sdk, SDK_HASHES) == report["sdk_hashes"]
        report["reference_source_isolated"] = True
    report["private_workspace_removed"] = not Path(report["private_workspace"]).exists()
    report.update(status="passed", measured=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--sdk-source", type=Path, required=True)
    parser.add_argument("--cases-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "running",
        "model": MODEL_ID,
        "model_id": MODEL_ID,
        "revision": REVISION,
        "base": BASE_ID,
        "base_revision": BASE_REVISION,
        "sdk_revision": SDK_REVISION,
        "sdk_commit": SDK_REVISION,
        "driver_sha256": sha256(__file__),
        "input_sha256": sha256(args.cases_json),
        "measured": False,
        "reference_kind": "original_kev_checkpoint_encoder_forward_fp32_cpu_eager",
        "single_question_per_forward": True,
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
        if "private_workspace" in report:
            report["private_workspace_removed"] = not Path(
                report["private_workspace"]
            ).exists()
        save(args.output, report)
        raise
    save(args.output, report)
    print(json.dumps({"status": report["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
