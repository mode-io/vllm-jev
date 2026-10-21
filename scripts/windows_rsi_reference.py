"""Independent RSI v1 release load_release/encode/evaluate on FP32 CPU.

The hash-pinned author files are copied unchanged into a private offline cache.
This runner imports no product modules and does not replace any author function.
"""

import argparse
import importlib.metadata
import json
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

import windows_kev_reference as common

MODEL_ID = "shgao/rsi-jev-v1.0-qwen3.5-0.8b"
REVISION = "f9248caceb89caf2e6c968ea33bf0d6eb7f957b0"
BASE_ID = "Qwen/Qwen3.5-0.8B-Base"
BASE_REVISION = "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"
SDK_REVISION = REVISION
SOURCE_HASHES = {
    "meta.json": "2dbae889af349197429e924865abd79a98017771ef6a017e5c7fd2479b956141",
    "tower.safetensors": "60f8ea114a6dc3070be268a1f95973b503bd300b2965555f8acad2d95cfcff67",
    "scorer.safetensors": "22cb924e0eb9432db467539e18df48ba837a09eee1106b94f04a21b430c11a06",
    "base/config.json": "b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204",
    "base/LICENSE": "50cbab8a892c5f2993b8c7351a99182507472def3b1374558308605d99b86b32",
    "base/model.safetensors.index.json": "ce9a885efdf27d3664fdef5d512ad365216f1074051ef840c7cd8e5431495d0a",
    "base/model.safetensors-00001-of-00001.safetensors": "c2b1e5a17d9c1e27685d92ed9b382911ebb99955ecd89052d1721241adfbab6c",
    "base/tokenizer.json": "fe000e3ed39ed12b8d2481d527d44f93c65d37e87645d2dcc80d1bf9d50d2927",
    "base/tokenizer_config.json": "e611fbccc7c29ef3b1cafb1cb7ea548d189968632901d678fd62be68c47885de",
}
SDK_HASHES = {
    "code/load_release.py": "55d909f8e62225cab3fd491a7ce14a3ae184880615dfec5ac5f8bc72e6c959b9",
    "code/rsijev/arch.py": "847d894fc8d590868c3b495109d3c3e4dfe57f129f0cfd40b027a56c9e3ebb8d",
    "code/rsijev/contract.py": "d412f9dd98f24fdc3f0aabfa3bc6b2481ce88c99f7e8498140a4c1d2352ea1d7",
    "code/rsijev/encode.py": "9f9c8318289b55d11dfbd6845958bffd686e9225edf67a6e5ac6d4d26c7e9f8e",
    "code/rsijev/evaluate.py": "9951b75f1649359ba79df9a3fab975e56ff2411f2370e23c37d4a5d542b04d51",
    "code/rsijev/metrics.py": "e7a0b9727a2bc9683c62cd0de28cf232f70984b0a8a902d4ba6562a13ed9f529",
    "code/rsijev/__init__.py": "df3d5d0cd69e78af31528c02ebc6ff878053355938039495b876d33cfb720a87",
}


def verify_checkpoint(path):
    return common.verify(path, SOURCE_HASHES)


def verify_sdk(path):
    return common.verify(path, SDK_HASHES)


def serialize(value):
    return (
        value
        if isinstance(value, str)
        else json.dumps(
            value, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        )
    )


def author_case(case):
    from rsijev.contract import Case, Question

    questions = []
    for identifier, request in case["questions"].items():
        kind, criteria = request["type"], request.get("criteria")
        if kind == "noul":
            options = ("false", "true")
            given = criteria or {}
            criteria = {
                key: serialize(given.get(key, value))
                for key, value in (("false", "No"), ("true", "Yes"))
            }
        elif kind == "score":
            options = tuple(str(i) for i in range(len(criteria)))
            criteria = {key: serialize(value) for key, value in zip(options, criteria)}
        else:
            options = tuple(criteria)
            criteria = (
                {
                    key: "" if value is None else serialize(value)
                    for key, value in criteria.items()
                }
                if isinstance(criteria, dict)
                else dict.fromkeys(options, "")
            )
        questions.append(
            Question(
                identifier, kind, serialize(request["instructions"]), options, criteria
            )
        )
    # The unmodified Case requires normalized gold, unused by predict.
    gold = {q.key: tuple([1.0] + [0.0] * (len(q.options) - 1)) for q in questions}
    return Case(
        case["id"],
        "windows-acceptance",
        serialize(case["state"]),
        tuple(questions),
        gold,
    )


def run(args, report):
    source, sdk = args.source.resolve(), args.sdk_source.resolve()
    report["source_hashes"] = verify_checkpoint(source)
    report["sdk_hashes"] = verify_sdk(sdk)
    cases = common.read_cases(args.cases_json)
    with tempfile.TemporaryDirectory(prefix="rsi-reference-", dir=Path.cwd()) as tmp:
        private = Path(tmp)
        report["private_workspace"] = str(private)
        common._copy_verified(sdk, private / "sdk", SDK_HASHES)
        release = {k: v for k, v in SOURCE_HASHES.items() if not k.startswith("base/")}
        common._copy_verified(source, private / "checkpoint", release, large_links=True)
        base = {
            k.removeprefix("base/"): v
            for k, v in SOURCE_HASHES.items()
            if k.startswith("base/")
        }
        hub = private / "hf/hub"
        model_cache = hub / ("models--" + BASE_ID.replace("/", "--"))
        snapshot = model_cache / "snapshots" / BASE_REVISION
        common._copy_verified(source / "base", snapshot, base, large_links=True)
        (model_cache / "refs").mkdir(parents=True)
        (model_cache / "refs/main").write_text(BASE_REVISION, encoding="utf-8")
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
        sys.path.insert(0, str(private / "sdk/code"))
        import torch
        from load_release import load_release
        from rsijev.encode import EncodeConfig, encode_question
        from rsijev.evaluate import predict

        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        report["versions"] = {
            n: importlib.metadata.version(n)
            for n in (
                "torch",
                "transformers",
                "safetensors",
                "huggingface_hub",
                "numpy",
                "tokenizers",
                "accelerate",
                "jinja2",
            )
        }
        if (
            report["versions"]["torch"] != "2.13.0+cpu"
            or report["versions"]["transformers"] != "5.17.0"
        ):
            raise ValueError(
                "RSI reference requires the pinned 2.13 CPU / 5.17 environment"
            )
        started = time.perf_counter()
        model, tokenizer, enc, meta = load_release(
            private / "checkpoint", device="cpu", infer_dtype=torch.float32
        )
        report["load_seconds"] = time.perf_counter() - started
        if any(
            p.dtype != torch.float32 or p.device.type != "cpu" or p.is_meta
            for p in model.parameters()
        ):
            raise ValueError(
                "Original RSI loader did not materialize FP32 CPU parameters"
            )
        if model.tower.config._attn_implementation != "sdpa":
            raise ValueError("Original RSI attention implementation changed")
        report["runtime"] = {
            "device": "cpu",
            "tower_dtype": "float32",
            "scorer_dtype": "float32",
            "temperature": 1.0,
            "attention_implementation": "sdpa",
            "use_cache": model.tower.config.use_cache,
            "threads": args.threads,
            "batch_size": 1,
            "max_options": meta["spec"]["max_options"],
            "original_loader": "load_release.load_release",
            "original_predict": "rsijev.evaluate.predict",
            "original_forward": "rsijev.arch.DecisionModel.forward",
        }
        for case in cases:
            original = author_case(case)
            preflight = [
                encode_question(
                    tokenizer, original.state, q, EncodeConfig(max_length=1_000_000)
                )
                for q in original.questions
            ]
            if any(len(row["input_ids"]) > 1024 for row in preflight):
                raise ValueError("Reference complete prompt exceeds product domain")
            observed, inputs = [], []

            def capture_input(_module, _args, kwargs):
                inputs.append(
                    {
                        name: value.detach().cpu().tolist()
                        for name, value in kwargs.items()
                    }
                )

            def capture(_module, _args, output):
                observed.append(output.detach().cpu().tolist())

            handle = model.register_forward_hook(capture)
            input_handle = model.register_forward_pre_hook(
                capture_input, with_kwargs=True
            )
            started = time.perf_counter()
            try:
                result = predict(
                    model,
                    tokenizer,
                    [original],
                    enc,
                    max_options=80,
                    device="cpu",
                    batch_size=1,
                    temperature=1.0,
                )
            finally:
                handle.remove()
                input_handle.remove()
            if len(inputs) != len(preflight):
                raise ValueError("Original forward row count differs")
            for row, actual in zip(preflight, inputs, strict=True):
                count = len(row["option_span"])
                if (
                    actual["input_ids"] != [row["input_ids"]]
                    or actual["decision_index"] != [row["decision_index"]]
                    or actual["option_span_start"][0][:count]
                    != [a for a, _ in row["option_span"]]
                    or actual["option_span_end"][0][:count]
                    != [b for _, b in row["option_span"]]
                    or actual["option_mask"]
                    != [[True] * count + [False] * (80 - count)]
                ):
                    raise ValueError(
                        "Original forward tokens, spans or padded option mask differ"
                    )
            probabilities = {q.key: list(pred.probs) for _, q, pred in result}
            raw_logits = {
                q.key: observed[i][0][: len(q.options)]
                for i, q in enumerate(original.questions)
            }
            report["cases"].append(
                {
                    **case,
                    "probabilities": probabilities,
                    "raw_logits": raw_logits,
                    "encoded": preflight,
                    "state_not_truncated": True,
                    "original_forward_inputs_verified": True,
                    "seconds": time.perf_counter() - started,
                    "original_readouts": {
                        q.key: {
                            "choice": pred.choice(q.options),
                            "score": pred.score(),
                            "noul": pred.noul() if q.mode == "noul" else None,
                        }
                        for _, q, pred in result
                    },
                }
            )
            common.save(args.output, report)
            print(json.dumps({"case": case["id"], "status": "passed"}), flush=True)
        del model
        report["private_source_unchanged"] = (
            common.verify(private / "checkpoint", release) == release
        )
        report["private_sdk_unchanged"] = (
            common.verify(private / "sdk", SDK_HASHES) == SDK_HASHES
        )
        report["private_base_unchanged"] = common.verify(snapshot, base) == base
    report["private_workspace_removed"] = not private.exists()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "sdk-source", "cases-json", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    report = {
        "status": "running",
        "model": MODEL_ID,
        "revision": REVISION,
        "base_revision": BASE_REVISION,
        "sdk_revision": SDK_REVISION,
        "runner_sha256": common.sha256(__file__),
        "reference_helpers_sha256": common.sha256(common.__file__),
        "cases_sha256": common.sha256(args.cases_json),
        "python": sys.version,
        "executable": sys.executable,
        "cases": [],
    }
    try:
        if sys.platform != "win32" or not 1 <= args.threads <= 4:
            raise ValueError("Native Windows with 1 to 4 threads is required")
        if any(
            args.output.resolve().is_relative_to(p.resolve())
            for p in (args.source, args.sdk_source)
        ):
            raise ValueError("Reference output must be outside source and SDK")
        if args.output.resolve() in (
            args.cases_json.resolve(),
            Path(sys.executable).resolve(),
        ):
            raise ValueError("Reference output must not overwrite cases or interpreter")
        run(args, report)
        report.update(status="passed", measured=True)
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        traceback.print_exc()
    finally:
        try:
            report["source_unchanged"] = verify_checkpoint(args.source) == SOURCE_HASHES
            report["sdk_unchanged"] = verify_sdk(args.sdk_source) == SDK_HASHES
        except Exception as error:
            report.update(status="failed", preservation_error=str(error))
        common.save(args.output, report)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
