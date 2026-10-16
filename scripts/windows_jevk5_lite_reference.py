"""Pinned, independent author JevK5-Lite/JevBench CPU reference.

The original four SDK files execute unchanged. The only JevBench shim is its
DecisionResult data container, following the author's test_lite_bench.py; it
does not tokenize, infer, calibrate, normalize, or map an answer. The parent
owns this process and its working directory, including scratch after timeout.
"""

import argparse
import hashlib
import importlib.abc
import importlib.metadata
import importlib.util
import json
import math
import os
import shutil
import sys
import tempfile
import time
import traceback
import types
from dataclasses import asdict, dataclass, field
from pathlib import Path

MODEL_ID = "alibiserikbay/JevK5-Lite"
REVISION = "315ee211f828a899161477c534cea23e84ba3568"
SDK_COMMIT = "f26426d16f59e8bbe1470e5b162cc89329e29b29"
MAX_SEQUENCE_TOKENS = 512
TEMPERATURE_SINGLE = 1.8660659830736146
TEMPERATURE_MULTI = 1.319507910772894
SOURCE_HASHES = {
    "config.json": "4b143dff2309fc312d0413c1d42b4c1f75cc5f33e70b593f4d4568877e56783e",
    "lite_config.json": "e80a53d67af90d2d8237cfb6a9dfd365501b0da3b8019ad2ed7827fee3b35777",
    "tokenizer_config.json": "d008608e43ac0a01895fb8a6a241316c2869ae5af65724195b4a28d4f600a8b1",
    "tokenizer.json": "5eebf550a388329ccc733e5e9b73f75c83e4e50ff0fe99a9885217cf20d274fd",
    "NOTICE": "ced5d9c7234168de5b3b057895305cd1c1377a3556b2afc09e74d3c063fa92ba",
    "model.safetensors": "152b56535b93874f3404ce421c86c88e0e064b4f321a7e41c709357843f714de",
    "scorer.safetensors": "1ca3c4f5f2e88b87a9951055f567fd9e64258e027d0ecd5d9b375e49399ee98d",
}
SDK_HASHES = {
    "jevk5/__init__.py": "3ec8170fe224e7b9b2ce0333a58ba1cb8066effd4e9d825fc913ef033b907e2b",
    "jevk5/prompt.py": "322d69ba6abea83602f27b7a5aa0d3e6ab00847fd03ddccd35a28ae5b23acc1a",
    "jevk5/lite.py": "8ca027ea70e9ddb30b1fe9e93e9f808d3cc2dd65fbb6562f6096b464f598e56d",
    "bench/jevk5_lite.py": "d5f24d5e8f83f9916e620beaff4716b920a2cd1d5ee23e6aa9281607e654f912",
}


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(root, expected):
    actual = {}
    for name, digest in expected.items():
        path = root / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Missing or linked pinned file: {path}")
        actual[name] = sha256(path)
        if actual[name] != digest:
            raise ValueError(f"Pinned SHA256 mismatch: {path}")
    return actual


def verify_source(root):
    for path in root.iterdir():
        if path.name == ".cache" and path.is_dir() and not path.is_symlink():
            continue
        if path.name not in SOURCE_HASHES or not path.is_file() or path.is_symlink():
            raise ValueError(f"Unexpected checkpoint loader input: {path}")
    return verify(root, SOURCE_HASHES)


def verify_sdk(root, *, isolated=False):
    # A complete author checkout is accepted, but only these verified files are
    # copied into the private import path. Other checkout code never executes.
    if isolated:
        for path in root.rglob("*"):
            name = path.relative_to(root).as_posix()
            if (
                path.is_symlink()
                or (path.is_file() and name not in SDK_HASHES)
                or (
                    path.is_dir()
                    and not any(key.startswith(name + "/") for key in SDK_HASHES)
                )
            ):
                raise ValueError(f"Unexpected private author package input: {path}")
    return verify(root, SDK_HASHES)


def copy_inputs(source, sdk, workspace):
    verify_source(source)
    verify_sdk(sdk)
    checkpoint, author = workspace / "checkpoint", workspace / "sdk"
    for root, target, names in (
        (source, checkpoint, SOURCE_HASHES),
        (sdk, author, SDK_HASHES),
    ):
        for name in names:
            dest = target / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if name.endswith(".safetensors"):
                try:
                    os.link(root / name, dest)
                except OSError:
                    shutil.copyfile(root / name, dest)
            else:
                shutil.copyfile(root / name, dest)
    verify_source(checkpoint)
    verify_sdk(author, isolated=True)
    return checkpoint, author


def read_cases(path):
    def invalid(value):
        raise ValueError(f"Nonfinite JSON: {value}")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    cases = json.loads(
        path.read_text(encoding="utf-8-sig"),
        object_pairs_hook=unique,
        parse_constant=invalid,
    )
    # JSON's exponent notation can overflow without invoking parse_constant.
    json.dumps(cases, allow_nan=False)
    if not isinstance(cases, list) or not cases:
        raise ValueError("Cases must be a nonempty list")
    identifiers = set()
    for case in cases:
        if (
            not isinstance(case, dict)
            or not {"id", "state", "questions"} <= case.keys()
        ):
            raise ValueError("Cases require id, state, questions")
        if (
            not isinstance(case["id"], str)
            or not case["id"]
            or case["id"] in identifiers
        ):
            raise ValueError("Case IDs must be distinct nonempty text")
        identifiers.add(case["id"])
        if not isinstance(case["state"], (str, dict, list)):
            raise ValueError("Reference state must be text or JSON")
        if not isinstance(case["questions"], dict) or not case["questions"]:
            raise ValueError("Reference questions must be a nonempty map")
        for identifier, question in case["questions"].items():
            if not identifier or not isinstance(question, dict):
                raise ValueError("Invalid reference question")
            if (
                not isinstance(question.get("instructions"), str)
                or not question["instructions"].strip()
            ):
                raise ValueError("Reference instructions must be nonempty text")
            kind, criteria = question.get("type"), question.get("criteria")
            if kind == "noul":
                if criteria is not None and criteria != {}:
                    raise ValueError("Noul does not support custom criteria")
            elif kind == "choice":
                if (
                    not isinstance(criteria, (dict, list))
                    or not 2 <= len(criteria) <= 255
                ):
                    raise ValueError("Choice requires 2 to 255 options")
                if not all(isinstance(key, str) and key for key in criteria):
                    raise ValueError("Choice option IDs must be nonempty text")
                if isinstance(criteria, list) and len(set(criteria)) != len(criteria):
                    raise ValueError("Choice option IDs must be unique")
                if isinstance(criteria, dict) and not all(
                    value is None or isinstance(value, str)
                    for value in criteria.values()
                ):
                    raise ValueError("Choice descriptions must be text or null")
            elif kind == "score":
                if (
                    not isinstance(criteria, list)
                    or not 2 <= len(criteria) <= 10
                    or not all(isinstance(level, str) for level in criteria)
                    or question.get("isolated", False) is not False
                ):
                    raise ValueError("Score requires 2 to 10 listwise text levels")
            else:
                raise ValueError("Unsupported reference question type")
    return cases


def versions():
    return {
        name: importlib.metadata.version(name)
        for name in (
            "torch",
            "transformers",
            "numpy",
            "huggingface_hub",
            "accelerate",
            "safetensors",
            "psutil",
            "tokenizers",
        )
    }


def memory_sample():
    import psutil

    info = psutil.Process().memory_info()
    return {
        key: getattr(info, key)
        for key in ("rss", "vms", "peak_wset", "private", "peak_pagefile")
        if hasattr(info, key)
    }


def save(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


class BlockProductImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] == "vllm_jev":
            raise RuntimeError(f"Reference cannot import product inference: {fullname}")
        return None


@dataclass
class DecisionResult:
    """JevBench transport fields, as in the author's lightweight adapter tests."""

    adapter: str
    ok: bool
    probs: dict | None = None
    probs_source: str = "unknown"
    model: str = ""
    error: str | None = None
    latency_s: float = 0.0
    usage: dict = field(default_factory=dict)
    raw: object = None
    request_body: object = None


def load_bench(sdk):
    """Load verified bench bytes with only a non-inference result container."""
    verify_sdk(sdk)
    name = "_jevk5_lite_author_bench"
    if any(key == name or key.startswith(name + ".") for key in sys.modules):
        raise RuntimeError("Run the author bench in a fresh process")
    package = types.ModuleType(name)
    package.__path__ = []
    base = types.ModuleType(name + ".base")
    base.DecisionResult = DecisionResult
    sys.modules.update({name: package, name + ".base": base})
    spec = importlib.util.spec_from_file_location(
        name + ".jevk5_lite", sdk / "bench/jevk5_lite.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        for key in list(sys.modules):
            if key == name or key.startswith(name + "."):
                sys.modules.pop(key)
        raise
    return module


def preflight_cases(cases, bench, author):
    """Restrict the comparison to complete, untruncated author input rows."""
    prompts = {}
    for case in cases:
        rows = {}
        for identifier, question in case["questions"].items():
            task, labels, keys = bench.lite_head(question)
            text = bench.state_text(case["state"])
            ids, positions, spans, (start, end) = author.encode(
                text, [(task, labels, False, None)]
            )
            total_state_tokens = len(
                author.tok(text, add_special_tokens=False)["input_ids"]
            )
            dropped = total_state_tokens - (end - start)
            if dropped != 0 or not 0 < len(ids) <= MAX_SEQUENCE_TOKENS:
                raise ValueError("Reference requires complete input within 512 tokens")
            if len(set(labels)) != len(labels) or spans != [(0, len(labels))]:
                raise ValueError("Author mapping produced an invalid label head")
            rows[identifier] = {
                "input_ids": ids,
                "label_positions": positions,
                "text_span": [start, end],
                "task": task,
                "labels": labels,
                "option_ids": keys,
                "truncated_state_tokens": dropped,
            }
        prompts[case["id"]] = rows
    return prompts


def result_probabilities(result, question, row):
    if result.ok is not True or result.error is not None:
        raise RuntimeError(f"Original author adapter failed: {result.error}")
    raw = result.raw
    keys = row["option_ids"]
    if (
        not isinstance(raw, dict)
        or not isinstance(raw.get("probabilities"), dict)
        or list(raw["probabilities"]) != keys
        or raw.get("labels") != dict(zip(keys, row["labels"]))
        or raw.get("truncated_state_tokens") != 0
        or raw.get("temperature_single") != TEMPERATURE_SINGLE
        or result.usage.get("input_tokens") != len(row["input_ids"])
    ):
        raise RuntimeError(
            "Original author output or untruncated input evidence differs"
        )
    probabilities = list(raw["probabilities"].values())
    if (
        len(probabilities) != len(keys)
        or any(
            type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1
            for p in probabilities
        )
        or not math.isclose(sum(probabilities), 1.0, abs_tol=1e-6)
    ):
        raise RuntimeError("Original author returned invalid probabilities")
    expected = (
        {"yes": probabilities[0], "no": probabilities[1]}
        if question["type"] == "noul"
        else raw["probabilities"]
    )
    if result.probs != expected:
        raise RuntimeError("Original author JevBench probability mapping differs")
    return probabilities


def infer(args, report, cases, sdk, checkpoint):
    if any(name.split(".")[0] in {"jevk5", "vllm_jev"} for name in sys.modules):
        raise RuntimeError("Run the reference in a fresh process")
    os.environ.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
        JEVK5_LITE_DTYPE="fp32",
        JEVK5_LITE_THREADS=str(args.threads),
    )
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(sdk))
    guard = BlockProductImports()
    sys.meta_path.insert(0, guard)
    hook = None
    try:
        import torch

        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        bench = load_bench(sdk)
        adapter = bench.JevK5LiteAdapter(
            endpoint=str(checkpoint), model=MODEL_ID, revision=REVISION
        )
        started = time.perf_counter()
        author = adapter.load()
        report["load_seconds"] = time.perf_counter() - started
        if Path(sys.modules["jevk5.lite"].__file__).resolve() != sdk / "jevk5/lite.py":
            raise RuntimeError("Reference imported an unexpected SDK location")
        parameters = [*author.encoder.parameters(), *author.scorer.parameters()]
        devices = sorted({str(p.device) for p in parameters})
        dtypes = sorted({str(p.dtype) for p in parameters})
        if (
            devices != ["cpu"]
            or dtypes != ["torch.float32"]
            or author.temperature_single != TEMPERATURE_SINGLE
            or author.temperature_multi != TEMPERATURE_MULTI
            or author.max_len != MAX_SEQUENCE_TOKENS
        ):
            raise RuntimeError("Original author CPU precision or calibration differs")
        report.update(
            memory_after_load=memory_sample(),
            runtime={
                "threads": torch.get_num_threads(),
                "dtype": "float32",
                "parameter_devices": devices,
                "parameter_dtypes": dtypes,
                "head_dtype": str(next(author.scorer.parameters()).dtype),
                "row_batch_size": 1,
                "padding": "none",
                "use_cache": False,
                "graphs": False,
                "max_sequence_tokens": MAX_SEQUENCE_TOKENS,
                "temperature_single": author.temperature_single,
                "temperature_multi": author.temperature_multi,
                "temperature": author.temperature_single,
                "truncation": False,
                "packed_questions": False,
                "cache_reuse": False,
                "truncated_state_tokens": 0,
                "encoder_class": type(author.encoder).__name__,
                "attention_implementation": author.encoder.config._attn_implementation,
                "decision_result_shim": "transport_only_author_test_fields",
            },
        )
        prompts = preflight_cases(cases, bench, author)
        observed = []

        def observe_forward(module, positional, keyword):
            ids, mask = keyword.get("input_ids"), keyword.get("attention_mask")
            if (
                ids is None
                or mask is None
                or list(ids.shape) != list(mask.shape)
                or ids.ndim != 2
                or ids.shape[0] != 1
                or not bool(torch.all(mask == 1))
                or keyword.get("past_key_values") is not None
                or keyword.get("use_cache", False) is not False
            ):
                raise RuntimeError("Author forward differs from one-row unpadded input")
            observed.append(ids.detach().cpu().tolist()[0])

        hook = author.encoder.register_forward_pre_hook(
            observe_forward, with_kwargs=True
        )
        save(args.output, report)
        for case in cases:
            record = {
                **case,
                "probabilities": {},
                "author_response": {},
                "prompt_rows": prompts[case["id"]],
            }
            started = time.perf_counter()
            for identifier, question in case["questions"].items():
                row = prompts[case["id"]][identifier]
                observed.clear()
                result = adapter.run(
                    types.SimpleNamespace(state=case["state"], question=question)
                )
                if observed != [row["input_ids"]]:
                    raise RuntimeError(
                        "Author did not execute exactly one expected encoder pass"
                    )
                record["probabilities"][identifier] = result_probabilities(
                    result, question, row
                )
                record["author_response"][identifier] = asdict(result)
            record.update(seconds=time.perf_counter() - started, memory=memory_sample())
            report["cases"].append(record)
            report["measured"] = True
            save(args.output, report)
            print(
                json.dumps(
                    {
                        "event": "reference_case_completed",
                        "id": case["id"],
                        "seconds": record["seconds"],
                    }
                ),
                flush=True,
            )
        report["memory_final"] = memory_sample()
    finally:
        if hook is not None:
            hook.remove()
        sys.meta_path.remove(guard)
        sys.path.remove(str(sdk))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--sdk-source", type=Path, required=True)
    parser.add_argument("--cases-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.threads <= (os.cpu_count() or 1):
        parser.error("threads must be between 1 and the logical CPU count")
    for name in ("source", "sdk_source", "cases_json", "output"):
        setattr(args, name, getattr(args, name).resolve())
    if args.output == args.cases_json or any(
        args.output.is_relative_to(root) for root in (args.source, args.sdk_source)
    ):
        parser.error("output must be outside source/SDK and not overwrite cases")
    return args


def main():
    args = parse_args()
    report = {
        "status": "running",
        "model_id": MODEL_ID,
        "revision": REVISION,
        "sdk_commit": SDK_COMMIT,
        "measured": False,
        "versions": versions(),
        "driver_sha256": sha256(__file__),
        "cases": [],
        "environment": {"python": sys.version, "executable": sys.executable},
    }
    started = time.perf_counter()
    try:
        report["source_hashes"] = verify_source(args.source)
        report["sdk_hashes"] = verify_sdk(args.sdk_source)
        cases = read_cases(args.cases_json)
        report.update(
            input_sha256=sha256(args.cases_json),
            selected_case_ids=[case["id"] for case in cases],
        )
        if args.check_only:
            report["status"] = "source_checked"
            return
        with tempfile.TemporaryDirectory(
            prefix="jevk5-lite-reference-", dir=Path.cwd()
        ) as temporary:
            checkpoint, sdk = copy_inputs(args.source, args.sdk_source, Path(temporary))
            report.update(
                sdk_execution_path=str(sdk),
                checkpoint_execution_path=str(checkpoint),
                sdk_source_isolated=True,
                checkpoint_source_isolated=True,
            )
            infer(args, report, cases, sdk, checkpoint)
            report["private_source_unchanged"] = (
                verify_source(checkpoint) == report["source_hashes"]
            )
            report["private_sdk_unchanged"] = (
                verify_sdk(sdk, isolated=True) == report["sdk_hashes"]
            )
        report["source_unchanged"] = (
            verify_source(args.source) == report["source_hashes"]
        )
        report["sdk_unchanged"] = verify_sdk(args.sdk_source) == report["sdk_hashes"]
        if len(report["cases"]) != len(cases) or report["measured"] is not True:
            raise RuntimeError("Reference did not complete every selected case")
        report["status"] = "passed"
    except BaseException as error:
        report.update(
            status="failed",
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
        )
        raise
    finally:
        report["total_seconds"] = time.perf_counter() - started
        save(args.output, report)


if __name__ == "__main__":
    main()
