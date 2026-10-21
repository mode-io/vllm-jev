"""Independent pinned JevK5 CPU reference; never imports product inference.

The parent owns the process and working directory. Only original verified SDK
bytes execute. No weights are downloaded and no packages are installed.
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

MODEL_ID = "alibiserikbay/JevK5-2B"
REVISION = "7922d1f55df137b72ef763fced56fd09efc5e99d"
SDK_COMMIT = "f26426d16f59e8bbe1470e5b162cc89329e29b29"
MAX_SEQUENCE_TOKENS = 1024
SOURCE_HASHES = {
    "config.json": "6cb8daca9fb653c61485ff7452fc068bacd5c27cbee659ecd24b47186b0d1b52",
    "generation_config.json": "62153eb6c69f2e1f426beaa8002b7186437e949c7588167085df14e10e9c0a73",
    "tokenizer_config.json": "9cf04fffe3d8c3b85e439fb35c7acad0761ab51c422a8c4256d9f887c3a0be7d",
    "chat_template.jinja": "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80",
    "jevk5_config.json": "c3ebf048dd55a95c3a45e368efc7534dc5ca5a5fd4276b3bc8ead068b38097d4",
    "model.safetensors": "847f8b2e6016f1c779cd01ff75428333ad2e9ab491853c36e9521529caf39892",
    "tokenizer.json": "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
}
SDK_HASHES = {
    "jevk5/__init__.py": "3ec8170fe224e7b9b2ce0333a58ba1cb8066effd4e9d825fc913ef033b907e2b",
    "jevk5/gguf.py": "6505de497ce115f0bf85f110cf708fc8535a774f1361b811e3415b3e91a10d03",
    "jevk5/lite.py": "8ca027ea70e9ddb30b1fe9e93e9f808d3cc2dd65fbb6562f6096b464f598e56d",
    "jevk5/prompt.py": "322d69ba6abea83602f27b7a5aa0d3e6ab00847fd03ddccd35a28ae5b23acc1a",
    "jevk5/runtime.py": "b05f2816135237b36694198bb3b8ffac1abc044ad23c70caa867b3be59a81be6",
    "jevk5/server.py": "2ebde38db58a1603ee390f72eb8737d5e969003973788640fa0e1edd58d75977",
}


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(root, expected):
    actual = {}
    for name in expected:
        if not (root / name).is_file():
            raise ValueError(f"Missing pinned file: {root / name}")
        actual[name] = sha256(root / name)
    if actual != expected:
        raise ValueError(f"Pinned SHA256 mismatch: {root}")
    return actual


def verify_source(root):
    for path in root.iterdir():
        if path.name == ".cache" and path.is_dir():
            continue
        if path.name not in SOURCE_HASHES or not path.is_file():
            raise ValueError(f"Unexpected checkpoint loader input: {path}")
    return verify(root, SOURCE_HASHES)


def verify_sdk(root):
    for path in (root / "jevk5").iterdir():
        if path.name == "__pycache__" and path.is_dir():
            continue
        if path.relative_to(root).as_posix() not in SDK_HASHES or not path.is_file():
            raise ValueError(f"Unexpected author package input: {path}")
    return verify(root, SDK_HASHES)


def copy_sdk(root, target):
    verify_sdk(root)
    for name in SDK_HASHES:
        path = target / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, path)
    verify_sdk(target)


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
            if (
                not identifier
                or not isinstance(question, dict)
                or question.get("type") not in {"noul", "choice", "score"}
            ):
                raise ValueError("Invalid reference question")
            if (
                not isinstance(question.get("instructions"), str)
                or not question["instructions"].strip()
            ):
                raise ValueError("Reference instructions must be nonempty text")
            kind, criteria = question["type"], question.get("criteria")
            if kind == "choice" and (
                not isinstance(criteria, (dict, list)) or not 2 <= len(criteria) <= 16
            ):
                raise ValueError("Reference Choice requires 2 to 16 options")
            if kind == "score" and (
                not isinstance(criteria, list)
                or not 2 <= len(criteria) <= 10
                or question.get("isolated", False) is not False
            ):
                raise ValueError("Reference Score requires 2 to 10 listwise levels")
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
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


class BlockProductImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] == "vllm_jev":
            raise RuntimeError(f"Reference cannot import product inference: {fullname}")
        return None


def infer(args, report, cases, sdk):
    if any(
        name == "jevk5" or name.startswith(("jevk5.", "vllm_jev"))
        for name in sys.modules
    ):
        raise RuntimeError("Run the reference in a fresh process")
    os.environ.update(
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false"
    )
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(sdk))
    guard = BlockProductImports()
    sys.meta_path.insert(0, guard)
    try:
        import torch
        from jevk5 import JevK5

        if (
            Path(sys.modules["jevk5.runtime"].__file__).resolve()
            != sdk / "jevk5/runtime.py"
        ):
            raise RuntimeError("Reference imported an unexpected SDK location")
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        started = time.perf_counter()
        author = JevK5(
            str(args.source), device="cpu", graphs=False, dtype=torch.bfloat16
        )
        report["load_seconds"] = time.perf_counter() - started
        devices = sorted({str(p.device) for p in author.model.parameters()})
        dtypes = sorted({str(p.dtype) for p in author.model.parameters()})
        if (
            devices != ["cpu"]
            or dtypes != ["torch.bfloat16"]
            or author.graphs
            or author.temperature != 1.42
        ):
            raise RuntimeError("Unexpected author runtime precision or calibration")
        report.update(
            measured=True,
            memory_after_load=memory_sample(),
            runtime={
                "threads": torch.get_num_threads(),
                "dtype": "bfloat16",
                "parameter_devices": devices,
                "parameter_dtypes": dtypes,
                "head_dtype": str(author.slot_weight.dtype),
                "head_shape": list(author.slot_weight.shape),
                "row_batch_size": 1,
                "padding": "none",
                "use_cache": False,
                "graphs": False,
                "temperature": author.temperature,
                "max_sequence_tokens": MAX_SEQUENCE_TOKENS,
                "attention_implementation": author.model.config._attn_implementation,
            },
        )
        # Validate every prompt before executing the first request. This does not
        # alter the author; it enforces the same bounded workload as the service.
        from jevk5.prompt import decision_options

        for case in cases:
            for question in case["questions"].values():
                options = decision_options(question)
                ids = author.encode(
                    case["state"],
                    question["instructions"],
                    [text for _, text in options],
                )
                if (
                    not 2 <= len(options) <= 16
                    or not 0 < len(ids) <= MAX_SEQUENCE_TOKENS
                ):
                    raise ValueError("Reference prompt exceeds single-pass limits")
        save(args.output, report)
        for case in cases:
            record = {
                **case,
                "probabilities": {},
                "author_response": {},
                "prompt_rows": {},
                "readout_dtypes": [],
            }
            started = time.perf_counter()
            for identifier, question in case["questions"].items():
                observed = {}

                def observe(frame, event, result):
                    if event != "return":
                        return
                    if frame.f_code is JevK5.probabilities.__code__ and isinstance(
                        result, tuple
                    ):
                        observed["probabilities"] = result[0]
                    elif frame.f_code is JevK5.encode.__code__ and isinstance(
                        result, list
                    ):
                        observed["ids"] = result
                    elif frame.f_code is JevK5._slot_logits.__code__ and isinstance(
                        result, torch.Tensor
                    ):
                        observed["dtype"] = str(result.dtype)
                        observed["shape"] = list(result.shape)

                previous = sys.getprofile()
                try:
                    sys.setprofile(observe)
                    response = author.decide(case["state"], question)
                finally:
                    sys.setprofile(previous)
                if (
                    observed.get("dtype") != "torch.bfloat16"
                    or observed.get("shape") != [1, 16]
                    or not observed.get("probabilities")
                ):
                    raise RuntimeError(
                        "Missing original BF16 16-letter readout observation"
                    )
                record["probabilities"][identifier] = list(
                    observed["probabilities"].values()
                )
                record["author_response"][identifier] = response
                record["prompt_rows"][identifier] = observed["ids"]
                record["readout_dtypes"].append(observed["dtype"])
            record.update(seconds=time.perf_counter() - started, memory=memory_sample())
            report["cases"].append(record)
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
        # Parent owns cwd, including this child SDK copy after a forced timeout.
        with tempfile.TemporaryDirectory(
            prefix="jevk5-sdk-", dir=Path.cwd()
        ) as temporary:
            sdk = Path(temporary)
            copy_sdk(args.sdk_source, sdk)
            report.update(sdk_execution_path=str(sdk), sdk_source_isolated=True)
            infer(args, report, cases, sdk)
            verify_sdk(sdk)
        report["source_unchanged"] = (
            verify_source(args.source) == report["source_hashes"]
        )
        report["sdk_unchanged"] = verify_sdk(args.sdk_source) == report["sdk_hashes"]
        if len(report["cases"]) != len(cases):
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
