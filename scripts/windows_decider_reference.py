"""Run the pinned author's Decider SDK independently of all product inference code.

Use a separate process and a local, verified checkpoint. The acceptance driver
supplies --cases-json and owns the process/workspace. This runner never downloads
weights or installs packages. --check-only verifies data without importing Torch.
sys.setprofile only observes original return frames; it changes no computation.
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

SDK_08 = {
    "decider/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "decider/engine.py": "fd2a485d5f2168fb1a24406e610c574195e1ed3ab79e46d38a4e319f27afd45b",
    "decider/fp8.py": "b0450e71249a85cf3a2c758214074a2deebd53e2b66e32589a1aece056311b4c",
    "decider/infer.py": "da075c07e07199d71cbb1b0693bc562914d7dc45509cd86127b3249ca5515ac5",
    "decider/metrics.py": "da9fc1758e74f348145dbdd72509a8b644b238e728e7a9ea9342c78269eee0ed",
    "decider/model.py": "0c6c6bd3410df1c117cbede9881c1dcbe9e4b04d3617dc32b886ddc3405b9987",
    "decider/prompt.py": "5a42134cf470c566e34ac38fb10e63851c21a4bb739475eef797849bcfe460a3",
    "decider/schema_engine.py": "5db350679a15430dfa2800db9f53b628cd8de348497288e7565ac33ce29c72ad",
    "decider/serve.py": "3d7f916ce9ed76825db619250cc9eb785b69e5fd6adf003063a06ae184a62a94",
    "decider/systemone.py": "627fc365583c92328b7b21d481b307eb4d2bfd27411d38cd6710ab3adf629a9a",
}
SDK_2B = {
    "decider/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "decider/batching.py": "c0c6f001850c3bc73d0707bf9309790cc00860a56dd22811be9bcae946861a99",
    "decider/calibrate.py": "9d01463ac07681d55a77b56098522c93cb449dd1701a8f6b24c1d209f98da6c9",
    "decider/engine.py": "829032095e4590152a8f1499ec110edd8004b011a1ee31c2908f3293d0a3b0ba",
    "decider/engine_v2.py": "2686f994dce11d39de40e2ccdb0c6f7df7e43a08686050bf475e4d08b0e77a2b",
    "decider/fp8.py": "b0450e71249a85cf3a2c758214074a2deebd53e2b66e32589a1aece056311b4c",
    "decider/infer.py": "6359e5989fe922054c99446115a25d22acf1a943f0dea097409eaa49e2ef84f1",
    "decider/metrics.py": "da9fc1758e74f348145dbdd72509a8b644b238e728e7a9ea9342c78269eee0ed",
    "decider/model.py": "2feae466895e84b7637ee3504522d809d88dded0b290f6c87b071e4b7f43e027",
    "decider/mps_moe.py": "9bbc924787c23e2ca848c4671d9fce3937ae909dd17a7ee3eb9e86b17553f93e",
    "decider/mps_ops.py": "58a2fc17e692e8402568ac53a63d96632524abef79280dd6e0b53e1fec54f66f",
    "decider/prompt.py": "46f8158cd00c708408f6340340517c5a0c50fb0a339d0e46cd0a30f8801624cd",
    "decider/prompt_fast.py": "fbfcc3a381a89785174cb9d29083ed14d404c7273d123ba8fe2b69584649ef25",
    "decider/schema_engine.py": "6872d2ccc0a91673e5d1583de7f390115075bdfc9403f3b9edd6ccf964937c4a",
    "decider/serve.py": "5a159804bda8eab98b9afbb4026d44abbed7b8e5e95879823623d3bdfaecf6a4",
    "decider/shared_prefix.py": "c37aac37f1f45b133c325d7838ade78bff581bd6de6a54941a8324d28f07052d",
    "decider/systemone.py": "884e4b0bae36f5e09dbd689b4390e94320cbbd77f7b8d675e50490b1d7c61d34",
    "decider/temperature.py": "1b88c270b788a21b01dd31803dadfcc3a61ba0246576c1de04881b1db10a8735",
}
SOURCE_08 = {
    "config.json": "3f56b6210db3c52e0bafb50da005eb52b6d340ab4ad980091879b397743675fc",
    "decider_config.json": "a817fa20524b2c062753f59fd5fae02207f1eff4808b729db6e0aa8d1bc3cce2",
    "generation_config.json": "62153eb6c69f2e1f426beaa8002b7186437e949c7588167085df14e10e9c0a73",
    "tokenizer_config.json": "8671bed7c852ce9e661be94f179a7b4ffd091c2a65aea0363e5501c20318ee45",
    "chat_template.jinja": "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80",
    "model.safetensors": "6926f82ef7e9ea408ad881555204daa6bb694ca82509c4d1514b7be2e713563d",
    "tokenizer.json": "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
}
SOURCE_2B = {
    **SOURCE_08,
    "config.json": "6cb8daca9fb653c61485ff7452fc068bacd5c27cbee659ecd24b47186b0d1b52",
    "decider_config.json": "6e4891f2754a1c18a10f8dadb0c04e439e7f79fab0333d56641491bd4a05e722",
    "tokenizer_config.json": "171ecbe7ddae98d11840698f7df2b8d5b4722139db0f0620d3bbf429bd656250",
    "model.safetensors": "acaef2228b134dcdc20cad4ee79219482c927ec819aa3687b9b8a575c338817f",
}
PROFILES = {
    "Mapika/decider-0.8b": {
        "revision": "a0a01d6f8135298f400a8c856b355793012ae971",
        "reference_kind": "original_hf_release_cpu_eager",
        "sdk_hashes": SDK_08,
        "source_hashes": SOURCE_08,
        "typed_temperatures": False,
    },
    "Mapika/decider-2b": {
        "revision": "ff99089b6810691bd79134ec6febb5d9afa6e619",
        "reference_kind": "original_hf_release_2b_v11_cpu_eager",
        "sdk_hashes": SDK_2B,
        "source_hashes": SOURCE_2B,
        "typed_temperatures": True,
    },
}


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(root, expected):
    for name in expected:
        if not (root / name).is_file():
            raise ValueError(f"Missing pinned file: {root / name}")
    actual = {name: sha256(root / name) for name in expected}
    if actual != expected:
        raise ValueError(f"Pinned SHA256 mismatch: {root}")
    return actual


def verify_sdk(root, profile):
    for path in (root / "decider").iterdir():
        if (
            path.relative_to(root).as_posix() in profile["sdk_hashes"]
            and path.is_file()
        ):
            continue
        if path.name == "__pycache__" and path.is_dir():
            continue
        raise ValueError(f"Unexpected author package file: {path}")
    return verify(root, profile["sdk_hashes"])


def copy_sdk(root, target, profile):
    """Only verified source bytes execute; existing bytecode is never copied."""
    verify_sdk(root, profile)
    for name in profile["sdk_hashes"]:
        path = target / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, path)
    verify_sdk(target, profile)


def verify_checkpoint(root, profile):
    for path in root.iterdir():
        if path.name in profile["source_hashes"] and path.is_file():
            continue
        if path.name == ".cache" and path.is_dir():
            continue
        if path.name in {"README.md", "LICENSE", ".gitattributes"} and path.is_file():
            continue
        raise ValueError(f"Unexpected checkpoint loader input: {path}")
    return verify(root, profile["source_hashes"])


def read_cases(path):
    def invalid(value):
        raise ValueError(f"nonfinite input JSON: {value}")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate input JSON key: {key}")
            result[key] = value
        return result

    data = json.loads(
        path.read_text(encoding="utf-8-sig"),
        object_pairs_hook=unique,
        parse_constant=invalid,
    )
    if isinstance(data, dict) and "cases" in data:
        data = data["cases"]
    elif isinstance(data, dict) and "state" in data:
        data = [dict(data, id=data.get("id", "case-1"))]
    if not isinstance(data, list) or not data:
        raise ValueError("cases JSON must be a nonempty array or {cases: [...]}.")
    ids = []
    for case in data:
        if (
            not isinstance(case, dict)
            or not {"id", "state", "questions"} <= case.keys()
        ):
            raise ValueError("Every case requires id, state, questions.")
        if not isinstance(case["id"], str) or not case["id"]:
            raise ValueError("Case ids must be nonempty strings.")
        if not isinstance(case["questions"], dict) or not case["questions"]:
            raise ValueError("Every case requires a nonempty questions map.")
        ids.append(case["id"])
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate case id.")
    return data


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


def environment():
    packages = {}
    for name in (
        "torch",
        "transformers",
        "numpy",
        "huggingface_hub",
        "safetensors",
        "psutil",
        "peft",
        "accelerate",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": sys.version, "executable": sys.executable, "packages": packages}


class BlockProductImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        # Transformers probes optional backends with find_spec; leave those
        # availability checks untouched and prohibit the project namespace only.
        if fullname.split(".")[0] == "vllm_jev":
            raise RuntimeError(
                f"Reference cannot import a product inference module: {fullname}"
            )
        return None


def observer(author, decider, torch, observed, temperature_function=None):
    """Capture unchanged return values and locals, never replace SDK callables."""

    def observe(frame, event, result):
        if event != "return":
            return
        if frame.f_code is author.m.slot_logits.__func__.__code__ and isinstance(
            result, torch.Tensor
        ):
            observed.setdefault("readout_dtypes", []).append(str(result.dtype))
            for local, name in (
                ("hs", "linear_input_dtypes"),
                ("W", "linear_weight_dtypes"),
            ):
                observed.setdefault(name, []).append(str(frame.f_locals[local].dtype))
            observed.setdefault("forward_input_shapes", []).append(
                list(frame.f_locals["input_ids"].shape)
            )
        if (
            temperature_function is not None
            and frame.f_code is temperature_function.__code__
            and isinstance(result, torch.Tensor)
        ):
            value = frame.f_locals["temperature"]
            observed.setdefault("row_temperatures", []).extend(
                value if isinstance(value, (list, tuple)) else [value] * result.shape[0]
            )
        if frame.f_code is decider.system_one.__code__ and isinstance(result, dict):
            local = frame.f_locals
            observed.update(
                row_probabilities=local["flatp"],
                row_index=local["index"],
                rendered_questions=local["rqs"],
            )
            observed["prompt_rows"] = [
                {
                    **{
                        key: item[key]
                        for key in ("ids", "slots", "nopts", "types")
                        if key in item
                    },
                    "padded_sequence_length": ((len(item["ids"]) + 63) // 64) * 64,
                }
                for item in local["items"]
            ]

    return observe


def validate_observation(observed, args, profile):
    if not observed or len(observed["row_probabilities"]) != len(
        observed["prompt_rows"]
    ):
        raise RuntimeError("Missing original unrounded row probabilities")
    if set(observed["readout_dtypes"]) != {"torch.float32"}:
        raise RuntimeError("Author readout returned an unexpected dtype")
    batch_sizes = sorted({shape[0] for shape in observed["forward_input_shapes"]})
    if args.max_fwd_tokens == 1 and batch_sizes != [1]:
        raise RuntimeError("Original reference did not use one-row batches")
    for key in ("linear_input_dtypes", "linear_weight_dtypes"):
        if set(observed[key]) != {"torch." + args.dtype}:
            raise RuntimeError("Author readout operands used an unexpected dtype")
    if profile["typed_temperatures"] and len(observed["row_temperatures"]) != len(
        observed["row_probabilities"]
    ):
        raise RuntimeError("Missing original per-row temperature observations")
    return batch_sizes[0] if len(batch_sizes) == 1 else batch_sizes


def infer(args, profile, report, cases, sdk):
    # Use a fresh child process. Existing modules or pre-existing bytecode must
    # not substitute for the verified author sources copied into this directory.
    if any(
        name == "decider" or name.startswith(("decider.", "vllm_jev"))
        for name in sys.modules
    ):
        raise RuntimeError("Run the independent reference in a fresh process")
    os.environ.update(
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false"
    )
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(sdk))
    guard = BlockProductImports()
    sys.meta_path.insert(0, guard)
    try:
        import torch
        from decider.infer import Decider

        temperature_function = None
        if profile["typed_temperatures"]:
            from decider.temperature import scaled_softmax

            temperature_function = scaled_softmax
        if (
            Path(sys.modules["decider.infer"].__file__).resolve()
            != sdk / "decider/infer.py"
        ):
            raise RuntimeError("Author module did not load from isolated pinned source")
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        torch.manual_seed(0)
        load_start = time.perf_counter()
        author = Decider(
            str(args.source),
            device="cpu",
            dtype=getattr(torch, args.dtype),
            use_graphs=False,
        )
        report["load_seconds"] = time.perf_counter() - load_start
        devices = sorted({str(parameter.device) for parameter in author.m.parameters()})
        dtypes = sorted({str(parameter.dtype) for parameter in author.m.parameters()})
        if devices != ["cpu"] or dtypes != ["torch." + args.dtype]:
            raise RuntimeError(
                "Reference loaded an unexpected parameter device or dtype"
            )
        runtime = {
            "threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(),
            "dtype": args.dtype,
            "parameter_devices": devices,
            "parameter_dtypes": dtypes,
            "actual_parameter_devices": devices,
            "use_graphs": False,
            "graphs": False,
            "actual_embedding_dtype": str(
                author.m.lm.get_input_embeddings().weight.dtype
            ),
            "actual_head_dtype": str(author.m.lm.lm_head.weight.dtype),
            "padding": "right_to_multiple_of_64",
            "padding_multiple": 64,
            "independent": True,
            "isolated_levels": author.isolated_levels,
            "layout": "state_first",
            "temperature": author.T,
            "max_state_tokens": args.max_state_tokens,
            "max_fwd_tokens": args.max_fwd_tokens,
            "config_use_cache": author.m.lm.config.use_cache,
            "cross_request_cache": False,
            "raw_probabilities_capture": "original_return_frame.flatp",
        }
        if profile["typed_temperatures"]:
            runtime.update(
                model_layout=author.layout,
                chat_template_applied=author.chat is not None,
                temperature_by_type=author.T_by_type,
            )
        report.update(runtime=runtime, memory_after_load=memory_sample(), measured=True)
        save(args.output, report)
        for case in cases:
            observed = {}
            observe = observer(author, Decider, torch, observed, temperature_function)
            previous = sys.getprofile()
            started = time.perf_counter()
            try:
                sys.setprofile(observe)
                response = author.system_one(
                    case["state"],
                    case["questions"],
                    independent=True,
                    max_state_tokens=args.max_state_tokens,
                    max_fwd_tokens=args.max_fwd_tokens,
                    layout="state_first",
                    isolated=True,
                )
            finally:
                sys.setprofile(previous)
            runtime.update(
                row_batch_size=validate_observation(observed, args, profile),
                actual_readout_dtype="torch.float32",
                actual_linear_input_dtype="torch." + args.dtype,
                actual_linear_weight_dtype="torch." + args.dtype,
            )
            record = {
                **case,
                "response": response,
                "answers": response["answers"],
                "seconds": time.perf_counter() - started,
                "memory": memory_sample(),
                **observed,
            }
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
    parser.add_argument(
        "--model", choices=list(PROFILES), default="Mapika/decider-0.8b"
    )
    parser.add_argument("--source", type=Path)
    parser.add_argument("--sdk-source", type=Path, required=True)
    parser.add_argument("--cases-json", "--input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--max-state-tokens", type=int, default=32768)
    parser.add_argument("--max-fwd-tokens", type=int, default=1)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--check-sdk-only", action="store_true")
    args = parser.parse_args()
    if min(args.threads, args.max_state_tokens, args.max_fwd_tokens) < 1:
        parser.error("threads and token limits must be positive")
    if not args.check_sdk_only and (args.source is None or args.cases_json is None):
        parser.error("--source and --cases-json are required unless --check-sdk-only")
    for key in ("source", "sdk_source", "cases_json", "output"):
        if getattr(args, key) is not None:
            setattr(args, key, getattr(args, key).resolve())
    if args.output == args.cases_json or any(
        root is not None and args.output.is_relative_to(root)
        for root in (args.source, args.sdk_source)
    ):
        parser.error(
            "output must be outside the checkpoint/SDK and must not overwrite input cases"
        )
    return args


def main():
    args = parse_args()
    profile = PROFILES[args.model]
    started = time.perf_counter()
    report = {
        "status": "running",
        "model_id": args.model,
        "revision": profile["revision"],
        "reference_kind": profile["reference_kind"],
        "measured": False,
        "environment": environment(),
        "driver_sha256": sha256(__file__),
        "cases": [],
    }
    try:
        report.update(
            sdk_source=str(args.sdk_source),
            sdk_hashes=verify_sdk(args.sdk_source, profile),
        )
        if args.check_sdk_only:
            report["status"] = "source_checked"
            return
        report.update(
            source=str(args.source),
            checkpoint_hashes=verify_checkpoint(args.source, profile),
        )
        cases = read_cases(args.cases_json)
        report.update(
            input_sha256=sha256(args.cases_json),
            selected_case_ids=[case["id"] for case in cases],
        )
        if args.check_only:
            report["status"] = "source_checked"
            return
        # The acceptance parent sets cwd to its owned temporary workspace, so a
        # forced child termination also leaves this SDK copy within its cleanup.
        with tempfile.TemporaryDirectory(
            prefix="decider-sdk-", dir=Path.cwd()
        ) as folder:
            sdk = Path(folder)
            copy_sdk(args.sdk_source, sdk, profile)
            report.update(sdk_source_isolated=True, sdk_execution_path=str(sdk))
            infer(args, profile, report, cases, sdk)
            verify_sdk(sdk, profile)
        report["source_unchanged"] = (
            verify_checkpoint(args.source, profile) == report["checkpoint_hashes"]
        )
        report["sdk_unchanged"] = (
            verify_sdk(args.sdk_source, profile) == report["sdk_hashes"]
        )
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
