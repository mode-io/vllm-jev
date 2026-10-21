"""Run the unmodified, pinned Decider 4B Q4_K_M author SDK on Windows CPU.

This independent runner never imports the product, downloads files, installs
packages, or substitutes author callables. It copies only seven verified SDK
files and observes original return frames with sys.setprofile. The acceptance
parent owns the child process and enforces the available-memory watchdog.
--check-only verifies assets without importing Torch or loading a model.
"""

import argparse
import hashlib
import importlib.abc
import importlib.metadata
import json
import math
import os
import shutil
import sys
import tempfile
import time
import traceback
import zipfile
from pathlib import Path

MODEL_ID = "Mapika/decider-4b-GGUF"
REVISION = "b79f09d9ba7837f1b744295ea267b55d08e958ec"
SDK_REVISION = "d6fc522aed7fc392576a9b66a330409bc6c382bd"
GGUF_FILE = "decider-4b-v2.1-Q4_K_M.gguf"
GGUF_BYTES = 2_708_804_640
WHEEL_SHA256 = "31590ea000d5aff6f05f1e428048e72318a83709288159a5bd4dabec530080bb"
WHEEL_BYTES = 7_086_788
TEMPERATURES = {"choice": 1.11, "noul": 1.56, "score": 1.287}
SDK_HASHES = {
    "decider/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "decider/infer.py": "a58e670c56f42855af94b8041b3bbd806f168d6b8aecf5c24776f069256aaa88",
    "decider/model.py": "74e0ca564bb52f7c2bd74c23cb34b16e0004ceadf7a13816f223c68829b5e1c5",
    "decider/prompt.py": "f21ae1016a11ca3500b5a8e2e8c3ef524434d196bbd4891199b96a33b3ad6314",
    "decider/systemone.py": "884e4b0bae36f5e09dbd689b4390e94320cbbd77f7b8d675e50490b1d7c61d34",
    "decider/temperature.py": "1dc5689164c0cc8482e46706b07547597c64c432231153aa0885e4fe11de1006",
    "decider/engine_gguf.py": "9db9ac5f4147d6236f5071edf08dcbe578df21079fe1cd10853a027aa5d7982a",
}
# Git's Windows checkout may use CRLF. These are independently checked against
# the exact commit's LF blobs; no source is normalised or rewritten at runtime.
SDK_CRLF_HASHES = {
    "decider/__init__.py": SDK_HASHES["decider/__init__.py"],
    "decider/infer.py": "1c7a03ebfea98f9065129177dc5b16eab80c30c9868fe1bc1559aaeb59c0fbbc",
    "decider/model.py": "25fba30ecb745c2e745f194409308eeb2915927ba2a5fbcf12d611ca544435cd",
    "decider/prompt.py": "d8856e0d0dbc2e50a055b4037efb9bdff932eed7b2fd413ebf1e0eefd138935e",
    "decider/systemone.py": "2b7e9fbfc520b77a18040ece9355f8e1e9f285ec01e735e5597362ea802526c7",
    "decider/temperature.py": "5483a033be60208ac1d27a9e05f10fe04a5c6df1b96e9e6fa0cc81162ce254c5",
    "decider/engine_gguf.py": "59167e4816f7283905bba4797b2b0459e7a542aa6e1d9213f33cc1605a9173e5",
}
SOURCE_HASHES = {
    GGUF_FILE: "c7083fcfc93f650cd66caeade4a3840ef9eeeeb1e80920907c554f70619d7c56",
    "decider_config.json": "fc83293b6f707172e20176d603ea88dd1a3be2abd3596fdb1ccbec9404626baf",
    "tokenizer_config.json": "171ecbe7ddae98d11840698f7df2b8d5b4722139db0f0620d3bbf429bd656250",
    "tokenizer.json": "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
    "chat_template.jinja": "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715",
    "README.md": "7fa5815c0b78d6fee9febdd8d734dbe95bd41138f7421ea28db0622d12cac7c2",
}


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def regular_file(root, name):
    path = root / name
    for entry in (path, *path.parents):
        if entry == root.parent:
            break
        if entry.is_symlink() or entry.is_junction():
            raise ValueError(f"Pinned inputs cannot use links: {entry}")
    if not path.is_file():
        raise ValueError(f"Missing pinned file: {path}")
    return path


def verify_sdk(root, isolated=False):
    actual = {}
    for name, expected in SDK_HASHES.items():
        value = sha256(regular_file(root, name))
        if value not in (expected, SDK_CRLF_HASHES[name]):
            raise ValueError(f"Pinned author SDK SHA256 mismatch: {name}")
        actual[name] = value
    if isolated:
        files = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
        if files != set(SDK_HASHES):
            raise ValueError("Private SDK contains files outside the verified seven")
    return actual


def copy_sdk(root, target):
    before = verify_sdk(root)
    for name in SDK_HASHES:
        path = target / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, path)
    if verify_sdk(target, isolated=True) != before:
        raise RuntimeError("Private SDK copy differs from the verified source")
    return before


def verify_checkpoint(root):
    for path in root.iterdir():
        if path.name in SOURCE_HASHES and path.is_file():
            continue
        if path.name == ".cache" and path.is_dir() and not path.is_symlink():
            continue
        if path.name in {"README.md", "LICENSE", ".gitattributes"} and path.is_file():
            continue
        raise ValueError(f"Unexpected checkpoint loader input: {path}")
    actual = {name: sha256(regular_file(root, name)) for name in SOURCE_HASHES}
    if actual != SOURCE_HASHES:
        raise ValueError("Pinned checkpoint SHA256 mismatch")
    if (root / GGUF_FILE).stat().st_size != GGUF_BYTES:
        raise ValueError("Pinned GGUF file size mismatch")
    cfg = json.loads((root / "decider_config.json").read_text(encoding="utf-8"))
    expected = {
        "temperature": 1.099,
        "temperature_by_type": TEMPERATURES,
        "version": "4b-v2.1",
        "neutralize_none": False,
        "layout": "plain",
        "max_options": 255,
        "schema_first": False,
        "isolated_levels": True,
    }
    if any(cfg.get(key) != value for key, value in expected.items()):
        raise ValueError("Unexpected pinned Decider GGUF calibration")
    return actual


def verify_llama_wheel(wheel):
    if "LLAMA_CPP_LIB_PATH" in os.environ:
        raise ValueError("Reference does not permit LLAMA_CPP_LIB_PATH overrides")
    if wheel.stat().st_size != WHEEL_BYTES or sha256(wheel) != WHEEL_SHA256:
        raise ValueError(
            "Expected the official llama-cpp-python 0.3.35 Windows CPU wheel"
        )
    distribution = importlib.metadata.distribution("llama-cpp-python")
    if distribution.version != "0.3.35":
        raise ValueError("Independent reference requires llama-cpp-python==0.3.35")
    package = Path(distribution.locate_file("llama_cpp")).resolve()
    expected = {}
    with zipfile.ZipFile(wheel) as archive:
        for item in archive.infolist():
            if not item.filename.startswith("llama_cpp/") or item.is_dir():
                continue
            name = item.filename.removeprefix("llama_cpp/")
            if ".." in Path(name).parts or Path(name).is_absolute():
                raise ValueError("Invalid path in pinned wheel")
            with archive.open(item) as stream:
                expected[name] = hashlib.file_digest(stream, "sha256").hexdigest()
    if not expected or "lib/llama.dll" not in expected:
        raise ValueError("Pinned wheel has no Windows llama runtime")
    for name, expected_hash in expected.items():
        if sha256(regular_file(package, name)) != expected_hash:
            raise ValueError(
                f"Installed llama_cpp differs from the official wheel: {name}"
            )
    for path in package.rglob("*"):
        if path.is_file() and path.relative_to(package).as_posix() not in expected:
            if "__pycache__" in path.parts and path.suffix == ".pyc":
                continue
            raise ValueError(f"Unexpected installed llama_cpp file: {path}")
    return {
        "version": distribution.version,
        "wheel_sha256": WHEEL_SHA256,
        "package_path": str(package),
        "package_hashes": expected,
    }


def read_cases(path):
    def invalid(value):
        raise ValueError(f"Nonfinite input JSON: {value}")

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
        raise ValueError("Cases JSON must be a nonempty array or {cases: [...]}")
    ids = set()
    for case in data:
        if (
            not isinstance(case, dict)
            or not {"id", "state", "questions"} <= case.keys()
        ):
            raise ValueError("Every case requires id, state, questions")
        if not isinstance(case["id"], str) or not case["id"] or case["id"] in ids:
            raise ValueError("Case ids must be unique nonempty strings")
        if not isinstance(case["state"], (str, dict, list)):
            raise ValueError("Case state must be text, a JSON object, or an array")
        if not isinstance(case["questions"], dict) or not case["questions"]:
            raise ValueError("Every case requires a nonempty questions map")
        ids.add(case["id"])
    return data


def memory_sample():
    import psutil

    info = psutil.Process().memory_info()
    return {
        key: getattr(info, key)
        for key in ("rss", "vms", "peak_wset", "private", "peak_pagefile")
        if hasattr(info, key)
    }


def native_libraries():
    import psutil

    names = ("ggml", "llama", "mtmd", "libomp", "libiomp", "vcomp")
    return sorted(
        {
            item.path
            for item in psutil.Process().memory_maps()
            if Path(item.path).name.lower().startswith(names)
            and item.path.lower().endswith(".dll")
        }
    )


def save(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def environment():
    packages = {}
    for name in (
        "torch",
        "transformers",
        "numpy",
        "huggingface_hub",
        "tokenizers",
        "safetensors",
        "jinja2",
        "psutil",
        "llama-cpp-python",
        "diskcache",
        "flash-linear-attention",
        "triton",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": sys.version, "executable": sys.executable, "packages": packages}


class BlockProductImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] == "vllm_jev":
            raise RuntimeError(f"Reference cannot import product code: {fullname}")
        return None


def observer(author, decider, torch, softmax, observed):
    """Observe raw original returns; no module, function, or tensor is replaced."""
    decode_code = author.eng._decode.__func__.__code__

    def observe(frame, event, result):
        if event != "return":
            return
        if frame.f_code is decode_code and result is not None:
            observed.setdefault("decode_calls", []).append(
                {
                    "rows": [
                        {"ids": list(ids), "slots": list(slots)}
                        for ids, slots in frame.f_locals["rows"]
                    ],
                    "raw_letter_logits": result.tolist(),
                    "readout_dtype": str(result.dtype),
                    "forward_count": author.eng.stats["forwards"],
                }
            )
        if frame.f_code is softmax.__code__ and isinstance(result, torch.Tensor):
            value = frame.f_locals["temperature"]
            observed.setdefault("row_temperatures", []).extend(
                list(value)
                if isinstance(value, (list, tuple))
                else [value] * result.shape[0]
            )
            observed.setdefault("softmax_calls", []).append(
                {
                    "shape": list(result.shape),
                    "input_dtype": str(frame.f_locals["lg"].dtype),
                    "output_dtype": str(result.dtype),
                    "device": str(result.device),
                }
            )
        if frame.f_code is decider.system_one.__code__ and isinstance(result, dict):
            local = frame.f_locals
            observed.update(
                row_probabilities=local["flatp"],
                row_index=local["index"],
                rendered_questions=local["rqs"],
                prompt_rows=[
                    {
                        key: item[key]
                        for key in ("ids", "slots", "nopts", "types", "perms")
                    }
                    for item in local["items"]
                ],
            )

    return observe


def canonical_answers(observed):
    """Independent project-shaped presentation of the author's unrounded rows."""
    answers, details = {}, {}
    rqs, rows = observed["rendered_questions"], observed["row_probabilities"]
    for identifier, mode, start, count in observed["row_index"]:
        question = rqs[identifier]
        if mode == "iso":
            fit = [float(rows[start + index][1]) for index in range(count)]
            mass = sum(fit)
            probabilities = [value / (mass or 1e-9) for value in fit]
            details[identifier] = {"level_fit": fit, "fit_mass": mass}
        else:
            probabilities = [
                float(value) for value in rows[start][: len(question["options"])]
            ]
        # format_answer normalises even an ordinary softmax before presentation.
        total = sum(probabilities) or 1.0
        probabilities = [value / total for value in probabilities]
        kind = question["type"]
        if kind == "noul":
            answers[identifier] = {"type": "noul", "noul": probabilities[1]}
            continue
        n = len(probabilities)
        selected = max(range(n), key=probabilities.__getitem__)
        keys = question["names"] if kind == "choice" else [str(i) for i in range(n)]
        result = {"type": kind, "probabilities": dict(zip(keys, probabilities))}
        if kind == "choice":
            total = sum(probabilities)
            confidence_p = [p / total for p in probabilities] if total else [1 / n] * n
            confidence = (n * max(confidence_p) - 1) / (n - 1)
            result.update(
                choice=keys[selected], confidence=max(0.0, min(1.0, confidence))
            )
        else:
            # The author's confidence helper uses a uniform distribution when
            # isolated fits are all zero, while its score remains zero.
            total = sum(probabilities)
            confidence_p = [p / total for p in probabilities] if total else [1 / n] * n
            mode = max(range(n), key=confidence_p.__getitem__)
            scale = sum(abs(i - (n - 1) / 2) for i in range(n)) / n
            distance = sum(p * abs(i - mode) for i, p in enumerate(confidence_p))
            result.update(
                score=sum(i * p for i, p in enumerate(probabilities)),
                confidence=max(0.0, min(1.0, 1 - distance / scale)),
                legend=dict(zip(keys, question["legend"])),
            )
        answers[identifier] = result
    return answers, details


def verify_wire(raw, canonical, details):
    """Check independent presentation against the actual author's rounded API."""
    if set(raw) != set(canonical):
        raise RuntimeError("Author wire and independently mapped question ids differ")
    for identifier, answer in canonical.items():
        original = raw[identifier]
        for key, value in answer.items():
            if key == "probabilities":
                expected = {name: round(p, 4) for name, p in value.items()}
            elif key in {"confidence", "noul"}:
                expected = round(value, 4)
            elif key == "score":
                expected = round(value, 2)
            else:
                expected = value
            if original.get(key) != expected:
                raise RuntimeError(
                    f"Independent author-output mapping differs: {identifier}.{key}"
                )
        if identifier in details:
            expected = {
                str(i): round(p, 4)
                for i, p in enumerate(details[identifier]["level_fit"])
            }
            if original.get("level_fit") != expected or original.get(
                "fit_mass"
            ) != round(details[identifier]["fit_mass"], 4):
                raise RuntimeError(
                    "Independent isolated Score details differ from author wire"
                )


def validate_observation(observed, args, preflight_rows):
    prompts = observed.get("prompt_rows", [])
    rows = observed.get("row_probabilities", [])
    decodes = observed.get("decode_calls", [])
    temperatures = observed.get("row_temperatures", [])
    if not prompts or len(prompts) != len(rows) or len(decodes) != len(prompts):
        raise RuntimeError("Missing original row/decode observations")
    if [row["ids"] for row in prompts] != [row["ids"] for row in preflight_rows]:
        raise RuntimeError("Original execution prompts differ from preflight")
    if len(temperatures) != len(rows):
        raise RuntimeError("Missing original per-row temperatures")
    for index, (prompt, probability, decode) in enumerate(zip(prompts, rows, decodes)):
        ids = prompt["ids"]
        if (
            not ids
            or len(ids) > args.max_sequence_tokens
            or prompt["slots"] != [len(ids) - 1]
        ):
            raise RuntimeError("Unexpected original answer position or prompt length")
        if len(decode["rows"]) != 1 or decode["rows"][0] != {
            "ids": ids,
            "slots": prompt["slots"],
        }:
            raise RuntimeError("Reference did not decode exactly one independent row")
        logits = decode["raw_letter_logits"]
        if (
            decode["readout_dtype"] != "float32"
            or len(logits) != 1
            or len(logits[0]) != 255
        ):
            raise RuntimeError("Unexpected original GGUF letter projection")
        if not all(math.isfinite(value) for value in logits[0]):
            raise RuntimeError("Nonfinite original GGUF logits")
        if len(probability) != 255 or not all(
            math.isfinite(p) and p >= 0 for p in probability
        ):
            raise RuntimeError("Invalid original softmax distribution")
        nopts = prompt["nopts"][0]
        if any(probability[nopts:]) or abs(sum(probability[:nopts]) - 1.0) > 1e-5:
            raise RuntimeError("Original candidate masking or normalisation differs")
        if temperatures[index] != TEMPERATURES[prompt["types"][0]]:
            raise RuntimeError("Original answer used the wrong typed temperature")
    for softmax in observed.get("softmax_calls", []):
        if (
            softmax["input_dtype"] != "torch.float32"
            or softmax["output_dtype"] != "torch.float32"
            or softmax["device"] != "cpu"
        ):
            raise RuntimeError("Original softmax must use CPU FP32")


def infer(args, report, cases, sdk):
    if any(
        name == "decider" or name.startswith(("decider.", "vllm_jev"))
        for name in sys.modules
    ):
        raise RuntimeError("Run the independent reference in a fresh process")
    if os.environ.get("KMP_DUPLICATE_LIB_OK", "").strip().lower() not in {
        "",
        "0",
        "false",
    }:
        raise RuntimeError("Do not suppress OpenMP conflicts with KMP_DUPLICATE_LIB_OK")
    os.environ.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
        CUDA_VISIBLE_DEVICES="",
        USE_TORCH="1",
    )
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(sdk))
    guard = BlockProductImports()
    sys.meta_path.insert(0, guard)
    try:
        import llama_cpp
        import torch
        from decider.infer import Decider
        from decider.systemone import render_state
        from decider.temperature import scaled_softmax

        if Path(llama_cpp.__file__).resolve().parent != Path(
            report["llama_runtime"]["package_path"]
        ):
            raise RuntimeError(
                "Imported llama_cpp is not the verified official package"
            )
        if torch.__version__.split("+")[0] != "2.8.0" or torch.version.cuda is not None:
            raise RuntimeError("Reference requires the isolated torch 2.8.0 CPU wheel")
        for name in SDK_HASHES:
            module_name = name.removesuffix(".py").replace("/", ".")
            module_name = module_name.removesuffix(".__init__")
            module = sys.modules.get(module_name)
            # engine_gguf is imported lazily by the author constructor below.
            if module is not None and Path(module.__file__).resolve() != sdk / name:
                raise RuntimeError(
                    f"Author module did not load from pinned private source: {name}"
                )
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        torch.manual_seed(0)
        started = time.perf_counter()
        author = Decider(
            str(args.source),
            device="cpu",
            dtype=torch.float32,
            use_graphs=False,
            gguf_file=GGUF_FILE,
            gguf_options={
                "n_ctx": args.n_ctx,
                "n_seq_max": 1,
                "n_gpu_layers": 0,
                "n_threads": args.threads,
            },
        )
        report["load_seconds"] = time.perf_counter() - started
        if (
            not author.gguf
            or author.layout != "plain"
            or author.chat is not None
            or author.schema_first
        ):
            raise RuntimeError("Unexpected original GGUF prompt/runtime route")
        if (
            not author.isolated_levels
            or author.neutralize_none
            or author.T != 1.099
            or author.T_by_type != TEMPERATURES
        ):
            raise RuntimeError("Original SDK did not retain the release calibration")
        if author.eng.n_seq_max != 1 or author.eng.n_ctx != args.n_ctx:
            raise RuntimeError("Original engine did not retain CPU context limits")
        for name in SDK_HASHES:
            module_name = (
                name.removesuffix(".py").replace("/", ".").removesuffix(".__init__")
            )
            if (
                module_name not in sys.modules
                or Path(sys.modules[module_name].__file__).resolve() != sdk / name
            ):
                raise RuntimeError(
                    f"Author module provenance check failed: {module_name}"
                )
        runtime = {
            "backend": "llama-cpp-python",
            "llama_cpp_python": "0.3.35",
            "quantization": "Q4_K_M",
            "device": "cpu",
            "n_gpu_layers": 0,
            "threads": args.threads,
            "torch_threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(),
            "n_ctx": args.n_ctx,
            "n_seq_max": 1,
            "independent": True,
            "row_batch_size": 1,
            "padding": "none",
            "actual_readout_dtype": "float32",
            "softmax_dtype": "torch.float32",
            "layout": "state_first",
            "model_layout": "plain",
            "chat_template_applied": False,
            "isolated_levels": True,
            "temperature": author.T,
            "temperature_by_type": author.T_by_type,
            "max_state_tokens": args.max_state_tokens,
            "max_sequence_tokens": args.max_sequence_tokens,
            "max_fwd_tokens": args.max_fwd_tokens,
            "use_graphs": False,
            "graphs": False,
            "cross_request_cache": False,
            "memory_clear_before_decode": True,
            "raw_probabilities_capture": "original_system_one_return_frame.flatp",
            "raw_logits_capture": "original_GGUFEngine._decode_return_value",
            "native_libraries": native_libraries(),
        }
        report.update(runtime=runtime, measured=True, memory_after_load=memory_sample())
        save(args.output, report)
        for case in cases:
            state_tokens = author.m.tok.encode(
                "Context:\n" + render_state(case["state"]), add_special_tokens=False
            )
            if len(state_tokens) > args.max_state_tokens:
                raise ValueError(
                    "Reference input would be silently truncated by the author state limit"
                )
            _, _, preflight = author._system_one_items(
                case["state"],
                case["questions"],
                independent=True,
                max_state_tokens=args.max_state_tokens,
                layout="state_first",
                isolated=True,
            )
            if not preflight or any(
                not it["ids"] or len(it["ids"]) > args.max_sequence_tokens
                for it in preflight
            ):
                raise ValueError(
                    f"Complete reference prompt exceeds {args.max_sequence_tokens} tokens"
                )
            before_forwards = author.eng.stats["forwards"]
            observed = {}
            previous = sys.getprofile()
            started = time.perf_counter()
            try:
                sys.setprofile(
                    observer(author, Decider, torch, scaled_softmax, observed)
                )
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
            seconds = time.perf_counter() - started
            validate_observation(observed, args, preflight)
            canonical, details = canonical_answers(observed)
            verify_wire(response["answers"], canonical, details)
            record = {
                **case,
                "response": response,
                "answers": response["answers"],
                "canonical_answers": canonical,
                "isolated_details": details,
                "seconds": seconds,
                "memory": memory_sample(),
                "forwards": author.eng.stats["forwards"] - before_forwards,
                "state_not_truncated": True,
                "preflight_complete_before_decode": True,
                **observed,
            }
            if record["forwards"] != len(preflight):
                raise RuntimeError(
                    "Original GGUF engine did not compute every row exactly once"
                )
            report["cases"].append(record)
            save(args.output, report)
            print(
                json.dumps(
                    {
                        "event": "reference_case_completed",
                        "id": case["id"],
                        "seconds": seconds,
                    }
                ),
                flush=True,
            )
        forbidden = [
            name
            for name in sys.modules
            if name.split(".")[0] in {"vllm_jev", "fla", "triton"}
        ]
        if forbidden:
            raise RuntimeError(
                f"Reference imported an unexpected product/GPU module: {forbidden}"
            )
        report.update(memory_final=memory_sample(), forbidden_modules_imported=False)
    finally:
        sys.meta_path.remove(guard)
        sys.path.remove(str(sdk))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=[MODEL_ID], default=MODEL_ID)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--sdk-source", type=Path, required=True)
    parser.add_argument("--llama-wheel", type=Path)
    parser.add_argument("--cases-json", "--input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--dtype", choices=["float32"], default="float32")
    parser.add_argument("--n-ctx", type=int, choices=(1024, 2048), default=2048)
    parser.add_argument("--max-sequence-tokens", type=int, default=1024)
    parser.add_argument("--max-state-tokens", type=int, default=32768)
    parser.add_argument("--max-fwd-tokens", type=int, default=1)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--check-sdk-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.max_sequence_tokens <= min(1024, args.n_ctx):
        parser.error("max-sequence-tokens must be between 1 and min(1024, n-ctx)")
    if min(args.max_state_tokens, args.max_fwd_tokens) < 1:
        parser.error("token limits must be positive")
    if not args.check_sdk_only and any(
        value is None for value in (args.source, args.cases_json, args.llama_wheel)
    ):
        parser.error(
            "--source, --cases-json and --llama-wheel are required unless --check-sdk-only"
        )
    for key in ("source", "sdk_source", "llama_wheel", "cases_json", "output"):
        if getattr(args, key) is not None:
            setattr(args, key, getattr(args, key).resolve())
    if args.output in (args.cases_json, args.llama_wheel) or any(
        root is not None and args.output.is_relative_to(root)
        for root in (args.source, args.sdk_source)
    ):
        parser.error(
            "output must be outside the checkpoint/SDK and must not overwrite any input"
        )
    if not (args.check_only or args.check_sdk_only) and any(
        root is not None and Path.cwd().resolve().is_relative_to(root)
        for root in (args.source, args.sdk_source)
    ):
        parser.error(
            "reference scratch working directory must be outside checkpoint/SDK inputs"
        )
    return args


def main():
    args = parse_args()
    started = time.perf_counter()
    report = {
        "status": "running",
        "model_id": MODEL_ID,
        "revision": REVISION,
        "sdk_revision": SDK_REVISION,
        "reference_kind": "original_decider_sdk_4b_q4_cpu_gguf",
        "measured": False,
        "environment": environment(),
        "driver_sha256": sha256(__file__),
        "cases": [],
    }
    try:
        report.update(
            sdk_source=str(args.sdk_source), sdk_hashes=verify_sdk(args.sdk_source)
        )
        if args.check_sdk_only:
            report["status"] = "source_checked"
            return
        report.update(
            source=str(args.source), checkpoint_hashes=verify_checkpoint(args.source)
        )
        report["llama_runtime"] = verify_llama_wheel(args.llama_wheel)
        cases = read_cases(args.cases_json)
        report.update(
            input_sha256=sha256(args.cases_json),
            selected_case_ids=[case["id"] for case in cases],
        )
        if args.check_only:
            report["status"] = "source_checked"
            return
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # The acceptance parent owns cwd, including leftovers after a forced
        # child termination; only copied, verified source executes here.
        with tempfile.TemporaryDirectory(
            prefix="decider-gguf-sdk-", dir=Path.cwd()
        ) as folder:
            sdk = Path(folder)
            copy_sdk(args.sdk_source, sdk)
            report.update(sdk_source_isolated=True, sdk_execution_path=str(sdk))
            infer(args, report, cases, sdk)
            if verify_sdk(sdk, isolated=True) != report["sdk_hashes"]:
                raise RuntimeError("Executing author source changed")
        report["source_unchanged"] = (
            verify_checkpoint(args.source) == report["checkpoint_hashes"]
        )
        report["sdk_unchanged"] = verify_sdk(args.sdk_source) == report["sdk_hashes"]
        report["llama_runtime_unchanged"] = (
            verify_llama_wheel(args.llama_wheel) == report["llama_runtime"]
        )
        if not all(
            report[key]
            for key in ("source_unchanged", "sdk_unchanged", "llama_runtime_unchanged")
        ):
            raise RuntimeError(
                "An independently verified input changed during reference execution"
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
