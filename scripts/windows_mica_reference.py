"""Independent pinned Mica Q5 reference; never imports product inference code.

Only original author modules execute. Profiling observes returned logits and
actual prompt token IDs without replacing functions, operators, or probabilities.
Run through check_windows_mica.py for the owned-process memory watchdog.
"""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import sys
import tempfile
import time
from pathlib import Path

MODEL_ID = "sky7350/Mica-v0.1-4B"
REVISION = "9b6428b272f4708ef54a4702b52bb453a4059363"
SDK_REVISION = "f6bccd76e9e42b9193d1cdf3549b971e7b200d59"
RUNTIME_REVISION = "b11010"
WEIGHT = "mica-v0.1-4b-Q5_K_M.gguf"
TEMPERATURE = 1.1244734010661372
# Populated from independently verified original HF/official CPU release bytes.
SOURCE_HASHES = {
    "runtime/LICENSE-LLVM-OpenMP": "fdad1758a9e1f9d5a81e18879b3406772115edc92c24bfa36b70c654f325e8e4",
    "calibration.json": "90076b77fd398dc42123dd970933d5af17ebca8cc9d8de6e2225bc2d0c76fbca",
    "chat_template.jinja": "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715",
    "config.json": "a9e9564540d0ab364e5aa08eb2a63d3f9ca3ba3216734c4c14f0ea629328378b",
    "tokenizer.json": "87a7830d63fcf43bf241c3c5242e96e62dd3fdc29224ca26fed8ea333db72de4",
    "tokenizer_config.json": "83febe739ecefd99f984f546dfdc10149cedf293b9ee0291b99894341188b613",
    "LICENSE": "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30",
    "NOTICE": "ce0b725c0171063827818bb5c914f7884e20eb2f105c7cd06bcc1e567edca30f",
    "mica-v0.1-4b-Q5_K_M.gguf": "7fbd1be2293ba158157bbbdcdc4ea57569bd9ed53901ed1766f81264d9da8292",
    "runtime/ggml-base.dll": "cd2603e0ba72b57f830455293185748a998be147b380f16965a973f484ada9e5",
    "runtime/ggml-cpu-alderlake.dll": "2bee7689149fcf3f0112002a85eca55c279c5c088a584dd83cd2433d9312bc15",
    "runtime/ggml-cpu-cannonlake.dll": "5124c61d027dfcb82a0ed988dfbf4c0ea7ec95f692d4574a824fb77c2e18eeec",
    "runtime/ggml-cpu-cascadelake.dll": "c6a79ad792b7e18002889c8b4d0c8fde7cd7b9f24344f08000a381ef9af23dcf",
    "runtime/ggml-cpu-cooperlake.dll": "9c708128b5a58005144d74f7224d5d1e56b4719151b63d70f9d88943e3c07bd1",
    "runtime/ggml-cpu-haswell.dll": "355dbeacd74b59a49129db49f9abfbf611acd7a9db827c8b4d7922b75070a34e",
    "runtime/ggml-cpu-icelake.dll": "bb96cf1a3324a87cbe43c44f3c30e2bacc4e0e1f0f11f4745eee5847353bcef1",
    "runtime/ggml-cpu-ivybridge.dll": "dae774e6f52b977e357bc3d97b93df05a00d0b397d71755962e237498beb3a7b",
    "runtime/ggml-cpu-piledriver.dll": "9ef3d5ee166de057bcc1b9276f42efeb54c0b58e6772f80c8243e750d7109c35",
    "runtime/ggml-cpu-sandybridge.dll": "9fdedb44733dbbec53c7d1087ccc7270a0e248bd89ae5e71c14f78206ef256a0",
    "runtime/ggml-cpu-sapphirerapids.dll": "f2246c1c47a2a8a3749cd74d71e5dd2a78577a5c68cbb0403175a7b08495cbda",
    "runtime/ggml-cpu-skylakex.dll": "4d57d36144f42d7c29b2a67d88ad7251c9dd1776bb6ef954bcf142d5b3c08f73",
    "runtime/ggml-cpu-sse42.dll": "597e4a12ef2b28ae1f5d5a0af1faa5dcc53212bdc4b15f966abac71f47239306",
    "runtime/ggml-cpu-x64.dll": "5ed04a90b6193377ba14e8c1dca88dfeb0544dae196d610d7a92b268e20543d8",
    "runtime/ggml-cpu-zen4.dll": "e051b474a192064f03b831cb223087f7df0daa25861658790f2ae88dd243805f",
    "runtime/ggml.dll": "52c375a006ca3aa802f47c50b883b29c9ba41c7c6dfc2a96a718beb4ed944b40",
    "runtime/libomp.dll": "a12116ba72d1d6820407cf30be23da04ce79d6bb8a71a5ee71759c5a1faa6f1c",
    "runtime/llama.dll": "6bb4edebf33785cf808dd7fc3964096317df5df63cc44c8918967903edabe633",
}
SDK_HASHES = {
    "mica/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "mica/calibration.py": "0c1a753df2c519c78760f83491cef26e617a7dae8d2bb4506e88752a2bb1d9c5",
    "mica/codebook.py": "9b87e060dbe5183eed31ab0c2eef97520df60f91a6d8596cc9822d470294560f",
    "mica/codebook_255.json": "9fa505527fb37fffa4b43ad4ef6d73a92f8f80455c98384734cb540cd16907f4",
    "mica/direct_backend.py": "91cdc4373632478061c8261abeb1b572c0c5a79903985ecdcc135f5ee61abae1",
    "mica/native.py": "92b1303bddaab3f56fc4956d2ca3f65c2576f0e871f636ee19df9421aed5d2c8",
    "mica/typesafe_server.py": "a8a32ce443d64ffebd6bb28fb6ebde35c335d1a5e80432ffe213b0ff7c0f30e9",
}
PACKAGES = ("transformers", "tokenizers", "huggingface_hub", "numpy", "jinja2", "torch")


def package_versions():
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )


def strict_json(text):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(f"nonfinite JSON: {value}")

    return json.loads(text, object_pairs_hook=unique, parse_constant=invalid)


def ordered(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def verify(root, expected):
    if not expected:
        raise ValueError("release hashes have not been pinned")
    observed = {}
    for name, checksum in expected.items():
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"missing or linked source file: {name}")
        observed[name] = digest(path)
        if observed[name] != checksum:
            raise ValueError(f"SHA256 differs: {name}")
    return observed


def verify_source(root):
    root = Path(root)
    if not SOURCE_HASHES:
        raise ValueError("release hashes have not been pinned")
    for entry in root.rglob("*"):
        relative = entry.relative_to(root)
        if relative.parts[0] == ".cache":
            continue
        name = relative.as_posix()
        if entry.is_symlink():
            raise ValueError(f"linked source entry: {name}")
        if entry.is_dir() and name == "runtime":
            continue
        if entry.is_file() and (
            name in SOURCE_HASHES
            or name in {"README.md", "LICENSE", "NOTICE", ".gitattributes"}
        ):
            continue
        raise ValueError(f"unexpected source entry: {name}")
    return verify(root, SOURCE_HASHES)


def verify_sdk(root):
    for entry in (Path(root) / "mica").rglob("*"):
        name = entry.relative_to(root).as_posix()
        if "__pycache__" in entry.parts:
            continue
        if entry.is_symlink() or not entry.is_file() or name not in SDK_HASHES:
            raise ValueError(f"unexpected author package entry: {name}")
    return verify(Path(root), SDK_HASHES)


def clone_pinned(source, target, expected, *, link_weight=False):
    target.mkdir(parents=True)
    for name in expected:
        destination = target / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if link_weight and name == WEIGHT:
            try:
                os.link(source / name, destination)
                continue
            except OSError:
                pass
        shutil.copyfile(source / name, destination)
    verify(target, expected)


def validate_cases(cases):
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases must be a nonempty list")
    ids = set()
    for case in cases:
        if not isinstance(case, dict) or set(case) != {"id", "state", "questions"}:
            raise ValueError("case must contain id/state/questions")
        if not isinstance(case["id"], str) or not case["id"] or case["id"] in ids:
            raise ValueError("case IDs must be unique nonempty strings")
        ids.add(case["id"])
        if not isinstance(case["state"], (str, dict, list)):
            raise ValueError("unsupported state")
        questions = case["questions"]
        if not isinstance(questions, dict) or not 1 <= len(questions) <= 64:
            raise ValueError("case needs 1..64 questions")
        for identifier, question in questions.items():
            if (
                not isinstance(identifier, str)
                or not identifier
                or not isinstance(question, dict)
            ):
                raise ValueError("invalid question")
            if not isinstance(question.get("instructions"), str):
                raise ValueError("instructions must be text")
            kind, criteria = question.get("type"), question.get("criteria")
            if kind == "noul":
                if criteria is not None and (
                    not isinstance(criteria, dict)
                    or not set(criteria).issubset({"false", "true"})
                    or any(not isinstance(value, str) for value in criteria.values())
                ):
                    raise ValueError("invalid Noul criteria")
            elif kind == "choice":
                if (
                    not isinstance(criteria, dict)
                    or not 2 <= len(criteria) <= 255
                    or any(
                        not isinstance(key, str) or not isinstance(value, str)
                        for key, value in criteria.items()
                    )
                ):
                    raise ValueError("invalid Choice criteria")
            elif kind == "score":
                if (
                    not isinstance(criteria, list)
                    or not 2 <= len(criteria) <= 10
                    or any(not isinstance(value, str) for value in criteria)
                ):
                    raise ValueError("invalid Score criteria")
            else:
                raise ValueError("unsupported question type")
    ordered(cases)
    return cases


def observe_judge(judge, events):
    logits_code, judge_code = (
        judge._logits_at.__func__.__code__,
        judge._judge.__func__.__code__,
    )

    def observe(frame, event, result):
        if event != "return":
            return
        if frame.f_code is logits_code and result is not None:
            events["logits"].append(list(result))
        elif frame.f_code is judge_code and result is not None:
            events["prompts"].extend([list(ids) for ids in frame.f_locals["prompts"]])
            events["labels"].extend([list(ids) for ids in frame.f_locals["labels"]])

    return observe


def configure_tokenizer_only():
    """Require a real tokenizer-only environment, not an availability override."""
    if any(
        name in {"torch", "transformers"}
        or name.startswith(("torch.", "transformers."))
        for name in sys.modules
    ):
        raise RuntimeError("tokenizer-only reference requires a fresh process")
    if os.environ.get("KMP_DUPLICATE_LIB_OK", "").upper() in {"1", "TRUE", "YES", "ON"}:
        raise RuntimeError("OpenMP duplicate-library bypass is not allowed")
    for framework in ("torch", "tensorflow", "jax"):
        if importlib.util.find_spec(framework) is not None:
            raise RuntimeError(f"reference requires {framework} to be uninstalled")


def tokenizer_runtime():
    imported = any(name == "torch" or name.startswith("torch.") for name in sys.modules)
    installed = imported or importlib.util.find_spec("torch") is not None
    if imported or installed:
        raise RuntimeError("Torch entered the tokenizer-only reference process")
    return {
        "tokenizer_runtime": "transformers-tokenizer-only",
        "torch_imported": imported,
        "torch_installed": installed,
    }


def infer(source, sdk, cases, threads, report, output=None, runtime_dir=None):
    if any(
        name == "mica" or name.startswith(("mica.", "vllm_jev")) for name in sys.modules
    ):
        raise RuntimeError("reference must start without author/product modules loaded")
    configure_tokenizer_only()
    sys.path.insert(0, str(sdk))
    from mica.direct_backend import DirectJudge
    from mica.native import prompt_for
    from mica.typesafe_server import answer, rows_from_request
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        source, local_files_only=True, trust_remote_code=False
    )
    tokenizer_runtime()
    # Keep the Windows search-directory handle alive; no author code is patched.
    runtime_dir = source / "runtime" if runtime_dir is None else runtime_dir
    dll_directory = (
        os.add_dll_directory(str(runtime_dir.resolve())) if os.name == "nt" else None
    )
    judge = None
    started = time.perf_counter()
    try:
        judge = DirectJudge.from_calibration(
            source / WEIGHT,
            tokenizer,
            source / "calibration.json",
            runtime_dir=runtime_dir,
            n_gpu_layers=0,
            n_threads=threads,
            n_ctx=1024,
            n_seq_max=1,
            max_length=1024,
            n_ubatch=512,
            flash_attn=-1,
        )
        report["load_seconds"] = time.perf_counter() - started
        report["runtime"] = {
            **tokenizer_runtime(),
            "runtime": "llama.cpp",
            "runtime_revision": RUNTIME_REVISION,
            "quantization": "Q5_K_M",
            "device": "cpu",
            "n_gpu_layers": 0,
            "threads": threads,
            "context_tokens": judge.n_ctx_seq,
            "n_seq_max": judge.n_seq_max,
            "n_batch": judge.n_batch,
            "n_ubatch": 512,
            "flash_attn": -1,
            "max_sequence_tokens": judge.max_length,
            "share_prefix": False,
            "session_id": None,
            "row_batch_size": 1,
            "temperature": judge.temperature,
            "temperatures": judge.temperatures,
            "noul_logit_bias": judge.noul_logit_bias,
            "backend_registration_count": int(judge.lib.ggml.ggml_backend_reg_count()),
        }
        if (
            judge.n_seq_max != 1
            or judge.n_ctx_seq != 1024
            or judge.temperature != TEMPERATURE
        ):
            raise RuntimeError("actual author runtime settings differ")
        for case in cases:
            rows = rows_from_request(case)
            prompts = [prompt_for(row, tokenizer) for row in rows]
            row_tokens = [
                judge.tokenize(prompt, add_special=False) for prompt in prompts
            ]
            if any(not ids or len(ids) > 1024 for ids in row_tokens):
                raise ValueError("case exceeds fixed complete-prompt context")
            result = {
                **case,
                "probabilities": {},
                "raw_logits": {},
                "evidence": {},
                "author_answers": {},
            }
            case_started = time.perf_counter()
            for row, prompt, ids in zip(rows, prompts, row_tokens):
                events = {"logits": [], "prompts": [], "labels": []}
                previous = sys.getprofile()
                try:
                    sys.setprofile(observe_judge(judge, events))
                    probabilities = judge.judge(
                        [row], share_prefix=False, session_id=None
                    )[0]
                finally:
                    sys.setprofile(previous)
                if (
                    len(events["logits"]) != 1
                    or events["prompts"] != [ids]
                    or len(events["labels"]) != 1
                ):
                    raise RuntimeError(
                        "did not observe exactly one unmodified author readout"
                    )
                identifier = row["id"]
                result["probabilities"][identifier] = probabilities
                result["raw_logits"][identifier] = events["logits"][0]
                result["evidence"][identifier] = {
                    "prompt": prompt,
                    "token_ids": ids,
                    "label_ids": events["labels"][0],
                    "row": row,
                    "prefix_cache_stats": dict(judge.prefix_cache_stats),
                }
                result["author_answers"][identifier] = answer(row, probabilities)
            result["seconds"] = time.perf_counter() - case_started
            report["runtime"].update(tokenizer_runtime())
            report["cases"].append(result)
            if output is not None:
                save(output, report)
            print(
                json.dumps(
                    {
                        "event": "author_case_completed",
                        "case": case["id"],
                        "seconds": result["seconds"],
                    }
                ),
                flush=True,
            )
    finally:
        if judge is not None:
            judge.close()
            report["judge_closed"] = judge.ctx is None and judge.model is None
        if dll_directory is not None:
            dll_directory.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "sdk-source", "cases-json", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    for name in ("source", "sdk_source", "cases_json", "output"):
        setattr(args, name, getattr(args, name).resolve())
    if not 1 <= args.threads <= min(4, os.cpu_count() or 1):
        parser.error("threads must be 1..min(4, CPU count)")
    if args.output == args.cases_json or any(
        args.output.is_relative_to(root) for root in (args.source, args.sdk_source)
    ):
        parser.error("output must be outside source/SDK and distinct from input")
    if not args.check_only and sys.platform != "win32":
        parser.error("native Windows is required")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cases = validate_cases(strict_json(args.cases_json.read_text(encoding="utf-8")))
    report = {
        "status": "running",
        "measured": False,
        "model_id": MODEL_ID,
        "revision": REVISION,
        "sdk_revision": SDK_REVISION,
        "runtime_revision": RUNTIME_REVISION,
        "source": str(args.source),
        "sdk_source": str(args.sdk_source),
        "driver_sha256": digest(Path(__file__)),
        "input_sha256": digest(args.cases_json),
        "environment_identity": {"python": sys.version, "executable": sys.executable},
        "platform": platform.platform(),
        "versions": package_versions(),
        "cases": [],
        "source_unchanged": False,
        "sdk_unchanged": False,
    }
    before = sdk_before = None
    try:
        before = verify_source(args.source)
        sdk_before = verify_sdk(args.sdk_source)
        report.update(source_hashes=before, sdk_hashes=sdk_before)
        if args.check_only:
            report["status"] = "checked"
        else:
            os.environ.update(
                HF_HUB_OFFLINE="1",
                TRANSFORMERS_OFFLINE="1",
                PYTHONDONTWRITEBYTECODE="1",
            )
            sys.dont_write_bytecode = True
            with tempfile.TemporaryDirectory(prefix="mica-author-") as directory:
                private = Path(directory)
                checkpoint, sdk = private / "checkpoint", private / "author"
                model_hashes = {
                    name: checksum
                    for name, checksum in SOURCE_HASHES.items()
                    if not name.startswith("runtime/")
                }
                # ctypes DLLs can stay mapped until this child exits. Load the
                # fully pinned original runtime read-only, not from a directory
                # this child tries to remove. Never manually unload C libraries.
                report.update(
                    runtime_source=str(args.source / "runtime"),
                    runtime_source_isolated=False,
                    isolated_source_hashes=model_hashes,
                )
                clone_pinned(args.source, checkpoint, model_hashes, link_weight=True)
                clone_pinned(args.sdk_source, sdk, SDK_HASHES)
                infer(
                    checkpoint,
                    sdk,
                    cases,
                    args.threads,
                    report,
                    args.output,
                    args.source / "runtime",
                )
                report["isolated_source_unchanged"] = (
                    verify(checkpoint, model_hashes) == model_hashes
                )
                report["isolated_sdk_unchanged"] = verify_sdk(sdk) == sdk_before
            report.update(
                status="passed",
                measured=True,
                private_workspace_removed=not private.exists(),
            )
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        try:
            if before is not None:
                report["source_unchanged"] = verify_source(args.source) == before
            if sdk_before is not None:
                report["sdk_unchanged"] = verify_sdk(args.sdk_source) == sdk_before
        except BaseException as error:
            report.update(source_unchanged=False, integrity_error=str(error))
        if not report["source_unchanged"] or not report["sdk_unchanged"]:
            report["status"] = "failed"
        save(args.output, report)


if __name__ == "__main__":
    main()
