"""Observe the unchanged pinned Tiny-Jev-1.7B author implementation on CPU.

Original decide, encoding, head, and softmax run independently of the product.
Forward hooks only record input IDs, marker positions, and calibrated logits.
"""

import argparse
import hashlib
import importlib.abc
import importlib.metadata
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

MODEL_ID = "lostargon/Tiny-Jev-1.7B"
REVISION = "c41bcf3f191407c779bd2ebce1b992dd1469437f"
SOURCE_HASHES = {
    "model.safetensors": "49ff7a60951abe35631b9f0632b8250acdb9142f79f39a57f56aa42befac7a3b",
    "config.json": "48cd095ffe1863dd003895a38a3359be84d76ef79ec95e8e76ecf92604e28839",
    "tokenizer.json": "e3fccc77adda606e2a461ffd17ca5a55a5a75ac40a5aa0a91c5307fb31b1efac",
    "tokenizer_config.json": "5e44295ef654157e02ca593d0be8c07ed11265d4553e7920005d721ff5a9feab",
    "chat_template.jinja": "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8",
    "modeling_tiny_jev.py": "5c2d146e2f92ffcb7bb8a0927b180a3d9458b5fc3c4f5d99835302910802e05d",
}
VERSIONS = {
    "torch": "2.6.0+cpu",
    "transformers": "5.17.0",
    "tokenizers": "0.23.3",
    "safetensors": "0.8.0",
    "numpy": "2.5.4",
}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(source):
    observed = {name: digest(Path(source) / name) for name in SOURCE_HASHES}
    if observed != SOURCE_HASHES:
        raise ValueError("checkpoint differs from the pinned Tiny-Jev-1.7B release")
    return observed


def save(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2),
        encoding="utf-8",
    )


class BlockProductImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "vllm_jev" or fullname.startswith("vllm_jev."):
            raise ImportError("product imports forbidden in author reference")
        return None


def render(value):
    return (
        value
        if isinstance(value, str)
        else json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
    )


def author_definition(question):
    """Explicit System One extensions, outside unchanged author inference.

    Custom Noul descriptions become instruction text. JSON descriptions become
    deterministic text; ordinary text definitions pass through unchanged.
    """
    kind = question["type"]
    result = {"kind": kind, "instructions": render(question["instructions"])}
    criteria = question.get("criteria")
    if kind == "noul":
        if criteria is not None:
            result["instructions"] += (
                "\nYes means: "
                + render(criteria["true"])
                + "\nNo means: "
                + render(criteria["false"])
            )
    elif kind == "choice":
        if any(value is None for value in criteria.values()):
            raise ValueError("typed parity fixture requires Choice descriptions")
        result["criteria"] = {key: render(value) for key, value in criteria.items()}
    elif kind == "score":
        result["criteria"] = [render(value) for value in criteria]
    else:
        raise ValueError("unsupported author fixture kind")
    return result


def original_module(source):
    file = source / "modeling_tiny_jev.py"
    if digest(file) != SOURCE_HASHES[file.name]:
        raise ValueError("refusing to execute unverified author code")
    spec = importlib.util.spec_from_file_location("tiny_jev_original_reference", file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def observe_forward(module, arguments, output):
    """Copy only small tensors, never retain model activations."""
    ids, attention, positions, mask = arguments
    if ids.shape[0] != 1 or not attention.bool().all() or not mask.all():
        raise RuntimeError("reference requires one unpadded question per forward")
    return {
        "input_ids": ids[0].tolist(),
        "positions": positions[0].tolist(),
        "calibrated_logits": output[0].detach().float().tolist(),
    }


def process_memory():
    """Windows lifetime peak counters include transient checkpoint-loading peaks."""
    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            *[
                (name, ctypes.c_size_t)
                for name in (
                    "PeakWorkingSetSize",
                    "WorkingSetSize",
                    "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage",
                    "QuotaPeakNonPagedPoolUsage",
                    "QuotaNonPagedPoolUsage",
                    "PagefileUsage",
                    "PeakPagefileUsage",
                    "PrivateUsage",
                )
            ],
        ]

    get_memory = ctypes.WinDLL("psapi", use_last_error=True).GetProcessMemoryInfo
    get_memory.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    get_memory.restype = wintypes.BOOL
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    if not get_memory(wintypes.HANDLE(-1), ctypes.byref(counters), counters.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return {
        "working_set_bytes": counters.WorkingSetSize,
        "peak_working_set_bytes": counters.PeakWorkingSetSize,
        "private_bytes": counters.PrivateUsage,
        "peak_commit_bytes": counters.PeakPagefileUsage,
    }


def run(args, report):
    source = args.source.resolve()
    report["source_hashes"] = verify(source)
    sys.dont_write_bytecode = True
    sys.meta_path.insert(0, BlockProductImports())
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    import torch
    from transformers import AutoTokenizer

    versions = {name: importlib.metadata.version(name) for name in VERSIONS}
    if versions != VERSIONS or torch.version.cuda is not None:
        raise RuntimeError("author reference requires the pinned CPU environment")
    report["versions"] = versions
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    original = original_module(source)
    config = original.TinyJevConfig.from_pretrained(source, local_files_only=True)
    # Runtime inference setting; the original files remain unchanged.
    config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    started = time.perf_counter()
    model, loading = original.TinyJevModel.from_pretrained(
        source,
        config=config,
        dtype=torch.float32,
        attn_implementation="sdpa",
        local_files_only=True,
        output_loading_info=True,
    )
    if any(
        loading.get(key)
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    ):
        raise RuntimeError("author checkpoint loading was incomplete")
    model.eval().requires_grad_(False)
    if any(
        p.is_meta or p.device.type != "cpu" or p.dtype != torch.float32
        for p in model.parameters()
    ):
        raise RuntimeError("author model was not entirely CPU FP32")
    report["load_seconds"] = time.perf_counter() - started
    report["memory_after_load"] = process_memory()
    report["runtime"] = {
        "device": "cpu",
        "dtype": "float32",
        "attention": "sdpa",
        "use_cache": False,
        "threads": args.threads,
        "row_batch_size": 1,
        "temperature": 1.15,
        "max_sequence_tokens": 4096,
    }
    captured = []
    hook = model.register_forward_hook(
        lambda module, arguments, output: captured.append(
            observe_forward(module, arguments, output)
        )
    )
    try:
        cases = json.loads(args.cases_json.read_text(encoding="utf-8"))
        for case in cases:
            row = {"id": case["id"], "rows": [], "answers": {}}
            started = time.perf_counter()
            for identifier, question in case["questions"].items():
                definition = author_definition(question)
                captured.clear()
                decision = model.decide(tokenizer, case["state"], [definition])[0]
                if len(captured) != 1:
                    raise RuntimeError("author decide must execute exactly one forward")
                observation = captured.pop()
                probabilities = torch.softmax(
                    torch.tensor(observation["calibrated_logits"], dtype=torch.float32),
                    dim=-1,
                ).tolist()
                row["rows"].append(
                    {
                        "id": identifier,
                        "definition": definition,
                        **observation,
                        "probabilities": probabilities,
                    }
                )
                row["answers"][identifier] = {"type": question["type"], **decision}
                if question["type"] == "score":
                    row["answers"][identifier]["legend"] = dict(
                        zip(definition["criteria"], question["criteria"])
                    )
            row["seconds"] = time.perf_counter() - started
            report["cases"].append(row)
            save(args.output, report)
            print(
                json.dumps(
                    {
                        "event": "author_case_completed",
                        "case": case["id"],
                        "seconds": row["seconds"],
                    }
                ),
                flush=True,
            )
    finally:
        hook.remove()
    report["memory_completed"] = process_memory()
    del model
    if any(name == "vllm_jev" or name.startswith("vllm_jev.") for name in sys.modules):
        raise RuntimeError("author reference imported product implementation")
    report["source_unchanged"] = verify(source) == report["source_hashes"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "cases-json", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if sys.platform != "win32" or not 1 <= args.threads <= 4:
        parser.error("native Windows and 1-4 CPU threads required")
    if args.output.resolve().is_relative_to(args.source.resolve()) or (
        args.output.resolve() == args.cases_json.resolve()
    ):
        parser.error("output must not overwrite source or fixtures")
    report = {
        "status": "running",
        "model": MODEL_ID,
        "revision": REVISION,
        "runner_sha256": digest(__file__),
        "cases_sha256": digest(args.cases_json),
        "cases": [],
        "scope": "Unmodified published TinyJevModel.decide and encoder; CPU FP32/SDPA, one question per forward, explicit use_cache=False. Synthetic parity, not quality or speed evaluation.",
    }
    try:
        run(args, report)
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save(args.output, report)


if __name__ == "__main__":
    main()
