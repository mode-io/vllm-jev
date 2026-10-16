"""Run unchanged pinned Bruv author encoder/collator/scorer in FP32 on CPU.

This is a minimal inference reference, not an installation of the author's CUDA
training project. Its evaluator defaults to BF16; our explicit FP32 profile is
compared only to the product FP32 profile, never to the published BF16 metrics.
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
from pathlib import Path

MODEL_ID = "bvolpato/bruv1-0.8b"
REVISION = "7d3660db482cae2afb1fe1ee091a659444791c13"
SDK_REVISION = "d10751e8fd3c2a917fea81fb350593eeed3c482d"
SOURCE_HASHES = {
    "config.json": "3f56b6210db3c52e0bafb50da005eb52b6d340ab4ad980091879b397743675fc",
    "generation_config.json": "62153eb6c69f2e1f426beaa8002b7186437e949c7588167085df14e10e9c0a73",
    "tokenizer.json": "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
    "tokenizer_config.json": "9cf04fffe3d8c3b85e439fb35c7acad0761ab51c422a8c4256d9f887c3a0be7d",
    "chat_template.jinja": "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80",
    "model.safetensors": "0da0a6084adc748d61952774f21ddc4e55cfa9f760d948588ecad6096f4b4589",
    "LICENSE": "bbedc3fda3305820b977265f01b8619d87570a6739de3a5582c3464840f1e57a",
    "training_run.json": "0b9dbb7e944ff0254a60f423c6590a3cf552504debfc73333ff1b812fbe33d57",
}
SDK_HASHES = {
    "src/bruv/__init__.py": "4876dda1a3d50c6dbe9f50dc162555e72d7f87df89ccb73abaf3e6255cdd430d",
    "src/bruv/data.py": "2585262f0c77974f0b2474dc444f7e92c393b77023d8608988ce9b81d331ec52",
    "src/bruv/model.py": "aa0038cf0cd46926387726ceb01750aee4dbf5e4ac968ad07987ba1fdcdc0d9e",
    "LICENSE": "6725369b5dd92753fe1a3b7e36a5d91b11eaad322f815a504f094c1b3f70ce81",
}
VERSIONS = {
    "torch": "2.9.1+cpu",
    "transformers": "5.17.0",
    "numpy": "2.5.4",
    "tokenizers": "0.23.3",
    "jinja2": "3.1.6",
}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(directory, pins):
    actual = {name: digest(Path(directory) / name) for name in pins}
    if actual != pins:
        raise ValueError("pinned Bruv source checksums differ")
    return actual


class BlockProductImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "vllm_jev" or fullname.startswith("vllm_jev."):
            raise ImportError("product imports forbidden in author reference")
        return None


def mapped_records(case):
    """Explicit project mapping; author API itself only accepts option records."""
    result = []
    for identifier, question in case["questions"].items():
        kind, criteria = question["type"], question.get("criteria")
        if kind == "choice":
            pairs = (
                list(criteria.items())
                if isinstance(criteria, dict)
                else [(key, None) for key in criteria]
            )
            descriptions = [
                key if not value else key + ": " + value for key, value in pairs
            ]
        elif kind == "noul":
            descriptions = []
            for key, name in (("false", "No"), ("true", "Yes")):
                description = (criteria or {}).get(key)
                descriptions.append(name + ": " + description if description else name)
        elif kind == "score":
            descriptions = list(criteria)
        else:
            raise ValueError("unsupported reference case")
        if not 2 <= len(descriptions) <= 16:
            raise ValueError("reference requires 2-16 options")
        result.append(
            {
                "id": identifier,
                "state": case["state"],
                "question": question["instructions"],
                "options": [
                    {"label": chr(65 + i), "description": text}
                    for i, text in enumerate(descriptions)
                ],
                "answer": "A",
            }
        )
    return result


def run(args, report):
    source = args.source.resolve()
    sdk = args.sdk_source.resolve()
    report["source_hashes"] = verify(source, SOURCE_HASHES)
    report["sdk_hashes"] = verify(sdk, SDK_HASHES)
    cases = json.loads(args.cases_json.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="bruv-author-", dir=Path.cwd()) as tmp:
        private = Path(tmp)
        report["private_workspace"] = str(private)
        for name in SDK_HASHES:
            destination = private / "sdk" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(sdk / name, destination)
        os.environ.update(
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            HF_HOME=str(private / "hf"),
            TOKENIZERS_PARALLELISM="false",
        )
        sys.dont_write_bytecode = True
        sys.meta_path.insert(0, BlockProductImports())
        sys.path.insert(0, str(private / "sdk/src"))
        import torch
        from bruv.data import encode_record, label_token_ids
        from bruv.model import collate, score_batch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        versions = {name: importlib.metadata.version(name) for name in VERSIONS}
        if versions != VERSIONS or torch.version.cuda is not None:
            raise RuntimeError("author reference requires fixed Windows CPU versions")
        report["versions"] = versions
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        tokenizer = AutoTokenizer.from_pretrained(
            source, local_files_only=True, trust_remote_code=False
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        labels = torch.tensor(label_token_ids(tokenizer), device="cpu")
        # Record exact author encodings before inference, never product encodings.
        prepared = [(case, mapped_records(case)) for case in cases]
        started = time.perf_counter()
        model, loading = AutoModelForCausalLM.from_pretrained(
            source,
            dtype=torch.float32,
            attn_implementation="eager",
            local_files_only=True,
            trust_remote_code=False,
            output_loading_info=True,
        )
        if any(
            loading.get(k)
            for k in (
                "missing_keys",
                "unexpected_keys",
                "mismatched_keys",
                "error_msgs",
            )
        ):
            raise RuntimeError("author reference model loading is incomplete")
        model.eval().requires_grad_(False)
        if any(
            p.device.type != "cpu" or p.dtype != torch.float32 or p.is_meta
            for p in model.parameters()
        ):
            raise RuntimeError("author model is not CPU FP32")
        report["load_seconds"] = time.perf_counter() - started
        report["runtime"] = {
            "device": "cpu",
            "dtype": "float32",
            "threads": args.threads,
            "attention_implementation": "eager",
            "use_cache": False,
            "row_batch_size": 1,
            "max_sequence_tokens": 1024,
            "prompt_format": "kevala-direct-options-v1",
            "temperature": 1.0,
            "published_evaluator_default_dtype": "bfloat16",
            "training_dependencies_installed": False,
        }
        with torch.inference_mode():
            for case, records in prepared:
                started = time.perf_counter()
                observed = {"id": case["id"], "rows": [], "probabilities": {}}
                for record in records:
                    row = encode_record(record, tokenizer, 1024)
                    batch = collate([row], tokenizer.pad_token_id, torch.device("cpu"))
                    scores = score_batch(model, batch, labels)[0]
                    count = len(record["options"])
                    if (
                        scores.dtype != torch.float32
                        or not torch.isfinite(scores[:count]).all()
                        or not torch.isneginf(scores[count:]).all()
                    ):
                        raise RuntimeError("invalid author candidate logits")
                    probabilities = torch.softmax(scores, dim=-1)[:count].tolist()
                    observed["rows"].append(
                        {
                            "id": record["id"],
                            "record": record,
                            "input_ids": row["input_ids"],
                            "logits": scores[:count].tolist(),
                            "probabilities": probabilities,
                        }
                    )
                    observed["probabilities"][record["id"]] = probabilities
                observed["seconds"] = time.perf_counter() - started
                report["cases"].append(observed)
                args.output.write_text(
                    json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2),
                    encoding="utf-8",
                )
                print(
                    json.dumps(
                        {
                            "event": "author_case_completed",
                            "case": case["id"],
                            "seconds": observed["seconds"],
                        }
                    ),
                    flush=True,
                )
        del model
        if any(
            name == "vllm_jev" or name.startswith("vllm_jev.") for name in sys.modules
        ):
            raise RuntimeError("product was imported by author reference")
        verify(private / "sdk", SDK_HASHES)
    report["private_workspace_removed"] = not private.exists()
    report["source_unchanged"] = (
        verify(source, SOURCE_HASHES) == report["source_hashes"]
    )
    report["sdk_unchanged"] = verify(sdk, SDK_HASHES) == report["sdk_hashes"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("source", "sdk-source", "cases-json", "output"):
        parser.add_argument("--" + option, required=True, type=Path)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if sys.platform != "win32" or not 1 <= args.threads <= 4:
        parser.error("native Windows CPU and 1-4 threads required")
    for root in (args.source, args.sdk_source):
        if args.output.resolve().is_relative_to(root.resolve()):
            parser.error("output cannot overwrite source or SDK")
    report = {
        "status": "running",
        "model": MODEL_ID,
        "revision": REVISION,
        "sdk_revision": SDK_REVISION,
        "runner_sha256": digest(__file__),
        "cases_sha256": digest(args.cases_json),
        "cases": [],
        "reference_scope": "Unchanged author data.encode_record, model.collate, model.score_batch; explicit same-precision FP32 reference; independent System One mapping, not an author System One API.",
    }
    try:
        run(args, report)
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
