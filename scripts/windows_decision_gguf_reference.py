"""Independent Eval Engine model-card reference, not an original-author SDK.

Uses the public task specification, the GGUF's embedded Jinja chat template,
llama-cpp-python's high-level eval API, and all option logits. It never imports
product modules and never uses sampled tokens or truncated top-k probabilities.
"""

import argparse
import hashlib
import importlib.abc
import importlib.metadata
import json
import os
import re
import sys
import time
from pathlib import Path

from windows_decider_gguf_reference import verify_llama_wheel

WHEEL_SHA256 = "31590ea000d5aff6f05f1e428048e72318a83709288159a5bd4dabec530080bb"
CONTEXT_FIELDS = (
    "n_ctx",
    "n_batch",
    "n_ubatch",
    "n_seq_max",
    "kv_unified",
    "flash_attn_type",
    "rope_scaling_type",
    "rope_freq_base",
    "rope_freq_scale",
    "yarn_ext_factor",
    "yarn_attn_factor",
    "yarn_beta_fast",
    "yarn_beta_slow",
    "yarn_orig_ctx",
    "pooling_type",
    "attention_type",
    "type_k",
    "type_v",
    "n_threads",
    "n_threads_batch",
    "offload_kqv",
    "op_offload",
    "swa_full",
    "embeddings",
    "no_perf",
)
MODELS = {
    "evalengine/decision-0.8b-gguf": {
        "revision": "ce993792ad283053004bf0230b1709ff7351bb24",
        "filename": "decision-0.8b-Q8_0.gguf",
        "quantization": "Q8_0",
        "files": {
            "decision-0.8b-Q8_0.gguf": "efa1614fe5c7d84acb6b8acdf9fd0dda4a26acf110302732baa573d3332a4fef",
            "README.md": "311a735e75169fc33a43e9374ad66c47ee389cc596263b069fad219011880332",
            "LICENSE": "faed6815ea11f8b1dd2486b27a6712a01f22d1bf5b674f55a77782cd7febaf4c",
            "NOTICE": "bb45d37de4c72d27785c3a4dffce6019619fcfa0c00bfa035600fda69375f03f",
        },
    },
    "evalengine/decision-4b-gguf": {
        "revision": "76b147cb5da4585aed9ae026dda18a3c9d78305b",
        "filename": "decision-4b-Q4_K_M.gguf",
        "quantization": "Q4_K_M",
        "files": {
            "decision-4b-Q4_K_M.gguf": "d67a3c72a3665f8add74a2f86c900f62d3080d3bcb554e4e39df09c6f046d0c7",
            "README.md": "79d782a1e9d677af10e1c609c5aa1f5d4c359583e4a3ca05d0b07f27d8d6c2d7",
            "LICENSE": "faed6815ea11f8b1dd2486b27a6712a01f22d1bf5b674f55a77782cd7febaf4c",
            "NOTICE": "f41a59abf984b18a34f7237e4d4e75d661dfb3f64b620da53c0af27868f9ccd9",
        },
    },
}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_checkpoint(source, model):
    expected = MODELS[model]["files"]
    if source.is_symlink() or source.is_junction():
        raise ValueError("Reference source cannot be linked")
    for path in source.iterdir():
        if path.is_symlink() or path.is_junction():
            raise ValueError("Reference source contains links")
        if path.name == ".cache" and path.is_dir():
            continue
        if path.name not in expected or not path.is_file():
            raise ValueError("Reference source contains unexpected files")
    actual = {name: digest(source / name) for name in expected}
    if actual != expected:
        raise ValueError("Reference checkpoint identity differs")
    return actual


def task_rows(case):
    """Map the project's three public question types into model-card options."""
    rows = []
    for name, question in case["questions"].items():
        kind = question["type"]
        criteria = question.get("criteria")
        if kind == "choice":
            entries = (
                criteria.items()
                if isinstance(criteria, dict)
                else [(k, k) for k in criteria]
            )
            options = [(key, key if value is None else value) for key, value in entries]
        elif kind == "noul":
            options = [
                (key, key if (criteria or {}).get(key) is None else criteria[key])
                for key in ["false", "true"]
            ]
        elif kind == "score":
            if question.get("isolated", False) is not False:
                raise ValueError("Only listwise rubric options are supported")
            options = [(str(i), text) for i, text in enumerate(criteria)]
        else:
            raise ValueError("Unsupported reference question type")
        if not 2 <= len(options) <= (10 if kind == "score" else 24):
            raise ValueError("Invalid number of reference options")
        rows.append(
            (
                name,
                {
                    "state": case["state"],
                    "question": question["instructions"],
                    "options": [
                        {
                            "label": chr(ord("A") + index),
                            "key": key,
                            "description": text,
                        }
                        for index, (key, text) in enumerate(options)
                    ],
                },
            )
        )
    return rows


def render(template, system, task):
    from jinja2 import TemplateError
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    env = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"]
    )

    def fail(message):
        raise TemplateError(message)

    env.globals["raise_exception"] = fail
    env.filters["tojson"] = lambda value, **kw: json.dumps(
        value, ensure_ascii=False, **kw
    )
    return env.from_string(template).render(
        messages=[
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": json.dumps(task, ensure_ascii=False, allow_nan=False),
            },
        ],
        add_generation_prompt=True,
        enable_thinking=False,
        tools=None,
        bos_token="",
        eos_token="<|im_end|>",
    )


class NoProduct(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"vllm_jev", "torch", "transformers"}:
            raise ImportError("Independent Decision reference blocks " + fullname)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    for name in ["source", "llama-wheel", "cases-json", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.threads <= 4 or "LLAMA_CPP_LIB_PATH" in os.environ:
        raise ValueError("Invalid threads or native library override")
    sys.meta_path.insert(0, NoProduct())
    spec = MODELS[args.model]
    source_hashes = verify_checkpoint(args.source, args.model)
    runtime = verify_llama_wheel(args.llama_wheel)
    cases = json.loads(args.cases_json.read_text(encoding="utf-8-sig"))
    report = {
        "status": "running",
        "model": args.model,
        "revision": spec["revision"],
        "reference_kind": "independent_public_model_card",
        "original_author_sdk": False,
        "driver_sha256": digest(__file__),
        "input_sha256": digest(args.cases_json),
        "source_hashes": source_hashes,
        "llama_runtime": runtime,
        "environment": {
            "executable": str(Path(sys.executable).resolve()),
            "packages": {
                n: importlib.metadata.version(n)
                for n in ["numpy", "llama-cpp-python", "jinja2"]
            },
        },
        "runtime": {
            "device": "cpu",
            "threads": args.threads,
            "n_gpu_layers": 0,
            "n_ctx": 2048,
            "max_sequence_tokens": 1024,
            "temperature": 1.0,
            "memory_clear_before_decode": True,
            "readout": "all_option_logits",
            "softmax_dtype": "float32",
        },
        "cases": [],
    }
    import llama_cpp
    import numpy as np

    model = None
    try:
        card = (args.source / "README.md").read_text(encoding="utf-8")
        system = re.search(r'"role": "system", "content": "([^"]+)"', card).group(1)
        model = llama_cpp.Llama(
            model_path=str(args.source / spec["filename"]),
            n_gpu_layers=0,
            n_ctx=2048,
            n_batch=2048,
            n_ubatch=2048,
            n_threads=args.threads,
            n_threads_batch=args.threads,
            logits_all=False,
            verbose=False,
        )
        template = model.metadata["tokenizer.chat_template"]
        report["native_context"] = {
            name: getattr(model.context_params, name) for name in CONTEXT_FIELDS
        }
        report["chat_template_sha256"] = hashlib.sha256(template.encode()).hexdigest()
        report["system_from_pinned_model_card"] = system
        for case in cases:
            prepared = []
            for name, task in task_rows(case):
                prompt = render(template, system, task)
                ids = model.tokenize(prompt.encode(), add_bos=False, special=True)
                if not 1 <= len(ids) <= 1024:
                    raise ValueError("Reference complete prompt exceeds bound")
                labels = []
                for option in task["options"]:
                    extended = model.tokenize(
                        (prompt + option["label"]).encode(), add_bos=False, special=True
                    )
                    if extended[:-1] != ids or len(extended) != len(ids) + 1:
                        raise ValueError("Reference option token boundary differs")
                    labels.append(extended[-1])
                if len(set(labels)) != len(labels):
                    raise ValueError("Reference candidate token IDs are not unique")
                prepared.append((name, prompt, ids, labels))
            observed = {**case, "rows": [], "probabilities": {}}
            for name, prompt, ids, labels in prepared:
                model.reset()
                llama_cpp.llama_memory_clear(
                    llama_cpp.llama_get_memory(model._ctx.ctx), True
                )
                started = time.perf_counter()
                model.eval(ids)
                # llama-cpp-python 0.3.35 intentionally does not populate .scores
                # when logits_all=False. Read the actual final output buffer.
                pointer = model._ctx.get_logits()
                if not pointer:
                    raise ValueError("Reference native output logits are missing")
                logits = np.ctypeslib.as_array(pointer, shape=(model.n_vocab(),))[
                    labels
                ].copy()
                if not np.isfinite(logits).all():
                    raise ValueError("Reference returned nonfinite logits")
                exps = np.exp(logits - logits.max(), dtype=np.float32)
                probabilities = (exps / np.sum(exps, dtype=np.float32)).tolist()
                observed["probabilities"][name] = probabilities
                observed["rows"].append(
                    {
                        "id": name,
                        "prompt": prompt,
                        "prompt_tokens": ids,
                        "label_ids": labels,
                        "logits": logits.tolist(),
                        "probabilities": probabilities,
                        "seconds": time.perf_counter() - started,
                    }
                )
            report["cases"].append(observed)
            print(json.dumps({"event": "reference_case", "id": case["id"]}), flush=True)
        report["status"] = "passed"
    finally:
        if model is not None:
            model.close()
        report["source_unchanged"] = (
            verify_checkpoint(args.source, args.model) == source_hashes
        )
        report["llama_runtime_unchanged"] = (
            verify_llama_wheel(args.llama_wheel) == runtime
        )
        report["forbidden_modules_imported"] = any(
            n.split(".")[0] in {"vllm_jev", "torch", "transformers"}
            for n in sys.modules
        )
        report["measured"] = True
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
