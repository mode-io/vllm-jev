"""Short-lived product-tokenizer probe; does not load any model weights."""

import argparse
import hashlib
import json
import sys
from pathlib import Path


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--cases-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(
        args.source.resolve()
    ) or args.output.resolve() in (
        args.cases_json.resolve(),
        Path(sys.executable).resolve(),
    ):
        parser.error("Probe output must not overwrite source or inputs")
    from transformers import AutoTokenizer

    from vllm_jev import windows_rsi as product

    tokenizer = AutoTokenizer.from_pretrained(
        args.source / "base", local_files_only=True, trust_remote_code=False
    )
    cases = json.loads(args.cases_json.read_text(encoding="utf-8"))
    records = []
    for case in cases:
        rows = []
        for key, kind, instructions, options, descriptions in product.parse_questions(
            case["questions"]
        ):
            ids, spans, index = product.encode_question(
                tokenizer,
                product._text(case["state"]),
                instructions,
                options,
                descriptions,
            )
            rows.append(
                {
                    "id": key,
                    "input_ids": ids,
                    "option_span": spans,
                    "decision_index": index,
                    "options": options,
                    "mode": kind,
                }
            )
        records.append({"id": case["id"], "encoded": rows})
    report = {
        "status": "passed",
        "model": product.MODEL_ID,
        "revision": product.REVISION,
        "python": sys.executable,
        "torch_imported": "torch" in sys.modules,
        "product_module": str(Path(product.__file__).resolve()),
        "product_sha256": digest(product.__file__),
        "probe_sha256": digest(__file__),
        "cases_sha256": digest(args.cases_json),
        "tokenizer_sha256": digest(args.source / "base/tokenizer.json"),
        "cases": records,
    }
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
