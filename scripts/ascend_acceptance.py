"""Exercise an already running System One server; never download/start a model.

Record one output per served model/family and preserve the exact launch command,
runtime versions, asset hashes and device ownership alongside it. Passing this
small functional probe does not establish reference quality or speedup.
"""

import argparse
import concurrent.futures
import json
import math
import time
import urllib.error
import urllib.request
from pathlib import Path


def validate_response(result, body):
    answers = result.get("answers")
    if not isinstance(answers, dict) or set(answers) != set(body["questions"]):
        raise ValueError("Missing or unexpected answer IDs")
    for key, spec in body["questions"].items():
        answer = answers[key]
        kind = spec["type"]
        if not isinstance(answer, dict) or answer.get("type") != kind:
            raise ValueError(f"Invalid answer type for {key}")
        if kind == "noul":
            v = answer.get("noul")
            if type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1:
                raise ValueError("Invalid Noul probability")
            continue
        probabilities = answer.get("probabilities")
        if not isinstance(probabilities, dict) or not probabilities:
            raise ValueError("Missing probability vector")
        values = list(probabilities.values())
        if any(
            type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1
            for v in values
        ):
            raise ValueError("Invalid probabilities")
        if abs(sum(values) - 1) > 0.02:
            raise ValueError("Probabilities are not normalized")
        if len(values) != len(spec["criteria"]):
            raise ValueError("Wrong probability count")
        if kind == "choice":
            if (
                set(probabilities) != set(spec["criteria"])
                or answer.get("choice") not in spec["criteria"]
            ):
                raise ValueError("Invalid selected option")
        elif kind == "score":
            v = answer.get("score")
            if (
                type(v) not in (int, float)
                or not math.isfinite(v)
                or not 0 <= v <= len(values) - 1
            ):
                raise ValueError("Invalid score")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8795")
    parser.add_argument("--requests", type=int, default=128)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.concurrency <= 128 or not 1 <= args.requests <= 10000:
        parser.error("concurrency must be 1..128 and requests 1..10000")
    body = {
        "state": "A customer asks for a refund after a duplicate payment.",
        "questions": {
            "route": {
                "type": "choice",
                "instructions": "Select a team.",
                "criteria": {"billing": "Payments", "technical": "Software"},
            },
            "refund": {"type": "noul", "instructions": "Is a refund requested?"},
            "urgency": {
                "type": "score",
                "instructions": "Rate the urgency.",
                "criteria": ["low", "medium", "high"],
            },
        },
    }

    def one(_):
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        request = urllib.request.Request(
            args.url.rstrip("/") + "/v1/systemone",
            json.dumps(body).encode(),
            {"Content-Type": "application/json"},
        )
        started = time.perf_counter()
        with opener.open(request, timeout=180) as response:
            result = json.load(response)
        validate_response(result, body)
        return time.perf_counter() - started

    started = time.perf_counter()
    latencies, failures = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(one, i) for i in range(args.requests)]
        for future in concurrent.futures.as_completed(futures):
            try:
                latencies.append(future.result())
            except Exception as error:
                failures.append(
                    {"type": type(error).__name__, "message": str(error)[:500]}
                )
    latencies.sort()
    result = {
        "scope": "functional API pressure only; hardware/backend must be verified separately",
        "successful": len(latencies),
        "failed": len(failures),
        "failures": failures[:20],
        "status": "passed" if not failures else "failed",
        "concurrency": args.concurrency,
        "wall_seconds": time.perf_counter() - started,
        "client_p50_seconds": latencies[len(latencies) // 2] if latencies else None,
        "client_p95_seconds": latencies[int((len(latencies) - 1) * 0.95)]
        if latencies
        else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
