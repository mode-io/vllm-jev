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
        assert set(result["answers"]) == set(body["questions"])
        for answer in result["answers"].values():
            if "probabilities" in answer:
                values = list(answer["probabilities"].values())
                assert all(math.isfinite(v) and 0 <= v <= 1 for v in values)
                assert abs(sum(values) - 1) < 0.02
            if "noul" in answer:
                assert math.isfinite(answer["noul"]) and 0 <= answer["noul"] <= 1
        return time.perf_counter() - started

    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        latencies = sorted(pool.map(one, range(args.requests)))
    result = {
        "scope": "functional API pressure only; hardware/backend must be verified separately",
        "successful": len(latencies),
        "concurrency": args.concurrency,
        "wall_seconds": time.perf_counter() - started,
        "client_p50_seconds": latencies[len(latencies) // 2],
        "client_p95_seconds": latencies[int((len(latencies) - 1) * 0.95)],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
