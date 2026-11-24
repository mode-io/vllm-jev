"""Kev prefix-cache A/B: does reading the cache change any answer?

Runs the same requests against two already-running servers, one with
``VLLM_JEV_KEV_PREFIX_CACHE=1`` and one with ``0``. Reports probability drift
and argmax flips per question, so a drift that never changes the chosen option
is not confused with one that does. Never starts or stops a server.
"""

import argparse
import json
import math
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path

# Prometheus counters vLLM exposes; the A/B is meaningless if the cached server
# never recorded a hit.
QUERY_METRIC = "vllm:prefix_cache_queries_total"
HIT_METRIC = "vllm:prefix_cache_hits_total"


def probabilities_of(kind, answer):
    """The answer's option probabilities, keyed as the request named them."""
    if kind == "noul":
        values = {"true": answer["noul"]}
    else:
        values = answer.get("probabilities")
    if not isinstance(values, dict) or not values:
        raise ValueError(f"{kind} answer has no probability vector")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value <= 1
        for value in values.values()
    ):
        raise ValueError("answer probabilities must be finite values in [0, 1]")
    if kind != "noul" and not math.isclose(sum(values.values()), 1.0, abs_tol=1e-5):
        raise ValueError("answer probabilities must sum to one")
    return {key: float(value) for key, value in values.items()}


def argmax_of(kind, answer):
    """The option the answer picks: ``noul`` picks true above one half."""
    values = probabilities_of(kind, answer)
    if kind == "noul":
        return "true" if values["true"] > 0.5 else "false"
    return max(values, key=values.__getitem__)


def percentile(values, fraction):
    """Nearest-rank percentile; ``fraction`` is 0 to 1."""
    if not values:
        raise ValueError("percentile of an empty sample")
    ordered = sorted(values)
    rank = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[rank]


def compare_answers(baseline, cached, questions):
    """Per-question drift and flips between two ``answers`` maps.

    ``questions`` is the request's question map, so a kind or option set that
    differs between the two runs is an error rather than a silent skip.
    """
    if set(baseline) != set(questions) or set(cached) != set(questions):
        raise ValueError("both runs must answer exactly the requested questions")
    rows = []
    for key, spec in questions.items():
        kind = spec["type"]
        left, right = baseline[key], cached[key]
        if left.get("type") != kind or right.get("type") != kind:
            raise ValueError(f"{key}: answer type does not match the question")
        left_p, right_p = probabilities_of(kind, left), probabilities_of(kind, right)
        if set(left_p) != set(right_p):
            raise ValueError(f"{key}: the two runs scored different options")
        deltas = {name: abs(right_p[name] - left_p[name]) for name in left_p}
        left_pick, right_pick = argmax_of(kind, left), argmax_of(kind, right)
        rows.append(
            {
                "question": key,
                "type": kind,
                "max_delta": max(deltas.values()),
                "deltas": deltas,
                "baseline_pick": left_pick,
                "cached_pick": right_pick,
                "flipped": left_pick != right_pick,
            }
        )
    return rows


def summarize(rows):
    """Aggregate the per-question rows into the numbers a decision needs."""
    deltas = [row["max_delta"] for row in rows]
    return {
        "questions": len(rows),
        "flips": sum(row["flipped"] for row in rows),
        "flipped": [row["question"] for row in rows if row["flipped"]],
        "max_delta": max(deltas) if deltas else 0.0,
        "mean_delta": statistics.fmean(deltas) if deltas else 0.0,
        "p50_delta": percentile(deltas, 0.50) if deltas else 0.0,
        "p99_delta": percentile(deltas, 0.99) if deltas else 0.0,
    }


def verdict(stats, max_delta, max_flips):
    """Whether this run is acceptable, and the reasons it is not."""
    reasons = []
    if not math.isfinite(max_delta) or max_delta < 0 or max_flips < 0:
        raise ValueError("comparison thresholds must be finite and nonnegative")
    if not math.isfinite(stats["max_delta"]):
        reasons.append("nonfinite probability drift")
    if stats.get("questions", 1) <= 0:
        reasons.append("no questions were compared")
    if stats["flips"] > max_flips:
        reasons.append(
            f"{stats['flips']} argmax flips exceed the {max_flips} allowed: "
            + ", ".join(stats["flipped"])
        )
    if stats["max_delta"] > max_delta:
        reasons.append(
            f"max probability delta {stats['max_delta']:.6f} exceeds {max_delta}"
        )
    return {"passed": not reasons, "reasons": reasons}


def parse_metric(text, name):
    """The summed value of one Prometheus counter across every label set."""
    total, seen = 0.0, False
    for line in text.splitlines():
        if not line.startswith(name):
            continue
        rest = line[len(name) :]
        if rest and not rest.startswith(("{", " ")):
            continue
        try:
            total += float(rest.rsplit(None, 1)[-1])
        except ValueError:
            raise ValueError(f"unreadable {name} sample: {line!r}") from None
        seen = True
    return total if seen else None


def request_json(url, body=None, timeout=120.0):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode()


def post_systemone(url, body, timeout=120.0):
    return json.loads(request_json(url.rstrip("/") + "/v1/systemone", body, timeout))


def metrics(url, timeout=30.0):
    text = request_json(url.rstrip("/") + "/metrics", None, timeout)
    return {
        "queries": parse_metric(text, QUERY_METRIC),
        "hits": parse_metric(text, HIT_METRIC),
    }


def state_text(index, clauses):
    return " ".join(
        f"Clause {i}: party {index} shall deliver item {i} within {i % 30 + 1} days."
        for i in range(clauses)
    )


def question_set():
    return {
        "party": {
            "type": "choice",
            "instructions": "Which party must deliver?",
            "criteria": {"supplier": None, "buyer": None},
        },
        "binding": {"type": "noul", "instructions": "Is delivery mandatory?"},
        "strictness": {
            "type": "score",
            "instructions": "How strict is the deadline?",
            "criteria": ["loose", "normal", "strict"],
        },
    }


def run_once(base_url, body, timeout):
    started = time.perf_counter()
    response = post_systemone(base_url, body, timeout)
    return response, time.perf_counter() - started


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-url",
        required=True,
        help="server started with VLLM_JEV_KEV_PREFIX_CACHE=0",
    )
    parser.add_argument(
        "--cached-url",
        required=True,
        help="server started with VLLM_JEV_KEV_PREFIX_CACHE=1",
    )
    parser.add_argument(
        "--states",
        type=int,
        default=4,
        help="distinct states; each is sent once to warm, then measured",
    )
    parser.add_argument(
        "--clauses",
        type=int,
        default=300,
        help="clauses per state; about 17 tokens each, so 300 is roughly 5,000",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="measured request rounds after the warm-up round",
    )
    parser.add_argument(
        "--max-delta",
        type=float,
        default=0.05,
        help="fail when any single probability moves further than this",
    )
    parser.add_argument(
        "--max-flips",
        type=int,
        default=0,
        help="fail when the chosen option changes more often than this",
    )
    parser.add_argument("--output", type=Path, help="write the full JSON report")
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args(argv)

    before = {
        name: metrics(url)
        for name, url in (
            ("baseline", args.baseline_url),
            ("cached", args.cached_url),
        )
    }
    rows, noise, timings = [], [], {"baseline": [], "cached": []}
    for index in range(args.states):
        # One salt per state lets the cached server reuse that state's blocks.
        # A server started with VLLM_JEV_KEV_PREFIX_CACHE=0 rejects cache_salt
        # with 422, so the baseline gets the same rows without it.
        body = {
            "state": state_text(index, args.clauses),
            "questions": question_set(),
        }
        salted = {**body, "cache_salt": f"kev-ab-state-{index}"}
        # Warm-up: the cached server's first request for a state is always a
        # miss, so it is neither timed nor compared.
        warm_baseline, _ = run_once(args.baseline_url, body, args.timeout)
        run_once(args.cached_url, salted, args.timeout)
        for _ in range(args.repeats):
            baseline, left_s = run_once(args.baseline_url, body, args.timeout)
            cached, right_s = run_once(args.cached_url, salted, args.timeout)
            timings["baseline"].append(left_s)
            timings["cached"].append(right_s)
            # Every measured round counts, not only the last one.
            rows += compare_answers(
                baseline["answers"], cached["answers"], body["questions"]
            )
            # The noise floor: the uncached server against itself. Drift the
            # cache adds is only what exceeds this.
            noise += compare_answers(
                warm_baseline["answers"], baseline["answers"], body["questions"]
            )

    stats = summarize(rows)
    noise_stats = summarize(noise)
    after = {
        name: metrics(url)
        for name, url in (
            ("baseline", args.baseline_url),
            ("cached", args.cached_url),
        )
    }
    hits = {}
    for name in before:
        old = before[name].get("hits") or 0.0
        new = after[name].get("hits") or 0.0
        queries_old = before[name].get("queries") or 0.0
        queries_new = after[name].get("queries") or 0.0
        hits[name] = {"queries": queries_new - queries_old, "hits": new - old}
    result = verdict(stats, args.max_delta, args.max_flips)
    evidence = {
        "questions": len(rows),
        **stats,
        "stats": stats,
        "noise_floor": noise_stats,
        "timing_s": {
            name: {
                "mean": statistics.fmean(values) if values else 0.0,
                "min": min(values) if values else 0.0,
            }
            for name, values in timings.items()
        },
        "cache_counters": hits,
        "limits": {"max_delta": args.max_delta, "max_flips": args.max_flips},
        **result,
    }
    if not math.isfinite(hits["cached"]["hits"]) or hits["cached"]["hits"] <= 0:
        evidence["passed"] = False
        evidence["reasons"].append(
            f"the cached server recorded no {HIT_METRIC} increase; check "
            "--enable-prefix-caching and that cache_salt repeats"
        )

    print("| Metric | Baseline | Cached |\n| --- | ---: | ---: |")
    print(
        f"| Mean latency (s) | {evidence['timing_s']['baseline']['mean']:.3f} "
        f"| {evidence['timing_s']['cached']['mean']:.3f} |"
    )
    print(
        f"| Cache queries | {hits['baseline']['queries']:.0f} "
        f"| {hits['cached']['queries']:.0f} |"
    )
    print(
        f"| Cache hits | {hits['baseline']['hits']:.0f} "
        f"| {hits['cached']['hits']:.0f} |"
    )
    print(f"| Max probability delta | | {stats['max_delta']:.6f} |")
    print(f"| p99 probability delta | | {stats['p99_delta']:.6f} |")
    print(f"| Argmax flips | | {stats['flips']} / {stats['questions']} |")
    print(
        f"| Noise floor: baseline vs itself | {noise_stats['max_delta']:.6f} "
        f"max, {noise_stats['flips']} flips | |"
    )
    print(f"| Verdict | | {'pass' if evidence['passed'] else 'FAIL'} |")
    for reason in evidence["reasons"]:
        print(f"  - {reason}")
    if args.output:
        args.output.write_text(json.dumps({"rows": rows, **evidence}, indent=2) + "\n")
    return 0 if evidence["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
