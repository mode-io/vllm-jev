"""Kev prefix-cache benchmark: state length, long questions, and concurrency.

Runs against two already-running servers: a baseline that does not read the
prefix cache and a cache-enabled one. Each request carries one choice, one
noul, and one score question of roughly 100 to 200 tokens.

``length``       for each target state length: cold and warm latency, cached
                 tokens, and answer drift of cached against baseline.
``concurrency``  at one state length and several client counts: throughput,
                 p50/p99 latency, and drift against a sequential reference,
                 for distinct states (one per client) and one shared state.

Writes a JSON and a Markdown report. Never starts or stops a server.
"""

import argparse
import json
import secrets
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kev_cache_ab import (  # noqa: E402
    compare_answers,
    metrics,
    percentile,
    post_systemone,
    summarize,
)

CONTEXT = (
    "You are reviewing a long commercial supply contract between two parties. "
    "Read every clause carefully, including payment schedules, delivery "
    "obligations, inspection windows, penalties, and termination rights. The "
    "final clause overrides any earlier allocation of responsibilities between "
    "the parties, so weigh it above the boilerplate on invoices and payment "
    "terms unless that boilerplate changes who carries the obligation. "
)


def questions():
    """Three questions, each roughly 100 to 200 tokens with its options."""
    return {
        "party": {
            "type": "choice",
            "instructions": CONTEXT
            + "Considering the agreement as a whole and the governing final "
            "clause, which party must deliver the goods?",
            "criteria": {
                "supplier": "The supplier carries the delivery obligation and "
                "bears the cost and risk of transport to the agreed site.",
                "buyer": "The buyer must collect the goods itself and bears the "
                "cost and risk of transport from the supplier's premises.",
            },
        },
        "binding": {
            "type": "noul",
            "instructions": CONTEXT
            + "Is the delivery obligation mandatory, meaning that a missed "
            "delivery would be a breach of the agreement rather than a "
            "discretionary target that either party may waive without notice?",
        },
        "strictness": {
            "type": "score",
            "instructions": CONTEXT
            + "How strict are the payment deadlines across the schedules, taking "
            "into account the range of days allowed and whether any grace "
            "period or late-payment penalty is mentioned?",
            "criteria": [
                "loose: long or open-ended deadlines with no penalty",
                "normal: ordinary commercial deadlines of a few weeks",
                "strict: short deadlines enforced by explicit penalties",
            ],
        },
    }


def contract(clauses, tag, deliverer="supplier"):
    """A state of ``clauses`` invoice clauses; ``tag`` keeps states distinct."""
    body = " ".join(
        f"Clause {i}: invoices under schedule {i} are payable within "
        f"{i % 30 + 1} days of receipt."
        for i in range(clauses)
    )
    return (
        f"Supply agreement {tag}. {body} "
        f"Final clause: the {deliverer} must deliver the goods."
    )


TOKEN_PROBE = {"probe": {"type": "noul", "instructions": "?"}}


def calibrate(url, target, timeout, tag="calibration"):
    """Clauses whose state is about ``target`` tokens, by measuring the server."""
    clauses = max(1, target // 22)
    for _ in range(4):
        response = post_systemone(
            url, {"state": contract(clauses, tag), "questions": TOKEN_PROBE}, timeout
        )
        tokens = response["usage"]["input_tokens"]
        if abs(tokens - target) <= target * 0.02:
            break
        clauses = max(1, round(clauses * target / tokens))
    return clauses, tokens


def timed(url, body, timeout):
    started = time.perf_counter()
    response = post_systemone(url, body, timeout)
    return response, time.perf_counter() - started


def meta(response):
    m = response.get("metadata") or {}
    return m.get("cached_tokens"), m.get("uncached_reruns")


def latency_stats(values):
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "p50": percentile(values, 0.50),
        "p99": percentile(values, 0.99),
        "max": max(values),
    }


def run_length(args, run_id, target):
    clauses, state_tokens = calibrate(args.cached_url, target, args.timeout)
    q = questions()
    body = {"state": contract(clauses, f"{run_id}-len{target}"), "questions": q}
    salted = {**body, "cache_salt": f"kev-bench-{run_id}-len{target}"}
    before = metrics(args.cached_url)
    baseline, base_s = [], []
    for _ in range(args.repeats):
        response, seconds = timed(args.baseline_url, body, args.timeout)
        baseline.append(response)
        base_s.append(seconds)
    cold, cold_s = timed(args.cached_url, salted, args.timeout)
    warm, warm_s = [], []
    for _ in range(args.repeats):
        response, seconds = timed(args.cached_url, salted, args.timeout)
        warm.append(response)
        warm_s.append(seconds)
    after = metrics(args.cached_url)
    rows = []
    for b, w in zip(baseline, [cold, *warm]):
        rows += compare_answers(b["answers"], w["answers"], q)
    for w in warm:
        rows += compare_answers(baseline[0]["answers"], w["answers"], q)
    input_tokens = cold["usage"]["input_tokens"]
    cached = [meta(w)[0] or 0 for w in warm]
    return {
        "target_tokens": target,
        "state_tokens": state_tokens,
        "question_tokens_each": round((input_tokens - state_tokens) / len(q)),
        "input_tokens": input_tokens,
        "compute_tokens": cold["internal_usage"]["compute_tokens"],
        "baseline_s": latency_stats(base_s),
        "cold_s": cold_s,
        "warm_s": latency_stats(warm_s),
        "speedup": statistics.fmean(base_s) / statistics.fmean(warm_s),
        "cold_cached_tokens": meta(cold)[0],
        "warm_cached_tokens": cached,
        "reruns": [meta(w)[1] for w in warm],
        "counters": {
            k: (after.get(k) or 0) - (before.get(k) or 0) for k in ("queries", "hits")
        },
        "drift": summarize(rows),
    }


def parallel(url, bodies, clients, timeout):
    """Send ``bodies`` with ``clients`` workers; answers in order and timings."""
    started = time.perf_counter()
    with ThreadPoolExecutor(clients) as pool:
        results = list(pool.map(lambda b: timed(url, b, timeout), bodies))
    wall = time.perf_counter() - started
    return [r for r, _ in results], [s for _, s in results], wall


def run_concurrency(args, run_id, clauses, scenario, clients):
    q = questions()
    count = max(clients * args.rounds, clients)
    if scenario == "distinct":
        tags = [f"{run_id}-{scenario}-{clients}-{i % clients}" for i in range(count)]
    else:
        tags = [f"{run_id}-{scenario}-{clients}"] * count
    bodies = [{"state": contract(clauses, t), "questions": q} for t in tags]
    salted = [{**b, "cache_salt": f"kev-bench-{t}"} for b, t in zip(bodies, tags)]
    unique = list(dict.fromkeys(tags))
    # Sequential references: what each state answers alone, on each server.
    reference = {}
    for t in unique:
        body = bodies[tags.index(t)]
        reference[t] = post_systemone(args.baseline_url, body, args.timeout)
        # Warm the cached server too, so the concurrent phase measures hits.
        post_systemone(args.cached_url, salted[tags.index(t)], args.timeout)
    out = {"scenario": scenario, "clients": clients, "requests": count}
    for name, url, payloads in (
        ("baseline", args.baseline_url, bodies),
        ("cached", args.cached_url, salted),
    ):
        before = metrics(url)
        answers, seconds, wall = parallel(url, payloads, clients, args.timeout)
        after = metrics(url)
        rows = []
        for t, response in zip(tags, answers):
            rows += compare_answers(reference[t]["answers"], response["answers"], q)
        out[name] = {
            "wall_s": wall,
            "throughput_rps": count / wall,
            "latency_s": latency_stats(seconds),
            "cached_tokens_mean": statistics.fmean(meta(r)[0] or 0 for r in answers),
            "counters": {
                k: (after.get(k) or 0) - (before.get(k) or 0)
                for k in ("queries", "hits")
            },
            "drift_vs_sequential": summarize(rows),
        }
    out["speedup"] = out["cached"]["throughput_rps"] / out["baseline"]["throughput_rps"]
    return out


def fmt(x, digits=3):
    return f"{x:.{digits}f}"


def markdown(report):
    lines = [
        "# Kev prefix cache 基准：state 长度 × 长问题 × 并发",
        "",
        f"- 基线：`{report['baseline_url']}`",
        f"- 开缓存：`{report['cached_url']}`",
        f"- 时间：{report['started']}；每个请求 3 个问题（choice / noul / score）",
        "",
        "## 一、state 长度",
        "",
        "| 目标 | state token | 问题 token/个 | 基线均值 s | 冷 s | 命中均值 s | 加速 "
        "| 命中 token | 最大概率差 | 翻转 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in report["length"]:
        lines.append(
            f"| {r['target_tokens'] // 1024}K | {r['state_tokens']} | "
            f"{r['question_tokens_each']} | {fmt(r['baseline_s']['mean'])} | "
            f"{fmt(r['cold_s'])} | {fmt(r['warm_s']['mean'])} | "
            f"{fmt(r['speedup'], 2)}× | {r['warm_cached_tokens'][0]} | "
            f"{r['drift']['max_delta']:.2e} | {r['drift']['flips']}/"
            f"{r['drift']['questions']} |"
        )
    if report["concurrency"]:
        c0 = report["concurrency"][0]
        lines += [
            "",
            f"## 二、并发（state ≈ {report['concurrency_state_tokens']} token）",
            "",
            "| 场景 | 并发 | 请求 | 基线 req/s | 开缓存 req/s | 吞吐提升 "
            "| 基线 p50/p99 s | 开缓存 p50/p99 s | 命中 token 均值 "
            "| 漂移 基线/开缓存 | 翻转 基线/开缓存 |",
            "|---|---:|---:|---:|---:|---:|---|---|---:|---|---|",
        ]
        del c0
        for r in report["concurrency"]:
            b, c = r["baseline"], r["cached"]
            lines.append(
                f"| {r['scenario']} | {r['clients']} | {r['requests']} | "
                f"{fmt(b['throughput_rps'])} | {fmt(c['throughput_rps'])} | "
                f"{fmt(r['speedup'], 2)}× | {fmt(b['latency_s']['p50'], 2)}/"
                f"{fmt(b['latency_s']['p99'], 2)} | {fmt(c['latency_s']['p50'], 2)}/"
                f"{fmt(c['latency_s']['p99'], 2)} | {c['cached_tokens_mean']:.0f} | "
                f"{b['drift_vs_sequential']['max_delta']:.2e}/"
                f"{c['drift_vs_sequential']['max_delta']:.2e} | "
                f"{b['drift_vs_sequential']['flips']}/"
                f"{c['drift_vs_sequential']['flips']} |"
            )
        lines += [
            "",
            "漂移与翻转对照的是同一 state 单独顺序请求的答案（各自服务器上），"
            "反映 batch 组合带来的数值差异；基线一列是不读缓存时的噪声底线。",
        ]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-url", required=True)
    parser.add_argument("--cached-url", required=True)
    parser.add_argument(
        "--lengths", default="4096,8192,16384", help="target state tokens"
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--concurrency", default="1,2,4,8,16", help="client counts; empty to skip"
    )
    parser.add_argument(
        "--concurrency-tokens", type=int, default=8192, help="state length there"
    )
    parser.add_argument(
        "--scenarios", default="distinct,shared", help="distinct and/or shared"
    )
    parser.add_argument(
        "--rounds", type=int, default=2, help="requests per client per level"
    )
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--output", type=Path, default=Path("kev_cache_bench.json"))
    args = parser.parse_args(argv)

    run_id = secrets.token_hex(4)
    report = {
        "baseline_url": args.baseline_url,
        "cached_url": args.cached_url,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "length": [],
        "concurrency": [],
    }

    def save():
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        args.output.with_suffix(".md").write_text(markdown(report))

    for target in [int(x) for x in args.lengths.split(",") if x]:
        result = run_length(args, run_id, target)
        report["length"].append(result)
        save()
        print(
            f"length {target}: state={result['state_tokens']} "
            f"q={result['question_tokens_each']} base={result['baseline_s']['mean']:.2f}s "
            f"warm={result['warm_s']['mean']:.2f}s cached={result['warm_cached_tokens']} "
            f"drift={result['drift']['max_delta']:.2e} flips={result['drift']['flips']}",
            flush=True,
        )
    levels = [int(x) for x in args.concurrency.split(",") if x]
    if levels:
        clauses, tokens = calibrate(
            args.cached_url, args.concurrency_tokens, args.timeout
        )
        report["concurrency_state_tokens"] = tokens
        for scenario in [s for s in args.scenarios.split(",") if s]:
            for clients in levels:
                result = run_concurrency(args, run_id, clauses, scenario, clients)
                report["concurrency"].append(result)
                save()
                b, c = result["baseline"], result["cached"]
                print(
                    f"{scenario} c={clients}: base {b['throughput_rps']:.2f} rps "
                    f"cached {c['throughput_rps']:.2f} rps "
                    f"drift {b['drift_vs_sequential']['max_delta']:.2e}/"
                    f"{c['drift_vs_sequential']['max_delta']:.2e} "
                    f"flips {b['drift_vs_sequential']['flips']}/"
                    f"{c['drift_vs_sequential']['flips']}",
                    flush=True,
                )
    save()
    print(f"report: {args.output} / {args.output.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
