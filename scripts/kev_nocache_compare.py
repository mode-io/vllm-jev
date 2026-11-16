"""Uncached throughput: vllm-jev-cache against native-kev when no state repeats.

Every request carries a state no server has seen, so no prefix cache can help
and both backends compute every token. This measures the engines themselves:
prefill kernels and batching, without caching.

For each state length and client count, sends ``rounds * clients`` requests,
each with a fresh state, and reports throughput (requests and tokens per
second), latency, and evidence that nothing hit: ``metadata.cached_tokens`` on
vllm-jev-cache, ``prefix_cache.hits`` on native-kev. The three question rows
inside one vllm-jev request share a salt and may share blocks with each other;
that is the per-request behavior of the patched adapter and is reported, not
hidden.

Writes a JSON and a Markdown report. Never starts or stops a server.
"""

import argparse
import json
import secrets
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kev_backend_compare import (  # noqa: E402
    Client,
    latency_stats,
    native_cache,
    parallel,
)
from kev_cache_bench import contract, questions  # noqa: E402

PROBE = {"probe": {"type": "noul", "instructions": "?"}}


def calibrate(client, target, tag):
    """Clauses whose state is about ``target`` tokens on this server."""
    clauses = max(1, target // 22)
    tokens = 0
    for _ in range(4):
        tokens = client.post(
            "/v1/systemone", {"state": contract(clauses, tag), "questions": PROBE}
        )["usage"]["input_tokens"]
        if abs(tokens - target) <= target * 0.02:
            break
        clauses = max(1, round(clauses * target / tokens))
    return clauses, tokens


def run_level(args, name, url, clauses, clients, run_id, target):
    count = clients * args.rounds
    q = questions()
    tags = [f"{run_id}-{name}-{target}-{clients}-{i}" for i in range(count)]
    bodies = []
    for tag in tags:
        body = {"state": contract(clauses, tag), "questions": q}
        if name == "vllm":
            # A unique salt per request: no request can read another's blocks.
            body["cache_salt"] = f"kev-nocache-{tag}"
        bodies.append(body)
    probe = Client(url, args.timeout)
    try:
        before = native_cache(probe) if name == "native" else None
        answers, seconds, wall = parallel(url, args.timeout, bodies, clients)
        after = native_cache(probe) if name == "native" else None
    finally:
        probe.close()
    tokens = sum(r["usage"]["input_tokens"] for r in answers)
    out = {
        "requests": count,
        "wall_s": wall,
        "throughput_rps": count / wall,
        "input_tokens_per_s": tokens / wall,
        "latency_s": latency_stats(seconds),
    }
    if name == "vllm":
        cached = [(r.get("metadata") or {}).get("cached_tokens") or 0 for r in answers]
        out["cached_tokens_mean"] = statistics.fmean(cached)
        out["requests_with_cross_request_hit"] = None  # unique salts rule it out
    else:
        out["cache_hits_delta"] = after["hits"] - before["hits"]
        out["cache_misses_delta"] = after["misses"] - before["misses"]
        out["batch_count_delta"] = after["batch_count"] - before["batch_count"]
        out["server_s"] = latency_stats(
            [r["latency_ms"] / 1000.0 for r in answers if "latency_ms" in r]
        )
    return out


def markdown(report):
    lines = [
        "# 无缓存吞吐对比：vllm-jev-cache vs native-kev",
        "",
        f"- vllm-jev-cache：`{report['vllm_url']}`",
        f"- native-kev：`{report['native_url']}`",
        f"- 时间：{report['started']}；每请求 3 个问题，每个请求的 state 都是全新的",
        "- 每个请求唯一 state（vllm 侧另加唯一 salt），跨请求缓存无法命中",
        "",
    ]
    for target in report["lengths"]:
        rows = [r for r in report["levels"] if r["target_tokens"] == target]
        if not rows:
            continue
        lines += [
            f"## state ≈ {rows[0]['state_tokens']} token",
            "",
            "| 并发 | 请求 | vllm req/s | native req/s | vllm/native "
            "| vllm tok/s | native tok/s | vllm p50/p99 s | native p50/p99 s "
            "| native 命中 | native batch 数 |",
            "|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|",
        ]
        for r in rows:
            v, n = r["vllm"], r["native"]
            lines.append(
                f"| {r['clients']} | {v['requests']} | {v['throughput_rps']:.3f} | "
                f"{n['throughput_rps']:.3f} | {r['ratio']:.2f}× | "
                f"{v['input_tokens_per_s']:.0f} | {n['input_tokens_per_s']:.0f} | "
                f"{v['latency_s']['p50']:.2f}/{v['latency_s']['p99']:.2f} | "
                f"{n['latency_s']['p50']:.2f}/{n['latency_s']['p99']:.2f} | "
                f"{n['cache_hits_delta']}/{v['requests']} | {n['batch_count_delta']} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-url", required=True)
    parser.add_argument("--native-url", required=True)
    parser.add_argument("--lengths", default="4096,8192")
    parser.add_argument("--concurrency", default="1,2,4,8,16")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--output", type=Path, default=Path("kev_nocache.json"))
    args = parser.parse_args(argv)

    run_id = secrets.token_hex(4)
    lengths = [int(x) for x in args.lengths.split(",") if x]
    report = {
        "vllm_url": args.vllm_url,
        "native_url": args.native_url,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "lengths": lengths,
        "levels": [],
    }

    def save():
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        args.output.with_suffix(".md").write_text(markdown(report))

    for target in lengths:
        calib = Client(args.vllm_url, args.timeout)
        try:
            clauses, tokens = calibrate(calib, target, f"{run_id}-calib-{target}")
        finally:
            calib.close()
        for clients in [int(x) for x in args.concurrency.split(",") if x]:
            row = {"target_tokens": target, "state_tokens": tokens, "clients": clients}
            row["vllm"] = run_level(
                args, "vllm", args.vllm_url, clauses, clients, run_id, target
            )
            row["native"] = run_level(
                args, "native", args.native_url, clauses, clients, run_id, target
            )
            row["ratio"] = (
                row["vllm"]["throughput_rps"] / row["native"]["throughput_rps"]
            )
            report["levels"].append(row)
            save()
            v, n = row["vllm"], row["native"]
            print(
                f"state {tokens} c={clients}: vllm {v['throughput_rps']:.3f} rps "
                f"({v['input_tokens_per_s']:.0f} tok/s, cached {v['cached_tokens_mean']:.0f}) | "
                f"native {n['throughput_rps']:.3f} rps ({n['input_tokens_per_s']:.0f} tok/s, "
                f"hits {n['cache_hits_delta']}, batches {n['batch_count_delta']}) | "
                f"{row['ratio']:.2f}x",
                flush=True,
            )
    save()
    print(f"report: {args.output} / {args.output.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
