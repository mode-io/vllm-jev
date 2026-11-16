"""Do the block-granular and whole-state Kev backends match at block-aligned states?

vllm-jev caches in blocks of ``block_size`` and only a prefix that lands on a
block boundary survives a hit; the rest of the state, and every question row, is
recomputed. native-kev keys its cache on the exact state prefix, so any repeat is
a full hit. This asks whether a state ending exactly on a block boundary makes
the block-granular backend cache the whole state and match the whole-state one.

Each length is built from words, calibrated against the server's own reported
count, sent cold and then warm to both servers. Cases marked aligned are exact
multiples of the block size; the others are controls: the same multiples minus a
tail, and the requested 736 multiples, which do not align to a 784 block.

Writes a JSON and a Markdown report. Never starts or stops a server.
"""

import argparse
import json
import secrets
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kev_backend_compare import Client, latency_stats, native_cache, timed  # noqa: E402
from kev_cache_ab import compare_answers, summarize  # noqa: E402
from kev_cache_bench import questions  # noqa: E402

PROBE = {"probe": {"type": "noul", "instructions": "?"}}


def build(words, tag):
    """A state of ``words`` one-token words, tagged so states stay distinct."""
    return f"agreement {tag} " + " ".join(["clause"] * words)


def input_tokens(client, state, questions_map=PROBE):
    response = client.post(
        "/v1/systemone", {"state": state, "questions": questions_map}
    )
    return response["usage"]["input_tokens"]


def overhead(client, tag="overhead"):
    """Tokens a probe question adds after the state, from two state lengths."""
    short = input_tokens(client, build(100, tag))
    long = input_tokens(client, build(300, tag))
    if long - short != 200:
        raise RuntimeError(
            f"a word is not one token here ({short} -> {long} for +200 words)"
        )
    return short - 101


def calibrate(client, target, tag, margin):
    """A state whose reported token count is ``target``."""
    words = max(1, target - margin)
    state, tokens = build(words, tag), None
    for _ in range(4):
        tokens = input_tokens(client, state) - overhead(client, tag)
        if tokens == target:
            break
        words += target - tokens
        state = build(max(1, words), tag)
    return state, tokens


def cached(response):
    return (response.get("metadata") or {}).get("cached_tokens") or 0


def per_row_cached(response, rows):
    """Cached tokens per question row; each row repeats the shared state."""
    return cached(response) // rows


def run_case(args, vllm, native, target, kind):
    q = questions()
    tag = f"{args.run_id}-{kind}-{target}"
    v_state, v_tokens = calibrate(vllm, target, tag, args.margin)
    # Calibrating against native-kev would prime its cache with this very state,
    # so its "cold" number would already be a hit. Measure its cold run on a
    # sibling state of the same length instead, then prime the real one.
    cold_tag = f"{tag}-cold"
    n_cold_state, n_cold_tokens = calibrate(native, target, cold_tag, args.margin)
    n_state = v_state
    n_tokens = n_cold_tokens
    salt = f"kev-align-{tag}"
    v_body = {"state": v_state, "questions": q, "cache_salt": salt}
    n_body = {"state": n_state, "questions": q}
    out = {
        "kind": kind,
        "target_tokens": target,
        "block": args.block,
        "remainder": target % args.block,
        "aligned": target % args.block == 0,
        "vllm_state_tokens": v_tokens,
        "native_state_tokens": n_tokens,
        # the timed requests carry one text on both servers
        "identical_state_text": n_state == v_state,
        "rows": len(q),
    }
    n_cold_body = {"state": n_cold_state, "questions": q}
    v_cold, v_cold_s = timed(vllm, v_body)
    n_before = native_cache(native)
    n_cold, n_cold_s = timed(native, n_cold_body)
    # First sight of the real state: a miss, discarded so every timed repeat hits.
    n_prime, _ = timed(native, n_body)
    n_primed = native_cache(native)
    v_warm, n_warm, v_s, n_s = [], [], [], []
    for _ in range(args.repeats):
        rv, sv = timed(vllm, v_body)
        rn, sn = timed(native, n_body)
        v_warm.append(rv)
        n_warm.append(rn)
        v_s.append(sv)
        n_s.append(sn)
    n_after = native_cache(native)
    rows = len(q)
    per_row = [per_row_cached(r, rows) for r in v_warm]
    out["vllm"] = {
        "cold_s": v_cold_s,
        "warm_s": latency_stats(v_s),
        "cold_cached_per_row": per_row_cached(v_cold, rows),
        "warm_cached_per_row": per_row,
        "warm_reruns": [
            (r.get("metadata") or {}).get("uncached_reruns") for r in v_warm
        ],
        "state_fully_cached": all(p >= v_tokens for p in per_row),
        "recomputed_state_tokens": [v_tokens - p for p in per_row],
    }
    hits = n_after["hits"] - n_primed["hits"]
    misses = n_after["misses"] - n_primed["misses"]
    del n_cold_s, n_before  # calibration already sent n_cold_state, so not cold
    out["native"] = {
        # No native cold time is reported: native-kev keys its cache on the
        # state alone, and calibration has already sent every state it sees.
        "calibration_state_tokens": n_cold_tokens,
        "warm_s": latency_stats(n_s),
        "warm_server_s": latency_stats([r["latency_ms"] / 1000.0 for r in n_warm]),
        # counted after the discarded prime, so these are the timed repeats only
        "cache_hits_delta": hits,
        "cache_misses_delta": misses,
        "warm_all_hit": hits >= args.repeats,
    }
    out["warm_ratio"] = out["vllm"]["warm_s"]["mean"] / out["native"]["warm_s"]["mean"]
    # Only answers to the same text are compared. n_cold ran a sibling state
    # (to time a genuine miss), so it is excluded here.
    # Across engines: native-kev's first sight of the state (n_prime) and its
    # hits, against vllm-jev-cache's cold run and hits.
    rows_out = compare_answers(n_prime["answers"], v_cold["answers"], q)
    for n, v in zip(n_warm, v_warm):
        rows_out += compare_answers(n["answers"], v["answers"], q)
    out["agreement"] = summarize(rows_out)
    # Within each engine: does a hit answer like that engine's own miss?
    within_v, within_n = [], []
    for v in v_warm:
        within_v += compare_answers(v_cold["answers"], v["answers"], q)
    for n in n_warm:
        within_n += compare_answers(n_prime["answers"], n["answers"], q)
    out["hit_vs_miss"] = {
        "vllm": summarize(within_v),
        "native": summarize(within_n),
    }
    # A degenerate state can leave a question's top two options within a few
    # 1e-3, so a small numeric difference between engines flips its argmax.
    # Keep the per-question detail to tell that apart from a real disagreement.
    out["rows"] = rows_out
    out["probabilities"] = {
        "native_cold": n_prime["answers"],
        "vllm_cold": v_cold["answers"],
        "native_warm": n_warm[0]["answers"],
        "vllm_warm": v_warm[0]["answers"],
    }
    return out


def table(cases, block):
    lines = [
        "| state token | block 余数 | vllm 每行命中 | 重算 | vllm 命中 s | native 命中 s "
        "| 命中耗时比 | native 命中 | 概率差 | 翻转 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in cases:
        v, n = r["vllm"], r["native"]
        lines.append(
            f"| {r['vllm_state_tokens']} | {r['remainder']} | "
            f"{v['warm_cached_per_row'][0]} | {v['recomputed_state_tokens'][0]} | "
            f"{v['warm_s']['mean']:.2f} | {n['warm_s']['mean']:.2f} | "
            f"{r['warm_ratio']:.2f}× | {n['cache_hits_delta']}/{n['cache_misses_delta']} | "
            f"{r['agreement']['max_delta']:.2e} | "
            f"{r['agreement']['flips']}/{r['agreement']['questions']} |"
        )
    del block
    return lines


def markdown(report):
    groups = {
        "aligned": "一、block 784 的整数倍（对齐）",
        "tail": "二、差一点不到边界（对照）",
        "736": "三、736 的整数倍（不对齐 784）",
    }
    lines = [
        "# Kev 块对齐对比：vllm-jev-cache vs native-kev",
        "",
        f"- vllm-jev-cache：`{report['vllm_url']}`，`block_size={report['block']}`",
        f"- native-kev：`{report['native_url']}`，按完整 state 前缀命中",
        f"- 时间：{report['started']}；每请求 3 个问题",
        "",
    ]
    for kind, title in groups.items():
        cases = [c for c in report["cases"] if c["kind"] == kind]
        if not cases:
            continue
        lines += [f"## {title}", ""] + table(cases, report["block"]) + [""]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-url", required=True)
    parser.add_argument("--native-url", required=True)
    parser.add_argument("--block", type=int, default=784)
    parser.add_argument("--aligned", default="1568,3136,6272,12544")
    parser.add_argument("--unaligned", default="1568,3136,6272")
    parser.add_argument("--seven36", default="1472,2944,5888")
    parser.add_argument("--tail", type=int, default=700)
    parser.add_argument("--margin", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--output", type=Path, default=Path("kev_align_compare.json"))
    args = parser.parse_args(argv)
    args.run_id = secrets.token_hex(4)

    vllm = Client(args.vllm_url, args.timeout)
    native = Client(args.native_url, args.timeout)
    report = {
        "vllm_url": args.vllm_url,
        "native_url": args.native_url,
        "block": args.block,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cases": [],
    }
    plan = []
    plan += [(int(x), "aligned") for x in args.aligned.split(",") if x]
    plan += [(int(x) - args.tail, "tail") for x in args.unaligned.split(",") if x]
    plan += [(int(x), "736") for x in args.seven36.split(",") if x]

    def save():
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        args.output.with_suffix(".md").write_text(markdown(report))

    try:
        for target, kind in plan:
            case = run_case(args, vllm, native, target, kind)
            report["cases"].append(case)
            save()
            v, n = case["vllm"], case["native"]
            print(
                f"{kind:8s} target {target} state {case['vllm_state_tokens']} "
                f"rem {case['remainder']:4d} | cached/row {v['warm_cached_per_row'][0]} "
                f"recomputed {v['recomputed_state_tokens'][0]} | warm "
                f"{v['warm_s']['mean']:.2f}s vs native {n['warm_s']['mean']:.2f}s "
                f"({case['warm_ratio']:.2f}x) | drift "
                f"{case['agreement']['max_delta']:.2e} flips {case['agreement']['flips']}",
                flush=True,
            )
    finally:
        vllm.close()
        native.close()
    save()
    print(f"report: {args.output} / {args.output.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
