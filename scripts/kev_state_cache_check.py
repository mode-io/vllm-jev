"""Does a cache-enabled Kev server cache the state, and cache it correctly?

Runs against one already-running server started with prefix caching on (the
default, ``VLLM_JEV_KEV_PREFIX_CACHE`` unset or ``1``). Each check sends real
``/v1/systemone`` requests and reads ``metadata.cached_tokens`` plus the
engine's ``vllm:prefix_cache_*`` counters:

1. patched   the server runs the patched code (metadata fields, salt accepted)
2. reuse     the same state under the same salt is a miss, then a hit
3. answers   the hit gives the same answer as the cold request
4. sharing   new questions about a cached state still reuse it
5. isolation another salt never reads the first salt's blocks
6. content   a state that differs early is not served from the old blocks,
             and two states sharing a long prefix keep their own answers

Writes a JSON report and a Markdown report. Never starts or stops a server.
"""

import argparse
import json
import secrets
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kev_cache_ab import (  # noqa: E402
    HIT_METRIC,
    QUERY_METRIC,
    compare_answers,
    metrics,
    post_systemone,
)

CHOICE = {
    "party": {
        "type": "choice",
        "instructions": "Which party must deliver the goods?",
        "criteria": {"supplier": None, "buyer": None},
    }
}
OTHER = {
    "binding": {"type": "noul", "instructions": "Is delivery mandatory?"},
}


def contract(clauses, deliverer="supplier", lead="Master supply agreement."):
    """A long state whose final clause decides who delivers."""
    body = " ".join(
        f"Clause {i}: invoices under schedule {i} are payable within "
        f"{i % 30 + 1} days of receipt."
        for i in range(clauses)
    )
    return f"{lead} {body} Final clause: the {deliverer} must deliver the goods."


class Check:
    def __init__(self, name, title):
        self.name, self.title = name, title
        self.passed, self.notes, self.data = True, [], {}

    def expect(self, ok, message):
        self.notes.append(("ok" if ok else "FAIL") + ": " + message)
        self.passed = self.passed and bool(ok)

    def as_dict(self):
        return {
            "name": self.name,
            "title": self.title,
            "passed": self.passed,
            "notes": self.notes,
            "data": self.data,
        }


class Runner:
    def __init__(self, url, timeout, run_id):
        self.url, self.timeout, self.run_id = url, timeout, run_id

    def salt(self, label):
        return f"kev-check-{self.run_id}-{label}"

    def ask(self, state, questions, salt=None):
        body = {"state": state, "questions": questions}
        if salt is not None:
            body["cache_salt"] = salt
        started = time.perf_counter()
        response = post_systemone(self.url, body, self.timeout)
        elapsed = time.perf_counter() - started
        meta = response.get("metadata") or {}
        return {
            "answers": response["answers"],
            "cached": meta.get("cached_tokens"),
            "reruns": meta.get("uncached_reruns"),
            "compute": (response.get("internal_usage") or {}).get("compute_tokens"),
            "input": (response.get("usage") or {}).get("input_tokens"),
            "seconds": elapsed,
            "metadata": meta,
        }

    def counters(self):
        return metrics(self.url, self.timeout)


def counter_delta(before, after):
    return {
        key: (after.get(key) or 0.0) - (before.get(key) or 0.0)
        for key in ("queries", "hits")
    }


def check_patched(run, state):
    check = Check("patched", "服务运行的是补丁版代码")
    try:
        result = run.ask(state, CHOICE, salt=run.salt("probe"))
    except Exception as error:  # 422 on unpatched code
        check.expect(False, f"带 cache_salt 的请求被拒绝：{error}")
        return check
    check.data = {"metadata": result["metadata"]}
    check.expect(
        "cached_tokens" in result["metadata"]
        and "uncached_reruns" in result["metadata"],
        "metadata 含 cached_tokens 与 uncached_reruns",
    )
    check.expect(True, "带 cache_salt 的请求被接受")
    return check


def check_reuse(run, state, repeats):
    check = Check("reuse", "同一 state、同一 salt：先未命中，再命中")
    salt = run.salt("reuse")
    before = run.counters()
    cold = run.ask(state, CHOICE, salt=salt)
    warm = [run.ask(state, CHOICE, salt=salt) for _ in range(repeats)]
    delta = counter_delta(before, run.counters())
    check.data = {
        "state_tokens_approx": cold["compute"],
        "cold": {k: cold[k] for k in ("cached", "reruns", "compute", "seconds")},
        "warm": [{k: w[k] for k in ("cached", "reruns", "seconds")} for w in warm],
        "counters": delta,
        "cold_seconds": cold["seconds"],
        "warm_seconds_mean": statistics.fmean(w["seconds"] for w in warm),
    }
    check.expect(cold["cached"] == 0, f"首个请求 cached_tokens={cold['cached']}")
    hits = [w["cached"] or 0 for w in warm]
    check.expect(all(h > 0 for h in hits), f"后续请求 cached_tokens={hits}")
    if cold["compute"]:
        check.expect(
            all(h < cold["compute"] for h in hits),
            f"命中数小于整条序列长度 {cold['compute']}（读数位置仍被重算）",
        )
    check.expect(
        all((w["reruns"] or 0) == 0 for w in warm),
        f"uncached_reruns={[w['reruns'] for w in warm]}（命中没有越过读数位置）",
    )
    check.expect(delta["queries"] > 0, f"{QUERY_METRIC} 增长 {delta['queries']:.0f}")
    check.expect(delta["hits"] > 0, f"{HIT_METRIC} 增长 {delta['hits']:.0f}")
    check._cold, check._warm = cold, warm
    return check


def check_answers(reuse, max_delta):
    check = Check("answers", "命中缓存的答案与冷启动答案一致")
    cold, warm = getattr(reuse, "_cold", None), getattr(reuse, "_warm", None)
    if not cold or not warm:
        check.expect(False, "reuse 检查没有产出数据")
        return check
    rows = []
    for w in warm:
        rows += compare_answers(cold["answers"], w["answers"], CHOICE)
    worst = max(row["max_delta"] for row in rows)
    flips = sum(row["flipped"] for row in rows)
    check.data = {
        "cold_probabilities": cold["answers"]["party"]["probabilities"],
        "warm_probabilities": [w["answers"]["party"]["probabilities"] for w in warm],
        "max_delta": worst,
        "flips": flips,
    }
    check.expect(flips == 0, f"argmax 翻转 {flips} 次")
    check.expect(worst <= max_delta, f"最大概率差 {worst:.6f}（上限 {max_delta}）")
    return check


def check_sharing(run, state):
    check = Check("sharing", "换一组问题，同一 state 仍复用缓存")
    salt = run.salt("sharing")
    run.ask(state, CHOICE, salt=salt)
    other = run.ask(state, OTHER, salt=salt)
    check.data = {"cached": other["cached"], "compute": other["compute"]}
    check.expect(
        (other["cached"] or 0) > 0,
        f"新问题请求 cached_tokens={other['cached']}（state 部分被复用）",
    )
    return check


def check_isolation(run, state):
    check = Check("isolation", "不同 salt 互不读取对方的缓存")
    run.ask(state, CHOICE, salt=run.salt("isolation-a"))
    other = run.ask(state, CHOICE, salt=run.salt("isolation-b"))
    fresh = run.ask(state, CHOICE)
    check.data = {
        "other_salt_cached": other["cached"],
        "no_salt_cached": fresh["cached"],
    }
    check.expect(
        other["cached"] == 0, f"另一个 salt 的 cached_tokens={other['cached']}"
    )
    check.expect(fresh["cached"] == 0, f"不带 salt 的 cached_tokens={fresh['cached']}")
    return check


def check_content(run, clauses):
    check = Check("content", "缓存按内容命中：改动的 state 不会读到旧块")
    salt = run.salt("content")
    supplier = run.ask(contract(clauses, "supplier"), CHOICE, salt=salt)
    # Same long prefix, different final clause: a correct cache reuses the
    # prefix and still reads the new ending.
    buyer = run.ask(contract(clauses, "buyer"), CHOICE, salt=salt)
    # Different first sentence: nothing after it may come from the cache.
    early = run.ask(
        contract(clauses, "supplier", lead="Amended distribution agreement."),
        CHOICE,
        salt=salt,
    )
    pick_s = supplier["answers"]["party"]["choice"]
    pick_b = buyer["answers"]["party"]["choice"]
    check.data = {
        "supplier": {"choice": pick_s, "cached": supplier["cached"]},
        "buyer": {"choice": pick_b, "cached": buyer["cached"]},
        "early_change": {"cached": early["cached"]},
        "supplier_probabilities": supplier["answers"]["party"]["probabilities"],
        "buyer_probabilities": buyer["answers"]["party"]["probabilities"],
    }
    check.expect(
        (buyer["cached"] or 0) > 0,
        f"共享长前缀的新 state 复用了前缀：cached_tokens={buyer['cached']}",
    )
    check.expect(
        pick_s == "supplier" and pick_b == "buyer",
        f"结尾不同、答案各自正确：supplier→{pick_s}，buyer→{pick_b}",
    )
    check.expect(
        early["cached"] == 0,
        f"开头就不同的 state cached_tokens={early['cached']}",
    )
    return check


def markdown(report):
    lines = [
        "# Kev state 缓存验证报告",
        "",
        f"- 服务：`{report['url']}`",
        f"- 时间：{report['started']}",
        f"- state：约 {report['clauses']} 条条款",
        f"- 结论：**{'通过' if report['passed'] else '未通过'}**"
        f"（{sum(c['passed'] for c in report['checks'])}/{len(report['checks'])}）",
        "",
        "| 检查 | 内容 | 结果 |",
        "|---|---|---|",
    ]
    for c in report["checks"]:
        lines.append(
            f"| {c['name']} | {c['title']} | {'✅' if c['passed'] else '❌'} |"
        )
    for c in report["checks"]:
        lines += ["", f"## {c['name']}：{c['title']}", ""]
        lines += [f"- {note}" for note in c["notes"]]
        if c["data"]:
            lines += [
                "",
                "```json",
                json.dumps(c["data"], ensure_ascii=False, indent=2),
            ]
            lines += ["```"]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="the cache-enabled server")
    parser.add_argument(
        "--clauses",
        type=int,
        default=180,
        help="clauses per state; the state must span several 784-token blocks",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-delta", type=float, default=0.05)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--output", type=Path, default=Path("kev_state_cache.json"))
    args = parser.parse_args(argv)

    run = Runner(args.url, args.timeout, secrets.token_hex(4))
    state = contract(args.clauses)
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    checks = [check_patched(run, state)]
    if checks[0].passed:
        reuse = check_reuse(run, state, args.repeats)
        checks += [
            reuse,
            check_answers(reuse, args.max_delta),
            check_sharing(run, state),
            check_isolation(run, state),
            check_content(run, args.clauses),
        ]
    report = {
        "url": args.url,
        "started": started,
        "clauses": args.clauses,
        "passed": all(c.passed for c in checks),
        "checks": [c.as_dict() for c in checks],
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    args.output.with_suffix(".md").write_text(markdown(report))
    for c in checks:
        print(f"{'PASS' if c.passed else 'FAIL'}  {c.name:9s} {c.title}")
        for note in c.notes:
            print(f"        {note}")
    print(f"report: {args.output} / {args.output.with_suffix('.md')}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
