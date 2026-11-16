"""Compare a vllm-jev Kev server against the native Kev server.

Both serve the same 27B weights and the same pointer head over the same
``/v1/systemone`` schema, so the same requests are sent to each. Keeps one
connection per worker: with keep-alive the native warm path is ~50 ms, and a
fresh TCP handshake per request would hide it.

``length``       cold and warm latency at several state lengths on both
                 backends, with the server's own ``latency_ms`` alongside.
``concurrency``  throughput and p50/p99 latency at several client counts, for
                 distinct states and for one shared state, which shows whether
                 either backend keeps a batch full.
``agreement``    the same request on both backends: do they pick the same
                 option? A different engine is expected to differ slightly;
                 a different option is a real discrepancy.

Writes a JSON and a Markdown report. Never starts or stops a server.
"""

import argparse
import http.client
import json
import secrets
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kev_cache_ab import compare_answers, percentile, summarize  # noqa: E402
from kev_cache_bench import contract, questions  # noqa: E402

TOKEN_PROBE = {"probe": {"type": "noul", "instructions": "?"}}


class Client:
    """One keep-alive connection, re-opened once if the server drops it."""

    def __init__(self, url, timeout):
        parts = urlsplit(url if "//" in url else "//" + url)
        self.host = parts.hostname
        self.port = parts.port or 80
        self.prefix = parts.path.rstrip("/")
        self.timeout = timeout
        self.conn = None

    def _connect(self):
        self.conn = http.client.HTTPConnection(
            self.host, self.port, timeout=self.timeout
        )
        return self.conn

    def request(self, method, path, body=None):
        payload = None if body is None else json.dumps(body).encode()
        headers = {"Content-Type": "application/json"} if payload else {}
        for attempt in (0, 1):
            conn = self.conn or self._connect()
            try:
                conn.request(method, self.prefix + path, payload, headers)
                response = conn.getresponse()
                text = response.read().decode()
                if response.status >= 400:
                    raise RuntimeError(f"{method} {path} -> {response.status}: {text}")
                return text
            except (http.client.HTTPException, ConnectionError, OSError):
                self.close()
                if attempt:
                    raise
        raise AssertionError("unreachable")

    def post(self, path, body):
        return json.loads(self.request("POST", path, body))

    def get(self, path):
        return json.loads(self.request("GET", path))

    def close(self):
        if self.conn is not None:
            self.conn.close()
            self.conn = None


def timed(client, body):
    started = time.perf_counter()
    response = client.post("/v1/systemone", body)
    return response, time.perf_counter() - started


def native_cache(client):
    """The native server's own cache and batch counters, from /v1/models."""
    model = client.get("/v1/models")["models"][0]
    cache, batches = model.get("prefix_cache") or {}, model.get("batches") or {}
    return {
        "hits": cache.get("hits", 0),
        "misses": cache.get("misses", 0),
        "cached_states": cache.get("cached_states", 0),
        "size": cache.get("size", 0),
        "batch_count": batches.get("count", 0),
        "batch_requests": batches.get("requests", 0),
    }


def vllm_cached_tokens(response):
    return (response.get("metadata") or {}).get("cached_tokens") or 0


def latency_stats(values):
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "p50": percentile(values, 0.50),
        "p99": percentile(values, 0.99),
        "max": max(values),
    }


def server_latencies(responses):
    """The server's own report, in seconds, where the backend gives one."""
    values = [
        r["latency_ms"] / 1000.0
        for r in responses
        if isinstance(r.get("latency_ms"), (int, float))
    ]
    return latency_stats(values) if values else None


def calibrate(client, target, tag):
    clauses = max(1, target // 22)
    for _ in range(4):
        response = client.post(
            "/v1/systemone",
            {"state": contract(clauses, tag), "questions": TOKEN_PROBE},
        )
        tokens = response["usage"]["input_tokens"]
        if abs(tokens - target) <= target * 0.02:
            break
        clauses = max(1, round(clauses * target / tokens))
    return clauses, tokens


def run_length(args, run_id, target):
    q = questions()
    fresh = Client(args.vllm_url, args.timeout)
    native = Client(args.native_url, args.timeout)
    clauses, state_tokens = calibrate(fresh, target, f"{run_id}-len{target}")
    state = contract(clauses, f"{run_id}-len{target}")
    body = {"state": state, "questions": q}
    salted = {**body, "cache_salt": f"kev-cmp-{run_id}-len{target}"}
    out = {
        "target_tokens": target,
        "state_tokens": state_tokens,
        "vllm": {},
        "native": {},
    }
    try:
        cold, cold_s = timed(fresh, salted)
        warm, warm_s = [], []
        for _ in range(args.repeats):
            response, seconds = timed(fresh, salted)
            warm.append(response)
            warm_s.append(seconds)
        out["vllm"] = {
            "cold_s": cold_s,
            "warm_s": latency_stats(warm_s),
            "cached_tokens": vllm_cached_tokens(warm[0]),
            "input_tokens": cold["usage"]["input_tokens"],
        }

        before = native_cache(native)
        n_cold, n_cold_s = timed(native, body)
        n_warm, n_warm_s = [], []
        for _ in range(args.repeats):
            response, seconds = timed(native, body)
            n_warm.append(response)
            n_warm_s.append(seconds)
        after = native_cache(native)
        out["native"] = {
            "cold_s": n_cold_s,
            "warm_s": latency_stats(n_warm_s),
            "warm_server_s": server_latencies(n_warm),
            "cold_server_s": server_latencies([n_cold]),
            "input_tokens": n_cold["usage"]["input_tokens"],
            "cache_hits_delta": after["hits"] - before["hits"],
            "cache_misses_delta": after["misses"] - before["misses"],
            "size": after["size"],
            "cached_states": after["cached_states"],
        }

        rows = compare_answers(n_cold["answers"], cold["answers"], q)
        out["agreement"] = {
            "cold": summarize(rows),
            "native_probabilities": n_warm[0]["answers"],
            "vllm_probabilities": warm[0]["answers"],
        }
        out["speedup_native_vs_vllm_warm"] = (
            out["vllm"]["warm_s"]["mean"] / out["native"]["warm_s"]["mean"]
        )
    finally:
        fresh.close()
        native.close()
    return out


def parallel(url, timeout, bodies, clients):
    """``clients`` workers, each on its own keep-alive connection.

    ``http.client`` connections are not thread-safe, so one is opened per
    worker thread and reused for that worker's requests.
    """
    local = threading.local()
    opened = []
    lock = threading.Lock()

    def send(body):
        client = getattr(local, "client", None)
        if client is None:
            client = local.client = Client(url, timeout)
            with lock:
                opened.append(client)
        return timed(client, body)

    started = time.perf_counter()
    try:
        with ThreadPoolExecutor(clients) as pool:
            results = list(pool.map(send, bodies))
    finally:
        for client in opened:
            client.close()
    wall = time.perf_counter() - started
    return [r for r, _ in results], [s for _, s in results], wall


def run_concurrency(args, run_id, clauses, scenario, clients):
    q = questions()
    count = max(clients * args.rounds, clients)
    if scenario == "distinct":
        tags = [f"{run_id}-{scenario}-{clients}-{i % clients}" for i in range(count)]
    else:
        tags = [f"{run_id}-{scenario}-{clients}"] * count
    bodies = [
        {"state": contract(clauses, t), "questions": q, "cache_salt": f"kev-cmp-{t}"}
        for t in tags
    ]
    unique = list(dict.fromkeys(tags))
    out = {"scenario": scenario, "clients": clients, "requests": count}
    for name, url, strips_salt in (
        ("vllm", args.vllm_url, False),
        ("native", args.native_url, True),
    ):
        client = Client(url, args.timeout)
        try:
            # Reference answers and a warm cache, one state at a time.
            reference = {}
            for t in unique:
                body = bodies[tags.index(t)]
                if strips_salt:
                    body = {k: v for k, v in body.items() if k != "cache_salt"}
                reference[t] = client.post("/v1/systemone", body)
            before = native_cache(client) if name == "native" else None
            payloads = [
                {k: v for k, v in b.items() if not (strips_salt and k == "cache_salt")}
                for b in bodies
            ]
            answers, seconds, wall = parallel(url, args.timeout, payloads, clients)
            after = native_cache(client) if name == "native" else None
            rows = []
            for t, response in zip(tags, answers):
                rows += compare_answers(reference[t]["answers"], response["answers"], q)
            hits = after["hits"] - before["hits"] if before and after else None
            out[name] = {
                "wall_s": wall,
                "throughput_rps": count / wall,
                "latency_s": latency_stats(seconds),
                "server_s": server_latencies(answers),
                "cache_hit_delta": hits,
                "cached_tokens_mean": statistics.fmean(
                    vllm_cached_tokens(r) for r in answers
                )
                if name == "vllm"
                else None,
                "batch_requests_delta": (
                    after["batch_requests"] - before["batch_requests"]
                    if before and after
                    else None
                ),
                "batch_count_delta": (
                    after["batch_count"] - before["batch_count"]
                    if before and after
                    else None
                ),
                "drift_vs_sequential": summarize(rows),
            }
        finally:
            client.close()
    out["throughput_gain"] = (
        out["vllm"]["throughput_rps"] / out["native"]["throughput_rps"]
    )
    return out


def fmt(value, digits=3):
    return "—" if value is None else f"{value:.{digits}f}"


def markdown(report):
    lines = [
        "# Kev 后端对比：vllm-jev（开缓存）vs 原生框架",
        "",
        f"- vllm-jev：`{report['vllm_url']}`",
        f"- 原生：`{report['native_url']}`",
        f"- 时间：{report['started']}；每个请求 3 个问题，各约 127 token",
        "- 延迟为客户端 wall-clock（长连接），括号内为服务端自报 `latency_ms`",
        "",
        "## 一、state 长度：冷启动与命中",
        "",
        "| state | vllm 冷 s | vllm 命中 s | 原生 冷 s | 原生 命中 s | 原生服务端 s "
        "| 原生快多少 | 原生命中增量 | 选项一致 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in report["length"]:
        v, n = r["vllm"], r["native"]
        same = r["agreement"]["cold"]["flips"] == 0
        lines.append(
            f"| {r['state_tokens']} | {fmt(v['cold_s'], 2)} | "
            f"{fmt(v['warm_s']['mean'], 2)} | {fmt(n['cold_s'], 2)} | "
            f"{fmt(n['warm_s']['mean'], 2)} | "
            f"{fmt((n['warm_server_s'] or {}).get('mean'), 3)} | "
            f"{fmt(r['speedup_native_vs_vllm_warm'], 1)}× | "
            f"{n['cache_hits_delta']} | {'是' if same else '否'} |"
        )
    if report["concurrency"]:
        lines += [
            "",
            "## 二、并发：continuous batching",
            "",
            "| 场景 | 并发 | 请求 | vllm req/s | 原生 req/s | vllm 更快 "
            "| vllm p50/p99 s | 原生 p50/p99 s | vllm 命中 token "
            "| 原生缓存命中 | 原生 batch 数 |",
            "|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|",
        ]
        for r in report["concurrency"]:
            v, n = r["vllm"], r["native"]
            lines.append(
                f"| {r['scenario']} | {r['clients']} | {r['requests']} | "
                f"{fmt(v['throughput_rps'])} | {fmt(n['throughput_rps'])} | "
                f"{fmt(r['throughput_gain'], 2)}× | "
                f"{fmt(v['latency_s']['p50'], 2)}/{fmt(v['latency_s']['p99'], 2)} | "
                f"{fmt(n['latency_s']['p50'], 2)}/{fmt(n['latency_s']['p99'], 2)} | "
                f"{v['cached_tokens_mean']:.0f} | {n['cache_hit_delta']} | "
                f"{n['batch_count_delta']} |"
            )
        lines += [
            "",
            "`batch 数` 是这一轮并发中服务端实际执行的批次数：远小于请求数说明"
            "请求被合并进同一个 batch（continuous batching 生效）；等于请求数"
            "说明每个请求各跑一趟。",
        ]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-url", required=True, help="vllm-jev, cache enabled")
    parser.add_argument("--native-url", required=True, help="the native Kev server")
    parser.add_argument("--lengths", default="4096,8192,16384")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--concurrency", default="1,2,4,8,16")
    parser.add_argument("--concurrency-tokens", type=int, default=8192)
    parser.add_argument("--scenarios", default="distinct,shared")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--output", type=Path, default=Path("kev_backend_compare.json"))
    args = parser.parse_args(argv)

    run_id = secrets.token_hex(4)
    report = {
        "vllm_url": args.vllm_url,
        "native_url": args.native_url,
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
        v, n = result["vllm"], result["native"]
        print(
            f"length {result['state_tokens']}: vllm cold {v['cold_s']:.2f} "
            f"warm {v['warm_s']['mean']:.2f} | native cold {n['cold_s']:.2f} "
            f"warm {n['warm_s']['mean']:.2f} "
            f"(server {(n['warm_server_s'] or {}).get('mean', 0):.3f}) "
            f"flips {result['agreement']['cold']['flips']}",
            flush=True,
        )
    levels = [int(x) for x in args.concurrency.split(",") if x]
    if levels:
        client = Client(args.vllm_url, args.timeout)
        try:
            clauses, tokens = calibrate(
                client, args.concurrency_tokens, f"{run_id}-conc"
            )
        finally:
            client.close()
        report["concurrency_state_tokens"] = tokens
        for scenario in [s for s in args.scenarios.split(",") if s]:
            for clients in levels:
                result = run_concurrency(args, run_id, clauses, scenario, clients)
                report["concurrency"].append(result)
                save()
                v, n = result["vllm"], result["native"]
                print(
                    f"{scenario} c={clients}: vllm {v['throughput_rps']:.2f} rps "
                    f"native {n['throughput_rps']:.2f} rps "
                    f"native batches {n['batch_count_delta']}/{result['requests']} "
                    f"hits {n['cache_hit_delta']}",
                    flush=True,
                )
    save()
    print(f"report: {args.output} / {args.output.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
