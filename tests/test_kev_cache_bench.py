"""Benchmark script: it runs end to end against a fake server pair.

No GPU and no vLLM: a stdlib HTTP server prices a request by its token count,
caches whole blocks per salt, and answers like Kev.
"""

import importlib.util
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BLOCK = 32
PER_CLAUSE = 20


def bench():
    sys.path.insert(0, str(ROOT / "scripts"))
    path = ROOT / "scripts/kev_cache_bench.py"
    spec = importlib.util.spec_from_file_location("kev_cache_bench", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Server:
    """Reads the cache only when ``reads`` is set, like the two real servers."""

    def __init__(self, reads):
        self.reads = reads
        self.blocks, self.queries, self.hits = set(), 0, 0
        self.httpd = HTTPServer(("127.0.0.1", 0), self.handler())
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join()

    @property
    def url(self):
        return f"http://127.0.0.1:{self.httpd.server_port}"

    def tokens(self, body):
        words = len(body["state"].split())
        per_question = sum(
            len((spec.get("instructions") or "").split())
            + len(json.dumps(spec.get("criteria") or []).split())
            for spec in body["questions"].values()
        )
        return words + per_question + 8

    def answer(self, body):
        total = self.tokens(body)
        cached = 0
        salt = body.get("cache_salt") or object()
        if self.reads:
            state_len = len(body["state"].split())
            for end in range(BLOCK, state_len, BLOCK):
                key = (salt, body["state"][: end * PER_CLAUSE])
                self.queries += BLOCK
                if key in self.blocks:
                    cached = end * PER_CLAUSE
                    self.hits += BLOCK
                self.blocks.add(key)
        deliverer = "buyer" if "the buyer must" in body["state"] else "supplier"
        other = "buyer" if deliverer == "supplier" else "supplier"
        probs = {deliverer: 0.98, other: 0.02}
        answers = {}
        for name, spec in body["questions"].items():
            kind = spec["type"]
            if kind == "noul":
                answers[name] = {"type": "noul", "noul": 0.75}
            elif kind == "score":
                scale = {str(i): w for i, w in enumerate((0.1, 0.8, 0.1))}
                answers[name] = {"type": "score", "probabilities": scale}
            else:
                answers[name] = {"type": "choice", "probabilities": probs}
        return {
            "answers": answers,
            "usage": {"input_tokens": total, "output_tokens": 0},
            "internal_usage": {"compute_tokens": total},
            "metadata": {"cached_tokens": cached, "uncached_reruns": 0},
        }

    def handler(self):
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def reply(self, payload, kind="application/json"):
                self.send_response(200)
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_GET(self):
                self.reply(
                    (
                        f'vllm:prefix_cache_queries_total{{engine="0"}} '
                        f"{outer.queries}\n"
                        f'vllm:prefix_cache_hits_total{{engine="0"}} {outer.hits}\n'
                    ).encode(),
                    "text/plain",
                )

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                self.reply(json.dumps(outer.answer(body)).encode())

        return Handler


def run(tmp_path, lengths="1000", concurrency="1,2", scenarios="distinct,shared"):
    output = tmp_path / "bench.json"
    with Server(reads=False) as baseline:
        with Server(reads=True) as cached:
            code = bench().main(
                [
                    "--baseline-url",
                    baseline.url,
                    "--cached-url",
                    cached.url,
                    "--lengths",
                    lengths,
                    "--repeats",
                    "2",
                    "--concurrency",
                    concurrency,
                    "--concurrency-tokens",
                    "1000",
                    "--scenarios",
                    scenarios,
                    "--rounds",
                    "1",
                    "--output",
                    str(output),
                ]
            )
    return code, json.loads(output.read_text()), output


def test_lengths_report_tokens_and_drift(tmp_path):
    code, report, output = run(tmp_path)
    assert code == 0
    row = report["length"][0]
    assert row["target_tokens"] == 1000
    assert 900 < row["state_tokens"] < 1150
    assert row["warm_cached_tokens"][0] > 0
    assert row["drift"]["max_delta"] == 0.0
    assert row["drift"]["flips"] == 0
    assert row["speedup"] > 0
    text = output.with_suffix(".md").read_text()
    assert "state 长度" in text and "并发" in text


def test_concurrency_covers_both_scenarios(tmp_path):
    _, report, _ = run(tmp_path, concurrency="1,2")
    seen = {(r["scenario"], r["clients"]) for r in report["concurrency"]}
    assert seen == {("distinct", 1), ("distinct", 2), ("shared", 1), ("shared", 2)}
    for row in report["concurrency"]:
        assert row["cached"]["cached_tokens_mean"] >= 0
        # The fake is deterministic, so no batch composition moves an answer.
        assert row["cached"]["drift_vs_sequential"]["max_delta"] == 0.0
        assert row["baseline"]["drift_vs_sequential"]["flips"] == 0
        assert row["requests"] >= row["clients"]
    shared = [r for r in report["concurrency"] if r["scenario"] == "shared"]
    assert shared[0]["cached"]["cached_tokens_mean"] > 0


def test_concurrency_can_be_skipped(tmp_path):
    _, report, _ = run(tmp_path, concurrency="")
    assert report["concurrency"] == []
