"""State-cache check: each check passes on a correct cache and fails a broken one.

No GPU and no vLLM: a stdlib HTTP server emulates block-granular prefix
caching keyed by (salt, token prefix), the way vLLM hashes blocks.
"""

import importlib.util
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BLOCK = 16


def checker():
    sys.path.insert(0, str(ROOT / "scripts"))
    path = ROOT / "scripts/kev_state_cache_check.py"
    spec = importlib.util.spec_from_file_location("kev_state_cache_check", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Server:
    """Caches whole blocks per salt; ``broken`` modes model real failure shapes."""

    def __init__(self, broken=None, patched=True):
        self.broken, self.patched = broken, patched
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

    def answer(self, body):
        state = body["state"]
        words = (state + " || " + json.dumps(body["questions"])).split()
        salt = body.get("cache_salt") or object()
        if self.broken == "shared_salts":
            salt = "everyone"
        cached = 0
        # Only full blocks strictly inside the state are cacheable, like the
        # readout positions that always follow it.
        state_len = len(state.split())
        for end in range(BLOCK, state_len, BLOCK):
            key = (salt, tuple(words[:end]))
            self.queries += BLOCK
            if key in self.blocks:
                cached = end
                self.hits += BLOCK
            self.blocks.add(key)
        if self.broken == "never_hits":
            cached, self.hits = 0, 0
        deliverer = "buyer" if "the buyer must" in state else "supplier"
        if self.broken == "stale" and cached:
            deliverer = "supplier"
        drift = 0.2 if self.broken == "drift" and cached else 0.0
        high = 0.9 - drift
        probs = {
            deliverer: high,
            ("buyer" if deliverer == "supplier" else "supplier"): 1 - high,
        }
        answers = {}
        for key, spec in body["questions"].items():
            if spec["type"] == "noul":
                answers[key] = {"type": "noul", "noul": 0.8}
            else:
                answers[key] = {
                    "type": "choice",
                    "probabilities": probs,
                    "choice": max(probs, key=probs.get),
                }
        meta = {"inference_seconds": 0.01}
        if self.patched:
            meta.update(cached_tokens=cached, uncached_reruns=0)
        return {
            "answers": answers,
            "usage": {"input_tokens": len(words)},
            "internal_usage": {"compute_tokens": len(words)},
            "metadata": meta,
        }

    def handler(self):
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def reply(self, code, payload, kind="application/json"):
                self.send_response(code)
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_GET(self):
                text = (
                    f'vllm:prefix_cache_queries_total{{engine="0"}} {outer.queries}\n'
                    f'vllm:prefix_cache_hits_total{{engine="0"}} {outer.hits}\n'
                )
                self.reply(200, text.encode(), "text/plain")

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if not outer.patched and "cache_salt" in body:
                    self.reply(422, b'{"error":{"code":422}}')
                    return
                self.reply(200, json.dumps(outer.answer(body)).encode())

        return Handler


def run(server, tmp_path):
    output = tmp_path / "report.json"
    code = checker().main(
        [
            "--url",
            server.url,
            "--clauses",
            "12",
            "--repeats",
            "2",
            "--output",
            str(output),
        ]
    )
    report = json.loads(output.read_text())
    return code, {c["name"]: c["passed"] for c in report["checks"]}, output


def test_a_correct_cache_passes_every_check(tmp_path):
    with Server() as server:
        code, checks, output = run(server, tmp_path)
    assert code == 0
    assert checks == {
        "patched": True,
        "reuse": True,
        "answers": True,
        "sharing": True,
        "isolation": True,
        "content": True,
    }
    assert "Kev state 缓存验证报告" in output.with_suffix(".md").read_text()


def test_unpatched_code_stops_after_the_first_check(tmp_path):
    with Server(patched=False) as server:
        code, checks, _ = run(server, tmp_path)
    assert code == 1
    assert checks == {"patched": False}


def test_a_cache_that_never_hits_fails_reuse(tmp_path):
    with Server(broken="never_hits") as server:
        code, checks, _ = run(server, tmp_path)
    assert code == 1
    assert checks["reuse"] is False
    assert checks["sharing"] is False


def test_salts_that_share_blocks_fail_isolation(tmp_path):
    with Server(broken="shared_salts") as server:
        code, checks, _ = run(server, tmp_path)
    assert code == 1
    assert checks["isolation"] is False


def test_a_hit_that_returns_stale_state_fails_content(tmp_path):
    with Server(broken="stale") as server:
        code, checks, _ = run(server, tmp_path)
    assert code == 1
    assert checks["content"] is False


def test_a_hit_that_moves_probabilities_fails_answers(tmp_path):
    with Server(broken="drift") as server:
        code, checks, _ = run(server, tmp_path)
    assert code == 1
    assert checks["answers"] is False
    assert checks["reuse"] is True
