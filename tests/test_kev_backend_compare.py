"""Backend comparison script: it runs end to end against a fake pair.

No GPU: one stdlib server answers like vllm-jev (``metadata.cached_tokens``),
the other like the native Kev server (``latency_ms`` plus cache and batch
counters on ``/v1/models``).
"""

import importlib.util
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def compare():
    sys.path.insert(0, str(ROOT / "scripts"))
    path = ROOT / "scripts/kev_backend_compare.py"
    spec = importlib.util.spec_from_file_location("kev_backend_compare", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def answers(body, shift=0.0):
    out = {}
    for name, spec in body["questions"].items():
        kind = spec["type"]
        if kind == "noul":
            out[name] = {"type": "noul", "noul": 0.8 - shift}
        elif kind == "score":
            out[name] = {
                "type": "score",
                "probabilities": {"0": 0.1, "1": 0.7 - shift, "2": 0.2 + shift},
            }
        else:
            out[name] = {
                "type": "choice",
                "probabilities": {"supplier": 0.97 - shift, "buyer": 0.03 + shift},
            }
    return out


class Server:
    def __init__(self, native):
        self.native = native
        self.seen, self.hits, self.misses = set(), 0, 0
        self.batches = self.requests = 0
        self.lock = threading.Lock()
        self.salted = []
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), self.handler())
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
        tokens = len(body["state"].split()) + 30
        with self.lock:
            key = (body.get("cache_salt"), body["state"])
            hit = key in self.seen
            self.seen.add(key)
            self.hits += hit
            self.misses += not hit
            self.batches += 1
            self.requests += 1
            if "cache_salt" in body:
                self.salted.append(body["cache_salt"])
        response = {
            "answers": answers(body, shift=0.001 if self.native else 0.0),
            "usage": {"input_tokens": tokens, "output_tokens": 0},
        }
        if self.native:
            response["latency_ms"] = 40.0 if hit else 900.0
        else:
            response["metadata"] = {"cached_tokens": tokens // 2 if hit else 0}
        return response

    def models(self):
        return {
            "models": [
                {
                    "name": "kev-latest",
                    "prefix_cache": {
                        "size": 4,
                        "hits": self.hits,
                        "misses": self.misses,
                        "cached_states": min(4, len(self.seen)),
                    },
                    "batches": {"count": self.batches, "requests": self.requests},
                }
            ]
        }

    def handler(self):
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def reply(self, payload):
                data = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                self.reply(outer.models())

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                self.reply(outer.answer(body))

        return Handler


def run(tmp_path, **overrides):
    options = {
        "lengths": "800",
        "concurrency": "1,3",
        "scenarios": "distinct,shared",
    }
    options.update(overrides)
    output = tmp_path / "compare.json"
    with Server(native=False) as vllm, Server(native=True) as native:
        code = compare().main(
            [
                "--vllm-url",
                vllm.url,
                "--native-url",
                native.url,
                "--lengths",
                options["lengths"],
                "--repeats",
                "2",
                "--concurrency",
                options["concurrency"],
                "--concurrency-tokens",
                "800",
                "--scenarios",
                options["scenarios"],
                "--rounds",
                "2",
                "--output",
                str(output),
            ]
        )
        salted_native = list(native.salted)
    return code, json.loads(output.read_text()), output, salted_native


def test_length_compares_cold_warm_and_agreement(tmp_path):
    code, report, output, _ = run(tmp_path, concurrency="")
    assert code == 0
    row = report["length"][0]
    assert row["vllm"]["cached_tokens"] > 0
    assert row["native"]["cache_hits_delta"] == 2
    assert row["native"]["warm_server_s"]["mean"] == 0.04
    # The engines differ a little and never pick a different option here.
    assert row["agreement"]["cold"]["flips"] == 0
    assert 0 < row["agreement"]["cold"]["max_delta"] < 0.01
    assert "原生" in output.with_suffix(".md").read_text()


def test_concurrency_reports_both_backends_and_batches(tmp_path):
    _, report, output, salted_native = run(tmp_path, lengths="")
    seen = {(r["scenario"], r["clients"]) for r in report["concurrency"]}
    assert seen == {("distinct", 1), ("distinct", 3), ("shared", 1), ("shared", 3)}
    for row in report["concurrency"]:
        assert row["requests"] == row["clients"] * 2
        assert row["native"]["batch_count_delta"] == row["requests"]
        assert row["native"]["cache_hit_delta"] == row["requests"]
        assert row["vllm"]["drift_vs_sequential"]["flips"] == 0
        assert row["throughput_gain"] > 0
    # The native server has no cache_salt; it is never sent one.
    assert salted_native == []
    assert "continuous batching" in output.with_suffix(".md").read_text()
