"""Alignment comparison script: block-granular vs exact-prefix caching.

No GPU: one stdlib server caches in blocks like vllm-jev and reports
``metadata.cached_tokens``, the other caches whole states like the native Kev
server and reports ``latency_ms`` plus counters on ``/v1/models``.
"""

import importlib.util
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BLOCK = 784
OVERHEAD = 127  # tokens a question adds after the state


def aligner():
    sys.path.insert(0, str(ROOT / "scripts"))
    path = ROOT / "scripts/kev_align_compare.py"
    spec = importlib.util.spec_from_file_location("kev_align_compare", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def answers(body):
    out = {}
    for name, spec in body["questions"].items():
        kind = spec["type"]
        if kind == "noul":
            out[name] = {"type": "noul", "noul": 0.8}
        elif kind == "score":
            out[name] = {
                "type": "score",
                "probabilities": {"0": 0.1, "1": 0.7, "2": 0.2},
            }
        else:
            out[name] = {
                "type": "choice",
                "probabilities": {"supplier": 0.97, "buyer": 0.03},
            }
    return out


class Server:
    """``blocks`` caches block prefixes; otherwise an exact-prefix cache."""

    def __init__(self, blocks=True):
        self.blocks = blocks
        self.seen, self.hits, self.misses = set(), 0, 0
        self.batches = self.requests = 0
        self.lock = threading.Lock()
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

    def state_tokens(self, body):
        return len(body["state"].split()) + 1

    def answer(self, body):
        state = self.state_tokens(body)
        row = state + OVERHEAD
        salt = body.get("cache_salt")
        key = (salt, body["state"], json.dumps(sorted(body["questions"])))
        with self.lock:
            hit = key in self.seen
            self.seen.add(key)
            self.hits += hit
            self.misses += not hit
            self.batches += 1
            self.requests += 1
        response = {
            "answers": answers(body),
            "usage": {"input_tokens": row, "output_tokens": 0},
        }
        if self.blocks:
            # vLLM sums cached tokens over a request's question rows, and each
            # row caches the full blocks of its own prompt.
            rows = len(body["questions"])
            per_row = (row // BLOCK) * BLOCK if hit else 0
            response["metadata"] = {
                "cached_tokens": per_row * rows,
                "uncached_reruns": 0,
            }
        else:
            response["latency_ms"] = 40.0 if hit else 900.0
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


def run(tmp_path, aligned="1568", tail="", seven36="", repeat_clause=""):
    output = tmp_path / "align.json"
    with Server(blocks=True) as vllm, Server(blocks=False) as native:
        code = aligner().main(
            [
                "--vllm-url",
                vllm.url,
                "--native-url",
                native.url,
                "--block",
                str(BLOCK),
                "--aligned",
                aligned,
                "--unaligned",
                tail,
                "--seven36",
                seven36,
                "--tail",
                "700",
                "--repeats",
                "2",
                "--output",
                str(output),
            ]
        )
    return code, json.loads(output.read_text()), output


def test_an_aligned_state_is_cached_whole_and_beats_the_whole_state_cache(tmp_path):
    code, report, output = run(tmp_path)
    assert code == 0
    case = report["cases"][0]
    assert case["aligned"] is True
    assert case["remainder"] == 0
    assert case["vllm_state_tokens"] == 1568
    # the 127-token question tail does not fill a third block, so exactly the
    # state's two blocks hit and nothing of the state is recomputed
    assert case["vllm"]["warm_cached_per_row"] == [1568, 1568]
    assert case["vllm"]["recomputed_state_tokens"] == [0, 0]
    assert case["vllm"]["state_fully_cached"] is True
    assert case["vllm"]["warm_reruns"] == [0, 0]
    # The fake's whole-state cache still answers; only the timing differs.
    assert case["agreement"]["flips"] == 0
    text = output.with_suffix(".md").read_text()
    assert "块对齐对比" in text and "整数倍" in text


def test_an_unaligned_state_recomputes_its_tail(tmp_path):
    _, report, _ = run(tmp_path, aligned="", tail="1568", seven36="1472")
    tail, seven = report["cases"]
    assert tail["kind"] == "tail" and tail["aligned"] is False
    assert tail["remainder"] == BLOCK - 700
    assert tail["vllm"]["recomputed_state_tokens"][0] > 0
    assert tail["vllm"]["state_fully_cached"] is False
    # 1472 is not a multiple of 784; its row still fills two blocks.
    assert seven["kind"] == "736"
    assert seven["remainder"] != 0


def test_the_native_server_is_shown_to_have_hit(tmp_path):
    _, report, _ = run(tmp_path)
    native = report["cases"][0]["native"]
    assert native["cache_hits_delta"] >= 2
    assert native["cache_misses_delta"] == 0
    assert native["warm_all_hit"] is True
