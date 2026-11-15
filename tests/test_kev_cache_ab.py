"""Kev prefix-cache A/B script: parsing, drift, and the end-to-end verdict.

No GPU and no vLLM: a stdlib HTTP server answers /v1/systemone and /metrics,
with the drift and the hit counters the script must detect.
"""

import importlib.util
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def runner():
    path = ROOT / "scripts/kev_cache_ab.py"
    spec = importlib.util.spec_from_file_location("kev_cache_ab", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


QUESTIONS = {
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


def answers(party, binding, strictness):
    return {
        "party": {
            "type": "choice",
            "probabilities": {"supplier": party, "buyer": 1 - party},
            "choice": "supplier" if party >= 0.5 else "buyer",
        },
        "binding": {"type": "noul", "noul": binding},
        "strictness": {
            "type": "score",
            "probabilities": {
                "0": strictness[0],
                "1": strictness[1],
                "2": strictness[2],
            },
            "score": strictness[2] * 2 + strictness[1],
        },
    }


# ---- pure functions


def test_noul_argmax_uses_one_half_not_the_option_order():
    # A Noul answer carries one probability; its pick is true above one half.
    assert runner().argmax_of("noul", {"noul": 0.6}) == "true"
    assert runner().argmax_of("noul", {"noul": 0.4}) == "false"


def test_identical_runs_report_no_drift_and_no_flips():
    left = answers(0.90, 0.80, [0.1, 0.2, 0.7])
    rows = runner().compare_answers(left, left, QUESTIONS)
    stats = runner().summarize(rows)
    assert stats["max_delta"] == 0.0
    assert stats["flips"] == 0
    assert {row["question"] for row in rows} == set(QUESTIONS)


def test_drift_without_a_flip_is_not_a_flip():
    left = answers(0.90, 0.80, [0.1, 0.2, 0.7])
    right = answers(0.70, 0.60, [0.2, 0.2, 0.6])
    stats = runner().summarize(runner().compare_answers(left, right, QUESTIONS))
    assert stats["flips"] == 0
    assert stats["max_delta"] == pytest.approx(0.20, abs=1e-9)


def test_a_crossed_option_is_reported_as_a_flip():
    left = answers(0.52, 0.51, [0.1, 0.2, 0.7])
    right = answers(0.48, 0.49, [0.1, 0.2, 0.7])
    stats = runner().summarize(runner().compare_answers(left, right, QUESTIONS))
    assert stats["flips"] == 2
    assert stats["flipped"] == ["party", "binding"]


def test_a_flip_fails_under_the_default_limits():
    left = answers(0.52, 0.51, [0.1, 0.2, 0.7])
    right = answers(0.48, 0.49, [0.1, 0.2, 0.7])
    rows = runner().compare_answers(left, right, QUESTIONS)
    result = runner().verdict(runner().summarize(rows), 0.05, 0)
    assert not result["passed"]
    assert any("argmax flips" in reason for reason in result["reasons"])


def test_pure_drift_beyond_the_limit_fails_even_without_a_flip():
    left = answers(0.90, 0.80, [0.1, 0.2, 0.7])
    right = answers(0.70, 0.60, [0.2, 0.2, 0.6])
    rows = runner().compare_answers(left, right, QUESTIONS)
    result = runner().verdict(runner().summarize(rows), 0.05, 0)
    assert not result["passed"]
    assert any("probability delta" in reason for reason in result["reasons"])


def test_small_drift_passes():
    left = answers(0.90, 0.80, [0.1, 0.2, 0.7])
    right = answers(0.895, 0.799, [0.1, 0.2, 0.7])
    rows = runner().compare_answers(left, right, QUESTIONS)
    assert runner().verdict(runner().summarize(rows), 0.05, 0)["passed"]


def test_a_different_option_set_is_an_error_not_a_skip():
    left = answers(0.9, 0.8, [0.1, 0.2, 0.7])
    right = json.loads(json.dumps(left))
    right["party"]["probabilities"] = {"supplier": 0.9, "vendor": 0.1}
    with pytest.raises(ValueError, match="different options"):
        runner().compare_answers(left, right, QUESTIONS)


def test_parse_metric_sums_every_label_set():
    text = (
        "# HELP vllm:prefix_cache_hits_total hits\n"
        "# TYPE vllm:prefix_cache_hits_total counter\n"
        'vllm:prefix_cache_hits_total{engine="0",model_name="kev"} 3.0\n'
        'vllm:prefix_cache_hits_total{engine="1",model_name="kev"} 4.0\n'
        "unrelated_metric_total 99.0\n"
    )
    assert runner().parse_metric(text, "vllm:prefix_cache_hits_total") == 7.0


def test_parse_metric_distinguishes_absent_from_zero():
    assert (
        runner().parse_metric("other_metric 1.0\n", "vllm:prefix_cache_hits_total")
        is None
    )


def test_percentile_is_nearest_rank_on_sorted_values():
    values = [0.5, 0.1, 0.9, 0.3]
    assert runner().percentile(values, 0.50) == 0.3
    assert runner().percentile(values, 0.99) == 0.9
    assert runner().percentile([0.2], 0.99) == 0.2


# ---- end to end over HTTP


class Server:
    """A System One server whose answers drift and whose cache counters move."""

    def __init__(self, drift, hits_each_request):
        self.drift = drift
        self.hits = 0
        self.hits_each_request = hits_each_request
        self.requests = 0
        self.salts = []
        handler = self.handler()
        self.httpd = HTTPServer(("127.0.0.1", 0), handler)
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

    def handler(self):
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                body = (
                    "# TYPE vllm:prefix_cache_queries_total counter\n"
                    f'vllm:prefix_cache_queries_total{{engine="0"}} {outer.requests}\n'
                    "# TYPE vllm:prefix_cache_hits_total counter\n"
                    f'vllm:prefix_cache_hits_total{{engine="0"}} {outer.hits}\n'
                ).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                length = int(self.headers["Content-Length"])
                body = json.loads(self.rfile.read(length))
                outer.requests += 1
                outer.salts.append(body.get("cache_salt"))
                outer.hits += outer.hits_each_request
                d = outer.drift
                payload = json.dumps(
                    {"answers": answers(0.90 - d, 0.80 - d, [0.1, 0.2 + d, 0.7 - d])}
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        return Handler


def test_end_to_end_passes_when_the_cache_only_moves_probabilities(tmp_path):
    output = tmp_path / "report.json"
    with Server(drift=0.0, hits_each_request=0) as baseline:
        with Server(drift=0.01, hits_each_request=8) as cached:
            code = runner().main(
                [
                    "--baseline-url",
                    baseline.url,
                    "--cached-url",
                    cached.url,
                    "--states",
                    "2",
                    "--clauses",
                    "5",
                    "--repeats",
                    "1",
                    "--output",
                    str(output),
                ]
            )
    report = json.loads(output.read_text())
    assert code == 0
    assert report["passed"] is True
    assert report["flips"] == 0
    assert report["max_delta"] == pytest.approx(0.01, abs=1e-9)
    assert report["cache_counters"]["cached"]["hits"] > 0
    # Every measured round is compared; the warm-up round is not.
    assert len(report["rows"]) == 2 * 1 * len(QUESTIONS)
    # The uncached server against itself: this fake is deterministic.
    assert report["noise_floor"]["max_delta"] == 0.0
    # One salt per state, reused by every round so the state can be cached.
    assert sorted(set(cached.salts)) == ["kev-ab-state-0", "kev-ab-state-1"]


def test_end_to_end_fails_when_cached_answers_do_not_match(tmp_path):
    output = tmp_path / "report.json"
    with Server(drift=0.0, hits_each_request=0) as baseline:
        with Server(drift=0.30, hits_each_request=8) as cached:
            code = runner().main(
                [
                    "--baseline-url",
                    baseline.url,
                    "--cached-url",
                    cached.url,
                    "--states",
                    "1",
                    "--clauses",
                    "5",
                    "--repeats",
                    "1",
                    "--output",
                    str(output),
                ]
            )
    report = json.loads(output.read_text())
    assert code == 1
    assert report["passed"] is False
    assert report["max_delta"] > 0.05


def test_end_to_end_fails_when_the_cache_never_hits(tmp_path):
    output = tmp_path / "report.json"
    with Server(drift=0.0, hits_each_request=0) as baseline:
        with Server(drift=0.0, hits_each_request=0) as cached:
            code = runner().main(
                [
                    "--baseline-url",
                    baseline.url,
                    "--cached-url",
                    cached.url,
                    "--states",
                    "1",
                    "--clauses",
                    "5",
                    "--repeats",
                    "1",
                    "--output",
                    str(output),
                ]
            )
    report = json.loads(output.read_text())
    assert code == 1
    assert any("no vllm:prefix_cache_hits_total" in r for r in report["reasons"])


def test_every_measured_round_is_compared_and_warm_up_is_not(tmp_path):
    output = tmp_path / "report.json"
    with Server(drift=0.0, hits_each_request=0) as baseline:
        with Server(drift=0.01, hits_each_request=8) as cached:
            runner().main(
                [
                    "--baseline-url",
                    baseline.url,
                    "--cached-url",
                    cached.url,
                    "--states",
                    "2",
                    "--clauses",
                    "5",
                    "--repeats",
                    "3",
                    "--output",
                    str(output),
                ]
            )
            # Two states, each one warm-up plus three measured rounds.
            assert cached.requests == 2 * (1 + 3)
    report = json.loads(output.read_text())
    assert len(report["rows"]) == 2 * 3 * len(QUESTIONS)
    assert len(report["timing_s"]) == 2
