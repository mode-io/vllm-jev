"""Uncached throughput script: every request carries a fresh state.

No GPU: reuses the backend-comparison fakes, one answering like vllm-jev and
one like the native Kev server, and checks that no request is ever a repeat.
"""

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    sys.path.insert(0, str(ROOT / "scripts"))
    sys.path.insert(0, str(ROOT / "tests"))
    path = ROOT / f"scripts/{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fakes():
    path = ROOT / "tests/test_kev_backend_compare.py"
    spec = importlib.util.spec_from_file_location("fakes", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_no_request_repeats_a_state(tmp_path):
    f = fakes()
    output = tmp_path / "nocache.json"
    with f.Server(native=False) as vllm, f.Server(native=True) as native:
        code = load("kev_nocache_compare").main(
            [
                "--vllm-url",
                vllm.url,
                "--native-url",
                native.url,
                "--lengths",
                "800",
                "--concurrency",
                "1,3",
                "--rounds",
                "2",
                "--output",
                str(output),
            ]
        )
        vllm_salts = list(vllm.salted)
    report = json.loads(output.read_text())
    assert code == 0
    assert [r["clients"] for r in report["levels"]] == [1, 3]
    for row in report["levels"]:
        assert row["vllm"]["requests"] == row["clients"] * 2
        # The fakes only hit on an exact repeat, so zero means no repeat.
        assert row["vllm"]["cached_tokens_mean"] == 0
        assert row["native"]["cache_hits_delta"] == 0
        assert row["native"]["cache_misses_delta"] == row["native"]["requests"]
        assert row["ratio"] > 0
    # Every vllm request carried its own salt.
    measured = [s for s in vllm_salts if s.startswith("kev-nocache-")]
    assert len(measured) == len(set(measured)) == 8
    assert "无缓存吞吐对比" in output.with_suffix(".md").read_text()
