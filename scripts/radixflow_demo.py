"""One real HTTP arm; repeat on a fresh matched service for the comparison."""

import argparse
import concurrent.futures
import hashlib
import json
import statistics
import time
import urllib.request
import uuid
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--url", required=True, help="Private vLLM service URL")
p.add_argument("--model", required=True)
p.add_argument("--label", required=True)
p.add_argument(
    "--inputs",
    type=Path,
    required=True,
    help='JSONL rows {"id":..., "prompts":[text or token IDs,...]}',
)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--output-tokens", type=int, default=1)
p.add_argument("--concurrency", type=int, default=4)
p.add_argument("--traces", type=Path, help="Fresh server trace directory for this run")
a = p.parse_args()
if min(a.output_tokens, a.concurrency) < 1:
    p.error("token count and concurrency must be positive")
inputs = [json.loads(x) for x in a.inputs.read_text().split("\n") if x.strip()]
if not inputs or len({r["id"] for r in inputs}) != len(inputs):
    p.error("nonempty unique input IDs required")
namespace = uuid.uuid4().hex


def ask(task):
    logical_id, rid, prompt = task
    data = dict(
        model=a.model,
        prompt=prompt,
        temperature=0,
        max_tokens=a.output_tokens,
        ignore_eos=True,
        cache_salt=namespace,
    )
    req = urllib.request.Request(
        a.url.rstrip("/") + "/v1/completions",
        json.dumps(data).encode(),
        {"Content-Type": "application/json", "X-Request-Id": rid},
    )
    begin = time.perf_counter()
    with urllib.request.urlopen(req, timeout=180) as response:
        result = json.load(response)
    elapsed = time.perf_counter() - begin
    usage = result["usage"]
    if usage["completion_tokens"] != a.output_tokens:
        raise ValueError("unexpected generation length")
    if "cached_tokens" not in (usage.get("prompt_tokens_details") or {}):
        raise ValueError("native cache usage missing; cannot report reuse")
    return dict(
        id=rid,
        logical_id=logical_id,
        begin=begin,
        done=begin + elapsed,
        seconds=elapsed,
        usage=usage,
        text=result["choices"][0]["text"],
    )


tasks = [
    (r["id"], f"radixflowdemo__{namespace}__{i}__{j}", prompt)
    for i, r in enumerate(inputs)
    for j, prompt in enumerate(r["prompts"])
]
a.output.mkdir(parents=True, exist_ok=False)
started = time.perf_counter()
rows = []
try:
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.concurrency) as pool:
        with (a.output / "responses.jsonl").open("w") as f:
            for value in pool.map(ask, tasks):
                rows.append(value)
                f.write(json.dumps(value) + "\n")
                f.flush()
except BaseException as error:
    (a.output / "FAILED.json").write_text(
        json.dumps({"error": repr(error), "completed": len(rows)})
    )
    raise
wall = time.perf_counter() - started
latencies = sorted(r["seconds"] for r in rows)
prompt = sum(r["usage"]["prompt_tokens"] for r in rows)
cache = sum(r["usage"]["prompt_tokens_details"]["cached_tokens"] for r in rows)
logical_times = [
    max(x["done"] for x in rows if x["logical_id"] == r["id"])
    - min(x["begin"] for x in rows if x["logical_id"] == r["id"])
    for r in inputs
]
summary = dict(
    model=a.model,
    concurrency=a.concurrency,
    mean_logical_s=statistics.mean(logical_times),
    label=a.label,
    input_sha256=hashlib.sha256(a.inputs.read_bytes()).hexdigest(),
    logical_requests=len(inputs),
    branches=len(rows),
    wall_s=wall,
    branch_req_s=len(rows) / wall,
    mean_branch_s=statistics.mean(latencies),
    p95_branch_s=latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))],
    native_cache_tokens=cache,
    prompt_tokens=prompt,
    reuse_pct=100 * cache / prompt,
    output_tokens=a.output_tokens,
    namespace=namespace,
    scope="Closed-loop demo; HTTP branch latency, not saturation throughput",
)
if a.traces:
    events = [
        json.loads(line)
        for path in a.traces.glob("*.jsonl")
        for line in path.read_text().split("\n")
        if line
    ]
    calls = [
        x
        for x in events
        if x["kind"] == "execute" and any(namespace in r["id"] for r in x["rows"])
    ]
    if not calls or any(
        any(namespace not in r["id"] for r in x["rows"]) for x in calls
    ):
        raise ValueError(
            "no attributable worker calls, or mixed unrelated service traffic"
        )
    nonempty = [x for x in calls if x["nonempty"]]
    summary["global_worker_steps"] = len(nonempty)
    completions = {
        (x["pid"], x["seq"]): x for x in events if x["kind"] == "execute_complete"
    }
    selected_completions = [completions[(x["pid"], x["seq"])] for x in nonempty]
    summary["mean_host_step_ms"] = (
        sum(x["host_completion_ns"] for x in selected_completions) / len(nonempty) / 1e6
    )
    summary["mean_host_enqueue_step_ms"] = (
        sum(x["host_ns"] for x in nonempty) / len(nonempty) / 1e6
    )
    core_steps = {(x["pid"], x["seq"]): x for x in events if x["kind"] == "core_step"}
    summary["mean_core_step_ms"] = (
        sum(core_steps[(x["pid"], x["seq"])]["host_ns"] for x in nonempty)
        / len(nonempty)
        / 1e6
    )
    summary["service_empty_calls"] = sum(
        not x["nonempty"] for x in events if x["kind"] == "execute"
    )
    summary["worker_time_scope"] = (
        "Host executor intervals; empty calls cover the entire trace directory"
    )
(a.output / "SUMMARY.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
