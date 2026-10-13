"""Display two verified demo summaries; never synthesize missing worker data."""

import argparse
import html
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("baseline", type=Path)
p.add_argument("radixflow", type=Path)
p.add_argument("--html", type=Path)
a = p.parse_args()
left, right = [json.loads(path.read_text()) for path in [a.baseline, a.radixflow]]
for key in [
    "model",
    "input_sha256",
    "output_tokens",
    "concurrency",
    "branches",
    "logical_requests",
]:
    if left[key] != right[key]:
        p.error(f"unmatched comparison: {key}")
metrics = [
    ("Wall time (s)", "wall_s", ".3f"),
    ("Mean logical latency (s)", "mean_logical_s", ".3f"),
    ("Branch throughput (req/s)", "branch_req_s", ".3f"),
    ("Global worker steps", "global_worker_steps", ".0f"),
    ("Mean host interval / step (ms)", "mean_host_step_ms", ".3f"),
    ("Mean complete core step (ms)", "mean_core_step_ms", ".3f"),
    ("Native token reuse (%)", "reuse_pct", ".3f"),
]
rows = [
    [
        name,
        format(left[key], fmt) if key in left else "unmeasured",
        format(right[key], fmt) if key in right else "unmeasured",
    ]
    for name, key, fmt in metrics
]
print("| Metric | Baseline | RadixFlow |\n| --- | ---: | ---: |")
for row in rows:
    print("| " + " | ".join(row) + " |")
if a.html:
    body = "".join(
        "<tr>" + "".join("<td>" + html.escape(x) + "</td>" for x in row) + "</tr>"
        for row in rows
    )
    a.html.write_text(
        """<!doctype html><meta charset="utf-8"><title>RadixFlow comparison</title>
<style>body{font:16px system-ui;color:#202630;background:#f4f6f8;max-width:850px;margin:60px auto}
main{background:white;padding:32px;border-radius:12px}h1{color:#1f63ad}table{border-collapse:collapse;width:100%}
th,td{padding:14px;border-bottom:1px solid #dfe5eb;text-align:right}th:first-child,td:first-child{text-align:left}
th:last-child,td:last-child{background:#eef4fb;color:#1f63ad}small{display:block;margin-top:22px;color:#596370}</style>
<main><h1>RadixFlow</h1><p>Matched shared-prefix serving comparison</p><table>
<tr><th>Metric</th><th>Baseline</th><th>RadixFlow</th></tr>"""
        + body
        + """</table><small>Host intervals include CPU-side execution and waiting. This is a closed-loop demo, not a saturation benchmark.</small></main>"""
    )
