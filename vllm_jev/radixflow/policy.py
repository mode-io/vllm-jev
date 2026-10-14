"""Bounded admission and per-round dependency accounting, without vLLM imports."""

import os
from collections import Counter
from dataclasses import dataclass

POLICIES = ("radix", "timer5", "timer10", "flex", "radixflow", "radixflow_frag")


@dataclass(frozen=True)
class Config:
    policy: str = "radixflow"
    wait_ms: float = 5.0
    max_deferrals: int = 64
    max_namespaces: int = 256
    max_tokens: int = 262144
    trace_dir: str | None = None
    library: str | None = None

    def __post_init__(self):
        if self.policy not in POLICIES:
            raise ValueError("unknown RadixFlow policy")
        if not 0 <= self.wait_ms <= 100:
            raise ValueError("wait_ms must be within [0, 100]")
        if min(self.max_deferrals, self.max_namespaces, self.max_tokens) < 1:
            raise ValueError("RadixFlow limits must be positive")

    @classmethod
    def environment(cls):
        return cls(
            policy=os.getenv("VLLM_JEV_RADIXFLOW_POLICY", "radixflow"),
            wait_ms=float(os.getenv("VLLM_JEV_RADIXFLOW_WAIT_MS", "5")),
            library=os.getenv("VLLM_JEV_RADIXFLOW_LIBRARY"),
            trace_dir=os.getenv("VLLM_JEV_RADIXFLOW_TRACE_DIR"),
        )

    @property
    def traversal(self):
        return self.policy in ("radixflow", "radixflow_frag")


class Admission:
    def __init__(self, config):
        self.config = config
        self.arrivals = {}
        self.released = set()
        self.counts = Counter()

    def add(self, rid, now):
        self.arrivals[rid] = now

    def remove(self, rid):
        self.arrivals.pop(rid, None)
        self.released.discard(rid)

    def snapshot(self, waiting, running, rows, budget, residual, now):
        # One eligibility snapshot: release cannot change halfway through a scan.
        peers = [r for r in waiting if r.request_id not in self.released]
        if not peers or self.config.policy == "radix":
            return set()
        delay = (
            10
            if self.config.policy == "timer10"
            else 5
            if self.config.policy == "timer5"
            else self.config.wait_ms
        ) / 1000
        oldest = min(self.arrivals.get(r.request_id, now) for r in peers)
        reason = "deadline" if now >= oldest + delay else None
        if len(peers) >= rows:
            reason = "rows"
        if self.config.policy in ("flex", "radixflow", "radixflow_frag"):
            if running:
                reason = "running"
            elif sum(residual(r) for r in peers) >= budget:
                reason = "tokens"
        if reason:
            self.released.update(r.request_id for r in peers)
            self.counts[reason] += 1
            return set()
        return {r.request_id for r in peers}


class RoundState:
    def __init__(self, config):
        self.config = config
        self.deferrals = Counter()
        self.totals = Counter()
        self.seq = 0
        self.begin()

    def begin(self):
        self.seq += 1
        self.producers = {}
        self.guards = {}
        self.events = []
        self.cut = None
        self.fragment_blocked = set()

    def reject(self, rid, selected, site):
        components = self.guards.pop(rid, [])
        known = bool(components) and all(c["producer"] in selected for c in components)
        pending = bool(components)
        kind = (
            "PENDING"
            if pending
            else ("FRAGMENT" if rid in self.fragment_blocked else "OTHER")
        )
        defer = (
            self.config.traversal
            and ((pending and known) or kind == "FRAGMENT")
            and self.deferrals[rid] < self.config.max_deferrals
        )
        if pending and self.cut is None:
            self.cut = dict(selected)
        if defer:
            self.deferrals[rid] += 1
        self.events.append(
            dict(
                consumer=rid,
                kind=kind,
                site=site,
                action="DEFER" if defer else "STOP",
                components=components,
                cut=dict(selected),
            )
        )
        self.totals[kind.lower() + ("_defers" if defer else "_stops")] += 1
        return defer

    def finish(self, selected):
        cut = self.cut or {}
        post = [rid for rid in selected if self.cut is not None and rid not in cut]
        pending = [e for e in self.events if e["kind"] == "PENDING"]
        preserved = all(rid in selected for rid in cut)
        bound = bool(pending) and all(
            c["producer"] in e["cut"] and c["producer"] in selected
            for e in pending
            for c in e["components"]
        )
        # A prior fragmentation skip is not credited to PENDING recovery.
        valid = bound and preserved and self.events[0]["kind"] == "PENDING"
        recovered = post if valid else []
        self.totals["post_pending_rows"] += len(post)
        self.totals["within_round_recovered_rows"] += len(recovered)
        return dict(
            seq=self.seq,
            events=self.events,
            selected=dict(selected),
            post_pending_selected=post,
            within_round_recovered=recovered,
            producer_cut_verified=bound,
            cut_preserved=preserved,
        )
