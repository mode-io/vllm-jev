"""Kev (kev_pointer_v1) prefix-cache reads through vLLM token embeddings.

vLLM 0.29 defaults ``skip_reading_prefix_cache=True`` for ``token_embed``
because a cache hit returns hidden states only for the tokens it computed.
Kev reads its pointer head after the shared state, so a hit that stops
before the first readout is safe. These tests use a fake engine with that
exact return shape; no GPU, vLLM, or weights are needed.
"""

import asyncio
import json
import string
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from vllm_jev import decision
from vllm_jev.decision_template import ENVIRONMENT, current_template

HIDDEN = 16
MARKERS = ["<|kev_state|>", "<|kev_q|>", "<|kev_opt|>", "<|kev_end|>", "<|kev_ask|>"]


class PoolingParams(SimpleNamespace):
    """vLLM 0.29 fields used here; None is resolved by the engine later."""

    def __init__(self, task, use_activation, skip_reading_prefix_cache=None):
        super().__init__(
            task=task,
            use_activation=use_activation,
            skip_reading_prefix_cache=skip_reading_prefix_cache,
        )


@pytest.fixture(autouse=True)
def vllm_stub(monkeypatch):
    """Stub vLLM only when it is not installed as a real package.

    Another test module may leave a bare ``vllm`` stub behind, so test for a
    real package (``__path__``) rather than for successful import.
    """
    import sys

    if hasattr(sys.modules.get("vllm"), "__path__"):
        yield
        return
    # Re-import endpoint against these stubs and drop it afterwards, so this
    # module neither depends on nor leaks into other tests' import state.
    monkeypatch.delitem(sys.modules, "vllm_jev.endpoint", raising=False)
    # Same stubs as test_systemone_cache.py: DecisionService.systemone
    # imports helpers from endpoint, which imports vLLM's API utilities.
    monkeypatch.setitem(
        sys.modules, "vllm", SimpleNamespace(PoolingParams=PoolingParams)
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.entrypoints.serve.utils.api_utils",
        SimpleNamespace(
            with_cancellation=lambda function: function,
            load_aware_call=lambda function: function,
        ),
    )
    yield
    # Drop the module imported against the stubs. Both the sys.modules entry and
    # the package attribute must go: another suite calls importorskip("vllm"),
    # which a leftover stub satisfies, and a leftover attribute would let a
    # later `from vllm_jev import endpoint` succeed where it should not.
    sys.modules.pop("vllm_jev.endpoint", None)
    import vllm_jev

    vllm_jev.__dict__.pop("endpoint", None)


@pytest.fixture(autouse=True)
def native_template(monkeypatch):
    monkeypatch.delenv(ENVIRONMENT, raising=False)
    monkeypatch.delenv("VLLM_JEV_KEV_PREFIX_CACHE", raising=False)
    current_template.cache_clear()
    yield
    current_template.cache_clear()


def write_checkpoint(path):
    from tokenizers import Regex, Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {t: i for i, t in enumerate([*MARKERS, "[UNK]", *string.printable])}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Split(Regex(r"[\s\S]"), "isolated")
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]")
    fast.add_special_tokens({"additional_special_tokens": MARKERS})
    fast.save_pretrained(path)
    (path / "decision_manifest.json").write_text(
        json.dumps(
            {
                "prompt_protocol": "kev_pointer_v1",
                "max_length": 4096,
                "max_options": 64,
                "hidden_size": HIDDEN,
                "temperature": 1.32,
                "markers": [fast.convert_tokens_to_ids(m) for m in MARKERS],
            }
        )
    )
    generator = torch.Generator().manual_seed(0)
    save_file(
        {
            "q.weight": torch.randn(8, HIDDEN, generator=generator),
            "q.bias": torch.randn(8, generator=generator),
            "k.weight": torch.randn(8, HIDDEN, generator=generator),
            "k.bias": torch.randn(8, generator=generator),
        },
        str(path / "decision_head.safetensors"),
    )


class CachingEngine:
    """vLLM 0.29 token_embed semantics: cached prefix tokens return no state."""

    def __init__(self, cached=0):
        self.cached = cached
        self.requests = []

    @staticmethod
    def states(ids):
        rows = []
        for position, token in enumerate(ids):
            generator = torch.Generator().manual_seed(token * 7919 + position)
            rows.append(torch.randn(HIDDEN, generator=generator))
        return torch.stack(rows)

    async def encode(self, prompt, pooling_params, request_id):
        self.requests.append((prompt, pooling_params))
        ids = list(prompt["prompt_token_ids"])
        # None mirrors vLLM: token_embed resolves to "skip" by default.
        skip = pooling_params.skip_reading_prefix_cache in (None, True)
        cached = 0 if skip else min(self.cached, len(ids) - 1)
        yield SimpleNamespace(
            finished=True,
            prompt_token_ids=ids,
            num_cached_tokens=cached,
            outputs=SimpleNamespace(data=self.states(ids)[cached:]),
        )

    async def abort(self, request_id):
        pass


PAYLOAD = {
    "state": "The supplier shall deliver within 30 days of the order.",
    "questions": {
        "party": {
            "type": "choice",
            "instructions": "Who must deliver?",
            "criteria": {"supplier": None, "buyer": None},
        },
        "binding": {"type": "noul", "instructions": "Is delivery mandatory?"},
    },
}


def run(service, payload=PAYLOAD):
    from vllm_jev.request_models import SystemOneRequest

    return asyncio.run(service.systemone(SystemOneRequest(**payload)))


def make_service(tmp_path, engine):
    write_checkpoint(tmp_path)
    return decision.DecisionService(engine, tmp_path, "kev-test", 4096)


def probabilities(response):
    return {k: v.get("probabilities", v.get("noul")) for k, v in response.items()}


def test_kev_requests_read_the_prefix_cache_by_default(tmp_path):
    engine = CachingEngine()
    run(make_service(tmp_path, engine))
    assert engine.requests
    assert all(
        params.skip_reading_prefix_cache is False for _, params in engine.requests
    )


def test_kev_prefix_cache_can_be_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_JEV_KEV_PREFIX_CACHE", "0")
    engine = CachingEngine(cached=10)
    response = run(make_service(tmp_path, engine))
    assert all(
        params.skip_reading_prefix_cache is True for _, params in engine.requests
    )
    assert response["metadata"]["cached_tokens"] == 0


@pytest.mark.parametrize("cached", [0, 8, 40])
def test_cached_prefix_before_readout_matches_uncached(tmp_path, cached):
    reference = run(make_service(tmp_path, CachingEngine(cached=0)))
    engine = CachingEngine(cached=cached)
    response = run(make_service(tmp_path, engine))
    for key, expected in probabilities(reference["answers"]).items():
        got = probabilities(response["answers"])[key]
        assert got == pytest.approx(expected, abs=1e-6)
    assert response["metadata"]["uncached_reruns"] == 0
    assert response["metadata"]["cached_tokens"] == cached * len(engine.requests)


def test_hit_covering_a_readout_reruns_without_the_cache(tmp_path):
    reference = run(make_service(tmp_path, CachingEngine(cached=0)))
    engine = CachingEngine(cached=10_000)
    response = run(make_service(tmp_path, engine))
    for key, expected in probabilities(reference["answers"]).items():
        got = probabilities(response["answers"])[key]
        assert got == pytest.approx(expected, abs=1e-6)
    questions = len(PAYLOAD["questions"])
    assert response["metadata"]["uncached_reruns"] == questions
    retries = [p for _, p in engine.requests if p.skip_reading_prefix_cache]
    assert len(retries) == questions


def test_trusted_salt_is_shared_across_requests(tmp_path):
    engine = CachingEngine()
    service = make_service(tmp_path, engine)
    run(service, {**PAYLOAD, "cache_salt": "tenant-a"})
    run(service, {**PAYLOAD, "cache_salt": "tenant-a"})
    assert {prompt["cache_salt"] for prompt, _ in engine.requests} == {"tenant-a"}


def test_missing_salt_keeps_requests_isolated(tmp_path):
    engine = CachingEngine()
    service = make_service(tmp_path, engine)
    run(service)
    first = {prompt["cache_salt"] for prompt, _ in engine.requests}
    engine.requests.clear()
    run(service)
    second = {prompt["cache_salt"] for prompt, _ in engine.requests}
    assert len(first) == len(second) == 1
    assert first != second


def test_endpoint_routes_trusted_salt_to_kev(tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from vllm_jev import endpoint

    engine = CachingEngine()
    app = FastAPI()
    app.state.vllm_decision_service = make_service(tmp_path, engine)
    endpoint.JevEndpointPlugin().attach_router(app)
    response = TestClient(app).post(
        "/v1/systemone", json={**PAYLOAD, "cache_salt": "tenant-a"}
    )
    assert response.status_code == 200, response.text
    assert {prompt["cache_salt"] for prompt, _ in engine.requests} == {"tenant-a"}


def test_endpoint_still_rejects_salt_for_non_kev_decisions(tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from vllm_jev import endpoint

    engine = CachingEngine()
    service = make_service(tmp_path, engine)
    service.compiler.protocol = "task_json_v1"
    app = FastAPI()
    app.state.vllm_decision_service = service
    endpoint.JevEndpointPlugin().attach_router(app)
    response = TestClient(app).post(
        "/v1/systemone", json={**PAYLOAD, "cache_salt": "tenant-a"}
    )
    assert response.status_code == 422
    assert not engine.requests


def test_legacy_pooling_api_keeps_full_state_path(tmp_path, monkeypatch):
    import sys

    class LegacyPoolingParams:
        def __init__(self, task, use_activation):
            self.skip_reading_prefix_cache = True

    monkeypatch.setattr(sys.modules["vllm"], "PoolingParams", LegacyPoolingParams)
    engine = CachingEngine(cached=40)
    service = make_service(tmp_path, engine)
    assert not service.supports_shared_prefix_cache
    response = run(service)
    assert response["metadata"]["cached_tokens"] == 0
    with pytest.raises(ValueError, match="cache_salt"):
        run(service, {**PAYLOAD, "cache_salt": "tenant-a"})


def test_ascend_does_not_enable_unvalidated_cache_reads(tmp_path, monkeypatch):
    from vllm_jev import backend

    monkeypatch.setattr(backend, "resolve_backend", lambda: "ascend")
    engine = CachingEngine(cached=40)
    service = make_service(tmp_path, engine)
    assert not service.supports_shared_prefix_cache
    response = run(service)
    assert response["metadata"]["cached_tokens"] == 0
    assert all(params.skip_reading_prefix_cache for _, params in engine.requests)


def test_disabled_cache_rejects_unexpected_partial_states(tmp_path, monkeypatch):
    class TruncatedEngine(CachingEngine):
        async def encode(self, prompt, pooling_params, request_id):
            ids = list(prompt["prompt_token_ids"])
            yield SimpleNamespace(
                finished=True,
                prompt_token_ids=ids,
                num_cached_tokens=1,
                outputs=SimpleNamespace(data=self.states(ids)[1:]),
            )

    monkeypatch.setenv("VLLM_JEV_KEV_PREFIX_CACHE", "0")
    with pytest.raises(RuntimeError, match="incomplete decision states"):
        run(make_service(tmp_path, TruncatedEngine()))


def test_protocol_gate_is_checked_when_reporting_support(tmp_path):
    service = make_service(tmp_path, CachingEngine())
    service.compiler.protocol = "task_json_v1"
    assert not service.supports_shared_prefix_cache
