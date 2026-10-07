"""Native cache ownership and optional live System One checks on vLLM 0.29.0.

Set VLLM_JEV_TEST_URL to an idle Open-Jev-2B/9B server with prefix caching.
For numerical parity, set VLLM_JEV_TEST_UNCACHED_URL to a second server using
the same checkpoint and settings, with --no-enable-prefix-caching. Run with
``python -m pytest tests/test_systemone_cache_integration.py -q``.
"""

import os
import uuid

import httpx
import pytest


def test_native_cache_namespaces_and_shared_block_lifetime():
    vllm = pytest.importorskip("vllm")
    assert vllm.__version__ == "0.29.0", "Run against the pinned vLLM dependency"

    import torch
    from vllm import PoolingParams
    from vllm.utils.hashing import sha256
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheConfig,
        KVCacheGroupSpec,
    )
    from vllm.v1.request import Request, RequestStatus

    init_none_hash(sha256)
    block_size = 16
    manager = KVCacheManager(
        KVCacheConfig(
            num_blocks=32,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec(
                    ["layer"],
                    FullAttentionSpec(
                        block_size=block_size,
                        num_kv_heads=1,
                        head_size=1,
                        dtype=torch.float32,
                    ),
                )
            ],
        ),
        max_model_len=256,
        scheduler_block_size=block_size,
        hash_block_size=block_size,
        enable_caching=True,
    )

    def request(salt, ids):
        return Request(
            request_id=uuid.uuid4().hex,
            prompt_token_ids=ids,
            sampling_params=None,
            pooling_params=PoolingParams(task="classify", use_activation=False),
            cache_salt=salt,
            block_hasher=get_request_block_hasher(block_size, sha256),
        )

    ids = list(range(59))
    first = request("caller-a", ids)
    assert manager.allocate_slots(first, len(ids)) is not None
    first.num_computed_tokens = len(ids)
    first.status = RequestStatus.RUNNING

    second = request("caller-a", ids)
    shared, cached, _ = manager.get_computed_blocks(second)
    assert cached == 48
    assert manager.get_computed_blocks(request("caller-b", ids))[1] == 0
    changed = request("caller-a", ids[:20] + [99] + ids[21:])
    assert manager.get_computed_blocks(changed)[1] == 16

    assert manager.allocate_slots(second, len(ids) - cached, cached, shared) is not None
    assert all(block.ref_cnt == 2 for block in shared.blocks[0])
    hashes = [block.block_hash for block in shared.blocks[0]]
    first.status = RequestStatus.FINISHED_ABORTED
    manager.free(first)
    assert all(block.ref_cnt == 1 for block in shared.blocks[0])
    assert [block.block_hash for block in shared.blocks[0]] == hashes
    assert manager.get_computed_blocks(request("caller-a", ids))[1] == cached
    second.num_computed_tokens = len(ids)
    manager.free(second)
    assert all(block.ref_cnt == 0 for block in shared.blocks[0])


@pytest.fixture(scope="module")
def server():
    url = os.environ.get("VLLM_JEV_TEST_URL")
    if not url:
        pytest.skip(
            "Set VLLM_JEV_TEST_URL to an Open-Jev-2B/9B server with prefix caching"
        )
    with httpx.Client(base_url=url, timeout=120) as client:
        version = client.get("/version")
        version.raise_for_status()
        assert version.json()["version"] == "0.29.0"
        yield client


def query(server, state, salt=None):
    response = server.post(
        "/v1/systemone",
        json={
            "state": state,
            "cache_salt": salt,
            "questions": {"urgent": {"type": "noul", "instructions": "Is it urgent?"}},
        },
    )
    response.raise_for_status()
    result = response.json()
    # One sequence avoids counting reuse between branches of the same request.
    assert result["metadata"]["candidate_sequences"] == 1
    return result


def test_live_namespace_isolation_and_partial_prefix(server):
    salt = uuid.uuid4().hex
    prefix = "A customer reports a duplicate payment. " * 40
    state = prefix + "The payment was made yesterday. " * 40
    cold = query(server, state, salt)
    warm = query(server, state, salt)
    assert cold["metadata"]["cached_tokens"] == 0
    assert warm["metadata"]["cached_tokens"] > 0
    assert query(server, state, uuid.uuid4().hex)["metadata"]["cached_tokens"] == 0
    assert query(server, state)["metadata"]["cached_tokens"] == 0
    assert query(server, state)["metadata"]["cached_tokens"] == 0
    changed = query(
        server, prefix + "A different customer cancelled the order. " * 40, salt
    )
    # Hybrid models may have no retained Mamba state at the shared boundary.
    assert 0 <= changed["metadata"]["cached_tokens"] < warm["metadata"]["cached_tokens"]


def test_live_cached_and_uncached_decisions_agree(server):
    url = os.environ.get("VLLM_JEV_TEST_UNCACHED_URL")
    if not url:
        pytest.skip(
            "Set VLLM_JEV_TEST_UNCACHED_URL to the same model with caching disabled"
        )
    salt = uuid.uuid4().hex
    state = "A customer reports a duplicate payment and requests a refund. " * 80
    query(server, state, salt)
    warm = query(server, state, salt)
    with httpx.Client(base_url=url, timeout=120) as uncached_server:
        version = uncached_server.get("/version")
        version.raise_for_status()
        assert version.json()["version"] == "0.29.0"
        uncached = query(uncached_server, state, salt)
    assert warm["model"] == uncached["model"]
    assert warm["metadata"]["cached_tokens"] > 0
    assert uncached["metadata"]["cached_tokens"] == 0
    assert warm["answers"]["urgent"]["noul"] == pytest.approx(
        uncached["answers"]["urgent"]["noul"], abs=1e-3, rel=1e-3
    )
