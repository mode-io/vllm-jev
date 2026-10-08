"""Optional online hybrid-prefix checkpoints for native vLLM 0.29.

The generic tree records exact token paths. A path becomes reusable only after
the native Full Attention and every Mamba/GDN group publish matching state.
This adapter is off by default while the model-specific path is evaluated.
"""

import os
from dataclasses import dataclass, field

from .prefix_components import ComponentEviction
from .prefix_tree import BoundedPrefixTrees


@dataclass
class _Runtime:
    pool: object | None = None
    policy: ComponentEviction | None = None
    trees: BoundedPrefixTrees | None = None
    counts: dict[str, int] = field(default_factory=dict)
    producers: dict[str, str] = field(default_factory=dict)
    hot: dict[str, tuple[int, str, tuple[int, ...]]] = field(default_factory=dict)
    pending: dict[str, dict[int, tuple[str, object]]] = field(default_factory=dict)
    configured: bool = False


_runtime = _Runtime()
_installed = False


def install() -> None:
    """Patch the pinned engine only when explicitly enabled before serving."""
    global _installed
    if _installed:
        return
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_utils import get_block_hash
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.core.single_type_kv_cache_manager import (
        FullAttentionManager,
        MambaManager,
    )
    from vllm.v1.request import RequestStatus

    original_init = Scheduler.__init__
    original_add = Scheduler.add_request
    original_split = Scheduler._mamba_block_aligned_split
    original_free = Scheduler._free_request
    original_full = FullAttentionManager._cache_partial_tail_block
    original_mamba = MambaManager._cache_partial_tail_block
    original_evict = BlockPool._maybe_evict_cached_block
    original_move = BlockPool.move_block_hashes
    original_reset = BlockPool.reset_prefix_cache
    max_tenants = int(os.getenv("VLLM_JEV_TREE_TENANTS", "8"))
    max_paths = int(os.getenv("VLLM_JEV_TREE_PATHS", "64"))
    max_checkpoints = int(os.getenv("VLLM_JEV_TREE_CHECKPOINTS", "4"))
    promote_after = int(os.getenv("VLLM_JEV_TREE_PROMOTE_AFTER", "16"))
    salt_prefix = os.getenv("VLLM_JEV_TREE_SALT_PREFIX", "")
    if min(max_tenants, max_paths, promote_after) < 1 or max_checkpoints < 0:
        raise ValueError("invalid online prefix-cache limits")

    def initialize(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if self.vllm_config.model_config.runner_type == "pooling":
            # Pooling emits no sampled token. Reserving a decode slot makes a
            # cold hybrid prefill of exactly max_model_len stall at its last
            # aligned block when partial-checkpoint reads are bypassed.
            self.num_sampled_tokens_per_step = 0

    def remove_node(key):
        salt, prefix = key
        assert _runtime.trees is not None
        assert _runtime.trees.detach(salt, prefix)

    def evict_blocks(ids):
        if ids:
            assert _runtime.pool is not None
            _runtime.pool.evict_blocks(set(ids))

    def close_tenant(salt):
        assert _runtime.policy is not None
        if salt in _runtime.producers:
            return False
        if not _runtime.policy.close_tenant(salt):
            return False
        _runtime.counts.pop(salt, None)
        return True

    def refresh_locks():
        if _runtime.policy is None or _runtime.pool is None:
            return
        for node in _runtime.policy.nodes.values():
            node.full_locked = any(
                _runtime.pool.blocks[i].ref_cnt > 0 for i in node.full_ids
            )
            node.mamba_locked = any(
                _runtime.pool.blocks[i].ref_cnt > 0 for i in node.mamba_ids
            )

    def ready(self):
        if _runtime.configured:
            return _runtime.pool is self.kv_cache_manager.block_pool
        managers = self.kv_cache_manager.coordinator.single_type_managers
        if not (
            any(isinstance(m, FullAttentionManager) for m in managers)
            and any(isinstance(m, MambaManager) for m in managers)
            and all(
                self.kv_cache_manager.block_pool.hash_block_size < m.block_size
                for m in managers
            )
        ):
            return False
        _runtime.pool = self.kv_cache_manager.block_pool
        _runtime.policy = ComponentEviction(
            max_checkpoints, max_checkpoints, evict_blocks, remove_node
        )
        _runtime.trees = BoundedPrefixTrees(max_tenants, max_paths, close_tenant)
        _runtime.configured = True
        return True

    def cow_headroom(self, prompt_len):
        for manager in self.kv_cache_manager.coordinator.single_type_managers:
            block_size = manager.block_size
            row_capacity = (self.max_model_len + block_size - 1) // block_size
            prompt_pages = (prompt_len + block_size - 1) // block_size
            if prompt_pages + 1 > row_capacity:
                return False
        return True

    def add_request(self, request):
        salt = request.cache_salt
        if (
            isinstance(salt, str)
            and salt.startswith(salt_prefix)
            and not request.skip_reading_prefix_cache
            and not request.resumable
            and ready(self)
        ):
            assert _runtime.trees is not None
            assert _runtime.policy is not None
            ids = tuple(request.prompt_token_ids or ())
            if not cow_headroom(self, len(ids)):
                # A long consumer can hit a checkpoint from a shorter producer.
                # Bypass reads too when its partial-page CoW cannot fit.
                request.skip_reading_prefix_cache = True
                return original_add(self, request)
            refresh_locks()
            matched = _runtime.trees.observe(salt, ids)
            if matched is not None:
                seen = _runtime.counts.get(salt, 0)
                _runtime.counts[salt] = seen + 1
                resident = any(
                    key[0] == salt and node.mamba_ids
                    for key, node in _runtime.policy.nodes.items()
                )
                if (
                    max_checkpoints > 0
                    and seen >= promote_after and seen % promote_after == 0
                    and not resident and salt not in _runtime.producers
                ):
                    unit = _runtime.pool.hash_block_size
                    boundary = matched // unit * unit
                    if (
                        boundary > unit
                        and boundary < len(ids)
                        and any(
                            boundary % manager.block_size
                            for manager in
                            self.kv_cache_manager.coordinator.single_type_managers
                        )
                    ):
                        _runtime.hot[request.request_id] = (
                            boundary, salt, ids[:boundary]
                        )
                        _runtime.producers[salt] = request.request_id
                group_ids = list(range(self.kv_cache_manager.num_kv_cache_groups))
                for key, node in list(_runtime.policy.nodes.items()):
                    if key[0] != salt or ids[: len(key[1])] != key[1]:
                        continue
                    blocks = (
                        _runtime.pool.get_cached_block(node.block_hash, group_ids)
                        if node.block_hash is not None
                        else None
                    )
                    if blocks is None:
                        _runtime.policy.native_evicted(
                            (node.full_ids + node.mamba_ids)[0]
                        )
                    else:
                        _runtime.policy.touch(key)
        return original_add(self, request)

    def split(self, request, num_new_tokens,
              num_new_local_computed_tokens=0,
              num_external_computed_tokens=0):
        ordinary = original_split(
            self, request, num_new_tokens,
            num_new_local_computed_tokens, num_external_computed_tokens,
        )
        hot = _runtime.hot.get(request.request_id)
        if hot is None:
            return ordinary
        boundary = hot[0]
        start = (request.num_computed_tokens + num_new_local_computed_tokens
                 + num_external_computed_tokens)
        if (
            start < boundary <= start + num_new_tokens
            and boundary < request.num_prompt_tokens
        ):
            return boundary - start
        return ordinary

    def register(request_id, group_id, kind, block_hash):
        if block_hash is not None:
            _runtime.pending.setdefault(request_id, {})[group_id] = (
                kind, block_hash
            )

    def partial(manager, request, num_tokens, kind, ordinary):
        hot = _runtime.hot.get(request.request_id)
        if hot is None or hot[0] != num_tokens:
            return ordinary(manager, request, num_tokens)
        unit = manager.block_pool.hash_block_size
        if num_tokens % unit or num_tokens % manager.block_size == 0:
            return ordinary(manager, request, num_tokens)
        index = num_tokens // manager.block_size
        blocks = manager.req_to_blocks[request.request_id]
        if index >= len(blocks) or blocks[index].is_null:
            return None
        source = blocks[index]
        result = manager.block_pool.cache_partial_block(
            request=request, block=source, num_tokens=num_tokens,
            kv_cache_group_id=manager.kv_cache_group_id,
            block_size=manager.block_size,
        )
        if result is not None:
            manager._partial_hit_reqs[request.request_id] = (index, source)
            manager.num_cached_block[request.request_id] = index
            if kind == "mamba":
                manager._producer_partial_tail_reqs[request.request_id] = num_tokens
            register(request.request_id, manager.kv_cache_group_id, kind, result)
        return result

    def discard_pending(pending):
        ids = set()
        for _, cache_key in pending.values():
            block = _runtime.pool.cached_block_hash_to_block.get_one_block(cache_key)
            if block is not None:
                ids.add(block.block_id)
        evict_blocks(ids)

    def free_request(self, request, delay_free_blocks=False):
        result = original_free(self, request, delay_free_blocks=delay_free_blocks)
        request_id = request.request_id
        pending = _runtime.pending.pop(request_id, {})
        hot = _runtime.hot.pop(request_id, None)
        if hot and _runtime.producers.get(hot[1]) == request_id:
            del _runtime.producers[hot[1]]
        completed = (
            request.status in (
                RequestStatus.FINISHED_STOPPED, RequestStatus.FINISHED_LENGTH_CAPPED
            )
            and not delay_free_blocks
            and request.last_sched_seq <= self.processed_step_seq
        )
        handled = False
        if (
            completed and hot
            and len(pending) == self.kv_cache_manager.num_kv_cache_groups
        ):
            hashes = {get_block_hash(item[1]) for item in pending.values()}
            if len(hashes) == 1:
                block_hash = next(iter(hashes))
                group_ids = list(range(self.kv_cache_manager.num_kv_cache_groups))
                blocks = _runtime.pool.get_cached_block(block_hash, group_ids)
                if blocks is not None:
                    full = [
                        blocks[i].block_id for i in group_ids
                        if pending[i][0] == "full"
                    ]
                    mamba = [
                        blocks[i].block_id for i in group_ids
                        if pending[i][0] == "mamba"
                    ]
                    key = (hot[1], hot[2])
                    if _runtime.trees.attach(*key):
                        refresh_locks()
                        _runtime.policy.insert(
                            key, full, mamba,
                            full_locked=any(
                                _runtime.pool.blocks[i].ref_cnt for i in full
                            ),
                            mamba_locked=any(
                                _runtime.pool.blocks[i].ref_cnt for i in mamba
                            ),
                        )
                        handled = True
                        if key in _runtime.policy.nodes:
                            _runtime.policy.nodes[key].block_hash = block_hash
        if pending and not handled:
            discard_pending(pending)
        if _runtime.pool is self.kv_cache_manager.block_pool and _runtime.policy:
            refresh_locks()
            _runtime.policy.enforce_caps()
        return result

    def native_evict(self, block):
        removed = original_evict(self, block)
        if removed and self is _runtime.pool and _runtime.policy is not None:
            _runtime.policy.native_evicted(block.block_id)
        return removed

    def native_move(self, source, destination):
        result = original_move(self, source, destination)
        if self is _runtime.pool and _runtime.policy is not None:
            _runtime.policy.moved(source.block_id, destination.block_id)
        return result

    def native_reset(self):
        result = original_reset(self)
        if result and self is _runtime.pool and _runtime.policy is not None:
            _runtime.policy.reset()
        return result

    Scheduler.__init__ = initialize
    Scheduler.add_request = add_request
    Scheduler._mamba_block_aligned_split = split
    Scheduler._free_request = free_request
    FullAttentionManager._cache_partial_tail_block = (
        lambda self, request, num_tokens: partial(
            self, request, num_tokens, "full", original_full
        )
    )
    MambaManager._cache_partial_tail_block = (
        lambda self, request, num_tokens: partial(
            self, request, num_tokens, "mamba", original_mamba
        )
    )
    BlockPool._maybe_evict_cached_block = native_evict
    BlockPool.move_block_hashes = native_move
    BlockPool.reset_prefix_cache = native_reset
    _installed = True
