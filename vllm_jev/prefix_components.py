"""Component-aware eviction for optional hybrid prefix checkpoints.

This follows SGLang UnifiedRadixCache's documented component ordering:
Mamba state may be tombstoned on an internal node; a leaf eviction drops all
components; Full leaf eviction cascades to Mamba. It manages only the extra
hot checkpoints created by the optional adapter, not vLLM's ordinary cache.
"""

from collections import OrderedDict
from dataclasses import dataclass


@dataclass
class Node:
    key: tuple
    full_ids: tuple[int, ...]
    mamba_ids: tuple[int, ...]
    full_locked: bool = False
    mamba_locked: bool = False
    block_hash: object | None = None


class ComponentEviction:
    def __init__(self, full_cap, mamba_cap, evict_blocks, on_remove):
        if full_cap < 0 or mamba_cap < 0:
            raise ValueError("negative component cap")
        self.full_cap = full_cap
        self.mamba_cap = mamba_cap
        self.evict_blocks = evict_blocks
        self.on_remove = on_remove
        self.nodes: dict[tuple, Node] = {}
        self.full_lru = OrderedDict()
        self.mamba_lru = OrderedDict()

    def _children(self, key):
        salt, path = key
        return [other for other in self.nodes
                if other != key and other[0] == salt and other[1][:len(path)] == path]

    def _is_leaf(self, key):
        return not self._children(key)

    def _drop(self, key, *, physical):
        node = self.nodes.pop(key, None)
        if node is None:
            return False
        self.full_lru.pop(key, None)
        self.mamba_lru.pop(key, None)
        self.on_remove(key)
        if physical:
            self.evict_blocks(set(node.full_ids + node.mamba_ids))
        return True

    def _tombstone_mamba(self, key, *, physical):
        node = self.nodes.get(key)
        if node is None or not node.mamba_ids:
            return False
        if self._is_leaf(key):
            full_ids = node.full_ids
            dropped = self._drop(key, physical=physical)
            if dropped and not physical:
                self.evict_blocks(set(full_ids))
            return dropped
        ids = node.mamba_ids
        node.mamba_ids = ()
        self.mamba_lru.pop(key, None)
        if physical:
            self.evict_blocks(set(ids))
        return True

    def touch(self, key):
        node = self.nodes.get(key)
        if node is None:
            return False
        if node.full_ids:
            self.full_lru.move_to_end(key)
        if node.mamba_ids:
            self.mamba_lru.move_to_end(key)
        return bool(node.full_ids and node.mamba_ids)

    def insert(self, key, full_ids, mamba_ids, *, full_locked=False,
               mamba_locked=False):
        node = Node(key, tuple(full_ids), tuple(mamba_ids),
                    full_locked=full_locked, mamba_locked=mamba_locked)
        if not node.full_ids or not node.mamba_ids:
            raise ValueError("hybrid checkpoint must include both components")
        old = self.nodes.get(key)
        old_ids = set(old.full_ids + old.mamba_ids) if old else set()
        new_ids = set(node.full_ids + node.mamba_ids)
        if old and (old.full_locked or old.mamba_locked):
            # The caller attached one reference for this candidate. Keep the
            # in-use resident node, release only the rejected candidate's ref.
            self.on_remove(key)
            self.evict_blocks(new_ids - old_ids)
            return False
        if old is not None:
            self._drop(key, physical=False)
        self.nodes[key] = node
        self.full_lru[key] = None
        self.mamba_lru[key] = None
        if old_ids - new_ids:
            self.evict_blocks(old_ids - new_ids)
        return self.enforce_caps()

    def enforce_caps(self):
        """Mamba LRU first, then Full leaf LRU; skip in-use nodes."""
        while len(self.mamba_lru) > self.mamba_cap:
            victim = next((key for key in self.mamba_lru
                           if not self.nodes[key].mamba_locked
                           and (not self._is_leaf(key)
                                or not self.nodes[key].full_locked)), None)
            if victim is None:
                return False
            self._tombstone_mamba(victim, physical=True)
        while len(self.full_lru) > self.full_cap:
            victim = next((key for key in self.full_lru
                           if self._is_leaf(key) and
                           not self.nodes[key].full_locked and
                           not self.nodes[key].mamba_locked), None)
            if victim is None:
                return False
            self._drop(victim, physical=True)
        return True

    def native_evicted(self, block_id):
        """Invalidate dependent hashes after one physical cache slot is lost.

        Detach all affected metadata before calling the native eviction API,
        which can synchronously call this method again for the remaining IDs.
        Native eviction clears cache hashes; it never releases in-use tensors.
        """
        retire = set()
        for key, node in list(self.nodes.items()):
            if key not in self.nodes:
                continue
            if block_id in node.full_ids:
                for victim in [*self._children(key), key]:
                    dependent = self.nodes.get(victim)
                    if dependent is not None:
                        retire.update(dependent.full_ids + dependent.mamba_ids)
                        self._drop(victim, physical=False)
            elif block_id in node.mamba_ids:
                retire.update(node.mamba_ids)
                if self._is_leaf(key):
                    retire.update(node.full_ids)
                    self._drop(key, physical=False)
                else:
                    node.mamba_ids = ()
                    node.mamba_locked = False
                    self.mamba_lru.pop(key, None)
        retire.discard(block_id)
        if retire:
            self.evict_blocks(retire)

    def close_tenant(self, salt):
        if any((node.full_locked or node.mamba_locked)
               for key, node in self.nodes.items() if key[0] == salt):
            return False
        for key in list(self.nodes):
            if key[0] == salt:
                self._drop(key, physical=True)
        return True

    def moved(self, source_id, destination_id):
        """Follow native copy-on-write when it relocates a cached hash."""
        for node in self.nodes.values():
            node.full_ids = tuple(
                destination_id if i == source_id else i for i in node.full_ids
            )
            node.mamba_ids = tuple(
                destination_id if i == source_id else i for i in node.mamba_ids
            )

    def reset(self):
        """Forget handles after the native pool has already cleared its hashes."""
        for key in list(self.nodes):
            self._drop(key, physical=False)
