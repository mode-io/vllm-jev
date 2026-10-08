"""Bounded exact-token radix index; independent of model and cache storage."""

from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class _Node:
    edge: tuple[int, ...] = ()
    children: dict[int, "_Node"] = field(default_factory=dict)
    terminal: int = 0

    def observe(self, tokens: tuple[int, ...]) -> int:
        node, offset = self, 0
        while offset < len(tokens):
            first = tokens[offset]
            child = node.children.get(first)
            if child is None:
                node.children[first] = _Node(tokens[offset:], terminal=1)
                return offset
            shared = 0
            while (
                shared < len(child.edge)
                and offset + shared < len(tokens)
                and child.edge[shared] == tokens[offset + shared]
            ):
                shared += 1
            if shared == len(child.edge):
                offset += shared
                node = child
                continue
            split = offset + shared
            old_edge = child.edge[shared:]
            old = _Node(old_edge, child.children, child.terminal)
            child.edge = child.edge[:shared]
            child.children = {old_edge[0]: old}
            child.terminal = int(split == len(tokens))
            if split < len(tokens):
                child.children[tokens[split]] = _Node(tokens[split:], terminal=1)
            return split
        node.terminal += 1
        return offset

    def forget(self, tokens: tuple[int, ...]) -> bool:
        node, offset = self, 0
        ancestors = []
        while offset < len(tokens):
            first = tokens[offset]
            child = node.children.get(first)
            if child is None or tokens[offset:offset + len(child.edge)] != child.edge:
                return False
            ancestors.append((node, first))
            offset += len(child.edge)
            node = child
        if not node.terminal:
            return False
        node.terminal -= 1
        for parent, first in reversed(ancestors):
            child = parent.children[first]
            if not child.terminal and not child.children:
                del parent.children[first]
            elif not child.terminal and len(child.children) == 1:
                grandchild = next(iter(child.children.values()))
                child.edge += grandchild.edge
                child.children = grandchild.children
                child.terminal = grandchild.terminal
        return True

    def counts(self) -> tuple[int, int]:
        nodes, edge_tokens = 0, 0
        pending = [self]
        while pending:
            node = pending.pop()
            nodes += 1
            edge_tokens += len(node.edge)
            pending.extend(node.children.values())
        return nodes, edge_tokens


class PrefixTree:
    """An exact compressed trie; splits never imply model-state availability."""

    def __init__(self) -> None:
        self._root = _Node()

    def observe(self, tokens: tuple[int, ...]) -> int:
        return self._root.observe(tokens)

    def forget(self, tokens: tuple[int, ...]) -> bool:
        return self._root.forget(tokens)

    def counts(self) -> tuple[int, int]:
        return self._root.counts()


class BoundedPrefixTrees:
    """Limit tenant trees and per-tenant request paths with LRU removal."""

    def __init__(
        self,
        max_tenants: int,
        max_paths: int,
        on_tenant_evicted: Callable[[str], bool],
    ) -> None:
        if max_tenants < 1 or max_paths < 1:
            raise ValueError("tree limits must be positive")
        self.max_tenants = max_tenants
        self.max_paths = max_paths
        self.on_tenant_evicted = on_tenant_evicted
        self._trees: OrderedDict[str, PrefixTree] = OrderedDict()
        self._paths: dict[str, deque[tuple[int, ...]]] = {}

    def observe(self, tenant: str, tokens: tuple[int, ...]) -> int | None:
        tree = self._trees.get(tenant)
        if tree is None:
            if len(self._trees) >= self.max_tenants:
                victim = next(
                    (key for key in list(self._trees) if self._evict(key)), None
                )
                if victim is None:
                    return None
            tree = PrefixTree()
            self._trees[tenant] = tree
            self._paths[tenant] = deque()
        else:
            self._trees.move_to_end(tenant)
        matched = tree.observe(tokens)
        paths = self._paths[tenant]
        paths.append(tokens)
        if len(paths) > self.max_paths:
            assert tree.forget(paths.popleft())
        return matched

    def attach(self, tenant: str, prefix: tuple[int, ...]) -> bool:
        tree = self._trees.get(tenant)
        if tree is None:
            return False
        tree.observe(prefix)
        return True

    def detach(self, tenant: str, prefix: tuple[int, ...]) -> bool:
        tree = self._trees.get(tenant)
        return tree.forget(prefix) if tree is not None else False

    def counts(self, tenant: str) -> tuple[int, int] | None:
        tree = self._trees.get(tenant)
        return tree.counts() if tree is not None else None

    def _evict(self, tenant: str) -> bool:
        if not self.on_tenant_evicted(tenant):
            return False
        del self._trees[tenant]
        del self._paths[tenant]
        return True
