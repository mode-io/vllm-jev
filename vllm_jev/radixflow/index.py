"""Bounded namespace-local Rust index; matches never grant cache reuse."""

import ctypes
from collections import OrderedDict


class RustIndex:
    def __init__(self, library, max_namespaces=256, max_tokens=262144):
        if min(max_namespaces, max_tokens) < 1:
            raise ValueError("Rust index limits must be positive")
        if not library:
            raise ValueError("build the Rust index and set VLLM_JEV_RADIXFLOW_LIBRARY")
        self.lib = ctypes.CDLL(library)
        self.lib.jev_radix_new.restype = ctypes.c_void_p
        self.lib.jev_radix_free.argtypes = [ctypes.c_void_p]
        for name in ("jev_radix_match", "jev_radix_observe"):
            fn = getattr(self.lib, name)
            fn.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint32),
                ctypes.c_size_t,
            ]
            fn.restype = ctypes.c_size_t
        self.entries = OrderedDict()
        self.max_namespaces, self.max_tokens = max_namespaces, max_tokens
        self.observations = self.evictions = self.refusals = 0

    def observe(self, namespace, tokens):
        if len(tokens) > self.max_tokens:
            self.refusals += 1
            return None
        if any(type(t) is not int or not 0 <= t <= 0xFFFFFFFF for t in tokens):
            raise ValueError("token IDs must be uint32 integers")
        # A global input-token charge bounds all namespace trees together.
        while (
            self.entries
            and sum(n for _, n in self.entries.values()) + len(tokens) > self.max_tokens
        ):
            self._drop(next(iter(self.entries)))
        if namespace not in self.entries:
            while len(self.entries) >= self.max_namespaces:
                self._drop(next(iter(self.entries)))
            pointer = self.lib.jev_radix_new()
            if not pointer:
                raise MemoryError("Rust index allocation failed")
            self.entries[namespace] = (pointer, 0)
        pointer, charged = self.entries[namespace]
        buf = (ctypes.c_uint32 * len(tokens))(*tokens)
        matched = self.lib.jev_radix_observe(pointer, buf, len(tokens))
        if matched == ctypes.c_size_t(-1).value:
            raise RuntimeError("Rust index rejected an observation")
        self.entries[namespace] = (pointer, charged + len(tokens))
        self.entries.move_to_end(namespace)
        self.observations += 1
        return matched

    def _drop(self, namespace):
        pointer, _ = self.entries.pop(namespace)
        self.lib.jev_radix_free(pointer)
        self.evictions += 1

    def close(self):
        for key in list(self.entries):
            self._drop(key)

    def stats(self):
        return dict(
            namespaces=len(self.entries),
            observed_tokens=sum(n for _, n in self.entries.values()),
            observations=self.observations,
            evictions=self.evictions,
            refusals=self.refusals,
        )
