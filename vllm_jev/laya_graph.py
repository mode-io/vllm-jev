"""Fixed-shape CUDA Graph readout for short Laya multilingual Choice requests."""

import threading
from dataclasses import dataclass

import torch


@dataclass
class _CapturedHead:
    graph: torch.cuda.CUDAGraph
    hidden: torch.Tensor
    attention_mask: torch.Tensor
    marker_pos: torch.Tensor
    marker_mask: torch.Tensor
    qtype: torch.Tensor
    outputs: tuple[torch.Tensor, torch.Tensor]


class LayaGraphHead:
    """Capture at most ten 32-token buckets; keep other shapes on eager."""

    def __init__(self, head: torch.nn.Module):
        self.head = head
        self._cache: dict[int, _CapturedHead] = {}
        self._lock = threading.Lock()

    def _capture(self, bucket: int, tensors: tuple[torch.Tensor, ...]) -> _CapturedHead:
        hidden, attention_mask, marker_pos, marker_mask, qtype = tensors
        static_hidden = torch.zeros(
            1, bucket, hidden.shape[-1], device=hidden.device, dtype=hidden.dtype
        )
        static_mask = torch.zeros(
            1, bucket, device=hidden.device, dtype=attention_mask.dtype
        )
        static_pos = torch.empty_like(marker_pos)
        static_marker_mask = torch.empty_like(marker_mask)
        static_qtype = torch.empty_like(qtype)
        static = (
            static_hidden,
            static_mask,
            static_pos,
            static_marker_mask,
            static_qtype,
        )
        self._copy(static, tensors)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(5):
                self.head(*static)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                outputs = self.head(*static)
        return _CapturedHead(graph, *static, outputs)

    @staticmethod
    def _copy(static: tuple[torch.Tensor, ...], inputs: tuple[torch.Tensor, ...]):
        static_hidden, static_mask, static_pos, static_marker_mask, static_qtype = (
            static
        )
        hidden, attention_mask, marker_pos, marker_mask, qtype = inputs
        length = hidden.shape[1]
        static_hidden.zero_()
        static_hidden[:, :length].copy_(hidden)
        static_mask.zero_()
        static_mask[:, :length].copy_(attention_mask)
        static_pos.copy_(marker_pos)
        static_marker_mask.copy_(marker_mask)
        static_qtype.copy_(qtype)

    def __call__(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        marker_pos: torch.Tensor,
        marker_mask: torch.Tensor,
        qtype: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            hidden.device.type != "cuda"
            or hidden.shape[0] != 1
            or marker_pos.shape[1] != 4
            or hidden.shape[1] > 320
        ):
            return self.head(hidden, attention_mask, marker_pos, marker_mask, qtype)
        bucket = ((hidden.shape[1] + 31) // 32) * 32
        inputs = (hidden, attention_mask, marker_pos, marker_mask, qtype)
        with self._lock:
            entry = self._cache.get(bucket)
            if entry is None:
                entry = self._capture(bucket, inputs)
                self._cache[bucket] = entry
            self._copy(
                (
                    entry.hidden,
                    entry.attention_mask,
                    entry.marker_pos,
                    entry.marker_mask,
                    entry.qtype,
                ),
                inputs,
            )
            entry.graph.replay()
            results = tuple(output.clone() for output in entry.outputs)
            torch.cuda.current_stream().synchronize()
        return results
