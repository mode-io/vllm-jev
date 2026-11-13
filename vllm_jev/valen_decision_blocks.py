"""Valen-Decision-1.5 block-structured attention for the full-attention layers.

A request is one sequence: the state, a user turn holding every question, and
an assistant turn holding one readout per question. In the full-attention
layers the state sees only itself, the shared turn markers see the state and
each other, and a question sees the state, the markers and its own block. With
``bbox_blocks`` the decision questions share one block and every bbox question
keeps its own. The linear-attention layers stay causal.

The masks follow ``valen/modeling/qwen/block_attention.py`` and the block
layout of Valen's vLLM plugin. This module imports neither vLLM nor the
tokenizer, so the masks can be tested on CPU.
"""

import torch
import torch.nn.functional as functional

PREFIX, COMMON = -1, 0
# The compiler closes the state with these two turns: questions, then readouts.
GLUE = ("<|im_start|>user\n", "<|im_end|>\n<|im_start|>assistant\n")
MASKS = ("block_bidirectional", "bbox_blocks")


def block_segments(mask: str, segments: list[int], bbox: set[int]) -> list[int]:
    """Block ids of the question tokens under a full-attention mask mode.

    ``segments`` holds the compiler's ids: turn markers 0, question i as i + 1.
    """
    if mask == "block_bidirectional":
        return [COMMON] * len(segments)
    if mask == "bbox_blocks":
        return [s if s == COMMON or s in bbox else 1 for s in segments]
    raise ValueError(f"unsupported Valen full-attention mask: {mask}")


def token_segments(ids, glue, bidirectional=False, encode=None):
    """Block ids of one prompt, read from its last two chat turns.

    ``ids`` is a 1-D tensor and ``glue`` the token ids of ``GLUE``. Decision
    questions share block 1. With the tokenizer's ``encode``, each bbox
    question keeps its own block, found by the question headers the compiler
    writes. A prompt without the two turns has no blocks (None).
    """
    turns = (ids == glue[0][0]).nonzero().flatten().tolist()
    if len(turns) < 2:
        return None
    user, assistant = turns[-2:]
    segments = torch.ones_like(ids)
    segments[:user] = PREFIX
    if bidirectional:
        segments[user:] = COMMON
        return segments
    segments[user : user + len(glue[0])] = COMMON
    start = assistant - glue[1].index(glue[0][0])
    segments[start : start + len(glue[1])] = COMMON
    if encode is None:
        return segments
    tokens = ids.tolist()
    newline = len(encode("\n"))
    cursor, readout, question, marks = user + len(glue[0]), start + len(glue[1]), 0, []
    while readout < len(tokens):
        question += 1
        header = encode(f"Question {question}\nTask:")
        while cursor < start and tokens[cursor : cursor + len(header)] != header:
            cursor += 1
        bbox = encode(f"Question {question}\nTask: bbox")
        end = readout + len(encode(f"Question {question} Decision:")) + newline
        marks.append(
            (cursor, readout, end, tokens[cursor : cursor + len(bbox)] == bbox)
        )
        cursor, readout = cursor + len(header), end
    stops = [mark[0] for mark in marks[1:]] + [start]
    for index, ((begin, readout, end, bbox), stop) in enumerate(zip(marks, stops)):
        if bbox:
            segments[begin:stop] = segments[readout:end] = index + 1
    return segments


def segment_mask(segments: torch.Tensor) -> torch.Tensor:
    """Boolean [L, L] visibility of one prompt; True means the key is visible."""
    query, key = segments[:, None], segments[None, :]
    key_prefix = key == PREFIX
    return torch.where(
        query == PREFIX, key_prefix, key_prefix | (key == COMMON) | (key == query)
    )


def block_attention(q, k, v, segments, scale):
    """Attention of one complete prompt under its block mask.

    ``q`` is [L, H, D]; ``k`` and ``v`` are [L, KV, D]. Returns [L, H, D].
    A prompt without blocks (``segments`` None) stays causal.
    """
    groups = q.shape[1] // k.shape[1]
    q = q.transpose(0, 1)
    k = k.transpose(0, 1).repeat_interleave(groups, 0)
    v = v.transpose(0, 1).repeat_interleave(groups, 0)
    if segments is None:
        out = functional.scaled_dot_product_attention(
            q, k, v, is_causal=True, scale=scale
        )
        return out.transpose(0, 1)
    out = torch.empty_like(q)
    prefix = segments == PREFIX
    shared = prefix | (segments == COMMON)
    for block in segments.unique():
        rows = segments == block
        keys = prefix if block == PREFIX else shared | rows
        out[:, rows] = functional.scaled_dot_product_attention(
            q[:, rows], k[:, keys], v[:, keys], scale=scale
        )
    return out.transpose(0, 1)


def prompt_blocks(
    input_ids, positions, metadata, layer_name, glue, bidirectional=False, encode=None
):
    """``(start, end, segments)`` of every prompt in one engine step.

    ``metadata`` is the step's attention metadata; ``layer_name`` names a
    full-attention layer whose entry reports where each prompt starts. Returns
    None when no prompt has blocks, so the engine kernel runs unchanged.
    """
    if not metadata:
        # Memory profiling runs without request metadata.
        return None
    if input_ids is None:
        # Without the prompt's tokens the mask would silently fall back to causal.
        raise RuntimeError(
            "Valen-Decision-1.5 needs the step's raw prompt tokens; this engine "
            "did not supply them"
        )
    if isinstance(metadata, list):
        raise RuntimeError("Valen-Decision-1.5 does not support micro-batched steps")
    starts = metadata[layer_name].query_start_loc.tolist()
    first = positions[0] if positions.ndim == 2 else positions
    blocks = []
    for start, end in zip(starts, starts[1:]):
        if end <= start:
            continue
        segments = token_segments(input_ids[start:end], glue, bidirectional, encode)
        if segments is not None and int(first[start]) != 0:
            # A cached prefix or a chunk would hide part of the block mask.
            raise RuntimeError(
                "Valen-Decision-1.5 needs each request as one complete prefill; "
                "disable prefix caching and chunked prefill"
            )
        blocks.append((start, end, segments))
    if all(segments is None for _, _, segments in blocks):
        return None
    return blocks


def backbone_weights(weights):
    """Stream the release's backbone under vLLM's names; the heads stay out."""
    for name, tensor in weights:
        if name.startswith(("head.", "grounding.")):
            continue
        if name.startswith("backbone."):
            name = "model." + name[len("backbone.") :]
        yield name, tensor


class BlockAttention(torch.nn.Module):
    """A full-attention layer's kernel under Valen-Decision-1.5 block masks.

    ``blocks`` lists ``(start, end, segments)`` for every prompt of the step.
    A block forward is one complete prefill, so it neither reads nor writes
    the KV cache. Without blocks the engine kernel runs unchanged.
    """

    def __init__(self, attn, scale):
        super().__init__()
        self.attn = attn
        self.scale = scale
        self.blocks = None

    def forward(self, query, key, value, *args, **kwargs):
        if self.blocks is None:
            return self.attn(query, key, value, *args, **kwargs)
        attn = self.attn
        q = query.view(-1, attn.num_heads, attn.head_size)
        k = key.view(-1, attn.num_kv_heads, attn.head_size)
        v = value.view(-1, attn.num_kv_heads, attn.head_size)
        output = q.new_zeros(q.shape)
        for start, end, segments in self.blocks:
            output[start:end] = block_attention(
                q[start:end], k[start:end], v[start:end], segments, self.scale
            )
        return output.view(-1, attn.num_heads * attn.head_size)
