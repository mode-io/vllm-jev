"""Native vLLM pooling models, imported only when vLLM loads a backbone."""

import torch
import triton
import triton.language as tl
from vllm.model_executor.models.adapters import as_embedding_model, as_seq_cls_model
from vllm.model_executor.models.interfaces import IsHybrid, SupportsMRoPE
from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ForCausalLM,
    Qwen3_5ForConditionalGeneration,
)


class JevQwen35CausalLM(Qwen3_5ForCausalLM, IsHybrid, SupportsMRoPE):
    # The native text class needs the hybrid hooks from its multimodal wrapper.
    is_hybrid = True
    supports_mrope = True

    def get_mrope_input_positions(self, input_tokens, mm_features):
        if mm_features:
            raise ValueError("vLLM Jev is text-only")
        positions = torch.arange(len(input_tokens), dtype=torch.long)
        return positions.unsqueeze(0).expand(3, -1).contiguous(), 0

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config):
        return Qwen3_5ForConditionalGeneration.get_mamba_state_dtype_from_config(
            vllm_config
        )

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config):
        return Qwen3_5ForConditionalGeneration.get_mamba_state_shape_from_config(
            vllm_config
        )

    @classmethod
    def get_mamba_state_copy_func(cls):
        return Qwen3_5ForConditionalGeneration.get_mamba_state_copy_func()


class VllmJevQwen35ForSequenceClassification(as_seq_cls_model(JevQwen35CausalLM)):
    """Uses vLLM's Qwen3.5 backbone, ReplicatedLinear score, and pooler.

    Model loading maps ``score.weight`` and ``score.bias`` from the exported
    safetensors checkpoint. The pooling task returns the final-position scalar
    when ``PoolingParams(use_activation=False)`` is supplied.
    """


class VllmJevQwen3ForSequenceClassification(as_seq_cls_model(Qwen3ForCausalLM)):
    """Native Qwen3 pooling backbone with a merged scalar decision head."""


class VllmValenQwen35ForTokenEmbedding(
    as_embedding_model(Qwen3_5ForConditionalGeneration)
):
    """Native Qwen3.5 token states for Valen's shared decision head."""


class VllmVjevQwen35ForTokenEmbedding(
    as_embedding_model(Qwen3_5ForConditionalGeneration)
):
    """Native Qwen3.5 image/text token states for vjev's listwise head."""


class VllmJevQwen3ForTokenEmbedding(as_embedding_model(Qwen3ForCausalLM)):
    """Native Qwen3 token embeddings for marker-position decision heads."""


@triton.jit
def _token_score_kernel(
    hidden,
    weight,
    bias,
    scores,
    size: tl.constexpr,
    row_stride: tl.constexpr,
    col_stride: tl.constexpr,
    block: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, block)
    value = tl.load(
        hidden + row * row_stride + col * col_stride, col < size, other=0
    ).to(tl.float32)
    scale = tl.load(weight + col, col < size, other=0)
    mean = tl.sum(value, 0) / size
    centered = tl.where(col < size, value - mean, 0.0)
    variance = tl.sum(centered * centered, 0) / size
    score = tl.sum(centered * tl.rsqrt(variance + 1e-5) * scale, 0)
    tl.store(scores + row, score + tl.load(bias))


class VllmJevQwen3ForTokenScoreEmbedding(as_embedding_model(Qwen3ForCausalLM)):
    """Project Tiny-Jev token scores before returning pooling outputs."""

    def _init_pooler(self, vllm_config, prefix: str = ""):
        from torch import nn
        from vllm.model_executor.layers.pooler.tokwise import pooler_for_token_embed

        hidden_size = vllm_config.model_config.get_hidden_size()
        self.score = nn.Linear(hidden_size, 1, bias=True, dtype=torch.float32)
        pooler_config = vllm_config.model_config.pooler_config
        assert pooler_config is not None
        return pooler_for_token_embed(pooler_config, projector=self._project_scores)

    def _project_scores(self, hidden_states: torch.Tensor) -> torch.Tensor:
        rows, size = hidden_states.shape
        if rows < 256:
            return hidden_states
        result = torch.empty(
            (rows, 1), device=hidden_states.device, dtype=torch.float32
        )
        _token_score_kernel[(rows,)](
            hidden_states,
            self.score.weight,
            self.score.bias,
            result,
            size,
            hidden_states.stride(0),
            hidden_states.stride(1),
            triton.next_power_of_2(size),
            num_warps=4,
        )
        return result
