"""Native vLLM pooling models, imported only when vLLM loads a backbone."""

import torch
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


class VllmJevQwen3ForTokenEmbedding(as_embedding_model(Qwen3ForCausalLM)):
    """Native Qwen3 token embeddings for marker-position decision heads."""
