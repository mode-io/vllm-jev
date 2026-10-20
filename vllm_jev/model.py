"""Native vLLM pooling models, imported only when vLLM loads a backbone."""

import torch
from vllm.model_executor.models.adapters import as_embedding_model, as_seq_cls_model
from vllm.model_executor.models.interfaces import IsHybrid, SupportsMRoPE
from vllm.model_executor.models.modernbert import ModernBertModel
from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ForCausalLM,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5MoeForConditionalGeneration,
)
from vllm.model_executor.models.qwen3_vl_moe import (
    Qwen3VLMoeForConditionalGeneration,
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


class VllmRsiJevQwen35ForTokenEmbedding(
    as_embedding_model(Qwen3_5ForConditionalGeneration)
):
    """Native Qwen3.5 final states for RSI-Jev's option cross-attention head."""


class VllmClefQwen35ForTokenEmbedding(
    as_embedding_model(Qwen3_5ForConditionalGeneration)
):
    """Native Qwen3.5 image/video/text states for Clef's joint schema head."""


class VllmJevAnyPointerQwen35ForTokenEmbedding(
    as_embedding_model(Qwen3_5ForConditionalGeneration)
):
    """Native dense Qwen3.5 image/text states for JevAny's Pointer head."""


class VllmDrexQwen35MoeForTokenEmbedding(
    as_embedding_model(Qwen3_5MoeForConditionalGeneration)
):
    """Native merged Drex MoE backbone; pointer head runs after pooling."""


class VllmJevAnyPointerQwen35MoeForTokenEmbedding(
    as_embedding_model(Qwen3_5MoeForConditionalGeneration)
):
    """Native MoE Qwen3.5 image/text states for JevAny's Pointer head."""


class VllmJevAnyPointerQwen3VLMoeForTokenEmbedding(
    as_embedding_model(Qwen3VLMoeForConditionalGeneration)
):
    """Native MoE Qwen3-VL image/text states for JevAny's Pointer head."""


class VllmJevQwen3ForTokenEmbedding(as_embedding_model(Qwen3ForCausalLM)):
    """Native Qwen3 token embeddings for marker-position decision heads."""


class VllmLayaModernBertForTokenEmbedding(as_embedding_model(ModernBertModel)):
    """Native bidirectional ModernBERT states for Laya's trained decision head."""


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
        from .token_score import project_token_scores

        return project_token_scores(hidden_states, self.score.weight, self.score.bias)
