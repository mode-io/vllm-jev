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


class VllmJevQwen35TextForTokenEmbedding(as_embedding_model(JevQwen35CausalLM)):
    """Pure text Qwen3.5 token pooling with native hybrid state hooks."""

    def load_weights(self, weights):
        from .checkpoint import text_backbone_weights

        return super().load_weights(text_backbone_weights(weights))


class VllmJevQwen3ForSequenceClassification(as_seq_cls_model(Qwen3ForCausalLM)):
    """Native Qwen3 pooling backbone with a merged scalar decision head."""


class VllmValenQwen35ForTokenEmbedding(
    as_embedding_model(Qwen3_5ForConditionalGeneration)
):
    """Native Qwen3.5 token states for Valen's shared decision head."""


class VllmValenDecisionQwen35ForTokenEmbedding(
    as_embedding_model(Qwen3_5ForConditionalGeneration)
):
    """Native Qwen3.5 token states under Valen-Decision-1.5 block attention.

    The full-attention layers follow the block mask the release was trained
    with; the linear-attention layers stay causal. The Mixer and grounding
    heads run after pooling. The Valen preview keeps its own class above.
    """

    # Block boundaries are read from the prompt's chat-turn tokens.
    requires_raw_input_tokens = True

    def __init__(self, *, vllm_config, prefix: str = "", **kwargs):
        super().__init__(vllm_config=vllm_config, prefix=prefix, **kwargs)
        import json
        from pathlib import Path

        from vllm.tokenizers.registry import cached_tokenizer_from_config

        from .valen_decision_blocks import GLUE, MASKS, BlockAttention

        manifest = json.loads(
            (
                Path(vllm_config.model_config.model) / "valen_decision_manifest.json"
            ).read_text()
        )
        mask = manifest["full_attention_mask"]
        if mask not in MASKS:
            raise ValueError(f"unsupported Valen full-attention mask: {mask}")
        tokenizer = cached_tokenizer_from_config(vllm_config.model_config)
        self._block_encode = lambda text: tokenizer.encode(
            text, add_special_tokens=False
        )
        self._block_glue = [self._block_encode(text) for text in GLUE]
        self._block_bidirectional = mask == "block_bidirectional"
        self._block_layers = []
        for layer in self.language_model.model.layers:
            if layer.layer_type != "full_attention":
                continue
            attention = layer.self_attn
            if attention.attn.attn_backend.get_name() not in (
                "FLASH_ATTN",
                "TRITON_ATTN",
            ):
                # Other backends do not report where each prompt starts.
                raise ValueError(
                    "Valen-Decision-1.5 requires --attention-backend FLASH_ATTN "
                    "or TRITON_ATTN"
                )
            attention.attn = BlockAttention(attention.attn, attention.scaling)
            self._block_layers.append(attention.attn)

    def forward(
        self,
        input_ids,
        positions,
        intermediate_tensors=None,
        inputs_embeds=None,
        **kwargs,
    ):
        from vllm.forward_context import get_forward_context

        from .valen_decision_blocks import prompt_blocks

        blocks = prompt_blocks(
            input_ids,
            positions,
            get_forward_context().attn_metadata,
            self._block_layers[0].attn.layer_name,
            self._block_glue,
            self._block_bidirectional,
            self._block_encode,
        )
        for layer in self._block_layers:
            layer.blocks = blocks
        try:
            return super().forward(
                input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs
            )
        finally:
            for layer in self._block_layers:
                layer.blocks = None

    def load_weights(self, weights):
        from .valen_decision_blocks import backbone_weights

        return super().load_weights(backbone_weights(weights))


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
