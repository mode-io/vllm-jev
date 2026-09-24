"""Register a Qwen3.5 scalar classifier using vLLM's own pooling adapter."""

ARCHITECTURE = "VllmJevQwen35ForSequenceClassification"
QWEN3_ARCHITECTURE = "VllmJevQwen3ForSequenceClassification"
QWEN3_TOKEN_ARCHITECTURE = "VllmJevQwen3ForTokenEmbedding"


def register() -> None:
    """The vLLM general-plugin entry point; defer model import until loading."""
    from vllm.model_executor.models import ModelRegistry

    ModelRegistry.register_model(
        ARCHITECTURE,
        "vllm_jev.model:VllmJevQwen35ForSequenceClassification",
    )
    ModelRegistry.register_model(
        QWEN3_ARCHITECTURE,
        "vllm_jev.model:VllmJevQwen3ForSequenceClassification",
    )
    ModelRegistry.register_model(
        QWEN3_TOKEN_ARCHITECTURE,
        "vllm_jev.model:VllmJevQwen3ForTokenEmbedding",
    )


def _model_class():
    import torch
    from vllm.model_executor.models.adapters import as_seq_cls_model
    from vllm.model_executor.models.interfaces import IsHybrid, SupportsMRoPE
    from vllm.model_executor.models.qwen3_5 import (
        Qwen3_5ForCausalLM,
        Qwen3_5ForConditionalGeneration,
    )

    class JevQwen35CausalLM(Qwen3_5ForCausalLM, IsHybrid, SupportsMRoPE):
        # Native vLLM Qwen3.5 text class lacks the hybrid marker. The same
        # cache-state hooks already exist on its multimodal wrapper.
        is_hybrid = True
        supports_mrope = True

        def get_mrope_input_positions(self, input_tokens, mm_features):
            if mm_features:
                raise ValueError("vLLM Jev is text-only")
            seq_len = len(input_tokens)
            positions = torch.arange(seq_len, dtype=torch.long)
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

    return as_seq_cls_model(JevQwen35CausalLM)


class VllmJevQwen35ForSequenceClassification(_model_class()):
    """Uses vLLM's Qwen3.5 backbone, ReplicatedLinear score, and pooler.

    Model loading maps ``score.weight`` and ``score.bias`` from the exported
    safetensors checkpoint. The pooling task returns the final-position scalar
    when ``PoolingParams(use_activation=False)`` is supplied.
    """


def _qwen3_model_class():
    from vllm.model_executor.models.adapters import as_seq_cls_model
    from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM

    return as_seq_cls_model(Qwen3ForCausalLM)


class VllmJevQwen3ForSequenceClassification(_qwen3_model_class()):
    """Native Qwen3 pooling backbone with a merged scalar decision head."""


def _qwen3_embedding_class():
    from vllm.model_executor.models.adapters import as_embedding_model
    from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM

    return as_embedding_model(Qwen3ForCausalLM)


class VllmJevQwen3ForTokenEmbedding(_qwen3_embedding_class()):
    """Native Qwen3 token embeddings for marker-position decision heads."""
