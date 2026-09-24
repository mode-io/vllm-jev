"""vLLM Jev: candidate scores via vLLM's native pooling engine."""

__version__ = "0.1.0"

ARCHITECTURE = "VllmJevQwen35ForSequenceClassification"
QWEN3_ARCHITECTURE = "VllmJevQwen3ForSequenceClassification"
QWEN3_TOKEN_ARCHITECTURE = "VllmJevQwen3ForTokenEmbedding"


def register() -> None:
    """Register models without importing worker-side implementations."""
    from vllm.model_executor.models import ModelRegistry

    for architecture in (
        ARCHITECTURE,
        QWEN3_ARCHITECTURE,
        QWEN3_TOKEN_ARCHITECTURE,
    ):
        ModelRegistry.register_model(architecture, f"vllm_jev.model:{architecture}")
