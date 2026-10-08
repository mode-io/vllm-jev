"""vLLM Jev: candidate scores via vLLM's native pooling engine."""

__version__ = "0.3.3"

ARCHITECTURE = "VllmJevQwen35ForSequenceClassification"
QWEN3_ARCHITECTURE = "VllmJevQwen3ForSequenceClassification"
QWEN3_TOKEN_ARCHITECTURE = "VllmJevQwen3ForTokenEmbedding"
QWEN3_PROJECTED_ARCHITECTURE = "VllmJevQwen3ForTokenScoreEmbedding"
VALEN_QWEN35_ARCHITECTURE = "VllmValenQwen35ForTokenEmbedding"
VJEV_QWEN35_ARCHITECTURE = "VllmVjevQwen35ForTokenEmbedding"
LAYA_MODERNBERT_ARCHITECTURE = "VllmLayaModernBertForTokenEmbedding"
RSIJEV_QWEN35_ARCHITECTURE = "VllmRsiJevQwen35ForTokenEmbedding"
CLEF_QWEN35_ARCHITECTURE = "VllmClefQwen35ForTokenEmbedding"


def register() -> None:
    """Register models without importing worker-side implementations."""
    import os

    from .precision import configure_matmul

    configure_matmul()
    from vllm.model_executor.models import ModelRegistry

    for architecture in (
        ARCHITECTURE,
        QWEN3_ARCHITECTURE,
        QWEN3_TOKEN_ARCHITECTURE,
        QWEN3_PROJECTED_ARCHITECTURE,
        VALEN_QWEN35_ARCHITECTURE,
        VJEV_QWEN35_ARCHITECTURE,
        LAYA_MODERNBERT_ARCHITECTURE,
        RSIJEV_QWEN35_ARCHITECTURE,
        CLEF_QWEN35_ARCHITECTURE,
    ):
        ModelRegistry.register_model(architecture, f"vllm_jev.model:{architecture}")
    if os.environ.get("VLLM_JEV_ONLINE_PREFIX_CACHE") == "1":
        from .online_prefix import install

        install()
