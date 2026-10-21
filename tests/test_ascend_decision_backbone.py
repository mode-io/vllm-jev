import json

import pytest
import torch
from safetensors.torch import save_file

from vllm_jev import QWEN35_TEXT_TOKEN_ARCHITECTURE
from vllm_jev.decision_export import decision_architecture


@pytest.mark.parametrize(
    "kind,backend,expected",
    [
        ("qwen3_5_text", "ascend", QWEN35_TEXT_TOKEN_ARCHITECTURE),
        ("qwen3_5_text", "cuda", "Qwen3_5ForCausalLM"),
        ("qwen3", "ascend", "Qwen3ForCausalLM"),
        ("qwen3_5_moe_text", "cuda", "Qwen3_5MoeForCausalLM"),
    ],
)
def test_decision_architecture_keeps_backend_specific_compatibility(
    kind, backend, expected
):
    assert decision_architecture(kind, backend) == expected


def test_ascend_export_rewrites_architecture_and_rehashes_manifest(
    monkeypatch, tmp_path
):
    from vllm_jev import backend
    from vllm_jev.checkpoint import sha256
    from vllm_jev.decision_export import _write, verify_decision

    monkeypatch.setattr(backend, "resolve_backend", lambda: "ascend")
    config = {
        "model_type": "qwen3_5_text",
        "architectures": ["Qwen3_5ForCausalLM"],
        "hidden_size": 2,
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    for name in ("tokenizer.json", "tokenizer_config.json"):
        (tmp_path / name).write_text("{}")
    save_file(
        {"model.embed_tokens.weight": torch.ones(4, 2)}, tmp_path / "model.safetensors"
    )
    original_weights_hash = sha256(tmp_path / "model.safetensors")
    manifest = _write(
        tmp_path,
        "test/fixture",
        "fixed",
        {
            "prompt_protocol": "decider_slot_v1",
            "max_options": 2,
            "temperature": 1.0,
            "labels": ["a", "b"],
            "label_token_ids": [1, 2],
        },
        {"weight": torch.ones(2, 2)},
    )
    assert json.loads((tmp_path / "config.json").read_text())["architectures"] == [
        QWEN35_TEXT_TOKEN_ARCHITECTURE
    ]
    assert sha256(tmp_path / "model.safetensors") == original_weights_hash
    assert manifest["files"]["config.json"] == sha256(tmp_path / "config.json")
    verify_decision(tmp_path, full=True)


def test_weight_alias_stream_keeps_values_and_rejects_collisions():
    from vllm_jev.checkpoint import text_backbone_weights

    first = torch.tensor([1.25, -3.0])
    second = torch.tensor([7.0])
    source = iter(
        [
            ("model.language_model.layers.0.input_layernorm.weight", first),
            ("language_model.lm_head.weight", second),
        ]
    )
    mapped = list(text_backbone_weights(source))
    assert [name for name, _ in mapped] == [
        "model.layers.0.input_layernorm.weight",
        "lm_head.weight",
    ]
    assert mapped[0][1] is first
    assert mapped[1][1] is second
    assert list(source) == []
    with pytest.raises(ValueError, match="collide"):
        list(
            text_backbone_weights(
                [
                    ("model.language_model.norm.weight", first),
                    ("model.norm.weight", second),
                ]
            )
        )


def test_flat_backbone_weights_are_not_remapped():
    from vllm_jev.checkpoint import text_backbone_weights

    value = torch.ones(2)
    assert (
        list(text_backbone_weights([("model.layers.0.linear_attn.A_log", value)]))[0][0]
        == "model.layers.0.linear_attn.A_log"
    )
