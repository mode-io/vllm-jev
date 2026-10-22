"""CPU-only contract tests for native Kev pointer export of composite bases.

These cover the large/composite-base branch of ``export_kev`` (the form used
by ``kev-4b`` and ``kev-27b``): a multimodal Qwen3.5 base config with the text
backbone under ``text_config``, a LoRA adapter merged one shard at a time, and
a Kev pointer head. No released weights are downloaded; tiny synthetic tensors
stand in for the backbone.
"""

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

try:
    import vllm  # noqa: F401
except ImportError:
    sys.modules["vllm"] = ModuleType("vllm")

from vllm_jev import decision_export as dx

MARKER_NAMES = [
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|box_start|>",
    "<|box_end|>",
    "<|fim_suffix|>",
]
MARKERS = [100, 101, 102, 103, 104]


class FakeTokenizer:
    def __init__(self):
        self.by_name = dict(zip(MARKER_NAMES, MARKERS))

    def convert_tokens_to_ids(self, value):
        return self.by_name.get(value)

    def save_pretrained(self, path):
        Path(path, "tokenizer.json").write_text("{}")
        Path(path, "tokenizer_config.json").write_text("{}")


def make_composite_fixture(tmp_path, monkeypatch, *, hidden=3):
    """A tiny multimodal Qwen3.5 base + LoRA adapter + Kev pointer head."""
    base, source, output = tmp_path / "base", tmp_path / "source", tmp_path / "out"
    base.mkdir()
    source.mkdir()
    # A language weight the LoRA targets, plus a vision weight to be dropped.
    language = (
        torch.arange(hidden * 4, dtype=torch.float32).reshape(4, hidden).bfloat16()
    )
    tensors = {
        "model.language_model.layers.0.self_attn.q_proj.weight": language,
        "model.visual.blocks.0.weight": torch.ones(2, 2).bfloat16(),
    }
    save_file(tensors, str(base / "model.safetensors"))
    (base / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: "model.safetensors" for name in tensors}})
    )
    (base / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "model_type": "qwen3_5",
                "tie_word_embeddings": True,
                "text_config": {
                    "model_type": "qwen3_5_text",
                    "hidden_size": hidden,
                    "max_position_embeddings": 8192,
                    "vocab_size": 1000,
                },
            }
        )
    )
    for name in ("tokenizer.json", "tokenizer_config.json"):
        (source / name).write_text("{}")
    rank = 2
    left = torch.tensor([[1.0, 2.0, 3.0], [0.5, 1.0, -1.0]])[:, :hidden]
    right = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 2.0]])
    save_file(
        {
            "base_model.model.layers.0.self_attn.q_proj.lora_A.weight": left,
            "base_model.model.layers.0.self_attn.q_proj.lora_B.weight": right,
        },
        str(source / "adapter_model.safetensors"),
    )
    (source / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": str(base),
                "peft_type": "LORA",
                "bias": "none",
                "r": rank,
                "lora_alpha": 4,
            }
        )
    )
    torch.manual_seed(0)
    head = {
        "q.weight": torch.randn(256, hidden),
        "q.bias": torch.randn(256),
        "k.weight": torch.randn(256, hidden),
        "k.bias": torch.randn(256),
    }
    torch.save(
        {
            "base": str(base),
            "option_isolation": False,
            "special_embeddings": False,
            "head_dim": 256,
            "head": head,
            "temperature": 2.4,
        },
        source / "head.pt",
    )
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: FakeTokenizer(),
    )
    return base, source, output, language, left, right


def test_export_merges_lora_per_shard_into_text_backbone(tmp_path, monkeypatch):
    base, source, output, language, left, right = make_composite_fixture(
        tmp_path, monkeypatch
    )
    manifest = dx.export_kev(base, source, output, "jaredpalmer/kev-27b", "rev1")

    assert manifest["prompt_protocol"] == "kev_pointer_v1"
    assert manifest["hidden_size"] == 3
    assert manifest["temperature"] == 2.4
    config = json.loads((output / "config.json").read_text())
    assert config["architectures"] == ["Qwen3_5ForCausalLM"]

    # The merged language weight is renamed to the text-backbone key and the
    # vision weight is dropped.
    index = json.loads((output / "model.safetensors.index.json").read_text())
    text_key = "model.layers.0.self_attn.q_proj.weight"
    assert text_key in index["weight_map"]
    assert not any("visual" in name for name in index["weight_map"])
    with safe_open(output / index["weight_map"][text_key], framework="pt") as file:
        merged = file.get_tensor(text_key)
    expected = (language.float() + (right @ left) * (4 / 2)).bfloat16()
    torch.testing.assert_close(merged, expected, rtol=0, atol=0)


def test_export_writes_pointer_head(tmp_path, monkeypatch):
    base, source, output, *_ = make_composite_fixture(tmp_path, monkeypatch)
    dx.export_kev(base, source, output, "jaredpalmer/kev-27b", "rev1")
    head = load_file(str(output / "decision_head.safetensors"))
    assert set(head) == {"q.weight", "q.bias", "k.weight", "k.bias"}
    assert head["q.weight"].shape == (256, 3)
    assert head["q.weight"].dtype == torch.float32


def test_export_verifies_the_result(tmp_path, monkeypatch):
    base, source, output, *_ = make_composite_fixture(tmp_path, monkeypatch)
    dx.export_kev(base, source, output, "jaredpalmer/kev-27b", "rev1")
    # Should already be verified by export; verifying again stays green.
    assert dx.verify_decision(output, full=True)["protocol"] == "kev_pointer_v1"


def test_export_rejects_option_isolated_head(tmp_path, monkeypatch):
    base, source, output, *_ = make_composite_fixture(tmp_path, monkeypatch)
    meta = torch.load(source / "head.pt", weights_only=True)
    meta["option_isolation"] = True
    torch.save(meta, source / "head.pt")
    with pytest.raises(ValueError, match="option-isolated"):
        dx.export_kev(base, source, output, "jaredpalmer/kev-27b", "rev1")


def test_export_rejects_mismatched_adapter_base(tmp_path, monkeypatch):
    base, source, output, *_ = make_composite_fixture(tmp_path, monkeypatch)
    adapter = json.loads((source / "adapter_config.json").read_text())
    adapter["base_model_name_or_path"] = "Qwen/some-other-base"
    (source / "adapter_config.json").write_text(json.dumps(adapter))
    with pytest.raises(ValueError, match="different bases"):
        dx.export_kev(base, source, output, "jaredpalmer/kev-27b", "rev1")


@pytest.mark.parametrize(
    "field,value",
    [
        ("use_rslora", True),
        ("use_dora", True),
        ("rank_pattern", {"layers.0.self_attn.q_proj": 8}),
        ("alpha_pattern", {"layers.0.self_attn.q_proj": 16}),
        ("fan_in_fan_out", True),
        ("lora_bias", True),
        ("modules_to_save", ["embed_tokens"]),
    ],
)
def test_export_rejects_non_vanilla_lora(tmp_path, monkeypatch, field, value):
    """rslora/dora/per-layer scales would silently corrupt the alpha/r merge."""
    base, source, output, *_ = make_composite_fixture(tmp_path, monkeypatch)
    adapter = json.loads((source / "adapter_config.json").read_text())
    adapter[field] = value
    (source / "adapter_config.json").write_text(json.dumps(adapter))
    with pytest.raises(ValueError, match="unsupported Kev adapter type"):
        dx.export_kev(base, source, output, "jaredpalmer/kev-27b", "rev1")


def test_export_falls_back_to_base_tokenizer(tmp_path, monkeypatch):
    """When the adapter tag ships no tokenizer, use the base's."""
    base, source, output, *_ = make_composite_fixture(tmp_path, monkeypatch)
    (source / "tokenizer.json").unlink()
    (source / "tokenizer_config.json").unlink()
    (base / "tokenizer.json").write_text("{}")
    (base / "tokenizer_config.json").write_text("{}")

    seen = {}

    def fake_from_pretrained(path, *args, **kwargs):
        seen["path"] = Path(path)
        return FakeTokenizer()

    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained", fake_from_pretrained
    )
    dx.export_kev(base, source, output, "jaredpalmer/kev-27b", "rev1")
    assert seen["path"] == base


def test_kev_27b_pins_the_lora_tag_for_dispatch():
    """kev-27b main is a full fine-tune; serving must use its v1-lora tag."""
    from vllm_jev import native_models

    assert (
        native_models.KEV_PINNED_REVISIONS["jaredpalmer/kev-27b"]
        == "01b81998019be550f0ae858727df49bac9511195"
    )


def test_kev_27b_dispatch_requests_the_pinned_revision(tmp_path, monkeypatch):
    from vllm_jev import native_models

    pinned = native_models.KEV_PINNED_REVISIONS["jaredpalmer/kev-27b"]
    requested = {}

    class FakeApi:
        def model_info(self, repo_id, **kwargs):
            requested["revision"] = kwargs.get("revision")
            return SimpleNamespace(
                sha=pinned,
                siblings=[
                    SimpleNamespace(rfilename=name)
                    for name in ("head.pt", "adapter_config.json")
                ],
            )

    def boom(**kwargs):
        raise AssertionError(f"stop after dispatch: downloading {kwargs['repo_id']}")

    monkeypatch.setattr(native_models, "HfApi", lambda: FakeApi())
    monkeypatch.setattr(native_models, "snapshot_download", boom)
    with pytest.raises(AssertionError, match="stop after dispatch"):
        native_models.prepare("jaredpalmer/kev-27b", tmp_path, "auto")
    assert requested["revision"] == pinned
