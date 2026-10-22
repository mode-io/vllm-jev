"""CPU-only contract tests for native Kev pointer export of composite bases.

These cover the composite-base LoRA branch of ``export_kev`` (``kev-4b`` and
the ``kev-27b`` v1-lora tag) and ``export_kev_full`` for full-weight releases
(``kev-27b`` main, v2). No released weights are downloaded; tiny synthetic
tensors stand in for the backbone, and every Hub call is stubbed.
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


def test_kev_27b_pins_the_v2_main_release():
    """kev-27b serves its pinned v2 full-weight main release by default."""
    from vllm_jev import native_models

    assert (
        native_models.KEV_PINNED_REVISIONS["jaredpalmer/kev-27b"]
        == "af0e6d551bdc2cc724f3e9d7a8bee1cd4fb8f7bf"
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


# --- Full-weight (kev-27b main / v2) form -------------------------------------


def make_full_fixture(tmp_path, monkeypatch, *, hidden=3):
    """A tiny flat Qwen3.5 text checkpoint with bare keys, as kev-27b main ships."""
    source, output = tmp_path / "full", tmp_path / "out"
    source.mkdir()
    tensors = {
        "embed_tokens.weight": torch.randn(1000, hidden).bfloat16(),
        "layers.0.linear_attn.in_proj_qkv.weight": torch.randn(6, hidden).bfloat16(),
        "layers.0.linear_attn.A_log": torch.randn(2),
        "layers.3.self_attn.q_proj.weight": torch.randn(4, hidden).bfloat16(),
        "norm.weight": torch.ones(hidden).bfloat16(),
    }
    names = sorted(tensors)
    shards = {
        "model-00001-of-00002.safetensors": names[:2],
        "model-00002-of-00002.safetensors": names[2:],
    }
    for shard, keys in shards.items():
        save_file({k: tensors[k] for k in keys}, str(source / shard))
    (source / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 1},
                "weight_map": {k: s for s, keys in shards.items() for k in keys},
            }
        )
    )
    (source / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3_5TextModel"],
                "model_type": "qwen3_5_text",
                "dtype": "bfloat16",
                "hidden_size": hidden,
                "vocab_size": 1000,
                "max_position_embeddings": 262144,
                "tie_word_embeddings": False,
                "mtp_num_hidden_layers": 1,
                "mtp_use_dedicated_embeddings": False,
            }
        )
    )
    for name in ("tokenizer.json", "tokenizer_config.json"):
        (source / name).write_text("{}")
    torch.manual_seed(1)
    head = {
        "q.weight": torch.randn(256, hidden),
        "q.bias": torch.randn(256),
        "k.weight": torch.randn(256, hidden),
        "k.bias": torch.randn(256),
    }
    torch.save(
        {
            "base": "Qwen/Qwen3.8-27B",
            "base_revision": "1d4bf0f2",
            "weights": "full",
            "weights_dtype": "bf16",
            "option_isolation": False,
            "special_embeddings": False,
            "head_dim": 256,
            "head": head,
            "temperature": 1.3195,
        },
        source / "head.pt",
    )
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: FakeTokenizer(),
    )
    return source, output, tensors


def test_full_export_prefixes_bare_keys_without_changing_values(
    tmp_path, monkeypatch
):
    source, output, tensors = make_full_fixture(tmp_path, monkeypatch)
    manifest = dx.export_kev_full(source, output, "jaredpalmer/kev-27b", "main1")

    assert manifest["prompt_protocol"] == "kev_pointer_v1"
    assert manifest["temperature"] == 1.3195
    assert manifest["hidden_size"] == 3
    index = json.loads((output / "model.safetensors.index.json").read_text())
    assert set(index["weight_map"]) == {"model." + name for name in tensors}
    for name, expected in tensors.items():
        shard = index["weight_map"]["model." + name]
        with safe_open(output / shard, framework="pt") as file:
            actual = file.get_tensor("model." + name)
        assert actual.dtype == expected.dtype
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_full_export_writes_a_native_text_config(tmp_path, monkeypatch):
    source, output, _ = make_full_fixture(tmp_path, monkeypatch)
    dx.export_kev_full(source, output, "jaredpalmer/kev-27b", "main1")
    config = json.loads((output / "config.json").read_text())
    assert config["architectures"] == ["Qwen3_5ForCausalLM"]
    assert config["model_type"] == "qwen3_5_text"
    # The pointer head reads hidden states; no LM head or MTP layer is loaded.
    assert config["tie_word_embeddings"] is True
    assert "mtp_num_hidden_layers" not in config
    assert dx.verify_decision(output, full=True)["protocol"] == "kev_pointer_v1"


def test_full_export_never_modifies_the_source(tmp_path, monkeypatch):
    from vllm_jev.checkpoint import sha256

    source, output, _ = make_full_fixture(tmp_path, monkeypatch)
    shards = sorted(source.glob("model-*.safetensors"))
    before = {path.name: sha256(path) for path in shards}
    dx.export_kev_full(source, output, "jaredpalmer/kev-27b", "main1")
    assert {path.name: sha256(path) for path in shards} == before
    for path in shards:
        assert not (output / path.name).samefile(path)


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("weights", "lora", "full-weight Kev"),
        ("option_isolation", True, "option-isolated"),
        ("special_embeddings", True, "delimiter embeddings"),
    ],
)
def test_full_export_rejects_unsupported_heads(
    tmp_path, monkeypatch, field, value, match
):
    source, output, _ = make_full_fixture(tmp_path, monkeypatch)
    meta = torch.load(source / "head.pt", weights_only=True)
    meta[field] = value
    torch.save(meta, source / "head.pt")
    with pytest.raises(ValueError, match=match):
        dx.export_kev_full(source, output, "jaredpalmer/kev-27b", "main1")


def test_full_export_rejects_an_adapter_release(tmp_path, monkeypatch):
    source, output, _ = make_full_fixture(tmp_path, monkeypatch)
    (source / "adapter_config.json").write_text("{}")
    with pytest.raises(ValueError, match="adapter"):
        dx.export_kev_full(source, output, "jaredpalmer/kev-27b", "main1")


def _stub_hub(monkeypatch, native_models, source, files, sha):
    calls = {"info": [], "file": [], "snapshot": []}

    class FakeApi:
        def model_info(self, repo_id, **kwargs):
            calls["info"].append((repo_id, kwargs.get("revision")))
            return SimpleNamespace(
                sha=sha, siblings=[SimpleNamespace(rfilename=n) for n in files]
            )

    def file_download(**kwargs):
        calls["file"].append(kwargs)
        return str(source / kwargs["filename"])

    def snapshot(**kwargs):
        calls["snapshot"].append(kwargs)
        return str(source)

    monkeypatch.setattr(native_models, "HfApi", lambda: FakeApi())
    monkeypatch.setattr(native_models, "hf_hub_download", file_download)
    monkeypatch.setattr(native_models, "snapshot_download", snapshot)
    return calls


def test_kev_27b_main_dispatches_to_full_export_without_a_base(
    tmp_path, monkeypatch
):
    from fnmatch import fnmatch

    from vllm_jev import native_models

    source, _, _ = make_full_fixture(tmp_path, monkeypatch)
    pinned = native_models.KEV_PINNED_REVISIONS["jaredpalmer/kev-27b"]
    files = [p.name for p in source.iterdir()] + ["README.md", "interpolation.json"]
    calls = _stub_hub(monkeypatch, native_models, source, files, pinned)
    workspace = tmp_path / "ws"
    path = native_models.prepare("jaredpalmer/kev-27b", workspace, "auto")

    assert calls["info"] == [("jaredpalmer/kev-27b", pinned)]
    assert [c["filename"] for c in calls["file"]] == ["head.pt"]
    assert [c["repo_id"] for c in calls["snapshot"]] == ["jaredpalmer/kev-27b"]
    snapshot = calls["snapshot"][0]
    assert snapshot["revision"] == pinned
    # One directory per revision, so files from another revision never mix in.
    expected_dir = workspace.resolve() / "public" / "jaredpalmer/kev-27b" / pinned
    assert Path(snapshot["local_dir"]) == expected_dir
    wanted = {
        name
        for name in files
        if any(fnmatch(name, pattern) for pattern in snapshot["allow_patterns"])
    }
    assert wanted == {
        "config.json",
        "head.pt",
        "model.safetensors.index.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
    }
    manifest = json.loads((path / "decision_manifest.json").read_text())
    assert manifest["source_revision"] == pinned
    assert manifest["prompt_protocol"] == "kev_pointer_v1"


def test_full_dispatch_rejects_a_foreign_head_before_downloading_shards(
    tmp_path, monkeypatch
):
    from vllm_jev import native_models

    source, _, _ = make_full_fixture(tmp_path, monkeypatch)
    meta = torch.load(source / "head.pt", weights_only=True)
    meta.pop("weights")
    torch.save(meta, source / "head.pt")
    files = [p.name for p in source.iterdir()]
    calls = _stub_hub(monkeypatch, native_models, source, files, "abc")
    with pytest.raises(ValueError, match="full-weight Kev"):
        native_models.prepare("someone/sharded-with-head", tmp_path / "ws", "auto")
    assert calls["snapshot"] == []


def test_adapter_release_with_config_and_index_still_uses_the_lora_export(
    tmp_path, monkeypatch
):
    from vllm_jev import native_models

    files = [
        "head.pt",
        "adapter_config.json",
        "adapter_model.safetensors",
        "config.json",
        "model.safetensors.index.json",
    ]
    calls = _stub_hub(monkeypatch, native_models, tmp_path, files, "abc")

    def stop(**kwargs):
        calls["snapshot"].append(kwargs)
        raise AssertionError("reached download")

    monkeypatch.setattr(native_models, "snapshot_download", stop)
    with pytest.raises(AssertionError, match="reached download"):
        native_models.prepare("someone/kev-lora", tmp_path / "ws", "auto")
    assert calls["file"] == []  # the full-weight head pre-check did not run
    assert "adapter_model.safetensors" in calls["snapshot"][0]["allow_patterns"]


def test_cached_kev_27b_from_another_revision_is_rejected(tmp_path, monkeypatch):
    from vllm_jev import native_models

    source, output, _ = make_full_fixture(tmp_path, monkeypatch)
    workspace = tmp_path / "ws"
    cached = workspace / "checkpoint" / "jaredpalmer/kev-27b"
    cached.parent.mkdir(parents=True)
    dx.export_kev_full(source, cached, "jaredpalmer/kev-27b", "01b81998old")
    monkeypatch.setattr(
        native_models, "HfApi", lambda: pytest.fail("unexpected Hub call")
    )
    with pytest.raises(ValueError, match="is pinned"):
        native_models.prepare("jaredpalmer/kev-27b", workspace, "auto")


def test_full_export_rejects_a_shard_that_disagrees_with_its_index(
    tmp_path, monkeypatch
):
    source, output, _ = make_full_fixture(tmp_path, monkeypatch)
    index = json.loads((source / "model.safetensors.index.json").read_text())
    moved = next(iter(index["weight_map"]))
    shards = sorted(set(index["weight_map"].values()))
    index["weight_map"][moved] = next(
        shard for shard in shards if shard != index["weight_map"][moved]
    )
    (source / "model.safetensors.index.json").write_text(json.dumps(index))
    with pytest.raises(ValueError, match="disagrees with index"):
        dx.export_kev_full(source, output, "jaredpalmer/kev-27b", "main1")


@pytest.mark.parametrize(
    "name", ["model.layers.0.extra.weight", "lm_head.weight", "mtp.fc.weight"]
)
def test_full_export_rejects_non_bare_keys(tmp_path, monkeypatch, name):
    source, output, _ = make_full_fixture(tmp_path, monkeypatch)
    shard = "model-00002-of-00002.safetensors"
    tensors = load_file(str(source / shard))
    tensors[name] = torch.zeros(2)
    save_file(tensors, str(source / shard))
    index = json.loads((source / "model.safetensors.index.json").read_text())
    index["weight_map"][name] = shard
    (source / "model.safetensors.index.json").write_text(json.dumps(index))
    with pytest.raises(ValueError, match="bare text-model keys"):
        dx.export_kev_full(source, output, "jaredpalmer/kev-27b", "main1")
