"""Synthetic Drex exporter/compiler/readout checks; no 35B weight downloads."""

import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from vllm_jev import drex, drex_export
from vllm_jev.decision_template import ENVIRONMENT, current_template


class Tokens:
    unk_token_id = None

    def encode(self, text, **kwargs):
        return [8 + ord(c) % 100 for c in text]

    def convert_tokens_to_ids(self, token):
        return drex_export.MARKERS.index(token) + 1


@pytest.fixture(autouse=True)
def native_template(monkeypatch):
    monkeypatch.delenv(ENVIRONMENT, raising=False)
    current_template.cache_clear()
    yield
    current_template.cache_clear()


@pytest.fixture
def source(tmp_path, monkeypatch):
    path = tmp_path / "source"
    path.mkdir()
    config = {
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "model_type": "qwen3_5_moe",
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "layer_types": ["linear_attention", "full_attention"],
            "hidden_size": 16,
            "vocab_size": 128,
            "max_position_embeddings": 256,
        },
    }
    (path / "config.json").write_text(json.dumps(config))
    save_file(
        {"model.language_model.embed_tokens.weight": torch.randn(128, 16)},
        str(path / "model.safetensors"),
    )
    (path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.language_model.embed_tokens.weight": "model.safetensors"
                }
            }
        )
    )
    torch.save(
        {
            "head_dim": 8,
            "state_bidir": False,
            "temperature": 1.3,
            "lora": 0,
            "head": {
                "q.weight": torch.randn(8, 16),
                "k.weight": torch.randn(8, 16),
                "q.bias": torch.randn(8),
                "k.bias": torch.randn(8),
            },
        },
        path / "head.pt",
    )
    (path / "tokenizer.json").write_text("{}")
    (path / "tokenizer_config.json").write_text("{}")
    monkeypatch.setattr(
        drex_export.AutoTokenizer, "from_pretrained", lambda *a, **kw: Tokens()
    )
    return path


@pytest.fixture
def exported(source, tmp_path):
    output = tmp_path / "export"
    drex_export.export_drex(source, output, "test/drex", "fixture")
    return output


def payload(**kwargs):
    values = dict(
        state={"ticket": "Refund"},
        model=None,
        cache_salt=None,
        images=None,
        videos=None,
        media=None,
        questions={
            "q": {
                "type": "choice",
                "instructions": "Choose",
                "criteria": {"a": "yes", "b": "no"},
            }
        },
    )
    return SimpleNamespace(**(values | kwargs))


def test_export_preserves_shards_and_source(source, exported):
    import os

    before = json.loads((source / "config.json").read_text())
    assert os.path.samefile(
        source / "model.safetensors", exported / "model.safetensors"
    )
    config = json.loads((exported / "config.json").read_text())
    assert config["architectures"] == [drex_export.ARCHITECTURE]
    assert before["architectures"] == ["Qwen3_5MoeForConditionalGeneration"]
    assert drex_export.verify_drex(exported)["full"]
    (exported / "tokenizer.json").write_text('{"tampered":true}')
    with pytest.raises(ValueError, match="checksum"):
        drex_export.verify_drex(exported, full=False)


@pytest.mark.parametrize(
    "field,value",
    [
        ("state_bidir", True),
        ("state_bidir", None),
        ("option_isolation", True),
        ("lora", 8),
        ("temperature", 0),
        ("temperature", float("nan")),
    ],
)
def test_unsupported_heads_rejected_before_export(source, tmp_path, field, value):
    meta = torch.load(source / "head.pt", weights_only=True)
    meta[field] = value
    torch.save(meta, source / "head.pt")
    with pytest.raises(ValueError):
        drex_export.export_drex(source, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_shard_traversal_is_rejected(source, tmp_path):
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": "../outside.safetensors"}})
    )
    with pytest.raises(ValueError, match="local"):
        drex_export.export_drex(source, tmp_path / "bad")


def test_copy_fallback_checks_space(monkeypatch, tmp_path):
    item = tmp_path / "weight"
    item.write_bytes(b"x")
    monkeypatch.setattr(
        drex_export.os,
        "link",
        lambda *a: (_ for _ in ()).throw(OSError("cross-device")),
    )
    monkeypatch.setattr(
        drex_export.shutil, "disk_usage", lambda *a: SimpleNamespace(free=0)
    )
    with pytest.raises(OSError, match="headroom"):
        drex_export._link_or_copy(item, tmp_path / "copy")
    assert not (tmp_path / "copy").exists()


def test_compiler_limits_control_tokens_and_single_level(exported):
    compiler = drex.DrexCompiler(exported, ["test/drex"], 256)
    p = payload(
        state="<|fim_prefix|>",
        questions={
            "one": {"type": "score", "criteria": ["only"]},
            "two": {"type": "noul"},
        },
    )
    rows, state_tokens = compiler.compile(p)
    assert len(rows) == 2 and rows[0][2] == ["0"] and rows[1][2] == ["false", "true"]
    assert all(1 not in row[-1][0][0][1:] for row in rows)
    exact = len(rows[1][-1][0][0])
    compiler.max_length = exact
    assert compiler.compile(payload(state=p.state, questions={"two": {"type": "noul"}}))
    compiler.max_length = exact - 1
    with pytest.raises(ValueError, match="not truncated"):
        compiler.compile(payload(state=p.state, questions={"two": {"type": "noul"}}))
    assert state_tokens > 1


@pytest.mark.parametrize(
    "updates",
    [
        {"cache_salt": "shared"},
        {"images": ["x"]},
        {"model": "wrong"},
        {"state": float("nan")},
        {"questions": {}},
        {"questions": {"q": {"type": "bad"}}},
    ],
)
def test_invalid_inputs_rejected(exported, updates):
    with pytest.raises(ValueError):
        drex.DrexCompiler(exported, ["test/drex"], 256).compile(payload(**updates))


def test_drex_confidence_matches_published_formula():
    p = [0.1, 0.2, 0.7]
    result = drex.answer("score", ["0", "1", "2"], ["low", "mid", "high"], p)
    assert result["score"] == 1.6 and result["confidence"] == 0.8
    assert result["legend"] == ["low", "mid", "high"]
    assert drex.answer("score", ["0"], ["only"], [1.0])["confidence"] == 1.0
    assert (
        drex.answer("noul", ["false", "true"], [], [0.123456, 0.876544])["noul"]
        == 0.8765
    )


def test_automatic_download_rejects_low_space_before_network(monkeypatch, tmp_path):
    import huggingface_hub

    monkeypatch.setattr(
        drex_export.shutil, "disk_usage", lambda *a: SimpleNamespace(free=0)
    )
    monkeypatch.setattr(
        huggingface_hub, "snapshot_download", lambda **kw: pytest.fail("network used")
    )
    with pytest.raises(OSError, match="headroom"):
        drex_export.prepare_drex(drex_export.MODEL_ID, tmp_path)


def test_service_pointer_math(exported):
    service = drex.DrexService.__new__(drex.DrexService)
    service.head = drex.load_file(str(exported / drex_export.HEAD_FILE))
    service.compiler = SimpleNamespace(manifest={"temperature": 1.3})
    hidden = torch.randn(6, 16)
    positions = [1, 3, 5]
    q = torch.nn.functional.linear(
        hidden[5], service.head["q.weight"], service.head["q.bias"]
    )
    k = torch.nn.functional.linear(
        hidden[[1, 3]], service.head["k.weight"], service.head["k.bias"]
    )
    expected = torch.softmax((k @ q) * (1 / 8**0.5) / 1.3, -1).tolist()
    assert service._probabilities(hidden, positions, 2, "choice") == expected


def test_linux_cli_uses_native_moe_and_uncached_pooling(monkeypatch, exported):
    import vllm_jev.laya_export  # noqa: F401
    from vllm_jev import cli

    observed = {}
    monkeypatch.setattr(cli.sys, "platform", "linux")
    monkeypatch.setattr(
        cli.sys, "argv", ["vllm-jev", "serve", str(exported), "--max-model-len", "129"]
    )
    monkeypatch.setattr(
        cli.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=0)
    )
    monkeypatch.setattr(
        cli.os,
        "execvpe",
        lambda exe, command, env: observed.update(command=command, env=env),
    )
    cli.main()
    command = observed["command"]
    assert command[command.index("--convert") + 1] == "none"
    assert (
        "--no-enable-prefix-caching" in command
        and "--enable-prefix-caching" not in command
    )
    assert command[command.index("--served-model-name") + 1] == "test/drex"
    assert command[command.index("--pooler-config") + 1] == '{"task":"token_embed"}'
    assert command[-2:] == ["--max-model-len", "129"]


@pytest.mark.parametrize(
    "flags",
    [
        ["--enable-prefix-caching"],
        ["--radixflow", "true"],
        ["--decision-template", "instructions-first"],
    ],
)
def test_cli_rejects_unsupported_drex_modes(monkeypatch, exported, flags):
    import vllm_jev.laya_export  # noqa: F401
    from vllm_jev import cli

    monkeypatch.setattr(cli.sys, "platform", "linux")
    monkeypatch.setattr(cli.sys, "argv", ["vllm-jev", "serve", str(exported), *flags])
    monkeypatch.setattr(
        cli.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=0)
    )
    monkeypatch.setattr(
        cli.os, "execvpe", lambda *a: pytest.fail("unsupported server started")
    )
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2


def test_lfs_pointer_does_not_bypass_disk_guard(monkeypatch, tmp_path):
    import huggingface_hub

    from vllm_jev import drex_release

    path = tmp_path / "public" / drex_export.MODEL_ID
    path.mkdir(parents=True)
    (path / "weight.safetensors").write_bytes(b"LFS pointer")
    monkeypatch.setattr(
        drex_release, "FILES", {"weight.safetensors": {"size": 42, "sha256": "0" * 64}}
    )
    monkeypatch.setattr(
        drex_export.shutil,
        "disk_usage",
        lambda *a: SimpleNamespace(free=drex_export.MIN_FREE_BYTES + 41),
    )
    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        lambda **kw: pytest.fail("insufficient space download"),
    )
    with pytest.raises(OSError, match="headroom"):
        drex_export.prepare_drex(drex_export.MODEL_ID, tmp_path)
