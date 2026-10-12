"""CPU-only tests for the JevAny Pointer adapter."""

import asyncio
import base64
import io
import json
import sys
from itertools import permutations
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from PIL import Image
from safetensors import safe_open
from safetensors.torch import save_file

try:
    import vllm  # noqa: F401
except ImportError:
    fake_vllm = ModuleType("vllm")

    class PoolingParams:
        def __init__(self, **kwargs):
            vars(self).update(kwargs)

    fake_vllm.PoolingParams = PoolingParams
    sys.modules["vllm"] = fake_vllm

from vllm_jev import (
    JEVANY_POINTER_QWEN3_VL_MOE_ARCHITECTURE,
    JEVANY_POINTER_QWEN35_ARCHITECTURE,
)
from vllm_jev import jevany_export as jx
from vllm_jev import jevany_pointer as jp

MARKER_NAMES = [
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|box_start|>",
    "<|box_end|>",
    "<|fim_suffix|>",
]
MARKERS = [100, 101, 102, 103, 104]
IMAGE = 200
VISION_START = 201
VISION_END = 202


def data_url():
    buffer = io.BytesIO()
    Image.new("RGB", (64, 48), (100, 20, 30)).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


class FakeTokenizer:
    def __init__(self):
        self.by_name = dict(zip(MARKER_NAMES, MARKERS))
        self.by_name["<image>"] = IMAGE
        self.by_name["<vision>"] = VISION_START
        self.by_name["</vision>"] = VISION_END
        self.by_id = {value: key for key, value in self.by_name.items()}
        self.all_special_tokens = list(self.by_name)

    def convert_tokens_to_ids(self, value):
        return self.by_name.get(value)

    def convert_ids_to_tokens(self, value):
        return self.by_id[value]

    def encode(self, text, add_special_tokens=False):
        ids = []
        while text:
            token = next((item for item in self.by_name if text.startswith(item)), None)
            if token:
                ids.append(self.by_name[token])
                text = text[len(token) :]
            else:
                ids.append(300 + ord(text[0]))
                text = text[1:]
        return ids

    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        ids = self.encode(text, add_special_tokens)
        if return_tensors is not None:
            return {"input_ids": torch.tensor([ids])}
        return SimpleNamespace(input_ids=ids)


class FakeProcessor:
    image_token = "<image>"
    vision_start_token = "<vision>"
    vision_end_token = "</vision>"

    def __init__(self):
        self.tokenizer = FakeTokenizer()

    def __call__(self, *, text, images, return_tensors, size):
        assert len(images) in (1, 2)
        ids = self.tokenizer.encode(text[0])
        expanded = []
        for token in ids:
            expanded += [token] * (3 if token == IMAGE else 1)
        return {"input_ids": torch.tensor([expanded])}


class FakeEngine:
    def __init__(self, hidden):
        self.hidden = hidden
        self.requests = []

    async def encode(self, prompt, pooling_params, request_id):
        self.requests.append((prompt, pooling_params))
        ids = []
        for token in prompt["prompt_token_ids"]:
            ids += [token] * (3 if token == IMAGE else 1)
        generator = torch.Generator().manual_seed(len(ids))
        states = torch.randn(len(ids), self.hidden, generator=generator).bfloat16()
        yield SimpleNamespace(
            finished=True,
            prompt_token_ids=ids,
            outputs=SimpleNamespace(data=states),
        )

    async def abort(self, request_id):
        pass


def make_service_dir(path: Path, hidden=8):
    (path / "jevany_pointer_manifest.json").write_text(
        json.dumps(
            {
                "hidden_size": hidden,
                "max_length": 4608,
                "max_state": 4096,
                "max_images": 2,
                "image_token_id": IMAGE,
                "markers": MARKERS,
                "temperature": 1.0,
            }
        )
    )
    torch.manual_seed(0)
    save_file(
        {
            "q.weight": torch.randn(4, hidden),
            "q.bias": torch.randn(4),
            "k.weight": torch.randn(4, hidden),
            "k.bias": torch.randn(4),
        },
        str(path / jx.HEAD_FILE),
    )


def test_pointer_readout_is_fp32():
    torch.manual_seed(1)
    hidden = torch.randn(7, 8).bfloat16()
    head = {
        "q.weight": torch.randn(4, 8),
        "q.bias": torch.randn(4),
        "k.weight": torch.randn(4, 8),
        "k.bias": torch.randn(4),
        "residual.0.weight": torch.randn(3, 4),
        "residual.0.bias": torch.randn(3),
        "residual.2.weight": torch.randn(1, 3),
        "residual.2.bias": torch.randn(1),
    }
    actual = jp.pointer_probabilities(head, hidden, [2, 4], 6, 1.0)
    query = torch.nn.functional.linear(
        hidden[6].float(), head["q.weight"], head["q.bias"]
    )
    keys = torch.nn.functional.linear(
        hidden[[2, 4]].float(), head["k.weight"], head["k.bias"]
    )
    residual = torch.nn.functional.linear(
        keys * query.unsqueeze(0),
        head["residual.0.weight"],
        head["residual.0.bias"],
    )
    residual = torch.nn.functional.gelu(residual)
    residual = torch.nn.functional.linear(
        residual, head["residual.2.weight"], head["residual.2.bias"]
    )
    expected = torch.softmax(keys @ query / 2 + residual.squeeze(-1), -1).tolist()
    assert actual == pytest.approx(expected, abs=1e-7)


def test_service_preserves_expanded_tokens_and_answers(tmp_path, monkeypatch):
    make_service_dir(tmp_path)
    processor = FakeProcessor()
    monkeypatch.setattr(
        jp.AutoProcessor, "from_pretrained", lambda *args, **kwargs: processor
    )
    engine = FakeEngine(8)
    service = jp.JevAnyPointerService(engine, tmp_path, "jevany-test", 4609)
    payload = SimpleNamespace(
        model=None,
        state={"policy": "check"},
        media=[{"type": "image", "uri": data_url()}],
        questions={
            "risk": {
                "type": "noul",
                "instructions": "Infringing?",
                "criteria": {"false": "safe", "true": "risk"},
            }
        },
    )
    response = asyncio.run(service.systemone(payload))
    prompt, params = engine.requests[0]
    assert prompt["prompt_token_ids"].count(IMAGE) == 1
    assert prompt["mm_processor_kwargs"]["size"] == {
        "shortest_edge": 1024,
        "longest_edge": 262144,
    }
    assert params.task == "token_embed" and params.use_activation is False
    assert response["answers"]["risk"]["type"] == "noul"
    assert 0 <= response["answers"]["risk"]["noul"] <= 1


def test_service_accepts_text_only_requests(tmp_path, monkeypatch):
    make_service_dir(tmp_path)
    processor = FakeProcessor()
    monkeypatch.setattr(
        jp.AutoProcessor, "from_pretrained", lambda *args, **kwargs: processor
    )
    engine = FakeEngine(8)
    service = jp.JevAnyPointerService(engine, tmp_path, "jevany-test", 4609)
    payload = SimpleNamespace(
        model=None,
        state="The invoice total is 10.",
        media=[],
        questions={
            "valid": {
                "type": "noul",
                "instructions": "Is the total positive?",
            }
        },
    )
    response = asyncio.run(service.systemone(payload))
    prompt, _ = engine.requests[0]
    assert "multi_modal_data" not in prompt
    assert "mm_processor_kwargs" not in prompt
    assert response["usage"]["input_tokens"] > 0


def make_export_fixture(tmp_path, monkeypatch):
    base, source, output = tmp_path / "base", tmp_path / "source", tmp_path / "out"
    base.mkdir()
    source.mkdir()
    base_weight = torch.arange(12, dtype=torch.float32).reshape(4, 3).bfloat16()
    visual = torch.ones(2, 2).bfloat16()
    tensors = {
        "model.language_model.layers.0.self_attn.q_proj.weight": base_weight,
        "model.visual.blocks.0.weight": visual,
    }
    save_file(tensors, str(base / "model.safetensors"))
    (base / "model.safetensors.index.json").write_text(
        json.dumps(
            {"weight_map": {name: "model.safetensors" for name in tensors}}
        )
    )
    (base / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "model_type": "qwen3_5",
                "image_token_id": IMAGE,
                "text_config": {
                    "model_type": "qwen3_5_text",
                    "hidden_size": 3,
                },
            }
        )
    )
    for name in ("preprocessor_config.json", "vocab.json", "merges.txt"):
        (base / name).write_text("{}")
    (source / "tokenizer.json").write_text("{}")
    (source / "tokenizer_config.json").write_text("{}")
    (source / "processor_config.json").write_text("{}")
    (source / "chat_template.jinja").write_text("")
    left = torch.tensor([[1.0, 2.0, 3.0], [0.5, 1.0, -1.0]])
    right = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 2.0]]
    )
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
                "r": 2,
                "lora_alpha": 4,
            }
        )
    )
    torch.manual_seed(2)
    head = {
        "q.weight": torch.randn(256, 3),
        "q.bias": torch.randn(256),
        "k.weight": torch.randn(256, 3),
        "k.bias": torch.randn(256),
        "residual.0.weight": torch.randn(5, 256),
        "residual.0.bias": torch.randn(5),
        "residual.2.weight": torch.randn(1, 5),
        "residual.2.bias": torch.randn(1),
    }
    torch.save(
        {
            "base": str(base),
            "decision_mode": "pointer",
            "multimodal": True,
            "backbone_adapter": "qwen_vl",
            "branch_mode": "rows",
            "option_isolation": False,
            "special_embeddings": False,
            "head_dim": 256,
            "head_residual_dim": 5,
            "head_type": "residual",
            "query_readout": "decide",
            "option_readout": "close",
            "head": head,
            "temperature": 1.0,
            "args": {"max_state": 4096, "max_branch": 4608},
        },
        source / "head.pt",
    )
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: FakeTokenizer(),
    )
    return base, source, output, base_weight, left, right


def test_export_merges_lora_shard_and_verifies(tmp_path, monkeypatch):
    base, source, output, weight, left, right = make_export_fixture(
        tmp_path, monkeypatch
    )
    manifest = jx.export_jevany_pointer(base, source, output, "jevany-test")
    assert manifest["architecture"] == JEVANY_POINTER_QWEN35_ARCHITECTURE
    assert manifest["head_residual_dim"] == 5
    assert jx.verify_jevany_pointer(output)["full"] is True
    with safe_open(output / "model.safetensors", framework="pt") as file:
        merged = file.get_tensor(
            "model.language_model.layers.0.self_attn.q_proj.weight"
        )
    expected = (weight.float() + (right @ left) * 2).bfloat16()
    torch.testing.assert_close(merged, expected, rtol=0, atol=0)


def test_export_rejects_unrecognized_adapter_tensor(tmp_path, monkeypatch):
    base, source, output, *_ = make_export_fixture(tmp_path, monkeypatch)
    adapter_path = source / "adapter_model.safetensors"
    with safe_open(adapter_path, framework="pt", device="cpu") as file:
        tensors = {name: file.get_tensor(name) for name in file.keys()}
    tensors["base_model.model.unexpected.weight"] = torch.ones(1)
    save_file(tensors, adapter_path)

    with pytest.raises(ValueError, match="unsupported JevAny adapter tensors"):
        jx.export_jevany_pointer(base, source, output, "jevany-test")


def test_export_supports_qwen3_vl_moe(tmp_path, monkeypatch):
    base, source, output, *_ = make_export_fixture(tmp_path, monkeypatch)
    config = json.loads((base / "config.json").read_text())
    config.update(
        architectures=["Qwen3VLMoeForConditionalGeneration"],
        model_type="qwen3_vl_moe",
    )
    config["text_config"]["model_type"] = "qwen3_vl_moe_text"
    (base / "config.json").write_text(json.dumps(config))

    manifest = jx.export_jevany_pointer(base, source, output, "jevany-qwen3-vl")

    assert manifest["architecture"] == JEVANY_POINTER_QWEN3_VL_MOE_ARCHITECTURE
    assert jx.verify_jevany_pointer(output)["full"] is True


def test_cli_serves_exported_jevany_pointer(tmp_path, monkeypatch):
    base, source, output, *_ = make_export_fixture(tmp_path, monkeypatch)
    jx.export_jevany_pointer(base, source, output, "jevany-test")
    from vllm_jev import cli

    calls = {}
    monkeypatch.setattr(cli.subprocess, "run", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        cli.os, "execvpe", lambda executable, argv, env: calls.update(argv=argv)
    )
    monkeypatch.setattr(cli.sys, "platform", "linux")
    monkeypatch.setattr(cli.sys, "argv", ["vllm-jev", "serve", str(output)])
    cli.main()
    argv = calls["argv"]
    assert argv[argv.index("--max-model-len") + 1] == "4609"
    assert "--no-enable-prefix-caching" in argv
    assert json.loads(argv[argv.index("--mm-processor-kwargs") + 1])["size"]["longest_edge"] == 512 * 512
    assert json.loads(argv[argv.index("--limit-mm-per-prompt") + 1]) == {
        "image": 2,
        "video": 0,
    }


@pytest.mark.parametrize(
    "order", list(permutations(["state", "instructions", "criteria"]))
)
def test_context_order_keeps_pointer_readouts_late(tmp_path, monkeypatch, order):
    from vllm_jev.decision_template import DecisionTemplate

    make_service_dir(tmp_path)
    processor = FakeProcessor()
    monkeypatch.setattr(jp.AutoProcessor, "from_pretrained", lambda *a, **k: processor)
    monkeypatch.setattr(
        jp, "current_template", lambda: DecisionTemplate({"context_order": list(order)})
    )
    compiler = jp.JevAnyPointerCompiler(tmp_path, "jevany-test", 4608)
    payload = SimpleNamespace(
        model=None,
        state="evidence",
        media=[],
        questions={
            "q": {
                "type": "choice",
                "instructions": "choose",
                "criteria": {"a": "alpha", "b": "beta"},
            }
        },
    )
    result = compiler.compile(payload)
    ids, positions, decision = result[4:7]
    q = ids.index(MARKERS[1])
    prefix = "".join(chr(t - 300) for t in ids[1:q])
    labels = {"state": "Context:", "instructions": "Question:", "criteria": "Criteria:"}
    assert [prefix.index(labels[field]) for field in order] == sorted(
        prefix.index(labels[field]) for field in order
    )
    assert len(positions) == 2 and all(q < p < decision for p in positions)
    assert ids[decision] == MARKERS[4]
    assert ids.count(MARKERS[1]) == 1


@pytest.mark.parametrize("protocol", ["auto", "jevany_pointer_v1"])
def test_native_prepare_jevany_uses_pinned_export(tmp_path, monkeypatch, protocol):
    from vllm_jev import native_models

    base, source, _, *_ = make_export_fixture(tmp_path, monkeypatch)
    meta = torch.load(source / "head.pt", weights_only=True)
    meta["base_revision"] = "fixture-base-revision"
    torch.save(meta, source / "head.pt")
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return str(source if kwargs["repo_id"].startswith("SimpleJev/") else base)

    monkeypatch.setattr(native_models, "snapshot_download", download)
    monkeypatch.setattr(native_models.sys, "platform", "linux")
    model = "SimpleJev/JevAny-Qwen3.5-4B-LoRA"
    workspace = tmp_path / "workspace"
    path = native_models.prepare(model, workspace, protocol)
    manifest = json.loads((path / jx.MANIFEST_FILE).read_text())
    assert manifest["source_repository"] == model
    assert manifest["source_revision"] == jx.MODELS[model]
    assert calls[0]["revision"] == jx.MODELS[model]
    assert calls[1]["revision"] == "fixture-base-revision"
    assert not (path / "jev_manifest.json").exists()
    assert native_models.prepare(model, workspace, protocol) == path
    assert len(calls) == 2


def test_native_prepare_jevany_rejects_other_protocol_before_download(
    tmp_path, monkeypatch
):
    from vllm_jev import native_models

    monkeypatch.setattr(native_models.sys, "platform", "linux")
    monkeypatch.setattr(
        native_models,
        "snapshot_download",
        lambda **kwargs: pytest.fail("unexpected download"),
    )
    with pytest.raises(ValueError, match="uses jevany_pointer_v1"):
        native_models.prepare(
            "SimpleJev/JevAny-Qwen3.5-4B-LoRA", tmp_path, "kev_pointer_v1"
        )


def test_native_preparation_cli_accepts_jevany_protocol(tmp_path, monkeypatch, capsys):
    from vllm_jev import native_models

    calls = []
    monkeypatch.setattr(
        native_models, "prepare", lambda *args: calls.append(args) or tmp_path
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "native_models",
            "SimpleJev/JevAny-Qwen3.5-4B-LoRA",
            "--workspace",
            str(tmp_path),
            "--protocol",
            "jevany_pointer_v1",
        ],
    )
    native_models.main()
    assert calls == [
        ("SimpleJev/JevAny-Qwen3.5-4B-LoRA", tmp_path, "jevany_pointer_v1")
    ]
    assert "JEV_CHECKPOINT_READY" in capsys.readouterr().out


def test_cli_rejects_jevany_on_mac_before_preparation(tmp_path, monkeypatch, capsys):
    from vllm_jev import cli
    monkeypatch.setattr(cli.sys, 'platform', 'darwin')
    monkeypatch.setattr(cli.sys, 'argv', ['vllm-jev', 'serve', 'SimpleJev/JevAny-Qwen3.5-4B-LoRA', '--workspace', str(tmp_path)])
    monkeypatch.setattr(cli.subprocess, 'run', lambda *a, **k: pytest.fail('unexpected preparation'))
    with pytest.raises(SystemExit) as e:
        cli.main()
    assert e.value.code == 2
    assert 'requires Linux' in capsys.readouterr().err
