"""CPU tests for the Clef adapter. No GPU, no weights download.

Run with ``python -m pytest tests/test_clef.py -q``. Tests marked
``reference`` compare against the release's own ``joint_schema_model.py`` and
skip unless ``CLEF_RELEASE`` names a local Clef-Flash release directory; only
its code, tokenizer, and processor files are read.
"""

import asyncio
import base64
import importlib.util
import io
import json
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from vllm_jev import CLEF_QWEN35_ARCHITECTURE
from vllm_jev import clef as cf
from vllm_jev import clef_export as cx

SMALL_HEAD = {
    "hidden_size": 64,
    "width": 32,
    "routing_layers": 2,
    "layers": 2,
    "heads": 4,
    "feedforward": 48,
}
QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "Which team should handle the message?",
        "criteria": {"technical": "Bugs or outages", "billing": "Payments"},
    },
    "urgency": {"type": "score", "criteria": ["Can wait", "This week", "Today"]},
    "outage": {"type": "noul", "instructions": "Is a service down?"},
    "lang": {"type": "choice", "criteria": {"中文": None, "English": {"x": 1}}},
    "custom": {"type": "noul", "criteria": {"true": "Yes, clearly"}},
}


def release():
    path = os.environ.get("CLEF_RELEASE")
    if not path or not (Path(path) / "joint_schema_model.py").is_file():
        pytest.skip("CLEF_RELEASE is not set")
    spec = importlib.util.spec_from_file_location(
        "clef_release", Path(path) / "joint_schema_model.py"
    )
    if "clef_release" not in sys.modules:
        module = importlib.util.module_from_spec(spec)
        # Its dataclasses resolve annotations through sys.modules.
        sys.modules["clef_release"] = module
        spec.loader.exec_module(module)
    return Path(path), sys.modules["clef_release"]


def processor(path):
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(path, local_files_only=True)


def data_url(width=64, height=48, color=(200, 30, 30)):
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def random_head(config=SMALL_HEAD, seed=0):
    torch.manual_seed(seed)
    head = cf.ClefHead(**config)
    for parameter in head.parameters():
        parameter.data.normal_(0, 0.3)
    return head.eval()


def test_questions_follow_the_jev_limits():
    cf.validate_question("q", {"type": "choice", "criteria": {"only": None}})
    for question in (
        {"type": "pick"},
        {"type": "choice", "criteria": {}},
        {"type": "choice", "criteria": {str(i): None for i in range(256)}},
        {"type": "choice", "criteria": {"": None}},
        {"type": "score", "criteria": ["one"]},
        {"type": "score", "criteria": [str(i) for i in range(11)]},
        {"type": "noul", "criteria": {"maybe": "?"}},
        {"type": "noul", "instructions": 3},
    ):
        with pytest.raises(ValueError):
            cf.validate_question("q", question)


def test_collapse_undoes_image_and_video_expansion():
    vision = cf.VisionTokens(image_pad=1, video_pad=2, vision_start=3, vision_end=4)
    image = [9, 3, 1, 1, 1, 4, 9]
    assert cf.collapse_media(image, vision) == [9, 3, 1, 4, 9]
    # Qwen3-VL video: timestamps and one nested group per temporal patch.
    video = [9, 3, 7, 3, 2, 2, 4, 8, 3, 2, 2, 4, 4, 9]
    assert cf.collapse_media(video, vision) == [9, 3, 2, 4, 9]
    assert cf.collapse_media([5, 6], vision) == [5, 6]


@pytest.mark.reference
def test_answers_equal_the_release():
    _, module = release()
    probabilities = {
        "department": {"billing": 0.123456, "technical": 0.876544},
        "urgency": {"0": 0.1, "1": 0.25, "2": 0.65},
        "outage": {"true": 0.712345, "false": 0.287655},
    }
    for key, values in probabilities.items():
        assert cf.answer(QUESTIONS[key], values) == module.systemone_answer(
            QUESTIONS[key], values
        )


@pytest.mark.reference
@pytest.mark.parametrize(
    "state",
    [
        "Our checkout started returning errors and orders are blocked.",
        {"invoice": {"vendor": "Acme", "total": 1250.0, "status": "overdue"}},
        "用户说：我昨天买的手机屏幕碎了，要求退货退款。",
        ["a", 1, None],
        "word " * 3000,
    ],
)
def test_encoding_equals_the_release(state):
    path, module = release()
    tokenizer = processor(path).tokenizer
    for max_length in (16384, 900):
        record = {"state": state, "questions": QUESTIONS}
        expected = module.encode_record(tokenizer, record, max_length=max_length)
        ids, questions = cf.encode(tokenizer, QUESTIONS, state, [], max_length)
        assert tuple(ids) == expected.input_ids
        assert [vars(q) for q in questions] == [vars(q) for q in expected.questions]


@pytest.mark.reference
def test_image_encoding_equals_the_release(monkeypatch):
    path, module = release()
    clef_processor = processor(path)
    images = [Image.new("RGB", (512, 512), "red"), Image.new("RGB", (300, 200))]
    record = {"state": "Two pictures.", "images": images, "questions": QUESTIONS}
    expected = module.encode_record(
        clef_processor.tokenizer, record, processor=clef_processor
    )
    service = cf.ClefService.__new__(cf.ClefService)
    service.processor = clef_processor
    urls = []
    for image in images:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        urls.append(
            "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
        )
    media_ids, data = service._media(SimpleNamespace(images=urls, videos=None))
    ids, questions = cf.encode(
        clef_processor.tokenizer, QUESTIONS, record["state"], media_ids, 16384
    )
    assert tuple(ids) == expected.input_ids
    assert [vars(q) for q in questions] == [vars(q) for q in expected.questions]
    assert len(data["image"]) == 2


@pytest.mark.reference
def test_head_equals_the_release_head():
    _, module = release()
    head = random_head()
    reference = module.JointSchemaHead(**SMALL_HEAD).eval()
    reference.load_state_dict(head.state_dict(), strict=True)
    tokenizer_ids = list(range(10, 90))
    vocabulary = torch.randn(100, SMALL_HEAD["hidden_size"])
    questions = (
        module.EncodedQuestion(
            "a", 1, (3, 9), ((10, 14), (14, 15), (16, 22)), ("x", "y", "z")
        ),
        module.EncodedQuestion(
            "b", 0, (30, 31), ((40, 45), (45, 50)), ("true", "false")
        ),
        module.EncodedQuestion("c", 2, (60, 62), ((62, 64), (64, 66)), ("0", "1")),
    )
    record = module.EncodedRecord(tuple(tokenizer_ids), questions, "r")
    hidden = torch.randn(len(tokenizer_ids), SMALL_HEAD["hidden_size"])
    input_ids = torch.tensor([tokenizer_ids])
    with torch.inference_mode():
        expected = reference(
            hidden.unsqueeze(0),
            input_ids,
            torch.ones_like(input_ids),
            [record],
            vocabulary,
        )[0]
    lexical = [
        torch.stack([vocabulary[tokenizer_ids[a:b]].mean(0) for a, b in q.option_spans])
        for q in questions
    ]
    ours = head(hidden, [cf.EncodedQuestion(**vars(q)) for q in questions], lexical)
    for mine, theirs in zip(ours, expected):
        torch.testing.assert_close(mine, theirs, rtol=0, atol=0)


class FakeEngine:
    def __init__(self, hidden_size, image_tokens=0, image_pad=None):
        self.hidden_size = hidden_size
        self.image_tokens = image_tokens
        self.image_pad = image_pad
        self.requests = []

    def states(self, ids):
        generator = torch.Generator().manual_seed(len(ids))
        return torch.randn(len(ids), self.hidden_size, generator=generator).bfloat16()

    async def encode(self, prompt, pooling_params, request_id):
        self.requests.append((prompt, pooling_params))
        ids = []
        for token in prompt["prompt_token_ids"]:
            ids += [token] * (self.image_tokens if token == self.image_pad else 1)
        yield SimpleNamespace(
            finished=True,
            prompt_token_ids=ids,
            outputs=SimpleNamespace(data=self.states(ids)),
        )

    async def abort(self, request_id):
        pass


def make_model_dir(tmp_path, source, head):
    from safetensors.torch import save_file

    for name in ("tokenizer.json", "tokenizer_config.json", "processor_config.json"):
        shutil.copy2(source / name, tmp_path / name)
    (tmp_path / "joint_head_config.json").write_text(json.dumps(SMALL_HEAD))
    save_file(
        {k: v.contiguous() for k, v in head.state_dict().items()},
        str(tmp_path / "joint_head.safetensors"),
    )
    torch.manual_seed(1)
    embedding = torch.randn(248320, SMALL_HEAD["hidden_size"]).bfloat16()
    save_file({"lm_head.weight": embedding}, str(tmp_path / "lm.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"lm_head.weight": "lm.safetensors"}})
    )
    return embedding


@pytest.mark.reference
def test_service_answers_like_the_release(tmp_path):
    source, module = release()
    head = random_head()
    embedding = make_model_dir(tmp_path, source, head)
    engine = FakeEngine(SMALL_HEAD["hidden_size"])
    service = cf.ClefService(engine, tmp_path, "Cloudflare/clef-flash", 16384)
    payload = SimpleNamespace(
        model=None,
        state={"ticket": "Checkout errors since 9am."},
        questions=QUESTIONS,
        images=None,
        videos=None,
    )
    response = asyncio.run(service.systemone(payload))

    reference = module.JointSchemaHead(**SMALL_HEAD).bfloat16().eval()
    reference.load_state_dict(head.state_dict(), strict=True)
    record = {"model": "m", "state": payload.state, "questions": QUESTIONS}
    encoded = module.encode_record(processor(tmp_path).tokenizer, record)
    batch = module.collate_records([encoded], 0, torch.device("cpu"))
    with torch.inference_mode():
        logits = reference(
            engine.states(list(encoded.input_ids)).unsqueeze(0),
            batch["input_ids"],
            batch["attention_mask"],
            batch["records"],
            embedding,
        )[0]
    expected = {
        q.question_id: module.systemone_answer(
            QUESTIONS[q.question_id],
            dict(zip(q.option_ids, row.float().softmax(-1).tolist())),
        )
        for q, row in zip(encoded.questions, logits)
    }
    assert response["answers"].keys() == expected.keys()
    # The reference and adapter use equivalent heads, but their bf16 execution
    # paths can differ slightly in numerical reduction order.
    for key, target in expected.items():
        actual = response["answers"][key]
        assert actual.keys() == target.keys()
        for field, value in target.items():
            if field == "probabilities":
                assert actual[field] == pytest.approx(value, abs=0.01)
            elif isinstance(value, float):
                assert actual[field] == pytest.approx(value, abs=0.01)
            else:
                assert actual[field] == value
    assert response["usage"] == {
        "input_tokens": len(encoded.input_ids),
        "output_tokens": 0,
    }
    prompt, params = engine.requests[0]
    assert prompt["prompt_token_ids"] == list(encoded.input_ids)
    assert params.task == "token_embed" and params.use_activation is False


@pytest.mark.reference
def test_service_sends_collapsed_images(tmp_path):
    source, _ = release()
    make_model_dir(tmp_path, source, random_head())
    clef_processor = processor(tmp_path)
    pad = clef_processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    engine = FakeEngine(SMALL_HEAD["hidden_size"], image_tokens=64, image_pad=pad)
    service = cf.ClefService(engine, tmp_path, "Cloudflare/clef-flash", 16384)
    payload = SimpleNamespace(
        model=None,
        state="Describe.",
        questions={"q": {"type": "choice", "criteria": {"red": None, "blue": None}}},
        images=[data_url(256, 256)],
        videos=None,
    )
    response = asyncio.run(service.systemone(payload))
    prompt, _ = engine.requests[0]
    assert prompt["prompt_token_ids"].count(pad) == 1
    assert isinstance(prompt["multi_modal_data"]["image"], Image.Image)
    assert response["usage"]["input_tokens"] > 64
    for bad, message in (
        (dict(images=[data_url()] * 9), "at most 8 images"),
        (dict(images=["https://example.com/a.png"]), "data URLs"),
        (dict(state="<|image_pad|>"), "reserved"),
        (dict(model="other"), "not loaded"),
    ):
        with pytest.raises(ValueError, match=message):
            asyncio.run(service.systemone(SimpleNamespace(**{**vars(payload), **bad})))


def test_large_option_grid_is_rejected_before_tokenization(monkeypatch):
    service = cf.ClefService.__new__(cf.ClefService)
    service.model_names = ["Cloudflare/clef-flash"]
    payload = SimpleNamespace(
        model=None,
        questions={
            f"q{i}": {
                "type": "choice",
                "criteria": {str(index): None for index in range(255)},
            }
            for i in range(64)
        },
    )
    monkeypatch.setattr(cf, "encode", lambda *a, **k: pytest.fail("tokenizer ran"))
    with pytest.raises(ValueError, match="at most 2048 options per request"):
        service._compile(payload)


def make_release(tmp_path, monkeypatch):
    from safetensors.torch import save_file

    source = tmp_path / "release"
    source.mkdir()
    save_file(
        {"model.language_model.norm.weight": torch.ones(4)},
        str(source / "a.safetensors"),
    )
    save_file({"lm_head.weight": torch.ones(8, 4)}, str(source / "b.safetensors"))
    (source / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.language_model.norm.weight": "a.safetensors",
                    "lm_head.weight": "b.safetensors",
                }
            }
        )
    )
    (source / "config.json").write_text(
        json.dumps({"architectures": ["Qwen3_5ForConditionalGeneration"]})
    )
    (source / "joint_head_config.json").write_text(json.dumps(SMALL_HEAD))
    save_file(
        {k: v.contiguous() for k, v in random_head().state_dict().items()},
        str(source / "joint_head.safetensors"),
    )
    for name in cx.COPIED_FILES[2:]:
        (source / name).write_text(name)
    model_id = "Cloudflare/clef-test"
    monkeypatch.setattr(cx, "HEAD_CONFIG", SMALL_HEAD)
    monkeypatch.setitem(cx.MODELS, model_id, "0" * 40)
    monkeypatch.setitem(
        cx.RELEASE_HASHES,
        model_id,
        {
            name: cx.sha256(source / name)
            for name in (
                "a.safetensors",
                "b.safetensors",
                "model.safetensors.index.json",
                "config.json",
                *cx.COPIED_FILES,
            )
        },
    )
    return source, model_id


def test_export_links_the_release_and_verifies(tmp_path, monkeypatch):
    source, model_id = make_release(tmp_path, monkeypatch)
    output = tmp_path / "out"
    manifest = cx.export_clef(source, output, model_id)
    assert manifest["max_length"] == 16384
    assert cx.verify_clef(output)["architecture"] == CLEF_QWEN35_ARCHITECTURE
    assert (output / "a.safetensors").stat().st_ino == (
        source / "a.safetensors"
    ).stat().st_ino
    config = json.loads((output / "config.json").read_text())
    assert config["architectures"] == [CLEF_QWEN35_ARCHITECTURE]
    (output / "joint_head_config.json").write_text(
        json.dumps(dict(SMALL_HEAD, layers=1))
    )
    with pytest.raises(ValueError, match="configuration"):
        cx.verify_clef(output)


def test_export_refuses_a_changed_release(tmp_path, monkeypatch):
    source, model_id = make_release(tmp_path, monkeypatch)
    (source / "tokenizer.json").write_text("changed")
    with pytest.raises(ValueError, match="checksum"):
        cx.export_clef(source, tmp_path / "out", model_id)


def test_cli_serves_a_prepared_clef_checkpoint(tmp_path, monkeypatch):
    from vllm_jev import cli

    source, model_id = make_release(tmp_path, monkeypatch)
    output = tmp_path / "out"
    cx.export_clef(source, output, model_id)
    calls = {}
    monkeypatch.setattr(cli.subprocess, "run", lambda *a, **k: None)
    monkeypatch.setattr(
        cli.os, "execvpe", lambda exe, argv, env: calls.update(argv=argv)
    )
    monkeypatch.setattr(cli.sys, "platform", "linux")
    monkeypatch.setattr(cli.sys, "argv", ["vllm-jev", "serve", str(output)])
    cli.main()
    argv = calls["argv"]
    assert argv[argv.index("--max-model-len") + 1] == "16385"
    assert argv[argv.index("--served-model-name") + 1] == model_id
    assert json.loads(argv[argv.index("--pooler-config") + 1]) == {
        "task": "token_embed"
    }
    assert json.loads(argv[argv.index("--limit-mm-per-prompt") + 1]) == {
        "image": 8,
        "video": 1,
    }
    assert "--no-enable-prefix-caching" in argv
    assert "--enable-prefix-caching" not in argv
    assert "--mamba-cache-mode" not in argv


def test_cli_rejects_clef_on_mac(tmp_path, monkeypatch, capsys):
    from vllm_jev import cli

    source, model_id = make_release(tmp_path, monkeypatch)
    output = tmp_path / "out"
    cx.export_clef(source, output, model_id)
    monkeypatch.setattr(cli.sys, "platform", "darwin")
    monkeypatch.setattr(cli.sys, "argv", ["vllm-jev", "serve", str(output)])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    assert "Clef serving requires Linux" in capsys.readouterr().err
