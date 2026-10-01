"""CPU tests for the RSI-Jev adapter. No GPU, no weights download.

Run with ``python -m pytest tests/test_rsijev.py -q``. Tests marked
``reference`` also compare against the published ``rsijev`` package and skip
when it is not importable; ``RSIJEV_RELEASE`` (a local release directory) and a
cached ``Qwen/Qwen3.5-2B-Base`` tokenizer enable the real-weight checks.
"""

import asyncio
import base64
import io
import json
import os
import string
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from vllm_jev import RSIJEV_QWEN35_ARCHITECTURE
from vllm_jev import rsijev as rj
from vllm_jev import rsijev_export as rx

SPECIALS = [rj.VISION_START, rj.IMAGE_PAD, rj.VISION_END, "<|endoftext|>"]
CONFIG = {
    "readout": "option_xattn",
    "xattn_combine": "mlp",
    "xattn_heads": 4,
    "xattn_mlp_hidden": 16,
    "xattn_dim": None,
    "max_options": 160,
    "max_length_text": 2048,
    "cal_mode": "oof_head_scorefloor",
    "vision": {"image_token_budget": 1024, "min_tokens_per_image": 64},
    "max_length_image": 3072,
}


def char_tokenizer(path: Path):
    """A character-level fast tokenizer; the vision markers are single tokens."""
    from tokenizers import Regex, Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    alphabet = string.printable + "éü"
    vocab = {token: i for i, token in enumerate([*SPECIALS, "[UNK]", *alphabet])}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Split(Regex(r"[\s\S]"), "isolated")
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="[UNK]", eos_token="<|endoftext|>"
    )
    fast.add_special_tokens({"additional_special_tokens": SPECIALS[:3]})
    fast.save_pretrained(path)
    return fast


def text_of(tokenizer, ids):
    return "".join(tokenizer.convert_ids_to_tokens(ids))


def data_url(width=64, height=48, color=(200, 30, 30)):
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def random_head(hidden=32, config=CONFIG, seed=0):
    torch.manual_seed(seed)
    head = rj.RsiJevHead(hidden, config)
    for parameter in head.parameters():
        parameter.data.normal_(0, 0.3)
    width = rj.CAL_PCA_DIM + 4 + len(rj.MODES)
    head.cal_pca_mean.normal_()
    head.cal_pca_W.normal_(0, 0.2)
    head.cal_feat_mu.normal_(0, 0.1)
    head.cal_feat_sd.uniform_(0.5, 2.0)
    head.cal_w.normal_(0, 0.3)
    head.cal_b.fill_(0.1)
    assert head.cal_w.shape == (width,)
    return head.eval()


# ---- wire mapping


def test_questions_follow_the_rsijev_wire_contract():
    key, kind, text, options, criteria = rj.to_question(
        "q",
        {"type": "noul", "instructions": "Urgent?", "criteria": {"true": "Act"}},
        160,
    )
    assert (kind, options, criteria) == (
        "noul",
        ("false", "true"),
        {"false": "No", "true": "Act"},
    )
    _, _, _, options, criteria = rj.to_question(
        "c",
        {
            "type": "choice",
            "instructions": {"ask": 1},
            "criteria": {"a": None, "b": "B"},
        },
        160,
    )
    assert options == ("a", "b") and criteria == {"a": "", "b": "B"}
    _, _, text, options, criteria = rj.to_question(
        "s",
        {"type": "score", "instructions": "How bad?", "criteria": ["low", {"x": 2}]},
        160,
    )
    assert options == ("0", "1") and criteria == {"0": "low", "1": '{"x":2}'}
    for bad in (
        {"type": "choice", "instructions": "x", "criteria": {"a": "A"}},
        {
            "type": "choice",
            "instructions": "x",
            "criteria": {str(i): "" for i in range(161)},
        },
        {"type": "score", "instructions": "x", "criteria": "low"},
        {"type": "noul", "instructions": "x", "criteria": {"maybe": "?"}},
        {"type": "rank", "instructions": "x"},
        {"type": "noul"},
        {"type": "noul", "instructions": "<|image_pad|>"},
    ):
        with pytest.raises(ValueError):
            rj.to_question("q", bad, 160)


def test_text_states_render_as_rsijev_serves_them():
    assert rj.state_parts("plain") == ("plain", [])
    assert rj.state_parts({"a": [1, "é"]}) == ('{"a":[1,"é"]}', [])
    chat = {"messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]}
    assert (
        rj.state_parts(chat)[0]
        == '[{"role":"user","content":[{"type":"text","text":"hi"}]}]'
    )
    with pytest.raises(ValueError):
        rj.state_parts({"messages": [{"role": "robot", "content": "x"}]})
    with pytest.raises(ValueError):
        rj.state_parts(3)


def test_image_parts_become_markers_in_order():
    def part(url=None, text=None):
        if text is not None:
            return {"type": "text", "text": text}
        return {"type": "image_url", "image_url": {"url": url or data_url()}}

    def state(*parts):
        return {"messages": [{"role": "user", "content": list(parts)}]}

    text, images = rj.state_parts(
        state(part(text="Left: "), part(), part(text="\nRight: "), part())
    )
    assert text == "Left: <image>\nRight: <image>" and len(images) == 2
    # An image, a newline and the text equal RSI-Jev's unmarked `images` request.
    text, _ = rj.state_parts(state(part(), part(text="\nLook.")))
    assert rj.expand_state(text, [3]) == rj.expand_state("Look.", [3])
    with pytest.raises(ValueError, match="at most 4"):
        rj.state_parts(state(*[part()] * 5))
    with pytest.raises(ValueError):
        rj.state_parts(state(part(), part(text="an <image> here")))
    with pytest.raises(ValueError):
        rj.state_parts(state(part(url="https://example.com/a.png")))


def test_image_budget_matches_the_qwen_processor():
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import (
        Qwen2VLImageProcessor,
    )

    for n_images, (width, height) in ((1, (640, 480)), (2, (1200, 300)), (4, (50, 40))):
        per_image, size = rj.image_size_limits(n_images, 1024, 64)
        assert per_image == max(64, 1024 // n_images)
        processor = Qwen2VLImageProcessor(patch_size=16, merge_size=2, size=size)
        image = Image.new("RGB", (width, height), (10, 20, 30))
        grid = processor(images=[image], return_tensors="pt")["image_grid_thw"][0]
        assert rj.image_tokens(image, size) == int(grid.prod()) // 4
        assert rj.image_tokens(image, size) <= per_image


# ---- encoding


def test_encoding_spans_decision_and_left_truncation(tmp_path):
    tokenizer = char_tokenizer(tmp_path)
    options, criteria = ("yes", "no"), {"yes": "Do it", "no": ""}
    ids, spans, decision = rj.encode_question(
        tokenizer, "The state.", "Pick one.", options, criteria, 2048
    )
    text = text_of(tokenizer, ids)
    assert text == "The state.\n\nPick one.\nOptions:\n\n- yes: Do it\n- no\n\nAnswer:"
    assert [text_of(tokenizer, ids[a:b]) for a, b in spans] == [
        "\n- yes: Do it",
        "\n- no",
    ]
    assert decision == len(ids) - 1
    short, short_spans, short_decision = rj.encode_question(
        tokenizer, "The state.", "Pick one.", options, criteria, len(ids) - 5
    )
    assert short == ids[5:] and short_decision == decision - 5
    assert short_spans == [(a - 5, b - 5) for a, b in spans]
    with pytest.raises(ValueError):
        rj.encode_question(tokenizer, "s", "q", options, criteria, 10)
    empty, _, _ = rj.encode_question(tokenizer, "  ", "Q", options, criteria, 2048)
    assert text_of(tokenizer, empty).startswith("Q\nOptions:")


@pytest.mark.reference
def test_encoding_equals_the_published_encoder():
    encode = pytest.importorskip("rsijev.encode")
    contract = pytest.importorskip("rsijev.contract")
    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            rx.BASE_ID, revision=rx.BASE_REVISION, local_files_only=True
        )
    except OSError:
        pytest.skip("Qwen3.5 tokenizer is not cached")
    state = "Ticket #42: the customer reports a double charge.\n\n" * 40
    questions = [
        ("noul", {"type": "noul", "instructions": "Refund due?"}),
        (
            "choice",
            {
                "type": "choice",
                "instructions": "Team?",
                "criteria": {"billing": "Payments", "tech": None, "1": "Numeric key"},
            },
        ),
        (
            "score",
            {"type": "score", "instructions": "Severity", "criteria": ["0", "1", "2"]},
        ),
    ]
    for max_length in (2048, 300):
        for name, spec in questions:
            key, kind, text, options, criteria = rj.to_question(name, spec, 160)
            question = contract.Question(key, kind, text, options, criteria)
            reference = encode.encode_question(
                tokenizer,
                state,
                question,
                encode.EncodeConfig(max_length=max_length, option_order="canonical"),
            )
            ids, spans, decision = rj.encode_question(
                tokenizer, state, text, options, criteria, max_length
            )
            assert ids == reference["input_ids"]
            assert spans == [tuple(span) for span in reference["option_span"]]
            assert decision == reference["decision_index"]


@pytest.mark.reference
def test_image_encoding_equals_the_published_vision_encoder():
    vision = pytest.importorskip("rsijev.vision")
    encode = pytest.importorskip("rsijev.encode")
    contract = pytest.importorskip("rsijev.contract")
    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            rx.BASE_ID, revision=rx.BASE_REVISION, local_files_only=True
        )
        prep = vision.ImagePrep(rx.BASE_ID, vision.VisionConfig(), rx.BASE_REVISION)
    except OSError:
        pytest.skip("Qwen3.5 tokenizer or image processor is not cached")
    pad = tokenizer.convert_tokens_to_ids(rj.IMAGE_PAD)
    spec = {
        "type": "choice",
        "instructions": "Which?",
        "criteria": {"a": "Left", "b": None},
    }
    key, kind, text, options, criteria = rj.to_question("q", spec, 160)
    question = contract.Question(key, kind, text, options, criteria)
    config = encode.EncodeConfig(max_length=3072, option_order="canonical")
    for sizes, layout in (
        ([(640, 480)], "first"),
        ([(1500, 400), (90, 700)], "between"),
    ):
        urls = [data_url(w, h, (i * 60, 90, 30)) for i, (w, h) in enumerate(sizes)]
        parts = [{"type": "image_url", "image_url": {"url": url}} for url in urls]
        if layout == "first":
            content = [*parts, {"type": "text", "text": "\nDescribe."}]
            reference_state = "Describe."
        else:
            content = [
                {"type": "text", "text": "Left: "},
                parts[0],
                {"type": "text", "text": "\nRight: "},
                parts[1],
            ]
            reference_state = "Left: <image>\nRight: <image>"
        state, images = rj.state_parts(
            {"messages": [{"role": "user", "content": content}]}
        )
        _, size = rj.image_size_limits(len(images), 1024, 64)
        counts = [rj.image_tokens(image, size) for image in images]
        ids, spans, decision = rj.encode_question(
            tokenizer, rj.expand_state(state, counts), text, options, criteria, 3072
        )
        reference = vision.encode_vision_question(
            tokenizer, prep, reference_state, images, question, config
        )
        assert ids == reference["input_ids"]
        assert spans == [tuple(span) for span in reference["option_span"]]
        assert decision == reference["decision_index"]
        assert ids.count(pad) == int(reference["image_grid_thw"].prod(-1).sum()) // 4


# ---- readout


def test_calibration_keeps_the_argmax_and_softens_score_rows():
    head = random_head()
    torch.manual_seed(1)
    rows = []
    for kind in ("choice", "noul", "score"):
        states = torch.randn(12, 32).bfloat16()
        rows.append(
            (states, [(0, 3), (3, 5), (5, 9)][: 2 if kind == "noul" else 3], 11, kind)
        )
    calibrated = head(rows)
    head.cal_mode = "none"
    raw = head(rows)
    for kind, a, b in zip(("choice", "noul", "score"), calibrated, raw):
        assert max(range(len(a)), key=a.__getitem__) == max(
            range(len(b)), key=b.__getitem__
        )
        assert abs(sum(a) - 1) < 1e-5
        if kind == "score":
            assert max(a) <= max(b) + 1e-6


def test_padding_other_questions_does_not_change_an_answer():
    head = random_head()
    torch.manual_seed(2)
    one = (torch.randn(10, 32).bfloat16(), [(0, 2), (2, 4)], 9, "choice")
    wide = (
        torch.randn(30, 32).bfloat16(),
        [(i, i + 2) for i in range(0, 20, 2)],
        29,
        "score",
    )
    alone = head([one])[0]
    together = head([one, wide])[0]
    assert max(abs(a - b) for a, b in zip(alone, together)) < 1e-5


@pytest.mark.reference
def test_head_equals_the_published_readout():
    arch = pytest.importorskip("rsijev.arch")
    from torch import nn

    head = random_head(hidden=32)
    config = arch.ArchConfig(
        readout="option_xattn",
        readout_layer=-1,
        max_options=160,
        option_pool="mean",
        xattn_combine="mlp",
        xattn_mlp_hidden=16,
    )
    reference = arch.DecisionModel(nn.Module(), 32, config)
    reference.scorer.load_state_dict(
        {k: v for k, v in head.state_dict().items() if not k.startswith("cal_")}
    )
    for name, value in head.state_dict().items():
        if name.startswith("cal_"):
            getattr(reference, name).copy_(value)
    reference.cal_mode = head.cal_mode
    reference.eval()
    _assert_head_matches(head, reference, hidden=32)


@pytest.mark.reference
def test_release_head_equals_the_published_readout():
    release = os.environ.get("RSIJEV_RELEASE")
    if not release:
        pytest.skip("set RSIJEV_RELEASE to a local release directory")
    arch = pytest.importorskip("rsijev.arch")
    calibrate = pytest.importorskip("rsijev.calibrate")
    from safetensors.torch import load_file
    from torch import nn

    release = Path(release)
    meta = json.loads((release / "meta.json").read_text())
    calibration = json.loads((release / "calibration.json").read_text())
    config = rx.readout_config(meta, calibration)
    head = rj.RsiJevHead(2048, config)
    head.load(
        load_file(str(release / "scorer.safetensors")),
        load_file(str(release / "calibration.safetensors")),
    )
    head.eval()
    spec = meta["spec"]
    reference = arch.DecisionModel(
        nn.Module(),
        2048,
        arch.ArchConfig(
            readout=spec["readout"],
            readout_layer=spec["readout_layer"],
            max_options=spec["max_options"],
            option_pool=spec["option_pool"],
            **dict(spec.get("arch_extra") or {}),
        ),
    )
    reference.scorer.load_state_dict(load_file(str(release / "scorer.safetensors")))
    calibrate.load_calibration(reference, release)
    reference.eval()
    _assert_head_matches(head, reference, hidden=2048, scale=3.0)


def _assert_head_matches(head, reference, hidden, scale=1.0):
    from rsijev.encode import collate  # noqa: F401  (the reference batch layout)

    torch.manual_seed(3)
    shapes = [
        ("choice", [(2, 7), (7, 9), (9, 15), (15, 300)], 320),
        ("noul", [(4, 6), (6, 8)], 12),
        ("score", [(1, 3), (3, 5), (5, 9)], 14),
    ]
    rows = [
        ((torch.randn(n, hidden) * scale).bfloat16(), spans, n - 1, kind)
        for kind, spans, n in shapes
    ]
    ours = head(rows)
    width = max(n for _, _, n in shapes)
    h = torch.zeros((len(rows), width, hidden), dtype=torch.bfloat16)
    k = max(len(spans) for _, spans, _ in shapes)
    start = torch.zeros((len(rows), k), dtype=torch.long)
    end = torch.zeros_like(start)
    mask = torch.zeros((len(rows), k), dtype=torch.bool)
    for r, (states, spans, index, kind) in enumerate(rows):
        h[r, : len(states)] = states
        start[r, : len(spans)] = torch.tensor([a for a, _ in spans])
        end[r, : len(spans)] = torch.tensor([b for _, b in spans])
        mask[r, : len(spans)] = True
    mode = torch.tensor([rj.MODES.index(kind) for _, _, _, kind in rows])
    b = torch.arange(len(rows))
    with torch.no_grad():
        logits, _ = reference._readout(
            h,
            h,
            b,
            torch.tensor([index for _, _, index, _ in rows]),
            None,
            start,
            end,
            None,
            mask,
            False,
            mode_id=mode,
        )
    for r, (_, spans, _, _) in enumerate(rows):
        expected = torch.softmax(logits[r, : len(spans)].float(), -1).tolist()
        assert max(abs(a - b) for a, b in zip(ours[r], expected)) < 1e-5


# ---- service with a fake engine


class FakeEngine:
    """Returns deterministic bf16 states, optionally only after a cached prefix."""

    def __init__(self, hidden=32, cached=0, image_tokens=0, image_pad=None):
        self.hidden, self.cached = hidden, cached
        self.image_tokens, self.image_pad = image_tokens, image_pad
        self.requests = []

    def states(self, ids):
        rows = []
        for position, token in enumerate(ids):
            generator = torch.Generator().manual_seed(token * 7919 + position)
            rows.append(torch.randn(self.hidden, generator=generator))
        return torch.stack(rows).bfloat16()

    async def encode(self, prompt, pooling_params, request_id):
        self.requests.append((prompt, pooling_params))
        ids = []
        for token in prompt["prompt_token_ids"]:
            if token == self.image_pad:
                ids += [token] * self.image_tokens
            else:
                ids.append(token)
        cached = 0 if pooling_params.skip_reading_prefix_cache else self.cached
        cached = min(cached, len(ids) - 1)
        yield SimpleNamespace(
            finished=True,
            prompt_token_ids=ids,
            num_cached_tokens=cached,
            outputs=SimpleNamespace(data=self.states(ids)[cached:]),
        )

    async def abort(self, request_id):
        pass


def make_model_dir(tmp_path, head):
    from safetensors.torch import save_file

    tokenizer = char_tokenizer(tmp_path)
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": [RSIJEV_QWEN35_ARCHITECTURE],
                "image_token_id": tokenizer.convert_tokens_to_ids(rj.IMAGE_PAD),
                "text_config": {"hidden_size": 32},
            }
        )
    )
    (tmp_path / "rsijev_config.json").write_text(json.dumps(CONFIG))
    tensors = head.state_dict()
    save_file(
        {k: v.contiguous() for k, v in tensors.items() if not k.startswith("cal_")},
        str(tmp_path / "rsijev_scorer.safetensors"),
    )
    save_file(
        {k: v.contiguous() for k, v in tensors.items() if k.startswith("cal_")},
        str(tmp_path / "rsijev_calibration.safetensors"),
    )
    return tokenizer


def run_service(service, payload):
    request = SimpleNamespace(model=None, **payload)
    return asyncio.run(service.systemone(request))


@pytest.mark.parametrize("cached", [0, 40, 10_000])
def test_service_answers_from_the_readout_rows(tmp_path, monkeypatch, cached):
    monkeypatch.setenv("VLLM_JEV_RSIJEV_DEVICE", "cpu")
    head = random_head()
    tokenizer = make_model_dir(tmp_path, head)
    engine = FakeEngine(cached=cached)
    service = rj.RsiJevService(engine, tmp_path, "shgao/rsi-jev-test", 4096)
    payload = {
        "state": "A customer was charged twice.",
        "questions": {
            "team": {
                "type": "choice",
                "instructions": "Route it.",
                "criteria": {"billing": "Payments", "tech": None, "other": "Else"},
            },
            "urgent": {"type": "noul", "instructions": "Urgent?"},
            "severity": {
                "type": "score",
                "instructions": "Severity",
                "criteria": ["low", "high"],
            },
        },
    }
    response = run_service(service, payload)
    expected = []
    for key, spec in payload["questions"].items():
        _, kind, text, options, criteria = rj.to_question(key, spec, 160)
        ids, spans, decision = rj.encode_question(
            tokenizer, payload["state"], text, options, criteria, 2048
        )
        expected.append((engine.states(ids), spans, decision, kind))
    reference = head(expected)
    answers = response["answers"]
    assert answers["team"]["probabilities"] == pytest.approx(
        dict(zip(("billing", "tech", "other"), reference[0])), abs=1e-6
    )
    assert answers["urgent"] == {
        "type": "noul",
        "noul": pytest.approx(reference[1][1], abs=1e-6),
    }
    assert answers["severity"]["score"] == pytest.approx(reference[2][1], abs=1e-6)
    reruns = response["metadata"]["uncached_reruns"]
    # A hit that reaches into the option blocks is computed again without the cache.
    assert reruns == (3 if cached == 10_000 else 0)
    first_attempts = [p for _, p in engine.requests if not p.skip_reading_prefix_cache]
    assert len(first_attempts) == 3


def test_service_sends_images_with_the_trained_budget(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_JEV_RSIJEV_DEVICE", "cpu")
    head = random_head()
    tokenizer = make_model_dir(tmp_path, head)
    pad = tokenizer.convert_tokens_to_ids(rj.IMAGE_PAD)
    per_image, size = rj.image_size_limits(2, 1024, 64)
    image = Image.new("RGB", (640, 480))
    count = rj.image_tokens(image, size)
    engine = FakeEngine(image_tokens=count, image_pad=pad)
    service = rj.RsiJevService(engine, tmp_path, "shgao/rsi-jev-test", 4096)
    url = data_url(640, 480, (0, 0, 0))
    state = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": url}},
                    {"type": "image_url", "image_url": {"url": url}},
                    {"type": "text", "text": "Which is brighter?"},
                ],
            }
        ]
    }
    response = run_service(
        service,
        {
            "state": state,
            "questions": {
                "q": {
                    "type": "choice",
                    "instructions": "Pick",
                    "criteria": {"left": None, "right": None},
                }
            },
        },
    )
    prompt, _ = engine.requests[0]
    assert prompt["mm_processor_kwargs"] == {"size": size}
    assert prompt["prompt_token_ids"].count(pad) == 2
    assert len(prompt["multi_modal_data"]["image"]) == 2
    assert set(response["answers"]["q"]["probabilities"]) == {"left", "right"}
    assert response["usage"]["input_tokens"] > 2 * count


def test_text_only_release_rejects_images(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_JEV_RSIJEV_DEVICE", "cpu")
    head = random_head()
    make_model_dir(tmp_path, head)
    config = dict(CONFIG, vision=None)
    config.pop("max_length_image")
    (tmp_path / "rsijev_config.json").write_text(json.dumps(config))
    service = rj.RsiJevService(FakeEngine(), tmp_path, "shgao/rsi-jev-test", 2048)
    state = {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": data_url()}}],
            }
        ]
    }
    with pytest.raises(ValueError, match="text-only"):
        run_service(
            service,
            {
                "state": state,
                "questions": {"q": {"type": "noul", "instructions": "Red?"}},
            },
        )


# ---- export


def make_release(tmp_path, monkeypatch, vision=True):
    from safetensors.torch import save_file

    base, source = tmp_path / "base", tmp_path / "release"
    base.mkdir()
    source.mkdir()
    torch.manual_seed(4)
    names = ["layers.0.mlp.up_proj.weight", "norm.weight"]
    base_tensors = {
        "model.language_model." + n: torch.randn(4, 8).bfloat16() for n in names
    }
    base_tensors["model.language_model.embed_tokens.weight"] = torch.randn(
        16, 8
    ).bfloat16()
    base_tensors["model.visual.blocks.0.weight"] = torch.randn(3, 3).bfloat16()
    base_tensors["mtp.fc.weight"] = torch.randn(2, 2).bfloat16()
    save_file(base_tensors, str(base / "model-00001.safetensors"))
    (base / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: "model-00001.safetensors" for k in base_tensors}})
    )
    (base / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "image_token_id": 5,
                "text_config": {"hidden_size": 8},
            }
        )
    )
    for name in ("tokenizer.json", "tokenizer_config.json", "preprocessor_config.json"):
        (base / name).write_text("{}")
    tower = {n: torch.randn(4, 8) for n in names}
    save_file(tower, str(source / "tower.safetensors"))
    head = random_head(hidden=8, config=dict(CONFIG, xattn_mlp_hidden=4))
    save_file(
        {k: v for k, v in head.state_dict().items() if not k.startswith("cal_")},
        str(source / "scorer.safetensors"),
    )
    save_file(
        {k: v for k, v in head.state_dict().items() if k.startswith("cal_")},
        str(source / "calibration.safetensors"),
    )
    release = {"cal_mode": "oof_head_scorefloor", "max_length_text": 2048}
    if vision:
        release["vision"] = {
            "budget": 1024,
            "model": rx.BASE_ID,
            "max_length_image": 3072,
        }
    (source / "meta.json").write_text(
        json.dumps(
            {
                "base_model": rx.BASE_ID,
                "release": release,
                "spec": {
                    "readout": "option_xattn",
                    "readout_layer": -1,
                    "option_pool": "mean",
                    "layout": "state_first",
                    "residual": False,
                    "logit_cap": None,
                    "head_input_norm": False,
                    "max_options": 160,
                    "arch_extra": {"xattn_combine": "mlp", "xattn_mlp_hidden": 4},
                },
            }
        )
    )
    (source / "calibration.json").write_text(
        json.dumps({"cal_mode": "oof_head_scorefloor"})
    )
    model_id = "shgao/rsi-jev-test"
    monkeypatch.setitem(rx.MODELS, model_id, "0" * 40)
    monkeypatch.setitem(
        rx.RELEASE_HASHES,
        model_id,
        {
            name: rx.sha256(source / name)
            for name in (
                "tower.safetensors",
                "scorer.safetensors",
                "calibration.safetensors",
                "calibration.json",
                "meta.json",
            )
        },
    )
    return base, source, model_id, tower


def test_export_writes_a_verified_bf16_vision_language_checkpoint(
    tmp_path, monkeypatch
):
    from safetensors import safe_open

    base, source, model_id, tower = make_release(tmp_path, monkeypatch)
    output = tmp_path / "out"
    manifest = rx.export_rsijev(source, base, output, model_id)
    assert manifest["max_length"] == 3072 and manifest["vision"] is True
    assert rx.verify_rsijev(output)["architecture"] == RSIJEV_QWEN35_ARCHITECTURE
    index = json.loads((output / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    assert "mtp.fc.weight" not in index and "model.visual.blocks.0.weight" in index
    with safe_open(output / rx.TEXT_SHARD, framework="pt") as file:
        weight = file.get_tensor("model.language_model.norm.weight")
        assert weight.dtype == torch.bfloat16
        assert torch.equal(weight, tower["norm.weight"].bfloat16())
    config = json.loads((output / "rsijev_config.json").read_text())
    assert config["vision"] == {"image_token_budget": 1024, "min_tokens_per_image": 64}
    (output / "rsijev_config.json").write_text(
        json.dumps(dict(config, cal_mode="none"))
    )
    with pytest.raises(ValueError, match="checksum"):
        rx.verify_rsijev(output)


def test_export_refuses_a_changed_release_or_readout(tmp_path, monkeypatch):
    base, source, model_id, _ = make_release(tmp_path, monkeypatch, vision=False)
    meta = json.loads((source / "meta.json").read_text())
    meta["spec"]["readout"] = "pointer"
    with pytest.raises(ValueError, match="unsupported RSI-Jev readout"):
        rx.readout_config(meta, {"cal_mode": "oof_head_scorefloor"})
    (source / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="checksum"):
        rx.export_rsijev(source, base, tmp_path / "out", model_id)


def test_cli_serves_a_prepared_rsijev_checkpoint(tmp_path, monkeypatch):
    from vllm_jev import cli

    base, source, model_id, _ = make_release(tmp_path, monkeypatch)
    output = tmp_path / "out"
    rx.export_rsijev(source, base, output, model_id)
    calls = {}
    monkeypatch.setattr(cli.subprocess, "run", lambda *a, **k: None)
    monkeypatch.setattr(
        cli.os, "execvpe", lambda exe, argv, env: calls.update(argv=argv)
    )
    monkeypatch.setattr(cli.sys, "argv", ["vllm-jev", "serve", str(output)])
    cli.main()
    argv = calls["argv"]
    assert argv[argv.index("--max-model-len") + 1] == "3072"
    assert argv[argv.index("--served-model-name") + 1] == model_id
    assert json.loads(argv[argv.index("--pooler-config") + 1]) == {
        "task": "token_embed"
    }
    assert json.loads(argv[argv.index("--limit-mm-per-prompt") + 1]) == {
        "image": 4,
        "video": 0,
    }
    assert "--enable-prefix-caching" in argv
