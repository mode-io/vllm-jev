"""This-That CPU and reference boundaries, without released model weights."""

import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm_jev import windows_thisthat as adapter
from vllm_jev.decision_template import ENVIRONMENT, current_template

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = {
    "route": {
        "type": "choice",
        "instructions": "Route?",
        "criteria": {"bill": "Refund", "tech": None},
    },
    "yes": {"type": "noul", "instructions": "Refund?"},
    "urgency": {
        "type": "score",
        "instructions": "Urgent?",
        "criteria": ["Later", "Soon", "Now"],
    },
}


@pytest.fixture(autouse=True)
def native_template(monkeypatch):
    monkeypatch.delenv(ENVIRONMENT, raising=False)
    current_template.cache_clear()
    yield
    current_template.cache_clear()


class Tokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return [ord(char) % 511 + 1 for char in text]


def make_agent(model_id=adapter.MODEL_ID):
    import torch

    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(217)
            self.embedding = torch.nn.Embedding(512, 16)
            self.calls = []

        def get_input_embeddings(self):
            return self.embedding

        def forward(self, *, input_ids, attention_mask, use_cache, return_dict):
            assert not use_cache and return_dict
            self.calls.append((input_ids.clone(), attention_mask.clone()))
            values = self.embedding(input_ids) * attention_mask[..., None]
            hidden = values.cumsum(1) / attention_mask.cumsum(1).clamp_min(1)[..., None]
            return SimpleNamespace(last_hidden_state=hidden)

    model = Backbone()
    torch.manual_seed(218)
    weight = torch.randn(255, 16)
    return adapter.ThisThatAgent(
        model,
        Tokenizer(),
        weight,
        list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + [f"L{i}" for i in range(229)],
        list(range(1, 256)),
        model_id=model_id,
    )


def expected_packed(agent, state):
    """Independent fixed author prompt and all-slot projection for the mixed case."""
    import numpy as np
    import torch

    text = "Context:\n" + (
        state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
    )
    blocks = [
        ("Route?\n\nOptions:\n- bill: Refund\n- tech", ["bill", "tech"]),
        ("Refund?", ["no", "yes"]),
        ("Urgent?\n\nLevels:\n- 0: Later\n- 1: Soon\n- 2: Now", ["0", "1", "2"]),
    ]
    slots = []
    for index, (question, options) in enumerate(blocks, 1):
        text += f"\n\nQuestion {index}: {question}\nOptions:"
        text += "".join(
            f"\n({chr(65 + i)}) {option}" for i, option in enumerate(options)
        )
        text += f"\nAnswer {index}: ("
        slots.append(len(Tokenizer().encode(text)) - 1)
    ids = Tokenizer().encode(text)
    with torch.no_grad():
        values = agent.model.embedding(torch.tensor(ids))
        hidden = values.cumsum(0) / torch.arange(1, len(ids) + 1)[:, None]
        logits = torch.nn.functional.linear(hidden[slots], agent.weight).numpy()
    logits[np.arange(255)[None, :] >= np.asarray([2, 2, 3])[:, None]] = -np.inf
    z = logits.astype(np.float32) / 1.0
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    p = e / e.sum(axis=-1, keepdims=True)
    rows = []
    for row, count in zip(p, [2, 2, 3]):
        values = [float(x) for x in row[:count]]
        rows.append([x / sum(values) for x in values])
    return ids, slots, rows


@pytest.mark.parametrize(
    "state", ["Duplicate charge.", {"问题": "重复扣款 🙏", "订单": "A 12"}]
)
def test_mixed_questions_use_one_packed_forward_and_exact_author_normalization(state):
    agent = make_agent()
    ids, slots, rows = expected_packed(agent, state)
    result = agent.system_one(state, QUESTIONS)
    assert len(agent.model.calls) == 1
    tokens, mask = agent.model.calls[0]
    assert tokens.shape == (1, (len(ids) + 63) // 64 * 64)
    assert tokens[0, : len(ids)].tolist() == ids
    assert mask[0, : len(ids)].all() and not mask[0, len(ids) :].any()
    assert not tokens[0, len(ids) :].any()
    assert slots[0] < slots[1] < slots[2] == len(ids) - 1
    answers = result["answers"]
    assert list(answers["route"]["probabilities"].values()) == rows[0]
    assert answers["yes"]["noul"] == rows[1][1]
    assert list(answers["urgency"]["probabilities"].values()) == rows[2]
    assert answers["urgency"]["score"] == sum(i * p for i, p in enumerate(rows[2]))
    assert answers["route"]["confidence"] == max(0.0, 2 * max(rows[0]) - 1)
    assert (
        result["usage"]["input_tokens"]
        == result["internal_usage"]["compute_tokens"]
        == len(ids)
    )
    assert result["metadata"]["backbone_dtype"] == "float32"
    assert result["metadata"]["softmax_backend"] == "numpy"


def test_wide_labels_are_encoded_as_one_token_and_padding_does_not_change_slots():
    agent = make_agent()
    criteria = [f"option{i}" for i in range(12)]
    result = agent.system_one(
        "x", {"q": {"type": "choice", "instructions": "Pick?", "criteria": criteria}}
    )
    prefix = (
        "Context:\nx\n\nQuestion: Pick?\n\nOptions:\n"
        + "\n".join(f"- {x}" for x in criteria)
        + "\nOptions:"
    )
    ids = Tokenizer().encode(prefix)
    for i, value in enumerate(criteria):
        ids += Tokenizer().encode("\n(") + [i + 1] + Tokenizer().encode(") " + value)
    ids += Tokenizer().encode("\nAnswer: (")
    tokens, mask = agent.model.calls[0]
    assert tokens[0, : int(mask.sum())].tolist() == ids
    assert list(result["answers"]["q"]["probabilities"]) == criteria
    assert sum(result["answers"]["q"]["probabilities"].values()) == pytest.approx(1)


@pytest.mark.parametrize(
    "change",
    [
        "too_long",
        "over_candidates",
        "no_instructions",
        "bad_choice",
        "bad_noul",
        "isolated",
        "nan",
    ],
)
def test_invalid_requests_are_rejected_before_forward(change):
    agent = make_agent()
    state, questions = "x", copy.deepcopy(QUESTIONS)
    if change == "too_long":
        state = "x" * 1100
    elif change == "over_candidates":
        questions["route"]["criteria"] = [f"id{i}" for i in range(255)]
    elif change == "no_instructions":
        questions["route"]["instructions"] = ""
    elif change == "bad_choice":
        questions["route"]["criteria"] = [None, True]
    elif change == "bad_noul":
        questions["yes"]["criteria"] = ["no", "yes"]
    elif change == "isolated":
        questions["urgency"]["isolated"] = True
    else:
        state = {"number": float("nan")}
    with pytest.raises(ValueError):
        agent.system_one(state, questions)
    assert not agent.model.calls


def test_precision_mismatch_and_closed_models_fail_without_forward():
    agent = make_agent()
    agent.model.bfloat16()
    with pytest.raises(ValueError, match="FP32"):
        agent.system_one("x", QUESTIONS)
    assert not agent.model.calls
    agent.close()
    with pytest.raises(RuntimeError, match="closed"):
        agent.system_one("x", QUESTIONS)


@pytest.mark.parametrize("model_id", tuple(adapter.MODELS))
def test_actual_small_qwen35_forward_packed_readout_matches_full_backbone(model_id):
    import numpy as np
    import torch
    from transformers import Qwen3_5TextConfig, Qwen3_5TextModel

    torch.manual_seed(220)
    torch.set_num_threads(1)
    config = Qwen3_5TextConfig(
        vocab_size=512,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        layer_types=["linear_attention", "full_attention"],
        max_position_embeddings=1024,
        pad_token_id=0,
    )
    config._attn_implementation = "sdpa"
    model = Qwen3_5TextModel(config).eval().requires_grad_(False)
    weight = torch.randn(255, 32)
    agent = adapter.ThisThatAgent(
        model,
        Tokenizer(),
        weight,
        [str(i) for i in range(255)],
        list(range(255)),
        model_id=model_id,
    )
    ids, positions, counts = [11, 12, 13, 14, 15], [2, 4], [2, 12]
    actual = agent._probabilities(ids, positions, counts)
    tokens = torch.tensor([ids + [0] * 59])
    mask = torch.tensor([[1] * 5 + [0] * 59])
    with torch.no_grad():
        states = model(
            input_ids=tokens, attention_mask=mask, use_cache=False
        ).last_hidden_state
        logits = torch.nn.functional.linear(states[0, positions], weight).numpy()
    logits[np.arange(255)[None] >= np.asarray(counts)[:, None]] = -np.inf
    exps = np.exp(logits - logits.max(axis=1, keepdims=True))
    rows = exps / exps.sum(axis=1, keepdims=True)
    for got, row, count in zip(actual, rows, counts):
        values = [float(x) for x in row[:count]]
        assert got == [x / sum(values) for x in values]


@pytest.fixture
def pinned_releases(tmp_path, monkeypatch):
    config = {
        "model_type": "qwen3_5_text",
        "architectures": ["Qwen3_5ForCausalLM"],
        "hidden_size": 2048,
        "num_hidden_layers": 24,
        "tie_word_embeddings": True,
    }
    profiles, sources = {}, {}
    for index, (model_id, profile) in enumerate(adapter.MODELS.items()):
        # Deliberately do not identify these directories by a release-shaped name.
        source = tmp_path / f"中文 checkpoint {index}"
        source.mkdir()
        hashes = {}
        for name in adapter.SOURCE_FILES:
            if name == "config.json":
                data = json.dumps(config).encode()
            elif name == "model.safetensors":
                data = model_id.encode()
            elif name == "tokenizer_config.json" and model_id == adapter.MODEL_ID_1_0:
                data = b"original 1.0 tokenizer configuration"
            else:
                data = b"fixture"
            (source / name).write_bytes(data)
            hashes[name] = hashlib.sha256(data).hexdigest()
        profiles[model_id] = {**profile, "source_hashes": hashes}
        sources[model_id] = source
    monkeypatch.setattr(adapter, "MODELS", profiles)
    return sources


@pytest.fixture
def pinned_source(pinned_releases):
    return pinned_releases[adapter.MODEL_ID]


def test_identity_checks_actual_weights_not_shared_decider_configuration(pinned_source):
    assert adapter.identify_source(pinned_source) == adapter.MODEL_ID
    (pinned_source / "model.safetensors").write_bytes(b"other model, same config")
    with pytest.raises(ValueError, match="checksum mismatch: model.safetensors"):
        adapter.identify_source(pinned_source)


@pytest.mark.parametrize("model_id", tuple(adapter.MODELS))
def test_each_release_identity_checks_all_files_and_reads_weights_once(
    pinned_releases, monkeypatch, model_id
):
    source = pinned_releases[model_id]
    original = hashlib.file_digest
    reads = []

    def digest(stream, algorithm):
        reads.append(Path(stream.name).name)
        return original(stream, algorithm)

    monkeypatch.setattr(hashlib, "file_digest", digest)
    assert adapter.identify_source(source) == model_id
    assert sorted(reads) == sorted(adapter.SOURCE_FILES)
    assert reads.count("model.safetensors") == 1
    adapter.verify_source(source, model_id)


@pytest.mark.parametrize("model_id", tuple(adapter.MODELS))
def test_release_does_not_accept_another_versions_weights(
    pinned_releases, monkeypatch, model_id
):
    import transformers

    loader = Mock(side_effect=AssertionError("must reject before loading"))
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", loader)
    for other_id, source in pinned_releases.items():
        if other_id != model_id:
            with pytest.raises(ValueError, match="checksum mismatch"):
                adapter.load_agent(source, model_id, threads=1)
    loader.assert_not_called()


@pytest.mark.parametrize("name", tuple(adapter.SOURCE_FILES))
def test_known_weights_cannot_hide_any_unverified_metadata(pinned_source, name):
    path = pinned_source / name
    if name == "config.json":
        config = json.loads(path.read_text())
        config["extra"] = "wrong release"
        path.write_text(json.dumps(config), encoding="utf-8")
    else:
        path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match=f"checksum mismatch: {name}"):
        adapter.identify_source(pinned_source)


@pytest.mark.parametrize(
    "name,directory",
    [
        ("adapter_config.json", False),
        ("submodel", True),
        ("decider_config.json", False),
    ],
)
def test_unverified_inputs_rejected_before_backend_loading(
    pinned_source, monkeypatch, name, directory
):
    import transformers

    path = pinned_source / name
    path.mkdir() if directory else path.write_text("{}", encoding="utf-8")
    loader = Mock(side_effect=AssertionError("model loader must not be called"))
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", loader)
    with pytest.raises(ValueError, match="unexpected"):
        adapter.load_agent(pinned_source)
    loader.assert_not_called()


@pytest.mark.parametrize("model_id", tuple(adapter.MODELS))
def test_prepare_only_publishes_a_verified_complete_fixed_release(
    tmp_path, pinned_releases, monkeypatch, model_id
):
    import shutil

    calls = []

    def download(**kw):
        calls.append(kw)
        shutil.copytree(pinned_releases[model_id], kw["local_dir"])

    monkeypatch.setattr(adapter, "snapshot_download", download)
    target = adapter.prepare(model_id, tmp_path / "workspace")
    assert adapter.identify_source(target) == model_id
    assert calls[0]["repo_id"] == model_id
    assert calls[0]["revision"] == adapter.MODELS[model_id]["revision"]
    assert set(calls[0]["allow_patterns"]) == set(adapter.SOURCE_HASHES)
    assert adapter.prepare(model_id, tmp_path / "workspace") == target
    assert len(calls) == 1


@pytest.mark.parametrize("existing", [False, True])
def test_prepare_rejects_mixed_release_without_publishing_or_overwriting(
    tmp_path, pinned_releases, monkeypatch, existing
):
    import shutil

    workspace = tmp_path / "workspace"
    target = workspace / "checkpoint" / adapter.MODEL_ID_1_1
    wrong_source = pinned_releases[adapter.MODEL_ID]
    if existing:
        shutil.copytree(wrong_source, target)
    calls = []

    def download(**kw):
        calls.append(kw)
        shutil.copytree(wrong_source, kw["local_dir"])

    monkeypatch.setattr(adapter, "snapshot_download", download)
    with pytest.raises(ValueError, match="checksum mismatch"):
        adapter.prepare(adapter.MODEL_ID_1_1, workspace)
    assert bool(calls) is not existing
    assert target.exists() is existing
    assert not list(target.parent.glob(".thisthat-download-*"))
    if existing:
        assert adapter.identify_source(target) == adapter.MODEL_ID


def test_prepare_failed_download_leaves_no_published_partial_release(
    tmp_path, monkeypatch
):
    workspace = tmp_path / "workspace"

    def download(**kw):
        kw["local_dir"].mkdir()
        (kw["local_dir"] / "model.safetensors").write_bytes(b"incomplete")
        raise ConnectionError("download interrupted")

    monkeypatch.setattr(adapter, "snapshot_download", download)
    with pytest.raises(ConnectionError, match="interrupted"):
        adapter.prepare(adapter.MODEL_ID_1_1, workspace)
    parent = workspace / "checkpoint" / "flock-io"
    assert not (parent / "this-that-model-1.1").exists()
    assert not list(parent.glob(".thisthat-download-*"))


@pytest.mark.parametrize("model_id", tuple(adapter.MODELS))
def test_selected_identity_propagates_without_changing_packed_probabilities(model_id):
    baseline = make_agent().system_one("x", QUESTIONS)
    agent = make_agent(model_id)
    result = agent.system_one("x", QUESTIONS)
    assert agent.model_id == result["model"] == result["metadata"]["model"] == model_id
    assert (
        agent.revision
        == result["metadata"]["revision"]
        == adapter.MODELS[model_id]["revision"]
    )
    assert agent.compiler.model_id == model_id
    assert agent.compiler.model_names == [model_id]
    assert result["answers"] == baseline["answers"]
    assert len(agent.model.calls) == 1


def test_default_12_aliases_keep_the_original_fixed_release():
    assert adapter.MODEL_ID == "flock-io/this-that-model-1.2"
    assert adapter.REVISION == "c4d1c30b8d512d278726de439b8cc81fccc70c8f"
    assert adapter.MODELS[adapter.MODEL_ID]["revision"] == adapter.REVISION
    assert adapter.MODELS[adapter.MODEL_ID]["source_hashes"] is adapter.SOURCE_HASHES
    assert tuple(adapter.SOURCE_HASHES) == adapter.SOURCE_FILES
    assert (
        adapter.SOURCE_HASHES["model.safetensors"]
        == "585295458a6b6e01162d2a72d33ba630c286b51c844869afc35272700c23f199"
    )
    assert make_agent().model_id == adapter.MODEL_ID


def test_release_profiles_are_distinct_and_complete():
    expected = {
        adapter.MODEL_ID: (
            "c4d1c30b8d512d278726de439b8cc81fccc70c8f",
            "585295458a6b6e01162d2a72d33ba630c286b51c844869afc35272700c23f199",
        ),
        adapter.MODEL_ID_1_1: (
            "fc083c82bd5872abfeaf0d44cf7482324efce87d",
            "63e1179554bc56654eb7aa1ccc5dc1edbaf99a2b32e1cd19f77fcf65a98be04d",
        ),
        adapter.MODEL_ID_1_0: (
            "3d927195c4f9845efe66c5715883a7a0f42b1239",
            "11bab4bbbce0214dcb4d70a88e74b3e4bde6fe8a95f239e3d0c355946e0000e0",
        ),
    }
    assert set(adapter.MODELS) == set(expected)
    for model_id, (revision, weight) in expected.items():
        profile = adapter.MODELS[model_id]
        assert profile["revision"] == revision
        assert profile["source_hashes"]["model.safetensors"] == weight
        assert set(profile["source_hashes"]) == set(adapter.SOURCE_FILES)
        differing = {
            name
            for name, digest in profile["source_hashes"].items()
            if digest != adapter.SOURCE_HASHES[name]
        }
        assert differing == (
            set()
            if model_id == adapter.MODEL_ID
            else {"model.safetensors", "tokenizer_config.json"}
            if model_id == adapter.MODEL_ID_1_0
            else {"model.safetensors"}
        )
    assert (
        adapter.MODELS[adapter.MODEL_ID_1_0]["source_hashes"]["tokenizer_config.json"]
        == "353c083805d863719e54973335a38e252a92e1ac64d32b83bb0b333cc1644ef2"
    )


def test_10_weights_with_newer_tokenizer_configuration_are_rejected(pinned_releases):
    source = pinned_releases[adapter.MODEL_ID_1_0]
    newer = pinned_releases[adapter.MODEL_ID] / "tokenizer_config.json"
    (source / "tokenizer_config.json").write_bytes(newer.read_bytes())
    with pytest.raises(ValueError, match="checksum mismatch: tokenizer_config.json"):
        adapter.identify_source(source)


def test_incomplete_loader_result_fails_without_mutating_pinned_source(
    pinned_source, monkeypatch
):
    import transformers

    from vllm_jev import decision_export

    before = {p.name: p.read_bytes() for p in pinned_source.iterdir()}
    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", lambda *a, **kw: Tokenizer()
    )
    monkeypatch.setattr(
        decision_export, "_labels", lambda tok: [(str(i), i) for i in range(255)]
    )
    loader = Mock(
        return_value=(object(), {"missing_keys": ["model.layers.0.mlp.up_proj.weight"]})
    )
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", loader)
    with pytest.raises(ValueError, match="not loaded completely"):
        adapter.load_agent(pinned_source)
    assert str(loader.call_args.kwargs["dtype"]) == "torch.float32"
    assert loader.call_args.kwargs["trust_remote_code"] is False
    assert before == {p.name: p.read_bytes() for p in pinned_source.iterdir()}


@pytest.mark.parametrize("model_id", tuple(adapter.MODELS))
def test_loader_preserves_selected_release_and_fp32_settings(
    pinned_releases, monkeypatch, model_id
):
    import torch
    import transformers

    from vllm_jev import decision_export

    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(256, 2048)
            self.config = SimpleNamespace(use_cache=True)

        def get_input_embeddings(self):
            return self.embedding

    source = pinned_releases[model_id]
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    backbone = Backbone()
    full = SimpleNamespace(
        model=backbone, get_output_embeddings=backbone.get_input_embeddings
    )
    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", lambda *a, **kw: Tokenizer()
    )
    monkeypatch.setattr(
        decision_export, "_labels", lambda tok: [(str(i), i) for i in range(255)]
    )
    loader = Mock(return_value=(full, {}))
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", loader)
    agent = adapter.load_agent(source, model_id, threads=1)
    assert agent.model_id == model_id
    assert agent.revision == adapter.MODELS[model_id]["revision"]
    assert agent.compiler.model_names == [model_id]
    assert agent.weight.dtype == torch.float32
    assert agent.weight.shape == (255, 2048)
    assert agent.weight.data_ptr() != backbone.embedding.weight.data_ptr()
    assert not backbone.config.use_cache
    assert loader.call_args.args == (source.resolve(),)
    assert loader.call_args.kwargs == {
        "dtype": torch.float32,
        "attn_implementation": "sdpa",
        "device_map": {"": "cpu"},
        "local_files_only": True,
        "trust_remote_code": False,
        "output_loading_info": True,
    }
    assert before == {p.name: p.read_bytes() for p in source.iterdir()}
    agent.close()
    assert agent.model is None and agent.weight is None


def load_reference():
    path = ROOT / "scripts/windows_thisthat_reference.py"
    spec = importlib.util.spec_from_file_location("thisthat_reference_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reference_has_independent_source_pins_and_no_backend_import():
    reference = load_reference()
    assert reference.SOURCE_HASHES == adapter.SOURCE_HASHES
    assert reference.MODELS == adapter.MODELS
    assert len(reference.SDK_HASHES) == 11
    code = """
import importlib.abc, importlib.util, sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname.split('.')[0] in {'torch','transformers','vllm','mlx','thisthat','vllm_jev'}:
   raise AssertionError('unexpected backend import: '+fullname)
sys.meta_path.insert(0, Block())
spec=importlib.util.spec_from_file_location('reference', sys.argv[1])
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
"""
    subprocess.run(
        [sys.executable, "-B", "-c", code, reference.__file__], check=True, cwd=ROOT
    )
    subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            "import sys; import vllm_jev.windows_thisthat; assert not any(n in sys.modules for n in ('torch','transformers','vllm','mlx'))",
        ],
        check=True,
        cwd=ROOT,
    )


def test_reference_isolates_only_pinned_python_and_configuration(tmp_path, monkeypatch):
    reference = load_reference()
    source, sdk = tmp_path / "source", tmp_path / "sdk"
    for root, name in ((source, "SOURCE_HASHES"), (sdk, "SDK_HASHES")):
        hashes = {}
        for relative in getattr(reference, name):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(relative.encode())
            hashes[relative] = reference.sha256(path)
        monkeypatch.setattr(reference, name, hashes)
        if name == "SOURCE_HASHES":
            monkeypatch.setitem(
                reference.MODELS,
                reference.MODEL_ID,
                {"revision": reference.REVISION, "source_hashes": hashes},
            )
    cache = sdk / "thisthat/backends/__pycache__"
    cache.mkdir()
    (cache / "torch_backend.pyc").write_bytes(b"stale bytecode")
    workspace = tmp_path / "private"
    workspace.mkdir()
    cloned_source, cloned_sdk = reference.copy_inputs(source, sdk, workspace)
    assert not (cloned_sdk / "thisthat/backends/__pycache__").exists()
    (cloned_source / "config.json").write_text("changed in runtime", encoding="utf-8")
    assert (source / "config.json").read_bytes() == b"config.json"
    (sdk / "thisthat/backends/unpinned.py").write_bytes(b"override")
    with pytest.raises(ValueError, match="Unexpected"):
        reference.verify_sdk(sdk)


@pytest.mark.parametrize("check_only", [False, True])
def test_reference_report_distinguishes_hash_checks_from_inference(
    tmp_path, monkeypatch, check_only
):
    reference = load_reference()
    source, sdk = tmp_path / "source", tmp_path / "sdk"
    source.mkdir()
    sdk.mkdir()
    cases = [{"id": "中文 case", "state": {"问题": "退款"}, "questions": QUESTIONS}]
    input_path, output_path = tmp_path / "cases.json", tmp_path / "report.json"
    input_path.write_text(json.dumps(cases, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(
        reference, "verify_source", lambda path, model_id: {"weights": "pinned"}
    )
    monkeypatch.setattr(reference, "verify_sdk", lambda path: {"sdk.py": "pinned"})
    monkeypatch.setattr(reference, "copy_inputs", lambda *a: (source, sdk))
    monkeypatch.setattr(reference.common, "environment", lambda: {"packages": {}})
    seen = []

    def infer(args, cloned_source, cloned_sdk, inputs, report):
        seen.append(inputs)
        report.update(measured=True, cases=inputs)

    monkeypatch.setattr(reference, "infer", infer)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            reference.__file__,
            "--source",
            str(source),
            "--sdk-source",
            str(sdk),
            "--cases-json",
            str(input_path),
            "--output",
            str(output_path),
            "--threads",
            "1",
        ]
        + (["--check-only"] if check_only else []),
    )
    reference.main()
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["input_sha256"] == reference.sha256(input_path)
    assert report["driver_sha256"] == reference.sha256(Path(reference.__file__))
    assert report["sdk_commit"] == report["sdk_revision"] == reference.SDK_REVISION
    if check_only:
        assert report["status"] == "source_checked"
        assert not report["measured"] and not report["cases"] and not seen
    else:
        assert report["status"] == "passed" and report["measured"]
        assert report["source_unchanged"] and report["sdk_unchanged"]
        assert report["cases"] == cases and seen == [cases]
