"""Pinned author-runner boundaries and passive observation; no model execution."""

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def reference():
    path = Path(__file__).resolve().parents[1] / "scripts/windows_decider_reference.py"
    spec = importlib.util.spec_from_file_location("decider_reference_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def pinned_fixture(reference, tmp_path):
    profile = copy.deepcopy(reference.PROFILES["Mapika/decider-0.8b"])
    sdk, source = tmp_path / "sdk", tmp_path / "checkpoint"
    for directory, field in ((sdk, "sdk_hashes"), (source, "source_hashes")):
        for name in profile[field]:
            content = f"# pinned fixture: {name}\n".encode()
            file = directory / name
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_bytes(content)
            profile[field][name] = hashlib.sha256(content).hexdigest()
    return profile, sdk, source


def test_profiles_pin_independent_models_and_sdk_releases(reference):
    small = reference.PROFILES["Mapika/decider-0.8b"]
    larger = reference.PROFILES["Mapika/decider-2b"]
    assert len(small["source_hashes"]) == len(larger["source_hashes"]) == 7
    assert len(small["sdk_hashes"]) == 10
    assert len(larger["sdk_hashes"]) == 18
    assert (
        small["source_hashes"]["model.safetensors"]
        != larger["source_hashes"]["model.safetensors"]
    )
    assert (
        small["sdk_hashes"]["decider/model.py"]
        != larger["sdk_hashes"]["decider/model.py"]
    )
    assert small["typed_temperatures"] is False
    assert larger["typed_temperatures"] is True


def test_sdk_copy_does_not_execute_or_copy_existing_bytecode(
    reference, pinned_fixture, tmp_path
):
    profile, sdk, _ = pinned_fixture
    cache = sdk / "decider/__pycache__"
    cache.mkdir()
    (cache / "infer.cpython-312.pyc").write_bytes(b"untrusted stale bytecode")
    destination = tmp_path / "isolated"
    reference.copy_sdk(sdk, destination, profile)
    assert reference.verify_sdk(destination, profile) == profile["sdk_hashes"]
    assert not (destination / "decider/__pycache__").exists()
    assert {
        p.relative_to(destination).as_posix()
        for p in destination.rglob("*")
        if p.is_file()
    } == set(profile["sdk_hashes"])


@pytest.mark.parametrize(
    "where,name",
    [
        ("sdk", "decider/unpinned.py"),
        ("source", "adapter_config.json"),
        ("source", "special_tokens_map.json"),
    ],
)
def test_unknown_code_and_loader_overrides_fail_before_hashing(
    reference, pinned_fixture, monkeypatch, where, name
):
    profile, sdk, source = pinned_fixture
    directory = sdk if where == "sdk" else source
    (directory / name).write_text("override", encoding="utf-8")
    hashing = Mock(side_effect=AssertionError("layout must be checked first"))
    monkeypatch.setattr(reference, "verify", hashing)
    with pytest.raises(ValueError, match="Unexpected"):
        (reference.verify_sdk if where == "sdk" else reference.verify_checkpoint)(
            directory, profile
        )
    hashing.assert_not_called()


def test_changed_pinned_file_is_rejected(reference, pinned_fixture):
    profile, sdk, _ = pinned_fixture
    (sdk / "decider/model.py").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        reference.verify_sdk(sdk, profile)


@pytest.mark.parametrize(
    "payload",
    [
        '[{"id":"same","state":"x","questions":{"q":{}}},{"id":"same","state":"y","questions":{"q":{}}}]',
        '{"state":"x","questions":{"q":{},"q":{}}}',
        '{"state":NaN,"questions":{"q":{}}}',
    ],
)
def test_ambiguous_or_nonfinite_cases_are_rejected(reference, tmp_path, payload):
    path = tmp_path / "cases.json"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError):
        reference.read_cases(path)


def test_check_only_cannot_be_reported_as_measured(
    reference, pinned_fixture, tmp_path, monkeypatch
):
    profile, sdk, source = pinned_fixture
    cases = tmp_path / "cases.json"
    cases.write_text(
        json.dumps(
            [
                {
                    "id": "one",
                    "state": "中文",
                    "questions": {"q": {"type": "noul", "instructions": "退款？"}},
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    output = tmp_path / "report.json"
    monkeypatch.setitem(reference.PROFILES, "Mapika/decider-0.8b", profile)
    monkeypatch.setattr(
        reference, "infer", Mock(side_effect=AssertionError("inference must not run"))
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "runner",
            "--source",
            str(source),
            "--sdk-source",
            str(sdk),
            "--cases-json",
            str(cases),
            "--output",
            str(output),
            "--check-only",
        ],
    )
    reference.main()
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "source_checked"
    assert report["measured"] is False
    assert report["cases"] == []
    reference.infer.assert_not_called()


@pytest.mark.parametrize("target", ["source", "sdk", "cases"])
def test_report_cannot_overwrite_inputs(
    reference, pinned_fixture, tmp_path, monkeypatch, target
):
    _, sdk, source = pinned_fixture
    cases = tmp_path / "cases.json"
    output = {
        "source": source / "config.json",
        "sdk": sdk / "report.json",
        "cases": cases,
    }[target]
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "runner",
            "--source",
            str(source),
            "--sdk-source",
            str(sdk),
            "--cases-json",
            str(cases),
            "--output",
            str(output),
            "--check-only",
        ],
    )
    with pytest.raises(SystemExit):
        reference.parse_args()


def test_observer_passively_records_original_frames(reference):
    class Tensor:
        def __init__(self, dtype, shape):
            self.dtype, self.shape = dtype, shape

    def scaled_softmax(logits, temperature):
        return Tensor("torch.float32", logits.shape)

    class Model:
        def slot_logits(self, input_ids):
            hs, W = Tensor("torch.bfloat16", [1, 4]), Tensor("torch.bfloat16", [255, 4])
            assert hs.dtype == W.dtype
            return Tensor("torch.float32", [1, 255])

    class Decider:
        def __init__(self):
            self.m = Model()
            self.original = {"answers": {"q": {"noul": 0.75}}}

        def system_one(self):
            scaled_softmax(self.m.slot_logits(Tensor("torch.int64", [1, 64])), [1.624])
            flatp = [[0.25, 0.75]]
            index = [["q", "list", 0, 1]]
            rqs = {"q": {"type": "noul"}}
            items = [{"ids": [1, 2, 3], "slots": [2], "nopts": [2], "types": ["noul"]}]
            assert flatp and index and rqs and items
            return self.original

    author, observed = Decider(), {}
    hook = reference.observer(
        author, Decider, SimpleNamespace(Tensor=Tensor), observed, scaled_softmax
    )
    before = sys.getprofile()
    try:
        sys.setprofile(hook)
        output = author.system_one()
    finally:
        sys.setprofile(before)
    assert output is author.original
    assert observed["row_probabilities"] == [[0.25, 0.75]]
    assert observed["row_temperatures"] == [1.624]
    assert observed["linear_input_dtypes"] == ["torch.bfloat16"]
    assert observed["linear_weight_dtypes"] == ["torch.bfloat16"]
    assert observed["forward_input_shapes"] == [[1, 64]]
    assert (
        reference.validate_observation(
            observed,
            SimpleNamespace(max_fwd_tokens=1, dtype="bfloat16"),
            {"typed_temperatures": True},
        )
        == 1
    )
    assert sys.getprofile() is before


@pytest.mark.parametrize(
    "name",
    ["vllm_jev", "vllm_jev.windows_decider", "vllm_jev.decision", "vllm_jev.prompt"],
)
def test_reference_rejects_product_inference_imports(reference, name):
    with pytest.raises(RuntimeError, match="cannot import"):
        reference.BlockProductImports().find_spec(name)


def test_optional_dependency_probes_are_not_changed(reference):
    for name in ("torch", "transformers", "vllm", "mlx"):
        assert reference.BlockProductImports().find_spec(name) is None
