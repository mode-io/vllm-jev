"""Pinned reference/acceptance boundaries. No DLLs or model weights are loaded."""

import argparse
import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "mica_driver_test", ROOT / "scripts/check_windows_mica.py"
)
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)
reference = driver.reference


@pytest.fixture
def tiny_source(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    pins = {}
    for name in ("model.gguf", "tokenizer.json", "runtime/llama.dll"):
        path = source / name
        path.parent.mkdir(exist_ok=True)
        path.write_text(name, encoding="utf-8")
        pins[name] = reference.digest(path)
    monkeypatch.setattr(reference, "SOURCE_HASHES", pins)
    return source, pins


@pytest.mark.parametrize(
    "name",
    [
        "runtime/ggml-cuda.dll",
        "runtime/ggml-rpc.dll",
        "runtime/subdir",
        "adapter_config.json",
        "model.py",
    ],
)
def test_source_extra_loader_input_is_rejected(tiny_source, name):
    source, _ = tiny_source
    target = source / name
    if name.endswith("subdir"):
        target.mkdir()
    else:
        target.write_text("unexpected", encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected"):
        reference.verify_source(source)


def test_source_tamper_fails_even_with_expected_filenames(tiny_source):
    source, _ = tiny_source
    (source / "runtime/llama.dll").write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA256"):
        reference.verify_source(source)


def test_clone_is_whitelist_only_and_config_is_not_hardlinked(tiny_source, tmp_path):
    source, pins = tiny_source
    (source / "README.md").write_text("ignored document", encoding="utf-8")
    target = tmp_path / "private"
    reference.clone_pinned(source, target, pins)
    assert reference.verify_source(target) == pins
    assert not (target / "README.md").exists()
    (target / "tokenizer.json").write_text("changed private copy", encoding="utf-8")
    assert reference.verify_source(source) == pins


def test_reference_uses_original_sdk_byte_pins_and_never_product_import():
    text = (ROOT / "scripts/windows_mica_reference.py").read_text(encoding="utf-8")
    assert "from vllm_jev" not in text
    assert "import vllm_jev" not in text
    assert len(reference.SOURCE_HASHES) == 27
    assert (
        reference.SOURCE_HASHES[reference.WEIGHT]
        == "7fbd1be2293ba158157bbbdcdc4ea57569bd9ed53901ed1766f81264d9da8292"
    )
    assert "runtime/ggml-rpc.dll" not in reference.SOURCE_HASHES


@pytest.mark.parametrize("text", ['{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}'])
def test_reference_json_cannot_hide_duplicate_or_nonfinite_fields(text):
    with pytest.raises(ValueError):
        reference.strict_json(text)


def test_observer_records_return_values_without_replacing_original_method():
    class Judge:
        def _logits_at(self):
            return [1.0, 2.0]

        def _judge(self):
            prompts, labels = [[10, 11]], [[1, 2]]
            self._logits_at()
            return prompts, labels

    judge = Judge()
    original = Judge._judge
    events = {"logits": [], "prompts": [], "labels": []}
    previous = sys.getprofile()
    try:
        sys.setprofile(reference.observe_judge(judge, events))
        assert judge._judge() == ([[10, 11]], [[1, 2]])
    finally:
        sys.setprofile(previous)
    assert Judge._judge is original
    assert events == {"logits": [[1.0, 2.0]], "prompts": [[10, 11]], "labels": [[1, 2]]}


def test_canonical_uses_true_second_and_expected_score_not_author_argmax():
    case = {
        "questions": {
            "n": {"type": "noul", "instructions": "q"},
            "s": {
                "type": "score",
                "instructions": "q",
                "criteria": ["low", "mid", "high"],
            },
            "c": {
                "type": "choice",
                "instructions": "q",
                "criteria": {"甲": "a", "乙": "b"},
            },
        }
    }
    output = driver.canonical(
        case, {"n": [0.8, 0.2], "s": [0.2, 0.5, 0.3], "c": [0.8, 0.2]}
    )
    assert output["n"]["noul"] == 0.2
    assert output["s"]["score"] == pytest.approx(1.1)
    assert output["c"]["confidence"] == pytest.approx(0.6)


@pytest.fixture
def good_report(tmp_path, monkeypatch):
    case = {
        "id": "中文",
        "state": {"a": "值"},
        "questions": {"q": {"type": "noul", "instructions": "True?"}},
    }
    monkeypatch.setattr(driver, "CASES", [case])
    cases_path = tmp_path / "cases.json"
    reference.save(cases_path, [case])
    monkeypatch.setattr(reference, "verify_sdk", lambda source: reference.SDK_HASHES)
    monkeypatch.setattr(
        driver,
        "reference_versions",
        lambda *args: {"transformers": "5.5.0", "torch": "not-installed"},
    )
    monkeypatch.setattr(
        driver.helpers,
        "_reference_python_identity",
        lambda *args: {"python": "version", "executable": "author.exe"},
    )
    value = {
        "status": "passed",
        "measured": True,
        "model_id": reference.MODEL_ID,
        "revision": reference.REVISION,
        "sdk_revision": reference.SDK_REVISION,
        "runtime_revision": "b11010",
        "source_hashes": reference.SOURCE_HASHES,
        "sdk_hashes": reference.SDK_HASHES,
        "driver_sha256": reference.digest(driver.RUNNER),
        "input_sha256": reference.digest(cases_path),
        "versions": {"transformers": "5.5.0", "torch": "not-installed"},
        "environment_identity": {"python": "version", "executable": "author.exe"},
        "source_unchanged": True,
        "sdk_unchanged": True,
        "isolated_source_unchanged": True,
        "isolated_sdk_unchanged": True,
        "judge_closed": True,
        "private_workspace_removed": True,
        "runtime_source_isolated": False,
        "isolated_source_hashes": {
            name: checksum
            for name, checksum in reference.SOURCE_HASHES.items()
            if not name.startswith("runtime/")
        },
        "runtime": {
            "tokenizer_runtime": "transformers-tokenizer-only",
            "torch_imported": False,
            "torch_installed": False,
            "runtime": "llama.cpp",
            "runtime_revision": "b11010",
            "quantization": "Q5_K_M",
            "device": "cpu",
            "n_gpu_layers": 0,
            "threads": 4,
            "context_tokens": 1024,
            "n_seq_max": 1,
            "n_batch": 1024,
            "n_ubatch": 512,
            "flash_attn": -1,
            "max_sequence_tokens": 1024,
            "row_batch_size": 1,
            "share_prefix": False,
            "session_id": None,
            "temperature": reference.TEMPERATURE,
            "temperatures": {
                kind: reference.TEMPERATURE for kind in ("noul", "choice", "score")
            },
            "noul_logit_bias": 0.0,
            "backend_registration_count": 1,
        },
        "cases": [
            {
                **case,
                "probabilities": {"q": [0.5, 0.5]},
                "raw_logits": {"q": [1.0, 1.0]},
                "evidence": {
                    "q": {
                        "prompt": "observed",
                        "token_ids": [1, 2],
                        "label_ids": [3, 4],
                        "row": {"id": "q", "kind": "noul"},
                        "prefix_cache_stats": {
                            "restores": 0,
                            "restored_prefix_tokens": 0,
                            "prefix_tokens_decoded": 0,
                            "checkpoint_bytes": 0,
                        },
                    }
                },
            }
        ],
    }
    args = argparse.Namespace(
        sdk_source=tmp_path, reference_python=tmp_path / "python.exe", threads=4
    )
    path = tmp_path / "report.json"
    return value, path, args, cases_path


def validate_fixture(data):
    value, path, args, cases_path = data
    reference.save(path, value)
    return driver.validate_reference(path, args, cases_path, reference.SOURCE_HASHES)


def test_correct_reference_provenance_and_probability_recomputation(good_report):
    assert validate_fixture(good_report)["status"] == "passed"


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("revision", "wrong"),
        ("sdk_revision", "wrong"),
        ("runtime_revision", "b9999"),
        ("driver_sha256", "wrong"),
        ("input_sha256", "wrong"),
        ("measured", False),
        ("source_unchanged", False),
        ("judge_closed", False),
        ("private_workspace_removed", False),
        ("versions", {"transformers": "5.17.0"}),
        ("source_hashes", {}),
        ("sdk_hashes", {}),
    ],
)
def test_reference_reuse_rejects_different_provenance(good_report, field, replacement):
    good_report[0][field] = replacement
    with pytest.raises(ValueError, match="provenance"):
        validate_fixture(good_report)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("quantization", "Q4_K_M"),
        ("threads", 8),
        ("context_tokens", 2048),
        ("n_seq_max", 2),
        ("share_prefix", True),
        ("session_id", "cache"),
        ("n_gpu_layers", -1),
    ],
)
def test_reference_reuse_rejects_different_execution(good_report, field, replacement):
    good_report[0]["runtime"][field] = replacement
    with pytest.raises(ValueError, match="runtime"):
        validate_fixture(good_report)


def test_reference_reuse_rejects_probs_not_computed_from_original_logits(good_report):
    good_report[0]["cases"][0]["raw_logits"]["q"] = [1.0, 2.0]
    with pytest.raises(AssertionError, match="decision differs"):
        validate_fixture(good_report)


def test_reference_reuse_rejects_changed_case_even_if_case_id_matches(good_report):
    good_report[0]["cases"][0]["state"] = "different"
    with pytest.raises(ValueError, match="case content"):
        validate_fixture(good_report)


def test_reference_command_uses_own_runner_and_fixed_inputs(tmp_path):
    args = argparse.Namespace(
        reference_python=tmp_path / "author/python.exe",
        source=tmp_path / "中文 模型",
        sdk_source=tmp_path / "author-sdk",
        threads=4,
    )
    command = driver.command_for_reference(
        args, tmp_path / "cases.json", tmp_path / "reference.json"
    )
    assert command[:3] == [str(args.reference_python), "-B", str(driver.RUNNER)]
    assert command[command.index("--source") + 1] == str(args.source)
    assert not any("windows_mica.py" in item for item in command)


def test_wrong_product_revision_or_backend_cannot_pass():
    response = {
        "model": reference.MODEL_ID,
        "metadata": {**driver.METADATA, "threads": 4},
    }
    driver.verify_metadata(response, 4)
    bad = copy.deepcopy(response)
    bad["metadata"]["backend"] = "pytorch"
    with pytest.raises(ValueError, match="backend"):
        driver.verify_metadata(bad, 4)


def test_reference_check_only_does_not_infer(tiny_source, tmp_path, monkeypatch):
    source, _ = tiny_source
    sdk = tmp_path / "sdk"
    sdk.mkdir()
    monkeypatch.setattr(reference, "verify_sdk", lambda root: {})
    monkeypatch.setattr(reference.importlib.metadata, "version", lambda name: "test")
    monkeypatch.setattr(
        reference,
        "infer",
        lambda *args: pytest.fail("check-only must not run inference"),
    )
    cases = tmp_path / "cases.json"
    reference.save(
        cases,
        [
            {
                "id": "short",
                "state": "hi",
                "questions": {"q": {"type": "noul", "instructions": "x"}},
            }
        ],
    )
    output = tmp_path / "checked.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "reference",
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
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["status"] == "checked"
    assert result["measured"] is False


def test_author_clones_model_but_keeps_dlls_at_verified_original_path(
    tiny_source, tmp_path, monkeypatch
):
    source, pins = tiny_source
    sdk = tmp_path / "sdk"
    (sdk / "mica").mkdir(parents=True)
    (sdk / "mica/__init__.py").write_text("# fixed", encoding="utf-8")
    monkeypatch.setattr(
        reference,
        "SDK_HASHES",
        {"mica/__init__.py": reference.digest(sdk / "mica/__init__.py")},
    )
    monkeypatch.setattr(reference.importlib.metadata, "version", lambda name: "test")
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "PYTHONDONTWRITEBYTECODE"):
        monkeypatch.setenv(name, "1")
    seen = {}

    def infer(checkpoint, author, cases, threads, report, output, runtime_dir):
        seen.update(checkpoint=checkpoint, author=author)
        assert checkpoint != source
        assert runtime_dir == source / "runtime"
        assert not (checkpoint / "runtime").exists()
        assert (checkpoint / "tokenizer.json").read_bytes() == (
            source / "tokenizer.json"
        ).read_bytes()
        assert (author / "mica/__init__.py").read_bytes() == (
            sdk / "mica/__init__.py"
        ).read_bytes()
        report["judge_closed"] = True

    monkeypatch.setattr(reference, "infer", infer)
    cases, output = tmp_path / "cases.json", tmp_path / "result.json"
    reference.save(
        cases,
        [
            {
                "id": "short",
                "state": "hi",
                "questions": {"q": {"type": "noul", "instructions": "x"}},
            }
        ],
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "reference",
            "--source",
            str(source),
            "--sdk-source",
            str(sdk),
            "--cases-json",
            str(cases),
            "--output",
            str(output),
        ],
    )
    reference.main()
    result = reference.strict_json(output.read_text(encoding="utf-8"))
    assert result["runtime_source_isolated"] is False
    assert result["isolated_source_hashes"] == {
        name: value for name, value in pins.items() if not name.startswith("runtime/")
    }
    assert not seen["checkpoint"].exists()
    assert not seen["author"].exists()
    assert result["source_unchanged"] is True
    assert (source / "runtime/llama.dll").is_file()


def test_tokenizer_only_setup_precedes_backend_import_and_records_actual_state(
    monkeypatch,
):
    monkeypatch.setattr(reference.sys, "modules", {})
    monkeypatch.setattr(reference.importlib.util, "find_spec", lambda name: None)
    monkeypatch.setenv("USE_TORCH", "1")
    monkeypatch.delenv("KMP_DUPLICATE_LIB_OK", raising=False)
    reference.configure_tokenizer_only()
    assert reference.tokenizer_runtime() == {
        "tokenizer_runtime": "transformers-tokenizer-only",
        "torch_imported": False,
        "torch_installed": False,
    }
    assert "KMP_DUPLICATE_LIB_OK" not in reference.os.environ
    assert reference.os.environ["USE_TORCH"] == "1"


@pytest.mark.parametrize(
    "module", ["torch", "torch._C", "transformers", "transformers.utils"]
)
def test_tokenizer_only_setup_rejects_already_initialized_framework(
    monkeypatch, module
):
    monkeypatch.setattr(reference.sys, "modules", {module: object()})
    with pytest.raises(RuntimeError, match="fresh process"):
        reference.configure_tokenizer_only()


def test_torch_imported_after_setup_cannot_be_reported_as_tokenizer_only(monkeypatch):
    monkeypatch.setattr(reference.sys, "modules", {"torch": object()})
    monkeypatch.setenv("USE_TORCH", "0")
    with pytest.raises(RuntimeError, match="Torch entered"):
        reference.tokenizer_runtime()


def test_reference_rejects_openmp_duplicate_library_bypass(monkeypatch):
    monkeypatch.setattr(reference.sys, "modules", {})
    monkeypatch.setenv("KMP_DUPLICATE_LIB_OK", "TRUE")
    with pytest.raises(RuntimeError, match="bypass"):
        reference.configure_tokenizer_only()


def test_reuse_rejects_torch_enabled_baseline(good_report):
    good_report[0]["runtime"]["torch_imported"] = True
    with pytest.raises(ValueError, match="runtime"):
        validate_fixture(good_report)


def test_installed_torch_is_rejected_even_when_use_torch_is_zero(monkeypatch):
    monkeypatch.setattr(reference.sys, "modules", {})
    monkeypatch.setattr(
        reference.importlib.util,
        "find_spec",
        lambda name: object() if name == "torch" else None,
    )
    monkeypatch.setenv("USE_TORCH", "0")
    monkeypatch.delenv("KMP_DUPLICATE_LIB_OK", raising=False)
    with pytest.raises(RuntimeError, match="torch to be uninstalled"):
        reference.configure_tokenizer_only()


def test_reference_checks_effective_batch_not_author_requested_batch(good_report):
    assert driver.REQUESTED_N_BATCH == 2048
    assert driver.EFFECTIVE_N_BATCH == 1024
    assert validate_fixture(good_report)["runtime"]["n_batch"] == 1024
    good_report[0]["runtime"]["n_batch"] = driver.REQUESTED_N_BATCH
    with pytest.raises(ValueError, match="runtime differs: n_batch"):
        validate_fixture(good_report)


@pytest.mark.parametrize(
    ("name", "wrong"),
    [
        ("n_batch", 2048),
        ("tokenizers_version", "0.23.0"),
        ("jinja2_version", "3.0.0"),
        ("transformers_imported", True),
    ],
)
def test_product_metadata_requires_actual_batch_and_tokenizer_runtime(name, wrong):
    response = {
        "model": reference.MODEL_ID,
        "metadata": {**driver.METADATA, "threads": 4},
    }
    driver.verify_metadata(response, 4)
    response["metadata"][name] = wrong
    with pytest.raises(ValueError, match=f"metadata differs: {name}"):
        driver.verify_metadata(response, 4)
