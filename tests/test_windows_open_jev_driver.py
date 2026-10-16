"""Reference-driver source boundaries, using tiny files and no model imports."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def driver():
    path = Path(__file__).resolve().parents[1] / "scripts/check_windows_open_jev.py"
    spec = importlib.util.spec_from_file_location("open_jev_driver_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def source(driver, tmp_path, monkeypatch):
    """Preserve all 18 release paths, substituting independently hashed tiny bytes."""
    root = tmp_path / "source"
    payloads = {name: f"fixture: {name}\n".encode() for name in driver.SOURCE_HASHES}
    hashes = {
        name: hashlib.sha256(value).hexdigest() for name, value in payloads.items()
    }
    for name, value in payloads.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
    monkeypatch.setattr(driver, "SOURCE_HASHES", hashes)
    monkeypatch.setattr(
        driver,
        "CHECKPOINT_FILES",
        {
            name: (
                len(payloads[f"package/checkpoint/{name}"]),
                hashes[f"package/checkpoint/{name}"],
            )
            for name in driver.CHECKPOINT_FILES
        },
    )
    base_files = {}
    for name in driver.BASE_FILES:
        payload = payloads.get(f"base/{name}", f"fixture: base/{name}\n".encode())
        base_files[name] = (len(payload), "sha256", hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(driver, "BASE_FILES", base_files)
    return root


def test_exact_source_and_hub_bookkeeping_are_accepted(driver, source):
    bookkeeping = source / "base/.cache/huggingface"
    bookkeeping.mkdir(parents=True)
    (bookkeeping / "metadata").write_text("download metadata")
    assert len(driver.verify_source(source)) == 18
    assert driver.verify_source(source) == driver.SOURCE_HASHES


@pytest.mark.parametrize(
    "relative,is_directory",
    [
        ("base/README.md", False),
        ("base/.gitattributes", False),
        ("base/preprocessor_config.json", False),
        ("base/video_preprocessor_config.json", False),
        ("base/adapter_config.json", False),
        ("base/templates", True),
        ("base/.cache", False),
        ("base/LICENSE", True),
        ("package/checkpoint/config.json", False),
        ("package/checkpoint/.cache", True),
        ("package/checkpoint/adapter/adapter_model.bin", False),
        ("package/checkpoint/adapter/.cache", True),
    ],
)
def test_incompatible_product_entries_fail_before_hashing(
    driver, source, monkeypatch, relative, is_directory
):
    entry = source / relative
    if entry.is_file():
        entry.unlink()
    if is_directory:
        entry.mkdir()
    else:
        entry.write_text("extra source entry")
    checkpoint = Mock(side_effect=AssertionError("hashing started before layout check"))
    monkeypatch.setattr(driver, "verify_checkpoint", checkpoint)
    with pytest.raises(ValueError, match="unexpected product source entry"):
        driver.verify_source(source)
    checkpoint.assert_not_called()


def test_missing_license_fails_before_hashing(driver, source, monkeypatch):
    (source / "LICENSE-CODE").unlink()
    checkpoint = Mock(side_effect=AssertionError("hashing started before file check"))
    monkeypatch.setattr(driver, "verify_checkpoint", checkpoint)
    with pytest.raises(ValueError, match="source artifact missing: LICENSE-CODE"):
        driver.verify_source(source)
    checkpoint.assert_not_called()


def test_official_cache_files_stay_supported_only_in_hf_cache(driver, source):
    for name in set(driver.BASE_FILES) - driver.BASE_REQUIRED:
        (source / "base" / name).write_bytes(f"fixture: base/{name}\n".encode())
    assert len(driver.verify_base(source / "base")) == 13
    with pytest.raises(ValueError, match="unexpected product source entry"):
        driver.verify_source(source)


def test_allowed_file_checksum_is_still_enforced(driver, source):
    (source / "base/config.json").write_text("corrupt")
    with pytest.raises(ValueError, match="base file size mismatch"):
        driver.verify_source(source)


def test_clone_contains_only_verified_product_files(driver, source, tmp_path):
    destination = tmp_path / "private product"
    driver.clone_source(source, destination)
    assert driver.verify_source(destination) == driver.SOURCE_HASHES
    assert {
        path.relative_to(destination).as_posix()
        for path in destination.rglob("*")
        if path.is_file()
    } == driver.SOURCE_HASHES.keys()


@pytest.fixture
def reference_fixture(driver, source, tmp_path, monkeypatch):
    sdk = tmp_path / "sdk/jev"
    sdk.mkdir(parents=True)
    sdk_hashes = {}
    for name in driver.SDK_HASHES:
        content = f"verified fixture {name}".encode()
        (sdk / name).write_bytes(content)
        sdk_hashes[name] = hashlib.sha256(content).hexdigest()
    monkeypatch.setattr(driver, "SDK_HASHES", sdk_hashes)
    cache = tmp_path / "hf-cache"
    snapshot = cache / "models--Qwen--Qwen3.5-2B/snapshots" / driver.BASE_REVISION
    snapshot.mkdir(parents=True)
    for name in driver.BASE_REQUIRED:
        (snapshot / name).write_bytes((source / "base" / name).read_bytes())
    environment = {
        "python": "3.12.5",
        "platform": "Windows-11",
        "packages": {
            "torch": "2.8.0+cpu",
            "transformers": "5.10.2",
            "peft": "0.19.1",
            "accelerate": "1.13.0",
        },
    }
    monkeypatch.setattr(driver, "environment", lambda: copy.deepcopy(environment))
    checkpoint = hashlib.sha256()
    for name in sorted(driver.CHECKPOINT_FILES):
        checkpoint.update(str(Path(name)).encode() + b"\0")
        checkpoint.update((source / "package/checkpoint" / name).read_bytes())
    answers = {
        "department": {
            "type": "choice",
            "choice": "billing",
            "probabilities": {"billing": 0.75, "technical": 0.25},
            "confidence": 0.5,
        },
        "refund": {"type": "noul", "noul": 0.7},
        "urgency": {
            "type": "score",
            "score": 1.1,
            "probabilities": {"0": 0.2, "1": 0.5, "2": 0.3},
            "confidence": 0.25,
            "legend": dict(enumerate(driver.QUESTIONS["urgency"]["criteria"])),
        },
    }
    answers["urgency"]["legend"] = {
        str(key): value for key, value in answers["urgency"]["legend"].items()
    }
    report = {
        "status": "passed",
        "measured": True,
        "source_unchanged": True,
        "hf_snapshot_unchanged": True,
        "private_checkpoint_unchanged": True,
        "model": driver.MODEL,
        "revision": driver.REVISION,
        "base_revision": driver.BASE_REVISION,
        "sdk_commit": driver.SDK_COMMIT,
        "sdk_source_sha256": sdk_hashes,
        "source_sha256": dict(driver.SOURCE_HASHES),
        "hf_snapshot_sha256": driver.verify_base(snapshot),
        "threads": 4,
        "selected_case_ids": [name for name, _ in driver.CASES],
        "reference": "Unmodified jev.serving.load_predictor().predict(), Torch CPU",
        "environment": environment,
        "inference": {
            "dtype": "bfloat16",
            "head_dtype": "float32",
            "batch_size": 1,
            "prefix_cache": False,
            "backend": "torch",
            "sdk_patches": False,
            "max_sequence_tokens": 1024,
            "actual_embedding_dtype": "torch.bfloat16",
            "actual_head_dtype": "torch.float32",
            "actual_backbone_dtypes": ["torch.bfloat16", "torch.float32"],
            "actual_text_backbone": "Qwen3_5TextModel",
            "cuda_available": False,
            "actual_parameter_devices": ["cpu"],
            "backbone_parameter_bytes": 1000,
            "head_parameter_bytes": 8196,
        },
        "cases": [],
        "cold_load_seconds": 1.0,
        "wall_seconds": 10.0,
        "peak_wset_bytes": 100,
        "memory": [
            {"stage": name, "peak_wset_bytes": 100}
            for name in [
                "before_load",
                "after_load",
                *[f"after_{name}" for name, _ in driver.CASES],
                "final",
            ]
        ],
    }
    for name, state in driver.CASES:
        report["cases"].append(
            {
                "id": name,
                "state": state,
                "questions": copy.deepcopy(driver.QUESTIONS),
                "answers": copy.deepcopy(answers),
                "seconds": 1.0,
                "response": {
                    "model": driver.BASE_MODEL,
                    "answers": copy.deepcopy(answers),
                    "usage": {"output_tokens": 0, "input_tokens": 100},
                    "metadata": {
                        "method": "lora_decision_head",
                        "temperature": 1.518796342858676,
                        "candidate_sequences": 6,
                        "prefix_cache": {
                            "enabled": False,
                            "mode": "independent_candidates",
                        },
                        "base_revision": driver.BASE_REVISION,
                        "max_length": 1024,
                        "backbone_parameter_dtypes": [
                            "torch.bfloat16",
                            "torch.float32",
                        ],
                        "checkpoint_sha256": checkpoint.hexdigest(),
                        "inference_seconds": 1.0,
                    },
                },
            }
        )
    path = tmp_path / "original-reference.json"
    driver.save(path, report)
    args = SimpleNamespace(
        source=source,
        sdk_source=sdk.parent,
        hf_cache=cache,
        threads=4,
        dtype="bfloat16",
        case=None,
    )
    return args, path, report


def run_reuse_main(driver, monkeypatch, reference_fixture):
    args, path, _ = reference_fixture
    service = Mock()
    reference = Mock(side_effect=AssertionError("reused reference must not run again"))
    monkeypatch.setattr(driver, "service_run", service)
    monkeypatch.setattr(driver, "run_reference_process", reference)
    monkeypatch.setattr(driver.sys, "platform", "win32")
    probe = Mock()
    probe.__enter__ = Mock(return_value=probe)
    probe.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(driver.socket, "socket", Mock(return_value=probe))
    monkeypatch.setattr(
        driver.sys,
        "argv",
        [
            "check_windows_open_jev.py",
            "--source",
            str(args.source),
            "--workspace",
            str(path.parent / "workspace"),
            "--sdk-source",
            str(args.sdk_source),
            "--hf-cache",
            str(args.hf_cache),
            "--output",
            str(path.parent / "result.json"),
            "--reference-report",
            str(path),
        ],
    )
    return service, reference


def test_matching_reference_reuse_runs_only_two_services(
    driver, monkeypatch, reference_fixture
):
    args, path, _ = reference_fixture
    service, reference = run_reuse_main(driver, monkeypatch, reference_fixture)
    driver.main()
    reference.assert_not_called()
    assert service.call_count == 2
    report = json.loads((path.parent / "result.json").read_text(encoding="utf-8"))
    assert report["reference_provenance"] == {
        "path": str(path.resolve()),
        "sha256": driver.digest(path),
        "reused": True,
        "rerun_this_invocation": False,
    }
    assert report["status"] == "passed"
    assert [row["id"] for row in service.call_args.args[1]["cases"]] == [
        name for name, _ in driver.CASES
    ]


@pytest.mark.parametrize(
    "keys,value",
    [
        (("status",), "running"),
        (("measured",), False),
        (("measured",), 1),
        (("source_unchanged",), False),
        (("hf_snapshot_unchanged",), False),
        (("private_checkpoint_unchanged",), False),
        (("model",), "other/model"),
        (("revision",), "other"),
        (("base_revision",), "other"),
        (("sdk_commit",), "other"),
        (("sdk_source_sha256", "model.py"), "0" * 64),
        (("source_sha256", "LICENSE"), "0" * 64),
        (("hf_snapshot_sha256", "config.json"), "0" * 64),
        (("threads",), 8),
        (("environment", "packages", "torch"), "2.6.0+cpu"),
        (("environment", "packages", "transformers"), "5.17.0"),
        (("environment", "packages", "peft"), "0.18.1"),
        (("environment", "packages", "accelerate"), "1.12.0"),
        (("inference", "dtype"), "float32"),
        (("inference", "actual_head_dtype"), "torch.bfloat16"),
        (("inference", "actual_embedding_dtype"), "torch.float32"),
        (("inference", "actual_parameter_devices"), ["cuda"]),
        (("inference", "batch_size"), 2),
        (("inference", "prefix_cache"), True),
        (("inference", "max_sequence_tokens"), 2048),
        (("inference", "sdk_patches"), True),
        (("selected_case_ids",), ["text_refund"]),
        (("cases",), []),
        (("cases", 0, "id"), "text_login"),
        (("cases", 0, "state"), "different state"),
        (("cases", 0, "questions", "refund", "instructions"), "different instruction"),
        (("cases", 0, "answers", "refund", "noul"), 0.1),
        (("cases", 0, "response", "metadata", "prefix_cache", "enabled"), True),
        (("cases", 0, "response", "metadata", "checkpoint_sha256"), "0" * 64),
        (("memory",), []),
        (("peak_wset_bytes",), 200),
    ],
)
def test_mismatched_reference_fails_before_any_cli(
    driver, monkeypatch, reference_fixture, keys, value
):
    args, path, report = reference_fixture
    target = report
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = value
    driver.save(path, report)
    service, reference = run_reuse_main(driver, monkeypatch, reference_fixture)
    with pytest.raises((ValueError, AssertionError)):
        driver.main()
    service.assert_not_called()
    reference.assert_not_called()


@pytest.mark.parametrize("flag", ["--reference-only", "--reference-child"])
def test_reference_reuse_rejects_incompatible_modes(
    driver, monkeypatch, reference_fixture, flag
):
    service, reference = run_reuse_main(driver, monkeypatch, reference_fixture)
    driver.sys.argv.append(flag)
    with pytest.raises(SystemExit) as error:
        driver.main()
    assert error.value.code == 2
    service.assert_not_called()
    reference.assert_not_called()


def test_current_sdk_tampering_is_rejected(driver, monkeypatch, reference_fixture):
    args, path, _ = reference_fixture
    (args.sdk_source / "jev/model.py").write_text("tampered")
    service, reference = run_reuse_main(driver, monkeypatch, reference_fixture)
    with pytest.raises(ValueError, match="current author source"):
        driver.main()
    service.assert_not_called()
    reference.assert_not_called()


@pytest.mark.parametrize(
    "content", ['{"status":"passed","status":"passed"}', '{"value": NaN}', '{"status":']
)
def test_malformed_reference_is_rejected(
    driver, monkeypatch, reference_fixture, content
):
    _, path, _ = reference_fixture
    path.write_text(content, encoding="utf-8")
    service, reference = run_reuse_main(driver, monkeypatch, reference_fixture)
    with pytest.raises(ValueError):
        driver.main()
    service.assert_not_called()
    reference.assert_not_called()
