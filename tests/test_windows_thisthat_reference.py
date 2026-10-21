"""Independent release selection and source isolation; no real model loading."""

import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "independent_thisthat_reference",
    Path(__file__).resolve().parents[1] / "scripts/windows_thisthat_reference.py",
)
reference = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reference)


@pytest.fixture
def release_inputs(tmp_path, monkeypatch):
    sources, models = {}, {}
    for model, profile in reference.MODELS.items():
        source = tmp_path / model.rsplit("/", 1)[1]
        source.mkdir()
        (source / "model.safetensors").write_bytes(model.encode())
        (source / "tokenizer_config.json").write_text(
            json.dumps({"local_files_only": model.endswith("1.0")}), encoding="utf-8"
        )
        models[model] = {
            "revision": profile["revision"],
            "source_hashes": {p.name: reference.sha256(p) for p in source.iterdir()},
        }
        sources[model] = source
    monkeypatch.setattr(reference, "MODELS", models)
    sdk = tmp_path / "sdk"
    package = sdk / "thisthat"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("# fixed original SDK\n", encoding="utf-8")
    monkeypatch.setattr(
        reference,
        "SDK_HASHES",
        {
            "thisthat/__init__.py": reference.sha256(package / "__init__.py"),
        },
    )
    return sources, sdk


@pytest.mark.parametrize("model", list(reference.MODELS))
def test_release_hashes_cannot_be_substituted(release_inputs, model):
    sources, _ = release_inputs
    assert (
        reference.verify_source(sources[model], model)
        == reference.MODELS[model]["source_hashes"]
    )
    for other, source in sources.items():
        if other != model:
            with pytest.raises(ValueError, match="SHA256"):
                reference.verify_source(source, model)


@pytest.mark.parametrize("model", list(reference.MODELS))
def test_copy_inputs_preserves_selected_raw_config_and_omits_unpinned_docs(
    release_inputs,
    tmp_path,
    model,
):
    sources, sdk = release_inputs
    source = sources[model]
    (source / "README.md").write_text("not a loader input", encoding="utf-8")
    workspace = tmp_path / "private"
    workspace.mkdir()
    cloned, author = reference.copy_inputs(source, sdk, workspace, model)
    assert (cloned / "tokenizer_config.json").read_bytes() == (
        source / "tokenizer_config.json"
    ).read_bytes()
    assert not (cloned / "README.md").exists()
    assert reference.verify_source(cloned, model) == reference.verify_source(
        source, model
    )
    assert reference.verify_sdk(author) == reference.verify_sdk(sdk)


@pytest.mark.parametrize("selected", [None, *reference.MODELS])
def test_check_only_records_explicit_release_without_inference(
    release_inputs,
    tmp_path,
    monkeypatch,
    selected,
):
    sources, sdk = release_inputs
    model = selected or reference.MODEL_ID
    cases = tmp_path / "cases.json"
    cases.write_text(
        json.dumps(
            [
                {
                    "id": "中文",
                    "state": {"文本": "hello"},
                    "questions": {"q": {"type": "noul", "instructions": "Greeting?"}},
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    output = tmp_path / "report.json"
    args = [
        "reference.py",
        "--source",
        str(sources[model]),
        "--sdk-source",
        str(sdk),
        "--cases-json",
        str(cases),
        "--output",
        str(output),
        "--check-only",
    ]
    if selected:
        args += ["--model", selected]
    monkeypatch.setattr(reference.sys, "argv", args)
    monkeypatch.setattr(reference.sys, "dont_write_bytecode", True)
    monkeypatch.setattr(reference.os, "environ", reference.os.environ.copy())
    monkeypatch.setattr(
        reference, "infer", lambda *a: pytest.fail("must not load a model")
    )
    reference.main()
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "source_checked"
    assert report["measured"] is False
    assert report["model_id"] == model
    assert report["revision"] == reference.MODELS[model]["revision"]
    assert report["source_hashes"] == reference.MODELS[model]["source_hashes"]
    assert report["input_sha256"] == reference.sha256(cases)
    assert report["cases"] == []


def test_release_pins_have_three_distinct_weights_and_exact_10_config():
    assert reference.MODEL_ID.endswith("1.2")
    assert (
        len(
            {p["source_hashes"]["model.safetensors"] for p in reference.MODELS.values()}
        )
        == 3
    )
    assert reference.MODELS["flock-io/this-that-model-1.0"]["source_hashes"][
        "tokenizer_config.json"
    ] == ("353c083805d863719e54973335a38e252a92e1ac64d32b83bb0b333cc1644ef2")
    with pytest.raises(ValueError, match="Unsupported"):
        reference.model_profile("flock-io/this-that-model-9.0")
