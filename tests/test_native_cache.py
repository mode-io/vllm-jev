"""CPU regressions for selecting a previously prepared checkpoint."""

import json

import pytest

from vllm_jev import clef_export, native_models


@pytest.mark.parametrize("model_id", sorted(clef_export.MODELS))
def test_clef_cache_belongs_to_requested_model(tmp_path, monkeypatch, model_id):
    output = tmp_path / "checkpoint" / model_id
    output.mkdir(parents=True)
    manifest = output / "clef_manifest.json"
    verified = []
    monkeypatch.setattr(
        clef_export,
        "verify_clef",
        lambda path, *, full: verified.append((path, full)),
    )
    monkeypatch.setattr(
        native_models,
        "snapshot_download",
        lambda **kwargs: pytest.fail("an existing checkpoint must not download"),
    )

    other_model = next(name for name in clef_export.MODELS if name != model_id)
    manifest.write_text(json.dumps({"source_repository": other_model}))
    with pytest.raises(ValueError, match="another repository"):
        native_models._prepare(model_id, tmp_path, "auto")
    assert not verified

    manifest.write_text(json.dumps({"source_repository": model_id}))
    assert native_models._prepare(model_id, tmp_path, "auto") == output
    assert verified == [(output, True)]
