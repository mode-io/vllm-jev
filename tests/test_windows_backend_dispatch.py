"""Merged backend selection must keep Windows on its explicit CPU route."""

import sys

import pytest

from vllm_jev import backend, cli

# Initialize the real host runtime before simulating sys.platform.
torch = pytest.importorskip("torch")


@pytest.mark.parametrize("selected", ["auto", "cpu"])
def test_windows_cpu_route_avoids_native_vllm(
    monkeypatch, tmp_path, legacy_windows_dependencies, selected
):
    from vllm_jev import windows_laya, windows_models

    calls = []
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "vllm-jev",
            "serve",
            "convaiinnovations/laya",
            "--backend",
            selected,
            "--workspace",
            str(tmp_path),
            "--cpu-threads",
            "1",
        ],
    )
    monkeypatch.setattr(
        backend, "server_module", lambda name: pytest.fail("native backend selected")
    )
    monkeypatch.setattr(windows_models, "prepare", lambda *args: tmp_path)
    monkeypatch.setattr(
        windows_laya, "serve", lambda *args, **kwargs: calls.append((args, kwargs))
    )
    cli.main()
    assert calls[0][1]["threads"] == 1
    assert calls[0][0][1] == "convaiinnovations/laya"


@pytest.mark.parametrize("selected", ["cuda", "ascend"])
def test_windows_explicit_accelerator_is_not_silently_ignored(monkeypatch, selected):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "vllm-jev",
            "serve",
            "convaiinnovations/laya",
            "--backend",
            selected,
        ],
    )
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
