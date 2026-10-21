"""Dependency profiles must agree with packaging without importing inference."""

import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
from packaging.markers import default_environment
from packaging.requirements import Requirement

from vllm_jev import windows_dependencies as dependencies

ROOT = Path(__file__).resolve().parents[1]
LEGACY_MODELS = (
    "convaiinnovations/laya",
    "convaiinnovations/laya-multilingual",
    "convaiinnovations/laya-typed-decisions",
    "lostargon/Tiny-Jev",
    "IamBusy/OpenJev-0.6B",
    "ZefanCai/Open-Jev-2B",
    "Mapika/decider-0.8b",
    "Mapika/decider-2b",
)
MODERN_MODELS = (
    "Mapika/decider-4b-GGUF",
    "alibiserikbay/JevK5",
    "alibiserikbay/JevK5-2B",
    "alibiserikbay/JevK5-Lite",
    "jaredpalmer/kev-0.8b",
    "jaredpalmer/kev-0.6b",
    "flock-io/this-that-model-1.2",
    "flock-io/this-that-model-1.1",
    "flock-io/this-that-model-1.0",
    "sky7350/Mica-v0.1-4B",
)
NEW_MODELS = {
    "evalengine/decision-0.8b-gguf": "modern",
    "evalengine/decision-4b-gguf": "modern",
    "bvolpato/bruv1-0.8b": "bruv",
    "shgao/rsi-jev-v1.0-qwen3.5-0.8b": "rsi",
}


@pytest.mark.parametrize("model", LEGACY_MODELS)
def test_existing_eight_keep_legacy_profile(model):
    profile = dependencies.profile_for(model)
    assert profile.name == "legacy"
    assert profile.requirements["transformers"] == "5.10.2"
    assert (
        dependencies.require_windows_dependencies(
            model, platform="win32", installed_versions=profile.requirements
        )
        is profile
    )


@pytest.mark.parametrize("model", MODERN_MODELS)
def test_new_candidates_use_modern_profile(model):
    profile = dependencies.profile_for(model)
    assert profile.name == "modern"
    assert profile.requirements["transformers"] == "5.17.0"
    versions = dict(profile.requirements)
    if model in ("sky7350/Mica-v0.1-4B", "Mapika/decider-4b-GGUF"):
        versions.update(dependencies.MICA_TOKENIZER_REQUIREMENTS)
    if model == "Mapika/decider-4b-GGUF":
        versions["llama-cpp-python"] = "0.3.35"
    assert (
        dependencies.require_windows_dependencies(
            model, platform="win32", installed_versions=versions
        )
        is profile
    )


@pytest.mark.parametrize("model", LEGACY_MODELS + MODERN_MODELS)
def test_cross_profile_use_fails_with_specific_environment_instruction(model):
    profile = dependencies.profile_for(model)
    wrong = dependencies.PROFILES["modern" if profile.name == "legacy" else "legacy"]
    with pytest.raises(ValueError) as error:
        dependencies.require_windows_dependencies(
            model, platform="win32", installed_versions=wrong.requirements
        )
    assert model in str(error.value)
    assert profile.extra in str(error.value)
    assert f'python -m pip install -e ".[{profile.extra}]"' in str(error.value)
    assert "separate virtual environment" in str(error.value)


@pytest.mark.parametrize("profile_name", ["legacy", "modern"])
@pytest.mark.parametrize(
    "package", ["torch", "transformers", "peft", "accelerate", "numpy"]
)
def test_every_key_dependency_is_required_and_checked(profile_name, package):
    profile = dependencies.PROFILES[profile_name]
    model = LEGACY_MODELS[0] if profile_name == "legacy" else MODERN_MODELS[0]
    versions = dict(profile.requirements)
    del versions[package]
    with pytest.raises(ValueError, match=package + ".*missing"):
        dependencies.require_windows_dependencies(
            model, platform="win32", installed_versions=versions
        )
    versions[package] = "999.0.0"
    with pytest.raises(ValueError, match=package + ".*999.0.0"):
        dependencies.require_windows_dependencies(
            model, platform="win32", installed_versions=versions
        )


def test_cpu_local_version_suffix_matches_packaging():
    versions = dict(dependencies.PROFILES["legacy"].requirements)
    versions["torch"] = "2.8.0+cpu"
    assert "2.8.0+cpu" in Requirement("torch==2.8.0").specifier
    assert (
        dependencies.require_windows_dependencies(
            LEGACY_MODELS[0], platform="win32", installed_versions=versions
        ).name
        == "legacy"
    )


@pytest.mark.parametrize("version", ["2.8.0rc1", "2.8.0.dev0", "2.8.1+cpu", "2.8.0+"])
def test_other_torch_releases_and_prereleases_are_not_silently_accepted(version):
    versions = dict(dependencies.PROFILES["legacy"].requirements, torch=version)
    with pytest.raises(ValueError, match="torch==2.8.0"):
        dependencies.require_windows_dependencies(
            LEGACY_MODELS[0], platform="win32", installed_versions=versions
        )


@pytest.mark.parametrize(
    "model", ["unknown/model", "vllm-jev", Path("local-checkpoint"), None]
)
def test_guard_requires_identified_model_id(model):
    with pytest.raises(ValueError, match="identify"):
        dependencies.require_windows_dependencies(
            model, platform="win32", installed_versions={}
        )


def test_real_metadata_reader_handles_missing_package_without_backend_import(
    monkeypatch,
):
    seen = []
    expected = dependencies.PROFILES["legacy"].requirements

    def version(package):
        seen.append(package)
        if package == "transformers":
            raise dependencies.importlib.metadata.PackageNotFoundError(package)
        return expected[package]

    monkeypatch.setattr(dependencies.importlib.metadata, "version", version)
    with pytest.raises(ValueError, match="transformers==5.10.2.*missing"):
        dependencies.require_windows_dependencies(LEGACY_MODELS[0], platform="win32")
    assert seen == list(expected)


@pytest.mark.parametrize("platform", ["linux", "darwin"])
def test_non_windows_is_noop_before_model_or_package_resolution(monkeypatch, platform):
    monkeypatch.setattr(
        dependencies.importlib.metadata,
        "version",
        lambda name: pytest.fail("metadata access on non-Windows"),
    )
    assert (
        dependencies.require_windows_dependencies(
            "not-a-Windows-model", platform=platform
        )
        is None
    )


def test_profiles_cannot_be_accidentally_mutated():
    with pytest.raises(TypeError):
        dependencies.PROFILES["legacy"].requirements["transformers"] = "5.17.0"
    with pytest.raises(TypeError):
        dependencies.MODEL_PROFILES[LEGACY_MODELS[0]] = "modern"


def test_import_and_guard_do_not_import_backends():
    code = """
import sys
from vllm_jev.windows_dependencies import PROFILES, require_windows_dependencies
p = require_windows_dependencies('alibiserikbay/JevK5-2B', platform='win32', installed_versions=PROFILES['modern'].requirements)
assert p.extra == 'windows-modern'
assert not any(name in sys.modules for name in ('torch', 'transformers', 'peft', 'accelerate', 'numpy', 'vllm', 'mlx'))
"""
    subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT, check=True)


def requirements_for(platform, extra=None):
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]
    environment = default_environment()
    environment.update(sys_platform=platform, extra=extra or "")
    values = (
        project["dependencies"]
        if extra is None
        else project["optional-dependencies"][extra]
    )
    return [
        requirement
        for value in values
        if (requirement := Requirement(value)).marker is None
        or requirement.marker.evaluate(environment)
    ]


@pytest.mark.parametrize("name", ["legacy", "modern", "bruv", "rsi"])
def test_extra_pins_match_guard_and_fit_base_constraints(name):
    profile = dependencies.PROFILES[name]
    extra = {
        requirement.name: requirement
        for requirement in requirements_for("win32", profile.extra)
    }
    base = {requirement.name: requirement for requirement in requirements_for("win32")}
    expected = dict(profile.requirements)
    if name == "modern":
        expected.update(dependencies.MICA_TOKENIZER_REQUIREMENTS)
    assert set(extra) == set(expected)
    for package, version in expected.items():
        assert str(extra[package].specifier) == "==" + version
    for package, version in profile.requirements.items():
        if package in base:
            assert version in base[package].specifier
        else:
            assert package in {"tokenizers", "jinja2"}
    assert not {"vllm", "mlx-vlm"} & base.keys()


@pytest.mark.parametrize("package", ["tokenizers", "jinja2"])
@pytest.mark.parametrize("value", [None, "999.0.0"])
def test_mica_requires_exact_tokenizer_versions(package, value):
    versions = {
        **dependencies.PROFILES["modern"].requirements,
        **dependencies.MICA_TOKENIZER_REQUIREMENTS,
        package: value,
    }
    with pytest.raises(ValueError, match=package) as error:
        dependencies.require_windows_dependencies(
            "sky7350/Mica-v0.1-4B", platform="win32", installed_versions=versions
        )
    assert "windows-modern" in str(error.value)
    assert "missing" in str(error.value) if value is None else value in str(error.value)


@pytest.mark.parametrize(
    "model",
    [
        m
        for m in LEGACY_MODELS + MODERN_MODELS
        if m not in ("sky7350/Mica-v0.1-4B", "Mapika/decider-4b-GGUF")
    ],
)
def test_existing_models_do_not_gain_a_tokenizer_dependency_guard(model, monkeypatch):
    required = dependencies.profile_for(model).requirements
    seen = []

    def version(package):
        assert package not in ("tokenizers", "jinja2")
        seen.append(package)
        return required[package]

    monkeypatch.setattr(dependencies.importlib.metadata, "version", version)
    dependencies.require_windows_dependencies(model, platform="win32")
    assert seen == list(required)


def test_mutually_exclusive_extras_do_not_claim_one_environment_can_use_both():
    legacy = {r.name: r for r in requirements_for("win32", "windows-legacy")}
    modern = {r.name: r for r in requirements_for("win32", "windows-modern")}
    assert "5.17.0" not in legacy["transformers"].specifier
    assert "5.10.2" not in modern["transformers"].specifier


@pytest.mark.parametrize("platform", ["linux", "darwin"])
def test_new_extras_do_not_change_non_windows_dependencies(platform):
    assert requirements_for(platform, "windows-legacy") == []
    assert requirements_for(platform, "windows-modern") == []
    assert requirements_for(platform, "windows-bruv") == []
    assert requirements_for(platform, "windows-rsi") == []
    base = {requirement.name: requirement for requirement in requirements_for(platform)}
    assert str(base["peft"].specifier) == "<0.21,>=0.20"
    assert "numpy" not in base and "accelerate" not in base
    assert (
        "5.10.2" in base["transformers"].specifier
        if platform == "linux"
        else "5.14.0" in base["transformers"].specifier
    )


@pytest.mark.parametrize("model,profile_name", NEW_MODELS.items())
def test_four_new_models_require_their_isolated_profile(model, profile_name):
    profile = dependencies.profile_for(model)
    assert profile.name == profile_name
    versions = dict(profile.requirements)
    if profile_name == "modern":
        versions.update(dependencies.MICA_TOKENIZER_REQUIREMENTS)
        versions["llama-cpp-python"] = "0.3.35"
    assert (
        dependencies.require_windows_dependencies(
            model, platform="win32", installed_versions=versions
        )
        is profile
    )
    for package in versions:
        wrong = dict(versions)
        wrong[package] = "999.0.0"
        with pytest.raises(ValueError, match=package):
            dependencies.require_windows_dependencies(
                model, platform="win32", installed_versions=wrong
            )


@pytest.mark.parametrize(
    "model", ["bvolpato/bruv1-0.8b", "shgao/rsi-jev-v1.0-qwen3.5-0.8b"]
)
def test_new_torch_profiles_do_not_accept_retained_modern_environment(model):
    old = dict(dependencies.PROFILES["modern"].requirements)
    old.update(dependencies.MICA_TOKENIZER_REQUIREMENTS)
    with pytest.raises(ValueError) as error:
        dependencies.require_windows_dependencies(
            model, platform="win32", installed_versions=old
        )
    profile = dependencies.profile_for(model)
    assert f"CPU Torch {profile.requirements['torch']}" in str(error.value)
    assert profile.extra in str(error.value)
