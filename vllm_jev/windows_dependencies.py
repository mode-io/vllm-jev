"""Version profiles for native Windows CPU inference, without backend imports.

Qwen3.5's BF16 fallback changed between Transformers 5.10.2 and 5.17.0.
These model-specific profiles preserve the existing environment while allowing
new author baselines in a separate environment. A matching dependency profile
is a prerequisite for inference, not a claim that a model passed validation.
"""

import importlib.metadata
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class WindowsDependencyProfile:
    name: str
    requirements: Mapping[str, str]

    @property
    def extra(self) -> str:
        return f"windows-{self.name}"


PROFILES = MappingProxyType(
    {
        "legacy": WindowsDependencyProfile(
            "legacy",
            MappingProxyType(
                {
                    "torch": "2.8.0",
                    "transformers": "5.10.2",
                    "peft": "0.19.1",
                    "accelerate": "1.13.0",
                    "numpy": "2.5.4",
                }
            ),
        ),
        "modern": WindowsDependencyProfile(
            "modern",
            MappingProxyType(
                {
                    "torch": "2.8.0",
                    "transformers": "5.17.0",
                    "peft": "0.21.0",
                    "accelerate": "1.15.0",
                    "numpy": "2.5.4",
                }
            ),
        ),
        "bruv": WindowsDependencyProfile(
            "bruv",
            MappingProxyType(
                {
                    "torch": "2.9.1",
                    "transformers": "5.17.0",
                    "peft": "0.21.0",
                    "accelerate": "1.15.0",
                    "numpy": "2.5.4",
                    "tokenizers": "0.23.3",
                    "jinja2": "3.1.6",
                }
            ),
        ),
        "rsi": WindowsDependencyProfile(
            "rsi",
            MappingProxyType(
                {
                    "torch": "2.13.0",
                    "transformers": "5.17.0",
                    "peft": "0.21.0",
                    "accelerate": "1.15.0",
                    "numpy": "2.5.4",
                    "tokenizers": "0.23.3",
                    "jinja2": "3.1.6",
                }
            ),
        ),
    }
)

# Model IDs, rather than prompt protocols, distinguish the required engine.
# A caller handling a local path must first identify its pinned source model.
MODEL_PROFILES = MappingProxyType(
    {
        "convaiinnovations/laya": "legacy",
        "convaiinnovations/laya-multilingual": "legacy",
        "convaiinnovations/laya-typed-decisions": "legacy",
        "lostargon/Tiny-Jev": "legacy",
        "lostargon/Tiny-Jev-1.7B": "legacy",
        "IamBusy/OpenJev-0.6B": "legacy",
        "ZefanCai/Open-Jev-2B": "legacy",
        "Mapika/decider-0.8b": "legacy",
        "Mapika/decider-2b": "legacy",
        "alibiserikbay/JevK5-2B": "modern",
        "alibiserikbay/JevK5": "modern",
        "alibiserikbay/JevK5-9B": "modern",
        "Mapika/decider-4b-GGUF": "modern",
        "evalengine/decision-0.8b-gguf": "modern",
        "evalengine/decision-4b-gguf": "modern",
        "bvolpato/bruv1-0.8b": "bruv",
        "shgao/rsi-jev-v1.0-qwen3.5-0.8b": "rsi",
        "alibiserikbay/JevK5-Lite": "modern",
        "jaredpalmer/kev-0.8b": "modern",
        "jaredpalmer/kev-0.6b": "modern",
        "flock-io/this-that-model-1.2": "modern",
        "flock-io/this-that-model-1.1": "modern",
        "flock-io/this-that-model-1.0": "modern",
        # Mica uses llama.cpp for inference and the pinned HF tokenizer only.
        "sky7350/Mica-v0.1-4B": "modern",
    }
)

# Mica's tokenizer-only process binds these libraries directly. Keep this
# additional guard model-specific so existing PyTorch profiles stay unchanged.
MICA_TOKENIZER_REQUIREMENTS = MappingProxyType(
    {"tokenizers": "0.23.3", "jinja2": "3.1.6"}
)


def profile_for(model_id: str) -> WindowsDependencyProfile:
    """Resolve a known model ID without importing an adapter or reading weights."""
    if not isinstance(model_id, str) or model_id not in MODEL_PROFILES:
        raise ValueError(
            f"No Windows dependency profile for {model_id!r}; "
            "identify a supported pinned model ID before checking dependencies."
        )
    return PROFILES[MODEL_PROFILES[model_id]]


def _matches(package: str, actual: str | None, required: str) -> bool:
    if actual == required:
        return True
    # CPU wheels report 2.8.0+cpu. PEP 440's torch==2.8.0 pin permits local
    # build suffixes; keep prereleases and other public versions excluded.
    return (
        package == "torch"
        and isinstance(actual, str)
        and actual.startswith(required + "+")
        and len(actual) > len(required) + 1
    )


def require_windows_dependencies(
    model_id: str,
    *,
    platform: str | None = None,
    installed_versions: Mapping[str, str | None] | None = None,
) -> WindowsDependencyProfile | None:
    """Fail before download/loading if Windows has the wrong model environment.

    The default reads distribution metadata only; it performs no imports of Torch,
    Transformers or adapters, and never changes the environment. Non-Windows calls
    are a no-op. ``installed_versions`` is an explicit complete snapshot for tests
    or callers that already read metadata; an omitted package counts as missing.
    """
    if (sys.platform if platform is None else platform) != "win32":
        return None
    profile = profile_for(model_id)
    requirements = dict(profile.requirements)
    if model_id == "sky7350/Mica-v0.1-4B":
        requirements.update(MICA_TOKENIZER_REQUIREMENTS)
    if model_id in (
        "Mapika/decider-4b-GGUF",
        "evalengine/decision-0.8b-gguf",
        "evalengine/decision-4b-gguf",
    ):
        requirements.update(MICA_TOKENIZER_REQUIREMENTS)
        requirements["llama-cpp-python"] = "0.3.35"
    if installed_versions is None:
        versions = {}
        for package in requirements:
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None
    else:
        versions = dict(installed_versions)
    mismatches = [
        f"{package}=={required} (installed: {versions.get(package) or 'missing'})"
        for package, required in requirements.items()
        if not _matches(package, versions.get(package), required)
    ]
    if mismatches:
        raise ValueError(
            f"{model_id} requires the {profile.extra} CPU environment: "
            + "; ".join(mismatches)
            + f". Use a separate virtual environment with CPU Torch {profile.requirements['torch']}, "
            f'then install this checkout with python -m pip install -e ".[{profile.extra}]". '
            "Keep the existing environment for models using the other profile."
        )
    return profile
