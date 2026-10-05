"""Configuration and plugin tests; no weights or CUDA device required."""

import sys
from types import SimpleNamespace

import pytest

from vllm_jev import precision


@pytest.fixture
def backend(monkeypatch):
    class Matmul:
        def __init__(self):
            self.reduced, self.splitk = True, True

        @property
        def allow_bf16_reduced_precision_reduction(self):
            return self.reduced

        @allow_bf16_reduced_precision_reduction.setter
        def allow_bf16_reduced_precision_reduction(self, value):
            self.reduced, self.splitk = (
                value if isinstance(value, tuple) else (value, True)
            )

        @property
        def allow_bf16_reduced_precision_reduction_split_k(self):
            return self.splitk

    state = SimpleNamespace(matmul=Matmul(), library="default", calls=[])

    def preferred_blas_library(value=None):
        if value is not None:
            state.calls.append(value)
            state.library = value
        return state.library

    state.preferred_blas_library = preferred_blas_library
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            backends=SimpleNamespace(cuda=state),
            version=SimpleNamespace(cuda="13.0"),
        ),
    )
    envs = SimpleNamespace(environment_variables={})
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(envs=envs))
    state.envs = envs
    monkeypatch.delenv(precision.MATMUL_ENV, raising=False)
    monkeypatch.setattr(precision, "_configured_mode", None)
    return state


def test_default_preserves_process_settings_and_registers_cache_factor(backend):
    backend.matmul.reduced = False
    precision.configure_matmul()
    assert backend.calls == []
    assert (backend.matmul.reduced, backend.matmul.splitk) == (False, True)
    assert backend.envs.environment_variables[precision.MATMUL_ENV]() == "default"


def test_no_splitk_sets_controls_and_distinct_cache_factor(backend, monkeypatch):
    default = precision.matmul_mode()
    monkeypatch.setenv(precision.MATMUL_ENV, "no_splitk")
    precision.configure_matmul()
    getter = backend.envs.environment_variables[precision.MATMUL_ENV]
    assert getter() != default
    assert backend.library == "cublaslt"
    assert (backend.matmul.reduced, backend.matmul.splitk) == (False, False)
    precision.configure_matmul()  # General plugin loading can be repeated.
    assert (backend.matmul.reduced, backend.matmul.splitk) == (False, False)


@pytest.mark.parametrize("initial", ["default", "no_splitk"])
def test_mode_change_requires_restart_and_preserves_applied_cache_factor(
    backend, monkeypatch, initial
):
    monkeypatch.setenv(precision.MATMUL_ENV, initial)
    precision.configure_matmul()
    getter = backend.envs.environment_variables[precision.MATMUL_ENV]
    previous = (backend.library, backend.matmul.reduced, backend.matmul.splitk)
    calls = list(backend.calls)

    other = "no_splitk" if initial == "default" else "default"
    monkeypatch.setenv(precision.MATMUL_ENV, other)
    assert getter() == initial
    with pytest.raises(RuntimeError, match="Restart the server"):
        precision.configure_matmul()
    assert backend.envs.environment_variables[precision.MATMUL_ENV]() == initial
    assert (backend.library, backend.matmul.reduced, backend.matmul.splitk) == previous
    assert backend.calls == calls


def test_invalid_mode_fails_before_backend_mutation(backend, monkeypatch):
    monkeypatch.setenv(precision.MATMUL_ENV, "no-splitk")
    with pytest.raises(ValueError, match="must be"):
        precision.configure_matmul()
    assert backend.calls == []


def test_unsupported_torch_fails_before_backend_mutation(backend, monkeypatch):
    monkeypatch.setenv(precision.MATMUL_ENV, "no_splitk")
    monkeypatch.setattr(sys.modules["torch"].version, "cuda", None)
    with pytest.raises(RuntimeError, match="requires a CUDA PyTorch"):
        precision.configure_matmul()
    assert backend.calls == []


def test_failed_backend_selection_restores_previous_settings(backend, monkeypatch):
    monkeypatch.setenv(precision.MATMUL_ENV, "no_splitk")
    previous = backend.preferred_blas_library

    def unsupported(value=None):
        if value == "cublaslt":
            backend.library = value
            raise RuntimeError("cuBLASLt unavailable")
        return previous(value)

    monkeypatch.setattr(backend, "preferred_blas_library", unsupported)
    with pytest.raises(RuntimeError, match="Cannot enable"):
        precision.configure_matmul()
    assert backend.library == "default"
    assert (backend.matmul.reduced, backend.matmul.splitk) == (True, True)
    assert precision._configured_mode is None
    assert precision.MATMUL_ENV not in backend.envs.environment_variables

    monkeypatch.setattr(backend, "preferred_blas_library", previous)
    precision.configure_matmul()
    assert backend.envs.environment_variables[precision.MATMUL_ENV]() == "no_splitk"
    assert (backend.matmul.reduced, backend.matmul.splitk) == (False, False)


def test_general_plugin_configures_before_model_registration(monkeypatch):
    import vllm_jev

    calls = []
    monkeypatch.setattr(
        precision, "configure_matmul", lambda: calls.append("precision")
    )
    registry = SimpleNamespace(register_model=lambda *args: calls.append("model"))
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.models",
        SimpleNamespace(ModelRegistry=registry),
    )
    vllm_jev.register()
    assert calls[0] == "precision"
    assert calls[1:] and all(call == "model" for call in calls[1:])
