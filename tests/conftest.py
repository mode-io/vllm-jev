import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "reference: compares with the published rsijev package"
    )


@pytest.fixture
def legacy_windows_dependencies(monkeypatch):
    """Supply pinned metadata only for mocked CLI-dispatch tests.

    These tests replace model preparation and serving. Real dependency mismatch
    rejection remains covered independently in test_windows_dependencies.py.
    """
    from vllm_jev import windows_dependencies

    original = windows_dependencies.require_windows_dependencies
    versions = dict(windows_dependencies.PROFILES["legacy"].requirements)

    def checked(model_id):
        return original(model_id, platform="win32", installed_versions=versions)

    monkeypatch.setattr(windows_dependencies, "require_windows_dependencies", checked)
