def pytest_configure(config):
    config.addinivalue_line(
        "markers", "reference: compares with the published rsijev package"
    )
