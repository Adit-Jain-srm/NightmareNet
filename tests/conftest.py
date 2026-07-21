"""Global test configuration for NightmareNet.

Disables API authentication during testing by unsetting NIGHTMARENET_API_KEY.
"""

from pathlib import Path

import pytest

try:
    import torchvision  # noqa: F401

    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False


def pytest_configure(config):
    """Skip vision distortion tests if torchvision is not available."""
    if not TORCHVISION_AVAILABLE:
        config.addinivalue_line(
            "markers", "skipif_no_torchvision: skip test if torchvision is not available"
        )
        # Add vision test file to collect_ignore
        config.option.ignore_paths = [str(Path("tests/test_distortion_vision.py").resolve())]


@pytest.fixture(autouse=True)
def _disable_api_auth(monkeypatch):
    """Remove API key from env so auth middleware is disabled during tests."""
    monkeypatch.delenv("NIGHTMARENET_API_KEY", raising=False)
