"""
pytest configuration for MedAide+ test suite.
Configures asyncio mode for async tests.
"""
import pytest


def pytest_configure(config):
    """Configure pytest-asyncio mode."""
    config.addinivalue_line(
        "markers", "asyncio: mark test as async"
    )
