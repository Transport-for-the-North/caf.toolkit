"""Configuration file for pytest."""
# Built-Ins
import os

# Third Party
import pytest


def pytest_configure() -> None:
    pytest.IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"
