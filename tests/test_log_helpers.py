# -*- coding: utf-8 -*-
"""
Tests for the `log_helpers` module in caf.toolkit
"""
# Built-Ins
import getpass
import platform
from typing import NamedTuple

# Third Party
import pydantic
import pytest

# Local Imports
from caf.toolkit import LogHelper, SystemInformation, TemporaryLogFile, ToolDetails


# # # Fixture # # #
class UnameResult(NamedTuple):
    """Result from `platform.uname()` for testing."""

    system: str
    node: str
    release: str
    version: str
    machine: str


@pytest.fixture(name="uname")
def fixture_monkeypatch_uname(monkeypatch: pytest.MonkeyPatch) -> UnameResult:
    """Monkeypatch `platform.uname()` to return constant."""
    result = UnameResult("Test System", "Test PC", "10", "10.0.1", "AMD64")
    monkeypatch.setattr(platform, "uname", lambda: result)
    return result


@pytest.fixture(name="python_version")
def fixture_monkeypatch_version(monkeypatch: pytest.MonkeyPatch) -> str:
    """Monkeypatch `platform.python_version()` to return constant."""
    version = "3.0.0"
    monkeypatch.setattr(platform, "python_version", lambda: version)
    return version


@pytest.fixture(name="username")
def fixture_monkeypatch_username(monkeypatch: pytest.MonkeyPatch) -> str:
    """Monkeypatch `getpass.getuser()` to return constant."""
    user = "Test User"
    monkeypatch.setattr(getpass, "getuser", lambda: user)
    return user


# # # Tests # # #
class TestToolDetails:
    """Test ToolDetails class validation."""

    @pytest.mark.parametrize(
        "version",
        ["0.0.4", "1.2.3", "10.20.30", "1.1.2-prerelease+meta", "1.1.2+meta", "1.0.0-alpha"],
    )
    @pytest.mark.parametrize(
        "url",
        ["https://www.github.com/", "http://www.github.com/", "http://github.com/", None],
    )
    def test_valid(self, version: str, url: str | None) -> None:
        """Test valid values of version and homepage / source URLs."""
        # Not testing different values for name because there's no validation on it
        # Ignoring mypy type stating str is incorrect type
        ToolDetails("test_name", version, url, url)  # type: ignore

    @pytest.mark.parametrize("url", [None, "http://github.com"])
    def test_str(self, url: str | None) -> None:
        """Test converting to formatted string with / without optional values."""
        name, version = "test1", "1.2.3"
        if url is None:
            # fmt: off
            correct = (
                "Tool Information\n"
                "----------------\n"
                "name    : test1\n"
                "version : 1.2.3"
            )

        else:
            # fmt: off
            correct = (
                "Tool Information\n"
                "----------------\n"
                "name       : test1\n"
                "version    : 1.2.3\n"
                "homepage   : http://github.com\n"
                "source_url : http://github.com"
            )

        assert str(ToolDetails(name, version, url, url)) == correct  # type: ignore

    @pytest.mark.parametrize("version", ["1", "1.2", "1.1.2+.123", "alpha"])
    def test_invalid_versions(self, version: str) -> None:
        """Test correctly raise ValidationError for invalid versions."""
        url = "https://github.com"
        with pytest.raises(pydantic.ValidationError):
            ToolDetails("test_name", version, url, url)  # type: ignore

    @pytest.mark.parametrize("url", ["github.com", "github", "www.github.com"])
    def test_invalid_urls(self, url: str) -> None:
        """Test correctly raise ValidationError for invalid homepage / source URLs."""
        with pytest.raises(pydantic.ValidationError):
            ToolDetails("test_name", "1.2.3", url, url)  # type: ignore


class TestSystemInformation:
    """Test SystemInformation class."""

    def test_load(self, uname: UnameResult, python_version: str, username: str) -> None:
        """Test loading system information."""
        os_label = f"{uname.system} {uname.release} ({uname.version})"

        info = SystemInformation.load()
        assert info.user == username, "incorrect username"
        assert info.pc_name == uname.node, "incorrect PC name"
        assert info.python_version == python_version, "incorrect Python version"
        assert info.operating_system == os_label, "incorrect OS"
        assert info.architecture == uname.machine, "incorrect architecture"

    def test_str(self) -> None:
        """Test string if formatted correctly."""
        user = "Test Name"
        pc_name = "Test PC"
        python_version = "3.0.0"
        operating_system = "Test 10 (10.0.1)"
        architecture = "AMD64"

        correct = (
            "System Information\n"
            "------------------\n"
            f"user             : {user}\n"
            f"pc_name          : {pc_name}\n"
            f"python_version   : {python_version}\n"
            f"operating_system : {operating_system}\n"
            f"architecture     : {architecture}"
        )

        info = SystemInformation(user, pc_name, python_version, operating_system, architecture)

        assert str(info) == correct, "incorrect string format"
