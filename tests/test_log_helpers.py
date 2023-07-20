# -*- coding: utf-8 -*-
"""
Tests for the `log_helpers` module in caf.toolkit
"""
# Built-Ins
import collections
import dataclasses
import getpass
import logging
import os
import pathlib
import platform
from typing import NamedTuple
import warnings
import psutil

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


@dataclasses.dataclass
class LogInitDetails:
    """Information for testing `LogHelper`."""

    details: ToolDetails
    details_message: str
    system: SystemInformation
    system_message: str
    init_message: list[str]


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


@pytest.fixture(name="cpu_count")
def fixture_monkeypatch_cpu_count(monkeypatch: pytest.MonkeyPatch) -> str:
    """Monkeypatch `os.cpu_count()` to return constant."""
    cpu_count = "10"
    monkeypatch.setattr(os, "cpu_count", lambda: cpu_count)
    return cpu_count


@pytest.fixture(name="total_ram")
def fixture_monkeypatch_total_ram(monkeypatch: pytest.MonkeyPatch) -> str:
    """Monkeypatch `psutil.virtual_memory()` to return constant."""
    ram = 30_000
    readable = "29.3K"

    memory = collections.namedtuple("memory", ["total"])
    monkeypatch.setattr(psutil, "virtual_memory", lambda: memory(ram))
    return readable


@pytest.fixture(name="log_init")
def fixture_log_init() -> LogInitDetails:
    """Initialise details for `LogHelper` tests."""
    name = "test tool"

    details = ToolDetails(name, "1.2.3")
    info = SystemInformation.load()

    msg = f"***  {name}  ***"
    init_message = ["", "*" * len(msg), msg, "*" * len(msg)]

    return LogInitDetails(details, f"\n{details}", info, f"\n{info}", init_message)


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

    def test_load(
        self,
        uname: UnameResult,
        python_version: str,
        username: str,
        cpu_count: str,
        total_ram: str,
    ) -> None:
        """Test loading system information."""
        os_label = f"{uname.system} {uname.release} ({uname.version})"

        info = SystemInformation.load()
        assert info.user == username, "incorrect username"
        assert info.pc_name == uname.node, "incorrect PC name"
        assert info.python_version == python_version, "incorrect Python version"
        assert info.operating_system == os_label, "incorrect OS"
        assert info.architecture == uname.machine, "incorrect architecture"
        assert info.cpu_count == cpu_count, "incorrect CPU count"
        assert info.total_ram == total_ram, "incorrect total RAM"

    def test_str(self) -> None:
        """Test string if formatted correctly."""
        user = "Test Name"
        pc_name = "Test PC"
        python_version = "3.0.0"
        operating_system = "Test 10 (10.0.1)"
        architecture = "AMD64"
        cpu_count = "16"
        total_ram = "30.3K"

        correct = (
            "System Information\n"
            "------------------\n"
            f"user             : {user}\n"
            f"pc_name          : {pc_name}\n"
            f"python_version   : {python_version}\n"
            f"operating_system : {operating_system}\n"
            f"architecture     : {architecture}\n"
            f"cpu_count        : {cpu_count}\n"
            f"total_ram        : {total_ram}"
        )

        info = SystemInformation(
            user, pc_name, python_version, operating_system, architecture, cpu_count, total_ram
        )

        assert str(info) == correct, "incorrect string format"


class TestLogHelper:
    """Test `LogHelper` class."""

    def _load_log(self, log_file: pathlib.Path) -> str:
        assert log_file.is_file(), "log file not created"
        with open(log_file, "rt", encoding="utf-8") as file:
            text = file.read()
        return text

    def test_initialising(
        self, caplog: pytest.LogCaptureFixture, log_init: LogInitDetails
    ) -> None:
        """Test initialising the logger without a file."""
        LogHelper("test", log_init.details, warning_capture=False)

        messages = [i.message for i in caplog.get_records("call")]

        assert messages[:4] == log_init.init_message, "initialisation messages"

        assert messages[4] == log_init.details_message, "incorrect tool details"
        assert messages[5] == log_init.system_message, "incorrect system info"

    def test_file_initialisation(
        self, tmp_path: pathlib.Path, log_init: LogInitDetails
    ) -> None:
        """Test initialising the logger with a file."""
        log_file = tmp_path / "test.log"
        assert not log_file.is_file(), "log file already exists"

        LogHelper(
            "test", log_init.details, console=False, log_file=log_file, warning_capture=False
        )

        text = self._load_log(log_file)

        for i, line in enumerate(log_init.init_message):
            assert line in text, f"line {i} init message"

        assert log_init.details_message in text, "incorrect tool details"
        assert log_init.system_message in text, "incorrect system information"

    def test_basic_file(self, tmp_path: pathlib.Path, log_init: LogInitDetails) -> None:
        """Test logging to file within `with` statement.

        Tests all log calls within `with` statement are logged to file
        and any outside are ignored.
        """
        root = "test"
        log = logging.getLogger(f"{root}.test_basic_file")
        log_file = tmp_path / "test.log"

        levels = list(range(10, 60, 10))
        messages = [f"testing level {i} - test basic file" for i in levels]

        with LogHelper(
            root, log_init.details, console=False, log_file=log_file, warning_capture=False
        ):
            for i, msg in zip(levels, messages):
                log.log(i, msg)

        # Messages logged after the log helper class has cleaned up so shouldn't be saved to file
        unlogged_messages = [f"not logging this message for level {i}" for i in levels]
        for i, msg in zip(levels, unlogged_messages):
            log.log(i, msg)

        text = self._load_log(log_file)

        for i, msg in zip(levels, messages):
            assert msg in text, f"missed logging level {i}"

        for i, msg in zip(levels, unlogged_messages):
            assert msg not in text, f"logged after closing class level {i}"

    def test_capture_warnings(self, tmp_path: pathlib.Path, log_init: LogInitDetails) -> None:
        """Test Python warnings are captured and saved to log file."""
        root = "test"
        log_file = tmp_path / "test.log"

        log_warnings = [
            ("testing runtime warning", RuntimeWarning),
            ("testing user warning", UserWarning),
        ]
        # Note: ImportWarnings aren't logged by default

        with LogHelper(root, log_init.details, console=False, log_file=log_file):
            for msg, warn in log_warnings:
                warnings.warn(msg, warn)

        text = self._load_log(log_file)

        for msg, warn in log_warnings:
            assert f"{warn.__name__}: {msg}" in text, f"missing {warn.__name__} warning"
