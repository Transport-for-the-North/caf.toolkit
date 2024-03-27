# -*- coding: utf-8 -*-
"""
Tests for the `log_helpers` module in caf.toolkit
"""
from __future__ import annotations

# Built-Ins
import collections
import dataclasses
import getpass
import logging
import os
import pathlib
import platform
import warnings
from typing import NamedTuple

# Third Party
import psutil
import pydantic
import pytest

# Local Imports
from caf.toolkit import LogHelper, SystemInformation, TemporaryLogFile, ToolDetails
from caf.toolkit.log_helpers import LoggingWarning, capture_warnings, get_logger

# # # Constants # # #
_LOG_WARNINGS = [
    ("testing warning: runtime warning", RuntimeWarning),
    ("testing warning: user warning", UserWarning),
]
# Note: ImportWarnings aren't logged by default


# # # Fixture # # #
class UnameResult(NamedTuple):
    """Result from `platform.uname()` for testing."""

    system: str
    node: str
    release: str
    version: str
    machine: str
    processor: str


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
    result = UnameResult(
        "Test System",
        "Test PC",
        "10",
        "10.0.1",
        "AMD64",
        "Intel64 Family 6 Model 85 Stepping 7, GenuineIntel",
    )
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
def fixture_monkeypatch_cpu_count(monkeypatch: pytest.MonkeyPatch) -> int:
    """Monkeypatch `os.cpu_count()` to return constant."""
    cpu_count = 10
    monkeypatch.setattr(os, "cpu_count", lambda: cpu_count)
    return cpu_count


@pytest.fixture(name="total_ram", params=[True, False])
def fixture_monkeypatch_total_ram(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> tuple[int | None, str]:
    """Monkeypatch `psutil.virtual_memory()` to return constant.

    If param is True set total ram to constant int, otherwise
    total ram is None.
    """
    memory = collections.namedtuple("memory", ["total"])

    if request.param:
        ram = 30_000
        value = memory(ram)
        readable = "29.3K"

    else:
        ram = None
        value = None
        readable = "unknown"

    monkeypatch.setattr(psutil, "virtual_memory", lambda: value)
    return ram, readable


@pytest.fixture(name="log_init")
def fixture_log_init() -> LogInitDetails:
    """Initialise details for `LogHelper` tests."""
    name = "test tool"

    details = ToolDetails(name, "1.2.3")
    info = SystemInformation.load()

    msg = f"***  {name}  ***"
    init_message = ["", "*" * len(msg), msg, "*" * len(msg)]

    return LogInitDetails(details, f"\n{details}", info, f"\n{info}", init_message)


@pytest.fixture(name="warnings_logger")
def fixture_warnings_logger() -> logging.Logger:
    """Get 'py.warnings' logger and clear handlers."""
    logger = logging.getLogger("py.warnings")

    for handler in logger.handlers:
        handler.close()

    logger.handlers.clear()
    return logger


def _log_messages(
    logger: logging.Logger, message: str = "testing logging level {level}"
) -> list[tuple[int, str]]:
    """Log messages for each logging level."""
    levels = list(range(10, 60, 10))
    messages = []
    for i in levels:
        msg = message.format(level=i)
        logger.log(i, msg)
        messages.append(msg)

    return list(zip(levels, messages))


def _load_log(log_file: pathlib.Path) -> str:
    """Assert log file exists and load the text."""
    assert log_file.is_file(), "log file not created"
    with open(log_file, "rt", encoding="utf-8") as file:
        text = file.read()

    return text


def _run_warnings() -> None:
    """Run all warnings."""
    for msg, warn in _LOG_WARNINGS:
        warnings.warn(msg, warn)


def _check_warnings(text: str) -> None:
    """Check warnings are in `text`."""
    for msg, warn in _LOG_WARNINGS:
        assert f"{warn.__name__}: {msg}" in text, f"missing {warn.__name__} warning"


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
                "homepage   : http://github.com/\n"
                "source_url : http://github.com/"
                # When validating URLs the ending '/' is added
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
        cpu_count: int,
        total_ram: tuple[int | None, str],
    ) -> None:
        """Test loading system information."""
        os_label = f"{uname.system} {uname.release} ({uname.version})"

        info = SystemInformation.load()
        assert info.user == username, "incorrect username"
        assert info.pc_name == uname.node, "incorrect PC name"
        assert info.python_version == python_version, "incorrect Python version"
        assert info.operating_system == os_label, "incorrect OS"
        assert info.architecture == uname.machine, "incorrect architecture"
        assert info.processor == uname.processor, "incorrect processor name"
        assert info.cpu_count == cpu_count, "incorrect CPU count"
        assert info.total_ram == total_ram[0], "incorrect total RAM"

    def test_str(self, total_ram: tuple[int | None, str]) -> None:
        """Test string if formatted correctly."""
        user = "Test Name"
        pc_name = "Test PC"
        python_version = "3.0.0"
        operating_system = "Test 10 (10.0.1)"
        architecture = "AMD64"
        processor = "Intel64 Family 6 Model 85 Stepping 7, GenuineIntel"
        cpu_count = 16
        ram, ram_readable = total_ram

        correct = (
            "System Information\n"
            "------------------\n"
            f"user             : {user}\n"
            f"pc_name          : {pc_name}\n"
            f"python_version   : {python_version}\n"
            f"operating_system : {operating_system}\n"
            f"architecture     : {architecture}\n"
            f"processor        : {processor}\n"
            f"cpu_count        : {cpu_count}\n"
            f"total_ram        : {ram_readable}"
        )

        info = SystemInformation(
            user,
            pc_name,
            python_version,
            operating_system,
            architecture,
            processor,
            cpu_count,
            ram,
        )

        assert str(info) == correct, "incorrect string format"

    def test_getpass_module_not_found(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test user is unknown if `getuser` raises ModuleNotFoundError."""

        def getuser() -> None:
            raise ModuleNotFoundError()

        monkeypatch.setattr(getpass, "getuser", getuser)
        info = SystemInformation.load()

        assert info.user == "unknown", "incorrect username"


class TestLogHelper:
    """Test `LogHelper` class."""

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

        text = _load_log(log_file)

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

        with LogHelper(
            root, log_init.details, console=False, log_file=log_file, warning_capture=False
        ):
            messages = _log_messages(log, "testing level {level} - test basic file")

        # Messages logged after the log helper class has
        # cleaned up so shouldn't be saved to file
        unlogged_messages = _log_messages(log, "not logging this message for level {level}")

        text = _load_log(log_file)

        for i, msg in messages:
            assert msg in text, f"missed logging level {i}"

        for i, msg in unlogged_messages:
            assert msg not in text, f"logged after closing class level {i}"

    def test_handler_cleanup(self, log_init: LogInitDetails) -> None:
        """Test handlers are added to logger and removed upon exiting with statement."""
        root = "test"
        logger = logging.getLogger(root)
        assert logger.handlers == [], "logger already has handlers"

        with LogHelper(root, log_init.details):
            assert len(logger.handlers) == 1, "incorrect number of handlers"
            assert isinstance(
                logger.handlers[0], logging.StreamHandler
            ), "incorrect stream handler"

        assert len(logger.handlers) == 0, "handlers not cleaned up"

    @pytest.mark.filterwarnings("ignore:testing warning")
    def test_setup_warnings_stream_handler(
        self, log_init: LogInitDetails, warnings_logger: logging.Logger
    ) -> None:
        """Test file handler is added to warnings logger."""
        with LogHelper("test", log_init.details):
            _run_warnings()

            assert len(warnings_logger.handlers) == 1, "incorrect number of handlers"
            assert isinstance(
                warnings_logger.handlers[0], logging.StreamHandler
            ), "error with stream handler"

        assert len(warnings_logger.handlers) == 0, "incorrect handler cleanup"

    @pytest.mark.filterwarnings("ignore:testing warning")
    def test_setup_warnings_file_handler(
        self, tmp_path: pathlib.Path, log_init: LogInitDetails, warnings_logger: logging.Logger
    ) -> None:
        """Test file handler is added to warnings logger and log file is created."""
        log_file = tmp_path / "test.log"

        assert not log_file.is_file(), "log file already exists"

        with LogHelper("test", log_init.details, console=False, log_file=log_file):
            _run_warnings()

            assert len(warnings_logger.handlers) == 1, "incorrect number of handlers"
            assert isinstance(
                warnings_logger.handlers[0], logging.FileHandler
            ), "error with file handler"

        assert log_file.is_file(), "log file not created"

    @pytest.mark.skip(
        reason="fails when running all tests, I think due to warnings be captured by pytest"
    )
    def test_capture_warnings(
        self,
        caplog: pytest.LogCaptureFixture,
        log_init: LogInitDetails,
        warnings_logger: logging.Logger,
    ) -> None:
        """Test Python warnings are captured."""
        del warnings_logger  # Fixture clears handlers from logger

        with LogHelper("test", log_init.details):
            _run_warnings()

        _check_warnings(caplog.text)

    def test_no_handlers_warning(self, log_init: LogInitDetails) -> None:
        """Test LogHelper warns when no handlers are defined."""
        with pytest.warns(
            LoggingWarning, match="LogHelper initialised without any logging handlers"
        ):
            with LogHelper(
                "test", log_init.details, console=False, warning_capture=False
            ) as helper:
                assert helper.logger.handlers == [], "incorrect handlers"

    @pytest.mark.parametrize("warning_capture", [False, True])
    def test_add_handler(
        self, log_init: LogInitDetails, warning_capture: bool, warnings_logger: logging.Logger
    ) -> None:
        """Test creating LogHelper without loggers and adding StreamHandler after."""
        # pylint: disable=protected-access
        stream = logging.StreamHandler()

        with LogHelper(
            "test", log_init.details, console=False, warning_capture=warning_capture
        ) as helper:
            assert helper.logger.handlers == [], "logger already has handlers"
            assert warnings_logger.handlers == [], "warnings logger already has handlers"

            helper.add_handler(stream)
            assert helper.logger.handlers == [stream], "list of handlers is incorrect"

            if warning_capture:
                assert warnings_logger.handlers == [
                    stream
                ], "handler not added to warnings logger"
            else:
                assert warnings_logger.handlers == [], "handlers added to warnings logger"


class TestTemporaryLogFile:
    # pylint: disable=too-few-public-methods
    """Test `TemporaryLogFile` class."""

    def test_log_file(self, tmp_path: pathlib.Path) -> None:
        """Test log file handler is added and removed correctly."""
        base_log_file = tmp_path / "base_log_file.log"
        log_file = tmp_path / "test.log"
        logger = logging.getLogger("test")
        logger.setLevel(logging.DEBUG)

        with TemporaryLogFile(logger, log_file, base_log_file=base_log_file):
            messages = _log_messages(logger)

        # Messages logged after the temporary log file class has
        # cleaned up so shouldn't be saved to file
        unlogged_messages = _log_messages(logger, "not logging this message for level {level}")

        text = _load_log(log_file)

        assert str(base_log_file) in text, "base log file missing"

        for i, msg in messages:
            assert msg in text, f"missed logging level {i}"

        for i, msg in unlogged_messages:
            assert msg not in text, f"logged after closing class level {i}"


class TestGetLogger:
    """Tests for `get_logger` function."""

    def test_console_handler(self) -> None:
        """Test a console handler is added correctly."""
        logger_name = "test_console_handler"
        logger = get_logger(logger_name, "1.2.3")

        assert logger.name == logger_name, "incorrect logger name"
        assert isinstance(
            logger.handlers[0], logging.StreamHandler
        ), "incorrect stream handler"

    def test_file_handler(self, tmp_path: pathlib.Path) -> None:
        """Test file handler is added and log file is created."""
        logger_name = "test_file_handler"
        log_file = tmp_path / "test.log"

        assert not log_file.is_file(), "log file already exists"

        logger = get_logger(
            logger_name, "1.2.3", console_handler=False, log_file_path=log_file
        )

        assert log_file.is_file(), "log file not created"
        assert isinstance(logger.handlers[0], logging.FileHandler), "incorrect file handler"

    def test_instantiate_message(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test instantiation message and code version are logged."""
        logger_name = "test_instantiate_message"
        version = "1.2.3-testing_version_logging"
        message = "testing instantiation message 123456789"

        logging.getLogger(logger_name).setLevel(logging.DEBUG)
        get_logger(logger_name, version, instantiate_msg=message)

        assert f"Code Version: v{version}" in caplog.messages, "missing version log"
        assert f"***  {message}  ***" in caplog.messages, "missing instantiation message"


class TestCaptureWarnings:
    """Test `capture_warnings` function."""

    def test_stream_handler_logger(self, warnings_logger: logging.Logger) -> None:
        """Test stream handler is added to warnings logger."""
        capture_warnings()

        assert len(warnings_logger.handlers) == 1, "incorrect number of handlers"
        assert isinstance(
            warnings_logger.handlers[0], logging.StreamHandler
        ), "error with stream handler"

    @pytest.mark.filterwarnings("ignore:testing warning")
    def test_file_handler_logger(
        self, tmp_path: pathlib.Path, warnings_logger: logging.Logger
    ) -> None:
        """Test file handler is added to warnings logger and log file is created."""
        log_file = tmp_path / "test.log"

        assert not log_file.is_file(), "log file already exists"

        capture_warnings(stream_handler=False, file_handler_args={"log_file": log_file})
        _run_warnings()

        assert log_file.is_file(), "log file not created"

        assert len(warnings_logger.handlers) == 1, "incorrect number of handlers"
        assert isinstance(
            warnings_logger.handlers[0], logging.FileHandler
        ), "error with file handler"

    @pytest.mark.filterwarnings("ignore:testing warning")
    @pytest.mark.skip(
        reason="fails when running all tests, I think due to warnings be captured by pytest"
    )
    def test_stream_handler_captures(
        self, caplog: pytest.LogCaptureFixture, warnings_logger: logging.Logger
    ) -> None:
        """Test Python warnings are captured."""
        del warnings_logger  # Fixture clears handlers from logger

        capture_warnings()
        _run_warnings()
        _check_warnings(caplog.text)

    @pytest.mark.skip(
        reason="fails when running all tests, I think due to warnings be captured by pytest"
    )
    def test_file_handler_captures(
        self, tmp_path: pathlib.Path, warnings_logger: logging.Logger
    ) -> None:
        """Test capturing warnings to log file."""
        del warnings_logger  # Fixture clears handlers from logger

        log_file = tmp_path / "test.log"

        assert not log_file.is_file(), "log file already exists"

        capture_warnings(stream_handler=False, file_handler_args={"log_file": log_file})
        _run_warnings()

        assert log_file.is_file(), "log file not created"

        with open(log_file, "rt", encoding="utf-8") as file:
            text = file.read()

        _check_warnings(text)
