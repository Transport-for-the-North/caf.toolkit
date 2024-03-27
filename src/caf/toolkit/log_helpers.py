# -*- coding: utf-8 -*-
"""
Helper functions for creating and managing a logger.

This module is designed to initialise Python logging consistently
and automatically log tool and system information.

Classes
-------
LogHelper
    Main context manager class for creating and managing loggers and
    logging key tool / system information.
TemporaryLogFile
    Context manager for adding a log file handler to a logger and
    removing it when done.
"""
from __future__ import annotations

# Built-Ins
import functools
import getpass
import logging
import os
import platform
import warnings
from typing import Annotated, Any, Iterable, Optional

# Third Party
import psutil
import pydantic
from psutil import _common
from pydantic import dataclasses, types

# # # CONSTANTS # # #
DEFAULT_CONSOLE_FORMAT = "[%(asctime)s - %(levelname)-8.8s] %(message)s"
DEFAULT_CONSOLE_DATETIME = "%H:%M:%S"
DEFAULT_FILE_FORMAT = "%(asctime)s [%(name)-40.40s] [%(levelname)-8.8s] %(message)s"
DEFAULT_FILE_DATETIME = "%d-%m-%Y %H:%M:%S"

# Regular expression for semantic versioning string from https://semver.org/
_SEMVER_REGEX = (
    r"^(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)\.(?P<patch>0|[1-9]\d*)"
    r"(?:-(?P<prerelease>(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)"
    r"(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?"
    r"(?:\+(?P<buildmetadata>[0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$"
)


# # # CLASSES # # #


class LoggingWarning(Warning):
    """Warnings from the `LogHelper` and other toolkit logging functionality."""


@dataclasses.dataclass
class ToolDetails:
    """Information about the current tool.

    Attributes
    ----------
    name: str
        Name of the tool.
    version: str
        Version of the tool, should be in semantic versioning
        format https://semver.org/.
    homepage: str, optional
        URL of the homepage for the tool.
    source_url: str, optional
        URL of the source code repository for the tool.
    """

    name: str
    version: Annotated[
        str, types.StringConstraints(strip_whitespace=True, pattern=_SEMVER_REGEX)
    ]
    homepage: Optional[pydantic.HttpUrl] = None
    source_url: Optional[pydantic.HttpUrl] = None

    def __str__(self) -> str:
        """Nicely formatted multi-line string."""
        message = ["Tool Information", "----------------"]

        # pylint false positive for __dataclass_fields__ no-member
        # pylint: disable=no-member
        length = functools.reduce(
            max, (len(i) for i in self.__dataclass_fields__ if getattr(self, i) is not None)
        )

        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if value is None:
                continue

            message.append(f"{name:<{length}.{length}} : {value}")

        return "\n".join(message)


@dataclasses.dataclass
class SystemInformation:
    """Information about the PC and Python version.

    Attributes
    ----------
    user: str
        Account name of the currently logged in user.
    pc_name: str
        Name of the PC.
    python_version: str
        Python version being used.
    operating_system: str
        Information about the name and version of OS.
    architecture: str
        Name of the machine architecture e.g. "AMD64".
    processor: str
        Name of the processor e.g. "Intel64 Family 6 Model 85 Stepping 7, GenuineIntel".
    cpu_count: int | None
        Number of logical CPU cores on the machine.
    total_ram: int | None
        Total virtual memory (bytes) on the machine.
    """

    user: str
    pc_name: str
    python_version: str
    operating_system: str
    architecture: str
    processor: str
    cpu_count: Optional[int]
    total_ram: Optional[int]

    @classmethod
    def load(cls) -> SystemInformation:
        """Load system information."""
        info = platform.uname()

        ram = psutil.virtual_memory()
        if ram is None:
            total_ram = None
        else:
            total_ram = ram.total

        try:
            user = getpass.getuser()
        except ModuleNotFoundError:
            # If LOGNAME, USER, LNAME and USERNAME are not set, then
            # uses the pwd module which isn't available on all systems
            user = "unknown"

        return SystemInformation(
            user=user,
            pc_name=info.node,
            python_version=platform.python_version(),
            operating_system=f"{info.system} {info.release} ({info.version})",
            architecture=info.machine,
            processor=info.processor,
            cpu_count=os.cpu_count(),
            total_ram=total_ram,
        )

    def __str__(self) -> str:
        """Nicely formatted multi-line string."""
        message = ["System Information", "------------------"]

        # pylint false positive for __dataclass_fields__ no-member
        # pylint: disable=no-member
        length = functools.reduce(max, (len(i) for i in self.__dataclass_fields__))

        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if value is None:
                value = "unknown"
            elif name == "total_ram":
                value = _common.bytes2human(value)

            message.append(f"{name:<{length}.{length}} : {value}")

        return "\n".join(message)


class LogHelper:
    """Class for managing Python loggers.

    Parameters
    ----------
    root_logger : str
        Name of the root logger to add handlers to,
        should be the name of the Python package.
    tool_details : ToolDetails
        Details of the tool being ran.
    console : bool, optional
        If True (default) output log messages to the console
        with default settings.
    log_file : os.PathLike, optional
        If given output log messages to a file with default
        settings.
    warning_capture : bool, optional
        If True (default) capture, and log, Python warnings.

    Examples
    --------
    When using Python's built-in logging functionality a module level
    logger constant should be used.

    >>> import logging
    >>>
    >>> LOG = logging.getLogger(__name__)

    This module constant should be used for logging any messages, in
    one of 5 levels.

    >>> LOG.debug("Log a debug message")
    >>> LOG.info("Log an info message")
    >>> LOG.warning("Log a warning message")
    >>> LOG.error("Log an error message")
    >>> LOG.critical("Log a critical message")

    To determine where log messages are written to (console / log file)
    the log handlers need to be setup, the `LogHelper` class can do
    this and will automatically clean-up upon exiting using a `with`
    statement.

    The example below shows how to setup logging with a log file using
    the `LogHelper` class, which will create a log file and write system
    and tool information to it automatically.

    >>> # Temp directory for testing purposes
    >>> tmp_path = getfixture('tmp_path')
    >>> path = tmp_path / "test.log"
    >>> details = ToolDetails("test", "1.2.3")
    >>>
    >>> with LogHelper(__package__, details, log_file=path):
    ...     # Add main function for running your tool here
    ...
    ...     # Any log messages within the with statement will be written to
    ...     # the log file, even if running in other functions / modules
    ...     LOG.info("Log messages using module logger")

    The following example shows how to setup logging with a custom console
    or file output, this also allows log files to be added after initial
    setup of `LogHelper` e.g. in another function after the output
    directory is known.

    >>> with LogHelper(__package__, details, console=False) as log_helper:
    ...     # Console handler with custom message format
    ...     log_helper.add_console_handler(ch_format="[%(levelname)-8.8s] %(message)s")
    ...     # File handler with custom message format
    ...     log_helper.add_file_handler(path, fh_format="[%(levelname)-8.8s] %(message)s")
    ...
    ...     # Write initialisation log message with system and tool information
    ...     log_helper.write_instantiate_message()
    """

    def __init__(
        self,
        root_logger: str,
        tool_details: ToolDetails,
        console: bool = True,
        log_file: os.PathLike | None = None,
        warning_capture: bool = True,
    ):
        self.logger_name = str(root_logger)
        self.logger = logging.getLogger(self.logger_name)
        self.logger.setLevel(logging.DEBUG)

        self.tool_details = tool_details
        self._warning_logger: logging.Logger | None = None

        if console:
            self.add_console_handler()

        if log_file is not None:
            self.add_file_handler(log_file)

        if len(self.logger.handlers) > 0:
            self.write_instantiate_message()
        else:
            warnings.warn(
                "LogHelper initialised without any logging handlers, "
                "`logging.basicConfig` will be called with default parameters "
                "at first log attempt if no handlers are added before that.",
                LoggingWarning,
            )

        if warning_capture:
            self.capture_warnings()

    def add_handler(self, handler: logging.Handler) -> None:
        """Add custom `handler` to the logger.

        This will also add handler to the warnings logger if
        warnings capture is enabled.

        Parameters
        ----------
        handler : logging.Handler
            Handler to add.
        """
        self.logger.addHandler(handler)

        if self._warning_logger is not None:
            self._warning_logger.addHandler(handler)

    def add_console_handler(
        self,
        ch_format: str = DEFAULT_CONSOLE_FORMAT,
        datetime_format: str = DEFAULT_CONSOLE_DATETIME,
        log_level: int = logging.INFO,
    ) -> None:
        """Add custom console handler to the logger.

        Parameters
        ----------
        ch_format:
            A string defining a custom formatting to use for the StreamHandler().
            Defaults to "[%(levelname)-8.8s] %(message)s".

        datetime_format:
            The datetime format to use when logging to the console.
            Defaults to "%H:%M:%S"

        log_level:
            The logging level to give to the StreamHandler.

        See Also
        --------
        `get_console_handler`
        """
        handler = get_console_handler(ch_format, datetime_format, log_level)
        self.add_handler(handler)

    def add_file_handler(
        self,
        log_file: os.PathLike,
        fh_format: str = DEFAULT_FILE_FORMAT,
        datetime_format: str = DEFAULT_FILE_DATETIME,
        log_level: int = logging.DEBUG,
    ) -> None:
        """Add custom file handler to the logger.

        Parameters
        ----------
        log_file:
            The path to a file to output the log

        fh_format:
            A string defining a custom formatting to use for the StreamHandler().
            Defaults to
            "%(asctime)s [%(name)-40.40s] [%(levelname)-8.8s] %(message)s".

        datetime_format:
            The datetime format to use when logging to the console.
            Defaults to "%d-%m-%Y %H:%M:%S"

        log_level:
            The logging level to give to the FileHandler.

        See Also
        --------
        `get_file_handler`
        """
        handler = get_file_handler(log_file, fh_format, datetime_format, log_level)
        self.add_handler(handler)

    def capture_warnings(self) -> None:
        """Capture warnings using logging.

        Runs `logging.captureWarnings(True)` to capture warnings then
        adds all the handlers from the root `logger`.

        See Also
        --------
        `capture_warnings`
        """
        logging.captureWarnings(True)

        self._warning_logger = logging.getLogger("py.warnings")

        for handler in self.logger.handlers:
            if handler in self._warning_logger.handlers:
                continue

            self._warning_logger.addHandler(handler)

    def write_instantiate_message(self) -> None:
        """Log instatiation message with tool and system information."""
        write_instantiate_message(self.logger, self.tool_details.name)
        write_information(self.logger, self.tool_details)

    @staticmethod
    def _cleanup_handlers(logger: logging.Logger) -> None:
        """Flush and close all handlers before removing from the `logger`."""
        for handler in logger.handlers:
            handler.flush()
            handler.close()

        logger.handlers.clear()

    def cleanup_handlers(self) -> None:
        """Flush and close all handlers before removing from the logger.

        Cleans up `logger` and warnings logger (if exists).
        """
        self._cleanup_handlers(self.logger)

        if self._warning_logger is not None:
            self._cleanup_handlers(self._warning_logger)

    def __enter__(self):
        """Initialise class with 'with' statement."""
        return self

    def __exit__(self, exc_type, exc, exc_tb):
        """Write any error to the logger and closes the file."""
        if exc_type is not None or exc is not None or exc_tb is not None:
            self.logger.critical("Oh no a critical error occurred", exc_info=True)
        else:
            self.logger.info("Program completed without any critical errors")

        self.logger.info("Closing log file")
        self.cleanup_handlers()
        logging.shutdown()


class TemporaryLogFile:
    """Add temporary log file to a logger.

    This context manager class is designed to temporarily add another
    log file to an existing logger for any messages in a `with`
    statement. When adding a new file handler to an existing logger
    any previous handlers will remain, so log messages will be written
    to the new and old log files for example.

    Parameters
    ----------
    logger : logging.Logger
        Logger to add FileHandler to.

    log_file : os.PathLike
        Path to new log file to create.

    base_log_file : os.PathLike, optional
        Path to base log file, location will be logged
        in new log file.

    kwargs : Keyword arguments, optional
        Any arguments to pass to `get_file_handler`.

    See Also
    --------
    `LogHelper`: for setting up logging for a tool.

    Examples
    --------
    When using Python's built-in logging functionality a module level
    logger constant should be used.

    >>> import logging
    >>>
    >>> LOG = logging.getLogger(__name__)

    The code below is defining the log file path for testing purposes.

    >>> log_file = getfixture('tmp_path') / "test.log"

    Setting up a new temporary log file for a single module can be done
    using the following:

    >>> with TemporaryLogFile(LOG, log_file):
    ...     LOG.info("Message logged to new file")
    ...     # Includes logging messages from functions which are
    ...     # in the current module only

    >>> LOG.info("Message not in new file")

    Logging all messages from the current package to the new file can
    be done by passing the package logger.

    >>> with TemporaryLogFile(logging.getLogger(__package__), log_file):
    ...     LOG.info("Message logged to new file")
    ...     # Includes logging messages from functions called here which
    ...     # are in other modules in the package
    """

    def __init__(
        self,
        logger: logging.Logger,
        log_file: os.PathLike,
        base_log_file: os.PathLike | None = None,
        **kwargs,
    ) -> None:
        self.logger = logger
        self.log_file = log_file

        self.logger.debug('Creating temporary log file: "%s"', self.log_file)
        self.handler = get_file_handler(log_file, **kwargs)
        self.logger.addHandler(self.handler)

        if base_log_file is not None:
            self.logger.debug('Base log file: "%s"', base_log_file)

    def __enter__(self) -> TemporaryLogFile:
        """Initialise TemporaryLogFile."""
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        """Close temporary log file."""
        # pylint: disable=invalid-name
        if exc_type is not None or exc is not None or exc_tb is not None:
            self.logger.critical("Oh no a critical error occurred", exc_info=True)

        self.logger.removeHandler(self.handler)
        self.logger.debug('Closed temporary log file: "%s"', self.log_file)


# # # FUNCTIONS # # #
def write_information(
    logger: logging.Logger, tool_details: ToolDetails | None = None, system_info: bool = True
) -> None:
    """Write tool and system information to `logger`.

    Parameters
    ----------
    logger : logging.Logger
        Logger to write to
    tool_details : ToolDetails, optional
        Tool details to write to logger, not written if None.
    system_info : bool, default True
        Whether, or not, to load `SystemInformation` and write to logger.
    """
    if tool_details is not None:
        logger.info("\n%s", tool_details)

    if system_info:
        info = SystemInformation.load()
        logger.debug("\n%s", info)


def write_instantiate_message(
    logger: logging.Logger,
    instantiate_message: str,
) -> None:
    """Write an instantiation message to logger.

    Instantiation message will be output at the logging.DEBUG level,
    and will be wrapped in a line of asterisk before and after.

    Parameters
    ----------
    logger:
        The logger to write the message to.

    instantiate_message:
        The message to output on instantiation. This will be output at the
        logging.DEBUG level, and will be wrapped in a line of hyphens before
        and after.

    Returns
    -------
    None
    """
    msg = f"***  {instantiate_message}  ***"

    logger.debug("")
    logger.debug("*" * len(msg))
    logger.debug(msg)
    logger.debug("*" * len(msg))


def get_custom_logger(
    logger_name: str,
    code_version: str,
    instantiate_msg: Optional[str] = None,
    log_handlers: Optional[Iterable[logging.Handler]] = None,
) -> logging.Logger:
    """Create a standard logger using the CAF template.

    Creates the logger, prints out the standard instantiation messages,
    and returns the logger.
    See `get_logger()` to get a default logger with default file and console
    handlers.

    Parameters
    ----------
    logger_name:
        The name of the new logger.

    code_version:
        A string describing the current version of the code being logged.

    log_handlers:
        A list of log handlers to add to the generated
        logger. Any valid logging handler can be accepted

    instantiate_msg:
        A message to output on instantiation. This will be output at the
        logging.DEBUG level, and will be wrapped in a line of asterisk before
        and after.

    Returns
    -------
    logger:
        A logger with the given handlers attached

    See Also
    --------
    `get_logger()`
    """
    # Init
    log_handlers = list() if log_handlers is None else log_handlers

    logger = logging.getLogger(logger_name)
    for handler in log_handlers:
        logger.addHandler(handler)

    if instantiate_msg is not None:
        write_instantiate_message(logger, instantiate_msg)

    if code_version:
        logger.info("Code Version: v%s", code_version)

    return logger


def get_logger(
    logger_name: str,
    code_version: str,
    console_handler: bool = True,
    instantiate_msg: Optional[str] = None,
    log_file_path: Optional[os.PathLike] = None,
) -> logging.Logger:
    """Create a standard logger using the CAF template.

    Creates and sets up the logger, prints out the standard instantiation
    messages, and returns the logger.
    If more custom handlers are needed, `get_custom_logger()` to follow the same
    standard with more flexibility.

    Parameters
    ----------
    logger_name:
        The name of the new logger.

    code_version:
        A string describing the current version of the code being logged.

    instantiate_msg:
        A message to output on instantiation. This will be output at the
        logging.DEBUG level, and will be wrapped in a line of asterisk before
        and after.

    log_file_path:
        The path to a file to output the log. This uses the default parameters
        from `get_file_handler()`

    console_handler:
        Whether to attach a default logging.StreamHandler object, generated
        by `get_console_handler()`.

    Returns
    -------
    logger:
        A logger with the given handlers attached.

    See Also
    --------
    `get_custom_logger()`
    `get_file_handler()`
    `get_console_handler()`
    """
    log_handlers = list()
    if log_file_path is not None:
        log_handlers.append(get_file_handler(log_file_path))

    if console_handler:
        log_handlers.append(get_console_handler())

    return get_custom_logger(
        logger_name=logger_name,
        code_version=code_version,
        instantiate_msg=instantiate_msg,
        log_handlers=log_handlers,
    )


def get_console_handler(
    ch_format: str = DEFAULT_CONSOLE_FORMAT,
    datetime_format: str = DEFAULT_CONSOLE_DATETIME,
    log_level: int = logging.INFO,
) -> logging.StreamHandler:
    """Create a console handles for a logger.

    Parameters
    ----------
    ch_format:
        A string defining a custom formatting to use for the StreamHandler().
        Defaults to "[%(levelname)-8.8s] %(message)s".

    datetime_format:
        The datetime format to use when logging to the console.
        Defaults to "%H:%M:%S"

    log_level:
        The logging level to give to the StreamHandler.

    Returns
    -------
    console_handler:
        A logging.StreamHandler object using the format in ch_format.
    """
    handler = logging.StreamHandler()
    handler.setLevel(log_level)
    handler.setFormatter(logging.Formatter(ch_format, datefmt=datetime_format))
    return handler


def get_file_handler(
    log_file: os.PathLike,
    fh_format: str = DEFAULT_FILE_FORMAT,
    datetime_format: str = DEFAULT_FILE_DATETIME,
    log_level: int = logging.DEBUG,
) -> logging.StreamHandler:
    """Create a console handles for a logger.

    Parameters
    ----------
    log_file:
        The path to a file to output the log

    fh_format:
        A string defining a custom formatting to use for the StreamHandler().
        Defaults to
        "%(asctime)s [%(name)-40.40s] [%(levelname)-8.8s] %(message)s".

    datetime_format:
        The datetime format to use when logging to the console.
        Defaults to "%d-%m-%Y %H:%M:%S"

    log_level:
        The logging level to give to the FileHandler.

    Returns
    -------
    console_handler:
        A logging.StreamHandler object using the format in ch_format.
    """
    handler = logging.FileHandler(log_file)
    handler.setLevel(log_level)
    handler.setFormatter(logging.Formatter(fh_format, datefmt=datetime_format))
    return handler


def capture_warnings(
    stream_handler: bool = True,
    stream_handler_args: Optional[dict[str, Any]] = None,
    file_handler_args: Optional[dict[str, Any]] = None,
) -> None:
    """Capture warnings using logging.

    Runs `logging.captureWarnings(True)` to capture warnings then
    sets up custom stream and file handlers if required.

    Parameters
    ----------
    stream_handler : bool, default True
        Add stream handler to warnings logger.

    stream_handler_args : Dict[str, Any], optional
        Custom arguments for the stream handler,
        passed to `get_console_handler`.

    file_handler_args : Dict[str, Any], optional
        Custom arguments for the file handler,
        passed to `get_file_handler`.
    """
    logging.captureWarnings(True)

    warning_logger = logging.getLogger("py.warnings")

    if stream_handler or stream_handler_args is not None:
        if stream_handler_args is None:
            stream_handler_args = {}
        warning_logger.addHandler(get_console_handler(**stream_handler_args))

    if file_handler_args is not None:
        warning_logger.addHandler(get_file_handler(**file_handler_args))
