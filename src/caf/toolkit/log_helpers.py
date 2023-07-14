# -*- coding: utf-8 -*-
"""
Helper functions for creating and managing a logger.

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
from typing import Any, Iterable, Optional

# Third Party
import pydantic
from pydantic import dataclasses

# Local Imports
# pylint: disable=import-error,wrong-import-position

# pylint: enable=import-error,wrong-import-position

# # # CONSTANTS # # #
DEFAULT_CONSOLE_FORMAT = "[%(asctime)s - %(levelname)-8.8s] %(message)s"
DEFAULT_CONSOLE_DATETIME = "%H:%M:%S"
DEFAULT_FILE_FORMAT = "%(asctime)s [%(name)-40.40s] [%(levelname)-8.8s] %(message)s"
DEFAULT_FILE_DATETIME = "%d-%m-%Y %H:%M:%S"


# # # CLASSES # # #
@dataclasses.dataclass
class ToolDetails:
    """Information about the current tool.

    Attributes
    ----------
    name: str
        Name of the tool.
    version: str
        Version of the tool.
    homepage: str, optional
        URL of the homepage for the tool.
    source_url: str, optional
        URL of the source code repository for the tool.
    """

    name: str
    version: str
    homepage: Optional[pydantic.HttpUrl] = None
    source_url: Optional[pydantic.HttpUrl] = None

    def __str__(self) -> str:
        """Nicely formatted multi-line string."""
        message = ["Tool Information", "----------------"]

        # pylint false positive for __dataclass_fields__ no-member
        # pylint: disable=no-member
        length = functools.reduce(max, (len(i) for i in self.__dataclass_fields__))

        for name in self.__dataclass_fields__:
            message.append(f"{name:<{length}.{length}} : {getattr(self, name)}")

        return "\n".join(message)


@dataclasses.dataclass
class SystemInformation:
    """Information about the PC and Python version.

    Attributes
    ----------
    user: str
        Account name of the currenly logged in user.
    pc_name: str
        Name of the PC.
    python_version: str
        Python version being used.
    operating_system: str
        Information about the name and version of OS.
    architecture: str
        Name of the machine architecture e.g. "AMD64".
    """

    user: str
    pc_name: str
    python_version: str
    operating_system: str
    architecture: str

    @classmethod
    def load(cls) -> SystemInformation:
        """Load system information."""
        info = platform.uname()

        return SystemInformation(
            user=getpass.getuser(),
            pc_name=info.node,
            python_version=platform.python_version(),
            operating_system=f"{info.system} {info.release} ({info.version})",
            architecture=info.machine,
        )

    def __str__(self) -> str:
        """Nicely formatted multi-line string."""
        message = ["System Information", "------------------"]

        # pylint false positive for __dataclass_fields__ no-member
        # pylint: disable=no-member
        length = functools.reduce(max, (len(i) for i in self.__dataclass_fields__))

        for name in self.__dataclass_fields__:
            message.append(f"{name:<{length}.{length}} : {getattr(self, name)}")

        return "\n".join(message)


class LogHelper:
    """Class for managing Python loggers."""

    # TODO(MB) Add examples to docstring

    def __init__(
        self,
        root_logger: str,
        tool_details: ToolDetails,
        console: bool = True,
        log_file: os.PathLike | None = None,
    ):
        self.logger_name = str(root_logger)
        self.logger = logging.getLogger(self.logger_name)
        self.logger.setLevel(logging.DEBUG)

        self.tool_details = tool_details

        if console:
            handler = get_console_handler()
            self.logger.addHandler(handler)

        if log_file is not None:
            handler = get_file_handler(log_file)
            self.logger.addHandler(handler)

        self.write_instantiate_message()

    def add_console_handler(
        self,
        ch_format: str = DEFAULT_CONSOLE_FORMAT,
        datetime_format: str = DEFAULT_CONSOLE_DATETIME,
        log_level: int = logging.INFO,
    ) -> None:
        """Wrapper around `get_console_handler` to add custom console handlers.

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
        """
        handler = get_console_handler(ch_format, datetime_format, log_level)
        self.logger.addHandler(handler)

    def add_file_handler(
        self,
        log_file: os.PathLike,
        fh_format: str = DEFAULT_FILE_FORMAT,
        datetime_format: str = DEFAULT_FILE_DATETIME,
        log_level: int = logging.DEBUG,
    ) -> None:
        """Wrapper around `get_file_handler` to add custom file handlers.

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
        """
        handler = get_file_handler(log_file, fh_format, datetime_format, log_level)
        self.logger.addHandler(handler)

    def capture_warnings(self) -> None:
        """Capture warnings using logging.

        Runs `logging.captureWarnings(True)` to capture warnings then
        adds all the handlers from the root `logger`.
        """
        logging.captureWarnings(True)

        warning_logger = logging.getLogger("py.warnings")

        for handler in self.logger.handlers:
            if handler in warning_logger.handlers:
                continue

            warning_logger.addHandler(handler)

    def write_instantiate_message(self) -> None:
        """Write instatiation message with tool / system information."""
        write_instantiate_message(self.logger, self.tool_details.name)
        write_information(self.logger, self.tool_details)

    def __enter__(self):
        """Called when initialising class with 'with' statement."""
        return self

    def __exit__(self, excepType, excepVal, traceback):
        """Called when exiting with statement, writes any error to the logger and closes the file."""
        if excepType is not None or excepVal is not None or traceback is not None:
            self.logger.critical("Oh no a critical error occurred", exc_info=True)
        else:
            self.logger.info("Program completed without any critical errors")

        self.logger.info("Closing log file")
        logging.shutdown()


class TemporaryLogFile:
    """Add temporary log file to a logger."""

    # TODO(MB) Add examples to docstring

    def __init__(self, logger: logging.Logger, log_file: os.PathLike, **kwargs) -> None:
        """Add temporary log file handler to `logger`.

        Parameters
        ----------
        logger : logging.Logger
            Logger to add FileHandler to.

        log_file : nd.PathLike
            Path to new log file to create.

        kwargs : Keyword arguments, optional
            Any arguments to pass to `get_file_handler`.
        """
        self.logger = logger
        self.log_file = log_file
        self.logger.debug('Creating temporary log file: "%s"', self.log_file)
        self.handler = get_file_handler(log_file, **kwargs)
        self.logger.addHandler(self.handler)
        # TODO(MB) Log location of base log file

    def __enter__(self) -> TemporaryLogFile:
        """Initialise TemporaryLogFile."""
        return self

    def __exit__(self, excepType, excepVal, traceback) -> None:
        """Close temporary log file."""
        # pylint: disable=invalid-name
        if excepType is not None or excepVal is not None or traceback is not None:
            self.logger.critical("Oh no a critical error occurred", exc_info=True)

        self.logger.removeHandler(self.handler)
        self.logger.debug('Closed temporary log file: "%s"', self.log_file)


# # # FUNCTIONS # # #
def write_information(
    logger: logging.Logger, tool_details: ToolDetails | None = None, system_info: bool = True
) -> None:
    """Write tool and system information to `logger`.

    _extended_summary_

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
