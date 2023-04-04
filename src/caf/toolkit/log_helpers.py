# -*- coding: utf-8 -*-
"""Helper functions for creating and managing a logger."""
from __future__ import annotations

# Built-Ins
import os
import logging

from typing import Any
from typing import Iterable
from typing import Optional

# Third Party

# Local Imports
# pylint: disable=import-error,wrong-import-position

# pylint: enable=import-error,wrong-import-position

# # # CONSTANTS # # #
LOG = logging.getLogger(__name__)

# # # CLASSES # # #


# # # FUNCTIONS # # #
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
    ch_format: Optional[str] = None,
    datetime_format: Optional[str] = None,
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
    if ch_format is None:
        ch_format = "[%(asctime)s - %(levelname)-8.8s] %(message)s"

    if datetime_format is None:
        datetime_format = "%H:%M:%S"

    # Create console handler
    handler = logging.StreamHandler()
    handler.setLevel(log_level)
    handler.setFormatter(logging.Formatter(ch_format, datefmt=datetime_format))
    return handler


def get_file_handler(
    log_file: os.PathLike,
    fh_format: Optional[str] = None,
    datetime_format: Optional[str] = None,
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
        The logging level to give to the StreamHandler.

    Returns
    -------
    console_handler:
        A logging.StreamHandler object using the format in ch_format.
    """
    # Init
    if fh_format is None:
        fh_format = "%(asctime)s [%(name)-40.40s] [%(levelname)-8.8s] %(message)s"

    if datetime_format is None:
        datetime_format = "%d-%m-%Y %H:%M:%S"

    # Create console handler
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


class TemporaryLogFile:
    """Add temporary log file to a logger."""

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
