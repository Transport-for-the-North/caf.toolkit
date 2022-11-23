# -*- coding: utf-8 -*-
"""
"""
# Built-Ins
import logging
import warnings

# Third Party

# Local Imports
# pylint: disable=import-error,wrong-import-position
from caf.toolkit.core import ErrorHandlingLiteral

# pylint: enable=import-error,wrong-import-position

# # # CONSTANTS # # #
LOG = logging.getLogger(__name__)

# # # CLASSES # # #


# # # FUNCTIONS # # #
# Private functions
def handle_error(
    msg: str,
    error: Exception,
    how: ErrorHandlingLiteral = "raise",
) -> None:
    """Handles internal errors depending on caller preference"""
    if how == "ignore":
        return

    msg = f"{error.__name__}: {msg}"
    if how == "raise":
        raise error(msg)

    if how == "print":
        print(msg)

    elif how == "warn":
        warnings.warn(msg)
    else:
        raise ValueError(
            f"Invalid value given to for 'how' expected one of: "
            f"{ErrorHandlingLiteral.__args__}"
        )
