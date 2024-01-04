# -*- coding: utf-8 -*-
"""Tools to validate items. These function return either True or False.

These are mostly commonly used validations across the codebase.
"""
# Built-Ins
import logging
from typing import Any

# Local Imports
from caf.toolkit import toolbox

# # # CONSTANTS # # #
LOG = logging.getLogger(__name__)

# # # CLASSES # # #


# # # FUNCTIONS # # #
# TODO(BT): Can this take a Collection instead?
def unique_list(unique_vals: list[Any], name: str = "unique_zones") -> None:
    """Validate that a list of unique values is unique.

    Parameters
    ----------
    unique_vals:
        The list of unique values to validate.

    name:
        The name to give to `unique_vals` if an error is raised.

    Returns
    -------
    None

    Raises
    ------
    ValueError:
        If `unique_vals` is not a unique list
    """
    if not toolbox.is_unique_list(unique_vals):
        raise ValueError(
            f"Duplicate values found in {name}, making it invalid."
            f"\n{unique_vals}\n{type(unique_vals[0])}"
        )
