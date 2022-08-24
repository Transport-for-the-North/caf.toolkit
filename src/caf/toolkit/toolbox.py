# -*- coding: utf-8 -*-
"""A toolbox of generic useful functionality.

Most of these tools will be used elsewhere in the codebase too
"""
# Built-Ins
from typing import Any

# Third Party

# Local Imports
# pylint: disable=import-error,wrong-import-position

# pylint: enable=import-error,wrong-import-position

# # # CONSTANTS # # #

# # # CLASSES # # #


# # # FUNCTIONS # # #
def list_safe_remove(
    lst: list[Any],
    remove: list[Any],
    throw_error: bool = False,
    inplace: bool = False,
) -> list[Any]:
    """
    Remove items from a list without raising an error.

    Parameters
    ----------
    lst:
        The list to remove items from

    remove:
        The items to remove from lst

    throw_error:
        Whether to raise an error or not when an item in `remove` is
        not contained in lst

    inplace:
        Whether to remove the items in-place, or return a copy of lst

    Returns
    -------
    lst:
        lst with removed items removed from it
    """
    # Init
    if not inplace:
        lst = lst.copy()

    for item in remove:
        try:
            lst.remove(item)
        except ValueError as exception:
            if throw_error:
                raise exception

    return lst
