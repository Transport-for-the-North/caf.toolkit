# -*- coding: utf-8 -*-
"""
Created on: 18/08/2022
Updated on:

Original author: Ben Taylor
Last update made by:
Other updates made by:

File purpose:

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
    raise_error: bool = False,
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

    raise_error:
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
            if raise_error:
                raise exception

    return lst
