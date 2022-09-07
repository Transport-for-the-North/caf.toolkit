# -*- coding: utf-8 -*-
"""A toolbox of generic useful functionality.

Most of these tools will be used elsewhere in the codebase too
"""
# Built-Ins
from typing import Any
from typing import Iterable

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


def is_none_like(obj: Any) -> bool:
    """Check if an object is None-like.

    An object is considered None-like if one of the following is True:
    - The `obj is None` equates to True
    - if obj is a string: it is equal to 'none' once stripped and lowered
    - if obj is a list: if each item in the list is none-like

    Parameters
    ----------
    obj:
        Object to check

    Returns
    -------
    bool:
        True if obj is none-like else False
    """
    if obj is None:
        return True

    if isinstance(obj, str):
        if obj.lower().strip() == "none":
            return True

    if isinstance(obj, list):
        return all(is_none_like(x) for x in obj)

    return False


def equal_ignore_order(one: Iterable[Any], two: Iterable[Any]) -> bool:
    """Check whether two iterables contain the same items, ignoring order.

    Only use when elements are neither hashable nor sortable, as this
    method is quite slow.
    if hashable use: set(a) == set(b)
    if sortable use: sorted(a) ==  sorted(b)
    """
    unmatched = list(two)
    for element in one:
        try:
            unmatched.remove(element)
        except ValueError:
            return False
    return not unmatched
