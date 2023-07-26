# -*- coding: utf-8 -*-
"""Basic utility functions for pandas objects."""
# Built-Ins
from typing import TypeVar
from typing import Protocol

# Third Party
import numpy as np

# Local Imports
# pylint: disable=import-error,wrong-import-position

# pylint: enable=import-error,wrong-import-position

# # # CONSTANTS # # #
# TYPES
_T = TypeVar("_T")


class CastProtocol(Protocol):
    # pylint: disable=too-few-public-methods
    """Type that as the `dtype` property and `astype` method."""

    dtype: np.dtype

    def astype(self: _T, dtype: np.dtype) -> _T:
        """Cast this object to a new type."""


_U = TypeVar("_U", bound=CastProtocol)
_V = TypeVar("_V", bound=CastProtocol)


# # # CLASSES # # #


# # # FUNCTIONS # # #
def cast_to_common_type(
    x1: _U,
    x2: _V,
) -> tuple[_U, _V]:
    """Cast two objects to the same datatype.

    The passed in objects must have the `dtype` attribute, and a call to
    `astype(new_type)` must return a copy of the object as `new_type`.
    Most, if not all, pandas objects meet this criteria.

    Parameters
    ----------
    x1:
        The first object to cast the type of.

    x2:
        The second object to cast the type of.

    Returns
    -------
    cast_x1:
        `x1` cast to a common type, as defined by `np.promote_types(x1.dtype, x2.dtype)`

    cast_x2:
        `x2` cast to a common type, as defined by `np.promote_types(x1.dtype, x2.dtype)`
    """
    # Simple case
    if x1.dtype == x2.dtype:
        return x1, x2

    # If one is object - cast to other type
    if x1.dtype == "object":
        return x1.astype(x2.dtype), x2
    if x2.dtype == "object":
        return x1, x2.astype(x1.dtype)

    # Cast to common type
    common_dtype = np.promote_types(x1.dtype, x2.dtype)
    return x1.astype(common_dtype), x2.astype(common_dtype)
