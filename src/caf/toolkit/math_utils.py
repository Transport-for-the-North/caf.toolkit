# -*- coding: utf-8 -*-
"""A toolbox of useful math related functionality.

Most will be used elsewhere in the codebase too
"""
# Built-Ins
import math
from typing import Union

# Third Party

# Local Imports
# pylint: disable=import-error,wrong-import-position

# pylint: enable=import-error,wrong-import-position

# # # CONSTANTS # # #

# # # CLASSES # # #


# # # FUNCTIONS # # #
def is_almost_equal(
    val1: Union[int, float],
    val2: Union[int, float],
    rel_tol: float = 0.0001,
    abs_tol: float = 0.0,
) -> bool:
    """Check if two values are similar

    Whether two values are considered close is determined according to given
    absolute and relative tolerances.
    Wrapper around ` math.isclose()` to set default values for `rel_tol` and
    `abs_tol`.

    Parameters
    -----------
    val1:
        The first value to check if close to `val2`

    val2:
        The second value to check if close to `val1`

    rel_tol:
        The relative tolerance – it is the maximum allowed difference
        between the sum of `val1` and `val2`,
        relative to the larger absolute value of `val1` or
        `val2`. By default, this is set to 0.0001,
        meaning the values must be within 0.01% of each other.

    abs_tol:
        The minimum absolute tolerance – useful for comparisons near
        zero.Must be at least zero.

    Returns
    -------
    is_close:
         True if `val1` and `val2` are similar. False otherwise.

    See Also
    --------
    `math.isclose()`
    """
    return math.isclose(val1, val2, rel_tol=rel_tol, abs_tol=abs_tol)
