# -*- coding: utf-8 -*-
"""A toolbox of useful math related functionality.

Most will be used elsewhere in the codebase too
"""
# Built-Ins
import math
from typing import Union
from typing import Collection

# Third Party
import numpy as np

# Local Imports
# pylint: disable=import-error,wrong-import-position

# pylint: enable=import-error,wrong-import-position

# # # CONSTANTS # # #

# # # CLASSES # # #


# # # FUNCTIONS # # #
def list_is_almost_equal(
    vals: list[Union[int, float]],
    rel_tol: float = 0.0001,
    abs_tol: float = 0.0,
) -> bool:
    """Check if a list of values are similar.

    Whether two values are considered close is determined according to given
    absolute and relative tolerances.
    Wrapper around ` math.isclose()` to set default values for `rel_tol` and
    `abs_tol`.

    Parameters
    ----------
    vals:
        The values to check if similar

    rel_tol:
        The relative tolerance – it is the maximum allowed difference
        between two values to be considered similar,
        relative to the largest absolute value .
        By default, this is set to 0.0001,
        meaning the values must be within 0.01% of each other.

    abs_tol:
        The minimum absolute tolerance – useful for comparisons near
        zero. Must be at least zero.

    Returns
    -------
    is_close:
         True if `vals` are all similar. False otherwise.

    See Also
    --------
    `math.isclose()`
    """
    first_val, rest = vals[0], vals[1:]
    return all(is_almost_equal(first_val, x, rel_tol, abs_tol) for x in rest)


def is_almost_equal(
    val1: Union[int, float],
    val2: Union[int, float],
    rel_tol: float = 0.0001,
    abs_tol: float = 0.0,
) -> bool:
    """Check if two values are similar.

    Whether two values are considered close is determined according to given
    absolute and relative tolerances.
    Wrapper around ` math.isclose()` to set default values for `rel_tol` and
    `abs_tol`.

    Parameters
    ----------
    val1:
        The first value to check if close to `val2`

    val2:
        The second value to check if close to `val1`

    rel_tol:
        The relative tolerance – it is the maximum allowed difference
        between `val1` and `val2`,
        relative to the larger absolute value of `val1` or
        `val2`. By default, this is set to 0.0001,
        meaning the values must be within 0.01% of each other.

    abs_tol:
        The minimum absolute tolerance – useful for comparisons near
        zero. Must be at least zero.

    Returns
    -------
    is_close:
         True if `val1` and `val2` are similar. False otherwise.

    See Also
    --------
    `math.isclose()`
    """
    return math.isclose(val1, val2, rel_tol=rel_tol, abs_tol=abs_tol)


def root_mean_squared_error(
    targets: Collection[np.ndarray],
    achieved: Collection[np.ndarray],
) -> float:
    """Calculate the root-mean-squared error between targets and achieved.

    Two lists of corresponding values are zipped together, differences taken
    (residuals) and the RMSE calculated.

    Parameters
    ----------
    targets:
        A list of all the targets that `achieved` should have reached. Must
        be the same length as `achieved`.

    achieved:
        A list of all the achieved values. Must be the same length as `targets`

    Returns
    -------
    rmse:
        A float value indicating the total root-mean-squared-error of targets
        and achieved

    Raises
    ------
    ValueError:
        If `targets` and `achieved` are not the same length
    """
    if len(targets) != len(achieved):
        raise ValueError(
            "targets and achieved must be the same length. "
            f"targets length: {len(targets)}, achieved length: {len(achieved)}"
        )

    squared_diffs = list()
    for target, ach in zip(targets, achieved):
        diffs = (target - ach) ** 2
        squared_diffs += diffs.flatten().tolist()

    return float(np.mean(squared_diffs) ** 0.5)
