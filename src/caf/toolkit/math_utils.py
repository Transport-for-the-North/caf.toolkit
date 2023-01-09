# -*- coding: utf-8 -*-
"""A toolbox of useful math related functionality.

Most will be used elsewhere in the codebase too
"""
# Built-Ins
import math
from typing import Union
from typing import Collection

# Third Party
import sparse
import numpy as np
import pandas as pd

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
    targets: Collection[Union[np.ndarray, sparse.COO]],
    achieved: Collection[Union[np.ndarray, sparse.COO]],
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
    try:
        if len(targets) != len(achieved):
            raise ValueError(
                "targets and achieved must be the same length. "
                f"targets length: {len(targets)}, achieved length: {len(achieved)}"
            )
    except TypeError as error:
        raise TypeError(
            "Expected a collection, got the following instead:\n"
            f"targets: {type(targets)}\n"
            f"achieved: {type(achieved)}"
        ) from error

    squared_diffs: list[float] = list()
    for i, (target, ach) in enumerate(zip(targets, achieved)):
        try:
            diffs = (target - ach) ** 2
        except ValueError as error:
            raise ValueError(
                "Could not broadcast target and achieved to the same shape at "
                f"index {i}. See above exception for mis-matching shapes."
            ) from error

        # Nice and easy with dense array
        if isinstance(diffs, np.ndarray):
            squared_diffs += diffs.flatten().tolist()

        elif isinstance(diffs, sparse.COO):
            # TODO(BT): Not ideal making this dense, but not sure on a smarter
            #  way to do this right now.
            squared_diffs += diffs.todense().flatten().tolist()
        else:
            raise TypeError(f"Cannot handle arrays of type '{type(diffs)}'.")

    return float(np.mean(squared_diffs) ** 0.5)


def nan_report_with_input(
    array: np.ndarray, input_dict: dict[str, np.ndarray]
) -> pd.DataFrame:
    """Create a report of NaN values in relative matrix locations.

    Uses an input `array` (usually the result of a calculation) to find the
    locations of all np.NaN values. Once found, the index of these values are
    found in the `input_dict` arrays. These values are then used to generate
    a report.

    Parameters
    ----------
    array:
        The array to find the NaN values in.

    input_dict:
        A dictionary of arrays of the same shape as `array`. The keys are the
        names to define each input, and the values the array.

    Returns
    -------
    report:
        A pandas DataFrame reporting where the np.NaN values are.
        Will have a column named "axis_{i}" for each axis in array.
        Will also have additional columns named "output" (for `array` values)
        and one named after each key in `input_dict` for the corresponding
        array values.
    """
    # Create the columns
    mat_idxs = np.isnan(array).nonzero()
    idx_cols = {f"axis_{i}": x for i, x in enumerate(mat_idxs)}
    output_col = {"output": array[mat_idxs]}
    in_cols = {k: v[mat_idxs] for k, v in input_dict.items()}

    # Combine and convert to DataFrame
    final_dict = dict()
    for ddict in [idx_cols, output_col, in_cols]:
        final_dict.update(ddict)
    return pd.DataFrame(final_dict)
