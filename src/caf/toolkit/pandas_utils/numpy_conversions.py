# -*- coding: utf-8 -*-
"""Conversion methods between numpy and pandas formats."""
# Built-Ins
import logging
import operator
import functools

from typing import Any
from typing import Union
from typing import Literal
from typing import overload
from typing import Collection

# Third Party
import sparse
import numpy as np
import pandas as pd

# Local Imports
# pylint: disable=import-error,wrong-import-position
from caf.toolkit.pandas_utils import df_handling
from caf.toolkit.core import _internal_utils as internal_utils

from caf.toolkit.core import ErrorHandlingLiteral

# pylint: enable=import-error,wrong-import-position

# # # CONSTANTS # # #
LOG = logging.getLogger(__name__)

# # # CLASSES # # #


# # # FUNCTIONS # # #
# ## Private Functions ## #
# ## Public functions ## #
def is_sparse_feasible(
    df: pd.DataFrame,
    dimension_cols: Collection[Any],
    error_handling: ErrorHandlingLiteral = "ignore",
) -> bool:
    """Check whether a sparse array is more efficient than a dense one

    Parameters
    ----------
    df:
        The potential dataframe to convert.

    dimension_cols:
        The columns of `df` that define the dimensions (the non-value columns).

    error_handling:
        How to handle errors if a sparse matrix is not a feasible option to
        reduce memory. By default, errors are ignored and a bool is returned.
        Can be set to either print, raise a warning, or raise an error when
        not feasible.

    Returns
    -------
    boolean:
        True if memory would be saved by converting to a sparse matrix rather
        than a dense one. Otherwise, False.

    Raises
    ------
    RuntimeError:
        If a sparse matrix is not a feasible option and
        `error_handling="raise"`.

    """
    # Init
    dimensions = [len(df[x].unique()) for x in dimension_cols]
    n_max_combinations = functools.reduce(operator.mul, dimensions, 1)
    n_dims = len(dimensions)

    # Calculate feasibility
    utilisation_threshold = 1 / (n_dims + 1)
    utilisation = len(df) / n_max_combinations
    if utilisation >= utilisation_threshold:
        msg = (
            "Utilisation is higher than the threshold at which sparse "
            "matrices are ineffective. The threshold of non-sparse values is "
            f"{utilisation_threshold * 100}% for a {n_dims}-dimensional "
            "array. Utilisation of the given array is "
            f"{utilisation * 100:.3f}%."
        )
        internal_utils.handle_error(msg, how=error_handling, error=RuntimeError)
        return False
    return True


def dataframe_to_n_dimensional_sparse_array(
    df: pd.DataFrame,
    dimension_cols: dict[str, list[Any]],
    value_col: Any,
    raise_error: bool = False,
    fill_value: Union[np.number, int, float] = 0,
) -> sparse.COO:
    """Convert a pandas.DataFrame to a sparse.COO matrix"""
    # Init
    final_shape = [len(x) for x in dimension_cols.values()]

    # Tidy and validate given DF
    mask = df[value_col] == fill_value
    df = df[~mask].copy()

    is_sparse_feasible(
        df=df,
        dimension_cols=dimension_cols.keys(),
        error_handling="raise" if raise_error else "warn",
    )

    # ## CONVERT TO SPARSE ## #
    # Build coordinate maps for values
    value_maps = dict()
    for col, vals in dimension_cols.items():
        if np.min(vals) == 0 and np.max(vals) == len(vals) - 1:
            continue
        value_maps[col] = dict(zip(vals, range(len(vals))))

    # Map each value to its coordinates
    for col, value_map in value_maps.items():
        df[col] = df[col].map(value_map)

    return sparse.COO(
        coords=np.array([df[col].values for col in dimension_cols.keys()]),
        data=np.array(df[value_col].values),
        shape=final_shape,
    )


@overload
def dataframe_to_n_dimensional_array(
    df: pd.DataFrame,
    dimension_cols: Union[list[str], dict[str, list[Any]]],
    sparse_ok: Literal[True],
    fill_val: Any = ...,
) -> Union[np.ndarray, sparse.COO]:
    ...


@overload
def dataframe_to_n_dimensional_array(
    df: pd.DataFrame,
    dimension_cols: Union[list[str], dict[str, list[Any]]],
    sparse_ok: Literal[False],
    fill_val: Any = ...,
) -> np.ndarray:
    ...


@overload
def dataframe_to_n_dimensional_array(
    df: pd.DataFrame,
    dimension_cols: Union[list[str], dict[str, list[Any]]],
    sparse_ok: Literal["force"],
    fill_val: Any = ...,
) -> sparse.COO:
    ...


@overload
def dataframe_to_n_dimensional_array(
    df: pd.DataFrame,
    dimension_cols: Union[list[str], dict[str, list[Any]]],
    sparse_ok: bool = ...,
    fill_val: Any = ...,
) -> np.ndarray:
    ...


def dataframe_to_n_dimensional_array(
    df: pd.DataFrame,
    dimension_cols: Union[list[str], dict[str, list[Any]]],
    sparse_ok: Union[bool, Literal["force"]] = False,
    fill_val: Any = np.nan,
) -> Union[np.ndarray, sparse.COO]:
    """Convert a pandas.DataFrame into an N-Dimensional numpy array.

    Each column listed in `dimension_cols` will be another dimension in the
    final array. E.g. if `dimension_cols` was a list of 4 items then a
    4D numpy array would be returned.

    Parameters
    ----------
    df:
        The pandas.DataFrame to convert.

    dimension_cols:
        Either a list of the columns to convert to dimensions, or a dictionary
        mapping the columns to convert to a list of the unique values in each
        column. If a list is provided than a dictionary is inferred from the
        unique values in each column in `df`.
        The resultant dimensions will be in order of `dimension_cols` if a
        list is provided, otherwise `dimension_cols.keys()`.

    fill_val:
        The value to use when filling any missing combinations of a product
        of all the `dimension_col` values.

    sparse_ok:
        Whether it is OK to return a sparse.COO matrix or not. If True then
        a sparse.COO matrix will be returned only if a MemoryError is caught
        while creating the matrix. If "force" then the return will always be
        a sparse.COO matrix.

    Returns
    -------
    ndarray:
        A N-dimensional numpy array made from `df`.
    """
    # Init
    if not isinstance(dimension_cols, dict):
        dimension_cols = {x: df[x].unique().tolist() for x in dimension_cols}
    final_shape = [len(x) for x in dimension_cols.values()]

    # Validate that only one value column exists
    value_cols = set(df.columns) - set(dimension_cols.keys())
    if len(value_cols) > 1:
        raise ValueError(
            "More than one value column found. Cannot convert to N-Dimensional "
            "array. The following columns have not been accounted for in the "
            f"`dimension_cols`:\n{value_cols}"
        )
    value_col = value_cols.pop()

    # Validate sparse_ok value
    if not isinstance(sparse_ok, bool) and sparse_ok != "force":
        raise ValueError(
            "Invalid value received for `sparse_ok`. Expected either a bool or "
            f"the string 'forced'. Got {sparse_ok} instead."
        )

    # ## CONVERT ## #
    # Just make sparse if we're forcing
    if sparse_ok == "force":
        return dataframe_to_n_dimensional_sparse_array(
            df=df,
            dimension_cols=dimension_cols,
            value_col=value_col,
            raise_error=False,
            fill_value=0,
        )
    assert isinstance(sparse_ok, bool)

    # Try make a dense matrix
    try:
        full_idx = df_handling.get_full_index(dimension_cols)
        np_df = df.set_index(list(dimension_cols.keys())).reindex(full_idx).fillna(fill_val)
        return np_df.values.reshape(final_shape)

    except MemoryError as err:
        if not sparse_ok:
            raise MemoryError(
                "Memory error while attempting to create a dense numpy matrix. "
                "This could be translated into a sparse matrix if 'sparse_ok=True'"
            ) from err

    # We ran out of memory making a dense matrix, and it's OK to make a sparse one
    return dataframe_to_n_dimensional_sparse_array(
        df=df,
        dimension_cols=dimension_cols,
        value_col=value_col,
        raise_error=True,
        fill_value=0,
    )


def n_dimensional_array_to_dataframe(
    mat: np.ndarray,
    dimension_cols: dict[str, list[Any]],
    value_col: str,
    drop_zeros: bool = False,
) -> pd.DataFrame:
    """Convert an N-dimensional numpy array to a pandas.Dataframe.

    Parameters
    ----------
    mat:
        The N-dimensional array to convert.

    dimension_cols:
        A dictionary of `{col_name: col_values}` pairs. `dimension_cols.keys()`
        MUST return a list of keys in the same order as the dimension that each
        `col_name` refers to. `dimension_cols.keys()` is defined by the order
        the keys are added to a dictionary. `col_values` MUST be in the same
        order as the values in the dimension they refer to.

    value_col:
        The name to give to the value columns in the output dataframe.

    drop_zeros:
        Whether to drop any rows in the final dataframe where the value is
        0. If False then a full product of all `dimension_cols` is
        returned as the index.

    Returns
    -------
    dataframe:
        A pandas.Dataframe of `mat` with the attached index defined by
        `dimension_cols`.

    Examples
    --------
    # TODO(BT): Add examples to this one. It's a bit confusing in abstract!
    """
    full_idx = df_handling.get_full_index(dimension_cols)
    df = pd.DataFrame(
        data=mat.flatten(),
        index=full_idx,
        columns=[value_col],
    )
    if not drop_zeros:
        return df

    # Drop any rows where the value is 0
    zero_mask = df[value_col] == 0
    return df[~zero_mask].copy()
