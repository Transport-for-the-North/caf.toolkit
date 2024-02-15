# -*- coding: utf-8 -*-
"""Conversion methods between numpy and pandas formats."""
from __future__ import annotations

# Built-Ins
import functools
import logging
import operator
import warnings
from typing import Any, Collection, Literal, Optional, overload

# Third Party
import numpy as np
import pandas as pd
import sparse

# Local Imports
from caf.toolkit.core import SparseLiteral, WarningActionKind
from caf.toolkit.pandas_utils import df_handling

# # # CONSTANTS # # #
LOG = logging.getLogger(__name__)

# # # CLASSES # # #


# # # FUNCTIONS # # #
# ## Private Functions ## #
def _pd_to_np_value_maps(dimension_cols: dict[Any, list[Any]]):
    """Create a map of column values to numpy dimensions."""
    value_maps = dict()
    for col, vals in dimension_cols.items():
        try:
            if np.min(vals) == 0 and np.max(vals) == len(vals) - 1:
                continue
        except TypeError:
            pass
        value_maps[col] = dict(zip(vals, range(len(vals))))
    return value_maps


# ## Public functions ## #
def is_sparse_feasible(
    df: pd.DataFrame,
    dimension_cols: Collection[Any],
    warning_action: WarningActionKind = "default",
) -> bool:
    """Check whether a sparse array is more efficient than a dense one.

    Parameters
    ----------
    df:
        The potential dataframe to convert.

    dimension_cols:
        The columns of `df` that define the dimensions (the non-value columns).

    warning_action:
        How to handle errors if a sparse matrix is not a feasible option to
        reduce memory. This argument will be passed directly to
        warnings.filterwarnings(warning_action). See
        https://docs.python.org/3/library/warnings.html for more information.

    Returns
    -------
    boolean:
        True if memory would be saved by converting to a sparse matrix rather
        than a dense one. Otherwise, False.

    Raises
    ------
    RuntimeWarning:
        If a sparse matrix is not a feasible option and
        `warning_action="error"`.

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
            f"{utilisation_threshold * 100:.3f}% for a {n_dims}-dimensional "
            "array. Utilisation of the given array is "
            f"{utilisation * 100:.3f}%."
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(warning_action)
            warnings.warn(msg, category=ResourceWarning)
        return False
    return True


def dataframe_to_n_dimensional_sparse_array(
    df: pd.DataFrame,
    dimension_cols: dict[Any, list[Any]],
    value_col: Any,
    sparse_value_maps: Optional[dict[Any, dict[Any, int]]] = None,
    warning_action: WarningActionKind = "default",
    fill_value: np.number | int | float = 0,
) -> tuple[sparse.COO, dict[Any, dict[Any, int]]]:
    """Convert a pandas.DataFrame to a sparse.COO matrix."""
    # Init
    final_shape = [len(x) for x in dimension_cols.values()]

    # Tidy and validate given DF
    mask = df[value_col] == fill_value
    df = df[~mask].copy()
    df = df.reindex(columns=list(dimension_cols.keys()) + [value_col])

    # Reduce inputs to just the needed columns
    dim_col_names = df.columns.tolist()
    dim_col_names.remove(value_col)
    dimension_cols = {x: dimension_cols[x] for x in dim_col_names}
    if sparse_value_maps is not None:
        sparse_value_maps = {x: sparse_value_maps[x] for x in dim_col_names}

    is_sparse_feasible(
        df=df,
        dimension_cols=dimension_cols.keys(),
        warning_action=warning_action,
    )

    # ## CONVERT TO SPARSE ## #
    if sparse_value_maps is None:
        sparse_value_maps = _pd_to_np_value_maps(dimension_cols)

    # Map each value to its coordinates
    for col, value_map in sparse_value_maps.items():
        df[col] = df[col].map(value_map)

    if np.any(np.isnan(df[dim_col_names].values)):
        raise ValueError(
            "Found NaN values after mapping `df` dimension columns to coordinate "
            "positions. If `sparse_value_maps` has been given, it likely wasn't "
            "a complete mapping."
        )

    array = sparse.COO(
        coords=np.array([df[col].values for col in dimension_cols.keys()]),
        data=np.array(df[value_col].values),
        shape=final_shape,
    )
    return array, sparse_value_maps


@overload
def dataframe_to_n_dimensional_array(
    df: pd.DataFrame,
    dimension_cols: list[Any] | dict[Any, list[Any]],
    sparse_ok: Literal["allow", "feasible"],
    sparse_value_maps: Optional[dict[Any, dict[Any, int]]] = ...,
    fill_val: Any = ...,
) -> tuple[np.ndarray | sparse.COO, dict[Any, dict[Any, int]]]: ...  # pragma: no cover


@overload
def dataframe_to_n_dimensional_array(
    df: pd.DataFrame,
    dimension_cols: list[Any] | dict[Any, list[Any]],
    sparse_ok: Literal["disallow"],
    sparse_value_maps: Optional[dict[Any, dict[Any, int]]] = ...,
    fill_val: Any = ...,
) -> tuple[np.ndarray, dict[Any, dict[Any, int]]]: ...  # pragma: no cover


@overload
def dataframe_to_n_dimensional_array(
    df: pd.DataFrame,
    dimension_cols: list[Any] | dict[Any, list[Any]],
    sparse_ok: Literal["force"],
    sparse_value_maps: Optional[dict[Any, dict[Any, int]]] = ...,
    fill_val: Any = ...,
) -> tuple[sparse.COO, dict[Any, dict[Any, int]]]: ...  # pragma: no cover


@overload
def dataframe_to_n_dimensional_array(
    df: pd.DataFrame,
    dimension_cols: list[Any] | dict[Any, list[Any]],
    sparse_ok: SparseLiteral = ...,
    sparse_value_maps: Optional[dict[Any, dict[Any, int]]] = ...,
    fill_val: Any = ...,
) -> tuple[np.ndarray, dict[Any, dict[Any, int]]]: ...  # pragma: no cover


def dataframe_to_n_dimensional_array(
    df: pd.DataFrame,
    dimension_cols: list[Any] | dict[Any, list[Any]],
    sparse_ok: SparseLiteral = "disallow",
    sparse_value_maps: Optional[dict[Any, dict[Any, int]]] = None,
    fill_val: Any = np.nan,
) -> tuple[np.ndarray | sparse.COO, dict[Any, dict[Any, int]]]:
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
        Whether it is OK to return a sparse.COO matrix or not.
        - "disallow" means that a sparse matrix cannot be returned, a memory
        error will be thrown if a sparse matrix is needed.
        - "allow" means that it is OK to convert to a sparse matrix if needed,
        but a dense matrix will be returned if it will fit into memory.
        - "feasible" means that a sparse matrix will always be returned if
        less memory would be consumed by the sparse matrix.
        - "force" means that a sparse matrix will always be returned regardless
        of the memory consumption of the dense matrix.

    sparse_value_maps:
        A nested dictionary of `{col_name: {col_val: coordinate_value}}` where
        `col_name` is the name of the column in `df`, `col_val` is the
        value in `col_name`, and `coordinate_value` is the coordinate value
        to assign to that value in the sparse array.

    Returns
    -------
    ndarray:
        A N-dimensional numpy array made from `df`.

    value_maps:
        A nested dictionary of `{col_name: {col_val: coordinate_value}}` where
        `col_name` is the name of the column in `df`, `col_val` is the
        value in `col_name`, and `coordinate_value` is the coordinate value
        assigned to that value in the sparse array.
        If `sparse_value_maps` is set then this return is the same value.
    """
    # Init
    if not isinstance(dimension_cols, dict):
        dimension_cols = {x: df[x].unique().tolist() for x in dimension_cols}
    final_shape = [len(x) for x in dimension_cols.values()]

    # Validate sparse_OK value
    valid_vals = SparseLiteral.__args__  # type: ignore
    if sparse_ok not in valid_vals:
        raise ValueError(
            f"Invalid value given for 'sparse_ok' expected one of: " f"{valid_vals}"
        )

    # Validate that only one value column exists
    value_cols = set(df.columns) - set(dimension_cols.keys())
    if len(value_cols) > 1:
        raise ValueError(
            "More than one value column found. Cannot convert to N-Dimensional "
            "array. The following columns have not been accounted for in the "
            f"`dimension_cols`:\n{value_cols}"
        )
    value_col = value_cols.pop()

    # ## CONVERT ## #
    # Just make sparse if we're forcing
    if sparse_ok == "force":
        return dataframe_to_n_dimensional_sparse_array(
            df=df,
            dimension_cols=dimension_cols,
            value_col=value_col,
            sparse_value_maps=sparse_value_maps,
            warning_action="ignore",
            fill_value=0,
        )

    # Only force if it would use less memory
    if sparse_ok == "feasible":
        sparse_feasible = is_sparse_feasible(
            df=df,
            dimension_cols=dimension_cols.keys(),
            warning_action="ignore",
        )
        if sparse_feasible:
            return dataframe_to_n_dimensional_sparse_array(
                df=df,
                dimension_cols=dimension_cols,
                value_col=value_col,
                sparse_value_maps=sparse_value_maps,
                fill_value=0,
            )

    # Try make a dense matrix
    try:
        full_idx = df_handling.get_full_index(dimension_cols)
        np_df = df.set_index(list(dimension_cols.keys())).reindex(full_idx).fillna(fill_val)
        array = np_df.values.reshape(final_shape)
        if sparse_value_maps is None:
            sparse_value_maps = _pd_to_np_value_maps(dimension_cols)
        return array, sparse_value_maps

    except MemoryError as err:
        if sparse_ok == "disallow":
            raise MemoryError(
                "Memory error while attempting to create a dense numpy matrix. "
                "This could be translated into a sparse matrix if 'sparse_ok=allow'"
            ) from err

    # We ran out of memory making a dense matrix, and it's OK to make a sparse one
    return dataframe_to_n_dimensional_sparse_array(
        df=df,
        dimension_cols=dimension_cols,
        value_col=value_col,
        sparse_value_maps=sparse_value_maps,
        warning_action="error",
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
