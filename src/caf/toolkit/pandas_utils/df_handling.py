# -*- coding: utf-8 -*-
"""Helper functions for handling pandas DataFrames."""
# Built-Ins
import functools

from typing import Any

# Third Party
import numpy as np
import pandas as pd

# Local Imports
# pylint: disable=import-error,wrong-import-position
from caf.toolkit import toolbox

# pylint: enable=import-error,wrong-import-position

# # # CONSTANTS # # #

# # # CLASSES # # #


# # # FUNCTIONS # # #
def reindex_cols(
    df: pd.DataFrame,
    columns: list[str],
    throw_error: bool = True,
    dataframe_name: str = "the given dataframe",
    **kwargs,
) -> pd.DataFrame:
    """
    Reindexes a pandas DataFrame. Will throw error if columns aren't in `df`.

    Parameters
    ----------
    df:
        The pandas.DataFrame that should be re-indexed

    columns:
        The columns to re-index `df` to.

    throw_error:
        Whether to throw an error or not if the given columns don't exist in
        `df`. If False, then operates exactly like calling `df.reindex()` directly.

    dataframe_name:
        The name to give to the dataframe in the error message being thrown.

    kwargs:
        Any extra arguments to pass into `df.reindex()`

    Returns
    -------
    re-indexed_df:
        `df`, re-indexed to only have `columns` as column names.

    Raises
    ------
    ValueError:
        If any of `columns` don't exist within `df` and `throw_error` is
        True.
    """
    # Init
    df = df.copy()

    if dataframe_name is None:
        dataframe_name = "the given dataframe"

    if throw_error:
        # Check that all columns actually exist in df
        for col in columns:
            if col not in df:
                raise ValueError(
                    f"No columns named '{col}' in {dataframe_name}.\n"
                    f"Only found the following columns: {list(df)}"
                )

    return df.reindex(columns=columns, **kwargs)


def reindex_rows_and_cols(
    df: pd.DataFrame,
    index: list[Any],
    columns: list[Any],
    fill_value: Any = np.nan,
    **kwargs,
) -> pd.DataFrame:
    """
    Reindex a pandas DataFrame, making sure index/col types don't clash.

    Wrapper around `df.reindex()`.
    If the type of the index of the column names in df does not match the
    types given in index or columns, the index types will be cast to the
    desired types before calling the reindex.

    Parameters
    ----------
    df:
        The pandas.DataFrame that should be re-indexed

    index:
        The index to reindex df to.

    columns:
        The columns to reindex df to.

    fill_value:
        Value to use for missing values. Defaults to NaN, but can be
        any “compatible” value.

    kwargs:
        Any extra arguments to pass into df.reindex

    Returns
    -------
    reindexed_df:
        The given df, re-indexed to the index and columns given.
    """
    # Init
    idx_dtype = type(index[0])
    col_dtype = type(columns[0])

    # Cast types if needed
    if df.index.dtype != idx_dtype:
        df.index = df.index.astype(idx_dtype)

    if df.columns.dtype != col_dtype:
        df.columns = df.columns.astype(idx_dtype)

    return df.reindex(columns=columns, index=index, fill_value=fill_value, **kwargs)


def reindex_and_groupby(
    df: pd.DataFrame,
    index_cols: list[str],
    value_cols: list[str],
    throw_error: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """
    Reindexes and groups a pandas DataFrame.

    Wrapper around `df.reindex()` and `df.groupby()`.
    Optionally throws an error if `index_cols` aren't in `df`. Will throw an
    error by default

    Parameters
    ----------
    df:
        The pandas.DataFrame that should be reindexed and grouped.

    index_cols:
        List of column names to reindex to.

    value_cols:
        List of column names that contain values. `df.groupby()` will be performed
        on any columns that are in `value_cols`, but not `index_cols`.

    throw_error:
        Whether to throw an error if not all `index_cols` are in the `df`.

    Returns
    -------
    new_df:
        A copy of `df` that has been reindexed and grouped.
    """
    # ## VALIDATE INPUTS ## #
    if throw_error:
        for col in index_cols:
            if col not in df:
                raise ValueError(
                    f"No columns named '{col}' in the given dataframe.\n"
                    f"Only found the following columns: {list(df)}"
                )

    for col in value_cols:
        if col not in index_cols:
            raise ValueError(
                f"Value '{col}' from value_cols is not in index_cols. "
                f"Can only accept value_cols that are in index_cols."
            )

    # Generate the group cols
    group_cols = toolbox.list_safe_remove(index_cols, value_cols)

    df = df.reindex(columns=index_cols, **kwargs)
    return df.groupby(group_cols).sum().reset_index()


def filter_df_mask(
    df: pd.DataFrame,
    df_filter: dict[str, Any],
) -> pd.DataFrame:
    """
    Generate a mask for filtering a pandas DataFrame by a filter.

    Parameters
    ----------
    df:
        The pandas.Dataframe to filter.

    df_filter:
        Dictionary of `{column: valid_values}` pairs to define the filter to be
        applied. `valid_values` can be a single value or a list of values.
        Will return only where all column conditions are met.

    Returns
    -------
    filter_mask:
        A mask, which when applied, will filter `df` down to `df_filter`.
    """
    # Init
    df_filter = df_filter.copy()

    # Wrap each item if a list to avoid errors
    for key, value in df_filter.items():
        if not pd.api.types.is_list_like(value):
            df_filter[key] = [value]

    needed_cols = list(df_filter.keys())
    mask = df[needed_cols].isin(df_filter).all(axis="columns")

    return mask


def filter_df(
    df: pd.DataFrame,
    df_filter: dict[str, Any],
    throw_error: bool = False,
) -> pd.DataFrame:
    """
    Filter a pandas DataFrame by a filter.

    Parameters
    ----------
    df:
        The pandas.Dataframe to filter.

    df_filter:
        Dictionary of `{column: valid_values}` pairs to define the filter to be
        applied. `valid_values` can be a single value or a list of values.
        Will return only where all column conditions are met.

    throw_error:
        Whether to throw an error if the filtered dataframe has no
        rows left

    Returns
    -------
    filtered_df:
        A copy of `df`, filtered down to `df_filter`.

    """
    # Generate and apply mask
    mask = filter_df_mask(df=df, df_filter=df_filter)
    return_df = df[mask].copy()

    if throw_error:
        if return_df.empty:
            raise ValueError(
                "An empty dataframe was returned after applying the filter. "
                "Are you sure the correct data was passed in?\n"
                f"Given filter: {df_filter}"
            )

    return return_df


def str_join_cols(
    df: pd.DataFrame,
    columns: list[str],
    separator: str = "_",
) -> pd.Series:
    """
    Equivalent to `separator.join(columns)` for all rows of pandas DataFrame.

    Joins the given columns together using separator. Returns a pandas Series
    with the return value in.

    Parameters
    ----------
    df:
        The dataframe containing the columns to join

    columns:
        The columns in df to concatenate together

    separator:
        The separator to use when joining columns together.

    Returns
    -------
    joined_column:
        a Pandas.Series containing all columns joined together using separator
    """
    # Define the accumulator function
    def reducer(accumulator, item):
        return accumulator + separator + item

    # Join the cols together
    join_cols = [df[x].astype(str) for x in columns]
    return functools.reduce(reducer, join_cols)
