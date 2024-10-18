# -*- coding: utf-8 -*-
"""Helper functions for handling pandas DataFrames."""

from __future__ import annotations

# Built-Ins
import functools
from collections.abc import Hashable
from typing import Any, Generator, overload

# Third Party
import numpy as np
import pandas as pd

# Local Imports
from caf.toolkit import toolbox

# # # CONSTANTS # # #


# # # CLASSES # # #
class ChunkDf:
    """Generator to split a dataframe into chunks.

    Similar to `chunk_df()`, but validates the input arguments and
    throws and error if not valid.

    Parameters
    ----------
    df:
        the pandas.DataFrame to chunk.

    chunk_size:
        The size of the chunks to use, in terms of rows.

    Raises
    ------
    ValueError:
        If `chunk_size` is less than or equal to 0. Or if it is not and
        integer value.

    TypeError:
        If `chunk_size` is not and integer

    See Also
    --------
    `caf.toolkit.pandas_utils.chunk_df()`
    """

    def __init__(
        self,
        df: pd.DataFrame,
        chunk_size: int,
    ):
        if not isinstance(chunk_size, int):
            raise TypeError(f"chunk_size must be an integer. Given: {chunk_size}")

        if chunk_size <= 0:
            raise ValueError(
                f"Cannot generate chunk sizes of size 0 or less. Given: {chunk_size}"
            )

        self.df = df
        self.chunk_size = chunk_size
        self.range_iterator = iter(range(0, len(self.df), self.chunk_size))

    def __iter__(self):
        """Get an iterator over `self.df` chunks of size `self.chunk_size`."""
        return self

    def __next__(self) -> pd.DataFrame:
        """Get the next chunk of `self.df` of size `self.chunk_size`."""
        i = next(self.range_iterator)
        chunk_end = i + self.chunk_size
        return self.df[i:chunk_end]


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

    Type checking wrapper around `df.reindex()`.
    If the type of the index or columns of `df` does not match the
    types given in `index` or `columns`, the index types will be cast to the
    desired types before calling the reindex.

    Parameters
    ----------
    df:
        The pandas.DataFrame that should be re-indexed

    index:
        The index to reindex `df` to.

    columns:
        The columns to reindex `df` to.

    fill_value:
        Value to use for missing values. Defaults to NaN, but can be
        any “compatible” value.

    kwargs:
        Any extra arguments to pass into `df.reindex()`

    Returns
    -------
    reindexed_df:
        The given `df`, re-indexed to the `index` and `columns` given,
        including typing
    """
    # Cast dtypes if needed
    if len(index) > 0:
        idx_dtype = type(index[0])
        if not isinstance(df.index.dtype, idx_dtype):
            df.index = df.index.astype(idx_dtype)

    if len(columns) > 0:
        col_dtype = type(columns[0])
        if not isinstance(df.columns.dtype, type(columns[0])):
            df.columns = df.columns.astype(col_dtype)

    return df.reindex(columns=columns, index=index, fill_value=fill_value, **kwargs)


def reindex_and_groupby_sum(
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
        List of column names that contain values. `df.groupby()` will be
        performed on any columns that remain in `index_cols` once all
        `value_cols` have been removed.

    throw_error:
        Whether to throw an error if not all `index_cols` are in the `df`.

    Returns
    -------
    new_df:
        A copy of `df` that has been reindexed and grouped.

    Raises
    ------
    ValueError:
        If any of `index_cols` don't exist within `df` and `throw_error` is
        True.

    See Also
    --------
    `caf.toolkit.pandas_utils.df_handling.reindex_cols()`
    """
    # Validate inputs
    for col in value_cols:
        if col not in index_cols:
            raise ValueError(
                f"Value '{col}' from value_cols is not in index_cols. "
                f"Can only accept value_cols that are in index_cols."
            )

    # Reindex and groupby
    df = reindex_cols(df=df, columns=index_cols, throw_error=throw_error, **kwargs)
    group_cols = toolbox.list_safe_remove(index_cols, value_cols)
    return df.groupby(group_cols).sum().reset_index()


def filter_df_mask(
    df: pd.DataFrame,
    df_filter: dict[str, Any],
) -> pd.Series:
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


def chunk_df(
    df: pd.DataFrame,
    chunk_size: int,
) -> Generator[pd.DataFrame, None, None]:
    """Split a dataframe into chunks, usually for multiprocessing.

    NOTE: If chunk_size is not a valid value (<=0, or not a integer) the
    generator will NOT throw an exception and instead return an empty list.
    This is a result of internal python functionality. If errors need to be
    thrown, use the generator class instead: `caf.toolkit.pandas_utils.ChunkDf`

    Parameters
    ----------
    df:
        the pandas.DataFrame to chunk.

    chunk_size:
        The size of the chunks to use, in terms of rows.

    Yields
    ------
    df_chunk:
        A chunk of `df` with `chunk_size` rows

    Raises
    ------
    ValueError:
        If `chunk_size` is less than or equal to 0. Or if it is not and
        integer value.

    TypeError:
            If `chunk_size` is not and integer

    See Also
    --------
    `caf.toolkit.pandas_utils.ChunkDf`
    """
    try:
        iterator = ChunkDf(df, chunk_size)
    except (ValueError, TypeError):
        return

    yield from iterator


@overload
def long_product_infill(
    data: pd.DataFrame,
    infill: Any = 0,
    check_totals: bool = False,
    index_dict: dict[str, list] | None = None,
) -> pd.DataFrame: ...
@overload
def long_product_infill(
    data: pd.Series,
    infill: Any = 0,
    check_totals: bool = False,
    index_dict: dict[str, list] | None = None,
) -> pd.Series: ...


# pylint: disable=too-many-branches
def long_product_infill(
    data: pd.DataFrame | pd.Series,
    infill: Any = 0,
    check_totals: bool = False,
    index_dict: dict[str, list] | None = None,
) -> pd.DataFrame | pd.Series:
    """Infill columns with a complete product of one another.

    Infills missing values of df in `index_dict.keys()` columns by generating
    a new MultiIndex from a product of the values in `index_cols.values()`.
    Where a None-like value is given, all unique values are taken from `df`
    in that column.

    Parameters
    ----------
    data:
        The data, as a pandas Series or DataFrame, to infill.

    infill:
        The value to use to infill any missing cells in the return DataFrame.

    check_totals:
        Whether to check if the totals are equal before and after infill. If
        'infill' is set to anything other than zero, this must be set to False
        or an error will be raised.

    index_dict:
        Define expected values in indices. This dict will form the index of the
        infilled Series. The keys of this dict must match the names of the index
        levels in your input Series, and all values in the input Series must
        be included in this dict. If this is left as None, the infilled index
        will simply be the product of all values in the current index.

    Returns
    -------
    infilled_df:
        An extended version of 'df' with a product of all `index_cols.values()`
        in `index_cols.keys()`.

    Raises
    ------
    TypeError:
        If none of the non-index columns are numeric and `check_totals` is True
    """
    if index_dict is None:
        index_dict = {}
        for ind in data.index.names:
            vals = data.index.get_level_values(ind).unique()
            index_dict[ind] = vals.tolist()

    else:
        mismatch = [i for i in data.index.names if i not in index_dict.keys()]
        if len(mismatch) > 0:
            raise ValueError(
                f"{mismatch} levels were found in the input data, " f"but not in index_dict."
            )

    if len(data.index.names) == 1:
        full_ind = pd.Index(index_dict[data.index.name], name=data.index.name)
    else:
        full_ind = pd.MultiIndex.from_product(index_dict.values(), names=index_dict.keys())
    filler = pd.DataFrame(data=["dummy"] * len(full_ind), index=full_ind, columns=["dummy"])

    # check there's an overlap
    overlap = data.index.intersection(filler.index)
    if len(overlap) == 0:
        raise ValueError(
            "There is no intersection between the index of the "
            "input data and the full index provided."
        )

    joined = filler.join(data, how="left")
    # For this infill to be valid all data should be the same type
    if isinstance(data, pd.Series):
        selector: pd.Index[str] | Hashable | None = data.name
        dtype = data.dtype
    else:
        selector = data.columns
        if all(i == data.dtypes[0] for i in data.dtypes):
            dtype = data.dtypes[0]
        else:
            raise TypeError(
                "All columns of the input data must have the same type for this to work."
            )
    # ints can be cast to floats in join
    filled = joined.fillna(infill)[selector].astype(dtype)

    if check_totals is True:
        if isinstance(data, pd.Series):
            diff = data.sum() - filled.sum()
        elif len(data.columns) == 1:
            # If data is a dataframe then select would be a list
            # of column names so filled would be a dataframe
            assert isinstance(filled, pd.DataFrame)
            diff = data.squeeze().sum() - filled.squeeze().sum()
        else:
            diff = data.sum().sum() - filled.sum().sum()
        if diff != 0:
            raise ValueError(
                "The total has changed in infilling. If "
                "you have set infill to anything other than zero "
                f"this is likely why. Difference = {diff} (after "
                "- before."
            )

    return filled


def long_to_wide_infill(
    matrix: pd.Series,
    infill: Any = 0,
    unstack_level: str | int = -1,
    check_totals: bool = False,
    correct_cols: list | None = None,
    correct_ind: list | None = None,
) -> pd.DataFrame:
    """Convert a DataFrame from long to wide format, infilling missing values.

    Parameters
    ----------
    matrix:
        The matrix, in long format (i.e. a Series), to convert to wide.

    infill:
        The value to use to infill any missing cells in the wide DataFrame.

    unstack_level:
        The level to unstack from the index. This can either be an int, i.e. the
        ordinal level of the multiindex to unstack, or a string, i.e. the name of
        the index level to unstack. See pd.DataFrame.unstack().

    check_totals:
        Whether to check if the totals are almost equal before and after the
        conversion.

    correct_cols:
        The correct columns for the resultant dataframe. If this is provided so
        must 'correct_ind' and vice versa.

    correct_ind:
        The correct index for the resultant dataframe. If this is provided so
        must 'correct_cols' and vice versa.

    Returns
    -------
    wide_df:
        A copy of `df`, in wide format, with index_col as the index,
        columns_col as the column names, and values_col as the values.

    Raises
    ------
    TypeError:
        If none of the `values_col` is not numeric and `check_totals` is True
    """
    if not isinstance(matrix.index, pd.MultiIndex):
        raise ValueError(
            "This function expects a multiindexed Series as an "
            "input. Generally this will be two index levels, one "
            "for origin/production and another for destination/attraction."
        )
    if correct_cols is not None and correct_ind is not None:
        ind_dict = {
            matrix.index.names[0]: correct_ind,
            matrix.index.names[1]: correct_cols,
        }
        matrix = long_product_infill(matrix, infill, index_dict=ind_dict)

    elif correct_cols is not None or correct_ind is not None:
        raise ValueError(
            "cannot infill correct columns without correct"
            " index, both are required for infilling"
        )

    unstacked = matrix.unstack(level=unstack_level, fill_value=infill)
    if check_totals is True:
        diff = unstacked.sum().sum() - matrix.sum()
        if diff != 0:
            raise ValueError(
                "The matrix total has changed in translating. If "
                "you have set infill to anything other than zero "
                f"this is likely why. Difference = {diff} (after "
                f"- before."
            )
    return unstacked


def wide_to_long_infill(
    df: pd.DataFrame,
    out_name: str | None = None,
    correct_cols: list | None = None,
    correct_ind: list | None = None,
    infill: Any = None,
) -> pd.Series:
    """Convert a matrix from wide to long format, infilling missing values.

    Parameters
    ----------
    df:
        The dataframe, in wide format, to convert to long. The index of `df`
        must be the values that are to become `index_col_1_name`, and the
        columns of `df` will be melted to become `index_col_2_name`.

    out_name:
        The 'name' attribute of the resultant pandas Series. This defaults to
        'val' if not provided.

    correct_cols:
        The correct columns of the input matrix.

    correct_ind:
        The correct index of the input matrix

    infill:
        Value to infill when the matrix is unstacked.

    Returns
    -------
    long_df:
        A copy of `df`, in long format, with 3 columns:
        `[index_col_1_name, index_col_2_name, value_col_name]`

    Raises
    ------
    TypeError:
        If none of the `value_col_name` is not numeric and `check_totals` is True
    """

    if isinstance(df.index, pd.MultiIndex):
        raise ValueError(
            "This expects a single index in the input matrix, being "
            "zones and matching the columns."
        )

    stacked = df.stack(future_stack=True)
    stacked.name = "val"
    if out_name is not None:
        stacked.name = out_name
    if correct_ind is not None and correct_cols is not None:
        ind_dict = {
            stacked.index.names[0]: correct_ind,
            stacked.index.names[1]: correct_cols,
        }
        stacked = long_product_infill(stacked, infill=infill, index_dict=ind_dict)

    assert isinstance(stacked, pd.Series)
    return stacked


def long_df_to_wide_ndarray(*args, **kwargs) -> np.ndarray:
    """Convert a DataFrame from long to wide format, infilling missing values.

    Similar to the `long_to_wide_infill()` function, but returns a numpy array
    instead.

    Parameters
    ----------
     matrix:
        The matrix, in long format (i.e. a Series), to convert to wide.

    infill:
        The value to use to infill any missing cells in the wide DataFrame.

    unstack_level:
        The level to unstack from the index. This can either be an int, i.e. the
        ordinal level of the multiindex to unstack, or a string, i.e. the name of
        the index level to unstack.

    check_totals:
        Whether to check if the totals are almost equal before and after the
        conversion.

    Returns
    -------
    wide_ndarray:
        An ndarray, in wide format, with index_col as the index,
        columns_col as the column names, and values_col as the values.

    See Also
    --------
    long_to_wide_infill()
    """
    df = long_to_wide_infill(*args, **kwargs)
    return df.to_numpy()


def get_full_index(dimension_cols: dict[str, list[Any]]) -> pd.Index:
    """Create a pandas Index from a mapping of {col_name: col_values}.

    Useful for N-dimensional conversions as MultiIndex can change types
    when only one index column is needed.

    Parameters
    ----------
    dimension_cols:
        A dictionary mapping `{col_name: col_values}`, where `col_values`
        is a list of the unique values in a column.

    Returns
    -------
    index:
        A pandas index of evey combination of values in `dimension_cols`
    """
    if len(dimension_cols) > 1:
        return pd.MultiIndex.from_product(
            iterables=dimension_cols.values(),
            names=dimension_cols.keys(),
        )

    return pd.Index(
        data=list(dimension_cols.values())[0],
        name=list(dimension_cols.keys())[0],
    )
