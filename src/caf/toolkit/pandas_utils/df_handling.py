# -*- coding: utf-8 -*-
"""Helper functions for handling pandas DataFrames."""
# Built-Ins
import warnings
import functools

from typing import Any
from typing import Mapping
from typing import Optional
from typing import Generator

# Third Party
import numpy as np
import pandas as pd

# Local Imports
# pylint: disable=import-error,wrong-import-position
from caf.toolkit import toolbox
from caf.toolkit import math_utils

# pylint: enable=import-error,wrong-import-position

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

    for item in iterator:
        yield item


# pylint: disable=too-many-branches
def long_product_infill(
    df: pd.DataFrame,
    index_dict: Mapping[str, Optional[list[Any]]],
    infill: Any = 0,
    check_totals: bool = False,
) -> pd.DataFrame:
    """Infill columns with a complete product of one another.

    Infills missing values of df in `index_cols.keys()` columns by generating
    a new MultiIndex from a product of the values in `index_cols.values()`.
    Where a None-like values is given, all unique values are taken from `df`
    in that column.

    Parameters
    ----------
    df:
        The dataframe, in long format, to infill.

    index_dict:
        A dictionary mapping the columns of `df` to infill, and with what
        values. Where a None-like values is given, all unique values are taken
        from `df` in that column.
        i.e, `df[index_col].unique()` will be used.

    infill:
        The value to use to infill any missing cells in the return DataFrame.

    check_totals:
        Whether to check if the totals are almost equal before and after the
        conversion. Can only be performed on numeric columns.

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
    # Init
    val_cols = set(df.columns) - set(index_dict.keys())
    index_dict = dict(index_dict)

    # Get original value column totals where we can
    orig_col_totals = dict()
    for col in val_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            orig_col_totals[col] = df[col].values.sum()

    # Validate we can check totals if been told to
    if check_totals and orig_col_totals == dict():
        raise TypeError(
            "Cannot check totals when none of the value columns of df are "
            f"numeric. Implied value columns:\n{val_cols}"
        )

    # Validate the input index columns
    for col, vals in index_dict.items():
        # Initialise any missing values
        if toolbox.is_none_like(vals):
            index_dict[col] = df[col].unique()
            vals = index_dict[col]

        # Assert for MyPY
        assert vals is not None

        # Make sure we're not dropping too much.
        # Indication of problems in arguments.
        missing_idx = set(vals) - set(df[col].unique().tolist())
        if len(missing_idx) >= len(vals) * 0.9:
            warnings.warn(
                f"Almost all values given for column {col} for not exist in "
                f"df['{col}']. Are the given data types matching?\n"
                f"There are {len(missing_idx)} missing values.",
                category=UserWarning,
            )

    # Handle a single index
    if len(index_dict) == 1:
        name = list(index_dict.keys())[0]
        vals = index_dict[name]
        new_index = pd.Index(name=name, data=vals)
    else:
        new_index = pd.MultiIndex.from_product(index_dict.values(), names=index_dict.keys())

    # Make sure every possible combination exists
    df = df.set_index(list(index_dict.keys()))
    df = df.reindex(index=new_index, fill_value=infill).reset_index()

    # Just return if we can't check totals
    if len(orig_col_totals) <= 0:
        return df

    #  ## Let the user know if the totals aren't similar ## #
    msg = (
        "Values have been dropped when reindexing the given dataframe.\n"
        "Starting total: {orig_total}\n"
        "Ending total: {after_total}."
    )

    # Check and warn / error about each column
    for col, orig_total in orig_col_totals.items():
        after_total = df[col].values.sum()

        if not math_utils.is_almost_equal(after_total, orig_total):
            final_msg = msg.format(orig_total=orig_total, after_total=after_total)

            if not check_totals:
                warnings.warn(final_msg, category=UserWarning)
            else:
                raise ValueError(final_msg)

    return df


# pylint: enable=too-many-branches


def long_to_wide_infill(
    df: pd.DataFrame,
    index_col: str,
    columns_col: str,
    values_col: str,
    index_vals: Optional[list[Any]] = None,
    column_vals: Optional[list[Any]] = None,
    infill: Any = 0,
    check_totals: bool = False,
) -> pd.DataFrame:
    """Convert a DataFrame from long to wide format, infilling missing values.

    Parameters
    ----------
    df:
        The dataframe, in long format, to convert to wide.

    index_col:
        The column of `df` to use as the index of the wide return DataFrame

    columns_col:
        The column of `df` to use as the columns of the wide return DataFrame

    values_col:
        The column of `df` to use as the values of the wide return DataFrame

    index_vals:
        The unique values to use as the index of the wide return DataFrame.
        If left as None, `df[index_col].unique()` will be used.

    column_vals:
        The unique values to use as the columns of the wide return DataFrame.
        If left as None, `df[columns_col].unique()` will be used.

    infill:
        The value to use to infill any missing cells in the wide DataFrame.

    check_totals:
        Whether to check if the totals are almost equal before and after the
        conversion.

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
    # Init
    index_vals = df[index_col].unique() if index_vals is None else index_vals
    column_vals = df[columns_col].unique() if column_vals is None else column_vals
    df = reindex_cols(df, [index_col, columns_col, values_col])

    index_dict = {index_col: index_vals, columns_col: column_vals}
    df = long_product_infill(
        df=df, index_dict=index_dict, infill=infill, check_totals=check_totals
    )

    # Convert to wide
    df = df.pivot(
        index=index_col,
        columns=columns_col,
        values=values_col,
    )

    return df


def wide_to_long_infill(
    df: pd.DataFrame,
    index_col_1_name: str,
    index_col_2_name: str,
    value_col_name: str,
    index_col_1_vals: Optional[list[Any]] = None,
    index_col_2_vals: Optional[list[Any]] = None,
    infill: Any = 0,
    check_totals: bool = False,
) -> pd.DataFrame:
    """Convert a matrix from wide to long format, infilling missing values.

    Parameters
    ----------
    df:
        The dataframe, in wide format, to convert to long. The index of `df`
        must be the values that are to become `index_col_1_name`, and the
        columns of `df` will be melted to become `index_col_2_name`.

    index_col_1_name:
        The name to give to the column that was the index of `df`.

    index_col_2_name:
        The name to give to the column that was the column names of `df`.

    value_col_name:
        The name to give to the column that was the values of `df`.

    index_col_1_vals:
        The unique values to use as the first index of the return dataframe.
        These unique values will be combined with every combination of
        `index_col_2_vals` to create the full index.
        If left as None, the unique values of `df` Index will be used.

    index_col_2_vals:
        The unique values to use as the second index of the return dataframe.
        These unique values will be combined with every combination of
        `index_col_1_vals` to create the full index.
        If left as None, the unique values of `df` columns will be used.

    infill:
        The value to use to infill any missing cells in the return DataFrame.

    check_totals:
        Whether to check if the totals are almost equal before and after the
        conversion.

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
    # Assume the index is the first ID
    df = df.reset_index()
    df = df.rename(columns={df.columns[0]: index_col_1_name})

    # Convert to long
    df = df.melt(
        id_vars=index_col_1_name,
        var_name=index_col_2_name,
        value_name=value_col_name,
    )

    # Infill anything that's missing
    index_dict = {
        index_col_1_name: index_col_1_vals,
        index_col_2_name: index_col_2_vals,
    }
    df = long_product_infill(
        df=df, index_dict=index_dict, infill=infill, check_totals=check_totals
    )
    return df


def long_df_to_wide_ndarray(*args, **kwargs) -> pd.DataFrame:
    """Convert a DataFrame from long to wide format, infilling missing values.

    Similar to the `long_to_wide_infill()` function, but returns a numpy array
    instead.

    Parameters
    ----------
    df:
        The dataframe, in long format, to convert to a wide numpy array.

    index_col:
        The column of `df` to use as the index of the wide return DataFrame

    columns_col:
        The column of `df` to use as the columns of the wide return DataFrame

    values_col:
        The column of `df` to use as the values of the wide return DataFrame

    index_vals:
        The unique values to use as the index of the wide return DataFrame.
        If left as None, `df[index_col].unique()` will be used.

    column_vals:
        The unique values to use as the columns of the wide return DataFrame.
        If left as None, `df[columns_col].unique()` will be used.

    infill:
        The value to use to infill any missing cells in the wide DataFrame.

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
    return df.values


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
