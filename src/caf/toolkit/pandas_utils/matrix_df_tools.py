# -*- coding: utf-8 -*-
"""Helper functions for handling pandas DataFrames as matrices"""
# Built-Ins
import operator

from typing import Any
from typing import Iterable
from typing import Callable

# Third Party
import numpy as np
import pandas as pd

# Local Imports
# pylint: disable=import-error,wrong-import-position
from caf.toolkit import toolbox
from caf.toolkit.pandas_utils import df_handling
# pylint: enable=import-error,wrong-import-position

# # # CONSTANTS # # #

# # # CLASSES # # #


# # # FUNCTIONS # # #
def check_wide_format(df: pd.DataFrame, df_name: str = "the given DataFrame"):
    """Check if the given df is in `wide matrix DataFrame` format.

    The following checks are carried out:
    - The `df.index` and `df.columns` are checked to ensure they are:
      - The same type
      - Contain the same values
    - All the `df.values` are the same type

    Parameters
    ----------
    df:
        The DataFrame to check the format of

    df_name:
        The name to give to `df` if an error is thrown

    Raises
    ------
    ValueError:
        If any of the conditions that a wide matrix must meet have not been met.
        See long function description for detailed checks.

    See Also
    --------
    conform_wide_format: Coerce a DataFrame into `wide matrix DataFrame` format.
    """
    # Check index and cols are same length
    if len(df.index) != len(df.columns):
        raise ValueError(
            f"{df_name} is not in the expected 'wide matrix DataFrame' format. "
            f"The index and columns lengths are not the same.\n"
            f"Index length: {len(df.index)}\n"
            f"Column length: {len(df.columns)}"
        )

    # Check index and cols have same types
    if df.index.dtype != df.columns.dtype:
        # Check if they're both numeric / similar
        numeric_index = pd.api.types.is_numeric_dtype(df.index.dtype)
        numeric_cols = pd.api.types.is_numeric_dtype(df.columns.dtype)
        if not (numeric_index and numeric_cols):
            raise ValueError(
                f"{df_name} is not in the expected 'wide matrix DataFrame' format. "
                f"The index and columns are not of the same type.\n"
                f"Index type: {df.index.dtype}\n"
                f"Column type: {df.columns.dtype}"
            )

    # Check index and cols contain the same values
    if not toolbox.equal_ignore_order(df.index.to_list(), df.columns.to_list()):
        i_not_c, c_not_i = toolbox.iterable_difference(
            df.index.to_list(),
            df.columns.to_list(),
        )
        raise ValueError(
            f"{df_name} is not in the expected `wide matrix DataFrame` format. "
            f"The index and columns do not contain all of the same values."
            f"In index, but not columns: {len(i_not_c)} values\n"
            f"In columns, but not index: {len(c_not_i)} values\n"
        )

    # Check all values are same types
    if len(df.dtypes.unique()) > 1:
        raise ValueError(
            f"{df_name} is not in the expected `wide matrix DataFrame` format. "
            f"The values are not all the same type. Found "
            f"{len(df.dtypes.unique())} unique data types."
        )


def conform_wide_format(
    df: pd.DataFrame,
    ids: Iterable[Any] = None,
    fill_value: Any = np.nan,
    df_name: str = "the given DataFrame",
) -> pd.DataFrame:
    """Attempt to coerce a DataFrame into the `wide matrix DataFrame`.

    Will attempt to set index and columns to the same dtype, then set the
    index and columns to the same values.

    Parameters
    ----------
    df:
        The dataframe to conform to correct format

    ids:
        An iterable of index/columns values the final `df` should have.

    fill_value:
        The value to fill any missing cells in with should any extra need adding.

    df_name:
        The name to give to `df` if any errors need to be thrown during the
        conformation process.

    Returns
    -------
    conformed_df:
        `df` converted into the `wide matrix DataFrame` format.

    Raises
    ------
    ValueError:
        If `df.columns` cannot be cast to the same data type as `df.index`

    See Also
    --------
    check_wide_format: Check if a DataFrame is in `wide matrix DataFrame` format.
    """
    # Avoid in-place operations
    df = df.copy()

    # Try cast the index and columns to the same type
    if df.index.dtype != df.columns.dtype:
        try:
            df.columns = df.columns.astype(df.index.dtype)
        except TypeError as err:
            raise ValueError(
                f"Cannot cast {df_name} columns to the same type as the index."
            ) from err

    # Assume ids if not given
    if ids is None:
        ids = set(df.index.to_list()) | set(df.columns.to_list())

    # Set the index and columns to the same ids
    df = df_handling.reindex_rows_and_cols(
        df=df,
        index=ids,
        columns=ids,
        fill_value=fill_value,
    )

    # Double-check df conforms to expected format
    check_wide_format(df, df_name)

    return df


def get_wide_mask(
    df: pd.DataFrame,
    ids: list[Any] = None,
    col_ids: list[Any] = None,
    index_ids: list[Any] = None,
    join_fn: Callable = operator.and_,
) -> np.ndarray:
    """Generate a mask for a `wide matrix DataFrame`.

    Returned mask will be same shape as df.

    Parameters
    ----------
    df:
        The dataframe to generate the mask for

    ids:
        The ids to match to in both the columns and index. If this value
        is set it will overwrite anything passed into col_ids and
        index_ids.

    col_ids:
        The ids to match to in the columns. This value is ignored if
        ids is set.

    index_ids:
        The ids to match to in the index. This value is ignored if
        ids is set.

    join_fn:
        The function to call on the column and index masks to join them.
        By default, a bitwise and is used. See pythons builtin operator
        library for more options.

    Returns
    -------
    mask:
        A mask of true and false values. Will be the same shape as df.
    """
    # Validate input args
    if ids is None:
        if col_ids is None or index_ids is None:
            raise ValueError(
                "If `ids` is not set, both `col_ids` and `index_ids` need to be set."
            )
    else:
        col_ids = ids
        index_ids = ids

    # Try and cast to the correct types for rows/cols
    try:
        # Assume columns are strings if they are an object
        col_dtype = df.columns.dtype
        col_dtype = str if col_dtype == object else col_dtype
        col_ids = np.array(col_ids, col_dtype)
    except ValueError as exc:
        raise ValueError(
            "Cannot cast the col_ids to the required dtype to match the "
            f"dtype of the given df columns. Tried to cast to: {str(df.columns.dtype)}"
        ) from exc

    try:
        index_ids = np.array(index_ids, df.index.dtype)
    except ValueError as exc:
        raise ValueError(
            "Cannot cast the index_ids to the required dtype to match the "
            f"dtype of the given df index. Tried to cast to: {str(df.index.dtype)}"
        ) from exc

    # Create square masks for the rows and cols
    col_mask = np.broadcast_to(df.columns.isin(col_ids), df.shape)
    index_mask = np.broadcast_to(df.index.isin(index_ids), df.shape).T

    # Combine to get the full mask
    return join_fn(col_mask, index_mask)


def get_internal_mask(
    df: pd.DataFrame,
    ids: list[Any],
) -> np.ndarray:
    """
    Generates a mask for a wide matrix. Returned mask will be same shape as df

    Parameters
    ----------
    df:
        The dataframe to generate the mask for

    zones:
        A list of zone numbers that make up the internal zones

    Returns
    -------
    mask:
        A mask of true and false values. Will be the same shape as df.
    """
    return get_wide_mask(df=df, zones=zones, join_fn=operator.and_)


def get_external_mask(
    df: pd.DataFrame,
    zones: list[Any],
) -> np.ndarray:
    """
    Generates a mask for a wide matrix. Returned mask will be same shape as df

    Parameters
    ----------
    df:
        The dataframe to generate the mask for

    zones:
        A list of zone numbers that make up the external zones

    Returns
    -------
    mask:
        A mask of true and false values. Will be the same shape as df.
    """
    return get_wide_mask(df=df, zones=zones, join_fn=operator.or_)


def get_external_values(
    df: pd.DataFrame,
    zones: list[Any],
) -> pd.DataFrame:
    """Get only the external values in df

    External values contains internal-external, external-internal, and
    external-external. All values not meeting this criteria will be set
    to 0.

    Parameters
    ----------
    df:
        The dataframe to get the external values from

    zones:
        A list of zone numbers that make up the external zones

    Returns
    -------
    external_df:
        A dataframe containing only the external demand from df.
        Will be the same shape as df.
    """
    return df * get_external_mask(df, zones)


def get_internal_values(
    df: pd.DataFrame,
    zones: list[Any],
) -> pd.DataFrame:
    """Get only the internal values in df

    Internal values contains internal-internal. All values not
    meeting this criteria will be set to 0.

    Parameters
    ----------
    df:
        The dataframe to get the external values from

    zones:
        A list of zone numbers that make up the internal zones

    Returns
    -------
    internal_df:
        A dataframe containing only the internal demand from df.
        Will be the same shape as df.
    """
    return df * get_internal_mask(df, zones)

