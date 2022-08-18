# -*- coding: utf-8 -*-
"""
Created on: 15/08/2022
Updated on:

Original author: Ben Taylor
Last update made by:
Other updates made by:

File purpose:

"""
# Built-Ins

# Third Party
import pandas as pd

# Local Imports
# pylint: disable=import-error,wrong-import-position

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
    Wrapper around `df.reindex()`. Will throw error if columns aren't in `df`

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
