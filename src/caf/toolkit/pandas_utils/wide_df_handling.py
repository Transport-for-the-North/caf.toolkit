# -*- coding: utf-8 -*-
"""Helper functions for handling wide pandas DataFrames, usually as demand matrices."""
# Built-Ins
import logging
import operator
import warnings
from typing import Any, Callable, Collection, Optional

# Third Party
import numpy as np
import pandas as pd
import pandas.api.types

# pylint: disable=import-error,wrong-import-position

# pylint: enable=import-error,wrong-import-position

# # # CONSTANTS # # #
LOG = logging.getLogger(__name__)

# # # CLASSES # # #


# # # FUNCTIONS # # #
def _guess_pandas_dtype(things):
    """Get the type of pandas object, avoiding object type."""
    # TODO(BT) Figure out a better way to do this. I don't like this way, but
    #  it avoids all the trouble you get into with pandas telling you that a
    #  column of integers is "object" type, but "object" type is also used
    #  to describe strings...
    if pandas.api.types.is_string_dtype(things):
        return str
    if pandas.api.types.is_numeric_dtype(things):
        return things.dtype
    raise ValueError("Can't figure out type!")


def get_wide_mask(
    df: pd.DataFrame,
    select: Optional[Collection[Any]] = None,
    col_select: Optional[Collection[Any]] = None,
    index_select: Optional[Collection[Any]] = None,
    join_fn: Callable = operator.and_,
) -> np.ndarray:
    """Generate an index/column mask for a wide Pandas matrix.

    Helper function to make selecting combinations of zones in a wide matrix
    easier. The index and column selections can be set individually using
    `col_select` and `index_select`, or set to the same value using
    `selection`.

    Parameters
    ----------
    df:
        The dataframe to generate the mask for.

    select:
        The IDs to select in both the columns and index. If this value
        is set it will overwrite anything passed into `col_select` and
        `index_select`.

    col_select:
        The IDs to select in the columns. This value is ignored if
        `selection` is set.

    index_select:
        The IDs to select in the index. This value is ignored if
        `selection` is set.

    join_fn:
        Individual masks are generated for the index and columns. This
        function is used to combine the two masks. By default a bitwise AND
        is used, meaning the final mask will only return `True` where both
        the index and column masks overlap. See pythons builtin operator
        library for more built-in options. Custom functions can be given.
        They must accept two numpy arrays as input and return one as output.


    Returns
    -------
    np.ndarray:
        A mask of True and False values. Will be the same shape as `df`.

    See Also
    --------
    :func:`get_wide_internal_only_mask`
    :func:`get_wide_all_external_mask`

    Examples
    --------
    Typical usage for travel demand matrices

    >>> df = pd.DataFrame(np.arange(16).reshape(4, 4))
    >>> df
        0   1   2   3
    0   0   1   2   3
    1   4   5   6   7
    2   8   9  10  11
    3  12  13  14  15

    >>> get_wide_mask(df,select=[0, 1])
    array([[ True,  True, False, False],
           [ True,  True, False, False],
           [False, False, False, False],
           [False, False, False, False]])

    It's possible to select differently for the index and columns

    >>> get_wide_mask(df,col_select=[0, 1],index_select=[1, 2, 3])
    array([[False, False, False, False],
           [ True,  True, False, False],
           [ True,  True, False, False],
           [ True,  True, False, False]])

    The operator for joining the column and index selections can also be changed

    >>> get_wide_mask(df,select=[0, 1],join_fn=operator.or_)
    array([[ True,  True,  True,  True],
           [ True,  True,  True,  True],
           [ True,  True, False, False],
           [ True,  True, False, False]])
    """
    # Validate input args
    if select is None:
        if col_select is None or index_select is None:
            raise ValueError(
                "If selection is not set, both col_select and row_zones need to be set. "
                "If you're interested in this function being able to filter on just rows "
                "or columns, please register your interest by commenting on this issue: "
                "https://github.com/Transport-for-the-North/caf.toolkit/issues/131"
            )
    else:
        col_select = select
        index_select = select

    # Validate matrix shape
    if len(df.shape) != 2 or df.shape[0] != df.shape[1]:
        raise ValueError(
            f"Only square matrices with 2 dimensions are supported. Got: {df.shape}"
        )

    # Try match dtypes in rows and cols
    if df.columns.dtype != type(col_select):
        col_select = np.array(col_select, dtype=_guess_pandas_dtype(df.columns))  # type: ignore
    if df.index.dtype != type(index_select):
        index_select = np.array(index_select, dtype=_guess_pandas_dtype(df.index))  # type: ignore

    # Create square masks for the rows and cols
    col_mask = np.broadcast_to(df.columns.isin(col_select), df.shape)
    index_mask = np.broadcast_to(df.index.isin(index_select), df.shape).T

    # Warn the user if nothing has matched
    if col_mask.sum() == 0:
        warnings.warn(
            "No columns matched the given selection. Please check values and datatypes."
        )
    if index_mask.sum() == 0:
        warnings.warn(
            "No index matched the given selection. Please check values and datatypes."
        )

    # Combine to get the full mask
    return join_fn(col_mask, index_mask)


def get_wide_internal_only_mask(
    df: pd.DataFrame,
    select: list[Any],
) -> np.ndarray:
    """Generate an internal only mask for a wide matrix.

    This is a common operation in wide travel demand matrices. When a matrix
    contains both "internal" and "external" demand, this function can be used
    to generate a mask that selects the "internal to internal" area only.

    To extract values from the given df perform `df * mask`.

    Parameters
    ----------
    df:
        The Pandas DataFrame to generate the mask for.

    select:
        A list of index/column identifiers that should be selected.

    Returns
    -------
    mask:
        A mask of True and False values. Will be the same shape as `df`.

    See Also
    --------
    :func:`get_wide_mask`
    :func:`get_wide_all_external_mask`

    Examples
    --------
    Internal zones in travel demand matrices tend to be the first in the matrix

    >>> df = pd.DataFrame(np.full((4, 4), 5))
    >>> df
       0  1  2  3
    0  5  5  5  5
    1  5  5  5  5
    2  5  5  5  5
    3  5  5  5  5

    >>> mask = get_wide_internal_only_mask(df,select=[0, 1])
    >>> mask
    array([[ True,  True, False, False],
           [ True,  True, False, False],
           [False, False, False, False],
           [False, False, False, False]])

    Values can be extracted using multiplication

    >>> df * mask
       0  1  2  3
    0  5  5  0  0
    1  5  5  0  0
    2  0  0  0  0
    3  0  0  0  0
    """
    return get_wide_mask(df=df, select=select, join_fn=operator.and_)


def get_wide_all_external_mask(
    df: pd.DataFrame,
    select: list[Any],
) -> np.ndarray:
    """Generate an external only mask for a wide matrix.

    This is a common operation in wide travel demand matrices. When a matrix
    contains both "internal" and "external" demand, this function can be used
    to generate a mask that selects the "external to external",
    "external_to_internal", and "internal to external" area.

    To extract values from the given df perform `df * mask`.

    Parameters
    ----------
    df:
        The Pandas DataFrame to generate the mask for.

    select:
        A list of index/column identifiers that should be selected.

    Returns
    -------
    mask:
        A mask of True and False values. Will be the same shape as `df`.

    See Also
    --------
    :func:`get_wide_mask`
    :func:`get_wide_internal_only_mask`

    Examples
    --------
    External zones in travel demand matrices tend to be the last in the matrix

    >>> df = pd.DataFrame(np.full((4, 4), 5))
    >>> df
       0  1  2  3
    0  5  5  5  5
    1  5  5  5  5
    2  5  5  5  5
    3  5  5  5  5

    >>> mask = get_wide_all_external_mask(df,select=[2, 3])
    >>> mask
    array([[False, False,  True,  True],
           [False, False,  True,  True],
           [ True,  True,  True,  True],
           [ True,  True,  True,  True]])

    Values can be extracted using multiplication

    >>> df * mask
       0  1  2  3
    0  0  0  5  5
    1  0  0  5  5
    2  5  5  5  5
    3  5  5  5  5
    """
    return get_wide_mask(df=df, select=select, join_fn=operator.or_)


def wide_matrix_internal_external_report(
    df: pd.DataFrame,
    int_select: Collection[Any],
    ext_select: Collection[Any],
) -> pd.DataFrame:
    """Generate a matrix report of value totals internal and externally.

    Generates a 3x3 Pandas DataFrame detailing the total of values across 4
    categories:
    internal-internal, internal-external, external-internal, external-external.
    Row and columns totals are also presented.

    Parameters
    ----------
    df:
        The dataframe to generate the report.

    int_select:
        A list of the column and index identifiers to mark as "internal".

    ext_select:
        A list of the column and index identifiers to mark as "external".

    Returns
    -------
    report:
        A report of internal and external demand in df.

    Warns
    -----
    UserWarning:
        If `internal_selection` and `external_selection` do not contain all the values
        listed in `df`, OR they have overlapping values - leading to double counting.

    Examples
    --------
    >>> df = pd.DataFrame(np.arange(25).reshape(5, 5))
    >>> df
        0   1   2   3   4
    0   0   1   2   3   4
    1   5   6   7   8   9
    2  10  11  12  13  14
    3  15  16  17  18  19
    4  20  21  22  23  24

    >>> wide_matrix_internal_external_report(df,[0, 1, 2],[3, 4])
              internal  external  total
    internal      54.0      51.0  105.0
    external     111.0      84.0  195.0
    total        165.0     135.0  300.0
    """
    # Warn if overlap of internal and external selection
    if len(overlap := set(int_select) & set(ext_select)):
        warnings.warn(
            "internal_selection and external_selection having overlapping values. "
            "The produced report will contain double counting and could be "
            f"unreliable. {overlap=}"
        )

    # Warn if not all given index and column values are included in the given selection
    df_ids = set(df.index.to_list()) | set(df.columns.to_list())
    select_ids = set(int_select) | set(ext_select)
    if len(missing := df_ids - select_ids):
        warnings.warn(
            "The given selection of internal and external values do not contain "
            f"all values in the dataframe index and columns. {missing=}"
        )

    # Build the initial report
    index = pd.Index(["internal", "external"])
    report = pd.DataFrame(index=index, columns=index, data=np.zeros((len(index), len(index))))

    # Build the kwargs to iterate over
    report_kwargs = {
        ("internal", "internal"): {"index_select": int_select, "col_select": int_select},
        ("internal", "external"): {"index_select": int_select, "col_select": ext_select},
        ("external", "internal"): {"index_select": ext_select, "col_select": int_select},
        ("external", "external"): {"index_select": ext_select, "col_select": ext_select},
    }

    # Build the report from the kwargs
    for (row_idx, col_idx), kwargs in report_kwargs.items():
        # TODO(BT): There is a way around ignoring kwarg types, but it feels like
        #  a lot of faff. See here:
        #  https://stackoverflow.com/questions/66120871/incompatible-type-in-mypy-seen-using-kwargs
        mask = get_wide_mask(df=df, join_fn=operator.and_, **kwargs)
        total = (df * mask).to_numpy().sum()

        # Feel like this indexing is backwards...
        report.loc[row_idx, col_idx] = total

    # Add a total row and column
    report["total"] = report.values.sum(axis=1)
    report.loc["total"] = report.values.sum(axis=0)
    return report
