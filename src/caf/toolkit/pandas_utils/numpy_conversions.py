# -*- coding: utf-8 -*-
"""Conversion methods between numpy and pandas formats."""
# Built-Ins
import logging

from typing import Any
from typing import Union

# Third Party
import numpy as np
import pandas as pd

# Local Imports
# pylint: disable=import-error,wrong-import-position
from caf.toolkit.pandas_utils import df_handling

# pylint: enable=import-error,wrong-import-position

# # # CONSTANTS # # #
LOG = logging.getLogger(__name__)

# # # CLASSES # # #


# # # FUNCTIONS # # #
def dataframe_to_n_dimensional_array(
    df: pd.DataFrame,
    dimension_cols: Union[list[str], dict[str, list[str]]],
    fill_val: Any = np.nan,
) -> np.ndarray:
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

    Returns
    -------
    ndarray:
        A N-dimensional numpy array made from `df`.
    """
    # Init
    if not isinstance(dimension_cols, dict):
        dimension_cols = {x: sorted(df[x].unique()) for x in dimension_cols}
    final_shape = [len(x) for x in dimension_cols.values()]

    # Validate that only one value column exists
    full_idx = df_handling.get_full_index(dimension_cols)
    value_cols = set(df.columns) - set(full_idx.names)
    if len(value_cols) > 1:
        raise ValueError(
            "More than one value column found. Cannot convert to N-Dimensional "
            "array. The following columns have not been accounted for in the "
            f"`dimension_cols`:\n{value_cols}"
        )

    # Convert
    np_df = df.set_index(list(dimension_cols.keys())).reindex(full_idx).fillna(fill_val)
    return np_df.values.reshape(final_shape)


def n_dimensional_array_to_dataframe(
    mat: np.ndarray,
    dimension_cols: dict[str, list[Any]],
    value_col: str,
    drop_zeros: bool,
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
