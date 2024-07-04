# -*- coding: utf-8 -*-
"""Basic utility functions for pandas objects."""
from __future__ import annotations

# Built-Ins
import typing
from typing import Literal, Sequence, overload

# Third Party
import numpy as np
import pandas as pd

# # # CONSTANTS # # #

# # # CLASSES # # #


# # # FUNCTIONS # # #
@typing.no_type_check
def cast_to_common_type(
    items_to_cast: Sequence[pd.Series],
) -> list[pd.Series]:
    """Cast N objects to the same datatype.

    The passed in objects must have the `dtype` attribute, and a call to
    `astype(new_type)` must return a copy of the object as `new_type`.
    Most, if not all, pandas objects meet the criteria.

    `np.result_type()` is used internally to find a common datatype.

    Parameters
    ----------
    items_to_cast:
        The items to cast to a common dtype.

    Returns
    -------
    cast_items:
        All of the items passed in, cast to a common datatype
    """
    # TODO(BT): Figure out how to do these type hints properly
    # Simple case
    base_dtype = items_to_cast[0].dtype
    if all(x.dtype == base_dtype for x in items_to_cast):
        return list(items_to_cast)

    # Try to convert objects to numeric types. To be here, some types are
    # already numeric, pandas doesn't cope well if you try to convert
    # integers to strings.
    return_items = list()
    for itm in items_to_cast:
        if itm.dtype == "object":
            return_items.append(pd.to_numeric(itm))
        else:
            return_items.append(itm)

    common_dtype = np.result_type(*return_items)
    return [x.astype(common_dtype) for x in return_items]


@overload
def to_numeric(
    arg: np.ndarray,
    errors: Literal["ignore", "raise", "coerce"] = "raise",
    downcast: Literal["integer", "signed", "unsigned", "float"] | None = None,
    **kwargs,
) -> np.ndarray: ...


@overload
def to_numeric(
    arg: pd.Index,
    errors: Literal["ignore", "raise", "coerce"] = "raise",
    downcast: Literal["integer", "signed", "unsigned", "float"] | None = None,
    **kwargs,
) -> pd.Index: ...


@overload
def to_numeric(
    arg: pd.Series,
    errors: Literal["ignore", "raise", "coerce"] = "raise",
    downcast: Literal["integer", "signed", "unsigned", "float"] | None = None,
    **kwargs,
) -> pd.Series: ...


def to_numeric(
    arg: pd.Series | pd.Index | np.ndarray,
    errors: Literal["ignore", "raise", "coerce"] = "raise",
    downcast: Literal["integer", "signed", "unsigned", "float"] | None = None,
    **kwargs,
) -> pd.Series | pd.Index | np.ndarray:
    """Convert argument to numeric type.

    Wraps `pandas.to_numeric` and adds option to ignore errors.

    Parameters
    ----------
    arg : 1-d array, Series or Index
        Argument to be converted.
    errors : {'ignore', 'raise','coerce'}, default 'raise'
        - If 'raise', then invalid parsing will raise an exception.
        - If 'coerce', then invalid parsing will be set as NaN.
        - If 'ignore', then invalid parsing will return the input.
    downcast : str, default None
        Can be 'integer', 'signed', 'unsigned', or 'float'. If not None,
        and if the data has been successfully cast to a numerical dtype
        (or if the data was numeric to begin with), downcast that
        resulting data to the smallest numerical dtype possible.

    Returns
    -------
    pd.Series | pd.Index | np.ndarray
        Numeric if parsing succeeded. Return type depends on input.
        Series if Series, Index if Index, otherwise ndarray.

    See Also
    --------
    pd.to_numeric
    """
    if errors != "ignore":
        return pd.to_numeric(arg, errors=errors, downcast=downcast, **kwargs)  # type: ignore[arg-type]

    try:
        return pd.to_numeric(arg, downcast=downcast, **kwargs)  # type: ignore[arg-type]
    except ValueError:
        if isinstance(arg, (pd.Series, pd.Index)):
            return arg

        return np.array(arg)
