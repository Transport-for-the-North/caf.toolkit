# -*- coding: utf-8 -*-
"""Basic utility functions for pandas objects."""
# Built-Ins
from typing import Sequence

# Third Party
import numpy as np
import pandas as pd

# Local Imports
# pylint: disable=import-error,wrong-import-position

# pylint: enable=import-error,wrong-import-position


# # # CONSTANTS # # #

# # # CLASSES # # #


# # # FUNCTIONS # # #
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
