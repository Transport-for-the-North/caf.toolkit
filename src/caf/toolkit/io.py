# -*- coding: utf-8 -*-
"""
"""
# Built-Ins
import time
import logging

# Third Party
import pandas as pd

# Local Imports
# pylint: disable=import-error,wrong-import-position

# pylint: enable=import-error,wrong-import-position

# # # CONSTANTS # # #
LOG = logging.getLogger(__name__)

# # # CLASSES # # #


# # # FUNCTIONS # # #
def safe_dataframe_to_csv(
    df: pd.DataFrame,
    *args,
    **kwargs,
) -> None:
    """
    Wrapper around `df.to_csv()`. Gives the user a chance to close the open file.

    Parameters
    ----------
    df:
        pandas.DataFrame to call `to_csv()` on

    args:
        Any arguments to pass to `df.to_csv()`

    kwargs:
        Any key-word arguments to pass to `df.to_csv()`

    Returns
    -------
        None
    """
    written_to_file = False
    waiting = False
    while not written_to_file:
        try:
            df.to_csv(*args, **kwargs)
            written_to_file = True
        except PermissionError:
            if not waiting:
                out_path = kwargs.get("path_or_buf", None)
                if out_path is None:
                    out_path = args[0]
                print(
                    f"Cannot write to file at {out_path}.\n"
                    "Please ensure it is not open anywhere.\n"
                    "Waiting for permission to write...\n"
                )
                waiting = True
            time.sleep(1)
