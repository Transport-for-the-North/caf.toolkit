# -*- coding: utf-8 -*-
"""Common utility functions for file input and output."""
from __future__ import annotations

# Built-Ins
import logging
import os
import pathlib
import re
import time
import warnings
from typing import Literal

# Third Party
import pandas as pd

# Local Imports
from caf.toolkit.pandas_utils import utility

# # # CONSTANTS # # #
LOG = logging.getLogger(__name__)

# # # CLASSES # # #


class MissingColumnsError(Exception):
    """Raised when columns are missing from input CSV."""

    def __init__(self, name: str, columns: list[str], *args, **kwargs):
        self.columns = columns
        cols = " and".join(", ".join(f"'{s}'" for s in columns).rsplit(",", 1))
        msg = f"Columns missing from {name}: {cols}"
        super().__init__(msg, *args, **kwargs)


# # # FUNCTIONS # # #
def safe_dataframe_to_csv(
    df: pd.DataFrame,
    *args,
    **kwargs,
) -> None:
    """Prompt the user to close a file before saving.

    Wrapper around `df.to_csv()`.

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


def read_csv(path: os.PathLike, name: str | None = None, **kwargs) -> pd.DataFrame:
    """Read CSV files, wraps `pandas.read_csv` to perform additional checks.

    Provides more detailed error messages about missing columns.

    Parameters
    ----------
    path : Path
        Path to the CSV file (can be ".csv" or ".txt").
    name : str, optional
        Human readable name of the file being read (used for error
        messages), if not given uses the filename.
    kwargs : keyword arguments
        All other keyword arguments are passed to `pandas.read_csv`.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the information from the CSV.

    Raises
    ------
    MissingColumnsError
        If any columns given in `usecols` don't exist in the CSV.
    ValueError
        If any of the columns in `dtype` cannot be converted
        to the given data type.
    """
    path = pathlib.Path(path)
    if name is None:
        name = path.stem

    if not path.is_file():
        raise FileNotFoundError(f"{name} file does not exist: '{path}'")

    try:
        df = pd.read_csv(path, **kwargs)
    except ValueError as err:
        match = re.match(
            r".*columns expected but not found:\s+\[((?:'[^']+',?\s?)+)\]",
            str(err),
            re.IGNORECASE,
        )
        if match:
            missing = re.findall(r"'([^']+)'", match.group(1))
            raise MissingColumnsError(name, missing) from err

        match = re.match(r"index (\S+) invalid", str(err), re.IGNORECASE)
        if match:
            raise MissingColumnsError(name, [match.group(1)]) from err

        if isinstance(kwargs.get("dtype"), dict):
            # Check what column can't be converted to dtypes
            columns: dict[str, type] = kwargs.pop("dtype")
            df = pd.read_csv(path, **kwargs)
            for col, _type in columns.items():
                try:
                    df[col].astype(_type)
                except ValueError:
                    raise ValueError(
                        f"Column '{col}' in {name} has values "
                        f"which cannot be converted to {_type}"
                    ) from err
        raise

    return df


def read_csv_matrix(
    path: os.PathLike, format_: Literal["square", "long"] | None = None, **kwargs
) -> pd.DataFrame:
    """Read matrix CSV in the square or long format.

    Sorts the index and column names and makes sure they're
    the same, doesn't infill any NaNs created when reindexing.

    Parameters
    ----------
    path : Path
        Path to CSV file
    format_ : str, optional
        Expected format of the matrix 'square' or 'long', if
        not given attempts to figure out the format by reading
        the top few lines of the file.
    kwargs : keyword arguments
        Additional keyword arguments passed to `read_csv`.

    Returns
    -------
    pd.DataFrame
        Matrix file in square format with sorted columns and indices

    Raises
    ------
    ValueError
        If the `format_` cannot be determined by reading the file
        or an invalid `format_` is given.
    """
    path = pathlib.Path(path)

    if format_ is None:
        # Determine format by reading top few lines of file
        matrix = read_csv(path, nrows=3)

        if len(matrix.columns) == 3:
            format_ = "long"

        elif len(matrix.columns) > 3:
            format_ = "square"

        else:
            raise ValueError(f"cannot determine format of matrix {path}")

    if "index_col" in kwargs and kwargs["index_col"] is None:
        kwargs.pop("index_col")

    format_ = format_.strip().lower()
    if format_ == "square":
        matrix = read_csv(path, index_col=kwargs.pop("index_col", 0), **kwargs)

    elif format_ == "long":
        matrix = read_csv(path, index_col=kwargs.pop("index_col", [0, 1]), **kwargs)

        # Matrix has MultiIndex so this returns a DataFrame
        matrix = matrix.unstack()  # type: ignore
        matrix.columns = matrix.columns.droplevel(0)

    else:
        raise ValueError(f"unknown format {format_}")

    # Attempt to convert to integers, which should work fine for pandas Index
    matrix.columns = utility.to_numeric(matrix.columns, errors="ignore", downcast="integer")
    matrix.index = utility.to_numeric(matrix.index, errors="ignore", downcast="integer")

    matrix = matrix.sort_index(axis=0).sort_index(axis=1)
    if not matrix.index.equals(matrix.columns):
        warnings.warn(
            f"matrix file ({path.name}) doesn't contain the same "
            "index and columns, these are reindexed so all unique "
            "values from both are included",
            RuntimeWarning,
        )
        # Reindex index to match columns then columns to match index
        if len(matrix.columns) > len(matrix.index):
            matrix = matrix.reindex(matrix.columns, axis=0)
            matrix = matrix.reindex(matrix.index, axis=1)
        else:
            matrix = matrix.reindex(matrix.index, axis=1)
            matrix = matrix.reindex(matrix.columns, axis=0)

    return matrix
