# -*- coding: utf-8 -*-
"""Common utility functions for file input and output."""
from __future__ import annotations

# Built-Ins
import collections
import itertools
import logging
import os
import pathlib
import re
import string
import time
import warnings
from collections.abc import Callable, Hashable, Iterable, Sequence
from typing import Any, Literal, TypeVar

# Third Party
import pandas as pd

# Local Imports
from caf.toolkit.pandas_utils import utility

# # # CONSTANTS # # #
LOG = logging.getLogger(__name__)

NORMALISED_PUNTUATION = r"!#$%£&\-.<=>+^_~\(\)"
NORMALISED_CHARACTERS = string.ascii_lowercase + string.digits + NORMALISED_PUNTUATION
"""Characted allowed in normalised column names."""

PD_COMPRESSION = {".zip", ".gzip", ".bz2", ".zstd", ".csv.bz2"}
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


def read_csv(
    path: os.PathLike, name: str | None = None, normalise_column_names: bool = False, **kwargs
) -> pd.DataFrame:
    """Read CSV files, wraps `pandas.read_csv` to perform additional checks.

    Provides more detailed error messages about missing columns.

    Parameters
    ----------
    path : Path
        Path to the CSV file (can be ".csv" or ".txt").
    name : str, optional
        Human readable name of the file being read (used for error
        messages), if not given uses the filename.
    normalise_columns : bool, default False
        Replace spaces with underscores, convert to lowercase
        and remove any characters not in :const:`NORMALISE_CHARACTERS`.
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

    column_lookup = None
    if normalise_column_names:
        column_lookup, *parameters = _normalise_read_csv(path, **kwargs)
        for nm, value in zip(("usecols", "dtype", "index_col"), parameters):
            if value is not None:
                kwargs[nm] = value

    try:
        df: pd.DataFrame = pd.read_csv(path, **kwargs)
    except ValueError as exc:
        _detailed_read_error(path, name, exc, kwargs)
        raise

    if column_lookup is not None:
        df = df.rename(columns=column_lookup)
        if not all(i is None for i in df.index):
            df.index.names = [
                i if i is None else column_lookup.get(str(i), i) for i in df.index.names
            ]

    return df


def _detailed_read_error(
    path: pathlib.Path, name: str, exc: ValueError, kwargs: dict[str, Any]
):
    """Parse `read_csv` error and provide more details.

    Raises
    ------
    MissingColumnsError
        If columns / indices are missing.
    ValueError
        If column data types are incorrect.
    """
    matched = re.match(
        r".*columns expected but not found:\s+\[((?:'[^']+',?\s?)+)\]",
        str(exc),
        re.IGNORECASE,
    )
    if matched:
        missing = re.findall(r"'([^']+)'", matched.group(1))
        raise MissingColumnsError(name, missing) from exc

    matched = re.match(r"index (\S+) invalid", str(exc), re.IGNORECASE)
    if matched:
        raise MissingColumnsError(name, [matched.group(1)]) from exc

    if isinstance(kwargs.get("dtype"), dict):
        # Check what column can't be converted to dtypes
        columns: dict[str, type] = kwargs.pop("dtype")
        df: pd.DataFrame = pd.read_csv(path, **kwargs)
        for col, _type in columns.items():
            try:
                df[col].astype(_type)
            except ValueError:
                raise ValueError(
                    f"Column '{col}' in {name} has values "
                    f"which cannot be converted to {_type}"
                ) from exc


_Usecols = TypeVar("_Usecols", Sequence[Hashable], Callable)
_Dtype = TypeVar("_Dtype", type, dict[Hashable, type])
_IndexCol = TypeVar("_IndexCol", Sequence[Hashable], Hashable, Literal[False])


def _normalise_read_csv(
    path: pathlib.Path,
    usecols: _Usecols | None = None,
    dtype: _Dtype | None = None,
    index_col: _IndexCol | None = None,
    **kwargs,
) -> tuple[dict[str, str], _Usecols | None, _Dtype | None, _IndexCol | None]:
    """Produce normalised column names and lookup from original.

    Reads headers only from CSV to quickly obtain current
    column names and produce normalised lookup.

    Parameters
    ----------
    path
        Path to the CSV.
    usecols : Sequence[Hashable] | None, optional
        Optional usecols parameters for :func:`pandas.read_csv`.
    dtype : type | dict[Hashable, type] | None, optional
        Optional dtype parameters for :func:`pandas.read_csv`.
    index_col : Sequence[Hashable] | False, optional
        Optional index_col parameters for :func:`pandas.read_csv`.

    Returns
    -------
    dict[str, str]
        Lookup between the original CSV column names (keys)
        and the normalised names (values).
    Sequence[Hashable] | Callable | None
        Normalised usecols names, None if usecols isn't given.
    type | dict[Hashable, type] | None
        Normalised dtype dictionary, None if dtype isn't a dictionary.
    Hashable | Sequence[Hashable] | Literal[False] | None
        Normalised index_col sequence, None if index_col not given.
    """
    if "names" in kwargs:
        raise ValueError("cannot normalise columns when new names are passed")
    if "header" in kwargs and kwargs["header"] is None:
        raise ValueError("cannot normalise columns when header is None")

    df: pd.DataFrame = pd.read_csv(path, nrows=1, **kwargs)

    if isinstance(df.columns, pd.MultiIndex):
        raise NotImplementedError("cannot normalise columns on a MultiIndex")

    original = df.columns.to_list()
    if any(i is not None for i in df.index.names):
        original.extend(df.index.names)  # type: ignore

    lookup: dict[str, str] = {}
    duplicates = collections.defaultdict(list)
    for col in original:
        normalised = _normalise_name(col)

        if normalised in lookup.values():
            duplicates[normalised].append(col)

        lookup[col] = normalised

    if len(duplicates) > 0:
        raise ValueError(
            f"multiple columns have the same name after normalisation:\n{duplicates}"
        )

    flipped = {j: i for i, j in lookup.items()}

    # Convert usecols and dtype to un-normalised names in CSV
    if isinstance(usecols, Sequence):
        _validate_normal_columns(usecols, "usecols")
        usecols = [flipped.get(str(i), i) for i in usecols]

    if isinstance(dtype, dict):
        _validate_normal_columns(dtype.keys(), "dtype")
        dtype = {flipped.get(str(i), i): j for i, j in dtype.items()}

    if index_col is None or index_col is False:
        return lookup, usecols, dtype, index_col

    if isinstance(index_col, Hashable):
        _validate_normal_columns([index_col], "index_col")
        index_col = flipped.get(str(index_col), index_col)  # type: ignore
    elif isinstance(index_col, Sequence):
        _validate_normal_columns(index_col, "index_col")
        index_col = [flipped.get(str(i), i) for i in index_col]

    return lookup, usecols, dtype, index_col


def _validate_normal_columns(columns: Iterable[Hashable], name: str):
    def normal_check(value) -> bool:
        return isinstance(value, str) and (value == _normalise_name(value))

    invalid = list(itertools.filterfalse(normal_check, columns))
    if len(invalid) > 0:
        raise ValueError(
            f"{len(invalid)} names given for {name} aren't normalised: "
            + ", ".join(repr(i) for i in invalid)
        )


def _normalise_name(col: str) -> str:
    normalised = re.sub(r"\s*-\s*", "-", string=col.strip().lower())
    normalised = re.sub(r"\s+", "_", normalised)
    normalised = re.sub(rf"[^{NORMALISED_CHARACTERS}]", "", normalised)
    return normalised


def read_df(
    path: os.PathLike,
    index_col: int = None,
    find_similar: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """
    Reads in the dataframe at path. Decompresses the df if needed.

    Parameters
    ----------
    path:
        The full path to the dataframe to read in

    index_col:
        Will set this column as the index if reading from a compressed
        file, and the index is not already set.
        If reading from a csv, this is passed straight to pd.read_csv()

    find_similar:
        If True and the given file at path cannot be found, files with the
        same name but different extensions will be looked for and read in
        instead. Will check for: '.csv', '.pbz2'

    Returns
    -------
    df:
        The read in df at path.
    """

    # Try and find similar files if we are allowed
    if not os.path.exists(path):
        if not find_similar:
            raise FileNotFoundError(f"No such file or directory: '{path}'")
        path = find_filename(path)

    # Determine how to read in df
    if pathlib.Path(path).suffix == ".pbz2":
        df = compress.read_in(path)

        # Optionally try and set the index
        if index_col is not None and not is_index_set(df):
            df = df.set_index(list(df)[index_col])

        # Unset the index col if it is set - this is how pd.read_csv() works
        if index_col is None and df.index.name is not None:
            df = df.reset_index()

        # Make sure no column name is set - this is how pd.read_csv() works
        df.columns.name = None
        return df

    if pathlib.Path(path).suffix == ".csv":
        return pd.read_csv(path, index_col=index_col, **kwargs)

    if pathlib.Path(path).suffix in PD_COMPRESSION:
        return pd.read_csv(path, index_col=index_col, **kwargs)

    raise ValueError(
        f"Cannot determine the filetype of the given path. "
        f"Expected either '.csv' or '{consts.COMPRESSION_SUFFIX}'\n"
        f"Got path: {path}"
    )


def write_df(df: pd.DataFrame, path: os.PathLike, **kwargs) -> None:
    """
    Writes the dataframe at path. Decompresses the df if needed.

    Parameters
    ----------
    df:
        The dataframe to write to disk

    path:
        The full path to the dataframe to read in

    **kwargs:
        Any arguments to pass to the underlying write function.

    Returns
    -------
    df:
        The read in df at path.
    """
    # Init
    path = pathlib.Path(path)

    # Determine how to read in df
    if pathlib.Path(path).suffix == ".pbz2":
        compress.write_out(df, path)

    elif pathlib.Path(path).suffix == ".csv":
        df.to_csv(path, **kwargs)

    elif pathlib.Path(path).suffix in PD_COMPRESSION:
        df.to_csv(path, **kwargs)

    else:
        raise ValueError(
            f"Cannot determine the filetype of the given path. Expected "
            f"either '.csv' or '{consts.COMPRESSION_SUFFIX}'"
        )


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


def find_file_with_name(
    folder: pathlib.Path, name: str, suffixes: Sequence[str]
) -> pathlib.Path:
    """Find a file in a folder matching _any_ acceptable suffix.

    Searches the given `folder` only, i.e. not sub-folders, and finds
    the first file existing based on the order of `suffixes`. Warnings
    are output if other files are found with the given `name`.

    Parameters
    ----------
    folder
        Folder to search for file with, doesn't search within sub-folders.
    name
        Filename to search for, this should **not** include suffixes
        (file extensions, e.g. ".csv", ".txt").
    suffixes
        Allowed suffixes to find, if multiple files are found with
        acceptable `suffixes` then the one with the suffix first
        in `suffixes` is returned.

    Returns
    -------
    pathlib.Path
        First file found in list of `suffixes`.

    Raises
    ------
    FileNotFoundError
        If no file can be found with `suffixes`.

    Warns
    -----
    RuntimeWarning
        If multiple files are found with the same name but different suffixes.

    Examples
    --------
    Import built-in modules used for creating temporary directory
    with example files.

    >>> import pathlib
    >>> import tempfile

    Ignoring any runtime warnings produced, for testing purposes.

    >>> import warnings
    >>> warnings.filterwarnings("ignore", category=RuntimeWarning)

    List of files which will be created in the temporary directory for examples.

    >>> filenames = [
    ...     "test_file.csv",
    ...     "test_file.csv.bz2",
    ...     "test_file.txt",
    ...     "test_file.xlsx",
    ...     "another_file.csv",
    ...     "another_file.csv.bz2",
    ...     "another_file.txt",
    ... ]

    Setup temporary folder and create empty files (above) for examples.

    >>> tmpdir = tempfile.TemporaryDirectory()
    >>> folder = pathlib.Path(tmpdir.name)
    >>> for name in filenames:
    ...     path = folder / name
    ...     path.touch()

    Find "test_file" which is either a CSV (.csv) or compressed CSV (.csv.bz2).
    Files exist with both of the suffixes but the function will only return
    the path to the preferred one, i.e. the one which shows up first in the
    list.

    >>> find_file_with_name(folder, "test_file", [".csv", ".csv.bz2"]).name
    'test_file.csv'

    Runtime warnings are returned if any other files exist with the correct name
    but different suffixes, a different warning is output if files exist with
    suffixes in the list versus files which exist with other (ignored) suffixes.

    Finding an Excel file in the folder, there isn't a file with suffix ".xls" so
    this will return a Path object pointing to "test_file.xlsx".

    >>> find_file_with_name(folder, "test_file", [".xls", ".xlsx"]).name
    'test_file.xlsx'

    If no files can be found with any of the suffixes given then a FileNotFoundError
    is raised.

    >>> # Deleting temporary directory and example files
    >>> tmpdir.cleanup()
    """
    found: list[pathlib.Path] = []
    unexpected: list[str] = []

    for path in folder.glob(f"{name}.*"):
        # Combines multiple suffixes into one, does nothing if only one suffix exists
        suffix = "".join(path.suffixes)

        if suffix in suffixes:
            found.append(path)
        else:
            unexpected.append(suffix)

    if len(unexpected) > 0:
        warnings.warn(
            f'Found {len(unexpected)} files named "{name}" with unexpected'
            f' suffixes ({", ".join(unexpected)}), these are ignored.',
            RuntimeWarning,
        )
    if len(found) > 1:
        warnings.warn(
            f'Found {len(found)} files named "{name}" with the expected'
            " suffixes, the highest priority suffix is used.",
            RuntimeWarning,
        )

    if len(found) == 0:
        raise FileNotFoundError(f'cannot find any files named "{name}" inside "{folder}"')

    # Order found based on expected_suffixes
    found = sorted(found, key=lambda x: suffixes.index("".join(x.suffixes)))

    return found[0]


def remove_suffixes(path: pathlib.Path) -> pathlib.Path:
    """Removes all suffixes from path

    Parameters
    ----------
    path:
        The path to remove the suffixes from

    Returns
    -------
    path:
        path with all suffixes removed
    """
    # Init
    parent = path.parent
    prev = pathlib.Path(path.name)

    # Remove a suffix then check if all are removed
    while True:
        new = pathlib.Path(prev.stem)

        # No more suffixes to remove
        if new.suffix == "":
            break

        prev = new

    return parent / new


def read_matrix(
    path: os.PathLike,
    format_: Optional[str] = None,
    find_similar: bool = False,
) -> pd.DataFrame:
    """Read matrix CSV in the square or long format.

    Sorts the index and column names and makes sure they're
    the same, doesn't infill any NaNs created when reindexing.

    Parameters
    ----------
    path : os.PathLike
        Path to CSV file
    format_ : str, optional
        Expected format of the matrix 'square' or 'long', if
        not given attempts to figure out the format by reading
        the top few lines of the file.
    find_similar : bool, default False
        If True and the given file at path cannot be found, files with the
        same name but different extensions will be looked for and read in
        instead. Will check for: '.csv', '.pbz2'

    Returns
    -------
    pd.DataFrame
        Matrix file in square format with sorted columns and indices

    Raises
    ------
    ValueError
        If the `format` cannot be determined by reading the file
        or an invalid `format` is given.
    """
    header = 0
    if format_ is None:
        # Determine format by reading top few lines of file
        matrix = read_df(path, nrows=3, find_similar=find_similar)

        if len(matrix.columns) == 3:
            format_ = "long"

            # Check if columns have a header
            if matrix.columns[0].strip().lower() in ("o", "origin", "p", "productions"):
                header = 0
            else:
                header = None

        elif len(matrix.columns) > 3:
            format_ = "square"
            header = 0
        else:
            raise ValueError(f"cannot determine format of matrix {path}")

    format_ = format_.strip().lower()
    if format_ == "square":
        matrix = read_df(path, index_col=0, find_similar=find_similar, header=header)
    elif format_ == "long":
        matrix = read_df(path, index_col=[0, 1], find_similar=True, header=header)

        matrix = matrix.unstack()
        matrix.columns = matrix.columns.droplevel(0)
    else:
        raise ValueError(f"unknown format {format_}")

    # Attempt to convert to integers
    matrix.columns = pd.to_numeric(matrix.columns, errors="ignore", downcast="integer")
    matrix.index = pd.to_numeric(matrix.index, errors="ignore", downcast="integer")

    matrix = matrix.sort_index(axis=0).sort_index(axis=1)
    if (matrix.index != matrix.columns).any():
        # Reindex index to match columns then columns to match index
        matrix = matrix.reindex(matrix.columns, axis=0)
        matrix = matrix.reindex(matrix.index, axis=1)

    return matrix


def starts_with(s: str, x: str) -> bool:
    """
    Boolean test to see if string s starts with string x or not.

    Parameters
    ----------
    s:
        The string to test

    x:
        The string to search for

    Returns
    -------
    Bool:
        True if s starts with x, else False.
    """
    search_string = "^" + x
    return re.search(search_string, s) is not None


def find_filename(
    path: os.PathLike,
    alt_types: list[str] | None = None,
    return_full_path: bool = True,
) -> pathlib.Path:
    """
    Checks if the file at path exists under a different file extension.

    If path ends in a file extension, will try find that file first. If
    that doesn't exist, it will look for a compressed, or '.csv' version.

    Parameters
    ----------
    path:
        The path to the file to try and find

    alt_types:
        A list of alternate filetypes to consider. By default, will be:
        ['.pbz2', '.csv']

    return_full_path:
        If False, will only return the name of the file, and not the full path

    Returns
    -------
    path:
        The path to a matching, or closely matching (differing only on
        filetype extension) file.

    Raises
    ------
    FileNotFoundError:
        If the file cannot be found under any of the given alt_types file
        extensions.
    """
    # Init
    path = pathlib.Path(path)

    # Wrapper around return to deal with full path or not
    def return_fn(ret_path):
        if return_full_path:
            return ret_path
        return ret_path.name

    if alt_types is None:
        alt_types = [".pbz2", ".csv"] + list(PD_COMPRESSION)

    # Make sure they all start with a dot
    temp_alt_types = list()
    for ftype in alt_types:
        if not starts_with(ftype, "."):
            ftype = "." + ftype
        temp_alt_types.append(ftype)
    alt_types = temp_alt_types.copy()

    # Try to find the path as is
    if path.suffix != "":
        if os.path.exists(path):
            return return_fn(path)

    # Try to find similar paths
    attempted_paths = list()
    base_path = remove_suffixes(path)
    for ftype in alt_types:
        i_path = base_path.with_suffix(ftype)
        attempted_paths.append(i_path)
        if os.path.exists(i_path):
            return return_fn(i_path)

    # If here, no paths were found!
    raise FileNotFoundError(
        "Cannot find any similar files. Tried all of the following paths: %s"
        % str(attempted_paths)
    )


def file_exists(file_path: os.PathLike) -> bool:
    """
    Checks if a file exists at the given path.

    Parameters
    ----------
    file_path:
        path to the file to check.

    Returns
    -------
    file_exists:
        True if a file exists, else False
    """
    if not os.path.exists(file_path):
        return False

    if not os.path.isfile(file_path):
        raise IOError(
            "The given path exists, but does not point to a file. "
            "Given path: %s" % str(file_path)
        )

    return True


def check_file_exists(
    file_path: os.PathLike,
    find_similar: bool = False,
) -> None:
    """
    Checks if a file exists at the given path. Throws an error if not.

    Parameters
    ----------
    file_path:
        path to the file to check.

    find_similar:
        Whether to look for files with the same name, but a different file
        type extension. If True, this will call find_filename() using the
        default alternate file types: ['.pbz2', '.csv']

    Returns
    -------
    None
    """
    if find_similar:
        find_filename(file_path)
        return

    if not file_exists(file_path):
        raise IOError("Cannot find a path to: %s" % str(file_path))
