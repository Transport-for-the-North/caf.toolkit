# -*- coding: utf-8 -*-
"""Tools to convert numpy/pandas vectors/matrices between different index systems.

In transport, these tools are very useful for translating data between different
zoning systems.
"""
from __future__ import annotations

# Built-Ins
import logging
import warnings

from typing import Any
from typing import TypeVar
from typing import Optional

# Third Party
import numpy as np
import pandas as pd

# Local Imports
# pylint: disable=import-error,wrong-import-position
from caf.toolkit import validators
from caf.toolkit import math_utils
from caf.toolkit import concurrency
from caf.toolkit import pandas_utils as pd_utils

# pylint: enable=import-error,wrong-import-position

# # # CONSTANTS # # #
_T = TypeVar("_T")

LOG = logging.getLogger(__name__)

# # # CLASSES # # #


# # # FUNCTIONS # # #
# ## PRIVATE FUNCTIONS ## #
def _lower_memory_row_translation(
    vector_chunk: np.ndarray,
    row_translation: np.ndarray,
) -> np.ndarray:
    """Run internal functionality of _lower_memory_matrix_zone_translation()."""
    # Translate the rows
    row_translated = list()
    for vector in np.hsplit(vector_chunk, vector_chunk.shape[1]):
        vector = vector.flatten()
        vector = np.broadcast_to(np.expand_dims(vector, axis=1), row_translation.shape)
        vector = vector * row_translation
        vector = vector.sum(axis=0)
        row_translated.append(vector)

    return np.array(row_translated)


def _lower_memory_col_translation(
    vector_chunk: np.ndarray,
    col_translation: np.ndarray,
) -> np.ndarray:
    """Run internal functionality of _lower_memory_matrix_zone_translation()."""
    # Translate the columns
    col_translated = list()
    for vector in np.vsplit(vector_chunk, vector_chunk.shape[0]):
        vector = vector.flatten()
        vector = np.broadcast_to(np.expand_dims(vector, axis=1), col_translation.shape)
        vector = vector * col_translation
        vector = vector.sum(axis=0)
        col_translated.append(vector)

    return np.array(col_translated)


def _lower_memory_matrix_zone_translation(
    matrix: np.ndarray,
    row_translation: np.ndarray,
    col_translation: np.ndarray,
    chunk_size: int,
    process_count: int = -2,
) -> np.ndarray:
    # TODO(BT): Can numba be used here to speed this
    #  up instead of multiprocessing??
    # Translate the rows
    kwargs = list()
    n_splits = matrix.shape[1] / chunk_size
    n_splits = 1 if n_splits <= 0 else n_splits
    for vector_chunk in np.array_split(matrix, n_splits, axis=1):  # type: ignore
        kwargs.append(
            {
                "vector_chunk": vector_chunk,
                "row_translation": row_translation,
            }
        )

    row_translated = concurrency.multiprocess(
        fn=_lower_memory_row_translation,
        kwarg_list=kwargs,
        process_count=process_count,
        in_order=True,
        pbar_kwargs={"desc": "Translating rows"},
    )

    row_translated = np.vstack(row_translated).T

    # Translate the rows
    kwargs = list()
    n_splits = row_translated.shape[0] / chunk_size
    n_splits = 1 if n_splits <= 0 else n_splits
    for vector_chunk in np.array_split(row_translated, n_splits, axis=0):  # type: ignore
        kwargs.append(
            {
                "vector_chunk": vector_chunk,
                "col_translation": col_translation,
            }
        )

    full_translated = concurrency.multiprocess(
        fn=_lower_memory_col_translation,
        kwarg_list=kwargs,
        process_count=process_count,
        in_order=True,
        pbar_kwargs={"desc": "Translating columns"},
    )

    return np.vstack(full_translated)


def _check_matrix_translation_shapes(
    matrix: np.ndarray,
    row_translation: np.ndarray,
    col_translation: np.ndarray,
) -> None:
    # Check matrix is square
    mat_rows, mat_columns = matrix.shape
    if mat_rows != mat_columns:
        raise ValueError(
            f"The given matrix is not square. Matrix needs to be square "
            f"for the numpy zone translations to work.\n"
            f"Given matrix shape: {str(matrix.shape)}"
        )

    # Check translations are the same shape
    if row_translation.shape != col_translation.shape:
        raise ValueError(
            f"Row and column translations are not the same shape. Both "
            f"need to be (n_in, n_out) shape for numpy zone translations "
            f"to work.\n"
            f"Row shape: {row_translation.shape}\n"
            f"Column shape: {col_translation.shape}"
        )

    # Check translation has the right number of rows
    n_zones_in, _ = row_translation.shape
    if n_zones_in != mat_rows:
        raise ValueError(
            f"Translation rows needs to match matrix rows for the "
            f"numpy zone translations to work.\n"
            f"Given matrix shape: {matrix.shape}\n"
            f"Given translation shape: {row_translation.shape}"
        )


# TODO(BT): Move to numpy_utils??
#  Would mean making array_utils sparse specific
def _convert_dtypes(
    arr: np.ndarray,
    to_type: np.dtype,
    arr_name: str = "arr",
) -> np.ndarray:
    """Convert a numpy array to a different datatype."""
    # Shortcut if already matching
    if to_type == arr.dtype:
        return arr

    # Make sure we're not going to introduce infs...
    mat_max = np.max(arr)
    mat_min = np.min(arr)

    dtype_max: np.floating | int
    dtype_min: np.floating | int
    if np.issubdtype(to_type, np.floating):
        dtype_max = np.finfo(to_type).max
        dtype_min = np.finfo(to_type).min
    elif np.issubdtype(to_type, np.integer):
        dtype_max = np.iinfo(to_type).max
        dtype_min = np.iinfo(to_type).min
    else:
        raise ValueError(f"Don't know how to get min/max info for datatype: {to_type}")

    if mat_max > dtype_max:
        raise ValueError(
            f"The maximum value of {to_type} cannot handle the maximum value "
            f"found in {arr_name}.\n"
            f"Maximum dtype value: {dtype_max}\n"
            f"Maximum {arr_name} value: {mat_max}"
        )

    if mat_min < dtype_min:
        raise ValueError(
            f"The minimum value of {to_type} cannot handle the minimum value "
            f"found in {arr_name}.\n"
            f"Minimum dtype value: {dtype_max}\n"
            f"Minimum {arr_name} value: {mat_max}"
        )

    return arr.astype(to_type)


# ## PUBLIC FUNCTIONS ## #
def numpy_matrix_zone_translation(
    matrix: np.ndarray,
    translation: np.ndarray,
    col_translation: Optional[np.ndarray] = None,
    translation_dtype: Optional[np.dtype] = None,
    check_shapes: bool = True,
    check_totals: bool = True,
    slow_fallback: bool = True,
    chunk_size: int = 100,
    _force_slow: bool = False,
) -> np.ndarray:
    """Efficiently translates a matrix between index systems.

    Uses the given translation matrices to translate a matrix of values
    from one index system to another. This has been written in pure numpy
    operations.
    NOTE:
    The algorithm optimises for speed by expanding the translation across
    3 dimensions. For large matrices this can result in `MemoryError`. In
    these cases the algorithm will fall back to a slower, more memory
    efficient algorithm when `slow_fallback` is `True`. `translation_dtype`
    can be set to a smaller data type, sacrificing accuracy for speed.

    Parameters
    ----------
    matrix:
        The matrix to translate. Must be square.
        e.g. (n_in, n_in)

    translation:
        A matrix defining the weights to use to translate.
        Should be of shape (n_in, n_out), where the output
        matrix shape will be (n_out, n_out). A value of `0.5` in
        `translation[0, 2]` Would mean that
        50% of the value in index 0 of `vector` should end up in index 2 of
        the output.
        When `col_translation` is None, this defines the translation to use
        for both the rows and columns. When `col_translation` is set, this
        defines the translation to use for the rows.

    col_translation:
        A matrix defining the weights to use to translate the columns.
        Takes an input of the same format as `translation`. When None,
        `translation` is used as the column translation.

    translation_dtype:
        The numpy datatype to use to do the translation. If None, then the
        dtype of the matrix is used. Where such high precision
        isn't needed, a more memory and time efficient data type can be used.

    check_shapes:
        Whether to check that the input and translation shapes look correct.
        Optionally set to `False` if checks have been done externally to speed
        up runtime.

    check_totals:
        Whether to check that the input and output matrices sum to the same
        total.

    slow_fallback:
        Whether to fall back to a slower, more memory efficient method if a
        `MemoryError` is raised during faster translation.

    chunk_size:
        The size of the chunks to use if falling back on a slower, more memory
        efficient method. Most of the time this can be ignored unless
        translating between two massive zoning systems.

    Returns
    -------
    translated_matrix:
        matrix, translated into (n_out, n_out) shape via translation.

    Raises
    ------
    ValueError:
        Will raise an error if matrix is not a square array, or if translation
        does not have the same number of rows as matrix.
    """
    # pylint: disable=too-many-locals
    # Init
    row_translation = translation
    if col_translation is None:
        col_translation = translation.copy()

    slow_fallback = True if _force_slow else slow_fallback

    # ## OPTIONALLY CHECK INPUT SHAPES ## #
    if check_shapes:
        _check_matrix_translation_shapes(
            matrix=matrix,
            row_translation=row_translation,
            col_translation=col_translation,
        )

    # ## CONVERT DTYPES ## #
    if translation_dtype is None:
        dtypes = [matrix.dtype, row_translation.dtype, col_translation.dtype]
        translation_dtype = np.find_common_type(dtypes, [])
    matrix = _convert_dtypes(
        arr=matrix,
        to_type=translation_dtype,
        arr_name="matrix",
    )
    row_translation = _convert_dtypes(
        arr=row_translation,
        to_type=translation_dtype,
        arr_name="row_translation",
    )
    col_translation = _convert_dtypes(
        arr=col_translation,
        to_type=translation_dtype,
        arr_name="col_translation",
    )

    # ## DO THE TRANSLATION ## #
    not_enough_memory = False
    translated_matrix = None
    n_in, n_out = row_translation.shape

    # If matrix is big enough, we might run out of memory
    try:
        # Force except
        if _force_slow:
            raise MemoryError

        # Translate rows
        mult_shape = (n_in, n_in, n_out)
        expanded_rows = np.broadcast_to(np.expand_dims(matrix, axis=2), mult_shape)
        row_trans = np.broadcast_to(np.expand_dims(row_translation, axis=1), mult_shape)
        temp = expanded_rows * row_trans

        # mat is transposed, but we need it this way
        rows_done = temp.sum(axis=0)

        # Translate cols
        mult_shape = (n_in, n_out, n_out)
        expanded_cols = np.broadcast_to(np.expand_dims(rows_done, axis=2), mult_shape)
        col_trans = np.broadcast_to(np.expand_dims(col_translation, axis=1), mult_shape)
        temp = expanded_cols * col_trans
        translated_matrix = temp.sum(axis=0)

    except MemoryError as err:
        if not slow_fallback:
            raise err
        not_enough_memory = True

    # Try translation again, a slower way.
    if not_enough_memory:
        warnings.warn(
            "Ran out of memory during translation. Falling back to a slower, "
            "more memory efficient method.",
            category=RuntimeWarning,
        )
        translated_matrix = _lower_memory_matrix_zone_translation(
            matrix=matrix,
            row_translation=row_translation,
            col_translation=col_translation,
            chunk_size=chunk_size,
        )

    # Ensure value has been set for MyPy
    assert translated_matrix is not None

    if not check_totals:
        return translated_matrix

    if not math_utils.is_almost_equal(matrix.sum(), translated_matrix.sum()):
        raise ValueError(
            f"Some values seem to have been dropped during the translation. "
            f"Check the given translation matrix isn't unintentionally "
            f"dropping values. If the difference is small, it's likely a "
            f"rounding error.\n"
            f"Before: {matrix.sum()}\n"
            f"After: {translated_matrix.sum()}"
        )

    return translated_matrix


def numpy_vector_zone_translation(
    vector: np.ndarray,
    translation: np.ndarray,
    translation_dtype: Optional[np.dtype] = None,
    check_shapes: bool = True,
    check_totals: bool = True,
) -> np.ndarray:
    """Efficiently translates a vector between index systems.

    Uses the given translation matrix to translate a vector of values from one
    index system to another. This has been written in pure numpy operations.
    This algorithm optimises for speed by expanding the translation across 2
    dimensions. For large vectors this can result in `MemoryError`. If
    this happens, the `translation_dtype` needs to be set to a smaller data
    type, sacrificing accuracy.

    Parameters
    ----------
    vector:
        The vector to translate. Must be one dimensional.
        e.g. (n_in, )

    translation:
        The matrix defining the weights to use to translate matrix. Should
        be of shape (n_in, n_out), where the output vector shape will be
        (n_out, ). A value of `0.5` in `translation[0, 2]` Would mean that
        50% of the value in index 0 of `vector` should end up in index 2 of
        the output.

    translation_dtype:
        The numpy datatype to use to do the translation. If None, then the
        dtype of the vector is used. Where such high precision
        isn't needed, a more memory and time efficient data type can be used.

    check_shapes:
        Whether to check that the input and translation shapes look correct.
        Optionally set to False if checks have been done externally to speed
        up runtime.

    check_totals:
        Whether to check that the input and output vectors sum to the same
        total.

    Returns
    -------
    translated_vector:
        vector, translated into (n_out, ) shape via translation.

    Raises
    ------
    ValueError:
        Will raise an error if `vector` is not a 1d array, or if `translation`
        does not have the same number of rows as vector.
    """
    # ## OPTIONALLY CHECK INPUT SHAPES ## #
    if check_shapes:
        # Check that vector is 1D
        if len(vector.shape) > 1:
            if len(vector.shape) == 2 and vector.shape[1] == 1:
                vector = vector.flatten()
            else:
                raise ValueError(
                    f"The given vector is not a vector. Expected a np.ndarray "
                    f"with only one dimension, but got {len(vector.shape)} "
                    f"dimensions instead."
                )

        # Check translation has the right number of rows
        n_zones_in, _ = translation.shape
        if n_zones_in != len(vector):
            raise ValueError(
                f"The given translation does not have the correct number of "
                f"rows. Translation rows needs to match vector rows for the "
                f"numpy zone translations to work.\n"
                f"Given vector shape: {vector.shape}\n"
                f"Given translation shape: {translation.shape}"
            )

    # ## CONVERT DTYPES ## #
    if translation_dtype is None:
        translation_dtype = np.find_common_type([vector.dtype, translation.dtype], [])
    vector = _convert_dtypes(
        arr=vector,
        to_type=translation_dtype,
        arr_name="vector",
    )
    translation = _convert_dtypes(
        arr=translation,
        to_type=translation_dtype,
        arr_name="translation",
    )

    # ## TRANSLATE ## #
    try:
        out_vector = np.broadcast_to(np.expand_dims(vector, axis=1), translation.shape)
        out_vector = out_vector * translation
        out_vector = out_vector.sum(axis=0)
    except ValueError as err:
        if not check_shapes:
            raise ValueError(
                "'check_shapes' was set to False, was there a shape mismatch? "
                "Set 'check_shapes' to True, or see above error for more "
                "information."
            ) from err
        raise err

    if not check_totals:
        return out_vector

    if not math_utils.is_almost_equal(vector.sum(), out_vector.sum()):
        raise ValueError(
            f"Some values seem to have been dropped during the translation. "
            f"Check the given translation matrix isn't unintentionally "
            f"dropping values. If the difference is small, it's "
            f"likely a rounding error.\n"
            f"Before: {vector.sum()}\n"
            f"After: {out_vector.sum()}"
        )

    return out_vector


def pandas_matrix_zone_translation(
    matrix: pd.DataFrame,
    translation_from_col: str,
    translation_to_col: str,
    translation_factors_col: str,
    from_unique_index: list[Any],
    to_unique_index: list[Any],
    translation: pd.DataFrame,
    col_translation: pd.DataFrame = None,
    translation_dtype: Optional[np.dtype] = None,
    matrix_infill: float = 0.0,
    translate_infill: float = 0.0,
    check_totals: bool = True,
    slow_fallback: bool = True,
    chunk_size: int = 100,
    _force_slow: bool = False,
) -> pd.DataFrame:
    """Efficiently translates a pandas matrix between index systems.

    Parameters
    ----------
    matrix:
        The matrix to translate. The index and columns need to be the
        values being translated. This CANNOT be a "long" matrix.

    translation:
        A pandas DataFrame defining the weights to translate use when
        translating.
        Needs to contain columns:
        `translation_from_col`, `translation_to_col`, `translation_factors_col`.
        When `col_translation` is None, this defines the translation to use
        for both the rows and columns. When `col_translation` is set, this
        defines the translation to use for the rows.

    col_translation:
        A matrix defining the weights to use to translate the columns.
        Takes an input of the same format as `translation`. When None,
        `translation` is used as the column translation.

    translation_from_col:
        The name of the column in `translation` and `col_translation`
        containing the current index and column values of `matrix`.

    translation_to_col:
        The name of the column in `translation` and `col_translation`
        containing the desired output index and column values. This
        will define the output index and column format.

    translation_factors_col:
        The name of the column in `translation` and `col_translation`
        containing the translation weights between `translation_from_col` and
        `translation_to_col`. Where zone pairs do not exist, they will be
        infilled with `translate_infill`.

    from_unique_index:
        A list of all the unique IDs in the input indexing system.

    to_unique_index:
        A list of all the unique IDs in the output indexing system.

    translation_dtype:
        The numpy datatype to use to do the translation. If None, then the
        dtype of `vector` is used. Where such high precision
        isn't needed, a more memory and time efficient data type can be used.

    matrix_infill:
        The value to use to infill any missing matrix values.

    translate_infill:
        The value to use to infill any missing translation factors.

    check_totals:
        Whether to check that the input and output matrices sum to the same
        total.

    slow_fallback:
        Whether to fall back to a slower, more memory efficient method if a
        `MemoryError` is raised during faster translation.

    chunk_size:
        The size of the chunks to use if falling back on a slower, more memory
        efficient method. Most of the time this can be ignored unless
        translating between two massive zoning systems.

    Returns
    -------
    translated_matrix:
        matrix, translated into to_unique_index system.

    Raises
    ------
    ValueError:
        If matrix is not a square array, or if translation any inputs are not
        the correct format.
    """
    # pylint: disable=too-many-arguments, too-many-locals
    # Init
    row_translation = translation
    if col_translation is None:
        col_translation = translation.copy()

    # Make sure columns exist in the translations
    columns = [translation_from_col, translation_to_col, translation_factors_col]
    row_translation = pd_utils.reindex_cols(
        df=row_translation,
        columns=columns,
        throw_error=True,
        dataframe_name="row_translation",
    )
    col_translation = pd_utils.reindex_cols(
        df=col_translation,
        columns=columns,
        throw_error=True,
        dataframe_name="col_translation",
    )

    # Check the matrix and translation dtypes match
    if matrix.index.dtype != row_translation[translation_from_col].dtype:
        raise ValueError(
            "dtypes of `matrix.index` and `translation` in `from_zone_col` "
            "must match.\n"
            f"Matrix index data type: {matrix.index.dtype}\n"
            f"Row Translation data type: {row_translation[translation_from_col].dtype}"
        )

    if matrix.columns.dtype != col_translation[translation_from_col].dtype:
        raise ValueError(
            "dtypes of `matrix.columns` and `col_translation` in `from_zone_col` "
            "must match.\n"
            f"Matrix column Dtype: {matrix.columns.dtype}\n"
            f"Col Translation data type: {col_translation[translation_from_col].dtype}"
        )

    validators.unique_list(from_unique_index, name="from_unique_index")
    validators.unique_list(to_unique_index, name="to_unique_index")

    # Make sure the matrix only has the zones defined in from_unique_zones
    missing_rows = set(matrix.index.to_list()) - set(from_unique_index)
    if len(missing_rows) > 0:
        warnings.warn(
            f"Some zones in `matrix.index` have not been defined in "
            f"`from_unique_zones`. These zones will be dropped "
            f"before the translation!\n"
            f"Additional rows count: {len(missing_rows)}"
        )

    missing_cols = set(matrix.columns.to_list()) - set(from_unique_index)
    if len(missing_cols) > 0:
        warnings.warn(
            f"Some zones in `matrix.columns` have not been defined in "
            f"`from_unique_zones`. These zones will be dropped "
            f"before the translation!\n"
            f"Additional cols count: {len(missing_cols)}"
        )

    # ## PREP AND TRANSLATE ## #
    # Square the translation
    row_translation = pd_utils.long_to_wide_infill(
        df=row_translation,
        index_col=translation_from_col,
        columns_col=translation_to_col,
        values_col=translation_factors_col,
        index_vals=from_unique_index,
        column_vals=to_unique_index,
        infill=translate_infill,
    )

    col_translation = pd_utils.long_to_wide_infill(
        df=col_translation,
        index_col=translation_from_col,
        columns_col=translation_to_col,
        values_col=translation_factors_col,
        index_vals=from_unique_index,
        column_vals=to_unique_index,
        infill=translate_infill,
    )

    # Make sure all zones are in the matrix and infill 0s
    matrix = matrix.reindex(
        index=from_unique_index,
        columns=from_unique_index,
        fill_value=matrix_infill,
    )

    # Translate
    translated = numpy_matrix_zone_translation(
        matrix=matrix.values,
        translation=row_translation.values,
        col_translation=col_translation.values,
        translation_dtype=translation_dtype,
        check_totals=check_totals,
        slow_fallback=slow_fallback,
        chunk_size=chunk_size,
        _force_slow=_force_slow,
    )

    # Stick into pandas
    return pd.DataFrame(
        data=translated,
        index=to_unique_index,
        columns=to_unique_index,
    )


def pandas_vector_zone_translation(
    vector: pd.Series | pd.DataFrame,
    translation: pd.DataFrame,
    translation_from_col: str,
    translation_to_col: str,
    translation_factors_col: str,
    from_unique_index: list[Any],
    to_unique_index: list[Any],
    translation_dtype: Optional[np.dtype] = None,
    vector_infill: float = 0.0,
    translate_infill: float = 0.0,
    check_totals: bool = True,
) -> pd.Series:
    # pylint: disable=too-many-arguments
    """Efficiently translates a pandas vector between index systems.

    Internally, checks and converts the pandas inputs into numpy arrays
    and calls `numpy_vector_zone_translation()`. The final output is then
    converted back into a pandas Series, using the same format as the input.

    Parameters
    ----------
    vector:
        The vector to translate. The index must be the values to be translated.

    translation:
        A pandas DataFrame defining the weights to translate use when
        translating.
        Needs to contain columns:
        `translation_from_col`, `translation_to_col`, `translation_factors_col`.

    translation_from_col:
        The name of the column in `translation` containing the current index
        values of `vector`.

    translation_to_col:
        The name of the column in `translation` containing the desired output
        index values. This will define the output index format.

    translation_factors_col:
        The name of the column in `translation` containing the translation
        weights between `translation_from_col` and `translation_to_col`.
        Where zone pairs do not exist, they will be infilled with
        `translate_infill`.

    from_unique_index:
        A list of all the unique IDs in the input indexing system.

    to_unique_index:
        A list of all the unique IDs in the output indexing system.

    translation_dtype:
        The numpy datatype to use to do the translation. If None, then the
        dtype of `vector` is used. Where such high precision
        isn't needed, a more memory and time efficient data type can be used.

    vector_infill:
        The value to use to infill any missing vector values.

    translate_infill:
        The value to use to infill any missing translation factors.

    check_totals:
        Whether to check that the input and output matrices sum to the same
        total.

    Returns
    -------
    translated_vector:
        vector, translated into to_zone system.

    See Also
    --------
    .numpy_vector_zone_translation()
    """
    # If dataframe given, try coerce
    if isinstance(vector, pd.DataFrame):
        if vector.shape[1] != 1:
            raise ValueError(
                "`vector` must be a pandas.Series, or a pandas.DataFrame with "
                f"one column. Got a DataFrame of shape {vector.shape} instead"
            )
        vector = vector[vector.columns[0]]

    # Check the matrix and translation dtypes match
    if vector.index.dtype != translation[translation_from_col].dtype:
        raise ValueError(
            "dtypes of `vector.index` and `translation` in `from_zone_col` "
            "must match.\n"
            f"vector index data type: {vector.index.dtype}\n"
            f"translation[from_zone_col] data type: {translation[translation_from_col].dtype}"
        )

    validators.unique_list(from_unique_index, name="from_unique_index")
    validators.unique_list(to_unique_index, name="to_unique_index")

    # Make sure the vector only has the zones defined in from_unique_zones
    missing_rows = set(vector.index.to_list()) - set(from_unique_index)
    if len(missing_rows) > 0:
        warnings.warn(
            f"Some zones in `vector.index` have not been defined in "
            f"`from_unique_zones`. These zones will be dropped before "
            f"translating.\n"
            f"Additional rows count: {len(missing_rows)}"
        )

    # Check all needed values are in from_zone_col
    trans_from_zones = set(translation[translation_from_col].unique())
    missing_zones = set(from_unique_index) - trans_from_zones
    if len(missing_zones) != 0:
        warnings.warn(
            f"Some zones in `vector.index` are missing in `translation`. "
            f"Infilling missing zones with `translation_infill`.\n"
            f"Missing zones count: {len(missing_zones)}"
        )

    # ## PREP AND TRANSLATE ## #
    # Square the translation
    translation = pd_utils.long_to_wide_infill(
        df=translation,
        index_col=translation_from_col,
        columns_col=translation_to_col,
        values_col=translation_factors_col,
        index_vals=from_unique_index,
        column_vals=to_unique_index,
        infill=translate_infill,
    )

    # Sort vector and infill 0s
    vector = vector.reindex(
        index=from_unique_index,
        fill_value=vector_infill,
    )

    # Translate and return
    translated = numpy_vector_zone_translation(
        vector=vector.values,
        translation=translation.values,
        translation_dtype=translation_dtype,
        check_totals=check_totals,
    )
    return pd.Series(
        data=translated,
        index=to_unique_index,
        name=vector.name,
    )


# TODO(BT): Bring over from normits_demand (once we have zoning systems):
#  translate_vector_zoning
#  translate_matrix_zoning
#  get_long_translation
#  get_long_pop_emp_translations
