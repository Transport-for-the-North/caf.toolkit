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
from typing import TypedDict

# Third Party
import numpy as np
import pandas as pd

# Local Imports
# pylint: disable=import-error,wrong-import-position
from caf.toolkit import validators
from caf.toolkit import math_utils
from caf.toolkit import pandas_utils as pd_utils

# pylint: enable=import-error,wrong-import-position

# # # CONSTANTS # # #
_T = TypeVar("_T")

LOG = logging.getLogger(__name__)


# # # CLASSES # # #
class _MultiVectorKwargs(TypedDict):
    """Typed dict for multi_vector_translation kwarg expansion."""

    translation_from_col: str
    translation_to_col: str
    translation_factors_col: str
    translation_dtype: Optional[np.dtype]
    check_totals: bool


# # # FUNCTIONS # # #
# ## PRIVATE FUNCTIONS ## #
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


def _pandas_vector_validation(
    vector: pd.Series | pd.DataFrame,
    translation: pd.DataFrame,
    translation_from_col: str,
    from_unique_index: list[Any],
    to_unique_index: list[Any],
    name: str = "vector",
) -> None:
    """Validate the given parameters for a vector zone translation.

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

    from_unique_index:
        A list of all the unique IDs in the input indexing system.

    to_unique_index:
        A list of all the unique IDs in the output indexing system.

    name:
        The name to use in any warnings messages when they are raised.

    Returns
    -------
    None
    """
    validators.unique_list(from_unique_index, name="from_unique_index")
    validators.unique_list(to_unique_index, name="to_unique_index")

    # Make sure the vector only has the zones defined in from_unique_zones
    missing_rows = set(vector.index.to_list()) - set(from_unique_index)
    if len(missing_rows) > 0:
        warnings.warn(
            f"Some zones in `{name}.index` have not been defined in "
            f"`from_unique_zones`. These zones will be dropped before "
            f"translating.\n"
            f"Additional rows count: {len(missing_rows)}"
        )

    # Check all needed values are in from_zone_col
    trans_from_zones = set(translation[translation_from_col].unique())
    missing_zones = set(from_unique_index) - trans_from_zones
    if len(missing_zones) != 0:
        warnings.warn(
            f"Some zones in `{name}.index` are missing in `translation`. "
            f"Missing zones count: {len(missing_zones)}"
        )


def _pandas_matrix_validation(
    matrix: pd.DataFrame,
    row_translation: pd.DataFrame,
    col_translation: pd.DataFrame,
    translation_from_col: str,
    name: str = "matrix",
) -> None:
    """Validate the given parameters for a matrix zone translation.

    Parameters
    ----------
    matrix:
        The matrix to translate. The index and columns must be the values
        to be translated.

    row_translation:
        A pandas DataFrame defining the weights to translate use when
        translating.
        Needs to contain columns:
        `translation_from_col`, `translation_to_col`, `translation_factors_col`.

    col_translation:
        A pandas DataFrame defining the weights to translate use when
        translating.
        Needs to contain columns:
        `translation_from_col`, `translation_to_col`, `translation_factors_col`.

    translation_from_col:
        The name of the column in `translation` containing the current index
        values of `vector`.

    name:
        The name to use in any warnings messages when they are raised.

    Returns
    -------
    None
    """
    # Throw a warning if any index values are in the matrix, but not in the
    # row_translation. These values will just be dropped.
    translation_from = row_translation[translation_from_col].unique()
    missing_rows = set(matrix.index.to_list()) - set(translation_from)
    if len(missing_rows) > 0:
        total_value_dropped = matrix.loc[list(missing_rows)].to_numpy().sum()
        warnings.warn(
            f"Some zones in `{name}.index` have not been defined in "
            f"`row_translation`. These zones will be dropped before "
            f"translating.\n"
            f"Additional rows count: {len(missing_rows)}\n"
            f"Total value dropped: {total_value_dropped}"
        )

    # Throw a warning if any index values are in the matrix, but not in the
    # col_translation. These values will just be dropped.
    translation_from = col_translation[translation_from_col].unique()
    missing_cols = set(matrix.index.to_list()) - set(translation_from)
    if len(missing_cols) > 0:
        total_value_dropped = matrix[list(missing_cols)].to_numpy().sum()
        warnings.warn(
            f"Some zones in `{name}.index` have not been defined in "
            f"`col_translation`. These zones will be dropped before "
            f"translating.\n"
            f"Additional rows count: {len(missing_cols)}\n"
            f"Total value dropped: {total_value_dropped}"
        )


# ## PUBLIC FUNCTIONS ## #
def numpy_matrix_zone_translation(
    matrix: np.ndarray,
    translation: np.ndarray,
    col_translation: Optional[np.ndarray] = None,
    translation_dtype: Optional[np.dtype] = None,
    check_shapes: bool = True,
    check_totals: bool = True,
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
    # Init
    translation_from_col = "from_id"
    translation_to_col = "to_id"
    translation_factors_col = "factors"

    # ## OPTIONALLY CHECK INPUT SHAPES ## #
    row_translation = translation
    if col_translation is None:
        col_translation = translation.copy()

    if check_shapes:
        _check_matrix_translation_shapes(
            matrix=matrix,
            row_translation=row_translation,
            col_translation=col_translation,
        )

    # Set the id vals
    from_id_vals = list(range(translation.shape[0]))
    to_id_vals = list(range(translation.shape[1]))

    # Convert numpy arrays into pandas arrays
    dimension_cols = {translation_from_col: from_id_vals, translation_to_col: to_id_vals}
    pd_row_translation = pd_utils.n_dimensional_array_to_dataframe(
        mat=row_translation, dimension_cols=dimension_cols, value_col=translation_factors_col
    ).reset_index()
    pd_col_translation = pd_utils.n_dimensional_array_to_dataframe(
        mat=col_translation, dimension_cols=dimension_cols, value_col=translation_factors_col
    ).reset_index()

    return pandas_matrix_zone_translation(
        matrix=pd.DataFrame(data=matrix, columns=from_id_vals, index=from_id_vals),
        translation=pd_row_translation,
        col_translation=pd_col_translation,
        translation_from_col=translation_from_col,
        translation_to_col=translation_to_col,
        translation_factors_col=translation_factors_col,
        translation_dtype=translation_dtype,
        check_totals=check_totals,
    ).to_numpy()


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
    translation: pd.DataFrame,
    translation_from_col: str,
    translation_to_col: str,
    translation_factors_col: str,
    col_translation: Optional[pd.DataFrame] = None,
    translation_dtype: Optional[np.dtype] = None,
    check_totals: bool = True,
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

    translation_dtype:
        The numpy datatype to use to do the translation. If None, then the
        dtype of `vector` is used. Where such high precision
        isn't needed, a more memory and time efficient data type can be used.

    check_totals:
        Whether to check that the input and output matrices sum to the same
        total.

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
    # Init
    row_translation = translation
    if col_translation is None:
        col_translation = translation.copy()
    assert col_translation is not None

    # Set the index dtypes to match and validate
    (
        matrix.index,
        row_translation[translation_from_col],
        col_translation[translation_from_col],
    ) = pd_utils.cast_to_common_type(
        [
            matrix.index,
            row_translation[translation_from_col],
            col_translation[translation_from_col],
        ]
    )

    _pandas_matrix_validation(
        matrix=matrix,
        row_translation=row_translation,
        col_translation=col_translation,
        translation_from_col=translation_from_col,
    )

    # Build dictionary of repeated kwargs
    common_kwargs: _MultiVectorKwargs = {
        "translation_from_col": translation_from_col,
        "translation_to_col": translation_to_col,
        "translation_factors_col": translation_factors_col,
        "translation_dtype": translation_dtype,
        "check_totals": check_totals,
    }

    half_done = pandas_multi_vector_zone_translation(
        vector=matrix,
        translation=row_translation,
        **common_kwargs,
    )
    translated = pandas_multi_vector_zone_translation(
        vector=half_done.transpose(),
        translation=col_translation,
        **common_kwargs,
    ).transpose()

    if not check_totals:
        return translated

    if not math_utils.is_almost_equal(matrix.to_numpy().sum(), translated.to_numpy().sum()):
        raise ValueError(
            f"Some values seem to have been dropped during the translation. "
            f"Check the given translation matrix isn't unintentionally "
            f"dropping values. If the difference is small, it's likely a "
            f"rounding error.\n"
            f"Before: {matrix.to_numpy().sum()}\n"
            f"After: {translated.to_numpy().sum()}"
        )

    return translated


# TODO(BT): Can uncomment once we have pandas stubs
# @overload
# def pandas_vector_zone_translation(
#     vector: pd.Series,
#     translation: pd.DataFrame,
#     translation_from_col: str,
#     translation_to_col: str,
#     translation_factors_col: str,
#     from_unique_index: list[Any],
#     to_unique_index: list[Any],
#     translation_dtype: Optional[np.dtype] = None,
#     vector_infill: float = 0.0,
#     translate_infill: float = 0.0,
#     check_totals: bool = True,
# ) -> pd.Series:
#     # pylint: disable=too-many-arguments
#     ...  # pragma: no cover
#
#
# @overload
# def pandas_vector_zone_translation(
#     vector: pd.DataFrame,
#     translation: pd.DataFrame,
#     translation_from_col: str,
#     translation_to_col: str,
#     translation_factors_col: str,
#     from_unique_index: list[Any],
#     to_unique_index: list[Any],
#     translation_dtype: Optional[np.dtype] = None,
#     vector_infill: float = 0.0,
#     translate_infill: float = 0.0,
#     check_totals: bool = True,
# ) -> pd.Series | pd.DataFrame:
#     # pylint: disable=too-many-arguments
#     ...  # pragma: no cover


def pandas_vector_zone_translation(
    vector: pd.Series | pd.DataFrame,
    translation: pd.DataFrame,
    translation_from_col: str,
    translation_to_col: str,
    translation_factors_col: str,
    from_unique_index: Optional[list[Any]] = None,
    to_unique_index: Optional[list[Any]] = None,
    translation_dtype: Optional[np.dtype] = None,
    vector_infill: float = 0.0,
    translate_infill: float = 0.0,
    check_totals: bool = True,
) -> pd.Series | pd.DataFrame:
    # pylint: disable=too-many-arguments
    """Efficiently translate a pandas vector between index systems.

    Works for either single (Series) or multi (DataFrame) columns data vectors.
    Essentially switches between `pandas_single_vector_zone_translation()` and
    `pandas_multi_vector_zone_translation()`.

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
    `pandas_single_vector_zone_translation()`
    `pandas_multi_vector_zone_translation()`
    """
    if isinstance(vector, pd.DataFrame):
        if len(vector.columns) > 1:
            return pandas_multi_vector_zone_translation(
                vector=vector,
                translation=translation,
                translation_from_col=translation_from_col,
                translation_to_col=translation_to_col,
                translation_factors_col=translation_factors_col,
                check_totals=check_totals,
            )
        vector = vector.squeeze()

    if from_unique_index is None:
        raise ValueError("from_unique_index must be set for single vector translations")

    if to_unique_index is None:
        raise ValueError("to_unique_index must be set for single vector translations")

    return pandas_single_vector_zone_translation(
        vector=vector,
        translation=translation,
        translation_from_col=translation_from_col,
        translation_to_col=translation_to_col,
        translation_factors_col=translation_factors_col,
        from_unique_index=from_unique_index,
        to_unique_index=to_unique_index,
        translation_dtype=translation_dtype,
        vector_infill=vector_infill,
        translate_infill=translate_infill,
        check_totals=check_totals,
    )


def pandas_single_vector_zone_translation(
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
    """Efficiently translate a single-column pandas vector between index systems.

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

    # Set the dtypes to match
    vector.index, translation[translation_from_col] = pd_utils.cast_to_common_type(
        [vector.index, translation[translation_from_col]],
    )

    _pandas_vector_validation(
        vector=vector,
        translation=translation,
        translation_from_col=translation_from_col,
        from_unique_index=from_unique_index,
        to_unique_index=to_unique_index,
        name="vector",
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


def pandas_multi_vector_zone_translation(
    vector: pd.DataFrame,
    translation: pd.DataFrame,
    translation_from_col: str,
    translation_to_col: str,
    translation_factors_col: str,
    translation_dtype: Optional[np.dtype] = None,
    check_totals: bool = True,
) -> pd.DataFrame:
    """Efficiently translate a multi-column pandas vector between index systems.

    Internally, checks and converts the pandas inputs into numpy arrays
    and calls `numpy_vector_zone_translation()`. The final output is then
    converted back into a pandas Series, using the same format as the input.

    Parameters
    ----------
    vector:
        The vector to translate. The index must be the values to be translated.
        Any further segmentation data (i.e. data which should not be factored
        or translated) must be either in the columns or part of a MultiIndex.
        If part of a MultiIndex, the level of the MultiIndex to translate on
        must be named share a name with translation_from_col.

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

    translation_dtype:
        The numpy datatype to use to do the translation. If None, then the
        dtype of the vector is used. Where such high precision
        isn't needed, a more memory and time efficient data type can be used.

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
    vector = vector.copy()
    translation = translation.copy()

    # Throw a warning if any index values are in the vector, but not in the
    # translation. These values will just be dropped.
    translation_from = translation[translation_from_col].unique()

    # ## CONVERT DTYPES ## #
    # Convert data dtypes if needed
    if translation_dtype is None:
        translation_dtype = np.promote_types(
            translation[translation_factors_col].dtype, vector.to_numpy().dtype
        )
    assert translation_dtype is not None

    new_values = _convert_dtypes(
        arr=vector.to_numpy(),
        to_type=translation_dtype,
        arr_name="vector",
    )
    vector = pd.DataFrame(index=vector.index, columns=vector.columns, data=new_values)
    translation[translation_factors_col] = _convert_dtypes(
        arr=translation[translation_factors_col].to_numpy(),
        to_type=translation_dtype,
        arr_name="row_translation",
    )

    # ## PREP AND TRANSLATE ## #
    # set index for translation
    indexed_translation = translation.set_index([translation_from_col, translation_to_col])

    # Correct index for vector
    if isinstance(vector.index, pd.MultiIndex):
        ind_names = list(vector.index.names)
        if translation_from_col in ind_names:
            warnings.warn(
                "The input vector is MultiIndexed. The translation "
                f"will be done using the {translation_from_col} level "
                "of the index. If this is unexpected, check your "
                "inputs."
            )
            vector.reset_index(inplace=True)
            (
                vector[translation_from_col],
                translation[translation_from_col],
            ) = pd_utils.cast_to_common_type(
                [vector[translation_from_col], translation[translation_from_col]],
            )
            vector.set_index(ind_names, inplace=True)
            # this will be used for final grouping
            ind_names.remove(translation_from_col)
            missing_rows = set(vector.index.get_level_values(translation_from_col)) - set(
                translation_from
            )
            if len(missing_rows) > 0:
                total_value_dropped = vector.loc[list(missing_rows)].to_numpy().sum()
                warnings.warn(
                    f"Some zones in `vector.index` have not been defined in "
                    f"`translation`. These zones will be dropped before "
                    f"translating.\n"
                    f"Additional rows count: {len(missing_rows)}\n"
                    f"Total value dropped: {total_value_dropped}"
                )

        else:
            raise ValueError(
                "The input vector is MultiIndexed and does not "
                f"contain the expected column, {translation_from_col}."
                "Either rename the correct index level, or input "
                "a vector with a single index to use."
            )
    else:
        missing_rows = set(vector.index.to_list()) - set(translation_from)
        if len(missing_rows) > 0:
            total_value_dropped = vector.loc[list(missing_rows)].to_numpy().sum()
            warnings.warn(
                f"Some zones in `vector.index` have not been defined in "
                f"`translation`. These zones will be dropped before "
                f"translating.\n"
                f"Additional rows count: {len(missing_rows)}\n"
                f"Total value dropped: {total_value_dropped}"
            )
        vector.index, translation[translation_from_col] = pd_utils.cast_to_common_type(
            [vector.index, translation[translation_from_col]],
        )
        # Doesn't matter if it already equals this, quicker than checking.
        vector.index.names = [translation_from_col]
        ind_names = []

    # trans_vector should now contain the correct index level if an error hasn't
    # been raised
    translated = (
        vector.mul(indexed_translation[translation_factors_col].squeeze(), axis="index")
        .groupby(level=[translation_to_col] + ind_names)
        .sum()
    )

    if check_totals:
        overall_diff = translated.sum().sum() - vector.sum().sum()
        if not math_utils.is_almost_equal(translated.sum().sum(), vector.sum().sum()):
            raise ValueError(
                "Some values seem to have been dropped. The difference "
                f"total is {overall_diff} (translated - original)."
            )

    # Sometimes we need to remove the index name to make sure the same style of
    # dataframe is returned as that which came in
    if vector.index.name is None:
        translated.index.name = None

    return translated


# TODO(BT): Bring over from normits_demand (once we have zoning systems):
#  translate_vector_zoning
#  translate_matrix_zoning
#  get_long_translation
#  get_long_pop_emp_translations
