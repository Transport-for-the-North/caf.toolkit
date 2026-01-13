"""Tools to convert numpy/pandas vectors/matrices between different index systems.

In transport, these tools are very useful for translating data between different
zoning systems.
"""

from __future__ import annotations

import contextlib

# Built-Ins
import copy
import logging
import pathlib
import warnings
from collections.abc import Collection, Sequence
from typing import TYPE_CHECKING, Any, Literal, Self, TypedDict, TypeVar, overload

# Third Party
import numpy as np
import pandas as pd
from pydantic import ConfigDict, Field, FilePath, dataclasses, model_validator

# Local Imports
from caf.toolkit import io, math_utils, validators
from caf.toolkit import pandas_utils as pd_utils

if TYPE_CHECKING:
    from collections.abc import Hashable

# # # CONSTANTS # # #
_T = TypeVar("_T")

LOG = logging.getLogger(__name__)
DP_TOLERANCE = 6
ValidZoneIdDtype = str | int


# # # CLASSES # # #
class _MultiVectorKwargs(TypedDict):
    """Typed dict for multi_vector_translation kwarg expansion."""

    translation_dtype: np.dtype | None
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
            f"Given matrix shape: {matrix.shape!s}"
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


# TODO(BT): Move to numpy_utils??  # noqa: TD003
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
    translation: ZoneCorrespondence | pd.DataFrame,
    from_unique_index: list[Any],
    to_unique_index: list[Any],
    translation_from_col: str = "from",
    name: str = "vector",
) -> None:
    # pylint: disable=too-many-positional-arguments
    """Validate the given parameters for a vector zone translation.

    Parameters
    ----------
    vector:
        The vector to translate. The index must be the values to be translated.

    translation:
        A `ZoneCorrespondence` object defining the weights to use when
        translating.

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
    translation = _correspondence_from_df(translation, translation_from_col)
    validators.unique_list(from_unique_index, name="from_unique_index")
    validators.unique_list(to_unique_index, name="to_unique_index")

    # Make sure the vector only has the zones defined in from_unique_zones
    missing_rows = set(vector.index.to_list()) - set(from_unique_index)
    if len(missing_rows) > 0:
        warnings.warn(
            f"Some zones in `{name}.index` have not been defined in "
            f"`from_unique_zones`. These zones will be dropped before "
            f"translating.\n"
            f"Additional rows count: {len(missing_rows)}",
            stacklevel=2,
        )

    # Check all needed values are in from_zone_col
    trans_from_zones = set(translation.from_column.unique())
    missing_zones = set(from_unique_index) - trans_from_zones
    if len(missing_zones) != 0:
        warnings.warn(
            f"Some zones in `{name}.index` are missing in `translation`. "
            f"Missing zones count: {len(missing_zones)}",
            stacklevel=2,
        )


def _pandas_matrix_validation(
    matrix: pd.DataFrame,
    row_translation: ZoneCorrespondence | pd.DataFrame,
    col_translation: ZoneCorrespondence | pd.DataFrame,
    translation_from_col: str = "from",
    name: str = "matrix",
) -> None:
    """Validate the given parameters for a matrix zone translation.

    Parameters
    ----------
    matrix:
        The matrix to translate. The index and columns must be the values
        to be translated.

    row_translation:
        `ZoneCorrespondence` object defining the weights to translate use when
        translating.
        Needs to contain columns:
        `translation_from_col`, `translation_to_col`, `translation_factors_col`.

    col_translation:
        `ZoneCorrespondence` object defining the weights to translate use when
        translating.
        Needs to contain columns:
        `translation_from_col`, `translation_to_col`, `translation_factors_col`.

    name:
        The name to use in any warnings messages when they are raised.

    Returns
    -------
    None
    """
    # Throw a warning if any index values are in the matrix, but not in the
    # row_translation. These values will just be dropped.
    row_translation = _correspondence_from_df(row_translation, translation_from_col)
    col_translation = _correspondence_from_df(col_translation, translation_from_col)
    translation_from = row_translation.from_column.unique()
    missing_rows = set(matrix.index.to_list()) - set(translation_from)
    if len(missing_rows) > 0:
        total_value_dropped = matrix.loc[list(missing_rows)].to_numpy().sum()
        warnings.warn(
            f"Some zones in `{name}.index` have not been defined in "
            f"`row_translation`. These zones will be dropped before "
            f"translating.\n"
            f"Additional rows count: {len(missing_rows)}\n"
            f"Total value dropped: {total_value_dropped}",
            stacklevel=2,
        )

    # Throw a warning if any column values are in the matrix, but not in the
    # col_translation. These values will just be dropped.
    translation_from = col_translation.from_column.unique()
    missing_cols = set(matrix.columns.to_list()) - set(translation_from)
    if len(missing_cols) > 0:
        total_value_dropped = matrix[list(missing_cols)].to_numpy().sum()
        warnings.warn(
            f"Some zones in `{name}.columns` have not been defined in "
            f"`col_translation`. These zones will be dropped before "
            f"translating.\n"
            f"Additional rows count: {len(missing_cols)}\n"
            f"Total value dropped: {total_value_dropped}",
            stacklevel=2,
        )


# ## PUBLIC FUNCTIONS ## #
def numpy_matrix_zone_translation(
    matrix: np.ndarray,
    translation: np.ndarray,
    *,
    col_translation: np.ndarray | None = None,
    translation_dtype: np.dtype | None = None,
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
    # pylint: disable=too-many-locals
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
    dimension_cols = {
        translation_from_col: from_id_vals,
        translation_to_col: to_id_vals,
    }
    pd_row_translation = pd_utils.n_dimensional_array_to_dataframe(
        mat=row_translation,
        dimension_cols=dimension_cols,
        value_col=translation_factors_col,
    ).reset_index()
    zero_mask = pd_row_translation[translation_factors_col] == 0
    pd_row_translation = pd_row_translation[~zero_mask]

    pd_col_translation = pd_utils.n_dimensional_array_to_dataframe(
        mat=col_translation,
        dimension_cols=dimension_cols,
        value_col=translation_factors_col,
    ).reset_index()
    zero_mask = pd_col_translation[translation_factors_col] == 0
    pd_col_translation = pd_col_translation[~zero_mask]

    return pandas_matrix_zone_translation(
        matrix=pd.DataFrame(data=matrix, columns=from_id_vals, index=from_id_vals),
        zone_correspondence=ZoneCorrespondence(
            pd_row_translation,
            translation_from_col,
            translation_to_col,
            translation_factors_col,
        ),
        col_translation=ZoneCorrespondence(
            pd_col_translation,
            translation_from_col,
            translation_to_col,
            translation_factors_col,
        ),
        translation_dtype=translation_dtype,
        check_totals=check_totals,
    ).to_numpy()


def numpy_vector_zone_translation(
    vector: np.ndarray,
    translation: np.ndarray,
    translation_dtype: np.dtype | None = None,
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
            if len(vector.shape) == 2 and vector.shape[1] == 1:  # noqa: PLR2004
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
        translation_dtype = np.promote_types(vector.dtype, translation.dtype)
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
        raise

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


def pandas_long_matrix_zone_translation(  # noqa: PLR0913
    matrix: pd.DataFrame | pd.Series,
    index_col_1_name: str,
    index_col_2_name: str,
    values_col: str,
    zone_correspondence: ZoneCorrespondence | pd.DataFrame,
    translation_from_col: str = "from",
    translation_to_col: str = "to",
    translation_factors_col: str = "factors",
    col_translation: ZoneCorrespondence | pd.DataFrame | None = None,
    translation_dtype: np.dtype | None = None,
    index_col_1_out_name: str | None = None,
    index_col_2_out_name: str | None = None,
    check_totals: bool = True,
) -> pd.Series:
    # pylint: disable=too-many-positional-arguments
    """Efficiently translates a pandas matrix between index systems.

    Parameters
    ----------
    matrix:
        The matrix to translate, in long format. Must contain columns:
        [`index_col_1_name`, `index_col_2_name`, `value_col`].

    index_col_1_name:
        The name of the first column in `matrix` to translate index system.

    index_col_2_name:
        The name of the second column in `matrix` to translate index system.

    values_col:
        The name of the column in `matrix` detailing the values to translate.

    zone_correspondence:
        A `ZoneCorrespondence` object defining the weights to use when translating.
        When `col_translation` is None, this defines the translation to use
        for both the rows and columns. When `col_translation` is set, this
        defines the translation to use for the rows.

    translation_from_col : str = "from"
        Name of zone ID column in translation which corresponds to the current vector
        zone ID. Deprecated, only provide if zone correspondence is a dataframe.

    translation_to_col : str = "to"
        Name of column in translation for the new zone IDs. Deprecated, only provide if zone
        correspondence is a dataframe.

    translation_factors_col : str = "factors
        Name of column in translation. Deprecated, only provide if zone correspondence is a
        dataframe.

    col_translation:
        A `ZoneCorrespondence` object defining the weights to use to translate the columns.
        Takes an input of the same format as `translation`. When None,
        `translation` is used as the column translation.

    translation_dtype:
        The numpy datatype to use to do the translation. If None, then the
        dtype of `vector` is used. Where such high precision
        isn't needed, a more memory and time efficient data type can be used.

    check_totals:
        Whether to check that the input and output matrices sum to the same
        total.

    index_col_1_out_name:
        The name to give to `index_col_1_name` on return.

    index_col_2_out_name:
        The name to give to `index_col_2_name` on return.

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
    zone_correspondence = _correspondence_from_df(
        zone_correspondence,
        translation_from_col,
        translation_to_col,
        translation_factors_col,
    )
    if col_translation is not None:
        col_translation = _correspondence_from_df(
            col_translation,
            translation_from_col,
            translation_to_col,
            translation_factors_col,
        )
    if (index_col_2_out_name is None) ^ (index_col_1_out_name is None):
        raise ValueError("If one of index_col_out_name is set, both must be set.")
    matrix = matrix.copy()
    keep_cols = [index_col_1_name, index_col_2_name, values_col]
    if isinstance(matrix, pd.DataFrame):
        all_cols = matrix.columns.tolist()
        # Drop any columns we're not keeping
        drop_cols = set(all_cols) - set(keep_cols)
        if len(drop_cols) > 0:
            warnings.warn(
                f"Extra columns found in matrix, dropping the following: {drop_cols}",
                stacklevel=2,
            )
        matrix = pd_utils.reindex_cols(df=matrix, columns=keep_cols)
        series_mat = matrix.set_index([index_col_1_name, index_col_2_name]).squeeze()
        if not isinstance(series_mat, pd.Series):
            raise TypeError("")
    else:
        series_mat = matrix

    # Convert to wide to translate
    wide_mat = pd_utils.long_to_wide_infill(matrix=series_mat)

    translated_wide_mat = pandas_matrix_zone_translation(
        matrix=wide_mat,
        zone_correspondence=zone_correspondence,
        col_translation=col_translation,
        translation_dtype=translation_dtype,
        check_totals=check_totals,
    )

    # Convert back
    out_mat = pd_utils.wide_to_long_infill(df=translated_wide_mat)
    if index_col_2_out_name is not None:
        # Check at the start of function makes sure if one is not None, both are
        assert index_col_1_out_name is not None  # noqa: S101
        out_mat.index.names = [index_col_1_out_name, index_col_2_out_name]

    return out_mat


def pandas_matrix_zone_translation(
    matrix: pd.DataFrame,
    zone_correspondence: ZoneCorrespondence | pd.DataFrame,
    translation_from_col: str = "from",
    translation_to_col: str = "to",
    translation_factors_col: str = "factors",
    col_translation: ZoneCorrespondence | pd.DataFrame | None = None,
    translation_dtype: np.dtype | None = None,
    check_totals: bool = True,
) -> pd.DataFrame:
    # pylint: disable=too-many-positional-arguments
    """Efficiently translates a pandas matrix between index systems.

    Only works on wide matrices and not long. If translating long matrices,
    use `pandas_long_matrix_zone_translation` instead.

    Parameters
    ----------
    matrix:
        The matrix to translate. The index and columns need to be the
        values being translated. This CANNOT be a "long" matrix.

    zone_correspondence:
        A `ZoneCorrespondence` object defining the weights to use when translating.
        When `col_translation` is None, this defines the translation to use
        for both the rows and columns. When `col_translation` is set, this
        defines the translation to use for the rows.

    translation_from_col : str = "from"
        Name of zone ID column in translation which corresponds to the current vector
        zone ID. Deprecated, only provide if zone correspondence is a dataframe.

    translation_to_col : str = "to"
        Name of column in translation for the new zone IDs. Deprecated, only provide if zone
        correspondence is a dataframe.

    translation_factors_col : str = "factors
        Name of column in translation. Deprecated, only provide if zone correspondence is a
        dataframe.

    col_translation:
        A `ZoneCorrespondence` object defining the weights to use to translate the columns.
        Takes an input of the same format as `translation`. When None,
        `translation` is used as the column translation.

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
    zone_correspondence = _correspondence_from_df(
        zone_correspondence,
        translation_from_col,
        translation_to_col,
        translation_factors_col,
    )
    row_translation = zone_correspondence
    if col_translation is not None:
        col_translation = _correspondence_from_df(
            col_translation,
            translation_from_col,
            translation_to_col,
            translation_factors_col,
        )
    else:
        col_translation = zone_correspondence.copy()

    # Set the index dtypes to match and validate
    (
        matrix.index,
        matrix.columns,
        row_translation.from_column,
        col_translation.from_column,
    ) = pd_utils.cast_to_common_type(
        [
            matrix.index,
            matrix.columns,
            row_translation.from_column,
            col_translation.from_column,
        ]
    )

    _pandas_matrix_validation(
        matrix=matrix,
        row_translation=row_translation,
        col_translation=col_translation,
    )

    # Build dictionary of repeated kwargs
    common_kwargs: _MultiVectorKwargs = {
        "translation_dtype": translation_dtype,
        "check_totals": False,
    }

    with warnings.catch_warnings():
        # Ignore the warnings we've already checked for
        msg = ".*zones will be dropped.*"
        warnings.filterwarnings(action="ignore", message=msg, category=UserWarning)

        half_done = pandas_vector_zone_translation(
            vector=matrix,
            zone_correspondence=row_translation,
            **common_kwargs,
        )
        translated = pandas_vector_zone_translation(
            vector=half_done.transpose(),
            zone_correspondence=col_translation,
            **common_kwargs,
        ).transpose()

    if not check_totals:
        return translated

    if not math_utils.is_almost_equal(matrix.to_numpy().sum(), translated.to_numpy().sum()):
        warnings.warn(
            f"Some values seem to have been dropped during the translation. "
            f"Check the given translation matrix isn't unintentionally "
            f"dropping values. If the difference is small, it's likely a "
            f"rounding error.\n"
            f"Before: {matrix.to_numpy().sum()}\n"
            f"After: {translated.to_numpy().sum()}",
            stacklevel=2,
        )

    return translated


@overload
def pandas_vector_zone_translation(
    vector: pd.DataFrame,
    zone_correspondence: ZoneCorrespondence | pd.DataFrame,
    translation_from_col: str = "from",
    translation_to_col: str = "to",
    translation_factors_col: str = "factors",
    check_totals: bool = True,
    translation_dtype: np.dtype | None = None,
) -> pd.DataFrame:
    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-positional-arguments
    ...  # pragma: no cover


@overload
def pandas_vector_zone_translation(
    vector: pd.Series,
    zone_correspondence: ZoneCorrespondence | pd.DataFrame,
    translation_from_col: str = "from",
    translation_to_col: str = "to",
    translation_factors_col: str = "factors",
    check_totals: bool = True,
    translation_dtype: np.dtype | None = None,
) -> pd.Series:
    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-positional-arguments
    ...  # pragma: no cover


def pandas_vector_zone_translation(
    vector: pd.Series | pd.DataFrame,
    zone_correspondence: ZoneCorrespondence | pd.DataFrame,
    translation_from_col: str = "from",
    translation_to_col: str = "to",
    translation_factors_col: str = "factors",
    check_totals: bool = True,
    translation_dtype: np.dtype | None = None,
) -> pd.Series | pd.DataFrame:
    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-positional-arguments
    """Efficiently translate a pandas vector between index systems.

    Works for either single (Series) or multi (DataFrame) columns data vectors.
    Essentially switches between `pandas_single_vector_zone_translation()` and
    `pandas_multi_vector_zone_translation()`.

    Parameters
    ----------
    vector:
        The vector to translate. The index must be the values to be translated.

    zone_correspondence:
        A `ZoneCorrespondence` object defining the weights to translate use when
        translating.

    translation_from_col : str = "from"
        Name of zone ID column in translation which corresponds to the current vector
        zone ID. Deprecated, only provide if zone correspondence is a dataframe.

    translation_to_col : str = "to"
        Name of column in translation for the new zone IDs. Deprecated, only provide if zone
        correspondence is a dataframe.

    translation_factors_col : str = "factors
        Name of column in translation. Deprecated, only provide if zone correspondence is a
        dataframe.

    check_totals:
        Whether to check that the input and output matrices sum to the same
        total.

    translation_dtype:
        The numpy datatype to use to do the translation. If None, then the
        dtype of `vector` is used. Where such high precision
        isn't needed, a more memory and time efficient data type can be used.

    Returns
    -------
    translated_vector:
        vector, translated into to_zone system.

    See Also
    --------
    `pandas_single_vector_zone_translation()`
    `pandas_multi_vector_zone_translation()`
    """
    zone_correspondence = _correspondence_from_df(
        zone_correspondence,
        translation_from_col,
        translation_to_col,
        translation_factors_col,
    )
    vector = vector.copy()
    zone_correspondence = zone_correspondence.copy()

    # Throw a warning if any index values are in the vector, but not in the
    # translation. These values will just be dropped.
    translation_from = zone_correspondence.from_column.unique()

    if translation_dtype is None:
        translation_dtype = np.promote_types(
            zone_correspondence.factors_column.to_numpy().dtype, vector.to_numpy().dtype
        )

    new_values = _convert_dtypes(
        arr=vector.to_numpy(),
        to_type=translation_dtype,
        arr_name="vector",
    )

    if isinstance(vector, pd.Series):
        vector = pd.Series(index=vector.index, name=vector.name, data=new_values)
    else:
        vector = pd.DataFrame(index=vector.index, columns=vector.columns, data=new_values)

    zone_correspondence.factors_column = _convert_dtypes(
        arr=zone_correspondence.factors_column.to_numpy(),
        to_type=translation_dtype,
        arr_name="row_translation",
    )

    # ## PREP AND TRANSLATE ## #
    # set index for translation
    indexed_translation = zone_correspondence.translation_vector

    # Fixing indices for the zone translation
    ind_names, vector, _ = _multi_vector_trans_index(
        vector,
        zone_correspondence.translation_vector.reset_index(),
        zone_correspondence.from_col_name,
        translation_from,
    )

    # trans_vector should now contain the correct index level if an error hasn't
    # been raised
    factors = indexed_translation[zone_correspondence.factors_col_name].squeeze()
    if not isinstance(factors, pd.Series):
        raise TypeError("Input translation vector is probably the wrong shape.")
    translated = (
        vector.mul(factors, axis=0)
        .groupby(level=[zone_correspondence.to_col_name, *ind_names])
        .sum()
    )

    if check_totals:
        overall_diff = translated.sum().sum() - vector.sum().sum()
        if not math_utils.is_almost_equal(translated.sum().sum(), vector.sum().sum()):
            warnings.warn(
                "Some values seem to have been dropped. The difference "
                f"total is {overall_diff} (translated - original).",
                stacklevel=2,
            )

    # Sometimes we need to remove the index name to make sure the same style of
    # dataframe is returned as that which came in
    if vector.index.name is None:
        translated.index.name = None

    # Make sure the output has the same name as input series
    if isinstance(vector, pd.Series) and isinstance(translated, pd.Series):
        translated.name = vector.name

    return translated


def _vector_missing_warning(vector: pd.DataFrame | pd.Series, missing_rows: list) -> None:
    """Warn when zones are missing from vector.

    Produces RuntimeWarning detailing the number of missing rows and
    the total value, with count of NaN values in the missing rows.
    """
    n_nans = np.sum(vector.loc[missing_rows].isna().to_numpy())
    n_cells = vector.loc[missing_rows].size
    total_value_dropped = np.nansum(vector.loc[missing_rows].to_numpy())
    if vector.index.names[0] is None:
        index_name = "`vector.index`"
    else:
        index_name = f"`vector.index` ({vector.index.names[0]})"

    warnings.warn(
        f"Some zones in {index_name} have not been defined in "
        "`translation`. These zones will be dropped before translating.\n"
        f"Missing rows count: {len(missing_rows)}\n"
        f"Total value dropped: {total_value_dropped}\n"
        f"NaN cells: {n_nans} / {n_cells} ({n_nans / n_cells:.0%} of missing rows)",
        stacklevel=2,
    )
    LOG.debug("Missing zones dropped before translation: %s", missing_rows)


def _multi_vector_trans_index(
    vector: pd.DataFrame | pd.Series,
    translation: pd.DataFrame,
    translation_from_col: str,
    translation_from: np.ndarray,
) -> tuple[list[Hashable], pd.DataFrame | pd.Series, pd.DataFrame]:
    """Create correct index for `pandas_multi_vector_zone_translation`."""
    if isinstance(vector.index, pd.MultiIndex):
        ind_names = list(vector.index.names)
        if translation_from_col in ind_names:
            warnings.warn(
                "The input vector is MultiIndexed. The translation "
                f"will be done using the {translation_from_col} level "
                "of the index. If this is unexpected, check your "
                "inputs.",
                stacklevel=2,
            )
            vector = vector.reset_index()
            (
                vector[translation_from_col],
                translation[translation_from_col],
            ) = pd_utils.cast_to_common_type(
                [vector[translation_from_col], translation[translation_from_col]],
            )
            vector = vector.set_index(ind_names)
            # this will be used for final grouping
            ind_names.remove(translation_from_col)
            missing_rows = set(vector.index.get_level_values(translation_from_col)) - set(
                translation_from
            )
            if len(missing_rows) > 0:
                _vector_missing_warning(vector, list(missing_rows))

        else:
            raise ValueError(
                "The input vector is MultiIndexed and does not "
                f"contain the expected column, {translation_from_col}."
                "Either rename the correct index level, or input "
                "a vector with a single index to use."
            )
    else:
        vector.index, translation[translation_from_col] = pd_utils.cast_to_common_type(
            [vector.index, translation[translation_from_col]],
        )
        missing_rows = set(vector.index.to_list()) - set(translation_from)
        if len(missing_rows) > 0:
            _vector_missing_warning(vector, list(missing_rows))

        # Doesn't matter if it already equals this, quicker than checking.
        vector.index.names = [translation_from_col]
        ind_names = []

    return ind_names, vector, translation


def _load_translation(
    path: pathlib.Path,
    from_column: int | str,
    to_column: int | str,
    factors_column: int | str,
) -> tuple[pd.DataFrame, tuple[str, str, str]]:
    """Load translation file and determine name of any column positions given.

    Returns
    -------
    pd.DataFrame
        Translation data.
    (str, str, str)
        From, to and factors column names.
    """
    LOG.info("Loading translation lookup data from: '%s'", path)
    data = io.read_csv(
        path,
        name="translation lookup",
        usecols=[from_column, to_column, factors_column],
        dtype={factors_column: float},
    )

    columns = (from_column, to_column, factors_column)
    str_columns = [data.columns[i] if isinstance(i, int) else i for i in columns]

    # Attempt to convert ID columns to integers,
    # but not necessarily a problem if they aren't
    for column in str_columns[:2]:
        with contextlib.suppress(ValueError):
            data[column] = pd.to_numeric(data[column], downcast="integer")

    columns = ("from_column", "to_column", "factors_column")
    LOG.info(
        "Translation loaded with following columns:\n\t%s",
        "\n\t".join(f"{i}: {j}" for i, j in zip(columns, str_columns, strict=True)),
    )

    #  MyPy is confused about the tuple
    return data, tuple(str_columns)  # type: ignore[return-value]


def _validate_column_name_parameters(params: dict[str, Any], *names: str) -> None:
    """Check all `names` are present and are strings in `params`.

    Raises TypeError for any `names` which aren't strings in `params`.
    """
    any_positions = False
    for name in names:
        value = params.get(name)
        if isinstance(value, int):
            any_positions = True

        elif not isinstance(value, str):
            raise TypeError(
                f"{name} parameter should be the name of a column not {type(value)}"
            )

    if any_positions:
        warnings.warn(
            "column positions are given instead of names,"
            " make sure the columns are in the correct order",
            stacklevel=2,
        )


def vector_translation_from_file(
    vector_path: pathlib.Path,
    zone_correspondence_path: ZoneCorrespondencePath | pathlib.Path,
    output_path: pathlib.Path,
    *,
    vector_zone_column: str | int,
    translation_from_column: str = "from",
    translation_to_column: str = "to",
    translation_factors_column: str = "factors",
) -> None:
    """Translate zoning system of vector CSV file.

    Load vector from CSV, perform translation and write to new CSV.

    Parameters
    ----------
    vector_path : pathlib.Path
        Path to CSV file containing data to be translated.
    zone_correspondence_path : ZoneCorrespondencePath
        Object defining the lookup CSV.
    output_path : pathlib.Path
        CSV path to save the translated data to.
    vector_zone_column : str | int
        Name, or position, of the zone ID column in `vector_path` file.
    translation_from_col : str = "from"
        Name of zone ID column in translation which corresponds to the current vector
        zone ID. Deprecated, only provide if zone correspondence is a dataframe.
    translation_to_col : str = "to"
        Name of column in translation for the new zone IDs. Deprecated, only provide if zone
        correspondence is a dataframe.
    translation_factors_col : str = "factors
        Name of column in translation. Deprecated, only provide if zone correspondence is a
        dataframe.
    """
    # TODO(MB): Add optional from / to unique index parameters  # noqa: TD003
    # pylint: disable=too-many-locals
    # otherwise infer from translation file
    zone_correspondence_path = _correspondence_path_from_path(
        zone_correspondence_path,
        translation_from_column,
        translation_to_column,
        translation_factors_column,
    )
    LOG.info("Loading vector data from: '%s'", vector_path)
    vector = io.read_csv(vector_path, index_col=[vector_zone_column])
    LOG.info(
        "Loaded vector data with zone ID (index) column '%s' and %s data columns: %s",
        vector.index.name,
        len(vector.columns),
        ", ".join(f"'{i}'" for i in vector.columns),
    )

    non_numeric = vector.select_dtypes(exclude="number").columns.tolist()
    if len(non_numeric) > 0:
        LOG.warning(
            "Ignoring %s columns containing non-numeric"
            " data, these will not be translated: %s",
            len(non_numeric),
            ", ".join(f"'{i}'" for i in non_numeric),
        )
        vector = vector.drop(columns=non_numeric)

    if len(vector.columns) == 0:
        LOG.error("no numeric columns in vector data to translate")
        return

    lookup = zone_correspondence_path.read()

    translated = pandas_vector_zone_translation(
        vector,
        lookup,
    )

    translated.to_csv(output_path)
    LOG.info("Written translated CSV to '%s'", output_path)


def matrix_translation_from_file(
    matrix_path: pathlib.Path,
    zone_correspondence_path: ZoneCorrespondencePath | pathlib.Path,
    output_path: pathlib.Path,
    *,
    matrix_zone_columns: tuple[int | str, int | str],
    matrix_values_column: int | str,
    translation_from_column: str = "from",
    translation_to_column: str = "to",
    translation_factors_column: str = "factors",
    format_: Literal["square", "long"] = "long",
) -> None:
    """Translate zoning system of matrix CSV file.

    Load matrix from CSV, perform translation and write to new
    CSV. CSV files are expected to be in the matrix 'long' format.

    Parameters
    ----------
    matrix_path : pathlib.Path
        Path to matrix CSV file.
    zone_correspondence_path : ZoneCorrespondencePath
        Definition of path and columns to use from translation.
    output_path : pathlib.Path
        CSV path to save the translated data to.
    matrix_zone_columns : tuple[int | str, int | str]
        Names, or positions, of the 2 columns containing
        the zone IDs in the matrix file.
    matrix_values_column : int | str
        Name, or position, of the column containing the matrix values.
    translation_from_col : str = "from"
        Name of zone ID column in translation which corresponds to the current vector
        zone ID. Deprecated, only provide if zone correspondence is a dataframe.
    translation_to_col : str = "to"
        Name of column in translation for the new zone IDs. Deprecated, only provide if zone
        correspondence is a dataframe.
    translation_factors_col : str = "factors
        Name of column in translation. Deprecated, only provide if zone correspondence is a
        dataframe.
    format_: Literal["square", "long"] = "long",
        Whether the matrix is in long or wide format.
    """
    zone_correspondence_path = _correspondence_path_from_path(
        zone_correspondence_path,
        translation_from_column,
        translation_to_column,
        translation_factors_column,
    )
    if format_ == "square":
        raise NotImplementedError("Square matrices are not yet supported.")

    matrix_zone_columns = tuple(matrix_zone_columns)
    not_strings = any(not isinstance(i, str) for i in matrix_zone_columns)
    expected_columns = 2
    if len(matrix_zone_columns) != expected_columns or not_strings:
        raise TypeError(
            "matrix_zone_columns should be a tuple containing the names"
            f" of {expected_columns} columns not {matrix_zone_columns}"
        )

    LOG.info("Loading matrix data from: '%s'", matrix_path)
    matrix = io.read_csv_matrix(
        matrix_path,
        format_=format_,
        index_col=matrix_zone_columns,
        usecols=[*matrix_zone_columns, matrix_values_column],
        dtype={matrix_values_column: float},
    )
    LOG.info(
        "Loaded matrix with index from '%s' and columns from '%s' containing %s cells",
        matrix.index.name,
        matrix.columns.name,
        matrix.size,
    )

    zone_correspondence = zone_correspondence_path.read()

    translated = pandas_matrix_zone_translation(
        matrix,
        zone_correspondence,
    )

    translated.index.name = matrix.index.name
    translated.columns.name = matrix.columns.name

    if format_ == "long":
        # Stack is returning a Series, MyPy is wrong
        translated = translated.stack().to_frame()  # type: ignore[operator]

        # Get name of value column
        if isinstance(matrix_values_column, str):
            translated.columns = [matrix_values_column]
        else:
            headers = io.read_csv(
                matrix_path,
                index_col=matrix_zone_columns,
                usecols=[*matrix_zone_columns, matrix_values_column],
                nrows=2,
            )
            translated.columns = headers.columns

    translated.to_csv(output_path)
    LOG.info("Written translated matrix CSV to '%s'", output_path)


@dataclasses.dataclass
class ZoneCorrespondencePath:
    """Defines the path and columns to use for a translation."""

    path: FilePath
    """Path to the translation file."""
    _from_col_name: str | int = Field(alias="from_col_name")
    """Column name for the from zoning IDs."""
    _to_col_name: str | int = Field(alias="to_col_name")
    """Column name for the to zoning IDs."""
    _factors_col_name: str | int | None = Field(alias="factors_col_name", default=None)
    """Column name for the translation factors."""

    @property
    def from_col_name(self) -> str:
        """Get _from_col_name."""
        if isinstance(self._from_col_name, str):
            return self._from_col_name
        headers = pd.read_csv(self.path, nrows=0).columns.tolist()
        return headers[self._from_col_name]

    @property
    def to_col_name(self) -> str:
        """Get _to_col_name."""
        if isinstance(self._to_col_name, str):
            return self._to_col_name
        headers = pd.read_csv(self.path, nrows=0).columns.tolist()
        return headers[self._to_col_name]

    @property
    def factors_col_name(self) -> str | None:
        """Get _factors_col_name."""
        if isinstance(self._factors_col_name, str):
            return self._factors_col_name
        if self._factors_col_name is None:
            return self._factors_col_name
        headers = pd.read_csv(self.path, nrows=0).columns.tolist()
        return headers[self._factors_col_name]

    @property
    def generic_from_col(self) -> str:
        """Column name to replace the `from_col_name` if reading with generic column names."""
        return "from"

    @property
    def generic_to_col(self) -> str:
        """Column name to replace the `to_col_name` when reading with generic column names."""
        return "to"

    @property
    def generic_factor_col(self) -> str:
        """Replace the `factors_col_name` when reading with generic column names."""
        return "factors"

    @property
    def _generic_column_name_lookup(self) -> dict[str, str]:
        """Lookup to use when replacing generic column names."""
        lookup: dict[str, str] = {
            self.from_col_name: self.generic_from_col,
            self.to_col_name: self.generic_to_col,
        }

        if self.factors_col_name is not None:
            lookup[self.factors_col_name] = self.generic_factor_col

        return lookup

    @property
    def _use_cols(self) -> list[str]:
        """Columns to use when reading in the csv translation vector."""
        cols = [self.from_col_name, self.to_col_name]
        if self.factors_col_name is not None:
            cols.append(self.factors_col_name)

        return cols

    def read(
        self, *, factors_mandatory: bool = True, generic_column_names: bool = False
    ) -> ZoneCorrespondence:
        """Read the translation file.

        Parameters
        ----------
        factors_mandatory
            If True (default), an error will be raised if the factors
            column is not present.
        generic_column_names
            If True (default), the columns will be renamed
            to "from", "to" and "factors".

        Returns
        -------
        ZoneCorrespondence
            translation vector read from path.
        """
        if factors_mandatory and self.factors_col_name is None:
            raise ValueError("Factors column name is mandatory.")

        translation = pd.read_csv(
            self.path,
            usecols=self._use_cols,
        )

        if factors_mandatory:
            if not pd.api.types.is_numeric_dtype(translation[self.factors_col_name]):
                raise ValueError(f"{self.factors_col_name} must contain numeric values only.")
            if (translation[self.factors_col_name] > 1).any():
                warnings.warn(
                    "%s contains values greater than one,"
                    " this does not make sense for a zone translation factor",
                    stacklevel=2,
                )
            if (translation[self.factors_col_name] < 0).any():
                warnings.warn(
                    "%s contains values less than one,"
                    " this does not make sense for a zone translation factor",
                    stacklevel=2,
                )
            factors_col = self.factors_col_name

        else:
            translation[self.generic_factor_col] = 1
            factors_col = self.generic_factor_col

        if generic_column_names:
            translation = translation.rename(columns=self._generic_column_name_lookup)
            from_col = self.generic_from_col
            to_col = self.generic_to_col

        else:
            from_col = self.from_col_name
            to_col = self.to_col_name

        if factors_col is None:
            raise ValueError("Should not be None here.")

        return ZoneCorrespondence(
            translation,
            from_col,
            to_col,
            factors_col,
        )


@dataclasses.dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class ZoneCorrespondence:
    """Container for a pandas based translation."""

    vector: pd.DataFrame
    """Translation vector."""
    _from_col_name: str | int = Field(alias="from_col_name")
    """Column name for the from zoning IDs."""
    _to_col_name: str | int = Field(alias="to_col_name")
    """Column name for the to zoning IDs."""
    _factors_col_name: str | int = Field(alias="factors_col_name")
    """Column name for the translation factors."""

    @property
    def from_col_name(self) -> str:
        """Get _from_col_name."""
        if isinstance(self._from_col_name, str):
            return self._from_col_name
        headers = self.vector.columns.tolist()
        return headers[self._from_col_name]

    @property
    def factors_col_name(self) -> str | None:
        """Get _factors_col_name."""
        if isinstance(self._factors_col_name, str):
            return self._factors_col_name
        if self._factors_col_name is None:
            return self._factors_col_name
        headers = self.vector.columns.tolist()
        return headers[self._factors_col_name]

    @property
    def to_col_name(self) -> str:
        """Get _to_col_name."""
        if isinstance(self._to_col_name, str):
            return self._to_col_name
        headers = self.vector.columns.tolist()
        return headers[self._to_col_name]

    @model_validator(mode="after")  # type: ignore[arg-type]
    def _validate_cols(self: Self) -> None:
        """Format and check translation vector."""
        try:
            self.vector = self.vector.reset_index()
            self.vector = self.vector[
                [self.from_col_name, self.to_col_name, self.factors_col_name]
            ]
        except KeyError as e:
            raise KeyError(
                "Zone correspondence must contain the columns "
                f"{self.from_col_name}, {self.to_col_name}, {self.factors_col_name}"
            ) from e
        self._validate_factors()

    def _validate_factors(self) -> None:
        """Validate the factors are in the right format and make logical sense."""
        if self.factors_col_name is None:
            raise TypeError("factor")
        factor_type = self.vector[self.factors_col_name].dtype.kind

        if factor_type not in ("i", "f"):
            raise ValueError(
                f"Factors must be numeric not {self.vector[self.factors_col_name].dtype}"
            )

        if self.vector[self.factors_col_name].isna().any():
            raise ValueError("Factors column contains NaNs")

        factor_col_sums = self.vector.groupby(self.from_col_name)[self.factors_col_name].sum()

        if any(factor_col_sums.round(DP_TOLERANCE) > 1):
            raise ValueError("Factors cannot be greater than one.")

        if any(self.vector[self.factors_col_name] < 0):
            raise ValueError("Factors cannot be negative.")

    @property
    def translation_vector(self) -> pd.DataFrame:
        """Translation vector table."""
        return self.vector.set_index([self.from_col_name, self.to_col_name])

    @property
    def from_column(self) -> pd.Series:
        """From zone column from the translation vector."""
        return self.vector[self.from_col_name].copy()

    @from_column.setter
    def from_column(self, new_col: pd.Series) -> None:
        if not isinstance(new_col, pd.Series):
            raise TypeError(f"New col must be a Series not {type(new_col)}")
        if len(self.vector) != len(new_col):
            raise ValueError(f"New col len {len(new_col)} != df len {len(self.vector)}")

        self.vector[self.from_col_name] = new_col

    @property
    def to_column(self) -> pd.Series:
        """To zone column from the translation vector."""
        return self.vector[self.to_col_name].copy()

    @to_column.setter
    def to_column(self, new_col: pd.Series) -> None:
        if not isinstance(new_col, pd.Series):
            raise TypeError(f"New col must be a Series not {type(new_col)}")
        if len(self.vector) != len(new_col):
            raise ValueError(f"New col len {len(new_col)} != df len {len(self.vector)}")

        self.vector[self.to_col_name] = new_col

    @property
    def factors_column(self) -> pd.Series:
        """Factors zone column from the translation vector."""
        return self.vector[self.factors_col_name].copy()

    @factors_column.setter
    def factors_column(self, new_col: Collection) -> None:
        try:
            new_col = pd.Series(new_col)
        except (ValueError, TypeError) as e:
            raise TypeError(f"New col must be a Series not {type(new_col)}") from e
        if len(self.vector) != len(new_col):
            raise ValueError(f"New col len {len(new_col)} != df len {len(self.vector)}")

        self.vector[self.factors_col_name] = new_col

        self._validate_factors()

    def get_correspondence(
        self,
        filter_zones: ValidZoneIdDtype | Sequence[ValidZoneIdDtype],
        filter_on: Literal["from", "to"] = "from",
    ) -> pd.DataFrame:
        """Retrieve the correspondence for a subset of zones.

        If the filter zones do not exist an empty dataframe is returned

        Parameters
        ----------
        filter_zones : ValidZoneIdDtype | Sequence[ValidZoneIdDtype]
            The zones for which to retrieve correspondence.
        filter_on : Literal["from", "to"], optional
            If set to "from" (default) the filter is applied to the from column in the
            translation vector. If "to" the filter is applied to the to column.

        Returns
        -------
        pd.DataFrame
            All columns in the zone correspondence for the selected zones.
        """
        if not isinstance(filter_zones, Sequence):
            filter_zones = [filter_zones]
        if not isinstance(filter_zones, list):
            filter_zones = list(filter_zones)

        if filter_on == "from":
            filter_col: pd.Series = self.from_column
            filter_col_name: str = self.from_col_name
        elif filter_on == "to":
            filter_col = self.to_column
            filter_col_name = self.to_col_name
        else:
            raise ValueError(f"filter_on must be either 'from' or 'to', not {filter_on}")

        if len(missing := (set(filter_zones) - set(filter_col.to_list()))) > 0:
            raise KeyError(f"Zones {missing} not in {filter_col} column")

        return self.vector[self.vector[filter_col_name].isin(filter_zones)]

    def copy(self) -> ZoneCorrespondence:
        """Deep copy of the ZoneCorrespondence object."""
        return copy.deepcopy(self)


# TO BE DELETED WHEN FULLY DEPRECATED
def _correspondence_from_df(
    translation: pd.DataFrame | ZoneCorrespondence,
    from_col: str = "from",
    to_col: str = "to",
    factors_col: str = "factors",
) -> ZoneCorrespondence:
    if isinstance(translation, pd.DataFrame):
        warnings.warn(
            "Zone translations in caf.toolkit should now use the ZoneCorrespondence "
            "class as an input, rather than a DataFrame. DataFrames will raise an error in "
            "caf.toolkit 1.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        # PyLint doesn't recognise pydantic Field aliases
        return ZoneCorrespondence(  # pylint: disable=unexpected-keyword-arg
            vector=translation,
            from_col_name=from_col,
            to_col_name=to_col,
            factors_col_name=factors_col,
        )
    return translation


def _correspondence_path_from_path(
    translation_path: pathlib.Path | ZoneCorrespondencePath,
    from_col: str,
    to_col: str,
    factors_col: str,
) -> ZoneCorrespondencePath:
    if isinstance(translation_path, pathlib.Path):
        warnings.warn(
            "Zone translations from file in caf.toolkit should now use the "
            "ZoneCorrespondencePath class as an input, rather than a simple Path. "
            "Paths will raise error in caf.toolkit 1.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        # PyLint doesn't recognise pydantic Field aliases
        return ZoneCorrespondencePath(  # pylint: disable=unexpected-keyword-arg
            path=translation_path,
            from_col_name=from_col,
            to_col_name=to_col,
            factors_col_name=factors_col,
        )
    return translation_path
