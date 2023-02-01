# -*- coding: utf-8 -*-
"""Implementation of iterative proportional fitting algorithm.

See: https://en.wikipedia.org/wiki/Iterative_proportional_fitting
"""
# Built-Ins
import logging
import warnings
import itertools

from typing import Any
from typing import Union
from typing import overload
from typing import Callable
from typing import Optional
from typing import Collection

# Third Party
import tqdm
import sparse
import numpy as np
import pandas as pd

# Local Imports
# pylint: disable=import-error,wrong-import-position
from caf.toolkit import math_utils
from caf.toolkit import array_utils
from caf.toolkit import pandas_utils as pd_utils

# pylint: enable=import-error,wrong-import-position

# # # CONSTANTS # # #
LOG = logging.getLogger(__name__)
IpfConvergenceFn = Callable[[Collection[np.ndarray], Collection[np.ndarray]], float]

# # # CLASSES # # #


# # # FUNCTIONS # # #
# ## Private Functions ## #
def _validate_seed_mat(seed_mat: Union[np.ndarray, sparse.COO], use_sparse: bool) -> None:
    """Check whether the seed matrix is valid."""
    if isinstance(seed_mat, pd.DataFrame):
        raise TypeError(
            "Given `seed_mat` is a pandas.DataFrame. "
            "`ipf()` cannot handle pandas.DataFrame. Perhaps you want "
            "to call `ipf_dataframe()` instead."
        )

    if use_sparse:
        if not isinstance(seed_mat, sparse.COO):
            raise TypeError(
                "Given `seed_mat` is not a sparse.COO matrix when 'use_sparse=True`. Cannot run."
            )
    else:
        if not isinstance(seed_mat, np.ndarray):
            raise TypeError("Given `seed_mat` is not an np.ndarray. Cannot run.")

    # Validate type
    if not np.issubdtype(seed_mat.dtype, np.number):
        raise TypeError(
            "`seed_mat` expected to be numeric type. Got " f"'{seed_mat.dtype}' instead."
        )


def _validate_marginals(
    target_marginals: list[Union[np.ndarray, sparse.COO]], use_sparse: bool
) -> None:
    """Check whether the marginals are valid."""
    # Check valid types
    if use_sparse:
        type_name_msg = "sparse.COO or np.array matrix when seed matrix is sparse"
        valid_types: Union[type, tuple[type, type]] = (np.ndarray, sparse.COO)
    else:
        type_name_msg = "np.ndarray"
        valid_types = np.ndarray

    invalid_types = list()
    for i, marginal in enumerate(target_marginals):
        if not isinstance(marginal, valid_types):
            invalid_types.append({"marginal_id": i, "type": type(marginal)})

    if len(invalid_types) > 0:
        raise TypeError(
            f"Marginals are expected to be {type_name_msg}."
            "Got the following invalid types:\n"
            f"{pd.DataFrame(invalid_types)}"
        )

    # Check valid data types
    invalid_dtypes = list()
    for i, marginal in enumerate(target_marginals):
        if not np.issubdtype(marginal.dtype, np.number):
            invalid_dtypes.append(
                {"marginal_id": i, "shape": marginal.shape, "dtype": marginal.dtype}
            )

    if len(invalid_dtypes) > 0:
        raise TypeError(
            "Marginals are expected to be numeric type. "
            "Got the following non-numeric types:\n"
            f"{pd.DataFrame(invalid_dtypes)}"
        )

    # Check sums
    marginal_sums = [x.sum() for x in target_marginals]
    if not math_utils.list_is_almost_equal(marginal_sums):
        warnings.warn(
            "Given target_marginals do not sum to similar amounts. The "
            "resulting matrix may not be very accurate.\n"
            f"Sums of given marginals: {marginal_sums}"
        )


def _validate_dimensions(
    target_dimensions: list[list[int]],
    seed_mat: np.ndarray,
) -> None:
    """Check whether the dimensions are valid."""
    # Check valid types
    invalid_dtypes = list()
    for dimension in target_dimensions:
        np_dimension = np.array(dimension)
        if not np.issubdtype(np_dimension.dtype, np.number):
            invalid_dtypes.append({"dimension": dimension, "dtype": np_dimension.dtype})

    if len(invalid_dtypes) > 0:
        raise TypeError(
            "Dimensions are expected to be numeric type. "
            "Got the following non-numeric types:\n"
            f"{pd.DataFrame(invalid_dtypes)}"
        )

    # Valid axis numbers
    seed_n_dims = len(seed_mat.shape)
    for dimension in target_dimensions:
        if len(dimension) > seed_n_dims:
            raise ValueError(
                "Too many dimensions. "
                "Cannot have more target dimensions than there are dimensions "
                f"in the seed matrix. Expected a maximum of {seed_n_dims} "
                f"dimensions, instead got a target of: {dimension}."
            )
        if np.max(dimension) > seed_n_dims - 1:
            raise ValueError(
                "Dimension numbers too high. "
                "Cannot have a higher axis number than is available in the "
                f"seed matrix. Expected a maximum axis number of "
                f"{seed_n_dims - 1}, got {np.max(dimension)} instead."
            )


def _validate_marginal_shapes(
    target_marginals: list[Union[np.ndarray, sparse.COO]],
    target_dimensions: list[list[int]],
    seed_mat: np.ndarray,
) -> None:
    """Check whether the marginal shapes are valid."""
    seed_shape = seed_mat.shape
    for marginal, dimensions in zip(target_marginals, target_dimensions):
        target_shape = tuple(np.array(seed_shape)[dimensions])
        if marginal.shape != target_shape:
            raise ValueError(
                "Marginal is not the expected shape for the given seed "
                f"matrix. Marginal with dimensions {dimensions} is expected "
                f"to have shape {target_shape}. Instead, got shape "
                f"{marginal.shape}."
            )


def _validate_seed_df(
    seed_df: pd.DataFrame,
    value_col: str,
) -> None:
    """Check whether the seed_df and value_col are valid."""
    if not isinstance(seed_df, pd.DataFrame):
        if isinstance(seed_df, np.ndarray):
            raise TypeError(
                "Given `seed_df` is a numpy array. "
                "`ipf_dataframe()` cannot handle numpy arrays. Perhaps you want "
                "to call `ipf()` instead."
            )

        raise TypeError("Given `seed_df` is not a pandas.DataFrame. Cannot run.")

    if value_col not in seed_df:
        raise ValueError("`value_col` is not in `seed_df`.")

    # Validate type
    if not pd.api.types.is_numeric_dtype(seed_df[value_col]):
        raise TypeError(
            "`seed_df` expected to be numeric type. Got "
            f"'{seed_df[value_col].dtype}' instead."
        )


def _validate_pd_marginals(
    target_marginals: list[pd.Series],
) -> None:
    """Check whether the pandas target marginals are valid."""
    if not all(isinstance(x, pd.Series) for x in target_marginals):
        raise TypeError(
            "`target_marginals` should be a list of pandas.Series where the "
            "index names of each series are the corresponding dimensions to "
            "control to with the marginal."
        )

    # Check valid types
    invalid_dtypes = list()
    for i, marginal in enumerate(target_marginals):
        if not pd.api.types.is_numeric_dtype(marginal.dtype):
            invalid_dtypes.append(
                {"marginal_id": i, "controls": marginal.index.names, "dtype": marginal.dtype}
            )

    if len(invalid_dtypes) > 0:
        raise TypeError(
            "Marginals are expected to be numeric types. Try using "
            "`pd.to_numeric()` to cast the marginals to the correct types. "
            "Got the following non-numeric types:\n"
            f"{pd.DataFrame(invalid_dtypes)}"
        )


def _validate_pd_dimensions(seed_cols: set[str], dimension_cols: set[str]) -> None:
    """Check whether the pandas target dimension columns are valid."""
    missing_cols = dimension_cols - seed_cols
    if len(missing_cols) > 0:
        raise ValueError(
            "Not all dimension control columns defined in `target_marginals` "
            "can be found in the `seed_df`. The following columns are "
            f"missing:\n{missing_cols}"
        )


def _infill_invalid_combos(
    marginal: np.ndarray,
    dimension_order: dict[str, list[Any]],
    valid_dimension_combos: pd.DataFrame,
    fill_val: float = 0,
) -> np.ndarray:
    """Infill invalid combinations of dimension values of a marginal."""
    valid_zeros = valid_dimension_combos[dimension_order.keys()].drop_duplicates()
    valid_zeros["val"] = 0
    invalid_ones, _ = pd_utils.dataframe_to_n_dimensional_array(
        df=valid_zeros,
        dimension_cols=dimension_order,
        fill_val=1,
    )
    return np.where(invalid_ones, fill_val, marginal)


def _ipf_dense_mat_result_to_df(
    results_array: np.ndarray,
    dimension_cols: dict[str, list[Any]],
    value_col: str,
    valid_dimension_combos: pd.DataFrame,
    drop_zeros_return: bool,
) -> pd.DataFrame:
    """Convert an np.ndarray ipf result back into a dataframe."""
    fit_df = pd_utils.n_dimensional_array_to_dataframe(
        mat=results_array,
        dimension_cols=dimension_cols,
        value_col=value_col,
    )

    col_names = valid_dimension_combos.columns.to_list()
    input_index = pd.MultiIndex.from_arrays(
        [valid_dimension_combos[x].values for x in col_names],
        names=col_names,
    )

    # Make sure no demand was dropped when fitting back in dataframe
    pre_convert_total = fit_df.values.sum()
    fit_df = fit_df.reindex(input_index)
    post_convert_total = fit_df.values.sum()
    if not math_utils.is_almost_equal(pre_convert_total, post_convert_total):
        raise RuntimeError(
            "When converting the resulting ipf matrix back into the `seed_df` "
            "format some values were dropped. This shouldn't happen and is an "
            "internal error."
        )

    # Drop any zeros
    if drop_zeros_return:
        zero_mask = fit_df[value_col] == 0
        fit_df = fit_df[~zero_mask].copy()

    return fit_df.reset_index()


def _ipf_sparse_mat_result_to_df(
    results_array: sparse.COO,
    value_maps: dict[str, dict[Any, int]],
    value_col: str,
    valid_dimension_combos: pd.DataFrame,
    drop_zeros_return: bool,
) -> pd.DataFrame:
    # pylint: disable=too-many-locals
    # Init
    dimension_col_names = valid_dimension_combos.columns.to_list()
    input_index = pd.MultiIndex.from_arrays(
        [valid_dimension_combos[x].values for x in dimension_col_names],
        names=dimension_col_names,
    )

    # Convert the sparse array to a dataframe
    df_ph = dict(zip(dimension_col_names, results_array.coords))
    df_ph[value_col] = results_array.data
    fit_df = pd.DataFrame(df_ph)
    pre_convert_total = fit_df[value_col].values.sum()

    # Map back to original values
    rev_value_maps = dict()
    for col, mapping in value_maps.items():
        rev_value_maps[col] = {v: k for k, v in mapping.items()}

    for col, value_map in rev_value_maps.items():
        fit_df[col] = fit_df[col].map(value_map)
    fit_df = fit_df.set_index(dimension_col_names)

    # Sparse matrices would have optimised 0 values away - infill back
    fit_df = fit_df.reindex(input_index).fillna(0)

    # Make sure no demand was dropped when fitting back in dataframe
    post_convert_total = fit_df.values.sum()
    if not math_utils.is_almost_equal(pre_convert_total, post_convert_total):
        raise RuntimeError(
            "When converting the resulting ipf matrix back into the `seed_df` "
            "format some values were dropped. This shouldn't happen and is an "
            "internal error.\n"
            f"{pre_convert_total = }\n"
            f"{post_convert_total = }\n"
        )

    # Drop any zeros
    if drop_zeros_return:
        zero_mask = fit_df[value_col] == 0
        fit_df = fit_df[~zero_mask].copy()

    return fit_df.reset_index()


# ## Public Functions ## #
def default_convergence(
    targets: Collection[np.ndarray],
    achieved: Collection[np.ndarray],
) -> float:
    """Calculate the default convergence used by ipfn.

    Two lists of corresponding values are zipped together, differences taken
    (residuals) and the RMSE calculated.

    Parameters
    ----------
    targets:
        The targets that `achieved` should have reached. Must
        be the same length as `achieved`.

    achieved:
        The achieved values. Must be the same length as `targets`

    Returns
    -------
    ipfn_convergence:
        A float value indicating the max convergence value achieved across
        residuals

    Raises
    ------
    ValueError:
        If `targets` and `achieved` are not the same length
    """
    if len(targets) != len(achieved):
        raise ValueError(
            "targets and achieved must be the same length. "
            f"targets length: {len(targets)}, achieved length: {len(achieved)}"
        )

    max_conv = 0
    for target, ach in zip(targets, achieved):
        conv = np.max(abs((ach / target) - 1))
        if conv > max_conv:
            max_conv = conv

    return max_conv


def pd_marginals_to_np(
    target_marginals: list[pd.Series],
    dimension_order: dict[str, list[Any]],
    valid_dimension_combos: pd.DataFrame,
    allow_sparse: bool = False,
    sparse_value_maps: Optional[dict[Any, dict[Any, int]]] = None,
) -> tuple[Union[list[np.ndarray], list[sparse.COO]], list[list[int]]]:
    """Convert pandas marginals to numpy format for `ipf()`.

    Parameters
    ----------
    target_marginals:
        A list of the aggregates to adjust `seed_df` towards. Aggregates are
        the target values to aim for when aggregating across one or several
        other axis. Each item should be a pandas.Series where the index names
        relate to the dimensions to control `seed_df` to.
        The index names relate to `target_dimensions` in `ipf()`. See there for
        more information

    dimension_order:
        A dictionary of `{col_name: col_values}` pairs. `dimension_cols.keys()`
        MUST return a list of keys in the same order as the seed matrix for
        this function to be accurate. `dimension_cols.keys()` is defined
        by the order the keys are added to a dictionary. `col_values` MUST
        be in the same order as the values in the dimension they refer to.
        The values are used to ensure the returned marginals are in the correct
        order.

    valid_dimension_combos:
        A dataframe defining all the valid combinations of each of the
        dimension values. This should be taken from the seed matrix. Used
        internally to ensure the generated numpy marginals are valid with
        no unexpected missing combinations.

    allow_sparse:
        Whether to allow the resultant marginals to become sparse.COO matrices.
        Usually used when the corresponding seed matrix is also sparse.
        If set to False, then the resultant marginals will not be allowed
        to be sparse and MUST be dense numpy matrices.

    sparse_value_maps:
        A nested dictionary of `{col_name: {col_val: coordinate_value}}` where
        `col_name` is the name of the column in `df`, `col_val` is the
        value in `col_name`, and `coordinate_value` is the coordinate value
        to assign to that value in the sparse array.

    Returns
    -------
    target_marginals:
        A list of the aggregates to adjust `matrix` towards. Aggregates are
        the target values to aim for when aggregating across one or several
        other axis. Directly corresponds to `target_dimensions`.

    target_dimensions:
        A list of target dimensions for each aggregate.
        Each target dimension lists the axes that should be preserved when
        calculating the achieved aggregates for the corresponding
        `target_marginals`.
        Another way to look at this is a list of the numpy axis which should
        NOT be summed from `mat` when calculating the achieved marginals.

    Raises
    ------
    ValueError:
        If any of the marginal index names do not exist in the keys of
        `dimension_order`.

    ValueError:
        If the passed in marginals do not contain all the valid combinations
        of `dimension_cols`, as defined in `valid_dimension_combos`
    """
    # pylint: disable=too-many-locals
    # Init
    dimension_col_order = list(dimension_order.keys())
    target_dimensions = [list(x.index.names) for x in target_marginals]

    # Validate inputs
    _validate_pd_dimensions(
        seed_cols=set(dimension_col_order),
        dimension_cols=set(itertools.chain.from_iterable(target_dimensions)),
    )

    # Convert targets and dimensions to numpy format
    np_marginals = list()
    np_dimensions = list()
    for pd_marginal, pd_dimension in zip(target_marginals, target_dimensions):
        # Init
        pd_dimension_ordered = [x for x in dimension_order if x in pd_dimension]
        dimension_cols_i = {x: dimension_order[x] for x in pd_dimension_ordered}

        # Convert marginals
        sparse_ok = "force" if allow_sparse else "disallow"
        # TODO(BT): Not sure what the overload error is here.
        np_marginal_i, _ = pd_utils.dataframe_to_n_dimensional_array(  # type: ignore
            df=pd_marginal.reset_index(),
            dimension_cols=dimension_cols_i,
            sparse_ok=sparse_ok,
            sparse_value_maps=sparse_value_maps,
            fill_val=np.nan,
        )

        # Remove any NaN where the combo isn't actually valid
        if np.any(np.isnan(np_marginal_i)):
            np_marginal_i = _infill_invalid_combos(
                marginal=np_marginal_i,
                dimension_order=dimension_cols_i,
                valid_dimension_combos=valid_dimension_combos,
                fill_val=0,
            )

        # Check all valid combinations have been accounted for
        if np.any(np.isnan(np_marginal_i)):
            full_idx = pd_utils.get_full_index(dimension_cols_i)
            err_df = pd.DataFrame(
                data=np_marginal_i.flatten(),
                index=full_idx,
                columns=["Value"],
            )
            err_df = err_df[np.isnan(err_df["Value"].values)]
            raise ValueError(
                "Not all seed matrix dimensions were given in a marginal. See "
                f"np.NaN below for missing values:\n{err_df}"
            )
        np_marginals.append(np_marginal_i)

        # Convert the dimensions
        axes = [dimension_col_order.index(x) for x in pd_dimension_ordered]
        np_dimensions.append(axes)

    return np_marginals, np_dimensions


def adjust_towards_aggregates(
    mat: np.ndarray,
    target_marginals: list[np.ndarray],
    target_dimensions: list[list[int]],
    convergence_fn: IpfConvergenceFn,
) -> tuple[np.ndarray, float]:
    """Adjust a matrix towards aggregate targets.

    Uses `target_aggregates` and `target_dimensions` to calculate adjustment
    factors across each of the dimensions, brining mat closer to the targets.

    Parameters
    ----------
    mat:
        The starting matrix that should be adjusted

    target_marginals:
        A list of the aggregates to adjust `matrix` towards. Aggregates are
        the target values to aim for when aggregating across one or several
        other axis. Directly corresponds to `target_dimensions`.

    target_dimensions:
        A list of target dimensions for each aggregate.
        Each target dimension lists the axes that should be preserved when
        calculating the achieved aggregates for the corresponding
        `target_marginals`.
        Another way to look at this is a list of the numpy axis which should
        NOT be summed from `mat` when calculating the achieved marginals.

    convergence_fn:
        The function that should be called to calculate the convergence of
        `mat` after all `target_marginals` adjustments have been made. If
        a callable is given it must take the form:
        `fn(targets: list[np.ndarray], achieved: list[np.ndarray])`

    Returns
    -------
    adjusted_mat:
        The input mat, adjusted once for each aggregate towards the
        `target_marginals`

    convergence:
        A float describing the convergence of `adjusted_mat` to
        `target_marginals`. Usually lower is better, but that depends on the
        exact `convergence_fn` in use.
    """
    # Init
    n_dims = len(mat.shape)
    out_mat = mat.copy().astype(float)

    # Adjust the matrix once for each marginal
    for target, dimensions in zip(target_marginals, target_dimensions):
        # Figure out which axes to sum across
        sum_axes = tuple(set(range(n_dims)) - set(dimensions))

        # Figure out the adjustment factor
        achieved = out_mat.sum(axis=sum_axes)
        adj_factor = np.divide(
            target,
            achieved,
            where=achieved != 0,
            out=np.ones_like(target, dtype=float),
        )

        # Apply factors
        adj_factor = np.broadcast_to(
            np.expand_dims(adj_factor, axis=sum_axes),
            mat.shape,
        )
        out_mat *= adj_factor

    # Calculate the achieved marginals
    achieved_aggregates = list()
    for dimensions in target_dimensions:
        sum_axes = tuple(set(range(n_dims)) - set(dimensions))
        achieved_aggregates.append(out_mat.sum(axis=sum_axes))

    return out_mat, convergence_fn(target_marginals, achieved_aggregates)


# Sparse doesn't handle DIV/0 very well
# We could use an infill value, but it's quicker to just ignore the errors
@np.errstate(divide="ignore", invalid="ignore")
def sparse_adjust_towards_aggregates(
    mat: sparse.COO,
    target_marginals: list[sparse.COO],
    target_dimensions: list[list[int]],
    convergence_fn: IpfConvergenceFn,
) -> tuple[np.ndarray, float]:
    """Adjust a matrix towards aggregate targets.

    Uses `target_aggregates` and `target_dimensions` to calculate adjustment
    factors across each of the dimensions, brining mat closer to the targets.

    Parameters
    ----------
    mat:
        The starting matrix that should be adjusted

    target_marginals:
        A list of the aggregates to adjust `matrix` towards. Aggregates are
        the target values to aim for when aggregating across one or several
        other axis. Directly corresponds to `target_dimensions`.

    target_dimensions:
        A list of target dimensions for each aggregate.
        Each target dimension lists the axes that should be preserved when
        calculating the achieved aggregates for the corresponding
        `target_marginals`.
        Another way to look at this is a list of the numpy axis which should
        NOT be summed from `mat` when calculating the achieved marginals.

    convergence_fn:
        The function that should be called to calculate the convergence of
        `mat` after all `target_marginals` adjustments have been made. If
        a callable is given it must take the form:
        `fn(targets: list[np.ndarray], achieved: list[np.ndarray])`

    Returns
    -------
    adjusted_mat:
        The input mat, adjusted once for each aggregate towards the
        `target_marginals`

    convergence:
        A float describing the convergence of `adjusted_mat` to
        `target_marginals`. Usually lower is better, but that depends on the
        exact `convergence_fn` in use.
    """
    # Init
    n_dims = len(mat.shape)
    out_mat = mat.copy().astype(float)

    # Adjust the matrix once for each marginal
    for target, dimensions in zip(target_marginals, target_dimensions):
        # Figure out which axes to sum across
        sum_axes = tuple(set(range(n_dims)) - set(dimensions))

        # Figure out the adjustment factor
        if isinstance(out_mat, sparse.COO):
            achieved = array_utils.sparse_sum(sparse_array=out_mat, axis=sum_axes)
        else:
            achieved = out_mat.sum(axis=sum_axes)
        adj_factor = target / achieved
        adj_factor = array_utils.remove_sparse_nan_values(adj_factor)

        # Apply factors
        adj_factor = array_utils.broadcast_sparse_matrix(
            array=adj_factor,
            target_array=out_mat,
            array_dims=dimensions,
        )
        out_mat *= adj_factor

    # Calculate the achieved marginals
    achieved_aggregates = list()
    for dimensions in target_dimensions:
        sum_axes = tuple(set(range(n_dims)) - set(dimensions))
        achieved_aggregates.append(array_utils.sparse_sum(sparse_array=out_mat, axis=sum_axes))

    return out_mat, convergence_fn(target_marginals, achieved_aggregates)


def ipf_dataframe(
    seed_df: pd.DataFrame,
    target_marginals: list[pd.Series],
    value_col: str,
    drop_zeros_return: bool = False,
    force_sparse: bool = False,
    **kwargs,
) -> tuple[pd.DataFrame, int, float]:
    """Adjust a matrix iteratively towards targets until convergence met.

    This is a pandas wrapper of ipf
    https://en.wikipedia.org/wiki/Iterative_proportional_fitting

    Parameters
    ----------
    seed_df:
        The starting pandas.DataFrame that should be adjusted.

    target_marginals:
        A list of the aggregates to adjust `seed_df` towards. Aggregates are
        the target values to aim for when aggregating across one or several
        other axis. Each item should be a pandas.Series where the index names
        relate to the dimensions to control `seed_df` to.
        The index names relate to `target_dimensions` in `ipf()`. See there for
        more information

    value_col:
        The column in `seed_df` that refers to the data. All other columns
        will be assumed to be dimensional columns.

    drop_zeros_return:
        Whether to drop any rows of the dataframe that contain 0 values on
        return or now. If False, the return dataframe will be in the same
        order as `seed_df`. That is, the return will be exactly the same as
        `seed_df` except in the `value_col` column.

    force_sparse:
        Whether to force the dataframe into a sparse array without first
        checking if the dense array would fit into memory.

    **kwargs:
        Any other arguments to pass to `iterative_proportional_fitting.ipf()`

    Returns
    -------
    fit_df:
        The final fit matrix, converted back to a DataFrame.

    completed_iterations:
        The number of completed iterations before exiting.

    achieved_convergence:
        The final achieved convergence - achieved by `fit_matrix`

    Raises
    ------
    ValueError:
        If any of the marginals or dimensions are not valid when passed in.

    See Also
    --------
    `iterative_proportional_fitting.ipf()`
    """
    # pylint: disable=too-many-locals
    # Validate inputs
    _validate_seed_df(seed_df, value_col)
    _validate_pd_marginals(target_marginals)

    # Set and check dimension cols
    seed_dimension_cols = seed_df.columns.tolist()
    seed_dimension_cols.remove(value_col)
    dimension_order = {x: seed_df[x].unique().tolist() for x in seed_dimension_cols}

    # Get a df of all the valid combinations of dimensions
    valid_dimension_combos = seed_df[seed_dimension_cols].copy()

    # Convert inputs to numpy
    np_seed, value_maps = pd_utils.dataframe_to_n_dimensional_array(
        df=seed_df,
        dimension_cols=dimension_order,
        fill_val=0,
        sparse_ok="force" if force_sparse else "allow",
    )

    np_marginals, np_dimensions = pd_marginals_to_np(
        target_marginals=target_marginals,
        dimension_order=dimension_order,
        valid_dimension_combos=valid_dimension_combos,
        allow_sparse=isinstance(np_seed, sparse.COO),
        sparse_value_maps=value_maps,
    )

    # Call numpy IPF
    fit_mat, iter_num, conv = ipf(
        seed_mat=np_seed,
        target_marginals=np_marginals,
        target_dimensions=np_dimensions,
        **kwargs,
    )

    # Fit results back into starting dataframe shape
    if isinstance(fit_mat, sparse.COO):
        fit_df = _ipf_sparse_mat_result_to_df(
            results_array=fit_mat,
            value_maps=value_maps,
            value_col=value_col,
            valid_dimension_combos=valid_dimension_combos,
            drop_zeros_return=drop_zeros_return,
        )
    else:
        fit_df = _ipf_dense_mat_result_to_df(
            results_array=fit_mat,
            dimension_cols=dimension_order,
            value_col=value_col,
            valid_dimension_combos=valid_dimension_combos,
            drop_zeros_return=drop_zeros_return,
        )

    return fit_df, iter_num, conv


@overload
def ipf(
    seed_mat: np.ndarray,
    target_marginals: list[np.ndarray],
    target_dimensions: list[list[int]],
    convergence_fn: Optional[IpfConvergenceFn] = ...,
    max_iterations: int = ...,
    tol: float = ...,
    min_tol_rate: float = ...,
    use_sparse: bool = ...,
    show_pbar: bool = ...,
    pbar_kwargs: Optional[dict[str, Any]] = ...,
) -> tuple[np.ndarray, int, float]:
    ...  # pragma: no cover


@overload
def ipf(
    seed_mat: sparse.COO,
    target_marginals: list[Union[np.ndarray, sparse.COO]],
    target_dimensions: list[list[int]],
    convergence_fn: Optional[IpfConvergenceFn] = ...,
    max_iterations: int = ...,
    tol: float = ...,
    min_tol_rate: float = ...,
    use_sparse: bool = ...,
    show_pbar: bool = ...,
    pbar_kwargs: Optional[dict[str, Any]] = ...,
) -> tuple[sparse.COO, int, float]:
    ...  # pragma: no cover


def ipf(
    seed_mat: Union[np.ndarray, sparse.COO],
    target_marginals: list[Union[np.ndarray, sparse.COO]],
    target_dimensions: list[list[int]],
    convergence_fn: Optional[IpfConvergenceFn] = None,
    max_iterations: int = 5000,
    tol: float = 1e-9,
    min_tol_rate: float = 1e-9,
    use_sparse: bool = False,
    show_pbar: bool = False,
    pbar_kwargs: Optional[dict[str, Any]] = None,
) -> tuple[Union[np.ndarray, sparse.COO], int, float]:
    """Adjust a matrix iteratively towards targets until convergence met.

    https://en.wikipedia.org/wiki/Iterative_proportional_fitting

    Parameters
    ----------
    seed_mat:
        The starting matrix that should be adjusted.

    target_marginals:
        A list of the aggregates to adjust `matrix` towards. Aggregates are
        the target values to aim for when aggregating across one or several
        other axis. Directly corresponds to `target_dimensions`.

    target_dimensions:
        A list of target dimensions for each aggregate.
        Each target dimension lists the axes that should be preserved when
        calculating the achieved aggregates for the corresponding
        `target_marginals`.
        Another way to look at this is a list of the numpy axis which should
        NOT be summed from `mat` when calculating the achieved marginals.

    convergence_fn:
        The function that should be called to calculate the convergence of
        `mat` after all `target_marginals` adjustments have been made. If
        a callable is given it must take the form:
        `fn(targets: list[np.ndarray], achieved: list[np.ndarray])`

    max_iterations:
        The maximum number of iterations to complete before exiting

    tol:
        The target convergence to achieve before exiting early. This is one
        condition which allows exiting before `max_iterations` is reached.
        The convergence is calculated via `convergence_fn`.

    min_tol_rate:
        The minimum value that the convergence can change by between
        iterations before exiting early. This is one
        condition which allows exiting before `max_iterations` is reached.
        The convergence is calculated via `convergence_fn`.

    use_sparse:
        Whether to use sparse matrices when doing the ipf calculations. This
        is useful then the given numpy array is very sparse. If a sparse.COO
        matrix is given, this argument is ignored.

    show_pbar:
        Whether to show a progress bar of the current ipf iterations or not.
        If `pbar_kwargs` is not None, then this argument is ignored.
        If `pbar_kwargs` is set, and you want to disable it, add
        `{"disable": True}` as a kwarg.

    pbar_kwargs:
        A dictionary of keyword arguments to pass into a progress bar.
        This dictionary is passed into `tqdm.tqdm(**pbar_kwargs)` when
        building the progress bar.

    Returns
    -------
    fit_matrix:
        The final fit matrix.

    completed_iterations:
        The number of completed iterations before exiting.

    achieved_convergence:
        The final achieved convergence - achieved by `fit_matrix`

    Raises
    ------
    ValueError:
        If any of the marginals or dimensions are not valid when passed in.
    """
    # pylint: disable=too-many-locals
    # pylint: disable=too-many-branches
    # Init
    if pbar_kwargs is None and not show_pbar:
        pbar_kwargs = {"disable": True}
    elif pbar_kwargs is None and show_pbar:
        pbar_kwargs = {"desc": "IPF Iterations"}

    if isinstance(seed_mat, sparse.COO):
        use_sparse = True

    # Validate inputs
    _validate_seed_mat(seed_mat, use_sparse)
    _validate_marginals(target_marginals, use_sparse)
    _validate_dimensions(target_dimensions, seed_mat)
    _validate_marginal_shapes(target_marginals, target_dimensions, seed_mat)

    if convergence_fn is None:
        convergence_fn = math_utils.root_mean_squared_error

    # Initialise variables for iterations
    iter_num = -1
    convergence = np.inf
    fit_mat = seed_mat.copy()
    early_exit = False

    # Can return early if all 0 - probably shouldn't happen!
    if all(x.sum() == 0 for x in target_marginals):
        warnings.warn("Given target_marginals of 0. Returning all 0's")
        return np.zeros(seed_mat.shape), iter_num, convergence

    # Set up numpy overflow errors
    with np.errstate(over="raise"):
        # Iteratively fit
        iterator = tqdm.tqdm(range(max_iterations), **pbar_kwargs)
        for iter_num in iterator:
            # Adjust matrix and log convergence changes
            prev_conv = convergence

            if isinstance(seed_mat, np.ndarray):
                fit_mat, convergence = adjust_towards_aggregates(
                    mat=fit_mat,
                    target_marginals=target_marginals,
                    target_dimensions=target_dimensions,
                    convergence_fn=convergence_fn,
                )
            elif isinstance(seed_mat, sparse.COO):
                fit_mat, convergence = sparse_adjust_towards_aggregates(
                    mat=fit_mat,
                    target_marginals=target_marginals,
                    target_dimensions=target_dimensions,
                    convergence_fn=convergence_fn,
                )
            else:
                raise RuntimeError(
                    "Apparently `seed_mat` is a valid type, but I don't know "
                    "how to adjust it! This is an internal error that needs "
                    "fixing."
                )

            # Check if we've hit targets
            if convergence < tol:
                early_exit = True
                break

            if iter_num > 1 and abs(convergence - prev_conv) < min_tol_rate:
                early_exit = True
                break

            # Check for errors
            if np.isnan(convergence):
                return np.zeros(seed_mat.shape), iter_num + 1, np.inf

    # Warn the user if we exhausted our number of loops
    if not early_exit:
        warnings.warn(
            f"The iterative proportional fitting exhausted its max "
            f"number of loops ({max_iterations}), while achieving a "
            f"convergence value of {convergence}. The values returned may "
            f"not be accurate."
        )

    return fit_mat, iter_num + 1, convergence
