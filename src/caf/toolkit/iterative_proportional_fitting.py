# -*- coding: utf-8 -*-
"""Implementation of iterative proportional fitting algorithm.

See: https://en.wikipedia.org/wiki/Iterative_proportional_fitting
"""
# Built-Ins
import logging
import warnings
import itertools

from typing import Any
from typing import Callable
from typing import Optional

# Third Party
import numpy as np
import pandas as pd

# Local Imports
# pylint: disable=import-error,wrong-import-position
from caf.toolkit import math_utils
from caf.toolkit import pandas_utils as pd_utils

# pylint: enable=import-error,wrong-import-position

# # # CONSTANTS # # #
LOG = logging.getLogger(__name__)

# # # CLASSES # # #


# # # PRIVATE FUNCTIONS # # #
def _validate_seed_mat(seed_mat: np.ndarray) -> None:
    """Check whether the seed matrix is valid."""
    if not isinstance(seed_mat, np.ndarray):
        if isinstance(seed_mat, pd.DataFrame):
            raise ValueError(
                "Given `seed_mat` is a pandas.DataFrame. "
                "`ipf()` cannot handle pandas.DataFrame. Perhaps you want "
                "to call `ipf_dataframe()` instead."
            )

        raise ValueError("Given `seed_mat` is not an np.ndarray. Cannot run.")

    # Validate type
    if not np.issubdtype(seed_mat.dtype, np.number):
        raise ValueError(
            "`seed_mat` expected to be numeric type. Got "
            f"'{seed_mat.dtype}' instead."
        )


def _validate_marginals(target_marginals: list[np.ndarray]) -> None:
    """Check whether the marginals are valid."""
    # Check valid types
    invalid_dtypes = list()
    for i, marginal in enumerate(target_marginals):
        if not np.issubdtype(marginal.dtype, np.number):
            invalid_dtypes.append({
                "marginal_id": i,
                "shape": marginal.shape,
                "dtype": marginal.dtype
            })

    if len(invalid_dtypes) > 0:
        raise ValueError(
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
            invalid_dtypes.append({
                "dimension": dimension,
                "dtype": np_dimension.dtype
            })

    if len(invalid_dtypes) > 0:
        raise ValueError(
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
    target_marginals: list[np.ndarray],
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
            raise ValueError(
                "Given `seed_df` is a numpy array. "
                "`ipf_dataframe()` cannot handle numpy arrays. Perhaps you want "
                "to call `ipf()` instead."
            )

        raise ValueError("Given `seed_df` is not a pandas.DataFrame. Cannot run.")

    if value_col not in seed_df:
        raise ValueError("`value_col` is not in `seed_df`.")

    # Validate type
    if not pd.api.types.is_numeric_dtype(seed_df[value_col]):
        raise ValueError(
            "`seed_df` expected to be numeric type. Got "
            f"'{seed_df[value_col].dtype}' instead."
        )


def _validate_pd_marginals(
    target_marginals: list[pd.Series],
) -> None:
    """Check whether the pandas target marginals are valid."""
    if not all(isinstance(x, pd.Series) for x in target_marginals):
        raise ValueError(
            "`target_marginals` should be a list of pandas.Series where the "
            "index names of each series are the corresponding dimensions to "
            "control to with the marginal."
        )

    # Check valid types
    invalid_dtypes = list()
    for i, marginal in enumerate(target_marginals):
        if not pd.api.types.is_numeric_dtype(marginal.dtype):
            invalid_dtypes.append({
                "marginal_id": i,
                "controls": marginal.index.names,
                "dtype": marginal.dtype}
            )

    if len(invalid_dtypes) > 0:
        raise ValueError(
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


def pd_marginals_to_np(
    target_marginals: list[pd.Series],
    dimension_order: dict[str, list[Any]],
) -> tuple[list[np.ndarray], list[list[int]]]:
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
    """
    # Init
    dimension_col_order = list(dimension_order.keys())
    target_dimensions = [list(x.index.names) for x in target_marginals]

    # Validate dimensions
    _validate_pd_dimensions(
        seed_cols=set(dimension_col_order),
        dimension_cols=set(itertools.chain.from_iterable(target_dimensions)),
    )

    # Convert targets and dimensions to numpy format
    np_marginals = list()
    np_dimensions = list()
    for pd_marginal, pd_dimension in zip(target_marginals, target_dimensions):
        # Convert marginals
        np_marginal_i = pd_utils.dataframe_to_n_dimensional_array(
            df=pd_marginal.reset_index(),
            dimension_cols={x: dimension_order[x] for x in pd_dimension},
            fill_val=np.nan,
        )

        # Validate marginal
        if np.any(np.isnan(np_marginal_i)):
            full_idx = pd_utils.get_full_index({x: dimension_order[x] for x in pd_dimension})
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
        axes = [dimension_col_order.index(x) for x in pd_dimension]
        np_dimensions.append(axes)

    return np_marginals, np_dimensions


# # # FUNCTIONS # # #
def default_convergence(
    targets: list[np.ndarray],
    achieved: list[np.ndarray],
) -> float:
    """Calculate the default convergence used by ipfn.

    Two lists of corresponding values are zipped together, differences taken
    (residuals) and the RMSE calculated.

    Parameters
    ----------
    targets:
        A list of all the targets that `achieved` should have reached. Must
        be the same length as `achieved`.

    achieved:
        A list of all the achieved values. Must be the same length as `targets`

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


def adjust_towards_aggregates(
    mat: np.ndarray,
    target_marginals: list[np.ndarray],
    target_dimensions: list[list[int]],
    convergence_fn: Callable,
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


def ipf_dataframe(
    seed_df: pd.DataFrame,
    target_marginals: list[pd.Series],
    value_col: str,
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
    # Validate inputs
    _validate_seed_df(seed_df, value_col)
    _validate_pd_marginals(target_marginals)

    # Set and check dimension cols
    seed_dimension_cols = seed_df.columns.tolist()
    seed_dimension_cols.remove(value_col)
    dimension_order = {x: seed_df[x].unique().tolist() for x in seed_dimension_cols}

    # Convert inputs to numpy
    np_seed = pd_utils.dataframe_to_n_dimensional_array(
        df=seed_df,
        dimension_cols=dimension_order,
        fill_val=0,
    )

    np_marginals, np_dimensions = pd_marginals_to_np(
        target_marginals=target_marginals,
        dimension_order=dimension_order,
    )

    # Call numpy IPF
    fit_mat, iter_num, conv = ipf(
        seed_mat=np_seed,
        target_marginals=np_marginals,
        target_dimensions=np_dimensions,
        **kwargs,
    )

    # Fit results back into a DataFrame
    fit_df = pd_utils.n_dimensional_array_to_dataframe(
        mat=fit_mat,
        dimension_cols=dimension_order,
        value_col=value_col,
        drop_zeros=True,
    )

    return fit_df.reset_index(), iter_num, conv


def ipf(
    seed_mat: np.ndarray,
    target_marginals: list[np.ndarray],
    target_dimensions: list[list[int]],
    convergence_fn: Optional[Callable] = None,
    max_iterations: int = 5000,
    tol: float = 1e-9,
    min_tol_rate: float = 1e-9,
) -> tuple[np.ndarray, int, float]:
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
    # Validate inputs
    _validate_seed_mat(seed_mat)
    _validate_marginals(target_marginals)
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
        for iter_num in range(max_iterations):
            # Adjust matrix and log convergence changes
            prev_conv = convergence
            fit_mat, convergence = adjust_towards_aggregates(
                mat=fit_mat,
                target_marginals=target_marginals,
                target_dimensions=target_dimensions,
                convergence_fn=convergence_fn,
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
