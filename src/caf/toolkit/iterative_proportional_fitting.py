# -*- coding: utf-8 -*-
"""Implementation of iterative proportional fitting algorithm.

See: https://en.wikipedia.org/wiki/Iterative_proportional_fitting
"""
# Built-Ins
import logging
import warnings

from typing import Callable
from typing import Optional

# Third Party
import numpy as np

# Local Imports
# pylint: disable=import-error,wrong-import-position
from caf.toolkit import math_utils

# pylint: enable=import-error,wrong-import-position

# # # CONSTANTS # # #
LOG = logging.getLogger(__name__)

# # # CLASSES # # #


# # # FUNCTIONS # # #
def _validate_marginals(target_marginals: list[np.ndarray]) -> None:
    """Check whether the marginals are valid."""
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
