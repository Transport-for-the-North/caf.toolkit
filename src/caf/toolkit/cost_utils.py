# -*- coding: utf-8 -*-
"""A toolbox of useful transport cost related functionality."""
from __future__ import annotations

# Built-Ins
import logging

from typing import Optional

# Third Party
import numpy as np

# Local Imports
# pylint: disable=import-error,wrong-import-position

# pylint: enable=import-error,wrong-import-position

# # # CONSTANTS # # #
LOG = logging.getLogger(__name__)

# # # CLASSES # # #


# # # FUNCTIONS # # #
def normalised_cost_distribution(
    matrix: np.ndarray,
    cost_matrix: np.ndarray,
    min_bounds: Optional[list[float] | np.ndarray] = None,
    max_bounds: Optional[list[float] | np.ndarray] = None,
    bin_edges: Optional[list[float] | np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the normalised distribution of costs across a matrix.

    Parameters
    ----------
    matrix:
        The matrix to calculate the cost distribution for. This matrix
        should be the same shape as cost_matrix

    cost_matrix:
        A matrix of cost relating to matrix. This matrix
        should be the same shape as matrix

    min_bounds:
        A list of minimum bounds for each edge of a distribution band.
        Corresponds to max_bounds.

    max_bounds:
        A list of maximum bounds for each edge of a distribution band.
        Corresponds to min_bounds.

    bin_edges:
        Defines a monotonically increasing array of bin edges, including the
        rightmost edge, allowing for non-uniform bin widths. This argument
        is passed straight into `numpy.histogram`

    Returns
    -------
    cost_distribution:
        A numpy array of the sum of trips by distance band.

    normalised_cost_distribution:
        Similar to `cost_distribution`, however the values in each band
        have been normalised to sum to 1.

    See Also
    --------
    `numpy.histogram`
    `cost_distribution`
    """
    distribution = cost_distribution(
        matrix=matrix,
        cost_matrix=cost_matrix,
        min_bounds=min_bounds,
        max_bounds=max_bounds,
        bin_edges=bin_edges,
    )

    # Normalise
    if distribution.sum() == 0:
        normalised = np.zeros_like(distribution)
    else:
        normalised = distribution / distribution.sum()

    return distribution, normalised


def cost_distribution(
    matrix: np.ndarray,
    cost_matrix: np.ndarray,
    min_bounds: Optional[list[float] | np.ndarray] = None,
    max_bounds: Optional[list[float] | np.ndarray] = None,
    bin_edges: Optional[list[float] | np.ndarray] = None,
) -> np.ndarray:
    """
    Calculate the distribution of costs across a matrix.

    Parameters
    ----------
    matrix:
        The matrix to calculate the cost distribution for. This matrix
        should be the same shape as cost_matrix

    cost_matrix:
        A matrix of cost relating to matrix. This matrix
        should be the same shape as matrix

    min_bounds:
        A list of minimum bounds for each edge of a distribution band.
        Corresponds to max_bounds.

    max_bounds:
        A list of maximum bounds for each edge of a distribution band.
        Corresponds to min_bounds.

    bin_edges:
        Defines a monotonically increasing array of bin edges, including the
        rightmost edge, allowing for non-uniform bin widths. This argument
        is passed straight into `numpy.histogram`

    Returns
    -------
    cost_distribution:
        A numpy array of the sum of trips by distance band.

    See Also
    --------
    `numpy.histogram`
    `normalised_cost_distribution`
    """
    # Use bounds to calculate bin edges
    if bin_edges is None:
        if min_bounds is None or max_bounds is None:
            raise ValueError(
                "Either `bin_edges` needs to be set, or both `min_bounds` and "
                "`max_bounds` needs to be set."
            )
        bin_edges = [min_bounds[0]] + list(max_bounds)

    distribution, _ = np.histogram(
        a=cost_matrix,
        bins=bin_edges,
        weights=matrix,
    )
    return distribution
