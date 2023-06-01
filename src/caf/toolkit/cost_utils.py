# -*- coding: utf-8 -*-
"""A toolbox of useful transport cost related functionality."""
from __future__ import annotations

# Built-Ins
import logging

from typing import Optional
from typing import TYPE_CHECKING

# Third Party
import pydantic
import numpy as np
import pandas as pd

# Local Imports
# pylint: disable=import-error,wrong-import-position
from caf.toolkit import pandas_utils as pd_utils
# pylint: enable=import-error,wrong-import-position

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from pydantic.dataclasses import dataclass

# # # CONSTANTS # # #
LOG = logging.getLogger(__name__)


# # # CLASSES # # #
@dataclass
class CostDistribution:
    """Distribution of cost values between variable bounds

    Parameters
    ----------
    df:
        The raw Pandas DataFrame containing the cost distribution data.

    min_vals:
        Minimum values of the cost distribution bin edges.

    max_vals:
        Maximum values of the cost distribution bin edges.

    bin_edges:
        Bin edges for the cost distribution.

    avg_vals:
        Average values for each of the cost distribution bins.

    trip_vals:
        Trip values for each of the cost distribution bins.

    band_share_vals:
        Band share values for each of the cost distribution bins.
    """

    df: pd.DataFrame

    # Default arguments
    min_col: str = "min"
    max_col: str = "max"
    avg_col: str = "trips"
    trips_col: str = "ave_km"

    # Ideas
    units: str = "km"

    @pydantic.root_validator
    def check_df_col_names(self, values):
        """Check the given columns are in the given dataframe"""
        # init
        col_names = ["min_col", "max_col", "avg_col", "trips_col"]
        cols = {k: v for k, v in values.items() if k in col_names}
        df = values.get("df")

        # Check columns are in df
        err_cols = {}
        for col_name, col_val in cols:
            if col_val not in df:
                err_cols.update({col_name: col_val})

        # Throw error if missing columns found
        if err_cols != dict():
            raise ValueError(
                "Not all of the given column names exist in the given df. "
                f"The following columns are missing:{err_cols}\n"
                f"With the following in the Df: {df.columns}"
            )

        # Tidy up df
        values["df"] = pd_utils.reindex_cols(df, cols)
        return values

    @property
    def min_vals(self) -> np.ndarray:
        """Minimum values of the cost distribution bin edges."""
        return self.df[self.min_col].values

    @property
    def max_vals(self) -> np.ndarray:
        """Maximum values of the cost distribution in edges."""
        return self.df[self.max_col].values

    @property
    def bin_edges(self) -> np.ndarray:
        """Bin edges for the cost distribution."""
        return np.append(self.min_vals, self.max_vals[-1])

    @property
    def avg_vals(self) -> np.ndarray:
        """Average values for each of the cost distribution bins."""
        return self.df[self.avg_col].values

    @property
    def trip_vals(self) -> np.ndarray:
        """Trip values for each of the cost distribution bins."""
        return self.df[self.trip_vals].values

    @property
    def band_share_vals(self) -> np.ndarray:
        """Band share values for each of the cost distribution bins"""
        trip_vals = self.trip_vals
        return trip_vals / np.sum(trip_vals)

    @staticmethod
    def from_data(
        matrix: np.ndarray,
        cost_matrix: np.ndarray,
        min_bounds: Optional[list[float] | np.ndarray] = None,
        max_bounds: Optional[list[float] | np.ndarray] = None,
        bin_edges: Optional[list[float] | np.ndarray] = None,
    ) -> CostDistribution:
        """Convert values and a cost matrix into a CostDistribution.

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
            An instance of CostDistribution containing the given data.

        See Also
        --------
        `cost_distribution`
        """
        # Define columns names
        min_col = "min"
        max_col = "max"
        avg_col = "trips"
        trips_col = "ave_km"

        # Calculate the cost distribution
        bin_edges = _validate_bin_edges(
            bin_edges=bin_edges,
            min_bounds=min_bounds,
            max_bounds=max_bounds,
        )
        distribution = cost_distribution(
            matrix=matrix,
            cost_matrix=cost_matrix,
            bin_edges=bin_edges
        )

        # Covert data into instance of this class
        df = pd.DataFrame({
            min_col: bin_edges[:-1],
            max_col: bin_edges[1:],
            avg_col: (np.array(bin_edges[:-1]) + np.array(bin_edges[1:])) / 2,
            trips_col: distribution,
        })

        return CostDistribution(
            df=df,
            min_col=min_col,
            max_col=max_col,
            avg_col=avg_col,
            trips_col=trips_col,
        )


# # # FUNCTIONS # # #
def _validate_bin_edges(
    min_bounds: Optional[list[float] | np.ndarray] = None,
    max_bounds: Optional[list[float] | np.ndarray] = None,
    bin_edges: Optional[list[float] | np.ndarray] = None,
) -> np.ndarray | list[float]:
    # Use bounds to calculate bin edges
    if bin_edges is None:
        if min_bounds is None or max_bounds is None:
            raise ValueError(
                "Either `bin_edges` needs to be set, or both `min_bounds` and "
                "`max_bounds` needs to be set."
            )
        bin_edges = [min_bounds[0]] + list(max_bounds)
    return bin_edges


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
    bin_edges = _validate_bin_edges(
        bin_edges=bin_edges,
        min_bounds=min_bounds,
        max_bounds=max_bounds,
    )
    distribution, _ = np.histogram(
        a=cost_matrix,
        bins=bin_edges,
        weights=matrix,
    )
    return distribution
