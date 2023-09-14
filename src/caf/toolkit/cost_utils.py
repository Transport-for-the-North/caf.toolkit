# -*- coding: utf-8 -*-
"""A toolbox of useful transport cost related functionality."""
from __future__ import annotations

# Built-Ins
import copy
import logging
import os

from typing import Optional
from typing import TYPE_CHECKING

# Third Party
import pydantic
import numpy as np
import pandas as pd

# Local Imports
# pylint: disable=import-error,wrong-import-position
from caf.toolkit import math_utils
from caf.toolkit import pandas_utils as pd_utils

# pylint: enable=import-error,wrong-import-position

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from pydantic.dataclasses import dataclass

# # # CONSTANTS # # #
LOG = logging.getLogger(__name__)


# # # CLASSES # # #
@dataclass(config={"arbitrary_types_allowed": True})  # type: ignore # pylint: disable=used-before-assignment, unexpected-keyword-arg
class CostDistribution:
    """Distribution of cost values between variable bounds.

    Attributes
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
    avg_col: str = "ave"
    trips_col: str = "trips"

    # Ideas
    units: str = "km"

    @pydantic.root_validator
    def check_df_col_names(cls, values):  # pylint:disable=no-self-argument
        """Check the given columns are in the given dataframe."""
        # init
        col_names = ["min_col", "max_col", "avg_col", "trips_col"]
        cols = {k: v for k, v in values.items() if k in col_names}
        df = values.get("df")

        # Check columns are in df
        err_cols = {}
        for col_name, col_val in cols.items():
            if col_val not in df:
                err_cols.update({col_name: col_val})

        # Throw error if missing columns found
        if err_cols != dict():
            raise ValueError(
                "Not all the given column names exist in the given df. "
                f"The following columns are missing:{err_cols}\n"
                f"With the following in the Df: {df.columns}"
            )

        # Tidy up df
        values["df"] = pd_utils.reindex_cols(df, cols.values())
        return values

    def __len__(self):
        """Get the number of bins in this cost distribution."""
        return len(self.bin_edges) - 1

    def __eq__(self, other):
        """Check if two items are the same."""
        if not isinstance(other, CostDistribution):
            return False
        return (self.df == other.df).values.all()

    def copy(self) -> CostDistribution:
        """Create a copy of this instance."""
        return copy.copy(self)

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
    def n_bins(self) -> int:
        """Bin edges for the cost distribution."""
        return len(self)

    @property
    def avg_vals(self) -> np.ndarray:
        """Average values for each of the cost distribution bins."""
        return self.df[self.avg_col].values

    @property
    def trip_vals(self) -> np.ndarray:
        """Trip values for each of the cost distribution bins."""
        return self.df[self.trips_col].values

    @property
    def band_share_vals(self) -> np.ndarray:
        """Band share values for each of the cost distribution bins."""
        trip_vals = self.trip_vals
        return trip_vals / np.sum(trip_vals)

    @classmethod
    def from_data(
        cls,
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
        # Calculate the cost distribution
        bin_edges = _validate_bin_edges(
            bin_edges=bin_edges,
            min_bounds=min_bounds,
            max_bounds=max_bounds,
        )
        distribution = cost_distribution(
            matrix=matrix, cost_matrix=cost_matrix, bin_edges=bin_edges
        )

        # Covert data into instance of this class
        df = pd.DataFrame(
            {
                cls.min_col: bin_edges[:-1],
                cls.max_col: bin_edges[1:],
                cls.avg_col: (np.array(bin_edges[:-1]) + np.array(bin_edges[1:])) / 2,
                cls.trips_col: distribution,
            }
        )
        return CostDistribution(df=df)

    @staticmethod
    def from_data_no_bins(
        matrix: np.ndarray,
        cost_matrix: np.ndarray,
        *args,
        **kwargs,
    ) -> CostDistribution:
        """Convert values and a cost matrix into a CostDistribution.

        `create_log_bins` will be used to generate some bin edges.

        Parameters
        ----------
        matrix:
            The matrix to calculate the cost distribution for. This matrix
            should be the same shape as cost_matrix

        cost_matrix:
            A matrix of cost relating to matrix. This matrix
            should be the same shape as matrix

        *args, **kwargs:
            arguments to pass through to `create_log_bins`

        Returns
        -------
        cost_distribution:
            An instance of CostDistribution containing the given data.

        See Also
        --------
        `cost_distribution`
        """
        bin_edges = create_log_bins(np.max(cost_matrix), *args, **kwargs)
        return CostDistribution.from_data(
            matrix=matrix,
            cost_matrix=cost_matrix,
            bin_edges=bin_edges,
        )

    @staticmethod
    def from_file(
        filepath: os.PathLike,
        min_col: str = "min",
        max_col: str = "max",
        avg_col: str = "ave",
        trips_col: str = "trips",
    ) -> CostDistribution:
        """Build an instance from a file on disk.

        Parameters
        ----------
        filepath:
            Path to the file to read in.

        min_col:
            The column of data at `filepath` that contains the minimum cost
            value of each band.

        max_col:
            The column of data at `filepath` that contains the maximum cost
            value of each band.

        avg_col:
            The column of data at `filepath` that contains the average cost
            value of each band.

        trips_col:
            The column of data at `filepath` that contains the number of trips
            of each cost band.

        Returns
        -------
        cost_distribution:
            An instance containing the data at filepath.
        """
        if not os.path.isfile(filepath):
            raise ValueError(f"'{filepath}' is not the location of a file.")
        use_cols = [min_col, max_col, avg_col, trips_col]
        return CostDistribution(
            df=pd.read_csv(filepath, usecols=use_cols),
            min_col=min_col,
            max_col=max_col,
            avg_col=avg_col,
            trips_col=trips_col,
        )

    def __validate_similar_bin_edges(self, other: CostDistribution) -> None:
        """Check whether other is using the same bins as self.

        Parameters
        ----------
        other:
            Another instance of CostDistribution using the same bins.

        Returns
        -------
        None

        Raises
        ------
        ValueError:
            When self and other do not have similar enough bin_edges.
        """
        if (
            self.bin_edges.shape != other.bin_edges.shape
            or not np.allclose(self.bin_edges, other.bin_edges)
        ):  # fmt: skip
            raise ValueError(
                "Bin edges are not similar enough.\n"
                f"{self.bin_edges=}\n"
                f"{other.bin_edges=}"
            )

    def create_similar(self, trip_vals: np.ndarray) -> CostDistribution:
        """Create a similar cost distribution with different trip values.

        Parameters
        ----------
        trip_vals:
            A numpy array of trip values that will replace the current trip
            values.

        Returns
        -------
        cost_distribution:
            A copy of this instance, with different trip values.
        """
        if trip_vals.shape != self.trip_vals.shape:
            raise ValueError(
                "The new trip_vals are not the correct shape to fit existing "
                f"data. Expected a shape of {self.trip_vals.shape}, got "
                f"{trip_vals.shape}."
            )
        new_distribution = self.copy()
        new_distribution.df[new_distribution.trips_col] = trip_vals
        return new_distribution

    def trip_residuals(self, other: CostDistribution) -> np.ndarray:
        """Calculate the trip residuals between this and other.

        Residuals are calculated as:
        `self.trip_vals - other.trip_vals`

        Parameters
        ----------
        other:
            Another instance of CostDistribution using the same bins.

        Returns
        -------
        residuals:
            The residual difference between this and other.
        """
        self.__validate_similar_bin_edges(other)
        return self.trip_vals - other.trip_vals

    def band_share_residuals(self, other: CostDistribution) -> np.ndarray:
        """Calculate the band share residuals between this and other.

        Residuals are calculated as:
        `self.band_share_vals - other.band_share_vals`

        Parameters
        ----------
        other:
            Another instance of CostDistribution using the same bins.

        Returns
        -------
        residuals:
            The residual difference between this and other.
        """
        self.__validate_similar_bin_edges(other)
        return self.band_share_vals - other.band_share_vals

    def band_share_convergence(self, other: CostDistribution) -> float:
        """Calculate the convergence between this and other.

        Residuals are calculated as:
        `math_utils.curve_convergence(self.band_share_vals, other.band_share_vals)`

        Parameters
        ----------
        other:
            Another instance of CostDistribution using the same bins.

        Returns
        -------
        convergence:
            A float value between 0 and 1. Values closer to 1 indicate a better
            convergence.

        See Also
        --------
        `math_utils.curve_convergence`
        """
        self.__validate_similar_bin_edges(other)
        return math_utils.curve_convergence(self.band_share_vals, other.band_share_vals)


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


def dynamic_cost_distribution(
    matrix: np.ndarray,
    cost_matrix: np.ndarray,
    *args,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate the distribution of costs across a matrix, using dynamic bins.

    Parameters
    ----------
    matrix:
        The matrix to calculate the cost distribution for. This matrix
        should be the same shape as cost_matrix

    cost_matrix:
        A matrix of cost relating to matrix. This matrix
        should be the same shape as matrix

    *args, **kwargs:
        arguments to pass through to `create_log_bins`

    Returns
    -------
    cost_distribution:
        A numpy array of the sum of trips by distance band.

    See Also
    --------
    `create_log_bins`
    """
    bin_edges = create_log_bins(np.max(cost_matrix), *args, **kwargs)
    distribution = cost_distribution(
        matrix=matrix,
        cost_matrix=cost_matrix,
        bin_edges=bin_edges,
    )
    return distribution, bin_edges


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


def create_log_bins(
    max_value: float,
    n_bin_pow: float = 0.51,
    log_factor: float = 2.2,
    final_val: float = 1500.0,
) -> np.ndarray:
    """Dynamically choose the bins based on the maximum possible value.

    `n_bins = int(max_value ** n_bin_pow)` Is used to choose the number of
    bins to use.
    `bins = (np.array(range(2, n_bins)) / n_bins) ** log_factor * max_value`
    is used to determine the bin edges being used.

    Parameters
    ----------
    max_value:
        The maximum value seen in the data, this is used to scale the bins
        appropriately.

    n_bin_pow:
        The power used to determine the number of bins to use, depending
        on the max value. This value should be between 0 and 1. (0, 1)
        `max_value ** n_bin_pow`.

    log_factor:
        The log factor to determine the bin spacing. This should be a
        value greater than 1. Larger numbers mean closer bins

    final_val:
        The final value to append to the end of the bin edges. The second
        to last bin will be less than `max_value`, therefore this number
        needs to be larger than the max value.

    Returns
    -------
    bin_edges:
        A numpy array of bin edges.
    """
    # Validate
    if final_val < max_value:
        raise ValueError("`final_val` is lower than `max_value`.")

    if not 0 < n_bin_pow < 1:
        raise ValueError(
            f"`n_bin_pow` should be in the range (0, 1). Got a value of " f"{n_bin_pow}."
        )

    if log_factor <= 0:
        raise ValueError(
            f"`log_factor` should be greater than 0. Got a value of " f"{log_factor}."
        )

    # Calculate
    n_bins = int(max_value**n_bin_pow)
    bins = (np.array(range(2, n_bins + 1)) / n_bins) ** log_factor * max_value
    bins = np.floor(bins)

    # Add the first and last item
    bins = np.insert(bins, 0, 0)
    return np.insert(bins, len(bins), final_val)
