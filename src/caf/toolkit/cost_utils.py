# -*- coding: utf-8 -*-
"""A toolbox of useful transport cost related functionality."""
from __future__ import annotations

# Built-Ins
import copy
import logging
import os
import warnings
from typing import Optional

# Third Party
import numpy as np
import pandas as pd

# Local Imports
from caf.toolkit import math_utils
from caf.toolkit import pandas_utils as pd_utils

# # # CONSTANTS # # #
LOG = logging.getLogger(__name__)


# # # CLASSES # # #
class CostDistribution:
    """Distribution of cost values between variable bounds.

    Alternate constructors are available in the See Also section

    Parameters
    ----------
    df:
        A DataFrame containing the binned cost distribution data. Must have columns
        named: `min_col`, `max_col`, `avg_col`, `trips_col`.

    min_col:
        The name of the columns in `df` that contains the lower bin edge
        value for each row.

    max_col:
        The name of the columns in `df` that contains the upper bin edge
        value for each row.

    avg_col:
        The name of the columns in `df` that contains the centre of the bin

    trips_col:
        The name of the columns in `df` that contains the value for each
        row.

    weighted_avg_col:
        The name of the columns in `df` that contains the weighted average
        value for each row. If available, this is different from `avg_col`
        as it takes into account this distribution of values within each
        bound when calculating averages.

    See Also
    --------
    :func:`from_data`
    :func:`from_data_no_bins`
    :func:`from_file`
    """

    # Ideas
    # units: str = "km"

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        min_col: str = "min",
        max_col: str = "max",
        avg_col: str = "avg",
        trips_col: str = "trips",
        weighted_avg_col: Optional[str] = None,
    ):
        # Keep as private. These shouldn't be needed outside of this class
        self.__df = df
        self.__min_col = min_col
        self.__max_col = max_col
        self.__avg_col = avg_col
        self.__trips_col = trips_col

        if weighted_avg_col is None:
            weighted_avg_col = "weighted_avg"
        self.__weighted_avg_col = weighted_avg_col

        self._validate_df_col_names()
        self._validate_df_values()

    def _validate_df_values(self) -> CostDistribution:
        """Sense check the values provided for the distribution.

        Raises
        ------
        ValueError
            If any columns contain anything other than positive numbers.

        Warns
        -----
        UserWarning
            If the bins aren't complete i.e. doesn't start at 0,
            gaps between bin edged or zero width bins.
        """
        basic_numerical_checks = {
            "min column": self.min_vals,
            "max column": self.max_vals,
            "trips": self.trip_vals,
            "average distance": self.avg_vals,
        }

        for name, check in basic_numerical_checks.items():
            if (check < 0).any():
                raise ValueError(f"Negatives are not allowed in the {name} column")
            if (np.isnan(check)).any():
                raise ValueError(f"NaNs are not allowed in the {name} column")
            if (np.isinf(check)).any():
                raise ValueError(f"Inf are not allowed in the {name} column")

        if self.min_vals.min() != 0:
            warnings.warn(
                "Minimum bound in min is not 0, consider recreating the"
                " distribution so no short distance trips are missed"
            )

        # we compare the max value to the min value of the next row to check for overlapping or disjoint bins
        gaps = self.max_vals[:-1] != self.min_vals[1:]
        # TODO(KF) this will do for now, but this could be made more specific
        if gaps.any():
            warnings.warn(
                "The bins do not nest (either overlapping or disjoint),"
                " there is a risk you will miss trips during your analysis"
            )

        zero_width = gaps = self.min_vals == self.max_vals

        if zero_width.any():
            warnings.warn(
                f"{zero_width.sum()} bins in the distribution have zero width, review if this makes sense"
            )

        return self

    def _validate_df_col_names(self) -> CostDistribution:
        """Check the given columns are in the given dataframe."""
        req_cols = {
            "min_col": self.__min_col,
            "max_col": self.__max_col,
            "avg_col": self.__avg_col,
            "trips_col": self.__trips_col,
        }

        # Check columns are in df
        err_cols = {}
        for col_name, col_val in req_cols.items():
            if col_val not in self.__df:
                err_cols.update({col_name: col_val})

        # Add in the weighted_avg_col if not already in df
        req_cols.update({"weighted_avg_col": self.__weighted_avg_col})
        if self.__weighted_avg_col not in self.__df.columns.to_list():
            self.__df[self.__weighted_avg_col] = self.__df[self.__avg_col]

        # Throw error if missing columns found
        if err_cols != dict():
            raise ValueError(
                "Not all the given column names exist in the given df. "
                f"The following columns are missing:{err_cols}\n"
                f"With the following in the DataFrame: {self.__df.columns}"
            )

        # Tidy up df
        self.__df = pd_utils.reindex_cols(self.__df, list(req_cols.values()))
        return self

    def __len__(self):
        """Get the number of bins in this cost distribution."""
        return len(self.bin_edges) - 1

    def __eq__(self, other):
        """Check if two items are the same."""
        if not isinstance(other, CostDistribution):
            return False
        # Optimisation: We want to compare the actual dataframes here rather than copies of them
        # pylint: disable-next=protected-access
        return (self.__df == other.__df).values.all()

    def copy(self) -> CostDistribution:
        """Create a copy of this instance."""
        return copy.copy(self)

    @property
    def df(self) -> pd.DataFrame:
        """A Pandas DataFrame containing the class data."""
        return self.__df.copy()

    @property
    def min_vals(self) -> np.ndarray:
        """Minimum values of the cost distribution bin edges."""
        return self.__df[self.__min_col].to_numpy()

    @property
    def max_vals(self) -> np.ndarray:
        """Maximum values of the cost distribution in edges."""
        return self.__df[self.__max_col].to_numpy()

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
        return self.__df[self.__avg_col].to_numpy()

    @property
    def weighted_avg_vals(self) -> np.ndarray:
        """Weighted average values for each of the cost distribution bins."""
        return self.__df[self.__weighted_avg_col].to_numpy()

    @property
    def trip_vals(self) -> np.ndarray:
        """Trip values for each of the cost distribution bins."""
        return self.__df[self.__trips_col].to_numpy()

    @property
    def band_share_vals(self) -> np.ndarray:
        """Band share values for each of the cost distribution bins."""
        trip_vals = self.trip_vals
        return trip_vals / np.sum(trip_vals)

    @staticmethod
    def calculate_weighted_averages(
        matrix: np.ndarray, cost_matrix: np.ndarray, bin_edges: list[float] | np.ndarray
    ):
        """
        Calculate weighted averages of bins in a cost distribution.

        Parameters
        ----------
        matrix: np.ndarray
            The matrix to calculate the cost distribution for. This matrix
            should be the same shape as cost_matrix

        cost_matrix: np.ndarray
            A matrix of cost relating to `matrix`. `cost_matrix`
            should be the same shape as `matrix`

        bin_edges: list[float] | np.ndarray
            Defines a monotonically increasing array of bin edges, including the
            rightmost edge, allowing for non-uniform bin widths. This argument
            is passed straight into `numpy.histogram`

        Returns
        -------
        np.ndarray
            An array to be passed into a dataframe as a column.
        """
        # Init and checks
        bin_edges = bin_edges.tolist() if isinstance(bin_edges, np.ndarray) else bin_edges
        if matrix.shape != cost_matrix.shape:
            raise ValueError(
                f"`matrix` and `cost_matrix` need to be the same shape. Got:\n"
                f"{matrix.shape=}\n"
                f"{cost_matrix.shape=}\n"
            )

        # Calculate distance weighted demand
        df = pd.DataFrame(
            {
                "cost": pd.DataFrame(cost_matrix).stack(),
                "demand": pd.DataFrame(matrix).stack(),
            }
        )
        df["weighted"] = df["cost"] * df["demand"]

        # Calculate the weighted average by bin
        df["bin"] = pd.cut(df["cost"], bins=bin_edges)
        grouped = df.groupby("bin", observed=False)[["weighted", "demand"]].sum().reset_index()
        grouped["averages"] = grouped["weighted"] / grouped["demand"]

        # Infill any missing values with bin midpoint
        grouped["bin_centres"] = grouped["bin"].apply(lambda x: x.mid)
        return grouped["averages"].fillna(grouped["bin_centres"].astype("float")).to_numpy()

    @classmethod
    def from_data(
        cls,
        matrix: np.ndarray,
        cost_matrix: np.ndarray,
        *,
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

        averages = cls.calculate_weighted_averages(matrix, cost_matrix, bin_edges)

        # Covert data into instance of this class
        df = pd.DataFrame(
            {
                "min": bin_edges[:-1],
                "max": bin_edges[1:],
                "avg": (np.array(bin_edges[:-1]) + np.array(bin_edges[1:])) / 2,
                "trips": distribution,
                "weighted_avg": averages,
            }
        )
        return CostDistribution(
            df=df,
            min_col="min",
            max_col="max",
            avg_col="avg",
            trips_col="trips",
            weighted_avg_col="weighted_avg",
        )

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
        *,
        min_col: str = "min",
        max_col: str = "max",
        avg_col: str = "avg",
        trips_col: str = "trips",
        weighted_avg_col: Optional[str] = None,
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

        weighted_avg_col:
            The column of data at 'filepath' that contains the weighted average
            cost value of each band. If the read in df does not contain this
            column, it will default to the avg_col.

        Returns
        -------
        cost_distribution:
            An instance containing the data at filepath.
        """
        # Validate the path
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"'{filepath}' is not the location of a file.")

        # Determine which columns to read in
        use_cols = [min_col, max_col, avg_col, trips_col]
        if weighted_avg_col is not None:
            use_cols.append(weighted_avg_col)

        return CostDistribution(
            df=pd.read_csv(filepath, usecols=use_cols),
            min_col=min_col,
            max_col=max_col,
            avg_col=avg_col,
            trips_col=trips_col,
            weighted_avg_col=weighted_avg_col,
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
        # Optimisation: We should make a new class with the constructor, however this is quicker
        # as it avoids the constructor validation that we know has already been done on this data
        # pylint: disable-next=protected-access
        new_distribution.__df[new_distribution.__trips_col] = trip_vals
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


def intrazonal_cost_infill(
    cost: np.ndarray,
    multiplier: float = 0.5,
    min_axis: int = 1,
) -> np.ndarray:
    """
    Infill the intra-zonal costs of a cost matrix.

    The intra-zonal costs are usually the diagonal of a cost matrix. Standard
    TAG procedure for infilling these costs is to take half the minimum cost
    for each zone. By default, this function takes the minimum value from each
    row (ignoring 0s) and multiplies that by 0.5 to get the infill value for
    each intra-zonal. Note that if any costs already exist for the
    intra-zonals, they will be overwritten.
    The diagonal infill is calculated similar to:
    `cost.min(axis=min_axis * multiplier`

    Parameters
    ----------
    cost:
        The square, 2D cost matrix to infill.

    multiplier:
        The value to multiply the minimum values by to calculate the infill value.

    min_axis:
        The axis to calculate the minimum value across.

    Returns
    -------
    infilled_cost:
        A copy of the input `cost`, but with the diagonal infilled.
    """
    # Ensure we don't pick up the diagonals or 0s in the minimum check
    nonzero_cost = np.where(cost == 0, np.inf, cost)
    np.fill_diagonal(nonzero_cost, np.inf)

    # Infill a copy of cost
    infill = nonzero_cost.min(axis=min_axis) * multiplier
    infilled_cost = cost.copy()
    np.fill_diagonal(infilled_cost, infill)
    return infilled_cost
