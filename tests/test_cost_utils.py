# -*- coding: utf-8 -*-
"""Tests for the {} module"""
from __future__ import annotations

# Built-Ins
import dataclasses
from typing import Any

# Third Party
import numpy as np
import pandas as pd
import pytest

# Local Imports
# pylint: disable=import-error,wrong-import-position
from caf.toolkit import cost_utils, math_utils

# pylint: enable=import-error,wrong-import-position

# # # CONSTANTS # # #


# # # DATACLASSES # # #
@dataclasses.dataclass
class CostDistFnResults:
    """Inputs and expected results for a cost_distribution function"""

    # Inputs
    matrix: np.ndarray
    cost_matrix: np.ndarray
    bin_edges: np.ndarray

    # Results
    distribution: np.ndarray

    def __post_init__(self):
        self.min_bounds = self.bin_edges[:-1]
        self.max_bounds = self.bin_edges[1:]

        if self.distribution.sum() == 0:
            self.normalised_distribution = np.zeros_like(self.distribution)
        else:
            self.normalised_distribution = self.distribution / self.distribution.sum()
            self.weighted_avg = cost_utils.CostDistribution.calculate_weighted_averages(
                matrix=self.matrix, cost_matrix=self.cost_matrix, bin_edges=self.bin_edges
            )


@dataclasses.dataclass
class DynamicCostDistFnResults(CostDistFnResults):
    """Inputs and expected results for a cost_distribution function"""

    # Inputs
    n_bin_pow: float
    log_factor: float
    final_val: float

    def get_kwargs(self) -> dict[str, Any]:
        """Get the kwarg dict for easy calls."""
        return {
            "matrix": self.matrix,
            "cost_matrix": self.cost_matrix,
            "n_bin_pow": self.n_bin_pow,
            "log_factor": self.log_factor,
            "final_val": self.final_val,
        }


@dataclasses.dataclass
class CostDistClassResults(CostDistFnResults):
    """Inputs and expected results for a CostDistribution class."""

    # Inputs
    min_col: str = "min"
    max_col: str = "max"
    avg_col: str = "avg"
    trips_col: str = "trips"
    weighted_avg_col: str = "weighted_avg"

    def __post_init__(self):
        super().__post_init__()

        self.avg_bounds = (self.min_bounds + self.max_bounds) / 2
        self.df = pd.DataFrame(
            {
                self.min_col: self.min_bounds,
                self.max_col: self.max_bounds,
                self.avg_col: self.avg_bounds,
                self.trips_col: self.distribution,
                self.weighted_avg_col: self.weighted_avg,
            }
        )
        self.normalised_df = pd.DataFrame(
            {
                self.min_col: self.min_bounds,
                self.max_col: self.max_bounds,
                self.avg_col: self.avg_bounds,
                self.trips_col: self.normalised_distribution,
                self.weighted_avg_col: self.weighted_avg,
            }
        )

        self.cost_dist_instance = cost_utils.CostDistribution(**self.constructor_kwargs)

    @property
    def constructor_kwargs(self) -> dict[str, Any]:
        """A kwarg dictionary for calling CostDistribution constructor."""
        return {
            "df": self.df,
            "min_col": self.min_col,
            "max_col": self.max_col,
            "avg_col": self.avg_col,
            "trips_col": self.trips_col,
            "weighted_avg_col": self.weighted_avg_col,
        }

    @property
    def constructor_kwargs_no_weighted_avg(self) -> dict[str, Any]:
        """A kwarg dictionary for calling CostDistribution constructor."""
        return {
            "df": self.df.drop(columns=[self.weighted_avg_col]),
            "min_col": self.min_col,
            "max_col": self.max_col,
            "avg_col": self.avg_col,
            "trips_col": self.trips_col,
        }

    @property
    def default_name_df(self) -> pd.DataFrame:
        """Get the internal pandas dataframe using the default col names."""
        naming_dict = {
            self.min_col: "min",
            self.max_col: "max",
            self.avg_col: "avg",
            self.trips_col: "trips",
            self.weighted_avg_col: "weighted_avg",
        }
        return self.df.rename(columns=naming_dict)


@dataclasses.dataclass
class DynamicCostDistClassResults(CostDistClassResults, DynamicCostDistFnResults):
    """Inputs and expected results for a CostDistribution class w/ dynamic bounds."""


@dataclasses.dataclass
class LogBinsResults:
    """Inputs and expected results for create_log_bins function."""

    # Inputs
    max_value: float
    n_bin_pow: float
    log_factor: float
    final_val: float

    # Results
    expected_bins: np.ndarray

    def get_kwargs(self) -> dict[str, Any]:
        """Get the kwarg dict for easy calls."""
        return {
            "max_value": self.max_value,
            "n_bin_pow": self.n_bin_pow,
            "log_factor": self.log_factor,
            "final_val": self.final_val,
        }


# # # FIXTURES # # #
@pytest.fixture(name="cost_dist_1d", scope="class")
def fixture_cost_dist_1d() -> CostDistFnResults:
    """Create a 1D matrix to distribute"""
    return CostDistFnResults(
        matrix=np.array([26.0, 43.0, 5.0, 8.0, 18.0, 51.0, 35.0, 39.0, 32.0, 37.0]),
        cost_matrix=np.array([77.0, 74.0, 53.0, 60.0, 94.0, 65.0, 13.0, 79.0, 39.0, 75.0]),
        bin_edges=np.array([0, 5, 10, 20, 40, 75, 100]),
        distribution=np.array([0.0, 0.0, 35.0, 32.0, 107.0, 120.0]),
    )


@pytest.fixture(name="cost_dist_2d", scope="class")
def fixture_cost_dist_2d() -> CostDistFnResults:
    """Create a 2D matrix to distribute"""
    matrix = np.array(
        [
            [60.0, 27.0, 79.0, 63.0, 8.0],
            [53.0, 85.0, 3.0, 45.0, 3.0],
            [19.0, 100.0, 75.0, 16.0, 62.0],
            [65.0, 37.0, 63.0, 69.0, 56.0],
            [87.0, 43.0, 5.0, 20.0, 57.0],
        ]
    )

    cost_matrix = np.array(
        [
            [54.0, 72.0, 61.0, 97.0, 72.0],
            [41.0, 84.0, 98.0, 32.0, 32.0],
            [4.0, 33.0, 67.0, 14.0, 26.0],
            [73.0, 46.0, 14.0, 8.0, 51.0],
            [2.0, 14.0, 58.0, 53.0, 40.0],
        ]
    )

    return CostDistFnResults(
        matrix=matrix,
        cost_matrix=cost_matrix,
        bin_edges=np.array([0, 5, 10, 20, 40, 75, 100]),
        distribution=np.array([106.0, 69.0, 122.0, 210.0, 542.0, 151.0]),
    )


@pytest.fixture(name="dynamic_cost_dist_1d", scope="class")
def fixture_dynamic_cost_dist_1d(cost_dist_1d) -> DynamicCostDistFnResults:
    """Create a 1d dynamic cost distribution and results"""
    dynamic_bin_edges = np.array([0, 2, 6, 12, 20, 30, 42, 57, 74, 94, 110.0])
    distribution = np.array([0, 0, 0, 35, 0, 32, 5, 59, 145, 18])
    return DynamicCostDistFnResults(
        matrix=cost_dist_1d.matrix,
        cost_matrix=cost_dist_1d.cost_matrix,
        n_bin_pow=0.51,
        log_factor=2.2,
        final_val=110,
        bin_edges=dynamic_bin_edges,
        distribution=distribution,
    )


@pytest.fixture(name="dynamic_cost_dist_2d", scope="class")
def fixture_dynamic_cost_dist_2d(cost_dist_2d) -> DynamicCostDistFnResults:
    """Create a 2d dynamic cost distribution and results"""
    distribution = np.array([0, 106, 69, 122, 62, 258, 178, 254, 148, 3])
    dynamic_bin_edges = np.array([0, 2, 6, 13, 21, 31, 44, 59, 77, 98, 110])
    return DynamicCostDistFnResults(
        matrix=cost_dist_2d.matrix,
        cost_matrix=cost_dist_2d.cost_matrix,
        n_bin_pow=0.51,
        log_factor=2.2,
        final_val=110,
        bin_edges=dynamic_bin_edges,
        distribution=distribution,
    )


@pytest.fixture(name="small_log_bins", scope="class")
def fixture_small_log_bins():
    """Create log bins with few values"""
    return LogBinsResults(
        max_value=10,
        n_bin_pow=0.51,
        log_factor=2.2,
        final_val=25,
        expected_bins=np.array([0, 4, 10, 25]),
    )


@pytest.fixture(name="med_log_bins", scope="class")
def fixture_med_log_bins():
    """Create log bins with few values"""
    return LogBinsResults(
        max_value=100,
        n_bin_pow=0.51,
        log_factor=2.2,
        final_val=300,
        expected_bins=np.array([0, 2, 7, 13, 21, 32, 45, 61, 79, 100, 300]),
    )


@pytest.fixture(name="large_log_bins", scope="class")
def fixture_large_log_bins():
    """Create log bins with few values"""
    # fmt: off
    expected_bins = np.array(
        [0, 1, 4, 8, 14, 21, 30, 40, 52, 66, 81, 99, 118, 139, 161, 186, 213,
         241, 272, 304, 339, 375, 414, 455, 497, 542, 589, 638, 690, 743, 799,
         856, 916, 979, 1043, 1110, 1179, 1250, 1324, 1400, 1500]
    )
    # fmt: on

    return LogBinsResults(
        max_value=1400,
        n_bin_pow=0.51,
        log_factor=2.2,
        final_val=1500,
        expected_bins=expected_bins,
    )


@pytest.fixture(name="cost_dist_1d_class", scope="function")
def fixture_cost_dist_1d_class(cost_dist_1d) -> CostDistClassResults:
    """Create a 1D array of values to pass to CostDistribution."""
    return CostDistClassResults(
        matrix=cost_dist_1d.matrix,
        cost_matrix=cost_dist_1d.cost_matrix,
        bin_edges=cost_dist_1d.bin_edges,
        distribution=cost_dist_1d.distribution,
    )


@pytest.fixture(name="cost_dist_2d_class", scope="function")
def fixture_cost_dist_2d_class(cost_dist_2d) -> CostDistClassResults:
    """Create a 1D array of values to pass to CostDistribution."""
    return CostDistClassResults(
        matrix=cost_dist_2d.matrix,
        cost_matrix=cost_dist_2d.cost_matrix,
        bin_edges=cost_dist_2d.bin_edges,
        distribution=cost_dist_2d.distribution,
    )


@pytest.fixture(name="cost_dist_2d_class_cols", scope="function")
def fixture_cost_dist_2d_class_cols(cost_dist_2d) -> CostDistClassResults:
    """Create a 1D array of values to pass to CostDistribution."""
    return CostDistClassResults(
        matrix=cost_dist_2d.matrix,
        cost_matrix=cost_dist_2d.cost_matrix,
        bin_edges=cost_dist_2d.bin_edges,
        distribution=cost_dist_2d.distribution,
        min_col="smallest",
        max_col="biggest34",
        avg_col="in the middle",
        trips_col="some random values",
    )


@pytest.fixture(name="dynamic_cost_dist_1d_class", scope="class")
def fixture_dynamic_cost_dist_1d_class(dynamic_cost_dist_1d) -> DynamicCostDistClassResults:
    """Create a 1d dynamic cost distribution and results"""
    return DynamicCostDistClassResults(
        matrix=dynamic_cost_dist_1d.matrix,
        cost_matrix=dynamic_cost_dist_1d.cost_matrix,
        n_bin_pow=dynamic_cost_dist_1d.n_bin_pow,
        log_factor=dynamic_cost_dist_1d.log_factor,
        final_val=dynamic_cost_dist_1d.final_val,
        bin_edges=dynamic_cost_dist_1d.bin_edges,
        distribution=dynamic_cost_dist_1d.distribution,
    )


@pytest.fixture(name="dynamic_cost_dist_2d_class", scope="class")
def fixture_dynamic_cost_dist_2d_class(dynamic_cost_dist_2d) -> DynamicCostDistClassResults:
    """Create a 1d dynamic cost distribution and results"""
    return DynamicCostDistClassResults(
        matrix=dynamic_cost_dist_2d.matrix,
        cost_matrix=dynamic_cost_dist_2d.cost_matrix,
        n_bin_pow=dynamic_cost_dist_2d.n_bin_pow,
        log_factor=dynamic_cost_dist_2d.log_factor,
        final_val=dynamic_cost_dist_2d.final_val,
        bin_edges=dynamic_cost_dist_2d.bin_edges,
        distribution=dynamic_cost_dist_2d.distribution,
    )


# # # TESTS # # #
@pytest.mark.usefixtures(
    "cost_dist_1d_class",
    "cost_dist_2d_class",
    "cost_dist_2d_class_cols",
    "dynamic_cost_dist_1d_class",
    "dynamic_cost_dist_2d_class",
)
class TestCostDistributionClassConstructors:
    """Tests for the construction methods for CostDistribution class."""

    @pytest.mark.parametrize(
        "io_str",
        ["cost_dist_1d_class", "cost_dist_2d_class", "cost_dist_2d_class_cols"],
    )
    def test_correct_init(self, io_str: str, request):
        """Test the class constructor creates the correct DF internally."""
        input_and_results: CostDistClassResults = request.getfixturevalue(io_str)
        cost_dist = cost_utils.CostDistribution(**input_and_results.constructor_kwargs)
        pd.testing.assert_frame_equal(cost_dist.df, input_and_results.df)

    def test_default_weighted_average(self, cost_dist_1d_class):
        """Test that the weighted_avg col is set to the same as avg when not given"""
        got = cost_utils.CostDistribution(
            **cost_dist_1d_class.constructor_kwargs_no_weighted_avg
        )
        expected = cost_dist_1d_class.df[cost_dist_1d_class.avg_col]
        np.testing.assert_almost_equal(expected, got.weighted_avg_vals)

    @pytest.mark.parametrize(
        "io_str",
        ["cost_dist_1d_class", "cost_dist_2d_class", "cost_dist_2d_class_cols"],
    )
    def test_init_wrong_col(self, io_str: str, request):
        """Test the class constructor throws errors with bad column names."""
        input_and_results: CostDistClassResults = request.getfixturevalue(io_str)
        with pytest.raises(ValueError, match="The following columns are missing"):
            cost_utils.CostDistribution(df=input_and_results.df, min_col="wrong_name")

    @pytest.mark.parametrize(
        "io_str",
        ["cost_dist_1d_class", "cost_dist_2d_class", "cost_dist_2d_class_cols"],
    )
    def test_correct_from_data_edges(self, io_str: str, request):
        """Test an alternate class constructor works correctly."""
        input_and_results: CostDistClassResults = request.getfixturevalue(io_str)
        cost_dist = cost_utils.CostDistribution.from_data(
            matrix=input_and_results.matrix,
            cost_matrix=input_and_results.cost_matrix,
            bin_edges=input_and_results.bin_edges,
        )
        pd.testing.assert_frame_equal(cost_dist.df, input_and_results.default_name_df)

    @pytest.mark.parametrize(
        "io_str",
        ["cost_dist_1d_class", "cost_dist_2d_class", "cost_dist_2d_class_cols"],
    )
    def test_correct_from_data_bounds(self, io_str: str, request):
        """Test an alternate class constructor works correctly."""
        input_and_results: CostDistClassResults = request.getfixturevalue(io_str)
        cost_dist = cost_utils.CostDistribution.from_data(
            matrix=input_and_results.matrix,
            cost_matrix=input_and_results.cost_matrix,
            min_bounds=input_and_results.min_bounds,
            max_bounds=input_and_results.max_bounds,
        )
        pd.testing.assert_frame_equal(cost_dist.df, input_and_results.default_name_df)

    @pytest.mark.parametrize("which_do", ["none", "min", "max"])
    def test_bad_bounds(self, cost_dist_2d_class: CostDistFnResults, which_do: str):
        """Check an error is raised when bad bounds given"""
        # Determine the kwargs
        kwargs = {
            "matrix": cost_dist_2d_class.matrix,
            "cost_matrix": cost_dist_2d_class.cost_matrix,
        }
        if which_do == "min":
            kwargs.update({"min_bounds": cost_dist_2d_class.min_bounds})
        elif which_do == "max":
            kwargs.update({"max_bounds": cost_dist_2d_class.max_bounds})
        # Implicit if "none", do nothing

        # Check fro error
        msg = (
            "Either `bin_edges` needs to be set, or both `min_bounds` and "
            "`max_bounds` needs to be set."
        )
        with pytest.raises(ValueError, match=msg):
            cost_utils.CostDistribution.from_data(
                matrix=cost_dist_2d_class.matrix,
                cost_matrix=cost_dist_2d_class.cost_matrix,
                min_bounds=cost_dist_2d_class.min_bounds,
            )

    @pytest.mark.parametrize(
        "io_str",
        ["dynamic_cost_dist_1d_class", "dynamic_cost_dist_2d_class"],
    )
    def test_correct_data_no_bins(self, io_str: str, request):
        """Test an alternate class constructor works correctly."""
        input_and_results: DynamicCostDistClassResults = request.getfixturevalue(io_str)
        cost_dist = cost_utils.CostDistribution.from_data_no_bins(
            **input_and_results.get_kwargs()
        )
        pd.testing.assert_frame_equal(
            cost_dist.df,
            input_and_results.default_name_df,
            check_dtype=False,
        )

    @pytest.mark.parametrize(
        "io_str", ["cost_dist_1d_class", "cost_dist_2d_class", "cost_dist_2d_class_cols"]
    )
    def test_correct_from_file(self, io_str: str, request, tmp_path):
        """Test that the constructor can be called correctly from file"""
        input_and_results: CostDistClassResults = request.getfixturevalue(io_str)
        expected_df = input_and_results.df

        # Create path and write out df
        filepath = tmp_path / "tld.csv"
        expected_df.to_csv(filepath, index=False)

        # Load back in using constructor
        result = cost_utils.CostDistribution.from_file(
            filepath=filepath,
            min_col=input_and_results.min_col,
            max_col=input_and_results.max_col,
            avg_col=input_and_results.avg_col,
            trips_col=input_and_results.trips_col,
            weighted_avg_col=input_and_results.weighted_avg_col,
        )

        # We only really care about the values here, so compare numpy arrays instead
        np.testing.assert_almost_equal(expected_df.to_numpy(), result.df.to_numpy())

    @pytest.mark.parametrize(
        "io_str", ["cost_dist_1d_class", "cost_dist_2d_class", "cost_dist_2d_class_cols"]
    )
    def test_correct_from_file_no_weighted(
        self,
        io_str: str,
        request,
        tmp_path,
    ):
        """Test that the constructor can be called correctly from file without weighted_avg col"""
        input_and_results: CostDistClassResults = request.getfixturevalue(io_str)

        # Create path and write out df
        filepath = tmp_path / "tld.csv"
        input_and_results.df.to_csv(filepath, index=False)

        # Load back in, but pretend the weighted avg col doesn't exist
        result = cost_utils.CostDistribution.from_file(
            filepath=filepath,
            min_col=input_and_results.min_col,
            max_col=input_and_results.max_col,
            avg_col=input_and_results.avg_col,
            trips_col=input_and_results.trips_col,
        )

        # We expect the result to just copy the avg col to weighted_avg
        expected_df = input_and_results.df
        expected_df[input_and_results.weighted_avg_col] = expected_df[
            input_and_results.avg_col
        ]

        # We only really care about the values here, so compare numpy arrays instead
        np.testing.assert_almost_equal(expected_df.to_numpy(), result.df.to_numpy())


@pytest.mark.usefixtures("cost_dist_1d_class", "cost_dist_2d_class", "cost_dist_2d_class_cols")
@pytest.mark.parametrize(
    "io_str",
    ["cost_dist_1d_class", "cost_dist_2d_class", "cost_dist_2d_class_cols"],
)
class TestCostDistributionClassProperties:
    """Tests for the properties of CostDistribution class."""

    def test_min_vals(self, io_str: str, request):
        """Test correct functionality."""
        input_and_results: CostDistClassResults = request.getfixturevalue(io_str)
        cost_dist = input_and_results.cost_dist_instance
        np.testing.assert_almost_equal(cost_dist.min_vals, input_and_results.min_bounds)

    def test_max_vals(self, io_str: str, request):
        """Test correct functionality."""
        input_and_results: CostDistClassResults = request.getfixturevalue(io_str)
        cost_dist = input_and_results.cost_dist_instance
        np.testing.assert_almost_equal(cost_dist.max_vals, input_and_results.max_bounds)

    def test_bin_edges(self, io_str: str, request):
        """Test correct functionality."""
        input_and_results: CostDistClassResults = request.getfixturevalue(io_str)
        cost_dist = input_and_results.cost_dist_instance
        np.testing.assert_almost_equal(cost_dist.bin_edges, input_and_results.bin_edges)

    def test_n_bins(self, io_str: str, request):
        """Test correct functionality."""
        input_and_results: CostDistClassResults = request.getfixturevalue(io_str)
        cost_dist = input_and_results.cost_dist_instance
        n_bins = len(input_and_results.bin_edges) - 1
        np.testing.assert_almost_equal(cost_dist.n_bins, n_bins)

    def test_avg_vals(self, io_str: str, request):
        """Test correct functionality."""
        input_and_results: CostDistClassResults = request.getfixturevalue(io_str)
        cost_dist = input_and_results.cost_dist_instance

        bin_edges = input_and_results.bin_edges
        avg_vals = (bin_edges[:-1] + bin_edges[1:]) / 2
        np.testing.assert_almost_equal(cost_dist.avg_vals, avg_vals)

    def test_weighted_avg_vals(self, io_str: str, request):
        """Test correct functionality."""
        input_and_results: CostDistClassResults = request.getfixturevalue(io_str)
        cost_dist = input_and_results.cost_dist_instance

        bin_edges = input_and_results.bin_edges
        avg_vals = (bin_edges[:-1] + bin_edges[1:]) / 2
        np.testing.assert_almost_equal(cost_dist.avg_vals, avg_vals)

    def test_trip_vals(self, io_str: str, request):
        """Test correct functionality."""
        input_and_results: CostDistClassResults = request.getfixturevalue(io_str)
        cost_dist = input_and_results.cost_dist_instance
        np.testing.assert_almost_equal(cost_dist.trip_vals, input_and_results.distribution)

    def test_band_share_vals(self, io_str: str, request):
        """Test correct functionality."""
        input_and_results: CostDistClassResults = request.getfixturevalue(io_str)
        cost_dist = input_and_results.cost_dist_instance

        distribution = input_and_results.distribution
        band_share_vals = distribution / np.sum(distribution)
        np.testing.assert_almost_equal(cost_dist.band_share_vals, band_share_vals)


@pytest.mark.usefixtures("cost_dist_1d_class", "cost_dist_2d_class", "cost_dist_2d_class_cols")
@pytest.mark.parametrize(
    "io_str",
    ["cost_dist_1d_class", "cost_dist_2d_class", "cost_dist_2d_class_cols"],
)
class TestCostDistributionClassMethods:
    """Tests for the methods of CostDistribution class."""

    def test_length(self, io_str: str, request):
        """Test the class __len__ method works correctly."""
        input_and_results: CostDistClassResults = request.getfixturevalue(io_str)
        cost_dist = input_and_results.cost_dist_instance
        assert len(cost_dist) == len(input_and_results.bin_edges) - 1

    def test_copy_equals(self, io_str: str, request):
        """Test the class copy and equal methods work correctly."""
        input_and_results: CostDistClassResults = request.getfixturevalue(io_str)
        cost_dist = input_and_results.cost_dist_instance
        assert cost_dist.copy() == cost_dist

    def test_create_similar(self, io_str: str, request):
        """Test the class create_similar method works correctly."""
        input_and_results: CostDistClassResults = request.getfixturevalue(io_str)
        cost_dist = input_and_results.cost_dist_instance

        # Create the expected output
        new_trip_vals = np.random.rand(*input_and_results.distribution.shape)
        expected_df = cost_dist.df.copy()
        expected_df[input_and_results.trips_col] = new_trip_vals

        # Create with function and assert
        got_df = cost_dist.create_similar(trip_vals=new_trip_vals).df
        pd.testing.assert_frame_equal(got_df, expected_df)

    def test_bad_shape_create_similar(self, io_str: str, request):
        """Test the class create_similar method correctly throws an error."""
        input_and_results: CostDistClassResults = request.getfixturevalue(io_str)
        cost_dist = input_and_results.cost_dist_instance

        # Create the expected output
        new_trip_vals = np.random.rand(*input_and_results.distribution.shape)
        new_trip_vals = new_trip_vals[:-1]

        with pytest.raises(ValueError, match="not the correct shape"):
            cost_dist.create_similar(trip_vals=new_trip_vals)

    @pytest.mark.parametrize(
        "io_str2",
        ["cost_dist_1d_class", "cost_dist_2d_class", "cost_dist_2d_class_cols"],
    )
    def test_trip_residuals(self, io_str: str, io_str2: str, request):
        """Test the class residuals method works correctly."""
        input_and_results: CostDistClassResults = request.getfixturevalue(io_str)
        cost_dist1 = input_and_results.cost_dist_instance
        input_and_results2: CostDistClassResults = request.getfixturevalue(io_str2)
        cost_dist2 = input_and_results2.cost_dist_instance

        desired = input_and_results.distribution - input_and_results2.distribution
        actual = cost_dist1.trip_residuals(cost_dist2)
        np.testing.assert_almost_equal(actual, desired)

    @pytest.mark.parametrize(
        "io_str2",
        ["cost_dist_1d_class", "cost_dist_2d_class", "cost_dist_2d_class_cols"],
    )
    def test_band_share_residuals(self, io_str: str, io_str2: str, request):
        """Test the class residuals method works correctly."""
        input_and_results: CostDistClassResults = request.getfixturevalue(io_str)
        input_and_results2: CostDistClassResults = request.getfixturevalue(io_str2)

        # Calc desired
        band_share1 = input_and_results.distribution / input_and_results.distribution.sum()
        band_share2 = input_and_results2.distribution / input_and_results2.distribution.sum()
        desired = band_share1 - band_share2

        # Calc actual
        cost_dist1 = input_and_results.cost_dist_instance
        cost_dist2 = input_and_results2.cost_dist_instance
        actual = cost_dist1.band_share_residuals(cost_dist2)

        np.testing.assert_almost_equal(actual, desired)

    @pytest.mark.parametrize(
        "io_str2",
        ["cost_dist_1d_class", "cost_dist_2d_class", "cost_dist_2d_class_cols"],
    )
    def test_band_share_convergence(self, io_str: str, io_str2: str, request):
        """Test the class convergence method works correctly."""
        input_and_results: CostDistClassResults = request.getfixturevalue(io_str)
        input_and_results2: CostDistClassResults = request.getfixturevalue(io_str2)

        # Calc desired
        band_share1 = input_and_results.distribution / input_and_results.distribution.sum()
        band_share2 = input_and_results2.distribution / input_and_results2.distribution.sum()
        desired = math_utils.curve_convergence(band_share1, band_share2)

        # Calc actual
        cost_dist1 = input_and_results.cost_dist_instance
        cost_dist2 = input_and_results2.cost_dist_instance
        actual = cost_dist1.band_share_convergence(cost_dist2)

        np.testing.assert_almost_equal(actual, desired)

    @pytest.mark.parametrize(
        "fn_str",
        ["trip_residuals", "band_share_residuals", "band_share_convergence"],
    )
    @pytest.mark.parametrize("bin_str", ["values", "shape"])
    def test_bad_bin_edges(self, io_str: str, fn_str: str, bin_str: str, request):
        """Test the class correctly throws an error with bad band edges."""
        input_and_results: CostDistClassResults = request.getfixturevalue(io_str)
        cost_dist = input_and_results.cost_dist_instance

        # Decide which function to use
        if fn_str == "trip_residuals":
            fn = cost_dist.trip_residuals
        elif fn_str == "band_share_residuals":
            fn = cost_dist.band_share_residuals
        elif fn_str == "band_share_convergence":
            fn = cost_dist.band_share_convergence
        else:
            raise ValueError

        # Decide which bad_bin edges to use
        if bin_str == "values":
            bad_bin_edges = input_and_results.bin_edges * 1.5
        elif bin_str == "shape":
            bad_bin_edges = input_and_results.bin_edges[:-1]
        else:
            raise ValueError

        # Create a bad class to call function with
        other = cost_utils.CostDistribution.from_data(
            matrix=input_and_results.matrix,
            cost_matrix=input_and_results.cost_matrix,
            bin_edges=bad_bin_edges,
        )

        with pytest.raises(ValueError, match="are not similar enough"):
            fn(other)


@pytest.mark.usefixtures("cost_dist_1d", "cost_dist_2d")
class TestCostDistributionFunction:
    """Tests for the cost distribution function"""

    @pytest.mark.parametrize(
        "dist_str",
        ["cost_dist_1d", "cost_dist_2d"],
    )
    def test_distribution_edges(self, dist_str: str, request):
        """Check that the expected distribution is returned when band edges given"""
        cost_dist = request.getfixturevalue(dist_str)
        result = cost_utils.cost_distribution(
            matrix=cost_dist.matrix,
            cost_matrix=cost_dist.cost_matrix,
            bin_edges=cost_dist.bin_edges,
        )
        np.testing.assert_almost_equal(result, cost_dist.distribution)

    @pytest.mark.parametrize(
        "dist_str",
        ["cost_dist_1d", "cost_dist_2d"],
    )
    def test_distribution_bounds(self, dist_str: str, request):
        """Check that the expected distribution is returned when bounds given"""
        cost_dist = request.getfixturevalue(dist_str)
        result = cost_utils.cost_distribution(
            matrix=cost_dist.matrix,
            cost_matrix=cost_dist.cost_matrix,
            min_bounds=cost_dist.min_bounds,
            max_bounds=cost_dist.max_bounds,
        )
        np.testing.assert_almost_equal(result, cost_dist.distribution)

    @pytest.mark.parametrize(
        "dist_str",
        ["cost_dist_1d", "cost_dist_2d"],
    )
    def test_norm_distribution(self, dist_str: str, request):
        """Check that the expected distribution is returned for normalised"""
        cost_dist = request.getfixturevalue(dist_str)
        dist, norm_dist = cost_utils.normalised_cost_distribution(
            matrix=cost_dist.matrix,
            cost_matrix=cost_dist.cost_matrix,
            bin_edges=cost_dist.bin_edges,
        )
        np.testing.assert_almost_equal(dist, cost_dist.distribution)
        np.testing.assert_almost_equal(norm_dist, cost_dist.normalised_distribution)

    @pytest.mark.parametrize(
        "dist_str",
        ["cost_dist_1d", "cost_dist_2d"],
    )
    def test_same_dist(self, dist_str: str, request):
        """Check that the same distribution is returned for both functions"""
        cost_dist = request.getfixturevalue(dist_str)
        dist1, _ = cost_utils.normalised_cost_distribution(
            matrix=cost_dist.matrix,
            cost_matrix=cost_dist.cost_matrix,
            bin_edges=cost_dist.bin_edges,
        )
        dist2 = cost_utils.cost_distribution(
            matrix=cost_dist.matrix,
            cost_matrix=cost_dist.cost_matrix,
            bin_edges=cost_dist.bin_edges,
        )
        np.testing.assert_almost_equal(dist1, dist2)

    @pytest.mark.parametrize("func_name", ["dist", "norm_dist"])
    def test_no_bounds(self, cost_dist_2d: CostDistFnResults, func_name: str):
        """Check an error is raised when no bounds given"""
        msg = (
            "Either `bin_edges` needs to be set, or both `min_bounds` and "
            "`max_bounds` needs to be set."
        )
        if func_name == "dist":
            func = cost_utils.cost_distribution
        elif func_name == "norm_dist":
            func = cost_utils.normalised_cost_distribution  # type: ignore
        else:
            raise ValueError

        with pytest.raises(ValueError, match=msg):
            func(
                matrix=cost_dist_2d.matrix,
                cost_matrix=cost_dist_2d.cost_matrix,
            )

    @pytest.mark.parametrize("func_name", ["dist", "norm_dist"])
    def test_only_min_bounds(self, cost_dist_2d: CostDistFnResults, func_name: str):
        """Check an error is raised when only min bounds given"""
        msg = (
            "Either `bin_edges` needs to be set, or both `min_bounds` and "
            "`max_bounds` needs to be set."
        )
        if func_name == "dist":
            func = cost_utils.cost_distribution
        elif func_name == "norm_dist":
            func = cost_utils.normalised_cost_distribution  # type: ignore
        else:
            raise ValueError

        with pytest.raises(ValueError, match=msg):
            func(
                matrix=cost_dist_2d.matrix,
                cost_matrix=cost_dist_2d.cost_matrix,
                min_bounds=cost_dist_2d.min_bounds,
            )

    @pytest.mark.parametrize("func_name", ["dist", "norm_dist"])
    def test_only_max_bounds(self, cost_dist_2d: CostDistFnResults, func_name: str):
        """Check an error is raised when only max bounds given"""
        msg = (
            "Either `bin_edges` needs to be set, or both `min_bounds` and "
            "`max_bounds` needs to be set."
        )
        if func_name == "dist":
            func = cost_utils.cost_distribution
        elif func_name == "norm_dist":
            func = cost_utils.normalised_cost_distribution  # type: ignore
        else:
            raise ValueError

        with pytest.raises(ValueError, match=msg):
            func(
                matrix=cost_dist_2d.matrix,
                cost_matrix=cost_dist_2d.cost_matrix,
                max_bounds=cost_dist_2d.max_bounds,
            )

    def test_misaligned_bounds_dist(self, cost_dist_2d: CostDistFnResults):
        """Check array of 0s is returned when bounds miss data"""
        new_bin_edges = cost_dist_2d.bin_edges * 1000
        new_bin_edges[0] = 1000
        result = cost_utils.cost_distribution(
            matrix=cost_dist_2d.matrix,
            cost_matrix=cost_dist_2d.cost_matrix,
            bin_edges=new_bin_edges,
        )
        np.testing.assert_almost_equal(result, np.zeros_like(cost_dist_2d.distribution))

    def test_misaligned_bounds_norm(self, cost_dist_2d: CostDistFnResults):
        """Check array of 0s is returned when bounds miss data"""
        new_bin_edges = cost_dist_2d.bin_edges * 1000
        new_bin_edges[0] = 1000
        result, norm_result = cost_utils.normalised_cost_distribution(
            matrix=cost_dist_2d.matrix,
            cost_matrix=cost_dist_2d.cost_matrix,
            bin_edges=new_bin_edges,
        )
        np.testing.assert_almost_equal(result, np.zeros_like(cost_dist_2d.distribution))
        np.testing.assert_almost_equal(norm_result, np.zeros_like(cost_dist_2d.distribution))


@pytest.mark.usefixtures("small_log_bins", "med_log_bins", "large_log_bins")
class TestCreateLogBins:
    """Tests for the create_log_bins function."""

    @pytest.mark.parametrize(
        "io_str",
        ["small_log_bins", "med_log_bins", "large_log_bins"],
    )
    def test_correct_result(self, io_str: str, request):
        """Check that the correct results are returned"""
        input_and_results: LogBinsResults = request.getfixturevalue(io_str)
        result = cost_utils.create_log_bins(**input_and_results.get_kwargs())
        np.testing.assert_almost_equal(result, input_and_results.expected_bins)

    def test_small_final_val(self, small_log_bins: LogBinsResults):
        """Check an error is thrown when the max value is too small."""
        with pytest.raises(ValueError, match="lower than"):
            cost_utils.create_log_bins(**(small_log_bins.get_kwargs() | {"final_val": 0}))

    @pytest.mark.parametrize("n_bin_pow", [-1, 0, 1, 2])
    def test_bad_power(self, small_log_bins: LogBinsResults, n_bin_pow: float):
        """Check an error is thrown when the power is an invalid value."""
        with pytest.raises(ValueError, match="should be in the range"):
            cost_utils.create_log_bins(
                **(small_log_bins.get_kwargs() | {"n_bin_pow": n_bin_pow})
            )

    @pytest.mark.parametrize("log_factor", [-1, 0])
    def test_bad_log_factor(self, small_log_bins: LogBinsResults, log_factor: float):
        """Check an error is thrown when the power is an invalid value."""
        with pytest.raises(ValueError, match="should be greater than 0"):
            cost_utils.create_log_bins(
                **(small_log_bins.get_kwargs() | {"log_factor": log_factor})
            )


@pytest.mark.usefixtures("dynamic_cost_dist_1d", "dynamic_cost_dist_2d")
class TestDynamicCostDistribution:
    """Tests for the dynamic_cost_distribution function.

    No error checks needed as these are carried out in other tests.
    """

    @pytest.mark.parametrize(
        "dist_str",
        ["dynamic_cost_dist_1d", "dynamic_cost_dist_2d"],
    )
    def test_correct_distribution(self, dist_str: str, request):
        """Check that the expected distribution is returned."""
        cost_dist: DynamicCostDistFnResults = request.getfixturevalue(dist_str)
        result, bins = cost_utils.dynamic_cost_distribution(
            matrix=cost_dist.matrix,
            cost_matrix=cost_dist.cost_matrix,
        )
        np.testing.assert_almost_equal(result, cost_dist.distribution)
        np.testing.assert_almost_equal(bins[:-1], cost_dist.bin_edges[:-1])

    @pytest.mark.parametrize(
        "dist_str",
        ["dynamic_cost_dist_1d", "dynamic_cost_dist_2d"],
    )
    def test_kwarg_passing(self, dist_str: str, request):
        """Check that the expected distribution is returned."""
        cost_dist: DynamicCostDistFnResults = request.getfixturevalue(dist_str)
        result, bins = cost_utils.dynamic_cost_distribution(**cost_dist.get_kwargs())
        np.testing.assert_almost_equal(result, cost_dist.distribution)
        np.testing.assert_almost_equal(bins, cost_dist.bin_edges)


class TestIntrazonalCostInfill:
    """Tests for the intrazonal_cost_infill function"""

    @dataclasses.dataclass
    class IzResults:
        """Inputs and expected results for intrazonal_cost_infill()"""

        # Inputs
        cost_matrix: np.ndarray
        multiplier: float
        min_axis: int

        # Results
        result: np.ndarray

        def get_kwargs(self) -> dict[str, Any]:
            return {
                "cost": self.cost_matrix,
                "multiplier": self.multiplier,
                "min_axis": self.min_axis,
            }

    @pytest.fixture(name="normal_array", scope="class")
    def fixture_normal_array(self):
        """Create a simple test with no edge cases"""
        cost_matrix = np.array(
            [
                [54.0, 72.0, 61.0, 97.0, 72.0],
                [41.0, 84.0, 98.0, 32.0, 32.0],
                [4.0, 33.0, 67.0, 14.0, 26.0],
                [73.0, 46.0, 14.0, 8.0, 51.0],
                [2.0, 14.0, 58.0, 53.0, 40.0],
            ]
        )
        result = np.array(
            [
                [61.0, 72.0, 61.0, 97.0, 72.0],
                [41.0, 32.0, 98.0, 32.0, 32.0],
                [4.0, 33.0, 4.0, 14.0, 26.0],
                [73.0, 46.0, 14.0, 14.0, 51.0],
                [2.0, 14.0, 58.0, 53.0, 2.0],
            ]
        )
        return self.IzResults(
            cost_matrix=cost_matrix,
            multiplier=1,
            min_axis=1,
            result=result,
        )

    @pytest.fixture(name="min_axis_array", scope="class")
    def fixture_min_axis_array(self, normal_array):
        """Test for min across columns (different min_axis)."""
        result = np.array(
            [
                [2.0, 72.0, 61.0, 97.0, 72.0],
                [41.0, 14.0, 98.0, 32.0, 32.0],
                [4.0, 33.0, 14.0, 14.0, 26.0],
                [73.0, 46.0, 14.0, 14.0, 51.0],
                [2.0, 14.0, 58.0, 53.0, 26.0],
            ]
        )
        return self.IzResults(
            cost_matrix=normal_array.cost_matrix,
            multiplier=1,
            min_axis=0,
            result=result,
        )

    @pytest.fixture(name="zeroes_array", scope="class")
    def fixture_zeroes_array(self, normal_array):
        """zero values should be returned in-place, expect in diagonal.

        Zero values should not count towards the minimum check.
        """
        idx = (np.array([0, 0, 2, 4, 4]), np.array([0, 1, 1, 2, 4]))
        non_diag_idx = (np.array([0, 2, 4]), np.array([1, 1, 2]))

        cost_matrix = normal_array.cost_matrix.copy()
        cost_matrix[idx] = 0

        result = normal_array.result.copy()
        result[non_diag_idx] = 0

        return self.IzResults(
            cost_matrix=cost_matrix,
            multiplier=normal_array.multiplier,
            min_axis=normal_array.min_axis,
            result=result,
        )

    @pytest.fixture(name="inf_array", scope="class")
    def fixture_inf_array(self, normal_array):
        """Inf values should be returned in-place, expect in diagonal."""
        idx = (np.array([0, 0, 2, 4, 4]), np.array([0, 1, 1, 2, 4]))
        non_diag_idx = (np.array([0, 2, 4]), np.array([1, 1, 2]))

        cost_matrix = normal_array.cost_matrix.copy()
        cost_matrix[idx] = np.inf

        result = normal_array.result.copy()
        result[non_diag_idx] = np.inf

        return self.IzResults(
            cost_matrix=cost_matrix,
            multiplier=normal_array.multiplier,
            min_axis=normal_array.min_axis,
            result=result,
        )

    @pytest.mark.parametrize(
        "io_str", ["normal_array", "zeroes_array", "inf_array", "min_axis_array"]
    )
    def test_correct_result(self, io_str: str, request):
        """Test that the correct results are achieved"""
        io: TestIntrazonalCostInfill.IzResults = request.getfixturevalue(io_str)
        result = cost_utils.intrazonal_cost_infill(**io.get_kwargs())
        np.testing.assert_almost_equal(result, io.result)

    @pytest.mark.parametrize(
        "io_str", ["normal_array", "zeroes_array", "inf_array", "min_axis_array"]
    )
    @pytest.mark.parametrize("multiplier", [0, 0.5, 2])
    def test_different_multiplier(self, io_str: str, multiplier: float, request):
        """Test that the multiplier is being applied correctly."""
        io: TestIntrazonalCostInfill.IzResults = request.getfixturevalue(io_str)

        # Calculate the new result
        new_diag = np.diagonal(io.result) * multiplier
        expected_result = io.result.copy()
        np.fill_diagonal(expected_result, new_diag)

        # Calculate and test the result
        result = cost_utils.intrazonal_cost_infill(
            **io.get_kwargs() | {"multiplier": multiplier}
        )
        np.testing.assert_almost_equal(result, expected_result)
