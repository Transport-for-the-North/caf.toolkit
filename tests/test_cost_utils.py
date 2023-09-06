# -*- coding: utf-8 -*-
"""Tests for the {} module"""
from __future__ import annotations

# Built-Ins
import dataclasses

from typing import Any

# Third Party
import pytest
import numpy as np


# Local Imports
# pylint: disable=import-error,wrong-import-position
from caf.toolkit import cost_utils

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
        return{
            "max_value": self.max_value,
            "n_bin_pow": self.n_bin_pow,
            "log_factor": self.log_factor,
            "final_val": self.final_val,
        }


# # # FIXTURES # # #
@pytest.fixture(name="cost_dist_1d", scope="class")
def fixture_cost_dist_1d():
    """Create a 1D matrix to distribute"""
    return CostDistFnResults(
        matrix=np.array([26.0, 43.0, 5.0, 8.0, 18.0, 51.0, 35.0, 39.0, 32.0, 37.0]),
        cost_matrix=np.array([77.0, 74.0, 53.0, 60.0, 94.0, 65.0, 13.0, 79.0, 39.0, 75.0]),
        bin_edges=np.array([0, 5, 10, 20, 40, 75, 100]),
        distribution=np.array([0.0, 0.0, 35.0, 32.0, 107.0, 120.0]),
    )


@pytest.fixture(name="cost_dist_2d", scope="class")
def fixture_cost_dist_2d():
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


# # # TESTS # # #
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


class TestCostDistributionClass:
    """Tests for the CostDistribution class."""

    pass


@pytest.mark.usefixtures("small_log_bins", "med_log_bins", "large_log_bins")
class TestCreateLogBins:
    """Tests for the create_log_bins function."""

    @pytest.mark.parametrize(
        "io_str",
        ["small_log_bins", "med_log_bins", "large_log_bins"],
    )
    def test_correct_result(self, io_str: str, request):
        """Check that the correct results are returned"""
        io: LogBinsResults = request.getfixturevalue(io_str)
        result = cost_utils.create_log_bins(**io.get_kwargs())
        np.testing.assert_almost_equal(result, io.expected_bins)

    def test_small_final_val(self, small_log_bins: LogBinsResults):
        """Check an error is thrown when the max value is too small."""
        with pytest.raises(ValueError, match="lower than"):
            cost_utils.create_log_bins(
                **(small_log_bins.get_kwargs() | {"final_val": 0})
            )

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
        with pytest.raises(ValueError, match="should be in the range"):
            cost_utils.create_log_bins(
                **(small_log_bins.get_kwargs() | {"log_factor": log_factor})
            )


class TestDynamicCostDistribution:
    """Tests for the cost distribution helper functions."""

    # Tests for:
    # create_log_bins
    # dynamic_cost_distribution

    pass
