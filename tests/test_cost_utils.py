# -*- coding: utf-8 -*-
"""Tests for the {} module"""
from __future__ import annotations

# Built-Ins
import dataclasses

# Third Party
import pytest
import numpy as np


# Local Imports
# pylint: disable=import-error,wrong-import-position
from caf.toolkit import cost_utils

# pylint: enable=import-error,wrong-import-position

# # # CONSTANTS # # #


# # # FIXTURES # # #
@dataclasses.dataclass
class CostDistResults:
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


@pytest.fixture(name="cost_dist_1d", scope="class")
def fixture_cost_dist_1d():
    """Create a 1D matrix to distribute"""
    return CostDistResults(
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

    return CostDistResults(
        matrix=matrix,
        cost_matrix=cost_matrix,
        bin_edges=np.array([0, 5, 10, 20, 40, 75, 100]),
        distribution=np.array([106.0, 69.0, 122.0, 210.0, 542.0, 151.0]),
    )


# # # TESTS # # #
@pytest.mark.usefixtures("cost_dist_1d", "cost_dist_2d")
class TestCostDistribution:
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
    def test_no_bounds(self, cost_dist_2d: CostDistResults, func_name: str):
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
    def test_only_min_bounds(self, cost_dist_2d: CostDistResults, func_name: str):
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
    def test_only_max_bounds(self, cost_dist_2d: CostDistResults, func_name: str):
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

    def test_misaligned_bounds_dist(self, cost_dist_2d: CostDistResults):
        """Check array of 0s is returned when bounds miss data"""
        new_bin_edges = cost_dist_2d.bin_edges * 1000
        new_bin_edges[0] = 1000
        result = cost_utils.cost_distribution(
            matrix=cost_dist_2d.matrix,
            cost_matrix=cost_dist_2d.cost_matrix,
            bin_edges=new_bin_edges,
        )
        np.testing.assert_almost_equal(result, np.zeros_like(cost_dist_2d.distribution))

    def test_misaligned_bounds_norm(self, cost_dist_2d: CostDistResults):
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
