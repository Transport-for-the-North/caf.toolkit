# -*- coding: utf-8 -*-
"""Tests for the caf.toolkit.math_utils module"""
from __future__ import annotations

# Built-Ins
import math
import dataclasses

from typing import Collection

# Third Party
import pytest
import sparse
import numpy as np


# Local Imports
# pylint: disable=import-error,wrong-import-position
from caf.toolkit import math_utils

# pylint: enable=import-error,wrong-import-position

# # # CONSTANTS # # #


# # # FIXTURES # # #


# # # TESTS # # #
class TestIsAlmostEqual:
    """Tests for caf.toolkit.math_utils.is_almost_equal"""

    @pytest.mark.parametrize("val1", [0, 0.5, 1])
    @pytest.mark.parametrize("val2", [0, 0.5, 1])
    @pytest.mark.parametrize("rel_tol", [0.0001, 0.05, 1.5])
    @pytest.mark.parametrize("abs_tol", [0, 0.5, 10])
    def test_equal_to_builtin(
        self,
        val1: int | float,
        val2: int | float,
        rel_tol: float,
        abs_tol: float,
    ):
        """Test it works exactly like math.isclose"""
        expected = math.isclose(val1, val2, rel_tol=rel_tol, abs_tol=abs_tol)
        got = math_utils.is_almost_equal(
            val1=val1,
            val2=val2,
            rel_tol=rel_tol,
            abs_tol=abs_tol,
        )
        assert expected == got


class TestRootMeanSquaredError:
    """Tests for caf.toolkit.math_utils.root_mean_squared_error"""

    @dataclasses.dataclass
    class RmseExample:
        """Collection of data to pass to an RMSE call"""

        targets: Collection[np.ndarray]
        achieved: Collection[np.ndarray]
        result: float

    @staticmethod
    def get_expected_rmse(
        targets: Collection[np.ndarray | sparse.COO],
        achieved: Collection[np.ndarray | sparse.COO],
    ) -> float:
        """Calculate the expected RMSE score"""
        # Calculate results
        squared_diffs = list()
        for t, a in zip(targets, achieved):
            diffs = (t - a) ** 2
            squared_diffs += diffs.flatten().tolist()
        return float(np.mean(squared_diffs) ** 0.5)

    @pytest.fixture(name="rmse_example", scope="class")
    def fixture_rmse_example(self) -> RmseExample:
        """Generate an example rmse call with result"""
        # Build the target and achieved
        targets = np.array(
            [
                [0, 0, 0, 1],
                [1, 1, 1, 1],
                [0, 1, 0, 1],
                [1, 1, 1, 1],
                [0, 0, 1, 0],
            ]
        )

        achieved = np.array(
            [
                [0.1, 0, 0, 1],
                [0.9, 1, 1, 1],
                [0.7, 1, 0, 1],
                [1.2, 1, 1, 1],
                [1.3, 0, 1, 0],
            ]
        )

        return self.RmseExample(
            targets=targets,
            achieved=achieved,
            result=self.get_expected_rmse(targets, achieved),
        )

    @pytest.fixture(name="rmse_example_1d", scope="class")
    def fixture_rmse_example_1d(self, rmse_example: RmseExample) -> RmseExample:
        """Generate an example 1d rmse call with result"""
        targets = rmse_example.targets[:1]
        achieved = rmse_example.achieved[:1]
        return self.RmseExample(
            targets=targets,
            achieved=achieved,
            result=self.get_expected_rmse(targets, achieved),
        )

    @pytest.mark.parametrize(
        "rmse_example_str",
        ["rmse_example", "rmse_example_1d"],
    )
    def test_numpy_arrays(self, rmse_example_str: str, request):
        """Test that the calculation works for numpy arrays"""
        rmse_example = request.getfixturevalue(rmse_example_str)
        result = math_utils.root_mean_squared_error(
            targets=rmse_example.targets,
            achieved=rmse_example.achieved,
        )
        np.testing.assert_almost_equal(result, rmse_example.result)

    @pytest.mark.parametrize(
        "rmse_example_str",
        ["rmse_example", "rmse_example_1d"],
    )
    def test_sparse_arrays(self, rmse_example_str: str, request):
        """Test that the calculation works for sparse arrays"""
        rmse_example = request.getfixturevalue(rmse_example_str)
        targets = [sparse.COO(x) for x in rmse_example.targets]
        achieved = [sparse.COO(x) for x in rmse_example.achieved]
        result = math_utils.root_mean_squared_error(
            targets=targets,
            achieved=achieved,
        )
        np.testing.assert_almost_equal(result, rmse_example.result)

    def test_unsupported_arrays(self, rmse_example: RmseExample):
        """Test that an error is raised for unsupported arrays"""
        targets = [sparse.GCXS(x) for x in rmse_example.targets]
        achieved = [sparse.GCXS(x) for x in rmse_example.achieved]
        with pytest.raises(TypeError, match="Cannot handle arrays of type"):
            math_utils.root_mean_squared_error(
                targets=targets,
                achieved=achieved,
            )

    def test_different_length_collections(self, rmse_example: RmseExample):
        """Test that an error is raised for different length collections"""
        targets = rmse_example.targets[:1]
        with pytest.raises(ValueError, match="must be the same length"):
            math_utils.root_mean_squared_error(
                targets=targets,
                achieved=rmse_example.achieved,
            )

    def test_non_collections(self, rmse_example: RmseExample):
        """Test that an error is raised for non-collections"""
        with pytest.raises(TypeError, match="Expected a collection"):
            math_utils.root_mean_squared_error(
                targets=1,
                achieved=rmse_example.achieved,
            )

        with pytest.raises(TypeError, match="Expected a collection"):
            math_utils.root_mean_squared_error(
                targets=rmse_example.targets,
                achieved=1,
            )

    def test_different_shape_values(self, rmse_example: RmseExample):
        """Test that an error is raised for different shape collection values"""
        # Create a new target in an un-broadcast-able shape
        targets = np.hstack([rmse_example.targets, rmse_example.targets])[:, :6]
        with pytest.raises(ValueError, match="Could not broadcast"):
            math_utils.root_mean_squared_error(
                targets=targets,
                achieved=rmse_example.achieved,
            )
