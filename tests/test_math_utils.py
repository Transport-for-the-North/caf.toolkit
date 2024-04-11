# -*- coding: utf-8 -*-
"""Tests for the caf.toolkit.math_utils module"""
from __future__ import annotations

# Built-Ins
import dataclasses
import math
import random
from typing import Collection

# Third Party
import numpy as np
import pytest
import sparse

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


class TestCurveConvergence:
    """Tests for caf.toolkit.math_utils.curve_convergence"""

    @dataclasses.dataclass
    class ConvergenceExample:
        """Collection of data to pass to an RMSE call"""

        target: np.ndarray
        achieved: np.ndarray
        result: float

    @staticmethod
    def get_expected_convergence(
        target: np.ndarray,
        achieved: np.ndarray,
    ) -> float:
        """Calculate the expected score"""
        convergence = np.sum((achieved - target) ** 2) / np.sum(
            (target - np.sum(target) / len(target)) ** 2
        )
        return max(1 - convergence, 0)

    @pytest.fixture(name="perfect_match_conv", scope="class")
    def fixture_perfect_match_conv(self) -> ConvergenceExample:
        """Data where there is a perfect match"""
        target = np.arange(10)
        return self.ConvergenceExample(target=target, achieved=target, result=1)

    @pytest.fixture(name="zero_match_conv", scope="class")
    def fixture_zero_match_conv(self) -> ConvergenceExample:
        """Data where there is no match"""
        target = np.arange(10)
        return self.ConvergenceExample(target=target, achieved=np.zeros_like(target), result=0)

    @pytest.fixture(name="random_conv", scope="class")
    def fixture_random_conv(self) -> ConvergenceExample:
        """Data where there is a random match"""
        target = np.arange(10)
        noise = [1 if random.random() > 0.5 else -1 for _ in range(target.shape[0])]
        achieved = target + noise
        return self.ConvergenceExample(
            target=target,
            achieved=achieved,
            result=self.get_expected_convergence(target, achieved),
        )

    @pytest.mark.parametrize(
        "conv_example_str",
        ["perfect_match_conv", "zero_match_conv", "random_conv"],
    )
    def test_correct_results(self, conv_example_str: str, request):
        """Test that the calculation works as expected"""
        conv_example = request.getfixturevalue(conv_example_str)
        result = math_utils.curve_convergence(
            target=conv_example.target,
            achieved=conv_example.achieved,
        )
        np.testing.assert_almost_equal(result, conv_example.result)

    def test_mismatch_shapes(self, perfect_match_conv: ConvergenceExample):
        """Test that an error is thrown with non-matching shapes"""
        new_achieved = perfect_match_conv.achieved.copy()
        new_achieved = np.hstack([new_achieved, new_achieved])
        msg = "Shape of target and achieved do not match"
        with pytest.raises(ValueError, match=msg):
            math_utils.curve_convergence(perfect_match_conv.target, new_achieved)

    def test_nan_target(self, perfect_match_conv: ConvergenceExample):
        """Test that an error is returned when NaN is one of the target values"""
        new_target = perfect_match_conv.target.copy().astype(float)
        new_target[0] = np.nan
        msg = "Found NaN in the target"
        with pytest.warns(UserWarning, match=msg):
            result = math_utils.curve_convergence(
                target=new_target,
                achieved=perfect_match_conv.achieved,
            )
            np.testing.assert_almost_equal(result, 0)

    def test_nan_achieved(self, perfect_match_conv: ConvergenceExample):
        """Test that 0 is returned when NaN is one of the achieved values"""
        new_achieved = perfect_match_conv.achieved.copy().astype(float)
        new_achieved[0] = np.nan
        result = math_utils.curve_convergence(
            target=perfect_match_conv.target,
            achieved=new_achieved,
        )
        np.testing.assert_almost_equal(result, 0)


class TestCheckNumeric:
    """Tests for check_numeric"""

    @pytest.mark.parametrize(
        "value",
        [
            1,
            1.1,
            np.float64(1.1),
            np.int32(1),
            np.uint(1),
            np.short(1),
            np.int_(1),
            np.double(1.1),
        ],
    )
    def test_correct(self, value):
        """Check that no error is raised when correct values passed in"""
        math_utils.check_numeric({"name": value})

    @pytest.mark.parametrize("value", ["str", list(), set(), dict()])
    def test_error(self, value):
        """Check that no error is raised when correct values passed in"""
        msg = "test_name should be a scalar number"
        with pytest.raises(ValueError, match=msg):
            math_utils.check_numeric({"test_name": value})


class TestClipSmallNonZero:
    """Tests for clip_small_non_zero"""

    @dataclasses.dataclass
    class ClipResults:
        """Collection of data to pass to an RMSE call"""

        array_in: np.ndarray
        min_val: float
        array_out: np.ndarray

    @pytest.mark.parametrize("min_val", [-1, -0.1, 0, 0.1, 1])
    def test_no_change(self, min_val: float):
        """Test that no change is made when min_val is too small"""
        array_in = np.arange(10) + 10
        result = math_utils.clip_small_non_zero(array_in, min_val=min_val)
        np.testing.assert_almost_equal(result, array_in)

    @pytest.mark.parametrize("min_val", [-1, -0.1, 0, 0.1, 1])
    def test_no_change_neg_array(self, min_val: float):
        """Test that no change is made when min_val is too small"""
        array_in = np.arange(10) + 10
        array_in[0] = -0.001
        result = math_utils.clip_small_non_zero(array_in, min_val=min_val)
        np.testing.assert_almost_equal(result, array_in)

    @pytest.mark.parametrize("min_val", [1, 5, 7, 20])
    def test_clip(self, min_val: float):
        """Test that no change is made when min_val is too small"""
        array_in = np.arange(10) + 1
        array_out = np.where(array_in < min_val, min_val, array_in)
        result = math_utils.clip_small_non_zero(array_in, min_val=min_val)
        np.testing.assert_almost_equal(result, array_out)
