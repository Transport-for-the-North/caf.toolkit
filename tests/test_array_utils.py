# -*- coding: utf-8 -*-
"""Tests for the caf.toolkit.array_utils module"""
# Built-Ins
import itertools

# Third Party
import pytest
import sparse
import numpy as np


# Local Imports
# pylint: disable=import-error,wrong-import-position
from caf.toolkit import array_utils

# pylint: enable=import-error,wrong-import-position

# # # CONSTANTS # # #


# # # FIXTURES # # #
def make_sparse_flag_matrix(shape: tuple[int, ...], sparsity: float) -> np.ndarray:
    """Make a matrix of 1s and 0s of a certain sparsity"""
    sparse_mat = np.zeros(shape)
    while sparse_mat.sum() == 0:
        sparse_mat = np.random.choice([0, 1], size=shape, p=[sparsity, 1 - sparsity])
    return sparse_mat


def random_sparse_nd_matrix(n_dims: int) -> np.ndarray:
    """Generate a random sparse matrix of N dimensions

    Made with prime numbers as dimensions to ensure broadcasting isn't happening
    behind the scenes to help tests pass
    """
    dim_lengths = [13, 11, 7, 5, 3]
    sparsity = 0.99
    shape = tuple(dim_lengths[:n_dims])

    # Create the sparse array
    array = np.random.random(shape)
    array *= make_sparse_flag_matrix(shape, sparsity)
    return sparse.COO(array)


@pytest.fixture(name="random_sparse_matrix", scope="function")
def fixture_random_sparse_matrix(request) -> np.ndarray:
    """Generate a random sparse matrix of N dimensions"""
    return random_sparse_nd_matrix(request.param)


@pytest.fixture(name="random_3d_sparse_matrix", scope="function")
def fixture_random_3d_sparse_matrix() -> sparse.COO:
    """Generate a random sparse matrix with 3 dimensions"""
    return random_sparse_nd_matrix(3)


def axis_permutations(n_dims: int) -> list[tuple[int, ...]]:
    """Generate all possible axis combinations for matrix of N dimensions"""
    perms = list()
    axis_list = range(n_dims)
    for i in range(1, n_dims):
        perms += list(itertools.permutations(axis_list, i))
    return perms


# # # TESTS # # #
class TestSparseSum:
    """Tests for the sparse_sum function"""

    @pytest.mark.parametrize("random_sparse_matrix", (1, 2, 3, 4), indirect=True)
    @pytest.mark.parametrize("repeat", range(3))
    def test_sum_all_axis(self, random_sparse_matrix: sparse.COO, repeat: int):
        """Test that all axis can be summed together"""
        del repeat
        target = random_sparse_matrix.sum()
        achieved = array_utils.sparse_sum(random_sparse_matrix)
        np.testing.assert_almost_equal(achieved, target)

    @pytest.mark.parametrize("sum_axis", axis_permutations(3))
    def test_sum_axis_subset(
        self, random_3d_sparse_matrix: sparse.COO, sum_axis: tuple[int, ...]
    ):
        """Test that all axis can be summed together"""
        target = random_3d_sparse_matrix.sum(axis=sum_axis)
        achieved = array_utils.sparse_sum(random_3d_sparse_matrix, axis=sum_axis)
        np.testing.assert_almost_equal(achieved.todense(), target.todense())

    def test_sum_axis_int(self, random_3d_sparse_matrix: sparse.COO):
        """Test that all axis can be summed together"""
        target = random_3d_sparse_matrix.sum(axis=1)
        achieved = array_utils.sparse_sum(random_3d_sparse_matrix, axis=1)
        np.testing.assert_almost_equal(achieved.todense(), target.todense())

    def test_non_sparse(self, random_3d_sparse_matrix: sparse.COO):
        """Test that an error is thrown when a dense matrix is given"""
        with pytest.raises(AttributeError, match="object has no attribute 'coords'"):
            array_utils.sparse_sum(random_3d_sparse_matrix.todense())


class TestSparseBroadcast:
    """Tests for the sparse_broadcast function"""

    @pytest.mark.parametrize("random_sparse_matrix", (2, 3, 4), indirect=True)
    @pytest.mark.parametrize("repeat", range(3))
    def test_1dim_broadcast(self, random_sparse_matrix: sparse.COO, repeat: int):
        """Test different random matrices of different dimensions"""
        # init
        del repeat
        sum_axis = 0

        # Generate the input and expected output
        input_mat = random_sparse_matrix.sum(axis=sum_axis)
        input_mat.data = np.ones_like(input_mat.data)

        expected_output = random_sparse_matrix.copy()
        expected_output.data = np.ones_like(expected_output.data)

        # Run and validate
        got = array_utils.broadcast_sparse_matrix(
            array=input_mat,
            target_array=random_sparse_matrix,
            array_dims=tuple(set(range(random_sparse_matrix.ndim)) - {sum_axis}),
        )
        np.testing.assert_almost_equal(got.todense(), expected_output.todense())

    @pytest.mark.parametrize("random_sparse_matrix", (1, 2, 3), indirect=True)
    @pytest.mark.parametrize("repeat", range(3))
    def test_scalar_broadcast(self, random_sparse_matrix: sparse.COO, repeat: int):
        """Test broadcast of scalar values to different dimensions"""
        # init
        del repeat
        expected_output = random_sparse_matrix.copy()
        expected_output.data = np.ones_like(expected_output.data)

        # Run and validate
        got = array_utils.broadcast_sparse_matrix(
            array=1,
            target_array=random_sparse_matrix,
        )
        np.testing.assert_almost_equal(got.todense(), expected_output.todense())

    @pytest.mark.parametrize("sum_axis", axis_permutations(3))
    def test_ndim_broadcast(
        self, random_3d_sparse_matrix: sparse.COO, sum_axis: tuple[int, ...]
    ):
        """Test that broadcast works across different axis"""
        # Generate the input and expected output
        input_mat = random_3d_sparse_matrix.sum(axis=sum_axis)
        input_mat.data = np.ones_like(input_mat.data)

        expected_output = random_3d_sparse_matrix.copy()
        expected_output.data = np.ones_like(expected_output.data)

        # Run and validate
        got = array_utils.broadcast_sparse_matrix(
            array=input_mat,
            target_array=random_3d_sparse_matrix,
            array_dims=tuple(set(range(random_3d_sparse_matrix.ndim)) - set(list(sum_axis))),
        )
        np.testing.assert_almost_equal(got.todense(), expected_output.todense())

    def test_scalar_array_dims(self, random_3d_sparse_matrix: sparse.COO):
        """Test that a scalar value can be given for aray_dims"""
        sum_axis = (1, 2)

        # Generate the input
        input_mat = random_3d_sparse_matrix.sum(axis=sum_axis)
        input_mat.data = np.ones_like(input_mat.data)

        expected_output = random_3d_sparse_matrix.copy()
        expected_output.data = np.ones_like(expected_output.data)

        # Run test
        got = array_utils.broadcast_sparse_matrix(
            array=input_mat,
            target_array=random_3d_sparse_matrix,
            array_dims=0,
        )
        np.testing.assert_almost_equal(got.todense(), expected_output.todense())

    def test_wrong_array_dims_count(self, random_3d_sparse_matrix: sparse.COO):
        """Test that an error is raised when array dims aren't expected shape"""
        # Generate the input
        input_mat = random_3d_sparse_matrix.sum(axis=0)
        input_mat.data = np.ones_like(input_mat.data)

        # Run test
        with pytest.raises(ValueError, match="array_dims must have the same number of values"):
            array_utils.broadcast_sparse_matrix(
                array=input_mat,
                target_array=random_3d_sparse_matrix,
                array_dims=(3, 4, 5, 6),
            )

    def test_no_array_dims(self, random_3d_sparse_matrix: sparse.COO):
        """Test that an error is raised when array dims aren't given when needed"""
        # Generate the input
        input_mat = random_3d_sparse_matrix.sum(axis=0)
        input_mat.data = np.ones_like(input_mat.data)

        # Run test
        with pytest.raises(ValueError, match="array_dims must be given"):
            array_utils.broadcast_sparse_matrix(
                array=input_mat,
                target_array=random_3d_sparse_matrix,
                array_dims=None,
            )

    def test_dimension_mismatch(self, random_3d_sparse_matrix: sparse.COO):
        """Test that an error is raised when axes don't line up"""
        # Generate the input
        input_mat = random_3d_sparse_matrix.sum(axis=0)
        input_mat.data = np.ones_like(input_mat.data)

        # Run test
        with pytest.raises(ValueError, match="Unable to broadcast"):
            array_utils.broadcast_sparse_matrix(
                array=input_mat,
                target_array=random_3d_sparse_matrix.T,
                array_dims=(1, 2),
            )


class TestValidateAxis:
    """Tests for the validate_axis function"""

    def test_duplicated_axis_vals(self):
        """Test that an error is raised when axis values are duplicated"""
        # Run test
        with pytest.raises(ValueError, match="axis values are not unique"):
            array_utils.validate_axis(
                axis=(1, 1),
                n_dims=2,
                name="axis",
            )

    def test_invalid_axis_vals(self):
        """Test that an error is raised when invalid axis values are given"""
        # Run test
        with pytest.raises(ValueError, match="axis values too high"):
            array_utils.validate_axis(
                axis=(3, 1),
                n_dims=3,
                name="axis",
            )

    def test_negative_axis_vals(self):
        """Test that an error is raised when negative axis values are given"""
        # Run test
        with pytest.raises(ValueError, match="axis values cannot be negative"):
            array_utils.validate_axis(
                axis=(-3, 1),
                n_dims=3,
                name="axis",
            )
