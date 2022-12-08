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
    """Generate a random sparse matrix of N dimensions"""
    dim_len = 10
    sparsity = 0.99
    shape = tuple(dim_len for _ in range(n_dims))

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
    """Tests for the spare_sum function"""

    @pytest.mark.parametrize("random_sparse_matrix", (1, 2, 3, 4), indirect=True)
    @pytest.mark.parametrize("repeat", range(3))
    def test_sum_all_axis(self, random_sparse_matrix: sparse.COO, repeat: int):
        """Test that all axis can be summed together"""
        del repeat
        target = random_sparse_matrix.sum()
        achieved = array_utils.sparse_sum(random_sparse_matrix)
        np.testing.assert_almost_equal(achieved, target)

    @pytest.mark.parametrize("sum_axis", axis_permutations(3))
    def test_sum_axis_subset(self, random_3d_sparse_matrix: sparse.COO, sum_axis: tuple[int, ...]):
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


# TODO: Tests for sparse broadcast
