# -*- coding: utf-8 -*-
"""Tests for the caf.toolkit.array_utils module"""
# Built-Ins


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


@pytest.fixture(name="random_sparse_matrix", scope="function")
def fixture_random_sparse_matrix(request) -> np.ndarray:
    """Generate a random sparse numpy matrix of N dimensions"""
    dim_len = 10
    sparsity = 0.99
    shape = tuple(dim_len for _ in range(request.param))

    # Create the sparse array
    array = np.random.random(shape)
    array *= make_sparse_flag_matrix(shape, sparsity)
    return sparse.COO(array)


# # # TESTS # # #
@pytest.mark.usefixtures("random_sparse_matrix")
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

    # TODO: Test summing various axis combinations 4d only?
    # TODO: Test giving non-sparse matrix (scalar, dense)
    # TODO: Test giving non-iterable


# TODO: Tests for sparse broadcast
