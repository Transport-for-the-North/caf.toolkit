# -*- coding: utf-8 -*-
"""Tests for the caf.toolkit.iterative_proportional_fitting module"""
from __future__ import annotations

# Built-Ins
import dataclasses

from typing import Any
from typing import Callable

import pandas as pd
# Third Party
import pytest

import numpy as np
from numpy import testing as np_testing


# Local Imports
# pylint: disable=import-error,wrong-import-position
from caf.toolkit import math_utils
from caf.toolkit import iterative_proportional_fitting

# pylint: enable=import-error,wrong-import-position

# # # CONSTANTS # # #


# # # Classes # # #
@dataclasses.dataclass
class IpfData:
    """Collection of data to pass to an IPF call"""

    matrix: np.ndarray
    marginals: list[np.ndarray]
    dimensions: list[list[int]]
    convergence_fn: Callable = None
    max_iterations: int = 5000
    tol: float = 1e-9
    min_tol_rate: float = 1e-9

    def to_kwargs(self) -> dict[str, Any]:
        """Return a dictionary of key-word arguments"""
        return {
            "seed_mat": self.matrix,
            "target_marginals": self.marginals,
            "target_dimensions": self.dimensions,
            "convergence_fn": self.convergence_fn,
            "max_iterations": self.max_iterations,
            "tol": self.tol,
            "min_tol_rate": self.min_tol_rate,
        }

    def copy(self):
        return IpfData(
            matrix=self.matrix,
            marginals=self.marginals,
            dimensions=self.dimensions,
            convergence_fn=self.convergence_fn,
            max_iterations=self.max_iterations,
            tol=self.tol,
            min_tol_rate=self.min_tol_rate,
        )


@dataclasses.dataclass
class IpfDataPandas:
    """Pandas version of IpfData"""
    matrix: pd.DataFrame
    marginals: list[pd.Series]
    convergence_fn: Callable = None
    max_iterations: int = 5000
    tol: float = 1e-9
    min_tol_rate: float = 1e-9

    def to_kwargs(self) -> dict[str, Any]:
        """Return a dictionary of key-word arguments"""
        return {
            "seed_mat": self.matrix,
            "target_marginals": self.marginals,
            "convergence_fn": self.convergence_fn,
            "max_iterations": self.max_iterations,
            "tol": self.tol,
            "min_tol_rate": self.min_tol_rate,
        }


@dataclasses.dataclass
class IpfDataAndResults:
    """Collection of inputs and expected outputs for and IPF call"""
    inputs: IpfData
    final_matrix: np.ndarray
    completed_iters: int
    final_convergence: float


@dataclasses.dataclass
class IpfDataAndResultsPandas:
    """Collection of inputs and expected outputs for and IPF call"""
    inputs: IpfDataPandas
    final_matrix: np.ndarray
    completed_iters: int
    final_convergence: float


# # # FIXTURES # # #
@pytest.fixture(name="ipf_example", scope="function")
def fixture_ipf_example() -> IpfData:
    """Basic collection of arguments for testing"""
    mat = np.array(
        [
            [[1, 2, 1], [3, 5, 5], [6, 2, 2], [1, 7, 2]],
            [[5, 4, 2], [5, 5, 5], [3, 8, 7], [2, 7, 6]],
        ]
    )

    # Marginals
    xipp = np.array([52, 48], dtype=float)
    xpjp = np.array([20, 30, 35, 15], dtype=float)
    xppk = np.array([35, 40, 25], dtype=float)
    xijp = np.array([[9, 17, 19, 7], [11, 13, 16, 8]], dtype=float)
    xpjk = np.array([[7, 9, 4], [8, 12, 10], [15, 12, 8], [5, 7, 3]], dtype=float)

    # Other params
    marginals = [xipp, xpjp, xppk, xijp, xpjk]
    dimensions = [[0], [1], [2], [0, 1], [1, 2]]

    return IpfData(
        matrix=mat,
        marginals=marginals,
        dimensions=dimensions,
    )


@pytest.fixture(name="ipf_rmse_example_results", scope="function")
def fixture_ipf_rmse_example_results(ipf_example: IpfData) -> IpfDataAndResults:
    """Collection of ipf arguments and results for testing"""
    # Set targets
    # fmt: off
    target_mat = np.array(
        [[[2.15512197, 4.73876153, 2.10611623],
          [3.79236005, 7.20416715, 6.00347262],
          [12.03141147, 4.03511611, 2.93345931],
          [2.02601482, 4.03702642, 0.93695875]],

         [[4.84487803, 4.26123847, 1.89388377],
          [4.20763995, 4.79583285, 3.99652738],
          [2.96858853, 7.96488389, 5.06654069],
          [2.97398518, 2.96297358, 2.06304125]]]
    )
    # fmt: on
    local_ipf_example = ipf_example.copy()
    local_ipf_example.convergence_fn = math_utils.root_mean_squared_error
    return IpfDataAndResults(
        inputs=local_ipf_example,
        final_matrix=target_mat,
        completed_iters=13,
        final_convergence=2.5840564976941704e-10
    )


@pytest.fixture(name="ipf_ipfn_example_results", scope="function")
def fixture_ipf_ipfn_example_results(ipf_example: IpfData) -> IpfDataAndResults:
    """Collection of ipf arguments and results for testing"""
    # Set targets
    # fmt: off
    target_mat = np.array(
        [[[2.15512205, 4.73876166, 2.10611629],
          [3.7923601, 7.20416722, 6.00347268],
          [12.03141598, 4.03512119, 2.93346283],
          [2.02601482, 4.03702643, 0.93695875]],

         [[4.84487795, 4.26123834, 1.89388371],
          [4.2076399, 4.79583278, 3.99652732],
          [2.96858402, 7.96487881, 5.06653717],
          [2.97398518, 2.96297357, 2.06304125]]]
    )
    # fmt: on
    local_ipf_example = ipf_example.copy()
    local_ipf_example.convergence_fn = iterative_proportional_fitting.default_convergence
    return IpfDataAndResults(
        inputs=local_ipf_example,
        final_matrix=target_mat,
        completed_iters=12,
        final_convergence=2.209143978859629e-10
    )


@pytest.fixture(name="pandas_ipf_example", scope="function")
def fixture_pandas_ipf_example(ipf_example: IpfData) -> IpfDataPandas:
    """Pandas wrapper for `fixture_ipf_example`"""
    # Convert the matrix
    # fmt: off
    dma = [
        501, 501, 501, 501, 501, 501, 501, 501, 501, 501, 501, 501,
        502, 502, 502, 502, 502, 502, 502, 502, 502, 502, 502, 502,
    ]
    size = [
        1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4,
        1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4,
    ]

    age = [
        '20-25', '30-35', '40-45',
        '20-25', '30-35', '40-45',
        '20-25', '30-35', '40-45',
        '20-25', '30-35', '40-45',
        '20-25', '30-35', '40-45',
        '20-25', '30-35', '40-45',
        '20-25', '30-35', '40-45',
        '20-25', '30-35', '40-45',
    ]
    # fmt: on

    pd_mat = pd.DataFrame(
        data=ipf_example.matrix.flatten(),
        index=pd.MultiIndex([dma, size, age], names=["dma", "size", "age"]),
        columns="total",
    )

    # Convert the marginals
    xipp = pd_mat.groupby("dma")["total"].sum()
    xpjp = pd_mat.groupby("size")["total"].sum()
    xppk = pd_mat.groupby("age")["total"].sum()
    xijp = pd_mat.groupby(["dma", "size"])["total"].sum()
    xpjk = pd_mat.groupby(["size", "age"])["total"].sum()

    xipp.values = ipf_example.marginals[0]
    xpjp.values = ipf_example.marginals[1]
    xppk.values = ipf_example.marginals[2]
    xijp.values = ipf_example.marginals[3]
    xpjk.values = ipf_example.marginals[4]

    marginals = [xipp, xpjp, xppk, xijp, xpjk]

    # Wrap in an object
    return IpfDataPandas(
        matrix=pd_mat,
        marginals=marginals
    )


# # # TESTS # # #
@pytest.mark.usefixtures("ipf_example", "ipf_rmse_example_results", "ipf_ipfn_example_results")
class TestIpf:
    """Tests for caf.toolkit.iterative_proportional_fitting.ipf"""
    def test_invalid_marginals(self, ipf_example: IpfData):
        """Test that invalid marginals raise a warning"""
        ipf_example.marginals[0] /= 2
        with pytest.warns(UserWarning, match="do not sum to similar amounts"):
            iterative_proportional_fitting.ipf(**ipf_example.to_kwargs())

    def test_too_many_dimensions(self, ipf_example: IpfData):
        """Test that invalid dimensions (too many) raise an error"""
        ipf_example.dimensions[0] = [0, 1, 2, 3, 4]
        with pytest.raises(ValueError, match="Too many dimensions"):
            iterative_proportional_fitting.ipf(**ipf_example.to_kwargs())

    def test_too_high_dimensions(self, ipf_example: IpfData):
        """Test that invalid dimensions (too high) raise an error"""
        ipf_example.dimensions[0] = [3, 4]
        with pytest.raises(ValueError, match="Dimension numbers too high."):
            iterative_proportional_fitting.ipf(**ipf_example.to_kwargs())

    def test_marginal_shapes(self, ipf_example: IpfData):
        """Test that invalid marginal shapes raises an error"""
        bad_marginal = ipf_example.marginals[0] / 2
        bad_marginal = np.broadcast_to(np.expand_dims(bad_marginal, axis=1), (2, 2))
        ipf_example.marginals[0] = bad_marginal

        with pytest.raises(ValueError, match="Marginal is not the expected shape"):
            iterative_proportional_fitting.ipf(**ipf_example.to_kwargs())

    def test_rmse_convergence(self, ipf_rmse_example_results: IpfDataAndResults):
        """Test that correct result calculated with RMSE convergence"""
        # Run
        mat, iters, conv = iterative_proportional_fitting.ipf(
            **ipf_rmse_example_results.inputs.to_kwargs()
        )

        # Check the results
        print(ipf_rmse_example_results.inputs.to_kwargs())
        np_testing.assert_allclose(mat, ipf_rmse_example_results.final_matrix, rtol=1e-4)
        assert iters == ipf_rmse_example_results.completed_iters
        assert math_utils.is_almost_equal(conv, ipf_rmse_example_results.final_convergence)

    def test_ipfn_convergence(self, ipf_ipfn_example_results: IpfDataAndResults):
        """Test that correct result calculated with ipfn convergence"""
        # Run
        mat, iters, conv = iterative_proportional_fitting.ipf(
            **ipf_ipfn_example_results.inputs.to_kwargs()
        )

        # Check the results
        np_testing.assert_allclose(mat, ipf_ipfn_example_results.final_matrix, rtol=1e-4)
        assert iters == ipf_ipfn_example_results.completed_iters
        assert math_utils.is_almost_equal(conv, ipf_ipfn_example_results.final_convergence)

    def test_zero_marginals(self, ipf_example: IpfData):
        """Test warning and return when marginals are 0"""
        # Set targets
        target_mat = np.zeros_like(ipf_example.matrix)
        target_iters = -1
        target_conv = np.inf

        # Create bad marginals
        bad_marginals = list()
        for marginal in ipf_example.marginals:
            bad_marginals.append(np.zeros_like(marginal))
        ipf_example.marginals = bad_marginals

        # Check for warning
        with pytest.warns(UserWarning, match="Given target_marginals of 0"):
            mat, iters, conv = iterative_proportional_fitting.ipf(**ipf_example.to_kwargs())

        # Check the results
        np_testing.assert_allclose(mat, target_mat, rtol=1e-4)
        assert iters == target_iters
        assert math_utils.is_almost_equal(conv, target_conv)

    def test_non_early_exit(self, ipf_example: IpfData):
        """Test warning and return when marginals are 0"""
        # Set targets
        target_mat = ipf_example.matrix
        target_iters = 0
        target_conv = np.inf

        # Check for warning
        with pytest.warns(UserWarning, match="exhausted its max number of loops"):
            ipf_example.max_iterations = 0
            mat, iters, conv = iterative_proportional_fitting.ipf(**ipf_example.to_kwargs())

        # Check the results
        np_testing.assert_allclose(mat, target_mat, rtol=1e-4)
        assert iters == target_iters
        assert math_utils.is_almost_equal(conv, target_conv)


# TODO(BT): Tests for ipf pandas
# TODO(BT): Test that ipf pandas conversions make numpy matrices