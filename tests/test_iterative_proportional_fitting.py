# -*- coding: utf-8 -*-
"""Tests for the caf.toolkit.iterative_proportional_fitting module"""
from __future__ import annotations

# Built-Ins
import re
import dataclasses

from typing import Any
from typing import Optional
from typing import Callable


# Third Party
import pytest
import sparse
import numpy as np
import pandas as pd

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
    convergence_fn: Optional[Callable] = None
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
        """Make a copt of this instance"""
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
    value_col: str = "total"
    convergence_fn: Optional[Callable] = None
    max_iterations: int = 5000
    tol: float = 1e-9
    min_tol_rate: float = 1e-9

    def to_kwargs(self) -> dict[str, Any]:
        """Return a dictionary of key-word arguments"""
        return {
            "seed_df": self.matrix,
            "target_marginals": self.marginals,
            "value_col": self.value_col,
            "convergence_fn": self.convergence_fn,
            "max_iterations": self.max_iterations,
            "tol": self.tol,
            "min_tol_rate": self.min_tol_rate,
        }

    def copy(self):
        """Make a copt of this instance"""
        return IpfDataPandas(
            matrix=self.matrix,
            marginals=self.marginals,
            value_col=self.value_col,
            convergence_fn=self.convergence_fn,
            max_iterations=self.max_iterations,
            tol=self.tol,
            min_tol_rate=self.min_tol_rate,
        )


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
@pytest.fixture(name="ipf_example_index", scope="function")
def fixture_ipf_example_index() -> pd.MultiIndex:
    """Generate the pandas MultiIndex for the pandas examples"""
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
    return pd.MultiIndex.from_arrays([dma, size, age], names=["dma", "size", "age"])


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
        final_convergence=2.5840564976941704e-10,
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
        final_convergence=2.209143978859629e-10,
    )


@pytest.fixture(name="pandas_ipf_example", scope="function")
def fixture_pandas_ipf_example(
    ipf_example: IpfData,
    ipf_example_index: pd.MultiIndex,
) -> IpfDataPandas:
    """Pandas wrapper for `fixture_ipf_example`"""
    # Convert the matrix
    pd_mat = pd.DataFrame(
        data=ipf_example.matrix.flatten(),
        index=ipf_example_index,
        columns=["total"],
    ).reset_index()

    # Convert the marginals
    xipp = pd_mat.groupby("dma")["total"].sum()
    xpjp = pd_mat.groupby("size")["total"].sum()
    xppk = pd_mat.groupby("age")["total"].sum()
    xijp = pd_mat.groupby(["dma", "size"])["total"].sum()
    xpjk = pd_mat.groupby(["size", "age"])["total"].sum()

    def _convert(template: pd.DataFrame, marginal_idx: int) -> pd.DataFrame:
        """Convert marginals"""
        return pd.Series(
            data=ipf_example.marginals[marginal_idx].flatten(),
            index=template.index,
        )

    marginals = [
        _convert(xipp, 0),
        _convert(xpjp, 1),
        _convert(xppk, 2),
        _convert(xijp, 3),
        _convert(xpjk, 4),
    ]

    # Wrap in an object
    return IpfDataPandas(matrix=pd_mat, marginals=marginals)


@pytest.fixture(name="ipf_pandas_rmse_example_results", scope="function")
def fixture_pandas_ipf_rmse_example_results(
    pandas_ipf_example: IpfDataPandas,
    ipf_rmse_example_results: IpfDataAndResults,
    ipf_example_index: pd.MultiIndex,
) -> IpfDataAndResultsPandas:
    """Pandas wrapper for `ipf_rmse_example_results`"""
    # Convert the matrix
    target_df = pd.DataFrame(
        data=ipf_rmse_example_results.final_matrix.flatten(),
        index=ipf_example_index,
        columns=["total"],
    ).reset_index()

    local_ipf_example = pandas_ipf_example.copy()
    local_ipf_example.convergence_fn = math_utils.root_mean_squared_error
    return IpfDataAndResultsPandas(
        inputs=local_ipf_example,
        final_matrix=target_df,
        completed_iters=ipf_rmse_example_results.completed_iters,
        final_convergence=ipf_rmse_example_results.final_convergence,
    )


@pytest.fixture(name="ipf_pandas_ipfn_example_results", scope="function")
def fixture_pandas_ipf_ipfn_example_results(
    pandas_ipf_example: IpfDataPandas,
    ipf_ipfn_example_results: IpfDataAndResults,
    ipf_example_index: pd.MultiIndex,
) -> IpfDataAndResultsPandas:
    """Pandas wrapper for `ipf_rmse_example_results`"""
    # Convert the matrix
    target_df = pd.DataFrame(
        data=ipf_ipfn_example_results.final_matrix.flatten(),
        index=ipf_example_index,
        columns=["total"],
    ).reset_index()

    local_ipf_example = pandas_ipf_example.copy()
    local_ipf_example.convergence_fn = iterative_proportional_fitting.default_convergence
    return IpfDataAndResultsPandas(
        inputs=local_ipf_example,
        final_matrix=target_df,
        completed_iters=ipf_ipfn_example_results.completed_iters,
        final_convergence=ipf_ipfn_example_results.final_convergence,
    )


@pytest.fixture(name="ipf_invalid_combos", scope="function")
def fixture_ipf_invalid_combos(ipf_example: IpfData):
    """Collection of arguments where the index is not a product of all values"""
    # Remove any combinations where 1st col is 501, and second is 1
    matrix = ipf_example.matrix.copy()
    matrix[0, 0, :] = 0

    marginals = ipf_example.marginals.copy()
    marginals[3] = np.array([[0, 18, 21, 9], [12, 14, 17, 9]], dtype=float)

    return IpfData(
        matrix=matrix,
        marginals=marginals,
        dimensions=ipf_example.dimensions.copy(),
    )


@pytest.fixture(name="ipf_pandas_invalid_combos", scope="function")
def fixture_ipf_pandas_invalid_combos(
    ipf_invalid_combos: IpfData,
    ipf_example_index: pd.MultiIndex,
) -> IpfDataPandas:
    """Pandas wrapper for `fixture_ipf_example`"""
    # Convert the matrix
    pd_mat = pd.DataFrame(
        data=ipf_invalid_combos.matrix.flatten().tolist(),
        index=ipf_example_index,
        columns=["total"],
    ).reset_index()
    pd_mat = pd_mat.loc[3:].copy()

    # Convert the marginals
    xipp = pd_mat.groupby("dma")["total"].sum()
    xpjp = pd_mat.groupby("size")["total"].sum()
    xppk = pd_mat.groupby("age")["total"].sum()
    xijp = pd_mat.groupby(["dma", "size"])["total"].sum()
    xpjk = pd_mat.groupby(["size", "age"])["total"].sum()

    def _convert(template: pd.DataFrame, marginal_idx: int) -> pd.DataFrame:
        """Convert marginals"""
        vals = ipf_invalid_combos.marginals[marginal_idx].flatten()
        if marginal_idx == 3:
            vals = vals[1:]

        return pd.Series(data=vals, index=template.index)

    marginals = [
        _convert(xipp, 0),
        _convert(xpjp, 1),
        _convert(xppk, 2),
        _convert(xijp, 3),
        _convert(xpjk, 4),
    ]

    # Wrap in an object
    return IpfDataPandas(matrix=pd_mat, marginals=marginals)


@pytest.fixture(name="ipf_invalid_combos_results", scope="function")
def fixture_ipf_invalid_combos_results(ipf_invalid_combos: IpfData) -> IpfDataAndResults:
    """Collection of ipf arguments and results for testing"""
    # Set targets
    # fmt: off
    target_mat = np.array(
        [[[0.,  0.,  0.],
          [3.75016167,  7.14309176,  5.95257647],
          [12.19680352,  4.22737148,  3.06652879],
          [2.19480828,  4.27076229,  1.02839926]],

         [[7., 9., 4.],
          [4.24983833,  4.85690824,  4.04742353],
          [2.80319648,  7.77262852,  4.93347121],
          [2.80519172,  2.72923771,  1.97160074]]]
    )
    # fmt: on
    local_ipf_example = ipf_invalid_combos.copy()
    local_ipf_example.convergence_fn = math_utils.root_mean_squared_error
    return IpfDataAndResults(
        inputs=local_ipf_example,
        final_matrix=target_mat,
        completed_iters=12,
        final_convergence=2.681540527090612,
    )


@pytest.fixture(name="ipf_pandas_invalid_combos_results", scope="function")
def fixture_ipf_pandas_invalid_combos_results(
    ipf_pandas_invalid_combos: IpfDataPandas,
    ipf_invalid_combos_results: IpfDataAndResults,
    ipf_example_index: pd.MultiIndex,
) -> IpfDataAndResultsPandas:
    """Pandas wrapper for `ipf_rmse_example_results`"""
    # Convert the matrix
    target_df = pd.DataFrame(
        data=ipf_invalid_combos_results.final_matrix.flatten().tolist(),
        index=ipf_example_index,
        columns=["total"],
    ).reset_index()
    target_df = target_df.loc[3:].reset_index(drop=True)

    local_ipf_example = ipf_pandas_invalid_combos.copy()
    local_ipf_example.convergence_fn = math_utils.root_mean_squared_error
    return IpfDataAndResultsPandas(
        inputs=local_ipf_example,
        final_matrix=target_df,
        completed_iters=ipf_invalid_combos_results.completed_iters,
        final_convergence=ipf_invalid_combos_results.final_convergence,
    )


# # # TESTS # # #
@pytest.mark.usefixtures(
    "ipf_example",
    "ipf_rmse_example_results",
    "ipf_ipfn_example_results",
    "ipf_invalid_combos_results",
)
class TestIpf:
    """Tests for caf.toolkit.iterative_proportional_fitting.ipf"""

    def test_seed_mat_value_type(self, ipf_example: IpfData):
        """Test that invalid seed_mat value type raises an error"""
        seed_mat = ipf_example.matrix.copy()
        seed_mat = seed_mat.astype(str)
        with pytest.raises(TypeError, match="expected to be numeric type"):
            iterative_proportional_fitting.ipf(
                **(ipf_example.to_kwargs() | {"seed_mat": seed_mat})
            )

    def test_marginal_value_type(self, ipf_example: IpfData):
        """Test that invalid marginal value type raises an error"""
        marginals = ipf_example.marginals.copy()
        marginals[0] = marginals[0].astype(str)
        with pytest.raises(TypeError, match="expected to be numeric type"):
            iterative_proportional_fitting.ipf(
                **(ipf_example.to_kwargs() | {"target_marginals": marginals})
            )

    def test_dimension_value_type(self, ipf_example: IpfData):
        """Test that invalid dimension value type raises an error"""
        dimensions = ipf_example.dimensions.copy()
        dimensions[0] = np.array(dimensions[0]).astype(str).tolist()
        with pytest.raises(TypeError, match="expected to be numeric type"):
            iterative_proportional_fitting.ipf(
                **(ipf_example.to_kwargs() | {"target_dimensions": dimensions})
            )

    def test_invalid_seed_mat(self, ipf_example: IpfData):
        """Test that invalid seed_mat type raises an error"""
        with pytest.raises(TypeError, match="is not an np.ndarray"):
            iterative_proportional_fitting.ipf(**(ipf_example.to_kwargs() | {"seed_mat": 1}))

    def test_pandas_seed_mat(self, ipf_example: IpfData):
        """Test that a pandas seed_mat raises an error"""
        seed_df = pd.DataFrame(ipf_example.matrix.flatten(), columns=["name"])
        with pytest.raises(TypeError, match=re.escape("call `ipf_dataframe()` instead")):
            iterative_proportional_fitting.ipf(
                **(ipf_example.to_kwargs() | {"seed_mat": seed_df})
            )

    def test_invalid_marginals(self, ipf_example: IpfData):
        """Test that invalid marginals raise a warning"""
        bad_marginals = ipf_example.marginals.copy()
        bad_marginals[0] /= 2
        with pytest.warns(UserWarning, match="do not sum to similar amounts"):
            iterative_proportional_fitting.ipf(
                **(ipf_example.to_kwargs() | {"target_marginals": bad_marginals})
            )

    def test_too_many_dimensions(self, ipf_example: IpfData):
        """Test that invalid dimensions (too many) raise an error"""
        bad_dimensions = ipf_example.dimensions.copy()
        bad_dimensions[0] = [0, 1, 2, 3, 4]
        with pytest.raises(ValueError, match="Too many dimensions"):
            iterative_proportional_fitting.ipf(
                **(ipf_example.to_kwargs() | {"target_dimensions": bad_dimensions})
            )

    def test_too_high_dimensions(self, ipf_example: IpfData):
        """Test that invalid dimensions (too high) raise an error"""
        bad_dimensions = ipf_example.dimensions.copy()
        bad_dimensions[0] = [3, 4]
        with pytest.raises(ValueError, match="Dimension numbers too high."):
            iterative_proportional_fitting.ipf(
                **(ipf_example.to_kwargs() | {"target_dimensions": bad_dimensions})
            )

    def test_marginal_shapes(self, ipf_example: IpfData):
        """Test that invalid marginal shapes raises an error"""
        marginals = ipf_example.marginals.copy()
        bad_marginal = marginals[0] / 2
        bad_marginal = np.broadcast_to(np.expand_dims(bad_marginal, axis=1), (2, 2))
        marginals[0] = bad_marginal

        with pytest.raises(ValueError, match="Marginal is not the expected shape"):
            iterative_proportional_fitting.ipf(
                **(ipf_example.to_kwargs() | {"target_marginals": marginals})
            )

    def test_rmse_convergence(self, ipf_rmse_example_results: IpfDataAndResults):
        """Test that correct result calculated with RMSE convergence"""
        # Run
        mat, iters, conv = iterative_proportional_fitting.ipf(
            **ipf_rmse_example_results.inputs.to_kwargs()
        )

        # Check the results
        np_testing.assert_allclose(mat, ipf_rmse_example_results.final_matrix, rtol=1e-4)
        assert iters == ipf_rmse_example_results.completed_iters
        np.testing.assert_almost_equal(conv, ipf_rmse_example_results.final_convergence)

    def test_ipfn_convergence(self, ipf_ipfn_example_results: IpfDataAndResults):
        """Test that correct result calculated with ipfn convergence"""
        # Run
        mat, iters, conv = iterative_proportional_fitting.ipf(
            **ipf_ipfn_example_results.inputs.to_kwargs()
        )

        # Check the results
        np_testing.assert_allclose(mat, ipf_ipfn_example_results.final_matrix, rtol=1e-4)
        assert iters == ipf_ipfn_example_results.completed_iters
        np.testing.assert_almost_equal(conv, ipf_ipfn_example_results.final_convergence)

    def test_invalid_combos_run(self, ipf_invalid_combos_results: IpfDataAndResults):
        """Test that correct result calculated with some invalid combos"""
        # Run
        mat, iters, conv = iterative_proportional_fitting.ipf(
            **ipf_invalid_combos_results.inputs.to_kwargs()
        )

        # Check the results
        np_testing.assert_allclose(mat, ipf_invalid_combos_results.final_matrix, rtol=1e-4)
        assert iters == ipf_invalid_combos_results.completed_iters
        np.testing.assert_almost_equal(conv, ipf_invalid_combos_results.final_convergence)

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

        # Check for warning
        with pytest.warns(UserWarning, match="Given target_marginals of 0"):
            mat, iters, conv = iterative_proportional_fitting.ipf(
                **(ipf_example.to_kwargs() | {"target_marginals": bad_marginals})
            )

        # Check the results
        np_testing.assert_allclose(mat, target_mat, rtol=1e-4)
        assert iters == target_iters
        np.testing.assert_almost_equal(conv, target_conv)

    def test_non_early_exit(self, ipf_example: IpfData):
        """Test warning and return when marginals are 0"""
        # Set targets
        target_mat = ipf_example.matrix
        target_iters = 0
        target_conv = np.inf

        # Check for warning
        with pytest.warns(UserWarning, match="exhausted its max number of loops"):
            mat, iters, conv = iterative_proportional_fitting.ipf(
                **(ipf_example.to_kwargs() | {"max_iterations": 0})
            )

        # Check the results
        np_testing.assert_allclose(mat, target_mat, rtol=1e-4)
        assert iters == target_iters
        np.testing.assert_almost_equal(conv, target_conv)


@pytest.mark.usefixtures(
    "pandas_ipf_example",
    "ipf_pandas_rmse_example_results",
    "ipf_pandas_ipfn_example_results",
    "ipf_pandas_invalid_combos_results",
)
class TestIpfDataFrame:
    """Tests for caf.toolkit.iterative_proportional_fitting.ipf_dataframe"""

    def test_seed_df_value_type(self, pandas_ipf_example: IpfDataPandas):
        """Test that invalid seed_df value type raises an error"""
        val_col = pandas_ipf_example.value_col
        seed_df = pandas_ipf_example.matrix.copy()
        seed_df[val_col] = seed_df[val_col].astype(str)
        with pytest.raises(TypeError, match="expected to be numeric type"):
            iterative_proportional_fitting.ipf_dataframe(
                **(pandas_ipf_example.to_kwargs() | {"seed_df": seed_df})
            )

    def test_marginal_value_type(self, pandas_ipf_example: IpfDataPandas):
        """Test that invalid marginal value type raises an error"""
        marginals = pandas_ipf_example.marginals.copy()
        marginals[0] = marginals[0].astype(str)
        with pytest.raises(TypeError, match="expected to be numeric types"):
            iterative_proportional_fitting.ipf_dataframe(
                **(pandas_ipf_example.to_kwargs() | {"target_marginals": marginals})
            )

    def test_invalid_seed_df(self, pandas_ipf_example: IpfDataPandas):
        """Test that invalid seed_df type raises an error"""
        with pytest.raises(TypeError, match="is not a pandas.DataFrame"):
            iterative_proportional_fitting.ipf_dataframe(
                **(pandas_ipf_example.to_kwargs() | {"seed_df": 1})
            )

    def test_numpy_array_error(self, pandas_ipf_example: IpfDataPandas):
        """Test an error is raised when a numpy matrix is passed"""
        numpy_seed = pandas_ipf_example.matrix.values
        with pytest.raises(TypeError, match=re.escape("call `ipf()` instead")):
            iterative_proportional_fitting.ipf_dataframe(
                **(pandas_ipf_example.to_kwargs() | {"seed_df": numpy_seed})
            )

    def test_value_col_error(self, pandas_ipf_example: IpfDataPandas):
        """Test an error is raised when the wrong value_col is passed"""
        with pytest.raises(ValueError, match=re.escape("`value_col` is not in")):
            iterative_proportional_fitting.ipf_dataframe(
                **(pandas_ipf_example.to_kwargs() | {"value_col": "broken"})
            )

    def test_dataframe_marginals_error(self, pandas_ipf_example: IpfDataPandas):
        """Test an error is raised when non-series marginals is passed"""
        new_marginals = pandas_ipf_example.marginals
        new_marginals[0] = new_marginals[0].reset_index()
        with pytest.raises(TypeError, match="list of pandas.Series"):
            iterative_proportional_fitting.ipf_dataframe(
                **(pandas_ipf_example.to_kwargs() | {"target_marginals": new_marginals})
            )

    def test_mismatch_columns_error(self, pandas_ipf_example: IpfDataPandas):
        """Test an error is raised when seed and marginal columns don't match"""
        missing_col = pandas_ipf_example.matrix.columns[0]
        new_seed = pandas_ipf_example.matrix.rename(columns={missing_col: "Broken"})
        with pytest.raises(ValueError, match=re.escape(missing_col)):
            iterative_proportional_fitting.ipf_dataframe(
                **(pandas_ipf_example.to_kwargs() | {"seed_df": new_seed})
            )

    def test_missing_marginals(self, pandas_ipf_example: IpfDataPandas):
        """Test an error is raised when some values are missing from a marginal"""
        new_marginals = pandas_ipf_example.marginals.copy()
        new_marginals[0] = new_marginals[0].iloc[1:].copy()

        with pytest.raises(ValueError, match="missing values"):
            iterative_proportional_fitting.ipf_dataframe(
                **(pandas_ipf_example.to_kwargs() | {"target_marginals": new_marginals})
            )

    def test_rmse_convergence(self, ipf_pandas_rmse_example_results: IpfDataAndResultsPandas):
        """Test that correct result calculated with ipfn convergence"""
        # Run
        mat, iters, conv = iterative_proportional_fitting.ipf_dataframe(
            **ipf_pandas_rmse_example_results.inputs.to_kwargs()
        )

        # Check the results
        pd.testing.assert_frame_equal(
            mat, ipf_pandas_rmse_example_results.final_matrix, rtol=1e-4
        )
        assert iters == ipf_pandas_rmse_example_results.completed_iters
        np.testing.assert_almost_equal(conv, ipf_pandas_rmse_example_results.final_convergence)

    def test_ipfn_convergence(self, ipf_pandas_ipfn_example_results: IpfDataAndResultsPandas):
        """Test that correct result calculated with ipfn convergence"""
        # Run
        mat, iters, conv = iterative_proportional_fitting.ipf_dataframe(
            **ipf_pandas_ipfn_example_results.inputs.to_kwargs()
        )

        # Check the results
        pd.testing.assert_frame_equal(
            mat, ipf_pandas_ipfn_example_results.final_matrix, rtol=1e-4
        )
        assert iters == ipf_pandas_ipfn_example_results.completed_iters
        np.testing.assert_almost_equal(conv, ipf_pandas_ipfn_example_results.final_convergence)

    def test_invalid_combos_run(
        self, ipf_pandas_invalid_combos_results: IpfDataAndResultsPandas
    ):
        """Test that correct result calculated with some invalid combos"""
        # Run
        mat, iters, conv = iterative_proportional_fitting.ipf_dataframe(
            **ipf_pandas_invalid_combos_results.inputs.to_kwargs()
        )

        # Check the results
        pd.testing.assert_frame_equal(
            mat, ipf_pandas_invalid_combos_results.final_matrix, rtol=1e-4
        )
        assert iters == ipf_pandas_invalid_combos_results.completed_iters
        np.testing.assert_almost_equal(
            conv, ipf_pandas_invalid_combos_results.final_convergence
        )


# TODO(BT): Test that ipf pandas conversions make numpy matrices


@pytest.mark.usefixtures(
    "ipf_example",
    "ipf_pandas_rmse_example_results",
    "ipf_pandas_ipfn_example_results",
)
class TestIpfSparse:
    """Tests for sparse caf.toolkit.iterative_proportional_fitting.ipf_dataframe"""

    def test_rmse_convergence(self, ipf_pandas_rmse_example_results: IpfDataAndResultsPandas):
        """Test that correct result calculated with ipfn convergence"""
        # Run
        input_kwargs = ipf_pandas_rmse_example_results.inputs.to_kwargs()
        mat, iters, conv = iterative_proportional_fitting.ipf_dataframe(
            **(input_kwargs | {"force_sparse": True})
        )

        # Check the results
        pd.testing.assert_frame_equal(
            mat, ipf_pandas_rmse_example_results.final_matrix, rtol=1e-4
        )
        assert iters == ipf_pandas_rmse_example_results.completed_iters
        np.testing.assert_almost_equal(conv, ipf_pandas_rmse_example_results.final_convergence)

    def test_ipfn_convergence(self, ipf_pandas_ipfn_example_results: IpfDataAndResultsPandas):
        """Test that correct result calculated with ipfn convergence"""
        # Run
        input_kwargs = ipf_pandas_ipfn_example_results.inputs.to_kwargs()
        mat, iters, conv = iterative_proportional_fitting.ipf_dataframe(
            **(input_kwargs | {"force_sparse": True})
        )

        # Check the results
        pd.testing.assert_frame_equal(
            mat, ipf_pandas_ipfn_example_results.final_matrix, rtol=1e-4
        )
        assert iters == ipf_pandas_ipfn_example_results.completed_iters
        np.testing.assert_almost_equal(conv, ipf_pandas_ipfn_example_results.final_convergence)

    def test_sparse_marginals_only(self, ipf_example: IpfData):
        """Test that dense seed and sparse marginals raises an error"""
        marginals = ipf_example.marginals.copy()
        marginals[0] = sparse.COO(marginals[0])
        with pytest.raises(TypeError, match="Marginals are expected to be np.ndarray"):
            iterative_proportional_fitting.ipf(
                **(ipf_example.to_kwargs() | {"target_marginals": marginals})
            )
