# -*- coding: utf-8 -*-
"""Tests for the caf.toolkit.pandas_utils.numpy_conversions module"""
# Built-Ins
import math
import psutil
import dataclasses

from typing import Any


# Third Party
import pytest
import sparse
import numpy as np
import pandas as pd

# Local Imports
# pylint: disable=import-error,wrong-import-position
from caf.toolkit import pandas_utils as pd_utils
from caf.toolkit.core import SparseLiteral

# pylint: enable=import-error,wrong-import-position

# # # CONSTANTS # # #


# # # CLASSES # # #
@dataclasses.dataclass
class ConversionData:
    """Store I/O for conversion tests"""

    np_matrix: np.ndarray
    pd_matrix: pd.DataFrame
    dimension_cols: dict[str, list[Any]]
    pd_value_col: str


@dataclasses.dataclass
class SparseConversionData:
    """Store I/O for conversion tests"""

    sparse_matrix: sparse.COO
    pd_matrix: pd.DataFrame
    dimension_cols: dict[str, list[Any]]
    pd_value_col: str


# # # FIXTURES # # #
@pytest.fixture(name="example_index", scope="function")
def fixture_example_index() -> pd.MultiIndex:
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


@pytest.fixture(name="all_conversion_data", scope="function")
def fixture_all_conversion_data(example_index: pd.MultiIndex):
    """Generate multiple pandas and numpy input and output conversions"""
    # Init
    value_col = "value"
    data_length = len(example_index)

    # Create the base dataframe - ensure 0 value
    df = pd.DataFrame(
        data=np.random.randint(0, 10, data_length - 1).tolist() + [0],
        index=example_index,
        columns=[value_col],
    ).reset_index()

    # Get the base dimension cols
    dimension_cols = df.columns.tolist()
    dimension_cols.remove(value_col)
    dimension_order = {x: df[x].unique().tolist() for x in dimension_cols}

    # Generate 1, 2, and 3 dimensional returns
    conversion_data = dict()
    for i in range(1, len(dimension_cols) + 1):
        keep_dims = dimension_cols[:i]
        drop_dims = dimension_cols[i:]
        local_dim_order = {x: dimension_order[x] for x in keep_dims}
        final_shape = [len(x) for x in local_dim_order.values()]

        local_df = df.drop(columns=drop_dims)
        local_df = local_df.groupby(keep_dims).sum().reset_index()

        conversion_data[i] = ConversionData(
            pd_matrix=local_df,
            np_matrix=local_df[value_col].values.reshape(final_shape),
            dimension_cols=local_dim_order,
            pd_value_col=value_col,
        )

    return conversion_data


@pytest.fixture(name="conversion_data_1d", scope="function")
def fixture_conversion_data_1d(all_conversion_data: pd.MultiIndex) -> ConversionData:
    """Generate 1d pandas and numpy input and output conversions"""
    return all_conversion_data[1]


@pytest.fixture(name="conversion_data_2d", scope="function")
def fixture_conversion_data_2d(all_conversion_data: pd.MultiIndex) -> ConversionData:
    """Generate 2d pandas and numpy input and output conversions"""
    return all_conversion_data[2]


@pytest.fixture(name="conversion_data_3d", scope="function")
def fixture_conversion_data_3d(all_conversion_data: pd.MultiIndex) -> ConversionData:
    """Generate 3d pandas and numpy input and output conversions"""
    return all_conversion_data[3]


@pytest.fixture(name="conversion_data_massive", scope="class")
def fixture_conversion_data_massive() -> SparseConversionData:
    """Generate a massive dataframe that can only be held in memory as a sparse matrix"""
    # init
    dim_cols = ["col1", "col2", "col3", "col4"]
    value_col = "value"

    # This should create an array that's approx double what can be handled
    max_ram = psutil.virtual_memory().total
    axis_0_1_2_size = 1024
    axis_3_size = math.ceil(max_ram / 8 / (1024 ** 3)) * 2

    # Scale axis sizes so all values appear, and they divide nicely
    axis_0_1_2_size += axis_3_size - (axis_0_1_2_size % axis_3_size)
    axis_2_repeats = axis_0_1_2_size / axis_3_size

    # Create the df
    df = pd.DataFrame(
        {
            "col1": range(axis_0_1_2_size),
            "col2": range(axis_0_1_2_size),
            "col3": range(axis_0_1_2_size),
            "col4": np.repeat(range(axis_3_size), axis_2_repeats),
            "value": np.random.random(axis_0_1_2_size),
        }
    )

    # Create the sparse conversion
    sparse_mat = sparse.COO(
        coords=np.array([df[col].values for col in dim_cols]),
        data=np.array(df[value_col].values),
        shape=(axis_0_1_2_size, axis_0_1_2_size, axis_0_1_2_size, axis_3_size),
    )

    return SparseConversionData(
        sparse_matrix=sparse_mat,
        pd_matrix=df,
        dimension_cols={x: df[x].values for x in dim_cols},
        pd_value_col=value_col,
    )


# # # TESTS # # #
@pytest.mark.usefixtures(
    "conversion_data_1d", "conversion_data_2d", "conversion_data_3d", "conversion_data_massive"
)
class TestDataframeToNDimensionalArray:
    """Tests for caf.toolkit.pandas_utils.numpy_conversions.dataframe_to_n_dimensional_array"""

    @pytest.mark.parametrize(
        "conversion_data_str",
        ["conversion_data_1d", "conversion_data_2d", "conversion_data_3d"],
    )
    def test_too_many_value_cols(self, conversion_data_str: ConversionData, request):
        """Test error raising when extra value cols given"""
        conversion_data = request.getfixturevalue(conversion_data_str)
        bad_df = conversion_data.pd_matrix.copy()
        bad_df["bad_value"] = 0
        with pytest.raises(ValueError, match="More than one value column found"):
            pd_utils.dataframe_to_n_dimensional_array(
                df=bad_df,
                dimension_cols=conversion_data.dimension_cols.keys(),
            )

    @pytest.mark.parametrize(
        "conversion_data_str",
        ["conversion_data_1d", "conversion_data_2d", "conversion_data_3d"],
    )
    def test_correct_conversion(self, conversion_data_str: ConversionData, request):
        """Test 1-3 dimension correct conversions"""
        conversion_data = request.getfixturevalue(conversion_data_str)
        got_return, _ = pd_utils.dataframe_to_n_dimensional_array(
            df=conversion_data.pd_matrix,
            dimension_cols=conversion_data.dimension_cols,
        )
        np.testing.assert_array_equal(got_return, conversion_data.np_matrix)

    @pytest.mark.parametrize(
        "conversion_data_str",
        ["conversion_data_1d", "conversion_data_2d", "conversion_data_3d"],
    )
    def test_list_cols(self, conversion_data_str: ConversionData, request):
        """Test dimension_cols given as a list"""
        conversion_data = request.getfixturevalue(conversion_data_str)
        got_return, _ = pd_utils.dataframe_to_n_dimensional_array(
            df=conversion_data.pd_matrix,
            dimension_cols=conversion_data.dimension_cols.keys(),
        )
        np.testing.assert_array_equal(got_return, conversion_data.np_matrix)

    @pytest.mark.parametrize("sparse_ok", list(SparseLiteral.__args__) + ["nsdtg", 23])
    def test_sparse_ok(self, conversion_data_2d: ConversionData, sparse_ok: str):
        """Test valid and invalid values of sparse_ok"""
        if sparse_ok in SparseLiteral.__args__:
            got_return, _ = pd_utils.dataframe_to_n_dimensional_array(
                df=conversion_data_2d.pd_matrix,
                dimension_cols=conversion_data_2d.dimension_cols.keys(),
                sparse_ok=sparse_ok,
            )
            if isinstance(got_return, sparse.COO):
                got_return = got_return.todense()
            np.testing.assert_array_equal(got_return, conversion_data_2d.np_matrix)
        else:
            with pytest.raises(ValueError, match="Invalid value given for "):
                pd_utils.dataframe_to_n_dimensional_array(
                    df=conversion_data_2d.pd_matrix,
                    dimension_cols=conversion_data_2d.dimension_cols.keys(),
                    sparse_ok=sparse_ok,
                )

    def test_auto_sparse_conversion(self, conversion_data_massive: SparseConversionData):
        """Test auto conversion to a sparse matrix when too big"""
        got_return, _ = pd_utils.dataframe_to_n_dimensional_array(
            df=conversion_data_massive.pd_matrix,
            dimension_cols=conversion_data_massive.dimension_cols.keys(),
            sparse_ok="allow",
        )

        # Check important components match
        np.testing.assert_array_equal(
            got_return.data, conversion_data_massive.sparse_matrix.data
        )
        np.testing.assert_array_equal(
            got_return.coords, conversion_data_massive.sparse_matrix.coords
        )
        assert got_return.shape == conversion_data_massive.sparse_matrix.shape

    @pytest.mark.parametrize("sparse_ok", ["feasible", "force"])
    def test_force_feasible_sparse_conversion(
        self, conversion_data_massive: SparseConversionData, sparse_ok: str
    ):
        """Test auto conversion to a sparse matrix when too big"""
        got_return, _ = pd_utils.dataframe_to_n_dimensional_array(
            df=conversion_data_massive.pd_matrix,
            dimension_cols=conversion_data_massive.dimension_cols.keys(),
            sparse_ok=sparse_ok,
        )

        # Check important components match
        np.testing.assert_array_equal(
            got_return.data, conversion_data_massive.sparse_matrix.data
        )
        np.testing.assert_array_equal(
            got_return.coords, conversion_data_massive.sparse_matrix.coords
        )
        assert got_return.shape == conversion_data_massive.sparse_matrix.shape


@pytest.mark.usefixtures("conversion_data_1d", "conversion_data_2d", "conversion_data_3d")
class TestNDimensionalArrayToDataframe:
    """Tests for caf.toolkit.pandas_utils.numpy_conversions.n_dimensional_array_to_dataframe"""

    @pytest.mark.parametrize(
        "conversion_data_str",
        ["conversion_data_1d", "conversion_data_2d", "conversion_data_3d"],
    )
    def test_correct_conversion(self, conversion_data_str: ConversionData, request):
        """Test 1-3 dimension correct conversions"""
        conversion_data = request.getfixturevalue(conversion_data_str)
        got_return = pd_utils.n_dimensional_array_to_dataframe(
            mat=conversion_data.np_matrix,
            dimension_cols=conversion_data.dimension_cols,
            value_col=conversion_data.pd_value_col,
        )
        pd.testing.assert_frame_equal(got_return.reset_index(), conversion_data.pd_matrix)

    @pytest.mark.parametrize(
        "conversion_data_str",
        ["conversion_data_1d", "conversion_data_2d", "conversion_data_3d"],
    )
    def test_drop_zeros(self, conversion_data_str: ConversionData, request):
        """Test 1-3 dimension correct conversions"""
        conversion_data = request.getfixturevalue(conversion_data_str)
        got_return = pd_utils.n_dimensional_array_to_dataframe(
            mat=conversion_data.np_matrix,
            dimension_cols=conversion_data.dimension_cols,
            value_col=conversion_data.pd_value_col,
            drop_zeros=True,
        )

        # Drop any rows where the value is 0
        zero_mask = conversion_data.pd_matrix[conversion_data.pd_value_col] == 0
        non_zero_df = conversion_data.pd_matrix[~zero_mask].reset_index(drop=True)

        pd.testing.assert_frame_equal(got_return.reset_index(), non_zero_df)
