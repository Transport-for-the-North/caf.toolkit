# -*- coding: utf-8 -*-
"""Tests for the caf.toolkit.pandas_utils.numpy_conversions module"""
# Built-Ins
import dataclasses

from typing import Any


# Third Party
import pytest
import numpy as np
import pandas as pd

# Local Imports
# pylint: disable=import-error,wrong-import-position
from caf.toolkit import pandas_utils as pd_utils

# pylint: enable=import-error,wrong-import-position

# # # CONSTANTS # # #

# # # CLASSES # # #
@dataclasses.dataclass
class ConversionData:
    np_matrix: np.ndarray
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

    # Create the base dataframe
    df = pd.DataFrame(
        data=np.random.randint(0, 10, data_length),
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
def fixture_conversion_data_1d(all_conversion_data: pd.MultiIndex):
    """Generate 1d pandas and numpy input and output conversions"""
    return all_conversion_data[1]


@pytest.fixture(name="conversion_data_2d", scope="function")
def fixture_conversion_data_2d(all_conversion_data: pd.MultiIndex):
    """Generate 2d pandas and numpy input and output conversions"""
    return all_conversion_data[2]


@pytest.fixture(name="conversion_data_3d", scope="function")
def fixture_conversion_data_3d(all_conversion_data: pd.MultiIndex):
    """Generate 3d pandas and numpy input and output conversions"""
    return all_conversion_data[3]


# # # TESTS # # #
@pytest.mark.usefixtures("conversion_data_1d", "conversion_data_2d", "conversion_data_3d")
class TestDataframeToNDimensionalArray:
    """Tests for caf.toolkit.pandas_utils.numpy_conversions.dataframe_to_n_dimensional_array"""

    # TODO(BT): Add Error tests

    @pytest.mark.parametrize(
        "conversion_data_str",
        ["conversion_data_1d", "conversion_data_2d", "conversion_data_3d"],
    )
    def test_correct_conversion(self, conversion_data_str: ConversionData, request):
        """Test 1-3 dimension correct conversions"""
        conversion_data = request.getfixturevalue(conversion_data_str)
        got_return = pd_utils.dataframe_to_n_dimensional_array(
            df=conversion_data.pd_matrix,
            dimension_cols=conversion_data.dimension_cols,
        )
        np.testing.assert_array_equal(got_return, conversion_data.np_matrix)


@pytest.mark.usefixtures("conversion_data_1d", "conversion_data_2d", "conversion_data_3d")
class TestNDimensionalArrayToDataframe:
    """Tests for caf.toolkit.pandas_utils.numpy_conversions.n_dimensional_array_to_dataframe"""

    # TODO(BT): Add Error tests

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
