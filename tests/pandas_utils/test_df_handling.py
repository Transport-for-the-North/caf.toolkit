# -*- coding: utf-8 -*-
"""
Created on: 18/08/2022
Updated on:

Original author: Ben Taylor
Last update made by:
Other updates made by:

File purpose:

"""
# Built-Ins

from typing import Any

# Third Party
import pytest
import pandas as pd
import numpy as np

# Local Imports
# pylint: disable=import-error,wrong-import-position
from caf.toolkit import pandas_utils as pd_utils

# pylint: enable=import-error,wrong-import-position

# # # CONSTANTS # # #


# # # FIXTURES # # #
@pytest.fixture(name="basic_long_df")
def fixture_basic_long_df():
    """Test long format dataframe"""
    columns = ["col1", "col2", "col3", "col4", "col5"]
    data = np.arange(50).reshape((-1, 5))
    return pd.DataFrame(data=data, columns=columns)


@pytest.fixture(name="basic_wide_df")
def fixture_basic_wide_df():
    """Test wide format dataframe"""
    col_idx = [1, 2, 3, 4, 5]
    data = np.arange(25).reshape((5, 5))
    return pd.DataFrame(data=data, columns=col_idx, index=col_idx)


# # # TESTS # # #
class TestReindexCols:
    """Tests for caf.toolkit.pandas_utils.reindex_cols"""

    @pytest.mark.parametrize(
        "columns", [["col1"], ["col1", "A"], [], ["col3", "A", "B", "col2"]]
    )
    @pytest.mark.parametrize("throw_error", [True, False])
    def test_reindex_and_error(
        self,
        basic_long_df: pd.DataFrame,
        columns: list[str],
        throw_error: bool,
    ):
        """Tests that it works exactly like `Dataframe.reindex()`"""
        # Check if an error should be thrown
        diff = set(columns) - set(basic_long_df.columns)
        all_cols_exist = len(diff) == 0

        # Set up kwargs
        kwargs = {
            "df": basic_long_df,
            "columns": columns,
            "dataframe_name": "test_df",
        }

        # Check and error is raise if it should be
        if throw_error and not all_cols_exist:
            with pytest.raises(ValueError):
                pd_utils.reindex_cols(**kwargs, throw_error=throw_error)

        else:
            # Should work exactly like pandas reindex of no error
            new_df = pd_utils.reindex_cols(**kwargs, throw_error=throw_error)
            base_df = basic_long_df.reindex(columns=columns)
            pd.testing.assert_frame_equal(new_df, base_df)

    @pytest.mark.parametrize("dataframe_name", ["df", "df_name", "some name"])
    def test_df_name(self, basic_long_df: pd.DataFrame, dataframe_name: str):
        """Tests the error is created correctly"""
        with pytest.raises(ValueError) as excinfo:
            pd_utils.reindex_cols(
                df=basic_long_df,
                columns=["A"],
                throw_error=True,
                dataframe_name=dataframe_name,
            )
        assert dataframe_name in str(excinfo.value)


class TestReindexRowsAndCols:
    """Tests for caf.toolkit.pandas_utils.reindex_rows_and_cols"""

    @pytest.mark.parametrize("index", [[1, 2], [], [5, 6]])
    @pytest.mark.parametrize("columns", [[1, 2], [], [5, 6]])
    @pytest.mark.parametrize("fill_value", [np.nan, 0])
    def test_reindex_with_matching_dtypes(
        self,
        basic_wide_df: pd.DataFrame,
        index: list[Any],
        columns: list[Any],
        fill_value: Any,
    ):
        """Tests that it works exactly like `Dataframe.reindex()`

        Tests only cover cases where dtypes match as functionality diverges
        where different datatypes are concerned
        """
        # Ensure the same types are used
        if len(index) > 0:
            basic_wide_df.index = basic_wide_df.index.astype(type(index[0]))
        if len(columns) > 0:
            basic_wide_df.columns = basic_wide_df.columns.astype(type(columns[0]))

        # Set up kwargs
        kwargs = {
            "df": basic_wide_df,
            "index": index,
            "columns": columns,
            "fill_value": fill_value,
        }

        # Should work exactly like pandas reindex
        new_df = pd_utils.reindex_rows_and_cols(**kwargs)
        base_df = basic_wide_df.reindex(
            columns=columns,
            index=index,
            fill_value=fill_value,
        )
        pd.testing.assert_frame_equal(new_df, base_df)

    def test_reindex_with_cast(self, basic_wide_df: pd.DataFrame):
        # Generate the expected output
        col_idx = ["1", "2"]
        data = np.array([[0, 1], [5, 6]], dtype=basic_wide_df.values.dtype)
        expected_df = pd.DataFrame(data=data, columns=col_idx, index=col_idx)

        # Run and compare
        new_df = pd_utils.reindex_rows_and_cols(
            df=basic_wide_df,
            index=col_idx,
            columns=col_idx,
        )

        pd.testing.assert_frame_equal(new_df, expected_df)
