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
from caf.toolkit import toolbox
from caf.toolkit import pandas_utils as pd_utils

# pylint: enable=import-error,wrong-import-position

# # # CONSTANTS # # #


# # # FIXTURES # # #


# # # TESTS # # #
class TestReindexCols:
    """Tests for caf.toolkit.pandas_utils.reindex_cols"""

    @pytest.fixture(name="basic_long_df", scope="class")
    def fixture_basic_long_df(self):
        """Test long format dataframe"""
        columns = ["col1", "col2", "col3", "col4", "col5"]
        data = np.arange(50).reshape((-1, 5))
        return pd.DataFrame(data=data, columns=columns)

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

    @pytest.fixture(name="basic_wide_df", scope="class")
    def fixture_basic_wide_df(self):
        """Test wide format dataframe"""
        col_idx = [1, 2, 3, 4, 5]
        data = np.arange(25).reshape((5, 5))
        return pd.DataFrame(data=data, columns=col_idx, index=col_idx)

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
        """Test that casting is working correctly"""
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


class TestReindexAndGroupbySum:
    """
    Tests for caf.toolkit.pandas_utils.reindex_and_groupby

    Builds on caf.toolkit.pandas_utils.reindex_cols(), so tests here are
    quite simple
    """

    @pytest.fixture(name="group_df", scope="class")
    def fixture_group_df(self):
        """Test Dataframe"""
        return pd.DataFrame(
            data=[[1, 2, 3], [1, None, 4], [2, 1, 3], [1, 2, 2]],
            columns=["a", "b", "c"],
        )

    @pytest.mark.parametrize(
        "index_cols,value_cols", [(["a", "b", "c"], ["c"]), (["a", "b"], ["a"])]
    )
    def test_groupby(
        self,
        group_df: pd.DataFrame,
        index_cols: list[str],
        value_cols: list[str],
    ):
        """Test that it works exactly like pandas operations"""
        # Function call
        new_df = pd_utils.reindex_and_groupby_sum(
            df=group_df,
            index_cols=index_cols,
            value_cols=value_cols,
        )

        # Generate the expected DF
        df = group_df.reindex(columns=index_cols)
        group_cols = toolbox.list_safe_remove(index_cols, value_cols)
        expected_df = df.groupby(group_cols).sum().reset_index()

        pd.testing.assert_frame_equal(new_df, expected_df)

    def test_error(self, group_df: pd.DataFrame):
        """Test that an error is thrown correctly"""
        with pytest.raises(ValueError):
            pd_utils.reindex_and_groupby_sum(
                df=group_df,
                index_cols=["a", "b"],
                value_cols=["c"],
            )


class TestFilterDf:
    """Tests for caf.toolkit.pandas_utils.filter_df()"""

    @pytest.fixture(name="filter_df", scope="class")
    def fixture_filter_df(self):
        """Test Dataframe"""
        return pd.DataFrame(
            data=[[1, 2, 3], [1, None, 4], [2, 1, 3], [1, 2, 2]],
            columns=["a", "b", "c"],
            dtype=float,
        )

    @pytest.mark.parametrize("filter_dict", [{"a": [1], "b": [2]}, {"a": 1, "b": 2}])
    def test_mask(self, filter_df: pd.DataFrame, filter_dict: dict[str, Any]):
        """Test the mask is generated correctly"""
        expected_mask = pd.Series(data=[True, False, False, True])
        new_mask = pd_utils.filter_df_mask(df=filter_df, df_filter=filter_dict)
        pd.testing.assert_series_equal(new_mask, expected_mask)

    @pytest.mark.parametrize("filter_dict", [{"a": [1], "b": [2]}, {"a": 1, "b": 2}])
    def test_filter_df(self, filter_df: pd.DataFrame, filter_dict: dict[str, Any]):
        """Test the mask is generated correctly"""
        expected_df = pd.DataFrame(
            data=[[1, 2, 3], [1, 2, 2]],
            columns=["a", "b", "c"],
            index=[0, 3],
            dtype=float,
        )

        new_df = pd_utils.filter_df(df=filter_df, df_filter=filter_dict)
        pd.testing.assert_frame_equal(new_df, expected_df)
