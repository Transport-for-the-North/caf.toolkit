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

# Third Party
import numpy
import pytest
import pandas as pd
import numpy as np

# Local Imports
# pylint: disable=import-error,wrong-import-position
from caf.toolkit import pandas_utils as pd_utils

# pylint: enable=import-error,wrong-import-position

# # # CONSTANTS # # #


# # # FIXTURES # # #
@pytest.fixture(name="basic_df")
def fixture_basic_df():
    columns = ["col1", "col2", "col3", "col4", "col5"]
    data = np.arange(50).reshape((-1, 5))
    return pd.DataFrame(data=data, columns=columns)


# # # TESTS # # #
class TestReindexCols:
    @pytest.mark.parametrize(
        "columns", [["col1"], ["col1", "A"], [], ["col3", "A", "B", "col2"]]
    )
    @pytest.mark.parametrize("throw_error", [True, False])
    def test_reindex_and_error(
        self, basic_df: pd.DataFrame, columns: list[str], throw_error: bool
    ):
        """Tests that it works exactly like `Dataframe.reindex()`"""
        # Check if an error should be thrown
        diff = set(columns) - set(basic_df.columns)
        all_cols_exist = len(diff) == 0

        # Set up kwargs
        kwargs = {
            "df": basic_df,
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
            base_df = basic_df.reindex(columns=columns)
            pd.testing.assert_frame_equal(new_df, base_df)

    @pytest.mark.parametrize("dataframe_name", ["df", "df_name", "some name"])
    def test_df_name(self, basic_df: pd.DataFrame, dataframe_name: str):
        """Tests the error is created correctly"""
        with pytest.raises(ValueError) as excinfo:
            pd_utils.reindex_cols(
                df=basic_df, columns=["A"], throw_error=True, dataframe_name=dataframe_name
            )
        assert dataframe_name in str(excinfo.value)
