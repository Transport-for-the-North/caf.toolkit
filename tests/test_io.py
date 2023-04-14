# -*- coding: utf-8 -*-
"""Tests for the {} module"""
# Built-Ins
import pathlib

# Third Party
import pandas as pd


# Local Imports
# pylint: disable=import-error,wrong-import-position
from caf.toolkit import io

# pylint: enable=import-error,wrong-import-position

# # # CONSTANTS # # #


# # # FIXTURES # # #


# # # TESTS # # #
def test_safe_dataframe_to_csv(tmp_path: pathlib.Path):
    """Test that this function correctly passes arguments to df.to_csv()"""
    df = pd.DataFrame(
        {
            "name": ["Raphael", "Donatello"],
            "mask": ["red", "purple"],
            "weapon": ["sai", "bo staff"],
        }
    )
    path = tmp_path / "test.csv"
    io.safe_dataframe_to_csv(df, path, index=False)
    pd.testing.assert_frame_equal(pd.read_csv(path), df)
