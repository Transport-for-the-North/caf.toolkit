"""Tests for the caf.toolkit.pandas_utils.df_handling module."""

# Built-Ins
import dataclasses
from typing import Any, NamedTuple

# Third Party
import numpy as np
import pandas as pd
import pytest

# Local Imports
# pylint: disable=import-error,wrong-import-position
from caf.toolkit import pandas_utils as pd_utils
from caf.toolkit import toolbox

# pylint: enable=import-error,wrong-import-position

# # # CONSTANTS # # #


# # # FIXTURES # # #


# # # TESTS # # #
class TestReindexCols:
    """Tests for caf.toolkit.pandas_utils.reindex_cols."""

    @pytest.fixture(name="basic_long_df", scope="class")
    def fixture_basic_long_df(self) -> pd.DataFrame:
        """Test long format dataframe."""
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
    ) -> None:
        """Tests that it works exactly like `Dataframe.reindex()`."""
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
    def test_df_name(self, basic_long_df: pd.DataFrame, dataframe_name: str) -> None:
        """Tests the error is created correctly."""
        with pytest.raises(ValueError) as excinfo:
            pd_utils.reindex_cols(
                df=basic_long_df,
                columns=["A"],
                throw_error=True,
                dataframe_name=dataframe_name,
            )
        assert dataframe_name in str(excinfo.value)


class TestReindexRowsAndCols:
    """Tests for caf.toolkit.pandas_utils.reindex_rows_and_cols."""

    @pytest.fixture(name="basic_wide_df", scope="class")
    def fixture_basic_wide_df(self) -> pd.DataFrame:
        """Test wide format dataframe."""
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
        fill_value: float | int,
    ) -> None:
        """Tests that it works exactly like `Dataframe.reindex()`.

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

    def test_reindex_with_cast(self, basic_wide_df: pd.DataFrame) -> None:
        """Test that casting is working correctly."""
        # Generate the expected output
        col_idx = ["1", "2"]
        data = np.array([[0, 1], [5, 6]], dtype=basic_wide_df.to_numpy().dtype)
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
    Tests for caf.toolkit.pandas_utils.reindex_and_groupby.

    Builds on caf.toolkit.pandas_utils.reindex_cols(), so tests here are
    quite simple
    """

    @pytest.fixture(name="group_df", scope="class")
    def fixture_group_df(self) -> pd.DataFrame:
        """Test Dataframe."""
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
    ) -> None:
        """Test that it works exactly like pandas operations."""
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

    def test_error(self, group_df: pd.DataFrame) -> None:
        """Test that an error is thrown correctly."""
        with pytest.raises(ValueError):
            pd_utils.reindex_and_groupby_sum(
                df=group_df,
                index_cols=["a", "b"],
                value_cols=["c"],
            )


class TestFilterDf:
    """Tests for caf.toolkit.pandas_utils.filter_df()."""

    @pytest.fixture(name="filter_df", scope="class")
    def fixture_filter_df(self) -> pd.DataFrame:
        """Test Dataframe."""
        return pd.DataFrame(
            data=[[1, 2, 3], [1, None, 4], [2, 1, 3], [1, 2, 2]],
            columns=["a", "b", "c"],
            dtype=float,
        )

    @pytest.mark.parametrize("filter_dict", [{"a": [1], "b": [2]}, {"a": 1, "b": 2}])
    def test_mask(self, filter_df: pd.DataFrame, filter_dict: dict[str, Any]) -> None:
        """Test the mask is generated correctly."""
        expected_mask = pd.Series(data=[True, False, False, True])
        new_mask = pd_utils.filter_df_mask(df=filter_df, df_filter=filter_dict)
        pd.testing.assert_series_equal(new_mask, expected_mask)

    @pytest.mark.parametrize("filter_dict", [{"a": [1], "b": [2]}, {"a": 1, "b": 2}])
    def test_filter_df(self, filter_df: pd.DataFrame, filter_dict: dict[str, Any]) -> None:
        """Test the mask is generated correctly."""
        expected_df = pd.DataFrame(
            data=[[1, 2, 3], [1, 2, 2]],
            columns=["a", "b", "c"],
            index=[0, 3],
            dtype=float,
        )

        new_df = pd_utils.filter_df(df=filter_df, df_filter=filter_dict)
        pd.testing.assert_frame_equal(new_df, expected_df)


class TestStrJoinCols:
    """Tests for caf.toolkit.pandas_utils.str_join_cols."""

    @pytest.mark.parametrize("col_vals", [["a", "b"], ["a", 1], [1.4, 1.0]])
    def test_str_joining(self, col_vals: list[Any]) -> None:
        """Test the whole process on different data types."""
        separator = "_"
        col_names = ["col1", "col2", "col3"]

        # Build the starting df
        starting_data = [[x] * 3 for x in col_vals]
        start_df = pd.DataFrame(
            data=starting_data,
            columns=col_names,
        )

        # Build the expected ending df
        exp_data = [[str(x)] * 3 for x in col_vals]
        exp_data = [separator.join(x) for x in exp_data]
        exp_series = pd.Series(data=exp_data)

        # Run and compare
        end_series = pd_utils.str_join_cols(
            df=start_df,
            columns=col_names,
            separator=separator,
        )
        pd.testing.assert_series_equal(exp_series, end_series)


class TestChunkDf:
    """Tests for caf.toolkit.pandas_utils.chunk_df."""

    @pytest.fixture(name="basic_long_df", scope="class")
    def fixture_basic_long_df(self) -> pd.DataFrame:
        """Test long format dataframe."""
        columns = ["col1", "col2"]
        data = np.arange(6).reshape((-1, 2))
        return pd.DataFrame(data=data, columns=columns)

    @pytest.mark.parametrize("chunk_size", [0, -1, 0.5, -0.5])
    def test_class_error(self, basic_long_df: pd.DataFrame, chunk_size: int) -> None:
        """Test that giving an incorrect chunk_size generates an error."""
        if not isinstance(chunk_size, int):
            with pytest.raises(TypeError):
                pd_utils.ChunkDf(df=basic_long_df, chunk_size=chunk_size)
        else:
            with pytest.raises(ValueError):
                pd_utils.ChunkDf(df=basic_long_df, chunk_size=chunk_size)

    @pytest.mark.parametrize("chunk_size", [0, -1, 0.5, -0.5])
    def test_function_error(self, basic_long_df: pd.DataFrame, chunk_size: int) -> None:
        """Test that giving an incorrect chunk_size works as python expects."""
        chunks = pd_utils.chunk_df(df=basic_long_df, chunk_size=chunk_size)
        assert list(chunks) == list()

    def test_chunk_size_one(self, basic_long_df: pd.DataFrame) -> None:
        """Test that chunk_size operates correctly."""
        # Create the expected output
        df = basic_long_df
        expected_output = [
            pd.DataFrame(data=[df.loc[0].to_numpy()], columns=df.columns),
            pd.DataFrame(data=[df.loc[1].to_numpy()], columns=df.columns),
            pd.DataFrame(data=[df.loc[2].to_numpy()], columns=df.columns),
        ]

        # Check outputs
        for expected, got in zip(expected_output, pd_utils.chunk_df(df, 1), strict=True):
            pd.testing.assert_frame_equal(expected, got.reset_index(drop=True))

    def test_chunk_size_two(self, basic_long_df: pd.DataFrame) -> None:
        """Test that chunk_size operates correctly."""
        # Create the expected output
        df = basic_long_df
        expected_output = [
            pd.DataFrame(data=df.loc[[0, 1]].to_numpy(), columns=df.columns),
            pd.DataFrame(data=[df.loc[2].to_numpy()], columns=df.columns),
        ]

        # Check outputs
        for expected, got in zip(expected_output, pd_utils.chunk_df(df, 2), strict=True):
            pd.testing.assert_frame_equal(expected, got.reset_index(drop=True))

    @pytest.mark.parametrize("chunk_size", [3, 4])
    def test_chunk_size_big(self, basic_long_df: pd.DataFrame, chunk_size: int) -> None:
        """Test that chunk_size operates correctly when bigger than DataFrame."""
        # Create the expected output
        df = basic_long_df
        expected_output = [df.copy()]

        # Check outputs
        for expected, got in zip(
            expected_output, pd_utils.chunk_df(df, chunk_size), strict=True
        ):
            pd.testing.assert_frame_equal(expected, got.reset_index(drop=True))

    # TODO(BT): Do we need a test for oddly shaped DataFrame edge cases?


class TestLongProductInfill:
    """Tests for caf.toolkit.pandas_utils.long_product_infill."""

    def test_single_idx_col(self) -> None:
        """Test function with one index column."""
        # Init
        infill_val = "z"
        col_names = ["idx1", "col1", "col2"]
        index_dict = {"idx1": [1, 2, 3]}

        # Build the input and expected dataframes
        data = [[1, "a", "b"], [2, "a", "b"]]
        expected_data = data.copy()
        expected_data.append([3, infill_val, infill_val])

        df = pd.DataFrame(data=data, columns=col_names).set_index("idx1")
        expected_df = pd.DataFrame(data=expected_data, columns=col_names).set_index("idx1")

        # Check
        df = pd_utils.long_product_infill(
            data=df,
            index_dict=index_dict,
            infill=infill_val,
            check_totals=False,
        )
        pd.testing.assert_frame_equal(df, expected_df)

    def test_double_idx_col(self) -> None:
        """Test function with two index columns.

        Also tests indexing things back into order
        """
        # Init
        infill_val = "z"
        col_names = ["idx1", "idx2", "col1"]
        index_dict = {"idx1": [1, 2], "idx2": [3, 4]}

        # Build the input and expected dataframes
        data = [[1, 3, "b"], [2, 4, "b"]]
        expected_data = [
            [1, 3, "b"],
            [1, 4, infill_val],
            [2, 3, infill_val],
            [2, 4, "b"],
        ]

        df = pd.DataFrame(data=data, columns=col_names).set_index(["idx1", "idx2"])
        expected_df = pd.DataFrame(data=expected_data, columns=col_names).set_index(
            ["idx1", "idx2"]
        )

        # Check
        df = pd_utils.long_product_infill(
            data=df,
            index_dict=index_dict,
            infill=infill_val,
            check_totals=False,
        )
        pd.testing.assert_frame_equal(df, expected_df)

    def test_triple_idx_col(self) -> None:
        """Test function with three index columns.

        Also tests:
         - string based indexes
         - index with only 1 value
         - Assuming index inputs
        """
        # Init
        infill_val = 100
        col_names = ["idx1", "idx2", "idx3", "col1"]
        index_dict = {"idx1": ["a", "b"], "idx2": ["c", "d"], "idx3": ["z"]}

        # Build the input and expected dataframes
        data = [["a", "c", "z", 1], ["b", "d", "z", 3]]
        expected_data = [
            ["a", "c", "z", 1],
            ["a", "d", "z", infill_val],
            ["b", "c", "z", infill_val],
            ["b", "d", "z", 3],
        ]

        df = (
            pd.DataFrame(data=data, columns=col_names)
            .set_index(["idx1", "idx2", "idx3"])
            .squeeze()
        )
        expected_df = (
            pd.DataFrame(data=expected_data, columns=col_names)
            .set_index(["idx1", "idx2", "idx3"])
            .squeeze()
        )

        df = pd_utils.long_product_infill(
            data=df,
            index_dict=index_dict,
            infill=infill_val,
            check_totals=False,
        )
        pd.testing.assert_series_equal(df, expected_df)

    def test_no_diff(self) -> None:
        """Test that no change is made when not needed."""
        col_names = ["idx1", "idx2", "col1"]
        index_dict = {"idx1": [1, 2], "idx2": [3, 4]}

        # Build the input and expected dataframes
        data = [
            [1, 3, "b"],
            [1, 4, "c"],
            [2, 3, "d"],
            [2, 4, "a"],
        ]
        df = pd.DataFrame(data=data, columns=col_names).set_index(["idx1", "idx2"])

        # Check
        generated_df = pd_utils.long_product_infill(
            data=df,
            index_dict=index_dict,
            check_totals=False,
        )
        pd.testing.assert_frame_equal(df, generated_df)

    def test_check_total(self) -> None:
        """Test that an error is thrown when values aren't numeric."""
        col_names = ["idx1", "idx2", "col1"]
        index_dict = {"idx1": [1, 2], "idx2": [3, 4]}

        # Build the input and expected dataframes
        data = [
            [1, 3, "a"],
            [1, 4, "b"],
            [2, 3, "c"],
            [2, 4, "d"],
        ]
        df = pd.DataFrame(data=data, columns=col_names).set_index(["idx1", "idx2"])

        # Check
        with pytest.raises(TypeError):
            pd_utils.long_product_infill(
                data=df,
                index_dict=index_dict,
                check_totals=True,
            )

    @pytest.mark.filterwarnings("ignore:Almost all values")
    @pytest.mark.parametrize("check_totals", [True, False])
    def test_drop(self, check_totals: bool) -> None:
        """Test error / warning if totals are being checked."""
        col_names = ["idx1", "idx2", "col1"]
        index_dict = {"idx1": [3, 4], "idx2": [3, 4]}

        # Build the input and expected dataframes
        data = [
            [1, 3, 10.3],
            [1, 4, 13.2],
            [2, 3, 76.4],
            [2, 4, 59.5],
        ]
        df = pd.DataFrame(data=data, columns=col_names).set_index(["idx1", "idx2"])

        # Check
        if check_totals:
            # Error should be throw if checking
            with pytest.raises(ValueError):
                pd_utils.long_product_infill(
                    data=df,
                    index_dict=index_dict,
                    check_totals=check_totals,
                )

    def test_warning(self) -> None:
        """Test that an error is raised when reindex is too different."""
        # Init
        infill_val = "z"
        col_names = ["idx1", "col1"]
        index_dict = {"idx1": list(range(20))}

        # Build the input dataframe
        data = [[100, "a"]]
        df = pd.DataFrame(data=data, columns=col_names).set_index("idx1")

        # Test for warning
        with pytest.raises(ValueError):
            pd_utils.long_product_infill(
                data=df,
                index_dict=index_dict,
                infill=infill_val,
                check_totals=False,
            )


class TestLongWideConversions:
    """Tests for long <-> wide conversions for dataframes.

    Covers:
    - caf.toolkit.pandas_utils.wide_to_long_infill
    - caf.toolkit.pandas_utils.long_to_wide_infill
    - caf.toolkit.pandas_utils.long_df_to_wide_ndarray
    """

    index_col1: str = "idx1"
    index_col2: str = "idx2"
    value_col: str = "val"

    class DfData(NamedTuple):
        """Collection of wide/long dataframe data."""

        long: pd.DataFrame
        wide: pd.DataFrame
        index_vals: list[Any]

    @pytest.fixture(name="df_data_complete", scope="class")
    def fixture_df_data_complete(self) -> DfData:
        """Long and Wide dataframes without any missing vals."""
        index_vals = [1, 2, 3]

        # Make the wide df
        data = np.arange(9).reshape((-1, 3))
        wide_df = pd.DataFrame(
            data=data,
            index=pd.Index(data=index_vals, name=self.index_col1),
            columns=pd.Index(data=index_vals, name=self.index_col2),
        )

        # Make the long df
        df = wide_df.reset_index()
        df = df.rename(columns={df.columns[0]: self.index_col1})

        # Convert to long
        long_df = df.melt(
            id_vars=self.index_col1,
            var_name=self.index_col2,
            value_name=self.value_col,
        )

        # Need to reindex to get in right order for tests
        index_cols = [self.index_col1, self.index_col2]
        new_index = pd.MultiIndex.from_product(
            [index_vals, index_vals],
            names=index_cols,
        )
        long_df = long_df.set_index(index_cols)
        long_df = long_df.reindex(index=new_index).reset_index()

        # Package values together
        return self.DfData(
            long=long_df,
            wide=wide_df,
            index_vals=index_vals,
        )

    def test_long_to_wide_full(self, df_data_complete: DfData) -> None:
        """Test long -> wide conversion with no add/del cols."""
        wide_df = pd_utils.long_to_wide_infill(
            matrix=df_data_complete.long.set_index(["idx1", "idx2"]).squeeze()
        )
        pd.testing.assert_frame_equal(wide_df, df_data_complete.wide)

    def test_long_to_wide_missing(self, df_data_complete: DfData) -> None:
        """Test that missing indexes are added back in correctly."""
        # Init
        infill_val = 0
        index = 2

        # Remove some random data
        long_df = df_data_complete.long.copy()
        mask = (long_df[self.index_col1] == index) | (long_df[self.index_col2] == index)
        long_df = long_df[~mask].copy().set_index(["idx1", "idx2"]).squeeze()

        # Create the expected output
        expected_wide = df_data_complete.wide.copy()
        expected_wide[index] = infill_val
        expected_wide.loc[index, :] = infill_val

        # Check
        wide_df = pd_utils.long_to_wide_infill(
            matrix=long_df, correct_cols=[1, 2, 3], correct_ind=[1, 2, 3], infill=0
        )
        pd.testing.assert_frame_equal(
            wide_df,
            expected_wide,
            check_dtype=False,
        )

    def test_wide_to_long_full(self, df_data_complete: DfData) -> None:
        """Test wide -> long conversion with no add/del cols."""
        long_df = pd_utils.wide_to_long_infill(df=df_data_complete.wide, out_name="val")
        pd.testing.assert_series_equal(
            long_df, df_data_complete.long.set_index(["idx1", "idx2"]).squeeze()
        )

    def test_wide_to_long_missing(self, df_data_complete: DfData) -> None:
        """Test that missing indexes are added back in correctly."""
        # Init
        infill_val = 0
        index = 2

        # Remove random data
        wide_df = df_data_complete.wide.copy()
        wide_df = wide_df.drop(columns=index, index=index)

        # Create expected output
        exp_long = df_data_complete.long.copy()
        mask = (exp_long[self.index_col1] == index) | (exp_long[self.index_col2] == index)
        exp_long.loc[mask, self.value_col] = infill_val

        # Check
        long_df = pd_utils.wide_to_long_infill(
            df=wide_df, correct_ind=[1, 2, 3], correct_cols=[1, 2, 3], infill=0
        )
        pd.testing.assert_series_equal(long_df, exp_long.set_index(["idx1", "idx2"]).squeeze())

    def test_long_df_to_wide_ndarray(self, df_data_complete: DfData) -> None:
        """Test function is a numpy wrapper around long_to_wide_infill."""
        # Init
        kwargs = {
            "matrix": df_data_complete.long.set_index(["idx1", "idx2"]),
        }

        # Check
        expected_array = pd_utils.long_to_wide_infill(**kwargs).to_numpy()
        wide_array = pd_utils.long_df_to_wide_ndarray(**kwargs)
        np.testing.assert_array_equal(expected_array, wide_array)


class TestGetFullIndex:
    """Tests for caf.toolkit.pandas_utils.get_full_index."""

    @dataclasses.dataclass
    class IndexData:
        """Store expected input and output of an index creation."""

        names_and_vals: dict[str, list[Any]]
        output_index: pd.Index

    @pytest.fixture(name="example_multi_index", scope="function")
    def fixture_example_multi_index(self) -> IndexData:
        """Generate the pandas MultiIndex for the pandas examples."""
        # Define example params
        names_and_vals = {
            "dma": [501, 502],
            "size": [1, 2, 3, 4],
            "age": ["20-25", "30-35", "40-45"],
        }

        # Create the example
        output_index = pd.MultiIndex.from_product(
            names_and_vals.values(),
            names=names_and_vals.keys(),
        )
        return self.IndexData(
            names_and_vals=names_and_vals,
            output_index=output_index,
        )

    @pytest.fixture(name="example_single_index", scope="function")
    def fixture_example_single_index(self) -> IndexData:
        """Generate the pandas MultiIndex for the pandas examples."""
        names_and_vals = {"age": ["20-25", "30-35", "40-45"]}
        return self.IndexData(
            names_and_vals=names_and_vals,
            output_index=pd.Index(data=names_and_vals["age"], name="age"),
        )

    def test_single_index(self, example_single_index: IndexData) -> None:
        """Test correct return when sending a single index."""
        gen_index = pd_utils.get_full_index(dimension_cols=example_single_index.names_and_vals)
        pd.testing.assert_index_equal(gen_index, example_single_index.output_index)

    def test_multi_index(self, example_multi_index: IndexData) -> None:
        """Test correct return when sending a single index."""
        gen_index = pd_utils.get_full_index(dimension_cols=example_multi_index.names_and_vals)
        pd.testing.assert_index_equal(gen_index, example_multi_index.output_index)
