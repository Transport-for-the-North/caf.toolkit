# -*- coding: utf-8 -*-
"""Tests for the io module"""
from __future__ import annotations

# Built-Ins
import dataclasses
import pathlib
import re
from typing import Iterator, Literal

# Third Party
import numpy as np
import pandas as pd
import pytest

# Local Imports
# pylint: disable=import-error,wrong-import-position
from caf.toolkit import io

# pylint: enable=import-error,wrong-import-position

# # # CONSTANTS # # #


# # # FIXTURES # # #


@dataclasses.dataclass
class DataFrameResults:
    """Stores `read_csv` test data."""

    data: pd.DataFrame
    columns: list[str]
    dtypes: dict[str, type]
    path: pathlib.Path
    incorrect_columns: list[str]
    incorrect_dtypes: dict[str, type]


@dataclasses.dataclass
class MatrixResults:
    """Stores `read_csv_matrix` test data."""

    matrix: pd.DataFrame
    path: pathlib.Path
    format: Literal["square", "long"]
    index_col: list[str] | str


class _FakeGlobPath:
    """Mock class replacing `pathlib.Path`."""

    def __init__(self, suffixes: list[str]):
        self._suffixes = suffixes

    def glob(self, name: str) -> Iterator[pathlib.Path]:
        """Mock `glob` method which just returns name with suffixes added.

        This does not create any files.
        """
        for suffix in self._suffixes:
            yield pathlib.Path(name.replace(".*", suffix))


@pytest.fixture(name="data", scope="module")
def fix_data(tmp_path_factory: pytest.TempPathFactory) -> DataFrameResults:
    """Create test DataFrame and save to CSV."""
    data = pd.DataFrame(
        {
            "string": [f"test_{i}" for i in range(10)],
            "integer": list(range(10)),
            "float": [i * 0.75 for i in range(10)],
        }
    )

    path = tmp_path_factory.mktemp("csv_data") / "test_data-hxoRI.csv"
    data.to_csv(path, index=False)
    assert path.is_file(), "error creating CSV file"

    return DataFrameResults(
        data=data,
        columns=["string", "integer", "float"],
        dtypes={"string": str, "integer": int, "float": float},
        path=path,
        incorrect_columns=["missing_ZO3yM", "missing_9GISV"],
        incorrect_dtypes={"string": int, "integer": str, "float": int},
    )


@pytest.fixture(name="matrix")
def fix_matrix() -> pd.DataFrame:
    """Create matrix DataFrame for tests."""
    index = ["zone1", "zone2", "zone3"]
    matrix = pd.DataFrame(np.arange(9).reshape((3, 3)), index=index, columns=index)

    return matrix


@pytest.fixture(name="square_matrix")
def fix_square_matrix(matrix: pd.DataFrame, tmp_path: pathlib.Path) -> MatrixResults:
    """Save matrix to CSV in square format."""
    matrix.index.name = "zone_id"
    path = tmp_path / "test_square_matrix.csv"
    matrix.to_csv(path)

    return MatrixResults(matrix, path, "square", "zone_id")


@pytest.fixture(name="long_matrix")
def fix_long_matrix(matrix: pd.DataFrame, tmp_path: pathlib.Path) -> MatrixResults:
    """Save matrix to CSV in long format."""
    long = matrix.stack()
    indices = ["origin", "destination"]
    long.index.names = indices
    matrix.index.name = indices[0]
    matrix.columns.name = indices[1]
    path = tmp_path / "test_long_matrix.csv"
    long.to_csv(path)

    return MatrixResults(matrix, path, "long", indices)


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


class TestReadCSV:
    """Tests for `read_csv` function."""

    @pytest.mark.parametrize("name", [None, "missing-dEg0N"])
    def test_missing(self, tmp_path_factory: pytest.TempPathFactory, name: str | None):
        """Test file not found error, with and without name."""
        path = tmp_path_factory.mktemp("empty")
        filename = "missing_file-bqiPI.csv"

        if name is None:
            name = filename
        error_pattern = re.compile(f"{name} file does not exist: '.*{filename}'", re.I)

        with pytest.raises(FileNotFoundError, match=error_pattern):
            io.read_csv(path / filename, name=name)

    def test_simple(self, data: DataFrameResults):
        """Test loading CSV with default parameters."""
        read = io.read_csv(data.path)

        pd.testing.assert_frame_equal(data.data, read, check_dtype=False)

    def test_full(self, data: DataFrameResults):
        """Test loading CSV with usecols, dtype and index_col parameters."""
        read = io.read_csv(
            data.path, usecols=data.columns, dtype=data.dtypes, index_col=data.columns[0]
        )

        correct = data.data.set_index(data.columns[0])
        # check_dtype set to False because this is very strict i.e. int32 != int64
        pd.testing.assert_frame_equal(read, correct, check_dtype=False)

    def test_missing_columns(self, data: DataFrameResults):
        """Test missing columns error is raised."""
        name = "missing_columns"
        pattern = re.compile(f"columns missing from {name}", re.I)
        with pytest.raises(io.MissingColumnsError, match=pattern) as excinfo:
            io.read_csv(data.path, name=name, usecols=data.incorrect_columns)

            assert set(excinfo.value.columns) == set(
                data.incorrect_columns
            ), "incorrect columns in exception"

    def test_incorrect_dtypes(self, data: DataFrameResults):
        """Test correct error is raised for incorrect dtypes."""
        # Find first column as error is raised for first column found
        column_name = None
        for col in data.columns:
            if col in data.incorrect_dtypes:
                column_name = col
                break
        assert column_name is not None, "incorrect column not found in test data???"

        name = "test_incorrect_dtypes-k77MC"
        pattern = re.compile(
            f"column '{column_name}' in {name} has values which cannot be "
            f"converted to {data.incorrect_dtypes[column_name]}",
            re.I,
        )
        with pytest.raises(ValueError, match=pattern):
            io.read_csv(data.path, name=name, dtype=data.incorrect_dtypes)


class TestReadCSVMatrix:
    """Tests for `read_csv_matrix` function."""

    @pytest.mark.parametrize("define_index", [False, True])
    @pytest.mark.parametrize("guess_format", [False, True])
    @pytest.mark.parametrize("data_name", ["long_matrix", "square_matrix"])
    def test_read_matrix(
        self, request, data_name: str, define_index: bool, guess_format: bool
    ):
        """Test reading matrix for square and long formats,
        with and without defined index columns."""
        data: MatrixResults = request.getfixturevalue(data_name)

        if define_index:
            index_col = data.index_col
        else:
            index_col = None

        if guess_format:
            format_ = None
        else:
            format_ = data.format

        read = io.read_csv_matrix(data.path, format_, index_col=index_col)
        pd.testing.assert_frame_equal(read, data.matrix, check_dtype=False)

    @pytest.mark.parametrize("columns", [True, False])
    def test_reindexing(self, tmp_path: pathlib.Path, columns: bool):
        """Test adjusting the index if the matrix has missing columns or indices."""
        zones = [f"zone{i}" for i in range(1, 7)]
        matrix = pd.DataFrame(np.arange(9).reshape((3, 3)), index=zones[:3], columns=zones[:3])
        new_zones = matrix.copy()
        if columns:
            new_zones.columns = zones[3:]
        else:
            new_zones.index = zones[3:]
        matrix = pd.concat([matrix, new_zones], axis=1 if columns else 0)

        path = tmp_path / "test_reindexing_matrix.csv"
        matrix.to_csv(path)

        pattern = re.compile(
            "matrix file (.*) doesn't contain the same "
            "index and columns, these are reindexed so all unique "
            "values from both are included",
            re.I,
        )
        with pytest.warns(RuntimeWarning, match=pattern):
            read = io.read_csv_matrix(path, "square")

        if columns:
            new_zones = pd.DataFrame([[np.nan] * 6] * 3, columns=zones, index=zones[3:])
        else:
            new_zones = pd.DataFrame([[np.nan] * 3] * 6, columns=zones[3:], index=zones)

        matrix = pd.concat([matrix, new_zones], axis=0 if columns else 1)
        pd.testing.assert_frame_equal(read, matrix, check_dtype=False)

    def test_unknown_format_guess(self, tmp_path: pathlib.Path):
        """Test that ValueError is raised when format can't be determined from file."""
        data = pd.DataFrame({"col1": [1, 2, 3], "col2": [1, 2, 3]})
        path = tmp_path / "test_matrix_incorrect.csv"
        data.to_csv(path, index=False)
        pattern = re.compile("cannot determine format of matrix .*", re.I)
        with pytest.raises(ValueError, match=pattern):
            io.read_csv_matrix(path)

    def test_invalid_format(self, tmp_path: pathlib.Path):
        """Test ValueError is raised when invalid format parameter is given."""
        path = tmp_path / "incorrect.csv"
        path.touch()

        format_ = "incorrect_17hSl"
        pattern = re.compile(f"unknown format {format_}", re.I)
        with pytest.raises(ValueError, match=pattern):
            io.read_csv_matrix(path, format_)


class TestFindFile:
    """Tests for the `find_file` function."""

    def test_correct(self):
        """Test single correct file exists and is found."""
        suffixes = [".csv.bz2"]
        expected = "test_file.csv.bz2"

        folder = _FakeGlobPath(suffixes)
        found = io.find_file_with_name(folder, "test_file", suffixes)

        assert found.name == expected, "incorrect file found"

    def test_correct_extras(self):
        """Test multiple file exists, highest priority is found and warning raised."""
        suffixes = [".csv.bz2", ".csv", ".txt"]
        expected = "test_file.csv.bz2"

        folder = _FakeGlobPath(suffixes)

        warn_msg = (
            f'Found {len(suffixes)} files named "test_file" with the expected'
            r" suffixes, the highest priority suffix is used\."
        )
        with pytest.warns(RuntimeWarning, match=warn_msg):
            found = io.find_file_with_name(folder, "test_file", suffixes)

        assert found.name == expected, "incorrect file found"

    def test_unexpected(self):
        """Test unexpected warning when additional files with different suffixes are found."""
        suffixes = [".csv.bz2"]
        extras = [".xlsx", ".test"]
        expected = "test_file.csv.bz2"

        folder = _FakeGlobPath(suffixes + extras)

        warn_msg = (
            f'Found {len(extras)} files named "test_file" with unexpected'
            rf' suffixes \({", ".join(re.escape(i) for i in extras)}\),'
            r" these are ignored\."
        )
        with pytest.warns(RuntimeWarning, match=warn_msg):
            found = io.find_file_with_name(folder, "test_file", suffixes)

        assert found.name == expected, "incorrect file found"

    def test_not_found(self):
        """Test `FileNotFoundError` is raised when no files are found."""
        with pytest.raises(FileNotFoundError):
            io.find_file_with_name(_FakeGlobPath([]), "test_file", [".csv"])
