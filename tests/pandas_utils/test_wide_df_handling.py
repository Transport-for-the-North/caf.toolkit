"""Tests for the {} module"""

# Built-Ins
import dataclasses
import operator
from collections.abc import Callable
from typing import Any

# Third Party
import numpy as np
import pandas as pd
import pytest

# Local Imports
from caf.toolkit import pandas_utils as pd_utils

# # # CONSTANTS # # #


# # # CLASSES # # #
@dataclasses.dataclass
class WideMatrixData:
    """Store testing data for a wide matrix.

    Some helper functions to replicate the functions we're testing at their
    most basic. This should make the tests more readable.
    """

    matrix: pd.DataFrame

    def index_mask(self, selection: list[int]) -> np.ndarray:
        """Get the mask for index matching in the shape of self.matrix."""
        mask = np.isin(self.matrix.index.to_numpy(), selection)
        return np.broadcast_to(mask, self.matrix.shape).T

    def col_mask(self, selection: list[int]) -> np.ndarray:
        """Get the mask for col matching in the shape of self.matrix."""
        mask = np.isin(self.matrix.columns.to_numpy(), selection)
        return np.broadcast_to(mask, self.matrix.shape)

    def full_mask(
        self,
        index_select: list[int],
        col_select: list[int],
        join_fn: Callable,
    ) -> np.ndarray:
        """Get the full mask of rows and columns in the shape of self.matrix."""
        return join_fn(self.col_mask(col_select), self.index_mask(index_select))


@dataclasses.dataclass
class ReportMatrixData(WideMatrixData):
    """Store testing data for the wide_matrix_internal_external_report()"""

    expected_report: pd.DataFrame
    int_select: list[int]
    ext_select: list[int]

    def get_kwargs(self) -> dict[str, Any]:
        """Produce a kwarg dict. Prevents repeated code."""
        return {
            "df": self.matrix,
            "int_select": self.int_select,
            "ext_select": self.ext_select,
        }


# # # FIXTURES # # #
def random_matrix(shape: tuple[int,]) -> pd.DataFrame:
    """Produce a random matrix of a given shape"""
    rng = np.random.default_rng()
    return pd.DataFrame(rng.random(shape))


@pytest.fixture(name="random_square_matrix", scope="function")
def fixture_random_square_matrix(request) -> WideMatrixData:
    """Fixture to create random matrix of set dimensions"""
    shape = (request.param, request.param)
    return WideMatrixData(matrix=random_matrix(shape))


@pytest.fixture(name="non_square_matrix", scope="module")
def fixture_non_square_matrix(request) -> WideMatrixData:
    """Fixture to create random matrix of set dimensions"""
    return WideMatrixData(matrix=random_matrix(request.param))


# # # TESTS # # #
class TestGetWideMask:
    """Tests for get_wide_mask()"""

    # TODO(BT): Add tests for datatypes. Different + fuzzy matching. Hoping some
    #  good test cases come out of real usage

    @pytest.mark.parametrize("random_square_matrix", (5, 11, 13), indirect=True)
    @pytest.mark.parametrize("select", [[0], [1, 2], [1, 3, 4]])
    def test_same_select_square_correct(
        self,
        random_square_matrix: WideMatrixData,
        select: list[int],
    ):
        """Test that we get the correct mask for a square matrix."""
        join_fn = operator.and_
        result = pd_utils.get_wide_mask(
            random_square_matrix.matrix, select=select, join_fn=join_fn
        )

        # Expected result
        expected = random_square_matrix.full_mask(
            index_select=select,
            col_select=select,
            join_fn=join_fn,
        )

        # Should be a boolean array, so exactly the same
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize("random_square_matrix", (5, 11, 13), indirect=True)
    @pytest.mark.parametrize("index_select", [[0], [1, 2], [1, 3, 4]])
    @pytest.mark.parametrize("col_select", [[0], [1, 2], [1, 3, 4]])
    def test_different_select_square_correct(
        self,
        random_square_matrix: WideMatrixData,
        index_select: list[int],
        col_select: list[int],
    ):
        """Test that we get the correct mask for a square matrix."""
        join_fn = operator.and_
        result = pd_utils.get_wide_mask(
            random_square_matrix.matrix,
            col_select=col_select,
            index_select=index_select,
            join_fn=join_fn,
        )

        # Expected result
        expected = random_square_matrix.full_mask(
            index_select=index_select,
            col_select=col_select,
            join_fn=join_fn,
        )

        # Should be a boolean array, so exactly the same
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize("random_square_matrix", (5, 11, 13), indirect=True)
    @pytest.mark.parametrize("select", [[0], [1, 2], [1, 3, 4]])
    @pytest.mark.parametrize("join_fn", [operator.or_, operator.and_])
    def test_join_fn_square_correct(
        self,
        random_square_matrix: WideMatrixData,
        select: list[int],
        join_fn: Callable,
    ):
        """Test that we get the correct mask for a square matrix."""
        result = pd_utils.get_wide_mask(
            random_square_matrix.matrix, select=select, join_fn=join_fn
        )

        # Expected result
        expected = random_square_matrix.full_mask(
            index_select=select,
            col_select=select,
            join_fn=join_fn,
        )

        # Should be a boolean array, so exactly the same
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize("non_square_matrix", ((6, 5), (11, 13), (7,)), indirect=True)
    def test_non_square_error(self, non_square_matrix: WideMatrixData):
        """Test an error is raised when the matrix is not square."""
        select = [1, 2]
        join_fn = operator.and_

        msg = "Only square matrices with 2 dimensions are supported"
        with pytest.raises(ValueError, match=msg):
            pd_utils.get_wide_mask(non_square_matrix.matrix, select=select, join_fn=join_fn)

    @pytest.mark.parametrize("random_square_matrix", [5], indirect=True)
    def test_bad_input_error(self, random_square_matrix: WideMatrixData):
        """Test an error is raised when bad input is used."""
        select = [1, 2]
        join_fn = operator.and_

        msg = "If selection is not set, both col_select and row_zones need to be set"
        with pytest.raises(ValueError, match=msg):
            pd_utils.get_wide_mask(
                random_square_matrix.matrix, col_select=select, join_fn=join_fn
            )

        with pytest.raises(ValueError, match=msg):
            pd_utils.get_wide_mask(
                random_square_matrix.matrix, index_select=select, join_fn=join_fn
            )

    @pytest.mark.parametrize("random_square_matrix", [5], indirect=True)
    @pytest.mark.parametrize("col_select", [[5], [7, 11]])
    @pytest.mark.parametrize("index_select", [[0], [1, 2], [1, 3, 4]])
    def test_no_col_match_warning(
        self,
        random_square_matrix: WideMatrixData,
        index_select: list[int],
        col_select: list[int],
    ):
        """Test that we get the correct mask for a square matrix."""
        join_fn = operator.and_

        # Expected result
        expected = random_square_matrix.full_mask(
            index_select=index_select,
            col_select=col_select,
            join_fn=join_fn,
        )

        msg = "No columns matched"
        with pytest.warns(UserWarning, match=msg):
            result = pd_utils.get_wide_mask(
                random_square_matrix.matrix,
                col_select=col_select,
                index_select=index_select,
                join_fn=join_fn,
            )

        # Should be a boolean array, so exactly the same
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize("random_square_matrix", [5], indirect=True)
    @pytest.mark.parametrize("index_select", [[5], [7, 11]])
    @pytest.mark.parametrize("col_select", [[0], [1, 2], [1, 3, 4]])
    def test_no_index_match_warning(
        self,
        random_square_matrix: WideMatrixData,
        index_select: list[int],
        col_select: list[int],
    ):
        """Test that we get the correct mask for a square matrix."""
        join_fn = operator.and_

        # Expected result
        expected = random_square_matrix.full_mask(
            index_select=index_select,
            col_select=col_select,
            join_fn=join_fn,
        )

        msg = "No index matched"
        with pytest.warns(UserWarning, match=msg):
            result = pd_utils.get_wide_mask(
                random_square_matrix.matrix,
                col_select=col_select,
                index_select=index_select,
                join_fn=join_fn,
            )

        # Should be a boolean array, so exactly the same
        np.testing.assert_array_equal(result, expected)


class TestInternalExternalHelpers:
    """Tests for get_wide_internal_only_mask() and get_wide_all_external_mask()"""

    @pytest.mark.parametrize("random_square_matrix", (5, 11, 13), indirect=True)
    @pytest.mark.parametrize("select", [[0], [1, 2], [1, 3, 4]])
    def test_internal_correct(
        self,
        random_square_matrix: WideMatrixData,
        select: list[int],
    ):
        """Test that we get the correct mask for a square matrix."""
        join_fn = operator.and_
        result = pd_utils.get_wide_mask(
            random_square_matrix.matrix, select=select, join_fn=join_fn
        )

        # Expected result
        expected = random_square_matrix.full_mask(
            index_select=select,
            col_select=select,
            join_fn=join_fn,
        )

        # Should be a boolean array, so exactly the same
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize("random_square_matrix", (5, 11, 13), indirect=True)
    @pytest.mark.parametrize("select", [[0], [1, 2], [1, 3, 4]])
    def test_external_correct(
        self,
        random_square_matrix: WideMatrixData,
        select: list[int],
    ):
        """Test that we get the correct mask for a square matrix."""
        join_fn = operator.or_
        result = pd_utils.get_wide_mask(
            random_square_matrix.matrix, select=select, join_fn=join_fn
        )

        # Expected result
        expected = random_square_matrix.full_mask(
            index_select=select,
            col_select=select,
            join_fn=join_fn,
        )

        # Should be a boolean array, so exactly the same
        np.testing.assert_array_equal(result, expected)


class TestWideMatrixIntExtReport:
    """Tests for wide_matrix_internal_external_report()"""

    @pytest.fixture(name="correct_report", scope="class")
    def fixture_correct_report(self) -> ReportMatrixData:
        """Create an example of a correct input and report"""
        int_select = [0, 1, 2]
        ext_select = [3, 4]
        mat = pd.DataFrame(np.arange(25).reshape(5, -1))

        cols = ["internal", "external", "total"]
        data = [[54, 51, 105], [111, 84, 195], [165, 135, 300]]
        report = pd.DataFrame(data=data, index=cols, columns=cols).astype(float)
        return ReportMatrixData(
            matrix=mat,
            expected_report=report,
            int_select=int_select,
            ext_select=ext_select,
        )

    @pytest.fixture(name="missing_bad_report", scope="class")
    def fixture_missing_bad_report(self, correct_report: ReportMatrixData) -> ReportMatrixData:
        """Create an example of a bad report where some df values are missed."""
        int_select = [1, 2, 3]
        ext_select = [4, 5]

        report = pd.DataFrame(
            data=[[108, 42, 150], [66, 24, 90], [174, 66, 240]],
            index=correct_report.expected_report.index,
            columns=correct_report.expected_report.columns,
        ).astype(float)

        return ReportMatrixData(
            matrix=correct_report.matrix,
            expected_report=report,
            int_select=int_select,
            ext_select=ext_select,
        )

    @pytest.fixture(name="overlap_bad_report", scope="class")
    def fixture_overlap_bad_report(self, correct_report: ReportMatrixData) -> ReportMatrixData:
        """Create an example of a bad report where some df values are counted twice."""
        int_select = [0, 1, 2]
        ext_select = [2, 3, 4]

        report = pd.DataFrame(
            data=[[54, 72, 126], [144, 162, 306], [198, 234, 432]],
            index=correct_report.expected_report.index,
            columns=correct_report.expected_report.columns,
        ).astype(float)

        return ReportMatrixData(
            matrix=correct_report.matrix,
            expected_report=report,
            int_select=int_select,
            ext_select=ext_select,
        )

    def test_correct(self, correct_report: ReportMatrixData):
        """Test that the correct result is produced with correct data."""
        result = pd_utils.wide_matrix_internal_external_report(**correct_report.get_kwargs())
        pd.testing.assert_frame_equal(result, correct_report.expected_report)

    def test_missing_warn(self, missing_bad_report: ReportMatrixData):
        """Test that a warning is raised when not all values used."""
        msg = "do not contain all values"
        with pytest.warns(UserWarning, match=msg):
            result = pd_utils.wide_matrix_internal_external_report(
                **missing_bad_report.get_kwargs()
            )
        pd.testing.assert_frame_equal(result, missing_bad_report.expected_report)

    def test_overlap_warn(self, overlap_bad_report: ReportMatrixData):
        """Test that a warning is raised when not all values used."""
        msg = "overlapping values"
        with pytest.warns(UserWarning, match=msg):
            result = pd_utils.wide_matrix_internal_external_report(
                **overlap_bad_report.get_kwargs()
            )
        pd.testing.assert_frame_equal(result, overlap_bad_report.expected_report)
