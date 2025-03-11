# Built-Ins
import pathlib

# Third Party
import numpy as np
import pandas as pd
import pytest

# Local Imports
from caf.toolkit import cost_utils
from caf.toolkit import pandas_utils as pd_utils
from caf.toolkit import translation

MATRIX_SIZE = 100


@pytest.fixture(name="matrix", scope="session")
def fixture_matrix():
    """Matrix to test matrix report functionality."""
    matrix_rows = []

    gen = np.random.default_rng(10)

    for i in range(MATRIX_SIZE):
        float_generator = pd_utils.random.FloatGenerator(i, MATRIX_SIZE, 1000)
        matrix_rows.append(float_generator.generate(gen))

    matrix = pd.DataFrame(matrix_rows)
    matrix.columns = matrix.index

    return matrix


@pytest.fixture(name="cost_matrix", scope="session")
def fixture_cost_matrix():
    """Matrix to test matrix report functionality."""
    matrix_rows = []

    gen = np.random.default_rng(30)

    for i in range(MATRIX_SIZE):
        float_generator = pd_utils.random.FloatGenerator(i, MATRIX_SIZE, 400)
        matrix_rows.append(float_generator.generate(gen))

    matrix = pd.DataFrame(matrix_rows)
    matrix.columns = matrix.index

    return matrix


@pytest.fixture(name="translation_vector", scope="session")
def fixture_translation() -> pd.DataFrame:
    """Translation to test matrix report functionality"""
    trans_data = []

    for i in range(MATRIX_SIZE):
        trans_data.append({"from": i, "to": int(str(i)[0]), "factor": 1})

    return pd.DataFrame(trans_data)


class TestMatrices:
    """Test Matrices functionality."""

    def test_matrix_describe(self, matrix: pd.DataFrame):
        """Test that the matrix describe function produces the expected outputs."""
        test = pd_utils.matrix_describe(matrix)

        almost_zero = 1 / matrix.size

        control = matrix.stack().describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
        control["columns"] = len(matrix.columns)
        control["rows"] = len(matrix.index)
        control["sum"] = matrix.sum().sum()
        control["zeros"] = (matrix == 0).sum().sum()
        control["almost_zeros"] = (matrix < almost_zero).sum().sum()
        control["NaNs"] = matrix.isna().sum().sum()

        pd.testing.assert_series_equal(test, control, check_exact=False)

    def test_trip_ends(self, matrix, translation_vector):
        """Test the trip ends property produces the expected output."""
        matrix_report = pd_utils.MatrixReport(
            matrix,
            translation_factors=translation_vector,
            translation_from_col="from",
            translation_to_col="to",
            translation_factors_col="factor",
        )

        test_trip_ends = matrix_report.trip_ends

        translated_matrix = translation.pandas_matrix_zone_translation(
            matrix, translation_vector, "from", "to", "factor"
        )

        control_trip_ends = pd.DataFrame(
            {
                "row_sums": translated_matrix.sum(axis=1),
                "col_sums": translated_matrix.sum(axis=0),
            }
        )

        pd.testing.assert_frame_equal(control_trip_ends, test_trip_ends)

    def test_trip_length_distribution(
        self,
        matrix: pd.DataFrame,
        translation_vector: pd.DataFrame,
        cost_matrix: pd.DataFrame,
    ):
        """Test whether writing out the matrix report classes executes without erroring."""
        matrix_report = pd_utils.MatrixReport(
            matrix,
            translation_factors=translation_vector,
            translation_from_col="from",
            translation_to_col="to",
            translation_factors_col="factor",
        )
        matrix_report.trip_length_distribution(
            cost_matrix, [0, 1, 2, 5, 10, 20, 50, 100, 200, 400]
        )

        test_tld = matrix_report.distribution

        control_tld = cost_utils.CostDistribution.from_data(
            matrix.to_numpy(), cost_matrix, bin_edges=[0, 1, 2, 5, 10, 20, 50, 100, 200, 400]
        )

        pd.testing.assert_frame_equal(test_tld.df, control_tld.df)

    def test_writing_matrix_report(
        self,
        matrix: pd.DataFrame,
        translation_vector: pd.DataFrame,
        cost_matrix: pd.DataFrame,
        tmp_path: pathlib.Path,
    ):
        """Test whether writing out the matrix report classes executes without erroring."""
        matrix_report = pd_utils.MatrixReport(
            matrix,
            translation_factors=translation_vector,
            translation_from_col="from",
            translation_to_col="to",
            translation_factors_col="factor",
        )
        matrix_report.trip_length_distribution(
            cost_matrix, [0, 1, 2, 5, 10, 20, 50, 100, 200, 400]
        )
        with pd.ExcelWriter(tmp_path / "test.xlsx", mode="w") as writer:
            matrix_report.write_to_excel(writer, "test", True)

        assert True
