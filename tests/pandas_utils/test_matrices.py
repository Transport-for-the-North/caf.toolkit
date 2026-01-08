"""Tests for `pandas_utils` matrices functionality."""

# Built-Ins
import pathlib

# Third Party
import numpy as np
import pandas as pd
import pytest

# Local Imports
from caf.toolkit import cost_utils, translation
from caf.toolkit import pandas_utils as pd_utils

MATRIX_SIZE = 100


@pytest.fixture(name="matrix", scope="session")
def fixture_matrix() -> pd.DataFrame:
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
def fixture_cost_matrix() -> pd.DataFrame:
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
def fixture_translation() -> translation.ZoneCorrespondencePath:
    """Translation to test matrix report functionality"""
    trans_data = []

    for i in range(MATRIX_SIZE):
        trans_data.append({"from": i, "to": int(str(i)[0]), "factor": 1})

    return translation.ZoneCorrespondence(pd.DataFrame(trans_data), "from", "to", "factor")


class TestMatrices:
    """Test Matrices functionality."""

    def test_matrix_describe(self, matrix: pd.DataFrame) -> None:
        """Test that the matrix describe function produces the expected outputs."""
        test = pd_utils.matrix_describe(matrix)

        almost_zero = 1 / matrix.size

        control = matrix.stack().describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])  # noqa: PD013
        control["columns"] = len(matrix.columns)
        control["rows"] = len(matrix.index)
        control["sum"] = matrix.sum().sum()
        control["zeros"] = (matrix == 0).sum().sum()
        control["almost_zeros"] = (matrix < almost_zero).sum().sum()
        control["NaNs"] = matrix.isna().sum().sum()

        pd.testing.assert_series_equal(test, control, check_exact=False)

    def test_trip_ends(self, matrix: pd.DataFrame, translation_vector: pd.DataFrame) -> None:
        """Test the trip ends property produces the expected output."""
        matrix_report = pd_utils.MatrixReport(
            matrix,
            translation_vector=translation_vector,
        )

        test_trip_ends = matrix_report.trip_ends

        translated_matrix = translation.pandas_matrix_zone_translation(
            matrix, translation_vector
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
    ) -> None:
        """Test whether trip length distribution calculation works as expected."""
        matrix_report = pd_utils.MatrixReport(
            matrix,
            translation_vector=translation_vector,
        )
        matrix_report.trip_length_distribution(
            cost_matrix, [0, 1, 2, 5, 10, 20, 50, 100, 200, 400]
        )

        test_tld = matrix_report.distribution

        control_tld = cost_utils.CostDistribution.from_data(
            matrix.to_numpy(),
            cost_matrix,
            bin_edges=[0, 1, 2, 5, 10, 20, 50, 100, 200, 400],
        ).df.set_index(["min", "max"])

        pd.testing.assert_frame_equal(test_tld, control_tld)

    def test_multi_vkms(
        self,
        matrix: pd.DataFrame,
        translation_vector: translation.ZoneCorrespondence,
        cost_matrix: pd.DataFrame,
    ) -> None:
        """Test whether multi area vkm calculation works as expected."""
        matrix_report = pd_utils.MatrixReport(
            matrix,
            translation_vector=translation_vector,
        )
        matrix_report.calc_vehicle_kms(
            cost_matrix,
            sector_zone_lookup=translation_vector,
        )

        test_vkms = matrix_report.vkms

        zonal_kms = matrix.multiply(cost_matrix)
        origin_kms = zonal_kms.sum(axis=1)
        sector_replace = translation_vector.vector.set_index("from")["to"].to_dict()
        sector_kms = origin_kms.rename(sector_replace).groupby(level=0).sum()
        sector_kms.name = "vkms"

        pd.testing.assert_series_equal(test_vkms, sector_kms)

    def test_vkms(
        self,
        matrix: pd.DataFrame,
        translation_vector: translation.ZoneCorrespondence,
        cost_matrix: pd.DataFrame,
    ) -> None:
        """Test whether vkm calculation works as expected."""
        matrix_report = pd_utils.MatrixReport(
            matrix,
            translation_vector=translation_vector,
        )
        matrix_report.calc_vehicle_kms(cost_matrix)

        test_vkms = matrix_report.vkms

        zonal_kms = matrix.multiply(cost_matrix)

        origin_kms = zonal_kms.sum(axis=1)

        vkms = pd.Series({"vkms": origin_kms.sum()}, name="vkms")

        pd.testing.assert_series_equal(test_vkms, vkms)

    def test_multi_trip_length_distribution(
        self,
        matrix: pd.DataFrame,
        translation_vector: translation.ZoneCorrespondence,
        cost_matrix: pd.DataFrame,
    ) -> None:
        """Test whether the multi area TLD works as expected."""
        matrix_report = pd_utils.MatrixReport(
            matrix,
            translation_vector=translation_vector,
        )
        matrix_report.trip_length_distribution(
            cost_matrix,
            [0, 1, 2, 5, 10, 20, 50, 100, 200, 400],
            sector_zone_lookup=translation_vector,
        )

        test_tld = matrix_report.distribution

        stacked_distribution = []
        for sector in translation_vector.vector["to"].unique():
            zones = translation_vector.vector.loc[
                translation_vector.vector["to"] == sector, "from"
            ]
            cut_matrix = matrix.loc[zones, :]
            cut_cost_matrix = cost_matrix.loc[cut_matrix.index, cut_matrix.columns]  # type: ignore[index]
            sector_distribution = cost_utils.CostDistribution.from_data(
                cut_matrix.to_numpy(),
                cut_cost_matrix.to_numpy(),
                bin_edges=[0, 1, 2, 5, 10, 20, 50, 100, 200, 400],
            ).df
            sector_distribution["to"] = sector
            stacked_distribution.append(sector_distribution)
        control_tld = pd.concat(stacked_distribution).set_index(["to", "min", "max"])

        pd.testing.assert_frame_equal(test_tld, control_tld)

    def test_writing_matrix_report(
        self,
        matrix: pd.DataFrame,
        translation_vector: translation.ZoneCorrespondence,
        cost_matrix: pd.DataFrame,
        tmp_path: pathlib.Path,
    ) -> None:
        """Test whether writing out the matrix report classes executes without erroring."""
        matrix_report = pd_utils.MatrixReport(
            matrix,
            translation_vector=translation_vector,
        )
        matrix_report.trip_length_distribution(
            cost_matrix,
            [0, 1, 2, 5, 10, 20, 50, 100, 200, 400],
            sector_zone_lookup=translation_vector,
        )
        matrix_report.calc_vehicle_kms(
            cost_matrix,
            sector_zone_lookup=translation_vector,
        )
        with pd.ExcelWriter(tmp_path / "test.xlsx", mode="w") as writer:
            matrix_report.write_to_excel(writer, "test", True)

        assert True


class TestCompareMatricesAndOutput:
    """Check compare_matrices_and_output functions as expected."""

    def test_write_comparison(
        self,
        tmp_path: pathlib.Path,
        matrix: pd.DataFrame,
        cost_matrix: pd.DataFrame,
        translation_vector: translation.ZoneCorrespondence,
    ):
        """Check that compare_matrices_and_output writes the expected
        sheets.

        Content of the sheets has not been tested as this should be covered
        by TestMatrixComparison.
        """
        matrix_report = pd_utils.MatrixReport(
            matrix,
            translation_vector=translation_vector,
        )

        matrix_report.trip_length_distribution(
            cost_matrix,
            [0, 1, 2, 5, 10, 20, 50, 100, 200, 400],
            sector_zone_lookup=translation_vector,
        )

        matrix_report.calc_vehicle_kms(
            matrix,
            sector_zone_lookup=translation_vector,
        )

        expected_sheets = [
            "a matrix",
            "b matrix",
            "matrix difference",
            "matrix percentage",
            "stats",
            "Trip Ends",
            "TLD comparison",
            "Vkms",
        ]

        path = tmp_path / "multi_tld_output.xlsx"

        with pd.ExcelWriter(path) as writer:
            pd_utils.compare_matrices_and_output(writer, matrix_report, matrix_report)

        output = pd.read_excel(path, sheet_name=None)

        for sheet in expected_sheets:
            assert sheet in output, f"{sheet} not in output"


class TestMatrixComparison:
    """Test the compare_matrices function."""

    @pytest.mark.filterwarnings("ignore:Trip .* has not been set:UserWarning")
    def test_comparison_sector_matrix(
        self, matrix: pd.DataFrame, translation_vector: translation.ZoneCorrespondence
    ) -> None:
        """Test sector matrix comparison produces expected results."""
        matrix_report = pd_utils.MatrixReport(
            matrix,
            translation_vector=translation_vector,
        )

        comparison = pd_utils.compare_matrices(matrix_report, matrix_report)

        assert len(comparison) != 0

        pd.testing.assert_frame_equal(comparison["a matrix"], matrix_report.sector_matrix)
        pd.testing.assert_frame_equal(comparison["b matrix"], matrix_report.sector_matrix)
        pd.testing.assert_frame_equal(
            comparison["matrix difference"],
            pd.DataFrame(
                0,
                index=matrix_report.sector_matrix.index,
                columns=matrix_report.sector_matrix.columns,
                dtype=np.float64,
            ),
        )
        pd.testing.assert_frame_equal(
            comparison["matrix percentage"],
            pd.DataFrame(
                0,
                index=matrix_report.sector_matrix.index,
                columns=matrix_report.sector_matrix.columns,
                dtype=np.float64,
            ),
        )
        pd.testing.assert_frame_equal(
            comparison["matrix abs difference"],
            pd.DataFrame(
                0,
                index=matrix_report.sector_matrix.index,
                columns=matrix_report.sector_matrix.columns,
                dtype=np.float64,
            ),
        )
        pd.testing.assert_frame_equal(
            comparison["matrix abs percentage"],
            pd.DataFrame(
                0,
                index=matrix_report.sector_matrix.index,
                columns=matrix_report.sector_matrix.columns,
                dtype=np.float64,
            ),
        )

    @pytest.mark.filterwarnings("ignore:Trip .* has not been set:UserWarning")
    def test_comparison_trip_ends(
        self, matrix: pd.DataFrame, translation_vector: translation.ZoneCorrespondence
    ) -> None:
        """Test Trip End Comparison produces expected results."""
        matrix_report = pd_utils.MatrixReport(
            matrix,
            translation_vector=translation_vector,
        )

        comparison = pd_utils.compare_matrices(matrix_report, matrix_report)

        assert len(comparison) != 0

        trip_ends = comparison["Trip Ends"]
        pd.testing.assert_series_equal(
            trip_ends["row_sums_a"],
            matrix_report.trip_ends["row_sums"],
            check_names=False,
        )
        pd.testing.assert_series_equal(
            trip_ends["col_sums_a"],
            matrix_report.trip_ends["col_sums"],
            check_names=False,
        )
        pd.testing.assert_series_equal(
            trip_ends["row_sums_b"],
            matrix_report.trip_ends["row_sums"],
            check_names=False,
        )
        pd.testing.assert_series_equal(
            trip_ends["col_sums_b"],
            matrix_report.trip_ends["col_sums"],
            check_names=False,
        )

        assert (trip_ends["row_sums_difference"] == 0).all()
        assert (trip_ends["col_sums_difference"] == 0).all()
        assert (trip_ends["col_sums_percentage"] == 0).all()
        assert (trip_ends["row_sums_percentage"] == 0).all()

    @pytest.mark.filterwarnings("ignore:Trip .* has not been set:UserWarning")
    def test_comparison_stats(
        
        self, matrix: pd.DataFrame, translation_vector: translation.ZoneCorrespondence
    
    ) -> None:
        """Check Stats produces expected results."""
        matrix_report = pd_utils.MatrixReport(
            matrix,
            translation_vector=translation_vector,
        )

        comparison = pd_utils.compare_matrices(matrix_report, matrix_report)

        assert len(comparison) != 0

        pd.testing.assert_frame_equal(comparison["a matrix"], matrix_report.sector_matrix)
        pd.testing.assert_frame_equal(comparison["b matrix"], matrix_report.sector_matrix)
        pd.testing.assert_frame_equal(
            comparison["matrix difference"],
            pd.DataFrame(
                0,
                index=matrix_report.sector_matrix.index,
                columns=matrix_report.sector_matrix.columns,
                dtype=np.float64,
            ),
        )
        pd.testing.assert_frame_equal(
            comparison["matrix percentage"],
            pd.DataFrame(
                0,
                index=matrix_report.sector_matrix.index,
                columns=matrix_report.sector_matrix.columns,
                dtype=np.float64,
            ),
        )
        pd.testing.assert_frame_equal(
            comparison["matrix abs difference"],
            pd.DataFrame(
                0,
                index=matrix_report.sector_matrix.index,
                columns=matrix_report.sector_matrix.columns,
                dtype=np.float64,
            ),
        )
        pd.testing.assert_frame_equal(
            comparison["matrix abs percentage"],
            pd.DataFrame(
                0,
                index=matrix_report.sector_matrix.index,
                columns=matrix_report.sector_matrix.columns,
                dtype=np.float64,
            ),
        )
        trip_ends = comparison["Trip Ends"]
        pd.testing.assert_series_equal(
            trip_ends["row_sums_a"],
            matrix_report.trip_ends["row_sums"],
            check_names=False,
        )
        pd.testing.assert_series_equal(
            trip_ends["col_sums_a"],
            matrix_report.trip_ends["col_sums"],
            check_names=False,
        )
        pd.testing.assert_series_equal(
            trip_ends["row_sums_b"],
            matrix_report.trip_ends["row_sums"],
            check_names=False,
        )
        pd.testing.assert_series_equal(
            trip_ends["col_sums_b"],
            matrix_report.trip_ends["col_sums"],
            check_names=False,
        )

        assert (trip_ends["row_sums_difference"] == 0).all()
        assert (trip_ends["col_sums_difference"] == 0).all()
        assert (trip_ends["col_sums_percentage"] == 0).all()
        assert (trip_ends["row_sums_percentage"] == 0).all()

        pd.testing.assert_series_equal(
            matrix_report.describe["Matrix"],
            comparison["stats"]["a"],
            check_names=False,
        )
        pd.testing.assert_series_equal(
            matrix_report.describe["Matrix"],
            comparison["stats"]["b"],
            check_names=False,
        )

    @pytest.mark.filterwarnings("ignore:Trip Length Distribution has not been set:UserWarning")
    def test_comparison_vkms(
        
        self, matrix: pd.DataFrame, translation_vector: translation.ZoneCorrespondence
    
    ) -> None:
        """Check Vkms produces expected results."""
        matrix_report = pd_utils.MatrixReport(
            matrix,
            translation_vector=translation_vector,
        )

        matrix_report.calc_vehicle_kms(
            matrix,
            sector_zone_lookup=translation_vector,
        )

        comparison = pd_utils.compare_matrices(matrix_report, matrix_report)

        pd.testing.assert_series_equal(
            comparison["Vkms"]["a"],
            matrix_report.vkms,
            check_exact=False,
            check_names=False,
        )
        pd.testing.assert_series_equal(
            comparison["Vkms"]["b"],
            matrix_report.vkms,
            check_exact=False,
            check_names=False,
        )

    @pytest.mark.filterwarnings("ignore:Trip Length Distribution has not been set:UserWarning")
    def test_comparison_multi_vkms(
        self,
       
        matrix: pd.DataFrame,
       
        cost_matrix: pd.DataFrame,
       
        translation_vector: translation.ZoneCorrespondence,
    ) -> None:
        """Checks Multi-Area Vkms comparisons functions as expected."""
        matrix_report = pd_utils.MatrixReport(
            matrix,
            translation_vector=translation_vector,
        )

        matrix_report.calc_vehicle_kms(
            cost_matrix,
            sector_zone_lookup=translation_vector,
        )

        comparison = pd_utils.compare_matrices(matrix_report, matrix_report)

        pd.testing.assert_series_equal(
            comparison["Vkms"]["a"],
            matrix_report.vkms,
            check_exact=False,
            check_names=False,
        )
        pd.testing.assert_series_equal(
            comparison["Vkms"]["b"],
            matrix_report.vkms,
            check_exact=False,
            check_names=False,
        )

    @pytest.mark.filterwarnings("ignore:Trip VKMs has not been set:UserWarning")
    def test_comparison_multi_tlds(
        self,
       
        matrix: pd.DataFrame,
       
        cost_matrix: pd.DataFrame,
       
        translation_vector: translation.ZoneCorrespondence,,
    ) -> None:
        """Check Multi-Area TLDs functions as expected."""
        matrix_report = pd_utils.MatrixReport(
            matrix,
            translation_vector=translation_vector,
        )

        matrix_report.trip_length_distribution(
            cost_matrix,
            [0, 1, 2, 5, 10, 20, 50, 100, 200, 400],
            sector_zone_lookup=translation_vector,
        )

        comparison = pd_utils.compare_matrices(matrix_report, matrix_report)

        pd.testing.assert_frame_equal(
            comparison["TLD comparison"],
            matrix_report.distribution.merge(
                matrix_report.distribution,
                suffixes=["_a", "_b"],
                left_index=True,
                right_index=True,
            ),
            check_exact=False,
        )

        assert len(comparison["TLD comparison"].columns) == 6  # noqa: PLR2004

    @pytest.mark.filterwarnings("ignore:Trip VKMs has not been set:UserWarning")
    def test_comparison_tlds(
        self,
       
        matrix: pd.DataFrame,
       
        cost_matrix: pd.DataFrame,
       
        translation_vector: translation.ZoneCorrespondence,,
    ) -> None:
        """Check TLD comparison functions as expected."""
        matrix_report = pd_utils.MatrixReport(
            matrix,
            translation_vector=translation_vector,
        )

        matrix_report.trip_length_distribution(
            cost_matrix, [0, 1, 2, 5, 10, 20, 50, 100, 200, 400]
        )

        comparison = pd_utils.compare_matrices(matrix_report, matrix_report)

        pd.testing.assert_frame_equal(
            comparison["TLD comparison"],
            matrix_report.distribution.merge(
                matrix_report.distribution,
                suffixes=["_a", "_b"],
                left_index=True,
                right_index=True,
            ),
            check_exact=False,
        )

        assert len(comparison["TLD comparison"].columns) == 6  # noqa: PLR2004
