"""Contains functions that perform checks and provide high level statistics."""

from __future__ import annotations

# Built-Ins
import warnings
from typing import TYPE_CHECKING

# Third Party
import pandas as pd

# Local Imports
from caf.toolkit import cost_utils, translation

if TYPE_CHECKING:
    import pathlib

_EXCEL_SHEET_NAME_LIMIT = 31


class MatrixReport:
    """Creates a high level summary of a matrix and its trip ends.

    Parameters
    ----------
    matrix : pd.DataFrame
         The matrix to be summarised.
    translation : translation.ZoneCorrespondence
        A translation object to be applied to the matrix,.

    See Also
    --------
    from_file, write_to_excel
    """

    def __init__(
        self,
        matrix: pd.DataFrame,
        translation_vector: translation.ZoneCorrespondence,
    ) -> None:

        self._matrix = matrix.sort_index(axis=0).sort_index(axis=1)
        self._describe: pd.DataFrame | None = None
        self._distribution: pd.DataFrame | None = None
        self._vkms: pd.Series | None = None
        self._translation_vector: translation.ZoneCorrespondence = translation_vector

        self._translated_matrix: pd.DataFrame = (
            translation.pandas_matrix_zone_translation(
                matrix, self._translation_vector, check_totals=True
            )
        )

    def calc_vehicle_kms(
        self,
        cost_matrix: pd.DataFrame,
        *,
        sector_zone_lookup: translation.ZoneCorrespondence | None = None,
    ) -> None:
        """Calculate vehicle kms from the matrix passed on initialisation.

        The result is stored within the object which can be accessed
        using the `MatrixReport.vkms` property.
        VKMs are calculated as the sum of the product of the cost matrix and the matrix.

        Parameters
        ----------
        cost_matrix : pd.DataFrame
            Cost matrix corresponding with the inputted matrix.
        sector_zone_lookup: pd.DataFrame | None = None,
            translation vector to translate zones to sectors to create a
            sectorised distribution
        """
        cost_matrix = cost_matrix.sort_index(axis=0).sort_index(axis=1)
        if not (
            cost_matrix.index.equals(self._matrix.index)
            and cost_matrix.columns.equals(self._matrix.columns)
        ):
            raise ValueError(
                "Cost matrix must have the same index and columns as the matrix"
            )

        zonal_kms = self._matrix.multiply(cost_matrix)

        origin_kms = zonal_kms.sum(axis=1)

        if sector_zone_lookup is None:
            self._vkms = pd.Series({"vkms": origin_kms.sum()}, name="vkms")

        else:
            sector_kms = translation.pandas_vector_zone_translation(
                origin_kms, sector_zone_lookup
            )
            sector_kms.name = "vkms"
            sector_kms.index.name = None
            self._vkms = sector_kms

    def trip_length_distribution(
        self,
        cost_matrix: pd.DataFrame,
        bins: list[int],
        *,
        sector_zone_lookup: translation.ZoneCorrespondence | None = None,
    ) -> None:
        """Calculate a distribution from the matrix passed on initialisation.

        Distribution is stored within the object which can be accessed using
        the `MatrixReport.distribution` property.

        Parameters
        ----------
        cost_matrix : pd.DataFrame
            Cost matrix corresponding with the inputted matrix.
        bins : list[int]
            Bins to use for the distribution.
        sector_zone_lookup: translation.ZoneCorrespondence | None = None,
            Lookup vector to translate zones to sectors to create a sectorised distribution
        """
        try:
            cost_matrix.index = pd.to_numeric(cost_matrix.index, downcast="integer")  # type: ignore[call-overload]
            cost_matrix.columns = pd.to_numeric(cost_matrix.columns, downcast="integer")  # type: ignore[call-overload]
        except ValueError:
            pass

        if not (
            cost_matrix.index.equals(self._matrix.index)
            and cost_matrix.columns.equals(self._matrix.columns)
        ):
            raise ValueError(
                "Cost matrix must have the same index and columns as the matrix"
            )

        if sector_zone_lookup is None:
            cost_matrix = cost_matrix.loc[self._matrix.index, self._matrix.columns]  # type: ignore[index]
            self._distribution = cost_utils.CostDistribution.from_data(
                self._matrix.to_numpy(), cost_matrix.to_numpy(), bin_edges=bins
            ).df.set_index(["min", "max"])

            return

        index_check: bool = (
            sector_zone_lookup.from_column.sort_values().tolist()
            == self._matrix.sort_index().index.tolist()
        )
        col_check: bool = (
            sector_zone_lookup.from_column.sort_values().tolist()
            == self._matrix.columns.sort_values().tolist()
        )

        if not (index_check and col_check):
            raise KeyError("Zones in sector_zone_lookup must contain all zones ")

        stacked_distribution = []
        for sector in sector_zone_lookup.to_column.unique():
            zones = sector_zone_lookup.get_correspondence(sector, filter_on="to")[
                sector_zone_lookup.from_col_name
            ]
            if len(zones) == 0:
                raise KeyError("No zones found")

            cut_matrix = self._matrix.loc[zones, :]
            cut_cost_matrix = cost_matrix.loc[cut_matrix.index, cut_matrix.columns]  # type: ignore[index]
            sector_distribution = cost_utils.CostDistribution.from_data(
                cut_matrix.to_numpy(), cut_cost_matrix.to_numpy(), bin_edges=bins
            ).df
            sector_distribution[sector_zone_lookup.to_col_name] = sector
            # sector_distribution.set_index([sector_column, "min", "max"], append=True)
            stacked_distribution.append(sector_distribution)
        self._distribution = pd.concat(stacked_distribution).set_index(
            [sector_zone_lookup.to_col_name, "min", "max"]
        )

    def write_to_excel(
        self,
        writer: pd.ExcelWriter,
        label: str | None = None,
        output_sector_matrix: bool = False,
    ) -> None:
        """Write the report to an Excel file.

        Parameters
        ----------
        writer : pd.ExcelWriter
            Excel writer to write the report with.
        label : Optional[str], optional
            Added to the sheet names to define the matrix, by default None.
        output_matrix : bool, optional
            Whether to output a sectorised matrix sheet to the Excel file, by default False.

        Raises
        ------
        ValueError
            If the `label` is over 30 characters long.
        """
        if label is not None:
            sheet_prefix: str = f"{label}_"
        else:
            sheet_prefix = ""

        # Name limit minus longest suffix (12)
        if len(sheet_prefix) >= _EXCEL_SHEET_NAME_LIMIT - 12:
            raise ValueError(
                f"label cannot be over {_EXCEL_SHEET_NAME_LIMIT} characters as"
                " the sheets names will be truncated and will not be unique"
            )

        self.describe.to_excel(writer, sheet_name=f"{sheet_prefix}Summary")

        self.trip_ends.to_excel(writer, sheet_name=f"{sheet_prefix}Trip_Ends")

        if output_sector_matrix is True:
            if self.sector_matrix is not None:
                self.sector_matrix.to_excel(writer, sheet_name=f"{sheet_prefix}Matrix")
            else:
                warnings.warn(
                    "Cannot output sectorised matrix unless you"
                    " pass the translation vector on init",
                    stacklevel=2,
                )

        if self.distribution is not None:
            self.distribution.to_excel(writer, sheet_name=f"{sheet_prefix}Distribution")

    @property
    def matrix(self) -> pd.DataFrame:
        """Matrix in the original zoning system."""
        return self._matrix.copy()

    def abs_difference(self, other: MatrixReport) -> pd.DataFrame:
        """Calculate the absolute difference between to matrices and sectorise output.

        Absolute difference is calculated in original zone system before aggregating

        Returns
        -------
        pd.DataFrame
            Square sector matrix containing the modulus of the difference
            between the matrix and other. Note that the difference is calculated
            at the zone level before summing the absolute values to sectors.


        Raises
        ------
        ValueError
            If other.matrix does not have identical columns and/or indices to self.matrix

        """
        if not (
            self.matrix.index.equals(other.matrix.index)
            and other.matrix.columns.equals(other.matrix.columns)
        ):
            raise ValueError("The matrices must have the same index and columns.")

        matrix_diff = (self.matrix - other.matrix).abs()

        return add_matrix_sums(
            translation.pandas_matrix_zone_translation(
                matrix_diff,
                self._translation_vector,
            )
        )

    @property
    def describe(self) -> pd.DataFrame:
        """High level statistics on the original and, if provided, sectorised matrix."""
        if self._describe is None:
            data = {"Matrix": matrix_describe(self._matrix)}
            if self.sector_matrix is not None:
                data["Translated_Matrix"] = matrix_describe(self.sector_matrix)

            self._describe = pd.DataFrame(data)

        return self._describe.copy()

    @property
    def sector_matrix(self) -> pd.DataFrame | None:
        """Sector matrix if translation vector provided, otherwise none."""
        if isinstance(self._translated_matrix, pd.DataFrame):
            return add_matrix_sums(self._translated_matrix)
        return self._translated_matrix

    @property
    def distribution(self) -> pd.DataFrame | None:
        """Distribution if `trip_length_distribution` has been called, otherwise none."""
        if self._distribution is None:
            warnings.warn("Trip Length Distribution has not been set", stacklevel=2)
        return self._distribution

    @property
    def vkms(self) -> pd.Series | None:
        """Vehicle kms if `calc_vehicle_kms` has been called, otherwise none."""
        if self._vkms is None:
            warnings.warn("Trip VKMs has not been set", stacklevel=2)
        return self._vkms

    @property
    def trip_ends(self) -> pd.DataFrame:
        """The row and column sums of the matrix."""
        return pd.DataFrame({"row_sums": self.row_sum, "col_sums": self.column_sum})

    @property
    def row_sum(self) -> pd.Series:
        """The row sums of the matrix."""
        if self._translated_matrix is not None:
            return self._translated_matrix.sum(axis=1)
        return self._matrix.sum(axis=1)

    @property
    def column_sum(self) -> pd.Series:
        """The column sums of the matrix."""
        if self._translated_matrix is not None:
            return self._translated_matrix.sum(axis=0)
        return self._matrix.sum(axis=0)

    @classmethod
    def from_file(
        cls,
        path: pathlib.Path,
        *,
        translation_path: translation.ZoneCorrespondencePath,
        factors_mandatory: bool = True,
    ) -> MatrixReport:
        """Create an instance of MatrixReport from file paths.

        Parameters
        ----------
        path : pathlib.Path
            Path to the matrix csv.
        translation_path : pathlib.Path
            Path to correspondence between matrix zoning and summary zoning.
        translation_from_col : str
            The column in the translation matrix with zoning to translate from.
        translation_to_col : str
            The column in the translation matrix with zoning to translate to.
        translation_factors_col : str
            The column in the translation matrix to use as factors.

        Returns
        -------
        MatrixReport
            Instance of MatrixReport created from the file paths.
        """
        matrix = pd.read_csv(path, index_col=0)
        translation_factors = translation_path.read(factors_mandatory=factors_mandatory)

        return cls(
            matrix,
            translation_vector=translation_factors,
        )


def matrix_describe(
    matrix: pd.DataFrame, almost_zero: float | None = None
) -> pd.Series:
    """Create a high level summary of a matrix.

    Stack Matrix before calling pandas describe with additional metrics added.

    Parameters
    ----------
    matrix : pd.DataFrame
        Matrix to be summarised.
    almost_zero : float, optional
        Below this value cells will be defined as almost zero.
        If not given, will be calculated as = 1 / (# of cells in the matrix).

    Returns
    -------
    pd.Series
        Matrix summary statistics, expands upon the standard pandas.Series.describe.
        Includes
        5%, 25%, 50%, 75%, 95% Percentiles
        Mean
        Count (total, zeros and almost zeros)
        Standard Deviation
        Minimum and Maximum

    See Also
    --------
    `pandas.Series.describe`
    """
    if almost_zero is None:
        almost_zero = 1 / matrix.size

    info = matrix.stack().describe(
        percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]
    )
    if not isinstance(info, pd.Series):
        raise TypeError(f"info should be Series not: {type(info)}")

    info["columns"] = len(matrix.columns)
    info["rows"] = len(matrix.index)
    info["sum"] = matrix.sum().sum()
    info["zeros"] = (matrix == 0).sum().sum()
    info["almost_zeros"] = (matrix < almost_zero).sum().sum()
    info["NaNs"] = matrix.isna().sum().sum()
    return info


def compare_matrices(
    matrix_report_a: MatrixReport,
    matrix_report_b: MatrixReport,
    name_a: str = "a",
    name_b: str = "b",
) -> dict[str, pd.DataFrame]:
    """Compare two matrix reports.

    Parameters
    ----------
    matrix_report_a : MatrixReport
        Matrix for comparison,
        this matrix will be the numerator in proportional comparisons.
    matrix_report_b : MatrixReport
        Other matrix for comparison,
        this matrix will be the denominator in proportional comparisons.
    name_a : str, optional
        name to label matrix_report_a, by default "a"
    name_b : str, optional
        name to label matrix_report_b, by default "b"

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary containing comparison statistics.

    Raises
    ------
    ValueError
        if either matrix report does not have a sector matrix.
    """
    comparisons = {}
    if matrix_report_a.sector_matrix is None or matrix_report_b.sector_matrix is None:
        raise ValueError("matrix reports must be sectorised to perform a comparison")

    comparisons[f"{name_a} matrix"] = matrix_report_a.sector_matrix
    comparisons[f"{name_b} matrix"] = matrix_report_b.sector_matrix

    comparisons["matrix difference"] = (
        matrix_report_a.sector_matrix - matrix_report_b.sector_matrix
    )

    comparisons["matrix percentage"] = (
        matrix_report_a.sector_matrix / matrix_report_b.sector_matrix
    ) - 1

    comparisons["matrix abs difference"] = matrix_report_a.abs_difference(
        matrix_report_b
    )

    comparisons["matrix abs percentage"] = (
        comparisons["matrix abs difference"] / matrix_report_a.sector_matrix
    )

    comparisons["stats"] = pd.DataFrame(
        {
            name_a: matrix_report_a.describe["Matrix"],
            name_b: matrix_report_b.describe["Matrix"],
        }
    )

    trip_ends = matrix_report_a.trip_ends.merge(
        matrix_report_b.trip_ends,
        left_index=True,
        right_index=True,
        suffixes=(f"_{name_a}", f"_{name_b}"),
    )

    for i in ("row", "col"):
        trip_ends[f"{i}_sums_difference"] = (
            trip_ends[f"{i}_sums_{name_a}"] - trip_ends[f"{i}_sums_{name_b}"]
        )
        trip_ends[f"{i}_sums_percentage"] = (
            trip_ends[f"{i}_sums_{name_a}"] / trip_ends[f"{i}_sums_{name_b}"]
        ) - 1

    comparisons["Trip Ends"] = pd.DataFrame(trip_ends)
    if matrix_report_a.vkms is not None and matrix_report_b.vkms is not None:
        comparisons["Vkms"] = pd.DataFrame(
            {name_a: matrix_report_a.vkms, name_b: matrix_report_b.vkms}
        )

    if (
        matrix_report_a.distribution is not None
        and matrix_report_b.distribution is not None
    ):
        comparisons["TLD comparison"] = matrix_report_a.distribution.merge(
            matrix_report_b.distribution,
            left_index=True,
            right_index=True,
            suffixes=(f"_{name_a}", f"_{name_b}"),
        )

    return comparisons


def add_matrix_sums(df: pd.DataFrame) -> pd.DataFrame:
    """Add a sum column and row to a dataframe containing a matrix in square format.

    Does not change the original matrix.

    Parameters
    ----------
    df : pd.DataFrame
        Square matrix to add sum row and columns.

    Returns
    -------
    pd.DataFrame
        square matrix with sum row and columns.
    """
    df_sums = df.copy()
    df_sums.loc["sum"] = df_sums.sum(axis=0)
    df_sums["sum"] = df_sums.sum(axis=1)
    return df_sums


def compare_matrices_and_output(
    excel_writer: pd.ExcelWriter,
    matrix_report_a: MatrixReport,
    matrix_report_b: MatrixReport,
    *,
    name_a: str = "a",
    name_b: str = "b",
    label: str | None = None,
) -> None:
    """Compare two matrix reports.

    Parameters
    ----------
    excel_writer: pd.ExcelWriter
        Excel writer to use to output the comparisons.
    matrix_report_a : MatrixReport
        Matrix for comparison,
        this matrix will be the numerator in proportional comparisons.
    matrix_report_b : MatrixReport
        Other matrix for comparison,
        this matrix will be the denominator in proportional comparisons.
    name_a : str, optional
        name to label matrix_report_a, by default "a"
    name_b : str, optional
        name to label matrix_report_b, by default "b"
    label : Optional[str]
        Label to add to the sheet names
    """
    comparisons = compare_matrices(matrix_report_a, matrix_report_b, name_a, name_b)
    for name, result in comparisons.items():
        if label is not None:
            sheet_name = f"{label}_{name}"
        else:
            sheet_name = name

        if len(sheet_name) > _EXCEL_SHEET_NAME_LIMIT:
            warnings.warn(
                f"Sheet name {sheet_name} is over {_EXCEL_SHEET_NAME_LIMIT}"
                " characters and will be truncated",
                stacklevel=2,
            )
        result.to_excel(excel_writer, sheet_name=sheet_name)
