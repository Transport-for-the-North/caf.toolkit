"""Contains functions that perform checks and provide high level statistics."""

from __future__ import annotations

# Built-Ins
import warnings
from pathlib import Path
from typing import Optional

# Third Party
import pandas as pd

# Local Imports
from caf.toolkit import cost_utils, translation


class MatrixReport:
    """Creates a high level summary of a matrix and its trip ends.

    Parameters
    ----------
    matrix : pd.DataFrame
         The matrix to be summarised.
    translation : Optional[pd.DataFrame], optional
        A translation matrix to be applied to the matrix,.
        If None no translations is applied, by default None.
    translation_from_col : Optional[str], optional
        The column in the translation matrix to translate from, by default None.
    translation_to_col : Optional[str], optional
        The column in the translation matrix to translate to, by default None.
    translation_factors_col : Optional[str], optional
        The column in the translation matrix to use as factors, by default None.

    See Also
    --------
    from_file, write_to_excel
    """

    def __init__(
        self,
        matrix: pd.DataFrame,
        *,
        translation_factors: Optional[pd.DataFrame] = None,
        translation_from_col: Optional[str] = None,
        translation_to_col: Optional[str] = None,
        translation_factors_col: Optional[str] = None,
    ):

        self._matrix = matrix
        self._translated_matrix: pd.DataFrame | None = None
        self._describe: pd.DataFrame | None = None
        self._distribution: cost_utils.CostDistribution | None = None

        if translation_factors is not None:
            if (
                (translation_factors_col is None)
                or (translation_from_col is None)
                or (translation_to_col is None)
            ):
                raise ValueError(
                    "If translation is provided translation_from_col,"
                    " translation_to_col and translation_factors_col "
                    "must also be given"
                )

            self._translated_matrix = translation.pandas_matrix_zone_translation(
                matrix,
                translation_factors,
                translation_from_col,
                translation_to_col,
                translation_factors_col,
            )

        elif (
            (translation_factors_col is not None)
            or (translation_from_col is not None)
            or (translation_to_col is not None)
        ):
            raise ValueError(
                "If translation_from_col,"
                " translation_to_col or translation_factors_col are provided,"
                " translation must also be given"
            )

    def trip_length_distribution(
        self,
        cost_matrix: pd.DataFrame,
        bins: list[int],
    ) -> None:
        """Calculate a distribution from the matrix passed on initilisation.

        Distribution is stored within the object which can be accessed using
        the `MatrixReport.distribution` property.

        Parameters
        ----------
        cost_matrix : pd.DataFrame
            Cost matrix corresponding with the inputted matrix.
        bins : list[int]
            Bins to use for the distribution.
        """
        try:
            cost_matrix.index = pd.to_numeric(cost_matrix.index, downcast="integer")  # type: ignore[call-overload]
            cost_matrix.columns = pd.to_numeric(cost_matrix.columns, downcast="integer")  # type: ignore[call-overload]
        except ValueError:
            pass
        cost_matrix = cost_matrix.loc[self._matrix.index, self._matrix.columns]  # type: ignore[index]
        self._distribution = cost_utils.CostDistribution.from_data(
            self._matrix.to_numpy(), cost_matrix, bin_edges=bins
        )

    def write_to_excel(
        self,
        writer: pd.ExcelWriter,
        label: Optional[str] = None,
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

        if len(sheet_prefix) >= 19:
            raise ValueError(
                "label cannot be over 30 characters as the sheets names will"
                " be truncated and will not be unique"
            )

        self.describe.to_excel(writer, sheet_name=f"{sheet_prefix}Summary")

        self.trip_ends.to_excel(writer, sheet_name=f"{sheet_prefix}Trip_Ends")

        if output_sector_matrix is True:
            if self.sector_matrix is not None:
                self.sector_matrix.to_excel(writer, sheet_name=f"{sheet_prefix}Matrix")
            else:
                warnings.warn(
                    "Cannot output sectorised matrix unless you pass the translation vector on init"
                )

        if self.distribution is not None:
            self.distribution.df.to_excel(writer, sheet_name=f"{sheet_prefix}Distribution")

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
        return self._translated_matrix

    @property
    def distribution(self) -> cost_utils.CostDistribution | None:
        """Distribution if `trip_length_distribution` has been called, otherwise none."""
        return self._distribution

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
        path: Path,
        *,
        translation_path: Optional[Path] = None,
        translation_from_col: Optional[str] = None,
        translation_to_col: Optional[str] = None,
        translation_factors_col: Optional[str] = None,
    ) -> MatrixReport:
        """Create an instance of MatrixReport from file paths.

        Parameters
        ----------
        path : Path
            Path to the matrix csv.
        translation_path : Optional[Path], optional
            Path to correspondence between matrix zoning and summary zoning, by default None
        translation_from_col : Optional[str], optional
            The column in the translation matrix with zoning to translate from, by default None.
        translation_to_col : Optional[str], optional
            The column in the translation matrix with zoning to translate to, by default None.
        translation_factors_col : Optional[str], optional
            The column in the translation matrix to use as factors, by default None.

        Returns
        -------
        MatrixReport
            Instance of MatrixReport created from the file paths.
        """
        matrix = pd.read_csv(path, index_col=0)

        if translation_path is not None:
            translation_factors = pd.read_csv(translation_path)
        else:
            translation_factors = None

        return cls(
            matrix,
            translation_factors=translation_factors,
            translation_from_col=translation_from_col,
            translation_to_col=translation_to_col,
            translation_factors_col=translation_factors_col,
        )


def matrix_describe(matrix: pd.DataFrame, almost_zero: Optional[float] = None) -> pd.Series:
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
        Matrix summary statistics, expands upon the standard pandas.Series.descibe.
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

    info = matrix.stack().describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
    assert isinstance(info, pd.Series)  # To stop MyPy whinging
    info["columns"] = len(matrix.columns)
    info["rows"] = len(matrix.index)
    info["sum"] = matrix.sum().sum()
    info["zeros"] = (matrix == 0).sum().sum()
    info["almost_zeros"] = (matrix < almost_zero).sum().sum()
    info["NaNs"] = matrix.isna().sum().sum()
    return info
