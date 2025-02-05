"""Contains functions that perform checks and provide high level statistics."""

from __future__ import annotations

# Built-Ins
from pathlib import Path
from typing import Optional

# Third Party
import pandas as pd

# Local Imports
from caf.toolkit import translation  # pylint: disable=redefined-outer-name


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

    Functions
    ---------
    `from_file` : Create a MatrixReport from a file.
    `write_to_excel` : Write the report to an Excel file.
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

        self.describe = pd.DataFrame()

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

            self.describe["Original_Matrix"] = matrix_describe(matrix)

            translated_describe_label = "Translated_Matrix"

            matrix = translation.pandas_matrix_zone_translation(
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
        else:
            translated_describe_label = "Matrix"

        self.matrix = matrix
        self.describe[translated_describe_label] = matrix_describe(matrix)

    def write_to_excel(
        self, writer: pd.ExcelWriter, label: Optional[str] = None, output_matrix: bool = False
    ) -> None:
        """Writes the report to an Excel file

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

        if len(sheet_prefix) >= 31:
            raise ValueError(
                "label cannot be over 30 characters as the sheets names will"
                " be truncated and will not be unique"
            )

        self.describe.to_excel(writer, sheet_name=f"{sheet_prefix}Summary")

        self.trip_ends.to_excel(writer, sheet_name=f"{sheet_prefix}Trip_Ends")

        if output_matrix is True:
            self.matrix.to_excel(writer, sheet_name=f"{sheet_prefix}Matrix")

    @property
    def trip_ends(self) -> pd.DataFrame:
        """The row and column sums of the matrix."""
        return pd.DataFrame({"row_sums": self.row_sum, "col_sums": self.column_sum})

    @property
    def row_sum(self) -> pd.DataFrame:
        """The row sums of the matrix."""
        return self.matrix.sum(axis=0)

    @property
    def column_sum(self) -> pd.DataFrame:
        """The column sums of the matrix."""
        return self.matrix.sum(axis=1)

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
    """Creates a high level summary of a matrix.

    Parameters
    ----------
    matrix : pd.DataFrame
        Matrix to be summarised.
    almost_zero : Optional[float], optional
        Below this value cells will be defined as almost zero.
        If None almost zero will be calculated as = 1 / (# of cells in the matrix), by default None

    Returns
    -------
    pd.Series
        Matrix summary statistics.
    """
    if almost_zero is None:
        almost_zero = 1 / matrix.size

    info = matrix.stack().describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])

    info["columns"] = len(matrix.columns)
    info["rows"] = len(matrix.index)
    info["sum"] = matrix.sum().sum()
    info["zeros"] = (matrix == 0).sum().sum()
    info["almost_zeros"] = (matrix < almost_zero).sum().sum()
    info["NaNs"] = matrix.isna().sum().sum()
    return info
