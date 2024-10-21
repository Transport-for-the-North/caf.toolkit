# -*- coding: utf-8 -*-
"""Front-end module for running toolkit functionality from command-line."""

##### IMPORTS #####

from __future__ import annotations

# Built-Ins
import argparse
import logging
import pathlib
import sys
import warnings
from typing import Union

# Third Party
import pydantic

# Local Imports
import caf.toolkit as ctk
from caf.toolkit import arguments, config_base, log_helpers, translation

##### CONSTANTS #####

LOG = logging.getLogger(__name__)
_TRACEBACK = arguments.getenv_bool("TOOLKIT_TRACEBACK", False)

##### CLASSES & FUNCTIONS #####


class _BaseTranslationArgs(config_base.BaseConfig):
    """Base class for arguments which are the same for matrix and vector translation."""

    data_file: pydantic.FilePath = pydantic.Field(
        description="CSV file containing data to be translated"
    )
    translation_file: pydantic.FilePath = pydantic.Field(
        description="CSV file defining how to translate and the weightings to use"
    )
    output_file: pathlib.Path = pydantic.Field(
        default=pathlib.Path("translated.csv"),
        description="Location to save the translated output",
    )
    from_column: Union[int, str] = pydantic.Field(
        default=0,
        description="The column (name or position) in the translation"
        " file containing the zone ids to translate from",
    )
    to_column: Union[int, str] = pydantic.Field(
        default=1,
        description="The column (name or position) in the translation"
        " file containing the zone ids to translate to",
    )
    factor_column: Union[int, str] = pydantic.Field(
        default=2,
        description="The column (name or position) in the translation"
        " file containing the weightings between from and to zones",
    )

    @pydantic.model_validator(mode="after")
    def check_translation_column_names(self) -> _BaseTranslationArgs:
        """Check all columns for translation are either str or int, not both.

        Pandas doesn't allow usecols to contain a mix of str and int.
        """
        columns = [self.from_column, self.to_column, self.factor_column]

        # pylint: disable=unidiomatic-typecheck
        if any(type(i) != type(columns[0]) for i in columns):
            raise TypeError(
                "from_column, to_column and factor_column should all be either"
                " a name or position, there cannot be a mix of names and"
                " positions. The default values for all three are positions"
                " so if one columnname is given then the other 2 columns"
                " names are also required."
            )

        return self


class TranslationArgs(_BaseTranslationArgs):
    """Command-line arguments for vector zone translation."""

    zone_column: Union[int, str] = pydantic.Field(
        default=0,
        description="The column (name or position) in the data file containing the zone ids",
    )

    def run(self):
        """Run vector zone translation with the given arguments."""
        translation.vector_translation_from_file(
            vector_path=self.data_file,
            translation_path=self.translation_file,
            output_path=self.output_file,
            vector_zone_column=self.zone_column,
            translation_from_column=self.from_column,
            translation_to_column=self.to_column,
            translation_factors_column=self.factor_column,
        )


class MatrixTranslationArgs(_BaseTranslationArgs):
    """Command-line arguments for matrix zone translation."""

    zone_column: tuple[Union[int, str], Union[int, str]] = pydantic.Field(
        default=(0, 1),
        description="The 2 columns (name or position) in"
        " the matrix file containing the zone ids",
    )
    value_column: Union[int, str] = pydantic.Field(
        default=2,
        description="The column (name or position) in the"
        " CSV file containing the matrix values",
    )

    def run(self):
        """Run matrix zone translation with the given arguments."""
        translation.matrix_translation_from_file(
            matrix_path=self.data_file,
            translation_path=self.translation_file,
            output_path=self.output_file,
            matrix_zone_columns=self.zone_column,
            matrix_values_column=self.value_column,
            translation_from_column=self.from_column,
            translation_to_column=self.to_column,
            translation_factors_column=self.factor_column,
        )

    @pydantic.model_validator(mode="after")
    def check_matrix_column_names(self) -> MatrixTranslationArgs:
        """Check all columns for matrix are either str or int, not both.

        Pandas doesn't allow usecols to contain a mix of str and int.
        """
        columns = [*self.zone_column, self.value_column]

        # pylint: disable=unidiomatic-typecheck
        if any(type(i) != type(columns[0]) for i in columns):
            raise TypeError(
                "zone_columns and value_column should all be either"
                " a name or position, there cannot be a mix of names and"
                " positions. The default values for all three are positions"
                " so if one columnname is given then the other 2 columns"
                " names are also required."
            )

        return self


def parse_args() -> TranslationArgs | MatrixTranslationArgs:
    """Parse and validate command-line arguments."""
    parser = _create_arg_parser()

    # Print help if no arguments are given
    args = parser.parse_args(None if len(sys.argv[1:]) > 0 else ["-h"])

    try:
        params = args.dataclass_parse_func(args)
    except (pydantic.ValidationError, FileNotFoundError) as exc:
        if _TRACEBACK:
            raise
        # Switch to raising SystemExit as this doesn't include traceback
        raise SystemExit(str(exc)) from exc

    return params


def _create_arg_parser():
    parser = argparse.ArgumentParser(
        __package__,
        description=ctk.__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-v",
        "--version",
        help="show caf.toolkit version and exit",
        action="version",
        version=f"{__package__} {ctk.__version__}",
    )

    subparsers = parser.add_subparsers(
        title="CAF Toolkit sub-commands",
        description="List of all available sub-commands",
    )

    translation_class = arguments.ModelArguments(TranslationArgs)
    translation_class.add_subcommands(
        subparsers,
        "translate",
        help="translate data file to a new zoning system",
        description="Translate data file to a new zoning "
        "system, using given translation lookup file",
        formatter_class=arguments.TidyUsageArgumentDefaultsHelpFormatter,
    )

    matrix_class = arguments.ModelArguments(MatrixTranslationArgs)
    matrix_class.add_subcommands(
        subparsers,
        "matrix-translate",
        help="translate a matrix file to a new zoning system",
        description="Translate a matrix file to a new zoning system, using"
        " given translation lookup file. Matrix CSV file should be in the"
        " long format i.e. 3 columns.",
        formatter_class=arguments.TidyUsageArgumentDefaultsHelpFormatter,
    )

    return parser


def main():
    """Parser command-line arguments and run CAF.toolkit functionality."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=arguments.TypeAnnotationWarning)
        parameters = parse_args()

    log_file = parameters.output_file.parent / "caf_toolkit.log"
    details = log_helpers.ToolDetails(
        __package__, ctk.__version__, homepage=ctk.__homepage__, source_url=ctk.__source_url__
    )

    with log_helpers.LogHelper(__package__, details, log_file=log_file):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "once",
                message=r".*column positions are given instead of names.*",
                category=UserWarning,
            )

            try:
                parameters.run()
            except Exception as exc:
                if _TRACEBACK:
                    raise
                # Switch to raising SystemExit as this doesn't include traceback
                raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
