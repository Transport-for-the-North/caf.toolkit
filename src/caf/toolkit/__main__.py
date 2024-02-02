# -*- coding: utf-8 -*-
"""Front-end module for running toolkit functionality from command-line."""

##### IMPORTS #####

from __future__ import annotations

# Built-Ins
import argparse
import dataclasses
import logging
import pathlib
import sys
import warnings

# Third Party
import pydantic

# Local Imports
import caf.toolkit as ctk
from caf.toolkit import arguments, log_helpers, translation

##### CONSTANTS #####

LOG = logging.getLogger(__name__)


##### CLASSES & FUNCTIONS #####


@pydantic.dataclasses.dataclass
class _BaseTranslationArgs(arguments.BaseArgs):
    """Base class for arguments which are the same for matrix and vector translation."""

    data_file: pydantic.FilePath = dataclasses.field(
        metadata={"help": "CSV file containing data to be translated"}
    )
    translation_file: pydantic.FilePath = dataclasses.field(
        metadata={"help": "CSV file defining how to translate and the weightings to use"}
    )
    output_file: pathlib.Path = dataclasses.field(
        default=pathlib.Path("translated.csv"),
        metadata={"help": "Location to save the translated output"},
    )
    from_column: int | str = dataclasses.field(
        default=0,
        metadata={
            "help": "The column (name or position) in the translation"
            " file containing the zone ids to translate from"
        },
    )
    to_column: int | str = dataclasses.field(
        default=1,
        metadata={
            "help": "The column (name or position) in the translation"
            " file containing the zone ids to translate to"
        },
    )
    factor_column: int | str = dataclasses.field(
        default=2,
        metadata={
            "help": "The column (name or position) in the translation"
            " file containing the weightings between from and to zones"
        },
    )


@pydantic.dataclasses.dataclass
class TranslationArgs(_BaseTranslationArgs):
    """Command-line arguments for vector zone translation."""

    zone_column: int | str = dataclasses.field(
        default=0,
        metadata={
            "help": "The column (name or position) in the data file containing the zone ids"
        },
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


@pydantic.dataclasses.dataclass
class MatrixTranslationArgs(_BaseTranslationArgs):
    """Command-line arguments for matrix zone translation."""

    zone_column: tuple[int | str, int | str] = dataclasses.field(
        default=(0, 1),
        metadata={
            "help": "The 2 columns (name or position) in"
            " the matrix file containing the zone ids"
        },
    )
    value_column: int | str = dataclasses.field(
        default=2,
        metadata={
            "help": "The column (name or position) in the"
            " CSV file containing the matrix values"
        },
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


def parse_args() -> TranslationArgs | MatrixTranslationArgs:
    """Parse and validate command-line arguments."""
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

    translate_parser = subparsers.add_parser(
        "translate",
        usage="caf.toolkit translate data_file translation_file [options]",
        help="translate data file to a new zoning system",
        description="Translate data file to a new zoning "
        "system, using given translation lookup file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    translate_parser = TranslationArgs.add_arguments(translate_parser)
    translate_parser.set_defaults(func=TranslationArgs.parse)

    matrix_parser = subparsers.add_parser(
        "matrix_translate",
        usage="caf.toolkit matrix_translate data_file translation_file [options]",
        help="translate a matrix file to a new zoning system",
        description="Translate a matrix file to a new zoning system, using"
        " given translation lookup file. Matrix CSV file should be in the"
        " long format i.e. 3 columns.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    matrix_parser = MatrixTranslationArgs.add_arguments(matrix_parser)
    matrix_parser.set_defaults(func=MatrixTranslationArgs.parse)

    # Print help if no arguments are given
    args = parser.parse_args(None if len(sys.argv[1:]) > 0 else ["-h"])

    return args.func(args)


def main():
    """Parser command-line arguments and run CAF.toolkit functionality."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "unexpected type format", UserWarning)
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
            parameters.run()


if __name__ == "__main__":
    main()
