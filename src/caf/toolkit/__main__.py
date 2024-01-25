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
        metadata={"help": "CSV file containing translation lookup"}
    )
    output_file: pathlib.Path = dataclasses.field(
        default=pathlib.Path("translated.csv"),
        metadata={"help": "file to save translated output to"},
    )
    from_column: str = dataclasses.field(
        default="from_id",
        metadata={"help": "name of column in translation containing from zone id"},
    )
    to_column: str = dataclasses.field(
        default="to_id",
        metadata={"help": "name of column in translation containing to zone id"},
    )
    factor_column: str = dataclasses.field(
        default="split_factor",
        metadata={"help": "name of column in translation containing split factors"},
    )


@pydantic.dataclasses.dataclass
class TranslationArgs(_BaseTranslationArgs):
    """Command-line arguments for vector zone translation."""

    zone_column: str = dataclasses.field(
        default="zone_id", metadata={"help": "name of column containing zone ID"}
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

    zone_column: tuple[str, str] = dataclasses.field(
        default=("origin_id", "destination_id"),
        metadata={"help": "name of 2 columns containing zone IDs for matrix"},
    )
    value_column: str = dataclasses.field(
        default="values",
        metadata={"help": "name of column in the CSV containing the matrix values"},
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
        description="translate data file to a new zoning "
        "system, using given translation lookup file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    translate_parser = TranslationArgs.add_arguments(translate_parser)
    translate_parser.set_defaults(func=TranslationArgs.parse)

    matrix_parser = subparsers.add_parser(
        "matrix_translate",
        usage="caf.toolkit matrix_translate data_file translation_file [options]",
        help="translate a matrix file to a new zoning system",
        description="translate a matrix file to a new zoning "
        "system, using given translation lookup file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "unexpected type format", UserWarning)
        matrix_parser = MatrixTranslationArgs.add_arguments(matrix_parser)
    matrix_parser.set_defaults(func=MatrixTranslationArgs.parse)

    # Print help if no arguments are given
    args = parser.parse_args(None if len(sys.argv[1:]) > 0 else ["-h"])

    return args.func(args)


def main():
    """Parser command-line arguments and run CAF.toolkit functionality."""
    parameters = parse_args()
    output_folder = parameters.output_file.parent

    details = log_helpers.ToolDetails(
        __package__, ctk.__version__, homepage=ctk.__homepage__, source_url=ctk.__source_url__
    )

    with log_helpers.LogHelper(
        __package__, details, log_file=output_folder / "caf_toolkit.log"
    ):
        parameters.run()


if __name__ == "__main__":
    main()
