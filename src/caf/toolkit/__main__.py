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

# Third Party
import pydantic

# Local Imports
import caf.toolkit as ctk
from caf.toolkit import arguments, log_helpers

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
    output_file: str = dataclasses.field(
        default="translated.csv",
        metadata={"help": "file name to save translated output to"},
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
        print(self)
        raise NotImplementedError("WIP!")


@pydantic.dataclasses.dataclass
class MatrixTranslationArgs(_BaseTranslationArgs):
    """Command-line arguments for matrix zone translation."""

    zone_column: tuple[str, str] = dataclasses.field(
        default=("origin_id", "destination_id"),
        metadata={"help": "name of 2 columns containing zone IDs for matrix"},
    )

    def run(self):
        """Run matrix zone translation with the given arguments."""
        print(self)
        raise NotImplementedError("WIP!")


@pydantic.dataclasses.dataclass
class Args:
    """Class for managing (and parsing) command-line arguments for CAF.toolkit."""

    output_folder: pathlib.Path
    parameters: TranslationArgs | MatrixTranslationArgs

    @classmethod
    def parse_args(cls) -> Args:
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
        parser.add_argument(
            "-o",
            "--output-folder",
            help="folder to save any outputs in, including log files",
            type=pathlib.Path,
            default=pathlib.Path().resolve(),
        )

        subparsers = parser.add_subparsers(
            title="Toolkit sub-commands",
            description="List of all available sub-commands",
        )

        translate_parser = subparsers.add_parser(
            "translate",
            usage="translate data_file translation_file [options]",
            help="translate data file to a new zoning system",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        translate_parser = TranslationArgs.add_arguments(translate_parser)
        translate_parser.set_defaults(func=TranslationArgs.parse)

        matrix_parser = subparsers.add_parser(
            "matrix_translate",
            help="caf.toolkit translate a matrix file to a new zoning system",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        matrix_parser = MatrixTranslationArgs.add_arguments(matrix_parser)
        matrix_parser.set_defaults(func=MatrixTranslationArgs.parse)

        # Print help if no arguments are given
        args = parser.parse_args(None if len(sys.argv[1:]) > 0 else ["-h"])

        return cls(output_folder=args.output_folder, parameters=args.func(args))


def main():
    """Parser command-line arguments and run CAF.toolkit functionality."""
    args = Args.parse_args()

    details = log_helpers.ToolDetails(
        __package__, ctk.__version__, homepage=ctk.__homepage__, source_url=ctk.__source_url__
    )

    with log_helpers.LogHelper(
        __package__, details, log_file=args.output_folder / "caf_toolkit.log"
    ):
        args.parameters.run()


if __name__ == "__main__":
    main()
