# -*- coding: utf-8 -*-
"""Functionality for handling command-line arguments."""

##### IMPORTS #####

from __future__ import annotations

# Built-Ins
import abc
import argparse
import collections
import dataclasses
import logging
import pathlib
import re
import warnings

# Third Party
import pydantic

##### CONSTANTS #####

LOG = logging.getLogger(__name__)
_ARG_TYPE_LOOKUP = collections.defaultdict(
    lambda: str,
    {
        "int": int,
        "str": str,
        "float": float,
        "filepath": pathlib.Path,
        "directorypath": pathlib.Path,
        "path": pathlib.Path,
    },
)

##### CLASSES & FUNCTIONS #####


def _get_arg_type(type_name: str) -> tuple[type, bool]:
    """Attempt to get argument type from annotation text.

    This only works with basic Python types (int, str, float and Path)
    for any other types (str, False) will be returned.
    """
    match = re.match(
        r"^(?:(\w+)?\[)?"  # Collection / typing e.g. list, dict, Union, Optional
        r"(\w+\.)?"  # Module name, can't handle multiple modules
        r"(\w+)"  # Type name(s) e.g. str, str
        r"\]?$",  # Optional closing bracket ']'
        type_name.strip(),
    )

    if match is None:
        warnings.warn(f"unexpected type format: '{type_name}'")
        return str, False

    prefix = match.group(1)
    type_ = match.group(3).lower()
    optional = prefix is not None and prefix.lower() == "optional"

    return _ARG_TYPE_LOOKUP[type_], optional


@pydantic.dataclasses.dataclass
class BaseArgs(abc.ABC):
    """Base class for defining command-line arguments and run functions."""

    # TODO(MB) Describe how this would be used, with examples

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add arguments to command-line `parser`.

        This method will add arguments to the `parser` with the same
        names as the class attributes. It will fill in the following:
        - default values, if defined in the attribute field function
        - help text, if defined in the field metadata dictionary with
          the "help" key
        - type if the attribute has a type annotation, only basic Python
          types are handled, any others will just be left as strings.
        - optional flag, if the field type annotation is optional or the
          field has a default value.

        For more complex argument requirements this method should be
        overwritten in the child class.
        """
        for field in dataclasses.fields(cls):
            type_, optional = _get_arg_type(str(field.type))

            if optional or (
                field.default is not None and field.default != dataclasses.MISSING
            ):
                name_or_flags = [f"--{field.name}"]
            else:
                name_or_flags = [field.name]

            parser.add_argument(
                *name_or_flags,
                type=type_,
                help=field.metadata.get("help", str(field.type)),
                default=field.default,
            )

        return parser

    @classmethod
    def parse(cls, args: argparse.Namespace) -> BaseArgs:
        """Parse `args` for command and return instance of class.

        This method will extract attributes from the `args` Namespace
        and use them to instantiate the class, then return the class
        instance.
        """
        data = {}
        for field in dataclasses.fields(cls):
            data[field.name] = getattr(args, field.name)

        return cls(**data)

    @abc.abstractmethod
    def run(self) -> None:
        """Run functionality for sub-command."""
        raise NotImplementedError("to be implemented by subclass")
