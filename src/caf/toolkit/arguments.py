# -*- coding: utf-8 -*-
"""Functionality for handling command-line arguments."""

##### IMPORTS #####

from __future__ import annotations

# Built-Ins
import argparse
import dataclasses
import logging
import pathlib
import re
import warnings

# Third Party
import pydantic
import pydantic_core

# Local Imports
from caf.toolkit import config_base

##### CONSTANTS #####

LOG = logging.getLogger(__name__)
# Dictionary is ordered from most generic (str) to least generic
_ARG_TYPE_LOOKUP: dict[str, type] = {
    "str": str,
    "float": float,
    "int": int,
    "filepath": pathlib.Path,
    "directorypath": pathlib.Path,
    "path": pathlib.Path,
}


##### CLASSES & FUNCTIONS #####


class TypeAnnotationWarning(UserWarning):
    """Warning for issues parsing type annotations for CLI arguments."""


def _parse_types(type_str: str) -> tuple[type, bool]:
    types = set()
    optional = False
    for type_ in type_str.split("|"):
        match = re.match(r"(?:(\w+)\.)?(\w+)", type_.strip(), re.I)
        if match is None:
            warnings.warn(f"unexpect type format: '{type_}'", TypeAnnotationWarning)
            return str, optional

        value = match.group(2).strip().lower()

        if value == "none":
            optional = True
        else:
            types.add(value)

    for name, type_ in _ARG_TYPE_LOOKUP.items():
        if name in types:
            return type_, optional

    warnings.warn(f"unexpected types: {types}", TypeAnnotationWarning)

    return str, optional


def parse_arg_details(annotation: str) -> tuple[type, bool, int | str | None]:
    """Attempt to get argument type from annotation text.

    This only works with basic Python types (int, str, float and Path)
    for any other types (str, False) will be returned.

    Parameters
    ----------
    annotation : str
        Type annotation to be parsed e.g. 'list[int | str]'.

    Returns
    -------
    type
        Argument type, will use the most generic type if multiple options
        are given e.g. 'int | str | float' will return str.
    bool
        True if the type annotation is optional e.g. 'Optional[str]' or 'str | None'.
        Otherwise False.
    int | str | None
        Number of arguments expected as required by `nargs` parameter in
        `argparse.ArgumentParser.add_argument`. This will return an int
        if the type is a tuple, '*' for lists and None otherwise.
    """
    match = re.match(r"^(?:(\w+)?\[)?([\w \t,.|]+)\]?$", annotation.strip())
    if match is None:
        warnings.warn(
            f"unexpected type annotation format: '{annotation}'", TypeAnnotationWarning
        )
        return str, False, None

    prefix = match.group(1)
    type_annotation = match.group(2)
    if prefix is None:
        return *_parse_types(type_annotation), None

    prefix = prefix.strip().lower()
    nargs: int | str | None = None

    if prefix == "optional":
        type_, _ = _parse_types(type_annotation)
        optional = True

    elif prefix == "union":
        type_, optional = _parse_types(type_annotation.replace(",", "|"))

    elif prefix == "tuple":
        # Can only determine single type for all arguments in tuple so treat the
        # separate tuple types as options and have parse types pick the most generic
        type_, optional = _parse_types(type_annotation.replace(",", "|"))
        nargs = len(type_annotation.split(","))

    elif prefix == "list":
        type_, optional = _parse_types(type_annotation)
        nargs = "*"

    else:
        warnings.warn(f"unexpected type annotation prefix: '{prefix}'", TypeAnnotationWarning)
        type_, optional = _parse_types(type_annotation)

    return type_, optional, nargs


class ModelArguments:
    """Base class for defining command-line arguments from `pydantic.BaseModel`."""

    # TODO(MB) Describe how this would be used, with examples

    def __init__(self, model: type) -> None:
        if not issubclass(model, pydantic.BaseModel):  # type: ignore
            raise TypeError(f"`dataclass` should be a pydantic dataclass not {type(model)}")

        self._model = model
        self._config = issubclass(model, config_base.BaseConfig)

    def add_arguments(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add arguments to command-line `parser` and adds `dataclass_parse_func`.

        This method will add arguments to the `parser` with the same
        names as the class attributes. It will fill in the following:
        - default values, if defined in the attribute field function
        - help text, if defined in the field metadata dictionary with
          the "help" key
        - type if the attribute has a type annotation, only basic Python
          types are handled, any others will just be left as strings.
        - optional flag, if the field type annotation is optional or the
          field has a default value.

        For more complex argument requirements these should be added manually.

        If a subclass of `config_base.BaseConfig` is provided then an optional
        sub-command for loading data from a config file will be included, this
        will allow arguments to be provided or to have them loaded from a config
        file but not both.

        Returns
        -------
        argparse.ArgumentParser
            Argument parser with arguments added and a default function
            set as the 'dataclass_parse_func' parameter which will parse
            the `argparse.Namespace` and output the given dataclass.
        """
        if self._config:
            subparsers = parser.add_subparsers(title="Sub-commands")
            config_parser = subparsers.add_parser(
                "config",
                prog=f"{parser.prog} config",
                description="Load parameters from config file instead of from arguments",
                help="load parameters from a config file",
            )
            config_parser.add_argument(
                "config_path",
                type=pathlib.Path,
                help="path to YAML config file containing run parameters",
            )
            config_parser.set_defaults(dataclass_parse_func=self._config_parse)

        else:
            parser.set_defaults(dataclass_parse_func=self._parse)

        for name, field in self._model.model_fields.items():
            type_, _, nargs = parse_arg_details(str(field.annotation))

            if field.is_required():
                name_or_flags = [name]
            else:
                name_or_flags = [f"--{name}"]

            if field.default == pydantic_core.PydanticUndefined:
                default = None
            else:
                default = field.default

            if field.description is None:
                description = str(field.annotation)
            else:
                description = field.description

            parser.add_argument(
                *name_or_flags, type=type_, help=description, default=default, nargs=nargs
            )

        return parser

    def _parse(self, args: argparse.Namespace) -> pydantic.dataclasses.PydanticDataclass:
        """Parse `args` for command and return instance of class.

        This method will extract attributes from the `args` Namespace
        and use them to instantiate the class, then return the class
        instance.
        """
        assert pydantic.dataclasses.is_pydantic_dataclass(self._model)  # type: ignore

        data = {}
        for field in dataclasses.fields(self._model):
            data[field.name] = getattr(args, field.name)

        return self._model(**data)

    def _config_parse(self, args: argparse.Namespace) -> config_base.BaseConfig:
        """Load parameters from config file.

        Parameters
        ----------
        args : argparse.Namespace
            Parsed command-line arguments with a `config_path` attribute.
        """
        assert issubclass(self._model, config_base.BaseConfig)
        return self._model.load_yaml(args.config_path)
