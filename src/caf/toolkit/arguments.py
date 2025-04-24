# -*- coding: utf-8 -*-
"""Functionality for handling command-line arguments."""

##### IMPORTS #####

from __future__ import annotations

# Built-Ins
import argparse
import logging
import os
import pathlib
import re
import warnings
from typing import Iterable

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
    "bool": bool,
    "filepath": pathlib.Path,
    "directorypath": pathlib.Path,
    "path": pathlib.Path,
}


##### CLASSES & FUNCTIONS #####


class TypeAnnotationWarning(UserWarning):
    """Warning for issues parsing type annotations for CLI arguments."""


def getenv_bool(name: str, default: bool) -> bool:
    """Get environment variable and converts to a boolean.

    Normalizes the variable value by converting to lowercase
    and stripping whitespace before bool conversion.

    Parameters
    ----------
    name : str
        Name of the environment variable.
    default : bool
        Default value for the environment variable.

    Returns
    -------
    bool
        True for: 'true', 'yes', 'y' or '1'.
        False for: 'false', 'no', 'n' or '0'.

    Raises
    ------
    ValueError
        If the environment variable has an unexpected value.
    """
    value = os.getenv(name, default)
    if isinstance(value, bool):
        return value

    true_str = ("true", "yes", "y", "1")
    false_str = ("false", "no", "n", "0")

    value = value.strip().lower()
    if value == "":
        return default
    if value in true_str:
        return True
    if value in false_str:
        return False

    raise ValueError(
        f"unexpected value ({value!r}) for '{name}' env "
        "variable should be one of the following:\n"
        f" For true: {true_str}\n For false: {false_str}"
    )


def _parse_types(type_str: str) -> tuple[type, bool]:
    types = set()
    optional = False
    for type_ in type_str.split("|"):
        match = re.match(r"(?:(\w+)\.)*(\w+)", type_.strip(), re.I)
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


def _replace_union(annotation: str) -> str:
    """Replace any 'Union[??]' annotations with '|'.

    Examples
    --------
    >>> _replace_union("union[int, str]")
    'int | str'

    >>> _replace_union("Union[   int  , str , float]")
    'int | str | float'

    >>> _replace_union("list[Union[float, int]]")
    'list[float | int]'

    >>> _replace_union("list[int]")
    'list[int]'

    >>> _replace_union("tuple[Union[int, float, str], Union[str, int]]")
    'tuple[int | float | str, str | int]'

    >>> _replace_union("tuple[int, Union[int, str]]")
    'tuple[int, int | str]'
    """
    annotation = re.sub(
        r"union\[(\s*(?:[\w\d.]+\s*,?\s*)+)\]",
        lambda x: " | ".join(i.strip() for i in x.group(1).split(",")),
        annotation,
        flags=re.IGNORECASE,
    )

    if "union" in annotation:
        return _replace_union(annotation)
    return annotation


def parse_arg_details(annotation: str) -> tuple[type, bool, int | str | None]:
    """Attempt to get argument type from annotation text.

    This only works with basic Python types (int, str, float, bool and Path)
    for any other types str will be returned.

    Parameters
    ----------
    annotation : str
        Type annotation to be parsed e.g. 'list[int | str]'. This can also
        handle the repr of a type e.g. "<class 'bool'>".

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
    annotation = _replace_union(annotation)

    match = re.match(r"^(?:(\w+)?\[|<class ')?([\w \t,.|]+)(?:\]|'>)?$", annotation.strip())
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
        """Check `model` type.

        Parameters
        ----------
        model : type
            Model used to define arguments, should be
            a subclass of `pydantic.BaseModel`.
        """
        if not issubclass(model, pydantic.BaseModel):  # type: ignore
            raise TypeError(f"`dataclass` should be a pydantic BaseModel not {type(model)}")

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

        Returns
        -------
        argparse.ArgumentParser
            Argument parser with arguments added and a default function
            set as the 'dataclass_parse_func' parameter which will parse
            the `argparse.Namespace` and output the given dataclass.
        """
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

            if type_ is not bool:
                parser.add_argument(
                    *name_or_flags, type=type_, help=description, default=default, nargs=nargs
                )
                continue

            # Use the store true / false action for boolean flags
            # Use false as default if not given
            if default is None or not default:
                action = "store_true"
            else:
                action = "store_false"
            parser.add_argument(*name_or_flags, action=action, help=description)

        parser.set_defaults(dataclass_parse_func=self._parse)
        return parser

    def add_config_arguments(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add config argument to command-line `parser` and adds `dataclass_parse_func`.

        Adds argument for reading parameters from a config and will automatically
        load the config file in the `dataclass_parse_func`. Will not work if
        class provided is not a subclass of `config_base.BaseConfig`.

        Returns
        -------
        argparse.ArgumentParser
            Argument parser with `config_path` argument added and a default
            function set as the 'dataclass_parse_func' parameter which will
            parse the `argparse.Namespace` and load the parameters from a
            config.

        Raises
        ------
        TypeError
            If the model class isn't a subclass of `BaseConfig`.
        """
        parser.add_argument(
            "config_path",
            type=pathlib.Path,
            help="path to YAML config file containing run parameters",
        )
        parser.set_defaults(dataclass_parse_func=self._config_parse)

        return parser

    def add_subcommands(
        self,
        subparsers: argparse._SubParsersAction,
        name: str,
        add_arguments: bool = True,
        add_config: bool = True,
        **kwargs,
    ):
        """Add sub-commands for CLI arguments and config (if possible).

        Note: config sub-command won't be added if model provided to class
        isn't a subclass of BaseConfig.

        Parameters
        ----------
        subparsers : argparse._SubParsersAction[ArgumentParser]
            Subparser to add new sub-commands to, created using
            `argparser.ArgumentParser().add_subparsers()`.
        name : str
            Name of the sub-command to add, if config sub-command is
            added it will be called '{name}-config'.
        add_arguments : bool
            If True (default) adds sub-command called '{name}' with arguments.
        add_config : bool
            If True (default), and model given is a :class:`BaseConfig`,
            adds sub-command called '{name}-config' with a single
            config path argument.
        kwargs
            Keyword arguments to pass to `subparsers.add_parser`.
        """
        if not add_arguments and not add_config:
            raise ValueError(
                "cannot add subcommands if add_arguments and add_config are both False."
            )

        parser = subparsers.add_parser(name, **kwargs)

        if add_arguments:
            self.add_arguments(parser)

        if self._config and add_config:
            kwargs.pop("help", None)
            parser = subparsers.add_parser(
                f"{name}-config", help=f"run {name} with parameters from config", **kwargs
            )
            self.add_config_arguments(parser)

    def _parse(self, args: argparse.Namespace) -> pydantic.BaseModel:
        """Parse `args` for command and return instance of class.

        This method will extract attributes from the `args` Namespace
        and use them to instantiate the class, then return the class
        instance.
        """
        assert issubclass(self._model, pydantic.BaseModel)

        data = {}
        for name in self._model.model_fields:
            data[name] = getattr(args, name)

        return self._model(**data)

    def _config_parse(self, args: argparse.Namespace) -> config_base.BaseConfig:
        """Load parameters from config file.

        Parameters
        ----------
        args : argparse.Namespace
            Parsed command-line arguments with a `config_path` attribute.
        """
        assert issubclass(self._model, config_base.BaseConfig)
        return self._model.load_yaml(args.config_path)  # type: ignore


class TidyUsageArgumentDefaultsHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """Format usage text to remove optional arguments and just have '[options]'."""

    def _format_usage(
        self,
        usage: str | None,
        actions: Iterable[argparse.Action],
        groups: Iterable[argparse._MutuallyExclusiveGroup],
        prefix: str | None,
    ) -> str:
        usage = super()._format_usage(usage, actions, groups, prefix)

        # Replace all optional arguments
        return re.sub(r"\[.*\]\s+", "[options] ", usage, flags=re.DOTALL)
