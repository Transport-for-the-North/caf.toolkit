# -*- coding: utf-8 -*-
"""Base config class for storing and reading parameters for any NorMITs demand script."""
from __future__ import annotations

# Built-Ins
import datetime as dt
import textwrap
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Optional, overload

# Third Party
import pydantic
import pydantic_core
import strictyaml

# # # CONSTANTS # # #


# # # CLASSES # # #


class BaseConfig(pydantic.BaseModel):
    r"""Base class for storing model parameters.

    Contains functionality for reading / writing parameters to
    config files in the YAML format.

    See Also
    --------
    [pydantic docs](https://pydantic-docs.helpmanual.io/):
        for more information about using pydantic's model classes.
    `pydantic.BaseModel`: which handles converting data to Python types.
    `pydantic.validator`: which allows additional custom validation methods.

    Examples
    --------
    Example of creating a config class and initialising it with
    values, values will be validated and converted to correct
    type on initialisation.

    >>> from pathlib import Path
    >>> from caf.toolkit import BaseConfig
    >>> class ExampleParameters(BaseConfig):
    ...    import_folder: Path
    ...    name: str
    ...    some_option: bool = True
    >>> parameters = ExampleParameters(
    ...    import_folder="Test Folder",
    ...    name="Test",
    ...    some_option=False,
    ... )

    Example of instance of class after initialisation, the path differs
    depending on operating system.

    >>> parameters # doctest: +SKIP
    ExampleParameters(
        import_folder=WindowsPath('Test Folder'),
        name='Test',
        some_option=False,
    )

    Config class can be converted to YAML or saved with `save_yaml`.

    >>> print(parameters.to_yaml())
    import_folder: Test Folder
    name: Test
    some_option: no

    Config class data can be loaded from a YAML config file using `load_yaml`.

    >>> yaml_text = '''
    ... import_folder: Test Folder
    ... name: Test
    ... some_option: no
    ... '''
    >>> loaded_parameters = ExampleParameters.from_yaml(yaml_text)
    >>> loaded_parameters == parameters
    True
    """

    @classmethod
    def from_yaml(cls, text: str):
        """Parse class attributes from YAML `text`.

        Parameters
        ----------
        text: str
            YAML formatted string, with parameters for
            the class attributes.

        Returns
        -------
        Instance of self
            Instance of class with attributes filled in from
            the YAML data.
        """
        data = strictyaml.load(text).data
        return cls.model_validate(data)

    @classmethod
    def load_yaml(cls, path: Path):
        """Read YAML file and load the data using `from_yaml`.

        Parameters
        ----------
        path: Path
            Path to YAML file containing parameters.

        Returns
        -------
        Instance of self
            Instance of class with attributes filled in from
            the YAML data.
        """
        # pylint: disable = unspecified-encoding
        with open(path, "rt") as file:
            text = file.read()
        return cls.from_yaml(text)

    def to_yaml(self) -> str:
        """Convert attributes from self to YAML string.

        Returns
        -------
        str
            YAML formatted string with the data from
            the class attributes.
        """
        data = self.model_dump(mode="json", exclude_unset=True, exclude_none=True)

        # Strictyaml cannot handle None so excluding from output
        data = _remove_none_dict(data)

        return strictyaml.as_document(data).as_yaml()

    def save_yaml(
        self,
        path: Path,
        datetime_comment: bool = True,
        other_comment: Optional[str] = None,
        format_comment: bool = False,
    ) -> None:
        """Write data from self to a YAML file.

        Parameters
        ----------
        path : Path
            Path to YAML file to output.
        datetime_comment : bool, default True
            Whether to include a comment at the top of
            the config file with the current date and time.
        other_comment : str, optional
            Additional comments to add to the top of the
            config file, "#" will be added to the start of
            each new line if it isn't already there.
        format_comment : bool, default False
            Whether to remove newlines from `other_comment` and
            format lines to a specific character length.
        """
        write_config(
            self.to_yaml(),
            path=path,
            datetime_comment=datetime_comment,
            name=self.__class__.__name__,
            other_comment=other_comment,
            format_comment=format_comment,
        )

    @classmethod
    def write_example(
        cls, path_: Path, /, comment_: Optional[str] = None, **examples: str
    ) -> None:
        """Write examples to a config file.

        Parameters
        ----------
        path_ : Path
            Path to the YAML file to write.
        comment_ : str, optional
            Comment to add to the top of the example config file,
            will be formatted to add "#" symbols and split across
            multiple lines.
        examples : str
            Fields of the config to write, any missing fields
            are filled in with their default value (if they have
            one) or 'REQUIRED' / 'OPTIONAL'.
        """
        data = {}
        for name, field in cls.model_fields.items():
            if field.default is not None and field.default != pydantic_core.PydanticUndefined:
                value = field.default
            else:
                value = "REQUIRED" if field.is_required() else "OPTIONAL"

            data[name] = examples.get(name, value)

            if is_dataclass(data[name]):
                data[name] = asdict(data[name])

            if isinstance(data[name], pydantic.BaseModel):
                data[name] = data[name].model_dump(
                    mode="json", exclude_unset=True, exclude_none=True
                )

        yaml = strictyaml.as_document(data).as_yaml()
        write_config(
            yaml,
            path_,
            datetime_comment=False,
            name=f"Example {cls.__name__}",
            other_comment=comment_,
            format_comment=True,
        )


# # # FUNCTIONS # # #
def _is_collection(obj: Any) -> bool:
    """
    Check if an object is any type of non-dict collection.

    Currently only checks for list, tuple or set,
    """
    return isinstance(obj, (list, tuple, set, dict))


@overload
def _remove_none_collection(data: list) -> list: ...  # pragma: no cover


@overload
def _remove_none_collection(data: set) -> set: ...  # pragma: no cover


@overload
def _remove_none_collection(data: tuple) -> tuple: ...  # pragma: no cover


def _remove_none_collection(data: list | set | tuple) -> list | set | tuple | None:
    """Remove items recursively from collections which are None."""
    filtered = []
    if len(data) == 0:
        return None
    for item in data:
        # Skip the None item so it's not included
        if item is None:
            continue

        # Clean and keep any other items
        if isinstance(item, dict):
            item = _remove_none_dict(item)
        elif _is_collection(item):
            item = _remove_none_collection(item)
        filtered.append(item)

    # return same type as input
    return type(data)(filtered)


def _remove_none_dict(data: dict) -> dict | None:
    """Remove items recursively from dictionary which are None."""
    filtered = {}
    if len(data) == 0:
        return None
    for key, value in data.items():
        if value is None:
            continue

        if isinstance(value, dict):
            value = _remove_none_dict(value)

        elif _is_collection(value):
            value = _remove_none_collection(value)

        if value is None:
            continue

        filtered[key] = value

    return filtered


def write_config(
    yaml: str,
    path: Path,
    datetime_comment: bool = True,
    name: Optional[str] = None,
    other_comment: Optional[str] = None,
    format_comment: bool = False,
) -> None:
    """Write data from self to a YAML file.

    Parameters
    ----------
    yaml : str
        YAML formatted text to write to config file.
    path : Path
        Path to YAML file to output.
    datetime_comment : bool, default True
        Whether to include a comment at the top of
        the config file with the current date and time.
    name : str
        Name of the type of config being written,
        used for datetime comments.
    other_comment : str, optional
        Additional comments to add to the top of the
        config file, "#" will be added to the start of
        each new line if it isn't already there.
    format_comment : bool, default False
        Whether to remove newlines from `other_comment` and
        format lines to a specific character length.
    """
    if other_comment is None or other_comment.strip() == "":
        comment_lines = []
    elif format_comment:
        comment_lines = textwrap.wrap(other_comment)
    else:
        comment_lines = other_comment.split("\n")

    if datetime_comment:
        if name is not None:
            name = f"{name} config"
        else:
            name = "Config"

        comment_lines.insert(0, f"{name} written on {dt.datetime.now():%Y-%m-%d at %H:%M}")

    if len(comment_lines) > 0:
        comment_lines = [i if i.startswith("#") else f"# {i}" for i in comment_lines]
        yaml = "\n".join(comment_lines + [yaml])

    # pylint: disable = unspecified-encoding
    with open(path, "wt") as file:
        file.write(yaml)
