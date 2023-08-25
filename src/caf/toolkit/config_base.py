# -*- coding: utf-8 -*-
"""Base config class for storing and reading parameters for any NorMITs demand script."""

# # # IMPORTS # # #
import datetime as dt
import json
from pathlib import Path
import textwrap
from typing import Optional

# pylint: disable=import-error
import pydantic
import strictyaml

# pylint: enable=import-error
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
        return cls.parse_obj(data)

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
        # Use pydantic to convert all types to json compatible,
        # then convert this back to a dictionary to dump to YAML
        json_dict = json.loads(self.json())

        # Strictyaml cannot handle None so excluding from output
        json_dict = _remove_none_dict(json_dict)

        return strictyaml.as_document(json_dict).as_yaml()

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
        path: Path
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
        if other_comment is None or other_comment.strip() == "":
            comment_lines = []
        elif format_comment:
            comment_lines = textwrap.wrap(other_comment)
        else:
            comment_lines = other_comment.split("\n")

        if datetime_comment:
            comment_lines.insert(
                0,
                f"{self.__class__.__name__} config written "
                f"on {dt.datetime.now():%Y-%m-%d at %H:%M}",
            )

        yaml = self.to_yaml()

        if len(comment_lines) > 0:
            comment_lines = [i if i.startswith("#") else f"# {i}" for i in comment_lines]
            yaml = "\n".join(comment_lines + [yaml])

        # pylint: disable = unspecified-encoding
        with open(path, "wt") as file:
            file.write(yaml)

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
        for name, field in cls.__fields__.items():
            if field.default is not None:
                value = field.default
            else:
                value = "REQUIRED" if field.required else "OPTIONAL"

            data[name] = examples.get(name, value)

        example = cls.construct(_fields_set=None, **data)
        example.save_yaml(
            path_,
            datetime_comment=False,
            other_comment=comment_,
            format_comment=True,
        )


# # # FUNCTIONS # # #
def _remove_none_dict(data: dict) -> dict:
    """Remove items recursively from dictionary which are None."""
    filtered = {}

    for key, value in data.items():
        if value is None:
            continue

        if isinstance(value, dict):
            value = _remove_none_dict(value)

        filtered[key] = value

    return filtered
