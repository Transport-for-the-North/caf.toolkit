# -*- coding: utf-8 -*-
"""Base config class for storing and reading parameters for any NorMITs demand script."""

# # # IMPORTS # # #
import json
from pathlib import Path

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
    >>> parameters
    ExampleParameters(import_folder=WindowsPath('Test Folder'), name='Test', some_option=False)
    >>> parameters.to_yaml()
    'import_folder: Test Folder\nname: Test\nsome_option: False\n'
    >>> yaml_text = '''
    ... import_folder: Test YAML Folder
    ... name: YAML test
    ... some_option: True
    ... '''
    >>> ExampleParameters.from_yaml(yaml_text)
    ExampleParameters(import_folder=WindowsPath('Test YAML Folder'), name='YAML test', some_option=True)
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
        # pylint: enable = unspecified-encoding

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

    def save_yaml(self, path: Path) -> None:
        """Write data from self to a YAML file.

        Parameters
        ----------
        path: Path
            Path to YAML file to output.
        """
        # pylint: disable = unspecified-encoding
        with open(path, "wt") as file:
            file.write(self.to_yaml())
        # pylint: enable = unspecified-encoding


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
