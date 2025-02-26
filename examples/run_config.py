# -*- coding: utf-8 -*-
"""
Using :class:`~caf.toolkit.BaseConfig`
======================================

The :class:`~caf.toolkit.BaseConfig` class in caf.toolkit is designed to load
and validated YAML configuration files. This example shows how
to create child classes of BaseConfig to load parameters.

See Also
--------
:class:`~caf.toolkit.arguments.ModelArguments`
    for producing command-line arguments from a BaseConfig class.
"""

# Built-Ins
import pathlib
import re
from typing import Any, Self

# Third Party
import pydantic
from pydantic import dataclasses

# Local Imports
import caf.toolkit as ctk

folder = pathlib.Path().parent

# %%
# Basic
# -----
#
# This example shows how to create a simple configuration file, without nesting, and
# with different types of parameters. The majority of Python built-in types can be used,
# additionally dataclasses and many "simple"[#simple]_ custom types can be used.
#
# .. seealso ::
#       :ref:`Extra Validation` for information on more complex validation.
#
# The :class:`pydantic.DirectoryPath` and :class:`pydantic.FilePath` types both return
# :class:`pathlib.Path` objects after validating that the directory, or file, exists.


class Config(ctk.BaseConfig):
    """Example of a basic configuration file without nesting."""

    years: list[int]
    name: str
    output_folder: pydantic.DirectoryPath
    input_file: pydantic.FilePath


# %%
# Example of the YAML config file which is loaded by the above class.
#
# .. literalinclude:: /../../examples/basic_config.yml
#       :language: yaml
#       :caption: Config file: examples/basic_config.yml
#
# Below shows how to load the config file and displays the class as text.

parameters = Config.load_yaml(folder / "basic_config.yml")
print(parameters)

# %%
# Use :meth:`~caf.toolkit.BaseConfig.to_yaml()` method to convert the class back to YAML,
# or :meth:`~caf.toolkit.BaseConfig.save_yaml()` to save the class as a YAML file.

print(parameters.to_yaml())

# %%
# Nesting
# -------
#
# More complex configuration files can be handled using dataclasses,
# :class:`pydantic.BaseModel` subclasses or :class:`~caf.toolkit.BaseConfig` subclasses.
#
# :class:`Bounds` is a simple dataclass containing 4 floats.


@dataclasses.dataclass
class Bounds:
    """Bounding box coordinates."""

    min_x: float
    min_y: float
    max_x: float
    max_y: float


# %%
# :class:`InputFile` is an example of a dataclass which contains another dataclass as an attribute.


@dataclasses.dataclass
class InputFile:
    """Example of an input file dataclass with some additional information."""

    name: str
    extent: Bounds
    path: pydantic.FilePath


# %%
# :class:`NestingConfig` is an example of a configuration file with a list of nested dataclasses.


class NestingConfig(ctk.BaseConfig):
    output_folder: pydantic.DirectoryPath
    model_run: str
    inputs: list[InputFile]


# %%
# Example of the YAML config file which is loaded by the above class.
#
# .. literalinclude:: /../../examples/nested_config.yml
#       :language: yaml
#       :caption: Config file: examples/nested_config.yml
#
# Below shows how to load the config file and displays the class as text.

parameters = NestingConfig.load_yaml(folder / "nested_config.yml")
print(parameters)

# %%
# Extra Validation
# ----------------
#
# Pydantic provides some functionality for adding additional validation to
# subclasses of :class:`pydantic.BaseModel` (or pydantic dataclasses),
# which :class:`caf.toolkit.BaseConfig` is based on.
#
# The simplest approach to pydantic's validation is using :class:`pydantic.Field` to
# add some additional validation.


@dataclasses.dataclass
class FieldValidated:
    """Example of addtion attribute validation with Field class."""

    # Numbers with restrictions
    positive: int = pydantic.Field(gt=0)
    small_number: float = pydantic.Field(ge=0, le=1)
    even: int = pydantic.Field(multiple_of=2)

    # Text restrictions
    short_text: str = pydantic.Field(max_length=10)
    # Regular expression pattern only allowing lowercase letters
    regex_text: str = pydantic.Field(pattern=r"[a-z]+")

    # Iterable restrictions e.g. lists and tuples
    short_list: list[int] = pydantic.Field(min_length=2, max_length=5)


# %%
# For more complex validation pydantic allow's custom methods to be defined
# to validate individual fields (:func:`pydantic.field_validator`), or the
# class as a whole (:func:`pydantic.model_validator`).[#valid]_
#
# :class:`CustomFieldValidated` gives an example of using the
# :func:`pydantic.field_validator` decorator to validate the whole class.


@dataclasses.dataclass
class CustomFieldValidated:
    """Example of using pydantics field validator decorator."""

    sorted_list: list[int]
    flexible_list: list[int]

    @pydantic.field_validator("sorted_list")
    @classmethod
    def validate_order(cls, value: list[int]) -> list[int]:
        """Validate the list is sorted."""
        previous = None
        for i, val in enumerate(value):
            if previous is not None and val < previous:
                raise ValueError(f"item {i} ({val}) is smaller than previous ({previous})")

            previous = val

        return value

    # This validation method is ran before pydantic does any validation
    @pydantic.field_validator("flexible_list", mode="before")
    @classmethod
    def csv_list(cls, value: Any) -> list:
        """Split text into list based on commas.."""
        if isinstance(value, str):
            return value.split(",")
        return value


# %%
# :class:`ModelValidated` gives an example of using the
# :func:`pydantic.model_validator` decorator to validate the whole class.


@dataclasses.dataclass
class ModelValidated:
    """Example of using pydantics model validator decorator."""

    possible_values: list[str]
    favourite_value: str

    @pydantic.model_validator(mode="after")
    def check_favourite(self) -> Self:
        """Checks if favourite value is in the list of possible values."""
        if self.favourite_value not in self.possible_values:
            raise ValueError(
                f"favourite value ({self.favourite_value})"
                " isn't found in list of possible values"
            )

        return self


# %%
# :class:`ExtraValidatedConfig` includes the additional validation methods
# discussed in the classes above in a config class.


class ExtraValidatedConfig(ctk.BaseConfig):
    """Config class showing examples of custom validation."""

    simple_validated: FieldValidated
    custom_validated: CustomFieldValidated
    model_validated: ModelValidated


# %%
# Example of the YAML config file which is loaded by the above class.
#
# .. literalinclude:: /../../examples/validated_config.yml
#       :language: yaml
#       :caption: Config file: examples/validated_config.yml
#
# Below shows how to load the config file and displays the class as text.

parameters = ExtraValidatedConfig.load_yaml(folder / "validated_config.yml")
print(parameters)


# %%
# .. rubric:: Footnotes
#
# .. [#simple] "simple" is referring to a type which is initialised with a single
#       string parameter.
#
# .. [#valid] `Pydantic validator documentation
#       <https://docs.pydantic.dev/latest/concepts/validators/>`_
