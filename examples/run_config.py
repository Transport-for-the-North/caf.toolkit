# -*- coding: utf-8 -*-
"""
Using `BaseConfig`
==================

The :class:`Baseconfig` class in caf.toolkit is designed to load
and validated YAML configuration files. This example shows how
to create child classes of `BaseConfig` to load parameters.
"""

# Built-Ins
import pathlib

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
# with different types of parameters.


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
# Use :func:`BaseConfig.to_yaml()` method to convert the class back to YAML,
# or :func:`BaseConfig.save_yaml()` to save the class as a YAML file.

print(parameters.to_yaml())

# %%
# Nesting
# -------


@dataclasses.dataclass
class Bounds:
    min_x: float
    min_y: float
    max_x: float
    max_y: float


@dataclasses.dataclass
class InputFile:
    name: str
    extent: Bounds
    path: pydantic.FilePath


class NestingConfig(ctk.BaseConfig):
    output_folder: pydantic.DirectoryPath
    model_run: str
    inputs: list[InputFile]


# %%
# Extra Validation
# ----------------


# %%
# Custom Validation Methods
# -------------------------
