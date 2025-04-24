# -*- coding: utf-8 -*-
"""Tests for the `arguments` module."""

##### IMPORTS #####

from __future__ import annotations

# Built-Ins
import argparse
import os
import pathlib
import sys

# Third Party
import pytest

# Local Imports
from caf.toolkit import arguments, config_base

##### CONSTANTS #####


##### FIXTURES & TESTS #####


CORRECT_ANNOTATIONS = [
    ("Optional[int]", (int, True, None)),
    ("int | None", (int, True, None)),
    ("pydantic.FilePath", (pathlib.Path, False, None)),
    ("pathlib.Path", (pathlib.Path, False, None)),
    ("int | str", (str, False, None)),
    ("str | int", (str, False, None)),
    ("int | float", (float, False, None)),
    ("int | str | None", (str, True, None)),
    ("tuple[int | str, int | str]", (str, False, 2)),
    ("list[int]", (int, False, "*")),
    ("Union[str, int]", (str, False, None)),
    ("<class 'bool'>", (bool, False, None)),
    ("<class 'int'>", (int, False, None)),
    ("<class 'str'>", (str, False, None)),
    ("<class 'float'>", (float, False, None)),
    ("<class 'pathlib.Path'>", (pathlib.Path, False, None)),
    ("<class 'pathlib._local.Path'>", (pathlib.Path, False, None)),
]


@pytest.fixture(name="parser")
def fix_parser() -> argparse.ArgumentParser:
    """Empty ArgumentParser."""
    return argparse.ArgumentParser()


class TestParseArgDetails:
    """Tests for `parse_arg_details` function."""

    @pytest.mark.parametrize("test_data", CORRECT_ANNOTATIONS)
    def test_correct(self, test_data: tuple[str, tuple[type, bool, int | str | None]]):
        """Test annotations the function can handle."""
        annotation, expected = test_data
        type_, optional, nargs = arguments.parse_arg_details(annotation)

        assert type_ == expected[0], "incorrect type found"
        assert optional is expected[1], "incorrect optional"

        if expected[2] is None:
            assert nargs is expected[2], "incorrect nargs"
        else:
            assert nargs == expected[2], "incorrect nargs"

    @pytest.mark.parametrize("annotation", ["dict[str, int]"])
    def test_unknown_formats(self, annotation: str) -> None:
        """Test annotations the function can't handle."""
        with pytest.warns(arguments.TypeAnnotationWarning):
            type_, optional, nargs = arguments.parse_arg_details(annotation)

        assert type_ == str, "incorrect default type"
        assert optional is False, "incorrect default optional"
        assert nargs is None, "incorrect default nargs"


class TestReplaceUnion:
    """Tests for the `_replace_union` function."""

    @pytest.mark.parametrize(
        "annotation, expected",
        [
            ("union[int, str]", "int | str"),
            ("Union[   int  , str , float]", "int | str | float"),
            ("list[Union[float, int]]", "list[float | int]"),
            ("list[int]", "list[int]"),
            (
                "tuple[Union[int, float, str], Union[str, int]]",
                "tuple[int | float | str, str | int]",
            ),
            ("tuple[int, Union[int, str]]", "tuple[int, int | str]"),
        ],
    )
    def test_replace_union(self, annotation: str, expected: str) -> None:
        """Test `_replace_union` function works as expected."""
        # pylint: disable=protected-access
        assert arguments._replace_union(annotation) == expected


class TestGetenvBool:
    """Tests for the `getenv_bool` function."""

    _variable_name = "TEST_TOOLKIT_ENV_VARIABLE"

    def test_default(self):
        """Test that the default parameter is correctly returned."""
        os.environ[self._variable_name] = ""

        assert arguments.getenv_bool(self._variable_name, True)
        assert not arguments.getenv_bool(self._variable_name, False)

    @pytest.mark.parametrize("value", ["TRUE", "Yes", "y", "1", "true  "])
    def test_true(self, value: str):
        """Test the possible values for True."""
        os.environ[self._variable_name] = value
        assert arguments.getenv_bool(self._variable_name, False)

    @pytest.mark.parametrize("value", ["false", "no", "n", "0", "FALSE", "n "])
    def test_false(self, value: str):
        """Test the possible values for False."""
        os.environ[self._variable_name] = value
        assert not arguments.getenv_bool(self._variable_name, True)

    @pytest.mark.parametrize("value", ["10", "01", "wrong", "t", "f"])
    def test_invalid(self, value: str):
        """Test an invalid value raises the correct error."""
        pattern = (
            r"unexpected value (.*) for '.*' env "
            r"variable should be one of the following:\n"
            r" For true: .*\n For false: .*"
        )
        os.environ[self._variable_name] = value

        with pytest.raises(ValueError, match=pattern):
            arguments.getenv_bool(self._variable_name, False)


if sys.version_info.minor <= 9:
    # Creating dummy class because the test is skipped
    class _ArgumentsConfigTest: ...  # pylint: disable=too-few-public-methods

else:
    # Ignore already defined
    class _ArgumentsConfigTest(config_base.BaseConfig):  # type: ignore
        """Class for testing `ModelArguments`."""

        text: str
        number: float | int
        path: pathlib.Path
        general_list: list[str | int]


class TestModelArguments:
    """Tests for the `ModelArguments` class."""

    @pytest.mark.skipif(
        sys.version_info.minor <= 9,
        reason="uses | in type annotations which was added in 3.10",
    )
    @pytest.mark.filterwarnings("error")
    def test_add_subcommands(self, parser: argparse.ArgumentParser):
        """Test the `add_subcommands` method works without errors / warnings.

        **Doesn't check the correct sub-commands have been added.**
        """
        subparsers = parser.add_subparsers()

        arg_class = arguments.ModelArguments(_ArgumentsConfigTest)
        arg_class.add_subcommands(subparsers, "test")
