# -*- coding: utf-8 -*-
"""Tests for the `arguments` module."""

##### IMPORTS #####

from __future__ import annotations

# Built-Ins
import pathlib

# Third Party
import pytest

# Local Imports
from caf.toolkit import arguments

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
]


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
