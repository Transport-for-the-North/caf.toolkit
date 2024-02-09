# -*- coding: utf-8 -*-
"""Tests for the `arguments` module."""

##### IMPORTS #####

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
