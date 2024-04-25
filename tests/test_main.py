# -*- coding: utf-8 -*-
"""Tests for the toolkit `__main__` module."""

##### IMPORTS #####

from __future__ import annotations

# Built-Ins
import pathlib
import sys

# Third Party
import pytest

# Local Imports
from caf.toolkit.__main__ import MatrixTranslationArgs, TranslationArgs, parse_args

##### CONSTANTS #####


##### FIXTURES & TESTS #####


@pytest.fixture(name="dummy_files", scope="module")
def fix_dummy_files(tmp_path_factory: pytest.TempPathFactory) -> dict[str, pathlib.Path]:
    """Write empty dummy files for the Translation / MatrixTranslation validation."""
    tmp_path = tmp_path_factory.mktemp("dummy_files")
    paths = {}
    for name in ("data_file", "translation_file", "output_file"):
        path = tmp_path / f"{name}.csv"
        path.touch()
        paths[name] = path

    return paths


@pytest.fixture(name="translate_config")
def fix_translate_config(
    dummy_files: dict[str, pathlib.Path], tmp_path: pathlib.Path
) -> tuple[TranslationArgs, pathlib.Path]:
    """Create translation config."""
    args = TranslationArgs(
        data_file=dummy_files["data_file"], translation_file=dummy_files["translation_file"]
    )

    path = tmp_path / "translate_config.yml"
    args.save_yaml(path)
    return args, path


@pytest.fixture(name="matrix_translate_config")
def fix_matrix_translate_config(
    dummy_files: dict[str, pathlib.Path], tmp_path: pathlib.Path
) -> tuple[MatrixTranslationArgs, pathlib.Path]:
    """Create matrix translation config."""
    args = MatrixTranslationArgs(
        data_file=dummy_files["data_file"], translation_file=dummy_files["translation_file"]
    )

    path = tmp_path / "matrix_translate_config.yml"
    args.save_yaml(path)
    return args, path


class TestParseArgs:
    """Test `parse_args` function to make sure it returns the correct arguments."""

    @pytest.mark.parametrize("type_", ["translate", "matrix-translate"])
    def test_min_parameters(self, dummy_files: dict[str, pathlib.Path], type_) -> None:
        """Testing running with bare minimum arguments."""
        sys.argv = [
            "caf.toolkit",
            type_,
            str(dummy_files["data_file"]),
            str(dummy_files["translation_file"]),
        ]

        if type_ == "translate":
            expected = TranslationArgs(
                data_file=dummy_files["data_file"],
                translation_file=dummy_files["translation_file"],
            )
        elif type_ == "matrix-translate":
            expected = MatrixTranslationArgs(
                data_file=dummy_files["data_file"],
                translation_file=dummy_files["translation_file"],
            )
        else:
            raise ValueError(f"incorrect value for {type_ = }")

        args = parse_args()

        assert args == expected, "incorrect translate arguments"

    @pytest.mark.parametrize(
        "fixture, name",
        [
            ("translate_config", "translate-config"),
            ("matrix_translate_config", "matrix-translate-config"),
        ],
    )
    def test_load_config(self, request: pytest.FixtureRequest, fixture, name):
        """Test running with path to config."""
        expected, path = request.getfixturevalue(fixture)

        sys.argv = ["caf.toolkit", name, str(path)]

        args = parse_args()

        assert args == expected, "incorrect config arguments"

    def test_complete_translate_parameters(self, dummy_files: dict[str, pathlib.Path]):
        """Test translate sub-command with full set of arguments."""
        expected = TranslationArgs(
            data_file=dummy_files["data_file"],
            translation_file=dummy_files["translation_file"],
            output_file=dummy_files["output_file"],
            from_column="from_zone_id",
            to_column="to_zone_id",
            factor_column="factor_col",
            zone_column="zone_id",
        )

        # Testing abreviated names too
        sys.argv = [
            "caf.toolkit",
            "translate",
            "--o",
            str(expected.output_file),
            "--from_column",
            expected.from_column,
            "--to_column",
            expected.to_column,
            "--factor",
            expected.factor_column,
            "--z",
            expected.zone_column,
            str(expected.data_file),
            str(expected.translation_file),
        ]

        args = parse_args()

        assert args == expected, "incorrect translate parameters"

    def test_complete_matrix_translate_parameters(self, dummy_files: dict[str, pathlib.Path]):
        """Test matrix_translate sub-command with full set of arguments."""
        expected = MatrixTranslationArgs(
            data_file=dummy_files["data_file"],
            translation_file=dummy_files["translation_file"],
            output_file=dummy_files["output_file"],
            from_column="from_zone_id",
            to_column="to_zone_id",
            factor_column="factor_col",
            zone_column=("origin", "destination"),
            value_column="value",
        )

        # Testing abreviated names too
        sys.argv = [
            "caf.toolkit",
            "matrix-translate",
            "--o",
            str(expected.output_file),
            "--from_column",
            expected.from_column,
            "--to_column",
            expected.to_column,
            "--factor",
            expected.factor_column,
            "--z",
            *expected.zone_column,
            "--val",
            expected.value_column,
            str(expected.data_file),
            str(expected.translation_file),
        ]

        args = parse_args()

        assert args == expected, "incorrect translate parameters"
