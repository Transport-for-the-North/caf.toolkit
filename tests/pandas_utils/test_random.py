"""Tests for the pandas utils random module."""
# Built-Ins
import dataclasses

# Third Party
import numpy as np
import pandas as pd
import pytest

# Local Imports
from caf.toolkit.pandas_utils import random


@dataclasses.dataclass
class DataGeneratorRun:
    """Data generator, the expected output from the numpy generator provided."""

    data_generator: random.DataGenerator
    """Data generator to test."""
    expected_output: pd.Series
    """Expected output from data generator when using generator"""
    generator: np.random.Generator
    """Generator to use for test run"""


@pytest.fixture(name="choice_not_all_values", scope="function")
def fixture_choice_not_all_values() -> DataGeneratorRun:
    """Test params for ChoiceGenerator without all values turned on."""
    return DataGeneratorRun(
        random.ChoiceGenerator("Developers", 10, ["Matt", "Isaac", "Ben"]),
        pd.Series(
            ["Ben", "Ben", "Matt", "Matt", "Ben", "Ben", "Isaac", "Matt", "Ben", "Isaac"],
            name="Developers",
        ),
        np.random.default_rng(10),
    )  # fmt: skip


@pytest.fixture(name="choice_all_values", scope="function")
def fixture_choice_all_values() -> DataGeneratorRun:
    """Test params for ChoiceGenerator with all values turned on."""
    return DataGeneratorRun(
        random.ChoiceGenerator(
            "Developers", 10, ["Matt", "Isaac", "Ben"], all_values=True
        ),
        pd.Series(
            ["Ben", "Ben", "Matt", "Matt", "Ben", "Ben", "Isaac", "Matt", "Isaac", "Ben"],
            name="Developers",
        ),
        np.random.default_rng(10),
    )  # fmt: skip


@pytest.fixture(name="float_lower", scope="function")
def fixture_float_lower() -> DataGeneratorRun:
    """Test params for FloatGenerator with lower values turned passed."""
    return DataGeneratorRun(
        random.FloatGenerator("Steves_lucky_numbers", 10, 1000, 10),
        pd.Series(
            [
                956.441693, 215.604992, 830.160436, 157.789302, 517.676570,
                144.560408, 692.146115, 843.330247, 431.253907, 957.356743,
            ],
            name="Steves_lucky_numbers",
        ),
        np.random.default_rng(10),
    )  # fmt: skip


@pytest.fixture(name="float_no_lower", scope="function")
def fixture_float_no_lower() -> DataGeneratorRun:
    """Test params for FloatGenerator without lower value passed."""
    return DataGeneratorRun(
        random.FloatGenerator("Steves_lucky_numbers", 10, 1000),
        pd.Series(
            [
                956.001710, 207.681810, 828.444885, 149.282123, 512.804616,
                135.919604, 689.036480, 841.747724, 425.508997, 956.926003
            ],
            name="Steves_lucky_numbers"
        ),
        np.random.default_rng(10),
    )  # fmt: skip


@pytest.fixture(name="int_no_lower", scope="function")
def fixture_int_no_lower() -> DataGeneratorRun:
    """Test params for IntGenerator without lower value passed."""
    return DataGeneratorRun(
        random.IntGenerator("Steves_phone_number", 11, 9),
        pd.Series([6, 8, 2, 1, 7, 7, 4, 1, 7, 4, 1], name="Steves_phone_number"),
        np.random.default_rng(10),
    )


@pytest.fixture(name="int_lower", scope="function")
def fixture_int_lower() -> DataGeneratorRun:
    """Test params for IntGenerator with lower value passed."""
    return DataGeneratorRun(
        random.IntGenerator("Steves_phone_number", 11, 9, 2),
        pd.Series([7, 8, 3, 3, 7, 7, 5, 3, 7, 5, 3], name="Steves_phone_number"),
        np.random.default_rng(10),
    )


@pytest.fixture(name="id_no_starting_val", scope="function")
def fixture_id_no_starting_val() -> DataGeneratorRun:
    """Test params for UnqiueIdGenerator without starting value passed."""
    return DataGeneratorRun(
        random.UniqueIdGenerator(
            "Steves_least_favourite_numbers",
            20,
        ),
        pd.Series(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
            name="Steves_least_favourite_numbers",
        ),
        np.random.default_rng(10),
    )


@pytest.fixture(name="id_starting_val", scope="function")
def fixture_id_starting_val() -> DataGeneratorRun:
    """Test params for UnqiueIdGenerator with starting value passed."""
    return DataGeneratorRun(
        random.UniqueIdGenerator("Steves_least_favourite_numbers", 20, 4),
        pd.Series(
            [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
            name="Steves_least_favourite_numbers",
        ),
        np.random.default_rng(10),
    )


class TestRandomBuild:
    """Test :class:`DataGeneratorRun`."""

    @pytest.mark.parametrize(
        "data_generator_run",
        [
            "choice_not_all_values",
            "choice_all_values",
            "float_lower",
            "float_no_lower",
            "int_lower",
            "int_no_lower",
            "id_no_starting_val",
            "id_starting_val",
        ],
    )
    def test_choice_not_all_values(
        self, data_generator_run: str, request: pytest.FixtureRequest
    ) -> None:
        """Tests whether data generators produce the expected output."""
        run: DataGeneratorRun = request.getfixturevalue(data_generator_run)
        test_values = run.data_generator.generate(run.generator)
        # Setting dtype to False since local run and github run choose different bit-ness
        pd.testing.assert_series_equal(
            test_values, run.expected_output, check_dtype=False
        )
