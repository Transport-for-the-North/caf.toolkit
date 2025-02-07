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
    data_generator: random.DataGenerator
    expected_output: pd.Series
    generator: np.random.Generator


@pytest.fixture(name="choice_not_all_values", scope="function")
def fixture_choice_not_all_values():
    return DataGeneratorRun(
        random.ChoiceGenerator("Developers", 10, ["Matt", "Isaac", "Ben"]),
        pd.Series(
            [
                "Ben",
                "Ben",
                "Matt",
                "Matt",
                "Ben",
                "Ben",
                "Isaac",
                "Matt",
                "Ben",
                "Isaac",
            ],
            name="Developers",
        ),
        np.random.default_rng(10),
    )


@pytest.fixture(name="choice_all_values", scope="function")
def fixture_choice_all_values():
    return DataGeneratorRun(
        random.ChoiceGenerator("Developers", 10, ["Matt", "Isaac", "Ben"], all_values=True),
        pd.Series(
            [
                "Ben",
                "Ben",
                "Matt",
                "Matt",
                "Ben",
                "Ben",
                "Isaac",
                "Matt",
                "Isaac",
                "Ben",
            ],
            name="Developers",
        ),
        np.random.default_rng(10),
    )


class TestRandomBuild:
    @pytest.mark.parametrize(
        "data_generator_run", ["choice_not_all_values", "choice_all_values"]
    )
    def test_choice_not_all_values(
        self, data_generator_run: str, request: pytest.FixtureRequest
    ):
        run: DataGeneratorRun = request.getfixturevalue(data_generator_run)
        test_values = run.data_generator.generate(run.generator)
        pd.testing.assert_series_equal(test_values, run.expected_output)
