"""Build Dummy datasets for the purposes of testing and demonstrations."""

from __future__ import annotations

# Built-Ins
import abc

# Third Party
import numpy as np
import pandas as pd


class DataGenerator(abc.ABC):
    """Asbstract base class for data generation."""

    # pylint: disable=too-few-public-methods
    name: str
    """ Name to be used as resultant series label."""
    length: int
    """ Number of rows to create."""

    def __init__(self, name: str, length: int):
        self.name = name
        self.length = length

    @abc.abstractmethod
    def generate(self, generator: np.random.Generator) -> pd.Series:
        """Generate data using the generator provided and specifications define in the attributes.

        Parameters
        ----------
        generator: np.random.Generator
            Generator used to produce data values.

        Returns
        -------
        pd.Series
            Generated data.
        """


class ChoiceGenerator(DataGenerator):
    """Generates data using a list of allowed values."""

    # pylint: disable=too-few-public-methods
    values: set[int | str]  # we use a set so there can't be duplicates
    """Values to select for data values"""
    all_values: bool
    """Whether to ensure the resultant data contains all elements defined in values."""

    def __init__(
        self, name: str, length: int, values: set[int | str], all_values: bool = False
    ):
        super().__init__(name, length)
        self.values = values

        if all_values and (len(values) >= length):
            raise ValueError(
                "all_values has been set to True when the number of choices "
                "is greater than length"
            )
        self.all_values = all_values

    def generate(self, generator: np.random.Generator) -> pd.Series:
        """Generate data using the generator provided and specifications define in the attributes.

        Parameters
        ----------
        generator: np.random.Generator
            Generator used to produce data values.

        Returns
        -------
        pd.Series
            Generated data.

        """
        if self.all_values:
            # check is made on init whether length < length(value) so we don't worry about this here
            if self.length == len(self.values):
                return pd.Series(list(self.values), name=self.name)

            generated_values = generator.choice(
                list(self.values), self.length - len(self.values)
            )
            generated_values = np.append(generated_values, list(self.values))

        else:
            generated_values = generator.choice(list(self.values), self.length)

        return pd.Series(generated_values, name=self.name)


class FloatGenerator(DataGenerator):
    """Generates float data between lower and upper ranges."""

    # pylint: disable=too-few-public-methods
    lower_range: float
    """Lower range of data."""
    upper_range: float
    """Upper range of data."""

    def __init__(self, name: str, length: int, upper_range: float, lower_range: float = 0):
        super().__init__(name, length)

        if lower_range >= upper_range:
            raise ValueError(
                f"upper_range ({upper_range}) should be greater than lower_range ({lower_range})"
            )

        self.lower_range = lower_range
        self.upper_range = upper_range

    def generate(self, generator: np.random.Generator) -> pd.Series:
        """Generate data using the generator provided and specifications define in the attributes.

        Parameters
        ----------
        generator: np.random.Generator
            Generator used to produce data values.

        Returns
        -------
        pd.Series
            Generated data.
        """
        # this generates random floats between 0 and 1
        generated_seed_values = generator.random(self.length)

        # this tranlates the seed values to floats within the specified range
        values = self.lower_range + (
            generated_seed_values * (self.upper_range - self.lower_range)
        )

        return pd.Series(values, name=self.name)


class UniqueIdGenerator(DataGenerator):
    """Generates a set of unique ID."""

    # pylint: disable=too-few-public-methods
    starting_val: int
    """Starting value for ID."""

    def __init__(self, name: str, length: int, starting_val: int = 0):
        super().__init__(name, length)
        self.starting_val = starting_val
        if length <= 0 or not isinstance(length, int):
            raise ValueError("length must be positive, non zero, integer")

    def generate(self, generator) -> pd.Series:
        """Generate data using the generator provided and specifications define in the attributes.

        Parameters
        ----------
        generator
            The generator is not used and only exists to conform to
            the method signature of the base class.

        Returns
        -------
        pd.Series
            Generated data.
        """
        del generator
        values = np.arange(start=self.starting_val, stop=self.starting_val + self.length)
        return pd.Series(values, name=self.name)


class IntGenerator(DataGenerator):
    """Generates integer data."""

    # pylint: disable=too-few-public-methods
    lower_range: int
    """Lower range of data, included in generated data."""
    upper_range: int
    """Upper range of data, maximum in generated data will be one less."""

    def __init__(self, name: str, length: int, upper_range: int = 0, lower_range: int = 0):
        super().__init__(name, length)

        if lower_range >= upper_range:
            raise ValueError(
                f"upper_range ({upper_range}) should be greater than lower_range ({lower_range})"
            )

        self.lower_range = lower_range
        self.upper_range = upper_range

    def generate(self, generator: np.random.Generator) -> pd.Series:
        """Generate data using the generator provided and specifications define in the attributes.

        Parameters
        ----------
        generator: np.random.Generator
            Generator used to produce data values.

        Returns
        -------
        pd.Series
            Generated data.
        """
        values = generator.integers(self.lower_range, self.upper_range, size=self.length)
        return pd.Series(values, name=self.name)
