# -*- coding: utf-8 -*-
"""Tests for the caf.toolkit.translation module"""
# Built-Ins
import dataclasses

from typing import Any
from typing import Optional

import pandas as pd

# Third Party
import pytest
import numpy as np


# Local Imports
# pylint: disable=import-error,wrong-import-position
from caf.toolkit import translation

# pylint: enable=import-error,wrong-import-position

# # # CONSTANTS # # #


# # # CLASSES # # #
@dataclasses.dataclass
class NumpyVectorResults:
    """Collection of I/O data for a numpy vector translation"""

    vector: np.ndarray
    translation: np.ndarray
    expected_result: np.ndarray
    translation_dtype: Optional[np.dtype] = None

    def input_kwargs(self, check_shapes: bool, check_totals: bool) -> dict[str, Any]:
        """Return a dictionary of key-word arguments"""
        return {
            "vector": self.vector,
            "translation": self.translation,
            "translation_dtype": self.translation_dtype,
            "check_shapes": check_shapes,
            "check_totals": check_totals,
        }


@dataclasses.dataclass
class PandasTranslation:
    """Container for a pandas based translation

    Takes a numpy translation and converts to a standard pandas format
    """

    np_translation: dataclasses.InitVar[np.ndarray]
    translation_from_col: str = "from_zone_id"
    translation_to_col: str = "to_zone_id"
    translation_factors_col: str = "factors"
    translation: pd.DataFrame = dataclasses.field(init=False)
    unique_from: list[Any] = dataclasses.field(init=False)
    unique_to: list[Any] = dataclasses.field(init=False)

    def __post_init__(self, np_translation: np.ndarray):
        """Convert numpy translation to pandas"""
        # Convert translation from numpy to long pandas
        df = pd.DataFrame(data=np_translation)
        df.index.name = self.translation_from_col
        df.columns.name = self.translation_to_col
        df.columns += 1
        df.index += 1
        df = df.reset_index()
        self.translation = df.melt(
            id_vars=self.translation_from_col,
            value_name=self.translation_factors_col,
        )

        # Get the unique from / to lists
        self.unique_from = self.translation[self.translation_from_col].unique().tolist()
        self.unique_to = self.translation[self.translation_to_col].unique().tolist()

    def to_kwargs(self) -> dict[str, Any]:
        """Return a dictionary of key-word arguments"""
        return {
            "translation": self.translation,
            "translation_from_col": self.translation_from_col,
            "translation_to_col": self.translation_to_col,
            "translation_factors_col": self.translation_factors_col,
        }


@dataclasses.dataclass
class PandasVectorResults:
    """Collection of I/O data for a pandas vector translation"""

    np_vector: dataclasses.InitVar[np.ndarray]
    np_expected_result: dataclasses.InitVar[np.ndarray]
    translation: PandasTranslation
    translation_dtype: Optional[np.dtype] = None

    vector: pd.Series = dataclasses.field(init=False)
    expected_result: pd.Series = dataclasses.field(init=False)
    from_unique_index: list[Any] = dataclasses.field(init=False)
    to_unique_index: list[Any] = dataclasses.field(init=False)

    def __post_init__(self, np_vector: np.ndarray, np_expected_result: np.ndarray):
        """Convert numpy objects to pandas"""
        # Input and results
        self.vector = pd.Series(data=np_vector)
        self.vector.index += 1
        self.expected_result = pd.Series(data=np_expected_result)
        self.expected_result.index += 1

        # Base from / to zones on translation
        self.from_unique_index = self.translation.unique_from
        self.to_unique_index = self.translation.unique_to

    def input_kwargs(
        self,
        check_totals: bool,
        vector_infill: float,
        translate_infill: float,
    ) -> dict[str, Any]:
        """Return a dictionary of key-word arguments"""
        return {
            "vector": self.vector,
            "from_unique_index": self.from_unique_index,
            "to_unique_index": self.to_unique_index,
            "translation_dtype": self.translation_dtype,
            "vector_infill": vector_infill,
            "translate_infill": translate_infill,
            "check_totals": check_totals,
        } | self.translation.to_kwargs()


# # # FIXTURES # # #
@pytest.fixture(name="simple_np_int_translation", scope="class")
def fixture_simple_np_int_translation() -> np.ndarray:
    """Generate a simple 5 to 3 complete translation array"""
    return np.array(
        [
            [1, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 1],
        ]
    )


@pytest.fixture(name="incomplete_np_int_translation", scope="class")
def fixture_incomplete_np_int_translation() -> np.ndarray:
    """Generate a simple 5 to 3 complete translation array"""
    return np.array(
        [
            [1, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0.6],
        ]
    )


@pytest.fixture(name="simple_np_float_translation", scope="class")
def fixture_simple_np_float_translation() -> np.ndarray:
    """Generate a simple 5 to 3 complete translation array"""
    return np.array(
        [
            [0.5, 0.0, 0.5],
            [0.25, 0.5, 0.25],
            [0, 1, 0],
            [0.5, 0.0, 0.5],
            [0.25, 0.5, 0.25],
        ]
    )


@pytest.fixture(name="simple_pd_int_translation", scope="class")
def fixture_simple_pd_int_translation(
    simple_np_int_translation: np.ndarray,
) -> PandasTranslation:
    """Generate a simple 5 to 3 complete translation array"""
    return PandasTranslation(simple_np_int_translation)


@pytest.fixture(name="incomplete_pd_int_translation", scope="class")
def fixture_incomplete_pd_int_translation(
    incomplete_np_int_translation: np.ndarray,
) -> PandasTranslation:
    """Generate a simple 5 to 3 complete translation array"""
    return PandasTranslation(incomplete_np_int_translation)


@pytest.fixture(name="simple_pd_float_translation", scope="class")
def fixture_simple_pd_float_translation(
    simple_np_float_translation: np.ndarray,
) -> PandasTranslation:
    """Generate a simple 5 to 3 complete translation array"""
    return PandasTranslation(simple_np_float_translation)


# TODO(BT): Pandas vector will build on this
@pytest.fixture(name="np_vector_aggregation", scope="class")
def fixture_np_vector_aggregation(simple_np_int_translation: np.ndarray) -> NumpyVectorResults:
    """Generate an aggregation vector, translation, and results"""
    return NumpyVectorResults(
        vector=np.array([8, 2, 8, 8, 5]),
        translation=simple_np_int_translation,
        expected_result=np.array([10, 8, 13]),
    )


@pytest.fixture(name="np_vector_split", scope="class")
def fixture_np_vector_split(simple_np_float_translation: np.ndarray) -> NumpyVectorResults:
    """Generate a splitting vector, translation, and results"""
    return NumpyVectorResults(
        vector=np.array([8, 2, 8, 8, 5]),
        translation=simple_np_float_translation,
        expected_result=np.array([9.75, 11.5, 9.75]),
    )


@pytest.fixture(name="np_incomplete", scope="class")
def fixture_np_incomplete(incomplete_np_int_translation: np.ndarray) -> NumpyVectorResults:
    """Generate an incomplete vector, translation, and results

    Incomplete meaning some demand will be dropped during the translation
    """
    return NumpyVectorResults(
        vector=np.array([8, 2, 8, 8, 5]),
        translation=incomplete_np_int_translation,
        expected_result=np.array([10, 8, 11]),
    )


@pytest.fixture(name="np_translation_dtype", scope="class")
def fixture_np_translation_dtype(simple_np_int_translation: np.ndarray) -> NumpyVectorResults:
    """Generate an incomplete vector, translation, and results

    Incomplete meaning some demand will be dropped during the translation
    """
    return NumpyVectorResults(
        vector=np.array([8.1, 2.2, 8.3, 8.4, 5.5]),
        translation=simple_np_int_translation,
        translation_dtype=np.int32,
        expected_result=np.array([10, 8, 11]),
    )


@pytest.fixture(name="pd_vector_aggregation", scope="class")
def fixture_pd_vector_aggregation(
    np_vector_aggregation: NumpyVectorResults,
    simple_pd_int_translation: PandasTranslation,
) -> PandasVectorResults:
    """Generate an aggregation vector, translation, and results"""
    return PandasVectorResults(
        np_vector=np_vector_aggregation.vector,
        np_expected_result=np_vector_aggregation.expected_result,
        translation=simple_pd_int_translation,
    )


@pytest.fixture(name="pd_vector_split", scope="class")
def fixture_pd_vector_split(
    np_vector_split: NumpyVectorResults,
    simple_pd_float_translation: PandasTranslation,
) -> PandasVectorResults:
    """Generate a splitting vector, translation, and results"""
    return PandasVectorResults(
        np_vector=np_vector_split.vector,
        np_expected_result=np_vector_split.expected_result,
        translation=simple_pd_float_translation,
    )


# # # TESTS # # #
@pytest.mark.usefixtures(
    "np_vector_aggregation",
    "np_vector_split",
    "np_incomplete",
    "np_translation_dtype",
)
class TestNumpyVector:
    """Tests for caf.toolkit.translation.numpy_vector_zone_translation"""

    @pytest.mark.parametrize("check_totals", [True, False])
    def test_dropped_totals(self, np_incomplete: NumpyVectorResults, check_totals: bool):
        """Test for total checking with dropped demand"""
        kwargs = np_incomplete.input_kwargs(check_shapes=True, check_totals=check_totals)
        if not check_totals:
            result = translation.numpy_vector_zone_translation(**kwargs)
            np.testing.assert_allclose(result, np_incomplete.expected_result)
        else:
            with pytest.raises(ValueError, match="Some values seem to have been dropped"):
                translation.numpy_vector_zone_translation(**kwargs)

    @pytest.mark.parametrize("check_shapes", [True, False])
    def test_non_vector(self, np_vector_split: NumpyVectorResults, check_shapes: bool):
        """Test for error when non-vector given"""
        # Convert vector to matrix
        new_vector = np_vector_split.vector
        new_vector = np.broadcast_to(new_vector, (new_vector.shape[0], new_vector.shape[0]))

        # Set expected error message
        if check_shapes:
            msg = "not a vector"
        else:
            msg = "was there a shape mismatch?"

        # Call with expected error
        kwargs = np_vector_split.input_kwargs(check_shapes=check_shapes, check_totals=True)
        with pytest.raises(ValueError, match=msg):
            translation.numpy_vector_zone_translation(**(kwargs | {"vector": new_vector}))

    @pytest.mark.parametrize("check_shapes", [True, False])
    def test_translation_shape(
        self,
        np_vector_split: NumpyVectorResults,
        check_shapes: bool,
    ):
        """Test for error when wrong shape translation"""
        # Convert vector to matrix
        new_trans = np_vector_split.translation
        new_trans = np.vstack([new_trans, new_trans])

        # Set expected error message
        if check_shapes:
            msg = "translation does not have the correct number of rows"
        else:
            msg = "was there a shape mismatch?"

        # Call with expected error
        kwargs = np_vector_split.input_kwargs(check_shapes=check_shapes, check_totals=True)
        with pytest.raises(ValueError, match=msg):
            translation.numpy_vector_zone_translation(**(kwargs | {"translation": new_trans}))

    @pytest.mark.parametrize(
        "np_vector_str",
        ["np_vector_aggregation", "np_vector_split"],
    )
    def test_vector_like(self, np_vector_str: str, request):
        """Test vector-like arrays (empty in 2nd dim)"""
        np_vector = request.getfixturevalue(np_vector_str)
        new_vector = np.expand_dims(np_vector.vector, 1)
        kwargs = np_vector.input_kwargs(check_shapes=True, check_totals=True)
        result = translation.numpy_vector_zone_translation(**(kwargs | {"vector": new_vector}))
        np.testing.assert_allclose(result, np_vector.expected_result)

    @pytest.mark.parametrize(
        "np_vector_str",
        ["np_vector_aggregation", "np_vector_split"],
    )
    @pytest.mark.parametrize("check_totals", [True, False])
    def test_translation_correct(
        self,
        np_vector_str: str,
        check_totals: bool,
        request,
    ):
        """Test that aggregation and splitting give correct results

        Also checks that totals are correctly checked.
        """
        np_vector = request.getfixturevalue(np_vector_str)
        kwargs = np_vector.input_kwargs(check_shapes=True, check_totals=check_totals)
        result = translation.numpy_vector_zone_translation(**kwargs)
        np.testing.assert_allclose(result, np_vector.expected_result)


@pytest.mark.usefixtures(
    "pd_vector_aggregation",
    "pd_vector_split",
)
class TestPandasVector:
    """Tests for caf.toolkit.translation.pandas_vector_zone_translation"""

    @pytest.mark.parametrize(
        "pd_vector_str",
        ["pd_vector_aggregation", "pd_vector_split"],
    )
    def test_series_like(self, pd_vector_str: str, request):
        """Test vector-like arrays (empty in 2nd dim)"""
        pd_vector = request.getfixturevalue(pd_vector_str)
        new_vector = pd.DataFrame(pd_vector.vector)
        kwargs = pd_vector.input_kwargs(check_totals=True, vector_infill=0, translate_infill=0)
        result = translation.pandas_vector_zone_translation(
            **(kwargs | {"vector": new_vector})
        )
        pd.testing.assert_series_equal(result, pd_vector.expected_result, check_names=False)

    @pytest.mark.parametrize(
        "pd_vector_str",
        ["pd_vector_aggregation", "pd_vector_split"],
    )
    @pytest.mark.parametrize("check_totals", [True, False])
    def test_translation_correct(
        self,
        pd_vector_str: str,
        check_totals: bool,
        request,
    ):
        """Test that aggregation and splitting give correct results

        Also checks that totals are correctly checked.
        """
        pd_vector = request.getfixturevalue(pd_vector_str)
        kwargs = pd_vector.input_kwargs(
            check_totals=check_totals, vector_infill=0, translate_infill=0
        )
        result = translation.pandas_vector_zone_translation(**kwargs)
        pd.testing.assert_series_equal(result, pd_vector.expected_result)
