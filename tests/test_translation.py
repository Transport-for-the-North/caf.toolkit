# -*- coding: utf-8 -*-
"""Tests for the caf.toolkit.translation module"""
# Built-Ins
import dataclasses

from typing import Any
from typing import Optional

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


# # # TESTS # # #
@pytest.mark.usefixtures(
    "np_vector_aggregation",
    "np_vector_split",
    "np_incomplete",
    "np_translation_dtype",
)
class TestNumpyVector:
    """Tests for caf.toolkit.translation.numpy_vector_zone_translation"""

    # TODO(BT): Use this class as framework for all other tests

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
            translation.numpy_vector_zone_translation(**kwargs | {"vector": new_vector})

    @pytest.mark.parametrize("check_shapes", [True, False])
    def test_translation_shape(
        self, np_vector_split: NumpyVectorResults, check_shapes: bool,
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
            translation.numpy_vector_zone_translation(**kwargs | {"translation": new_trans})

    @pytest.mark.parametrize(
        "np_vector_str",
        ["np_vector_aggregation", "np_vector_split"],
    )
    def test_vector_like(self, np_vector_str: NumpyVectorResults, request):
        """Test vector-like arrays (empty in 2nd dim)"""
        np_vector = request.getfixturevalue(np_vector_str)
        new_vector = np.expand_dims(np_vector.vector, 1)
        kwargs = np_vector.input_kwargs(check_shapes=True, check_totals=True)
        result = translation.numpy_vector_zone_translation(**kwargs | {"vector": new_vector})
        np.testing.assert_allclose(result, np_vector.expected_result)

    @pytest.mark.parametrize(
        "np_vector_str",
        ["np_vector_aggregation", "np_vector_split"],
    )
    @pytest.mark.parametrize("check_totals", [True, False])
    def test_translation_correct(self, np_vector_str: NumpyVectorResults, check_totals: bool, request,):
        """Test that aggregation and splitting give correct results

        Also checks that totals are correctly checked.
        """
        np_vector = request.getfixturevalue(np_vector_str)
        kwargs = np_vector.input_kwargs(check_shapes=True, check_totals=check_totals)
        result = translation.numpy_vector_zone_translation(**kwargs)
        np.testing.assert_allclose(result, np_vector.expected_result)
