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
# TODO(BT): Pandas vector will build on this
@pytest.fixture(name="numpy_vector_aggregation", scope="class")
def fixture_numpy_vector_aggregation() -> NumpyVectorResults:
    """Generate an aggregation vector, translation, and results"""
    trans = np.array(
        [
            [1, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 1],
        ]
    )

    vector = np.array([8, 2, 8, 8, 5])

    expected_result = np.array([10, 8, 13])

    return NumpyVectorResults(
        vector=vector,
        translation=trans,
        expected_result=expected_result,
    )


@pytest.fixture(name="numpy_vector_split", scope="class")
def fixture_numpy_vector_split() -> NumpyVectorResults:
    """Generate a splitting vector, translation, and results"""
    trans = np.array(
        [
            [0.5, 0.0, 0.5],
            [0.25, 0.5, 0.25],
            [0, 1, 0],
            [0.5, 0.0, 0.5],
            [0.25, 0.5, 0.25],
        ]
    )

    vector = np.array([8, 2, 8, 8, 5])

    expected_result = np.array([9.75, 11.5, 9.75])

    return NumpyVectorResults(
        vector=vector,
        translation=trans,
        expected_result=expected_result,
    )


# # # TESTS # # #
@pytest.mark.usefixtures(
    "numpy_vector_aggregation",
    "numpy_vector_split",
)
class TestNumpyVector:
    """Tests for caf.toolkit.translation.numpy_vector_zone_translation"""

    # TODO(BT): Check totals, and check translation dtype works
    # TODO(BT): Use this class as framework for all other tests

    @pytest.mark.parametrize("check_shapes", [True, False])
    def test_non_vector(self, numpy_vector_split: NumpyVectorResults, check_shapes: bool):
        """Test for error when non-vector given"""
        # Convert vector to matrix
        new_vector = numpy_vector_split.vector
        new_vector = np.broadcast_to(new_vector, (new_vector.shape[0], new_vector.shape[0]))

        # Set expected error message
        if check_shapes:
            msg = "not a vector"
        else:
            msg = "was there a shape mismatch?"

        # Call with expected error
        kwargs = numpy_vector_split.input_kwargs(check_shapes=check_shapes, check_totals=True)
        with pytest.raises(ValueError, match=msg):
            translation.numpy_vector_zone_translation(**kwargs | {"vector": new_vector})

    @pytest.mark.parametrize("check_shapes", [True, False])
    def test_translation_shape(
        self, numpy_vector_split: NumpyVectorResults, check_shapes: bool
    ):
        """Test for error when wrong shape translation"""
        # Convert vector to matrix
        new_trans = numpy_vector_split.translation
        new_trans = np.vstack([new_trans, new_trans])

        # Set expected error message
        if check_shapes:
            msg = "translation does not have the correct number of rows"
        else:
            msg = "was there a shape mismatch?"

        # Call with expected error
        kwargs = numpy_vector_split.input_kwargs(check_shapes=check_shapes, check_totals=True)
        with pytest.raises(ValueError, match=msg):
            translation.numpy_vector_zone_translation(**kwargs | {"translation": new_trans})

    @pytest.mark.parametrize(
        "numpy_vector_str",
        ["numpy_vector_aggregation", "numpy_vector_split"],
    )
    def test_vector_like(self, numpy_vector_str: NumpyVectorResults, request):
        """Test vector-like arrays (empty in 2nd dim)"""
        numpy_vector = request.getfixturevalue(numpy_vector_str)
        new_vector = np.expand_dims(numpy_vector.vector, 1)
        kwargs = numpy_vector.input_kwargs(check_shapes=True, check_totals=True)
        result = translation.numpy_vector_zone_translation(**kwargs | {"vector": new_vector})
        np.testing.assert_allclose(result, numpy_vector.expected_result)

    @pytest.mark.parametrize(
        "numpy_vector_str",
        ["numpy_vector_aggregation", "numpy_vector_split"],
    )
    def test_translation_correct(self, numpy_vector_str: NumpyVectorResults, request):
        """Test that aggregation and splitting"""
        numpy_vector = request.getfixturevalue(numpy_vector_str)
        kwargs = numpy_vector.input_kwargs(check_shapes=True, check_totals=True)
        result = translation.numpy_vector_zone_translation(**kwargs)
        np.testing.assert_allclose(result, numpy_vector.expected_result)
