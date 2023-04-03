# -*- coding: utf-8 -*-
"""Tests for the caf.toolkit.translation module"""
from __future__ import annotations

# Built-Ins
import copy
import dataclasses

from typing import Any
from typing import Optional


# Third Party
import pytest
import numpy as np
import pandas as pd


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
    translation_dtype: Optional[type] = None

    def input_kwargs(
        self,
        check_shapes: bool = True,
        check_totals: bool = True,
    ) -> dict[str, Any]:
        """Return a dictionary of key-word arguments"""
        return {
            "vector": self.vector,
            "translation": self.translation,
            "translation_dtype": self.translation_dtype,
            "check_shapes": check_shapes,
            "check_totals": check_totals,
        }


@dataclasses.dataclass
class NumpyMatrixResults:
    """Collection of I/O data for a numpy matrix translation"""

    mat: np.ndarray
    translation: np.ndarray
    expected_result: np.ndarray

    translation_dtype: Optional[type] = None
    col_translation: Optional[np.ndarray] = None

    def input_kwargs(
        self,
        check_shapes: bool = True,
        check_totals: bool = True,
        force_slow: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        """Return a dictionary of key-word arguments"""
        return {
            "matrix": self.mat,
            "translation": self.translation,
            "col_translation": self.col_translation,
            "translation_dtype": self.translation_dtype,
            "check_shapes": check_shapes,
            "check_totals": check_totals,
            "_force_slow": force_slow,
        } | kwargs


@dataclasses.dataclass
class PandasTranslation:
    """Container for a pandas based translation

    Takes a numpy translation and converts to a standard pandas format
    """

    np_translation: dataclasses.InitVar[np.ndarray]
    translation_from_col: str = "from_zone_id"
    translation_to_col: str = "to_zone_id"
    translation_factors_col: str = "factors"
    df: pd.DataFrame = dataclasses.field(init=False)
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
        df = df.melt(
            id_vars=self.translation_from_col,
            value_name=self.translation_factors_col,
        )
        df[self.translation_from_col] = df[self.translation_from_col].astype(np.int64)
        df[self.translation_to_col] = df[self.translation_to_col].astype(np.int64)
        self.df = df

        # Get the unique from / to lists
        self.unique_from = sorted(self.df[self.translation_from_col].unique().tolist())
        self.unique_to = sorted(self.df[self.translation_to_col].unique().tolist())

    @property
    def from_col(self) -> pd.Series:
        """The data from the "from zone col" of the translation"""
        return self.df[self.translation_from_col]

    @from_col.setter
    def from_col(self, value: pd.Series):
        """Set the "factor zone col" data"""
        self.df[self.translation_from_col] = value

    @property
    def factor_col(self) -> pd.Series:
        """The data from the "to zone col" of the translation"""
        return self.df[self.translation_factors_col]

    @factor_col.setter
    def factor_col(self, value: pd.Series):
        """Set the "factor col" data"""
        self.df[self.translation_factors_col] = value

    def to_kwargs(self) -> dict[str, Any]:
        """Return a dictionary of key-word arguments"""
        return {
            "translation": self.df,
            "translation_from_col": self.translation_from_col,
            "translation_to_col": self.translation_to_col,
            "translation_factors_col": self.translation_factors_col,
        }

    def copy(self) -> PandasTranslation:
        """Make a copy of this class"""
        return copy.deepcopy(self)


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
        check_totals: bool = True,
        vector_infill: float = 0,
        translate_infill: float = 0,
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


@dataclasses.dataclass
class PandasMatrixResults:
    """Collection of I/O data for a pandas matrix translation"""

    np_matrix: dataclasses.InitVar[np.ndarray]
    np_expected_result: dataclasses.InitVar[np.ndarray]
    translation: PandasTranslation

    translation_dtype: Optional[type] = None
    col_translation: Optional[PandasTranslation] = None

    def __post_init__(self, np_matrix: np.ndarray, np_expected_result: np.ndarray):
        """Convert numpy objects to pandas"""
        # Base from / to zones on translation
        self.from_unique_index = self.translation.unique_from
        self.to_unique_index = self.translation.unique_to

        # Input and results
        self.mat = pd.DataFrame(
            data=np_matrix,
            index=self.from_unique_index,
            columns=self.from_unique_index,
        )
        self.expected_result = pd.DataFrame(
            data=np_expected_result,
            index=self.to_unique_index,
            columns=self.to_unique_index,
        )

    def input_kwargs(
        self,
        matrix_infill: float = 0.0,
        translate_infill: float = 0.0,
        check_totals: bool = True,
        **kwargs,
    ) -> dict[str, Any]:
        """Return a dictionary of key-word arguments"""
        kwargs = (
            {
                "matrix": self.mat,
                "translation_dtype": self.translation_dtype,
                "from_unique_index": self.from_unique_index,
                "to_unique_index": self.to_unique_index,
                "matrix_infill": matrix_infill,
                "translate_infill": translate_infill,
                "check_totals": check_totals,
            }
            | self.translation.to_kwargs()
            | kwargs
        )

        if self.col_translation is None:
            return kwargs
        return kwargs | {"col_translation": self.col_translation.df}


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


@pytest.fixture(name="simple_np_int_translation2", scope="class")
def fixture_simple_np_int_translation2() -> np.ndarray:
    """Generate a simple 5 to 3 complete translation array"""
    return np.array(
        [
            [0, 0, 1],
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
            [1, 0, 0],
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


@pytest.fixture(name="simple_pd_int_translation2", scope="class")
def fixture_simple_pd_int_translation2(
    simple_np_int_translation2: np.ndarray,
) -> PandasTranslation:
    """Generate a simple 5 to 3 complete translation array"""
    return PandasTranslation(simple_np_int_translation2)


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
        expected_result=np.array([10, 8, 13]),
    )


# ## PANDAS VECTOR FIXTURES ## #
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


@pytest.fixture(name="pd_incomplete", scope="class")
def fixture_pd_incomplete(
    np_incomplete: NumpyVectorResults,
    incomplete_pd_int_translation: PandasTranslation,
) -> PandasVectorResults:
    """Generate a splitting vector, translation, and results"""
    return PandasVectorResults(
        np_vector=np_incomplete.vector,
        np_expected_result=np_incomplete.expected_result,
        translation=incomplete_pd_int_translation,
    )


# ## NUMPY MATRIX FIXTURES ## #
@pytest.fixture(name="np_matrix_aggregation", scope="class")
def fixture_np_matrix_aggregation(simple_np_int_translation: np.ndarray) -> NumpyMatrixResults:
    """Generate a matrix, translation, and results"""
    mat = np.array(
        [
            [4, 2, 3, 1, 1],
            [2, 8, 3, 6, 6],
            [5, 5, 6, 5, 9],
            [4, 3, 3, 6, 8],
            [8, 4, 8, 8, 1],
        ]
    )
    expected_result = np.array(
        [
            [16, 6, 14],
            [10, 6, 14],
            [19, 11, 23],
        ]
    )
    return NumpyMatrixResults(
        mat=mat,
        translation=simple_np_int_translation,
        expected_result=expected_result,
    )


@pytest.fixture(name="np_matrix_aggregation2", scope="class")
def fixture_np_matrix_aggregation2(
    np_matrix_aggregation: NumpyMatrixResults,
    simple_np_int_translation2: np.ndarray,
) -> NumpyMatrixResults:
    """Generate a matrix, translation, and results"""
    expected_result = np.array(
        [
            [14, 6, 16],
            [14, 6, 10],
            [23, 11, 19],
        ]
    )
    return NumpyMatrixResults(
        mat=np_matrix_aggregation.mat,
        translation=np_matrix_aggregation.translation,
        col_translation=simple_np_int_translation2,
        expected_result=expected_result,
    )


@pytest.fixture(name="np_matrix_split", scope="class")
def fixture_np_matrix_split(
    np_matrix_aggregation: NumpyMatrixResults,
    simple_np_float_translation: np.ndarray,
) -> NumpyMatrixResults:
    """Generate a matrix, translation, and results"""
    expected_result = np.array(
        [
            [9.6875, 11.625, 9.6875],
            [16.875, 23.25, 16.875],
            [9.6875, 11.625, 9.6875],
        ]
    )
    return NumpyMatrixResults(
        mat=np_matrix_aggregation.mat,
        translation=simple_np_float_translation,
        expected_result=expected_result,
    )


@pytest.fixture(name="np_matrix_dtype", scope="class")
def fixture_np_matrix_dtype(simple_np_int_translation: np.ndarray) -> NumpyMatrixResults:
    """Generate an incomplete vector, translation, and results

    Incomplete meaning some demand will be dropped during the translation
    """
    mat = np.array(
        [
            [4.2, 2.4, 3.6, 1.8, 1.1],
            [2.2, 8.4, 3.6, 6.8, 6.1],
            [5.2, 5.4, 6.6, 5.8, 9.1],
            [4.2, 3.4, 3.6, 6.8, 8.1],
            [8.2, 4.4, 8.6, 8.8, 1.1],
        ]
    )
    expected_result = np.array(
        [
            [16, 6, 14],
            [10, 6, 14],
            [19, 11, 23],
        ]
    )
    return NumpyMatrixResults(
        mat=mat,
        translation=simple_np_int_translation,
        translation_dtype=np.int32,
        expected_result=expected_result,
    )


@pytest.fixture(name="np_matrix_incomplete", scope="class")
def fixture_np_matrix_incomplete(
    incomplete_np_int_translation: np.ndarray, np_matrix_aggregation: NumpyMatrixResults
) -> NumpyMatrixResults:
    """Generate an incomplete vector, translation, and results

    Incomplete meaning some demand will be dropped during the translation
    """
    expected_result = np.array(
        [
            [16, 6, 11.2],
            [10, 6, 10.4],
            [14.2, 7.8, 15.96],
        ]
    )
    return NumpyMatrixResults(
        mat=np_matrix_aggregation.mat,
        translation=incomplete_np_int_translation,
        expected_result=expected_result,
    )


# ## PANDAS MATRIX FIXTURES ## #
@pytest.fixture(name="pd_matrix_aggregation", scope="class")
def fixture_pd_matrix_aggregation(
    np_matrix_aggregation: NumpyMatrixResults,
    simple_pd_int_translation: PandasTranslation,
) -> PandasMatrixResults:
    """Generate an aggregation matrix, translation, and results"""
    return PandasMatrixResults(
        np_matrix=np_matrix_aggregation.mat,
        np_expected_result=np_matrix_aggregation.expected_result,
        translation=simple_pd_int_translation,
    )


@pytest.fixture(name="pd_matrix_aggregation2", scope="class")
def fixture_pd_matrix_aggregation2(
    np_matrix_aggregation2: NumpyMatrixResults,
    simple_pd_int_translation: PandasTranslation,
    simple_pd_int_translation2: PandasTranslation,
) -> PandasMatrixResults:
    """Generate an aggregation matrix, translation, and results"""
    return PandasMatrixResults(
        np_matrix=np_matrix_aggregation2.mat,
        np_expected_result=np_matrix_aggregation2.expected_result,
        translation=simple_pd_int_translation,
        col_translation=simple_pd_int_translation2,
    )


@pytest.fixture(name="pd_matrix_split", scope="class")
def fixture_pd_matrix_split(
    np_matrix_split: NumpyMatrixResults,
    simple_pd_float_translation: PandasTranslation,
) -> PandasMatrixResults:
    """Generate an aggregation matrix, translation, and results"""
    return PandasMatrixResults(
        np_matrix=np_matrix_split.mat,
        np_expected_result=np_matrix_split.expected_result,
        translation=simple_pd_float_translation,
    )


@pytest.fixture(name="pd_matrix_dtype", scope="class")
def fixture_pd_matrix_dtype(
    np_matrix_dtype: NumpyMatrixResults,
    simple_pd_int_translation: PandasTranslation,
) -> PandasMatrixResults:
    """Generate an aggregation matrix, translation, and results"""
    return PandasMatrixResults(
        np_matrix=np_matrix_dtype.mat,
        np_expected_result=np_matrix_dtype.expected_result,
        translation_dtype=np_matrix_dtype.translation_dtype,
        translation=simple_pd_int_translation,
    )


@pytest.fixture(name="pd_matrix_incomplete", scope="class")
def fixture_pd_matrix_incomplete(
    np_matrix_incomplete: NumpyMatrixResults,
    incomplete_pd_int_translation: PandasTranslation,
) -> PandasMatrixResults:
    """Generate an aggregation matrix, translation, and results"""
    return PandasMatrixResults(
        np_matrix=np_matrix_incomplete.mat,
        np_expected_result=np_matrix_incomplete.expected_result,
        translation=incomplete_pd_int_translation,
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
        kwargs = np_incomplete.input_kwargs(check_totals=check_totals)
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
        kwargs = np_vector_split.input_kwargs(check_shapes=check_shapes)
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
        kwargs = np_vector_split.input_kwargs(check_shapes=check_shapes)
        with pytest.raises(ValueError, match=msg):
            translation.numpy_vector_zone_translation(**(kwargs | {"translation": new_trans}))

    @pytest.mark.parametrize(
        "np_vector_str",
        ["np_vector_aggregation", "np_vector_split", "np_translation_dtype"],
    )
    def test_vector_like(self, np_vector_str: str, request):
        """Test vector-like arrays (empty in 2nd dim)"""
        np_vector = request.getfixturevalue(np_vector_str)
        new_vector = np.expand_dims(np_vector.vector, 1)
        kwargs = np_vector.input_kwargs()
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
        kwargs = np_vector.input_kwargs(check_totals=check_totals)
        result = translation.numpy_vector_zone_translation(**kwargs)
        np.testing.assert_allclose(result, np_vector.expected_result)


@pytest.mark.usefixtures(
    "pd_vector_aggregation",
    "pd_vector_split",
    "pd_incomplete",
)
class TestPandasVector:
    """Tests for caf.toolkit.translation.pandas_vector_zone_translation"""

    @pytest.mark.parametrize("check_totals", [True, False])
    def test_dropped_totals(self, pd_incomplete: PandasVectorResults, check_totals: bool):
        """Test for total checking with dropped demand"""
        kwargs = pd_incomplete.input_kwargs(check_totals=check_totals)
        if not check_totals:
            result = translation.pandas_vector_zone_translation(**kwargs)
            pd.testing.assert_series_equal(
                result, pd_incomplete.expected_result, check_dtype=False
            )
        else:
            with pytest.raises(ValueError, match="Some values seem to have been dropped"):
                translation.pandas_vector_zone_translation(**kwargs)

    def test_missing_translation_zones(self, pd_vector_split: PandasVectorResults):
        """Check a warning is raised with missing translation from values"""
        # make the bad translation
        new_trans = pd_vector_split.translation.df.copy()
        from_col_name = pd_vector_split.translation.translation_from_col
        from_3_mask = new_trans[from_col_name] == 3
        new_trans = new_trans[~from_3_mask].copy()

        # need to turn off to avoid error
        kwargs = pd_vector_split.input_kwargs(check_totals=False)
        msg = "Some zones in `vector.index` are missing in `translation`"
        with pytest.warns(UserWarning, match=msg):
            translation.pandas_vector_zone_translation(**(kwargs | {"translation": new_trans}))

    def test_missing_unique_zones(self, pd_vector_split: PandasVectorResults):
        """Check a warning is raised if from_unique_zones and the vector disagree"""
        new_from_unq = pd_vector_split.from_unique_index[:-1]
        msg = "zones in `vector.index` have not been defined in `from_unique_zones`"
        with pytest.warns(UserWarning, match=msg):
            translation.pandas_vector_zone_translation(
                **(pd_vector_split.input_kwargs() | {"from_unique_index": new_from_unq})
            )

    @pytest.mark.parametrize("which", ["from", "to", "both"])
    def test_non_unique_index(self, pd_vector_split: PandasVectorResults, which: str):
        """Check that an error is thrown when bad unique index args given"""
        # Build bad arguments
        new_from_unq = pd_vector_split.from_unique_index * 2
        new_to_unq = pd_vector_split.to_unique_index * 2

        if which == "from":
            new_kwargs = {"from_unique_index": new_from_unq}
            msg = "Duplicate values found in from_unique_index"
        elif which == "to":
            new_kwargs = {"to_unique_index": new_to_unq}
            msg = "Duplicate values found in to_unique_index"
        elif which == "both":
            new_kwargs = {
                "from_unique_index": new_from_unq,
                "to_unique_index": new_to_unq,
            }
            msg = "Duplicate values found in from_unique_index"
        else:
            raise ValueError("This shouldn't happen! Debug code.")

        # Run with errors
        with pytest.raises(ValueError, match=msg):
            translation.pandas_vector_zone_translation(
                **(pd_vector_split.input_kwargs() | new_kwargs)
            )

    @pytest.mark.parametrize("dtype1", [np.float64, np.int64, str])
    @pytest.mark.parametrize("dtype2", [np.float64, np.int64, str])
    def test_col_dtypes(
        self,
        pd_vector_split: PandasVectorResults,
        dtype1: type,
        dtype2: type,
    ):
        """Check that correct errors are thrown when types don't match"""
        # Cast types
        new_vector = pd_vector_split.vector.copy()
        new_vector.index = new_vector.index.astype(dtype1)
        new_trans = pd_vector_split.translation.copy()
        new_trans.from_col = new_trans.from_col.astype(dtype2)
        new_kwargs = {"vector": new_vector, "translation": new_trans.df}

        if dtype1 == str and dtype2 == str:
            # Pandas handles strings weirdly (as objects), making this hard to test
            pass
        elif dtype1 == dtype2:
            # Should run normally with matching dtypes
            result = translation.pandas_vector_zone_translation(
                **(pd_vector_split.input_kwargs() | new_kwargs)
            )
            pd.testing.assert_series_equal(
                result,
                pd_vector_split.expected_result,
                check_names=False,
            )
        else:
            # Only throw error when not matching types
            msg = "dtypes of `vector.index` and `translation` in `from_zone_col` must match."
            with pytest.raises(ValueError, match=msg):
                translation.pandas_vector_zone_translation(
                    **(pd_vector_split.input_kwargs() | new_kwargs)
                )

    def test_non_vector(self, pd_vector_split: PandasVectorResults):
        """Test that an error is thrown when a non-vector is passed in"""
        new_vector = pd.DataFrame(pd_vector_split.vector)
        new_vector[1] = new_vector[0].copy()
        with pytest.raises(ValueError, match="must be a pandas.Series"):
            translation.pandas_vector_zone_translation(
                **(pd_vector_split.input_kwargs() | {"vector": new_vector})
            )

    @pytest.mark.parametrize(
        "pd_vector_str",
        ["pd_vector_aggregation", "pd_vector_split"],
    )
    def test_series_like(self, pd_vector_str: str, request):
        """Test vector-like arrays (empty in 2nd dim)"""
        pd_vector = request.getfixturevalue(pd_vector_str)
        new_vector = pd.DataFrame(pd_vector.vector)
        result = translation.pandas_vector_zone_translation(
            **(pd_vector.input_kwargs() | {"vector": new_vector})
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
        result = translation.pandas_vector_zone_translation(
            **pd_vector.input_kwargs(check_totals=check_totals)
        )
        pd.testing.assert_series_equal(result, pd_vector.expected_result)


@pytest.mark.usefixtures(
    "np_matrix_aggregation",
    "np_matrix_aggregation2",
    "np_matrix_split",
    "np_matrix_dtype",
    "np_matrix_incomplete",
)
class TestNumpyMatrix:
    """Tests for caf.toolkit.translation.numpy_matrix_zone_translation"""

    # TODO(BT): Test running out of memory - will need to skip for GitHub

    def test_mismatch_translations(self, np_matrix_aggregation2: NumpyMatrixResults):
        """Check an error is raised with a non-square matrix given"""
        col_trans = np_matrix_aggregation2.col_translation
        col_trans = np.delete(col_trans, 0, axis=0)
        msg = "Row and column translations are not the same shape"

        with pytest.raises(ValueError, match=msg):
            translation.numpy_matrix_zone_translation(
                **(np_matrix_aggregation2.input_kwargs() | {"col_translation": col_trans})
            )

    def test_bad_translation(self, np_matrix_aggregation2: NumpyMatrixResults):
        """Check an error is raised with a non-square matrix given"""
        new_trans = np_matrix_aggregation2.translation
        col_trans = np_matrix_aggregation2.col_translation
        new_kwargs = {
            "translation": np.delete(new_trans, 0, axis=0),
            "col_translation": np.delete(col_trans, 0, axis=0),
        }
        msg = "Translation rows needs to match matrix rows"

        with pytest.raises(ValueError, match=msg):
            translation.numpy_matrix_zone_translation(
                **(np_matrix_aggregation2.input_kwargs() | new_kwargs)
            )

    def test_bad_matrix(self, np_matrix_split: NumpyMatrixResults):
        """Check an error is raised with a non-square matrix given"""
        new_mat = np_matrix_split.mat
        new_mat = np.delete(new_mat, 0, axis=0)
        msg = "given matrix is not square"

        with pytest.raises(ValueError, match=msg):
            translation.numpy_matrix_zone_translation(
                **(np_matrix_split.input_kwargs() | {"matrix": new_mat})
            )

    @pytest.mark.parametrize("check_totals", [True, False])
    def test_dropped_totals(
        self, np_matrix_incomplete: NumpyMatrixResults, check_totals: bool
    ):
        """Test for total checking with dropped demand"""
        kwargs = np_matrix_incomplete.input_kwargs(check_totals=check_totals)
        if not check_totals:
            result = translation.numpy_matrix_zone_translation(**kwargs)
            np.testing.assert_allclose(result, np_matrix_incomplete.expected_result)
        else:
            with pytest.raises(ValueError, match="Some values seem to have been dropped"):
                translation.numpy_matrix_zone_translation(**kwargs)

    @pytest.mark.parametrize(
        "np_matrix_str",
        [
            "np_matrix_aggregation",
            "np_matrix_aggregation2",
            "np_matrix_split",
            "np_matrix_dtype",
        ],
    )
    @pytest.mark.parametrize("force_slow", [True, False])
    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_slow_translation(
        self,
        np_matrix_str: str,
        force_slow: bool,
        request,
    ):
        """Test translation works as expected

        Tests the matrix aggregation, using 2 different translations, and
        translation splitting.
        """
        np_mat = request.getfixturevalue(np_matrix_str)
        result = translation.numpy_matrix_zone_translation(
            **np_mat.input_kwargs(force_slow=force_slow, chunk_size=2)
        )
        np.testing.assert_allclose(result, np_mat.expected_result)

    @pytest.mark.parametrize(
        "np_matrix_str",
        [
            "np_matrix_aggregation",
            "np_matrix_aggregation2",
            "np_matrix_split",
            "np_matrix_dtype",
        ],
    )
    @pytest.mark.parametrize("check_totals", [True, False])
    def test_translation_correct(
        self,
        np_matrix_str: str,
        check_totals: bool,
        request,
    ):
        """Test translation works as expected

        Tests the matrix aggregation, using 2 different translations, and
        translation splitting.
        """
        np_mat = request.getfixturevalue(np_matrix_str)
        result = translation.numpy_matrix_zone_translation(
            **np_mat.input_kwargs(check_totals=check_totals)
        )
        np.testing.assert_allclose(result, np_mat.expected_result)


@pytest.mark.usefixtures(
    "pd_matrix_aggregation",
    "pd_matrix_aggregation2",
    "pd_matrix_split",
    "pd_matrix_dtype",
    "pd_matrix_incomplete",
)
class TestPandasMatrix:
    """Tests for caf.toolkit.translation.pandas_matrix_zone_translation"""

    @pytest.mark.parametrize("check_totals", [True, False])
    def test_dropped_totals(
        self, pd_matrix_incomplete: PandasMatrixResults, check_totals: bool
    ):
        """Test for total checking with dropped demand"""
        kwargs = pd_matrix_incomplete.input_kwargs(check_totals=check_totals)
        if not check_totals:
            result = translation.pandas_matrix_zone_translation(**kwargs)
            pd.testing.assert_frame_equal(
                result, pd_matrix_incomplete.expected_result, check_dtype=False
            )
        else:
            with pytest.raises(ValueError, match="Some values seem to have been dropped"):
                translation.pandas_matrix_zone_translation(**kwargs)

    def test_missing_unique_zones(self, pd_matrix_split: PandasMatrixResults):
        """Check a warning is raised if from_unique_zones and the vector disagree"""
        new_from_unq = pd_matrix_split.from_unique_index[:-1]
        msg = "zones in `matrix.index` have not been defined in `from_unique_zones`"
        with pytest.warns(UserWarning, match=msg):
            translation.pandas_matrix_zone_translation(
                **(pd_matrix_split.input_kwargs() | {"from_unique_index": new_from_unq})
            )

    @pytest.mark.parametrize("which", ["from", "to", "both"])
    def test_non_unique_index(self, pd_matrix_split: PandasMatrixResults, which: str):
        """Check that an error is thrown when bad unique index args given"""
        # Build bad arguments
        new_from_unq = pd_matrix_split.from_unique_index * 2
        new_to_unq = pd_matrix_split.to_unique_index * 2

        if which == "from":
            new_kwargs = {"from_unique_index": new_from_unq}
            msg = "Duplicate values found in from_unique_index"
        elif which == "to":
            new_kwargs = {"to_unique_index": new_to_unq}
            msg = "Duplicate values found in to_unique_index"
        elif which == "both":
            new_kwargs = {
                "from_unique_index": new_from_unq,
                "to_unique_index": new_to_unq,
            }
            msg = "Duplicate values found in from_unique_index"
        else:
            raise ValueError("This shouldn't happen! Debug code.")

        # Run with errors
        with pytest.raises(ValueError, match=msg):
            translation.pandas_matrix_zone_translation(
                **(pd_matrix_split.input_kwargs() | new_kwargs)
            )

    @pytest.mark.parametrize("mat_dtype", [np.float64, np.int64, str])
    @pytest.mark.parametrize("row_translation_dtype", [np.float64, np.int64, str])
    def test_col_dtypes(
        self,
        pd_matrix_split: PandasMatrixResults,
        mat_dtype: type,
        row_translation_dtype: type,
        col_translation_dtype: type = np.float64,
    ):
        """Check that correct errors are thrown when types don't match"""
        # Cast types
        new_matrix = pd_matrix_split.mat.copy()
        new_matrix.index = new_matrix.index.astype(mat_dtype)
        new_matrix.columns = new_matrix.columns.astype(mat_dtype)

        new_trans = pd_matrix_split.translation.copy()
        new_trans.from_col = new_trans.from_col.astype(row_translation_dtype)
        col_trans = pd_matrix_split.translation.copy()
        col_trans.from_col = col_trans.from_col.astype(col_translation_dtype)
        new_kwargs = {
            "matrix": new_matrix,
            "translation": new_trans.df,
            "col_translation": col_trans.df,
        }

        if mat_dtype == str and row_translation_dtype == str:
            # Pandas handles strings weirdly (as objects), making this hard to test
            pass
        elif mat_dtype == row_translation_dtype == col_translation_dtype:
            # Should run normally with matching dtypes
            result = translation.pandas_matrix_zone_translation(
                **(pd_matrix_split.input_kwargs() | new_kwargs)
            )
            pd.testing.assert_frame_equal(
                result,
                pd_matrix_split.expected_result,
                check_names=False,
            )
        elif mat_dtype == row_translation_dtype:
            # If the row matches, but the column doesn't
            msg = "dtypes of `matrix.columns` and `col_translation` in `from_zone_col` must match."
            with pytest.raises(ValueError, match=msg):
                translation.pandas_matrix_zone_translation(
                    **(pd_matrix_split.input_kwargs() | new_kwargs)
                )
        else:
            # If row row and data dtypes don't match
            msg = "dtypes of `matrix.index` and `translation` in `from_zone_col` must match."
            with pytest.raises(ValueError, match=msg):
                translation.pandas_matrix_zone_translation(
                    **(pd_matrix_split.input_kwargs() | new_kwargs)
                )

    @pytest.mark.parametrize(
        "pd_matrix_str",
        [
            "pd_matrix_aggregation",
            "pd_matrix_aggregation2",
            "pd_matrix_split",
            "pd_matrix_dtype",
        ],
    )
    @pytest.mark.parametrize("force_slow", [True, False])
    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_slow_translation(
        self,
        pd_matrix_str: str,
        force_slow: bool,
        request,
    ):
        """Test translation works as expected

        Tests the matrix aggregation, using 2 different translations, and
        translation splitting.
        """
        pd_mat = request.getfixturevalue(pd_matrix_str)
        result = translation.pandas_matrix_zone_translation(
            **pd_mat.input_kwargs(_force_slow=force_slow, chunk_size=3)
        )
        pd.testing.assert_frame_equal(result, pd_mat.expected_result)

    @pytest.mark.parametrize(
        "pd_matrix_str",
        [
            "pd_matrix_aggregation",
            "pd_matrix_aggregation2",
            "pd_matrix_split",
            "pd_matrix_dtype",
        ],
    )
    @pytest.mark.parametrize("check_totals", [True, False])
    def test_translation_correct(
        self,
        pd_matrix_str: str,
        check_totals: bool,
        request,
    ):
        """Test translation works as expected

        Tests the matrix aggregation, using 2 different translations, and
        translation splitting.
        """
        pd_mat = request.getfixturevalue(pd_matrix_str)
        result = translation.pandas_matrix_zone_translation(
            **pd_mat.input_kwargs(check_totals=check_totals)
        )
        pd.testing.assert_frame_equal(result, pd_mat.expected_result)
