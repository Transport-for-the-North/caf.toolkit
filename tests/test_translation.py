# -*- coding: utf-8 -*-
"""Tests for the caf.toolkit.translation module"""
from __future__ import annotations

# Built-Ins
import copy
import dataclasses
import pathlib
import sys
from typing import Any, Optional

# Third Party
import numpy as np
import pandas as pd
import pytest

# Local Imports
from caf.toolkit import io
from caf.toolkit import pandas_utils as pd_utils
from caf.toolkit import translation

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
        **kwargs,
    ) -> dict[str, Any]:
        """Return a dictionary of key-word arguments"""
        return {
            "matrix": self.mat,
            "translation": self.translation,
            "col_translation": self.col_translation,
            "translation_dtype": self.translation_dtype,
            "check_shapes": check_shapes,
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
    def to_col(self) -> pd.Series:
        """The data from the "to zone col" of the translation"""
        return self.df[self.translation_to_col]

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

    def create_dummy_rows(self):
        """Create some dummy rows that can be attached."""
        ret_val = pd.Series(data=self.to_col.unique(), name=self.translation_to_col)
        ret_val = pd.DataFrame(ret_val)
        ret_val[self.translation_from_col] = self.from_col.max() + 1
        ret_val[self.translation_factors_col] = 0
        ret_val.loc[0, self.translation_factors_col] = 1
        return ret_val.reindex(
            columns=[
                self.translation_from_col,
                self.translation_to_col,
                self.translation_factors_col,
            ]
        )


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
        self.expected_result.index.name = "to_zone_id"
        self.expected_result.index += 1

        # Base from / to zones on translation
        self.from_unique_index = self.translation.unique_from
        self.to_unique_index = self.translation.unique_to

    def input_kwargs(self, check_totals: bool = True) -> dict[str, Any]:
        """Return a dictionary of key-word arguments"""
        return {
            "vector": self.vector,
            "translation_dtype": self.translation_dtype,
            "check_totals": check_totals,
        } | self.translation.to_kwargs()


@dataclasses.dataclass
class PandasMultiVectorResults:
    """Collection of I/O data for a pandas multi-vector translation"""

    np_vector: dataclasses.InitVar[np.ndarray]
    np_expected_result: dataclasses.InitVar[np.ndarray]
    translation: PandasTranslation
    translation_dtype: Optional[np.dtype] = None

    vector: pd.DataFrame = dataclasses.field(init=False)
    expected_result: pd.DataFrame = dataclasses.field(init=False)
    from_unique_index: list[Any] = dataclasses.field(init=False)
    to_unique_index: list[Any] = dataclasses.field(init=False)

    n_cols = 10

    def __post_init__(self, np_vector: np.ndarray, np_expected_result: np.ndarray):
        """Convert numpy objects to pandas"""
        # Input and results
        multi_vector_data = np.tile(np_vector, (self.n_cols, 1)).T
        self.vector = pd.DataFrame(data=multi_vector_data)
        self.vector.index += 1
        # self.vector.index.names = ['to_zone_id']

        multi_vector_res = np.tile(np_expected_result, (self.n_cols, 1)).T
        self.expected_result = pd.DataFrame(data=multi_vector_res)
        self.expected_result.index += 1
        self.expected_result.index.names = ["to_zone_id"]

        # Base from / to zones on translation
        self.from_unique_index = self.translation.unique_from
        self.to_unique_index = self.translation.unique_to

    def input_kwargs(
        self,
        check_totals: bool = True,
    ) -> dict[str, Any]:
        """Return a dictionary of key-word arguments"""
        return {
            "vector": self.vector,
            "translation_dtype": self.translation_dtype,
            "check_totals": check_totals,
        } | self.translation.to_kwargs()

    def get_similar_vector(self, data: np.ndarray):
        """Create a vector in the same shape as this one from a 1d array."""
        multi_vector_data = np.tile(data, (self.n_cols, 1)).T
        return pd.DataFrame(data=multi_vector_data)


@dataclasses.dataclass
class PandasMultiVectorSeriesResults(PandasMultiVectorResults):
    """Collection of I/O data for a pandas multi-vector Series translation"""

    np_vector: dataclasses.InitVar[np.ndarray]
    np_expected_result: dataclasses.InitVar[np.ndarray]
    translation: PandasTranslation
    translation_dtype: Optional[np.dtype] = None

    vector: pd.Series = dataclasses.field(init=False)
    expected_result: pd.Series = dataclasses.field(init=False)
    from_unique_index: list[Any] = dataclasses.field(init=False)
    to_unique_index: list[Any] = dataclasses.field(init=False)

    n_cols = 1

    def __post_init__(self, np_vector: np.ndarray, np_expected_result: np.ndarray):
        """Convert numpy objects to pandas"""
        super().__post_init__(np_vector, np_expected_result)
        self.vector = self.vector.iloc[:, 0]
        self.expected_result = self.expected_result.iloc[:, 0]

    def get_similar_vector(self, data: np.ndarray):
        """Create a vector in the same shape as this one from a 1d array."""
        return super().get_similar_vector(data).iloc[:, 0]


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

        if self.col_translation is None:
            self.col_translation = self.translation.copy()

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
        self.expected_result.index.names = ["to_zone_id"]
        self.expected_result.columns.names = ["to_zone_id"]

    def input_kwargs(
        self,
        check_totals: bool = True,
        **kwargs,
    ) -> dict[str, Any]:
        """Return a dictionary of key-word arguments"""
        kwargs = (
            {
                "matrix": self.mat,
                "translation_dtype": self.translation_dtype,
                "check_totals": check_totals,
            }
            | self.translation.to_kwargs()
            | kwargs
        )

        if self.col_translation is None:
            return kwargs
        return kwargs | {"col_translation": self.col_translation.df}


@dataclasses.dataclass
class PandasLongMatrixResults:
    """Collection of I/O data for a pandas matrix translation"""

    wide_results: dataclasses.InitVar[PandasMatrixResults]
    index_col_1_name: str = "production"
    index_col_2_name: str = "attraction"
    values_col: str = "values"
    index_col_1_out_name: Optional[str] = None
    index_col_2_out_name: Optional[str] = None

    def __post_init__(self, wide_results: PandasMatrixResults):
        """Produce the new expected input and outputs."""
        if self.index_col_1_out_name is None:
            self.index_col_1_out_name = self.index_col_1_name
        if self.index_col_2_out_name is None:
            self.index_col_2_out_name = self.index_col_2_name

        self.df = pd_utils.wide_to_long_infill(
            df=wide_results.mat,
        )

        self.expected_result = pd_utils.wide_to_long_infill(
            df=wide_results.expected_result,
        )
        self.expected_result.index.names = [self.index_col_1_name, self.index_col_2_name]

        # Misc args to carry over
        self.translation_dtype = wide_results.translation_dtype
        self.translation = wide_results.translation
        self.col_translation = wide_results.col_translation

    def update_output_cols(self, index_col_1_out_name: str, index_col_2_out_name: str) -> None:
        """Update the expected result alongside this."""
        renamed = self.expected_result.copy()
        renamed.index.names = [index_col_1_out_name, index_col_2_out_name]

        # self.index_col_1_out_name = index_col_1_out_name
        # self.index_col_2_out_name = index_col_2_out_name
        return renamed

    def input_kwargs(
        self,
        check_totals: bool = True,
        **kwargs,
    ) -> dict[str, Any]:
        """Return a dictionary of key-word arguments"""
        kwargs = (
            {
                "matrix": self.df,
                "index_col_1_name": self.index_col_1_name,
                "index_col_2_name": self.index_col_2_name,
                "values_col": self.values_col,
                "index_col_1_out_name": self.index_col_1_out_name,
                "index_col_2_out_name": self.index_col_2_out_name,
                "translation_dtype": self.translation_dtype,
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


@pytest.fixture(name="pd_multi_vector_aggregation", scope="class")
def fixture_pd_multi_vector_aggregation(
    np_vector_aggregation: NumpyVectorResults,
    simple_pd_int_translation: PandasTranslation,
) -> PandasVectorResults:
    """Generate an aggregation vector, translation, and results"""
    return PandasMultiVectorResults(
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


@pytest.fixture(name="pd_multi_vector_split", scope="class")
def fixture_pd_multi_vector_split(
    np_vector_split: NumpyVectorResults,
    simple_pd_float_translation: PandasTranslation,
) -> PandasVectorResults:
    """Generate a splitting vector, translation, and results"""
    return PandasMultiVectorResults(
        np_vector=np_vector_split.vector,
        np_expected_result=np_vector_split.expected_result,
        translation=simple_pd_float_translation,
    )


@pytest.fixture(name="pd_multi_vector_series", scope="class")
def fixture_pd_multi_vector_series(
    np_vector_split: NumpyVectorResults,
    simple_pd_float_translation: PandasTranslation,
) -> PandasVectorResults:
    """Generate a splitting vector, translation, and results for Series."""
    return PandasMultiVectorSeriesResults(
        np_vector=np_vector_split.vector,
        np_expected_result=np_vector_split.expected_result,
        translation=simple_pd_float_translation,
    )


@pytest.fixture(name="pd_multi_vector_multiindex", scope="function")
def fixture_pd_multi_vector_multiindex(
    pd_multi_vector_split: PandasMultiVectorResults,
) -> PandasMultiVectorResults:
    """Generate a splitting vector, translation, and results"""
    return pd_multi_vector_split


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


@pytest.fixture(name="pd_multi_incomplete", scope="class")
def fixture_pd_multi_incomplete(
    np_incomplete: NumpyVectorResults,
    incomplete_pd_int_translation: PandasTranslation,
) -> PandasVectorResults:
    """Generate a splitting vector, translation, and results"""
    return PandasMultiVectorResults(
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
        ],
        dtype=np.int32,
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


# ## PANDAS LONG MATRIX FIXTURES ## #
@pytest.fixture(name="pd_long_matrix_aggregation", scope="class")
def fixture_pd_long_matrix_aggregation(
    pd_matrix_aggregation: PandasMatrixResults,
) -> PandasLongMatrixResults:
    """Convert fixture to long format."""
    return PandasLongMatrixResults(wide_results=pd_matrix_aggregation)


@pytest.fixture(name="pd_long_matrix_aggregation2", scope="class")
def fixture_pd_long_matrix_aggregation2(
    pd_matrix_aggregation2: PandasMatrixResults,
) -> PandasLongMatrixResults:
    """Convert fixture to long format."""
    return PandasLongMatrixResults(wide_results=pd_matrix_aggregation2)


@pytest.fixture(name="pd_long_matrix_split", scope="class")
def fixture_pd_long_matrix_split(
    pd_matrix_split: PandasMatrixResults,
) -> PandasLongMatrixResults:
    """Convert fixture to long format."""
    return PandasLongMatrixResults(wide_results=pd_matrix_split)


@pytest.fixture(name="pd_long_matrix_dtype", scope="class")
def fixture_pd_long_matrix_dtype(
    pd_matrix_dtype: PandasMatrixResults,
) -> PandasLongMatrixResults:
    """Convert fixture to long format."""
    return PandasLongMatrixResults(wide_results=pd_matrix_dtype)


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
    "pd_multi_vector_aggregation",
    "pd_multi_vector_split",
    "pd_multi_incomplete",
    "pd_vector_aggregation",
    "pd_vector_split",
)
class TestPandasMultiVector:
    """Tests for caf.toolkit.translation.pandas_vector_zone_translation"""

    @pytest.mark.parametrize("check_totals", [True, False])
    def test_dropped_totals(
        self, pd_multi_incomplete: PandasMultiVectorResults, check_totals: bool
    ):
        """Test for total checking with dropped demand"""
        kwargs = pd_multi_incomplete.input_kwargs(check_totals=check_totals)
        if not check_totals:
            result = translation.pandas_vector_zone_translation(**kwargs)
            pd.testing.assert_frame_equal(
                result, pd_multi_incomplete.expected_result, check_dtype=False
            )
        else:
            msg = "Some values seem to have been dropped"
            with pytest.warns(UserWarning, match=msg):
                translation.pandas_vector_zone_translation(**kwargs)

    @pytest.mark.parametrize(
        "pd_vector_str",
        [
            "pd_multi_vector_aggregation",
            "pd_multi_vector_split",
            "pd_multi_vector_series",
            "pd_vector_aggregation",
            "pd_vector_split",
        ],
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
        pd_vector: PandasMultiVectorResults | PandasVectorResults = request.getfixturevalue(
            pd_vector_str
        )
        result = translation.pandas_vector_zone_translation(
            **pd_vector.input_kwargs(check_totals=check_totals)
        )
        if isinstance(pd_vector.expected_result, pd.DataFrame):
            pd.testing.assert_frame_equal(result, pd_vector.expected_result)
        else:
            pd.testing.assert_series_equal(result, pd_vector.expected_result)

    @pytest.mark.parametrize(
        "pd_vector_str",
        ["pd_multi_vector_aggregation", "pd_multi_vector_split", "pd_multi_vector_series"],
    )
    def test_additional_vector_index(self, pd_vector_str: str, request):
        """Check a warning is raised when there are missing translation indexes."""
        pd_vector: PandasMultiVectorResults = request.getfixturevalue(pd_vector_str)

        # Add some additional data to the vector
        additional_rows = pd_vector.get_similar_vector(np.arange(5))
        new_vector = pd_vector.vector.copy()
        new_vector = pd.concat([new_vector, additional_rows], ignore_index=True)
        new_vector.index += 1

        # Check that an error is thrown and the additional values dropped
        msg = "Some zones in `vector.index` have not been defined in `translation`."
        with pytest.warns(UserWarning, match=msg):
            kwargs = pd_vector.input_kwargs(check_totals=False)
            result = translation.pandas_vector_zone_translation(
                **(kwargs | {"vector": new_vector})
            )

        if isinstance(pd_vector.expected_result, pd.DataFrame):
            pd.testing.assert_frame_equal(result, pd_vector.expected_result)
        else:
            pd.testing.assert_series_equal(result, pd_vector.expected_result)

    @pytest.mark.parametrize(
        "pd_vector_str",
        [
            "pd_multi_vector_aggregation",
            "pd_multi_vector_split",
            "pd_multi_vector_series",
            "pd_vector_aggregation",
            "pd_vector_split",
        ],
    )
    def test_additional_translation_index(self, pd_vector_str: str, request):
        """Check that additional translation values are ignored."""
        pd_vector: PandasMultiVectorResults | PandasVectorResults = request.getfixturevalue(
            pd_vector_str
        )

        # Add some additional data to the translation
        new_rows = pd_vector.translation.create_dummy_rows()
        new_trans = pd_vector.translation.df.copy()
        new_trans = pd.concat([new_trans, new_rows], ignore_index=True)

        # Check that the translation still works as before
        result = translation.pandas_vector_zone_translation(
            **(pd_vector.input_kwargs() | {"translation": new_trans})
        )

        if isinstance(pd_vector.expected_result, pd.DataFrame):
            pd.testing.assert_frame_equal(result, pd_vector.expected_result, check_dtype=False)
        else:
            pd.testing.assert_series_equal(
                result, pd_vector.expected_result, check_dtype=False
            )

    @pytest.mark.parametrize(
        "pd_vector_str",
        [
            "pd_multi_vector_aggregation",
            "pd_multi_vector_split",
            "pd_multi_vector_series",
            "pd_vector_aggregation",
            "pd_vector_split",
        ],
    )
    def test_check_allow_similar_types(
        self,
        pd_vector_str: str,
        request,
    ):
        """Test that similar types are allowed in translation and data."""
        pd_vector: PandasMultiVectorResults | PandasVectorResults = request.getfixturevalue(
            pd_vector_str
        )
        new_trans = pd_vector.translation.copy()
        new_trans.from_col = new_trans.from_col.astype(np.int32)
        result = translation.pandas_vector_zone_translation(
            **(pd_vector.input_kwargs() | {"translation": new_trans.df})
        )

        if isinstance(pd_vector.expected_result, pd.DataFrame):
            pd.testing.assert_frame_equal(result, pd_vector.expected_result)
        else:
            pd.testing.assert_series_equal(result, pd_vector.expected_result)

    def test_indexing_error(self, pd_multi_vector_multiindex: PandasMultiVectorResults):
        """Check that an error is thrown when the index is not unique."""
        new_vector = pd_multi_vector_multiindex.vector.copy()
        new_vector["wrong_1"] = new_vector.index
        new_vector["wrong_2"] = new_vector.index
        new_vector.set_index(["wrong_1", "wrong_2"], inplace=True)
        with pytest.raises(ValueError, match="The input vector is MultiIndexed"):
            translation.pandas_vector_zone_translation(
                **(pd_multi_vector_multiindex.input_kwargs() | {"vector": new_vector})
            )

    def test_multiindex(self, pd_multi_vector_multiindex: PandasMultiVectorResults):
        """Test that a multiindex is allowed."""
        # Setup
        new_vector = pd_multi_vector_multiindex.vector.copy()
        new_vector.index.names = ["from_zone_id"]
        factors = [1, 2, 3, 2, 5, 3, 6, 8, 2, 3]
        multiindex = pd.MultiIndex.from_product(
            [new_vector.index, ["A", "B"]], names=["from_zone_id", "extra_seg"]
        )
        multi_vector = new_vector.mul(pd.Series(data=factors, index=multiindex), axis="index")

        # Expect a multiindex warning
        with pytest.warns(UserWarning, match="input vector is MultiIndexed"):
            result = translation.pandas_vector_zone_translation(
                **(pd_multi_vector_multiindex.input_kwargs() | {"vector": multi_vector})
            )

        # Calculate the expected result
        expected = pd.DataFrame(
            {
                0: {
                    (1, "A"): 32.0,
                    (1, "B"): 44.75,
                    (2, "A"): 48.0,
                    (2, "B"): 33.5,
                    (3, "A"): 32.0,
                    (3, "B"): 44.75,
                }
            }
        )
        expected[1, 2, 3, 4, 5, 6, 7, 8, 9] = expected[0]
        expected.index.names = ["to_zone_id", "extra_seg"]
        assert expected.equals(result)


@pytest.mark.usefixtures(
    "np_matrix_aggregation",
    "np_matrix_aggregation2",
    "np_matrix_split",
    "np_matrix_dtype",
    "np_matrix_incomplete",
)
class TestNumpyMatrix:
    """Tests for caf.toolkit.translation.numpy_matrix_zone_translation"""

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
            msg = "Some values seem to have been dropped"
            with pytest.warns(UserWarning, match=msg):
                translation.numpy_matrix_zone_translation(**kwargs)

    @pytest.mark.parametrize(
        "np_matrix_str, check_totals",
        [
            ("np_matrix_aggregation", True),
            ("np_matrix_aggregation", False),
            ("np_matrix_aggregation2", True),
            ("np_matrix_aggregation2", False),
            ("np_matrix_split", True),
            ("np_matrix_split", False),
            ("np_matrix_dtype", False),
        ],
    )
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


@pytest.mark.usefixtures("pd_matrix_incomplete")
class TestPandasMatrixEdges:
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
            msg = "Some values seem to have been dropped"
            with pytest.warns(UserWarning, match=msg):
                translation.pandas_matrix_zone_translation(**kwargs)


@pytest.mark.parametrize(
    "pd_matrix_str, check_totals",
    [
        ("pd_matrix_aggregation", True),
        ("pd_matrix_aggregation", False),
        ("pd_matrix_aggregation2", True),
        ("pd_matrix_aggregation2", False),
        ("pd_matrix_split", True),
        ("pd_matrix_split", False),
        ("pd_matrix_dtype", False),
    ],
)
class TestPandasMatrixParams:
    """Tests for caf.toolkit.translation.pandas_matrix_zone_translation"""

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
        pd_mat: PandasMatrixResults = request.getfixturevalue(pd_matrix_str)
        result = translation.pandas_matrix_zone_translation(
            **pd_mat.input_kwargs(check_totals=check_totals)
        )
        pd.testing.assert_frame_equal(result, pd_mat.expected_result)

    def test_additional_index(self, pd_matrix_str: str, check_totals: bool, request):
        """Check a warning is raised if no translation exists for an index value."""
        pd_mat: PandasMatrixResults = request.getfixturevalue(pd_matrix_str)

        # Make new row to add
        add_df = pd.DataFrame(
            data=np.expand_dims(np.zeros(pd_mat.mat.shape[0]), 0),
            columns=pd_mat.mat.columns,
            index=[pd_mat.mat.index.max() + 1],
        )
        new_df = pd.concat([pd_mat.mat, add_df])

        # Expect the error
        msg = "zones in `matrix.index` have not been defined in `row_translation`"
        with pytest.warns(UserWarning, match=msg):
            translation.pandas_matrix_zone_translation(
                **pd_mat.input_kwargs(check_totals=check_totals) | {"matrix": new_df}
            )

    def test_additional_column(self, pd_matrix_str: str, check_totals: bool, request):
        """Check a warning is raised if no translation exists for a column value."""
        pd_mat: PandasMatrixResults = request.getfixturevalue(pd_matrix_str)

        # Add an additional columns
        new_df = pd_mat.mat.copy()
        new_df[pd_mat.mat.columns.max() + 1] = np.zeros(pd_mat.mat.shape[1])

        # Expect the error
        msg = "zones in `matrix.columns` have not been defined in `col_translation`"
        with pytest.warns(UserWarning, match=msg):
            translation.pandas_matrix_zone_translation(
                **pd_mat.input_kwargs(check_totals=check_totals) | {"matrix": new_df}
            )

    @pytest.mark.parametrize("row", [True, False])
    def test_check_allow_similar_types(
        self,
        pd_matrix_str: str,
        row: bool,
        check_totals: bool,
        request,
    ):
        """Test that similar types are allowed in translation and data."""
        pd_mat = request.getfixturevalue(pd_matrix_str)

        # Change the dtype of the row / col
        if row:
            new_trans = pd_mat.translation.copy()
            keyword = "translation"
        else:
            new_trans = pd_mat.col_translation.copy()
            keyword = "col_translation"

        # Run the translation
        new_trans.from_col = new_trans.from_col.astype(np.int32)
        result = translation.pandas_matrix_zone_translation(
            **(pd_mat.input_kwargs(check_totals=check_totals) | {keyword: new_trans.df})
        )

        # Need to enforce types so this works in linux
        if sys.platform.startswith("linux"):
            if any(x in ["int32", "int64"] for x in result.dtypes):
                result = result.astype(pd_mat.expected_result.dtypes[1])

        pd.testing.assert_frame_equal(result, pd_mat.expected_result)

    @pytest.mark.parametrize("row", [True, False])
    @pytest.mark.parametrize("trans_dtype", [str, int, float])
    @pytest.mark.parametrize("matrix_dtype", [str, int, float])
    def test_allow_different_types(
        self,
        pd_matrix_str: str,
        row: bool,
        trans_dtype: type,
        matrix_dtype: type,
        check_totals: bool,
        request,
    ):
        """Test that similar types are allowed in translation and data."""
        pd_mat: PandasMatrixResults = request.getfixturevalue(pd_matrix_str)

        # Change the dtype of the row / col
        new_matrix = pd_mat.mat.copy()
        if row:
            new_trans = pd_mat.translation.copy()
            keyword = "translation"
            new_matrix.index = new_matrix.index.astype(matrix_dtype)
        else:
            new_trans = pd_mat.col_translation.copy()
            keyword = "col_translation"
            new_matrix.columns = new_matrix.columns.astype(matrix_dtype)

        # Run the translation
        new_trans.from_col = new_trans.from_col.astype(trans_dtype)
        result = translation.pandas_matrix_zone_translation(
            **(pd_mat.input_kwargs(check_totals=check_totals) | {keyword: new_trans.df})
        )

        # Need to enforce types so this works in linux
        if sys.platform.startswith("linux"):
            if any(x in ["int32", "int64"] for x in result.dtypes):
                result = result.astype(pd_mat.expected_result.dtypes[1])

        pd.testing.assert_frame_equal(result, pd_mat.expected_result)


@pytest.mark.parametrize(
    "pd_matrix_str, check_totals",
    [
        ("pd_long_matrix_aggregation", True),
        ("pd_long_matrix_aggregation", False),
        ("pd_long_matrix_aggregation2", True),
        ("pd_long_matrix_aggregation2", False),
        ("pd_long_matrix_split", True),
        ("pd_long_matrix_split", False),
        ("pd_long_matrix_dtype", False),
    ],
)
class TestLongPandasMatrixParams:
    """Tests for caf.toolkit.translation.pandas_long_matrix_zone_translation"""

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
        pd_mat: PandasLongMatrixResults = request.getfixturevalue(pd_matrix_str)
        result = translation.pandas_long_matrix_zone_translation(
            **pd_mat.input_kwargs(check_totals=check_totals)
        )

        # Dtypes are checked in TestPandasMatrixParams.test_correct_results test. Ignore here.
        pd.testing.assert_series_equal(result, pd_mat.expected_result, check_dtype=False)

    def test_different_output_names(
        self,
        pd_matrix_str: str,
        check_totals: bool,
        request,
    ):
        """Test translation works with different output column names."""
        pd_mat: PandasLongMatrixResults = request.getfixturevalue(pd_matrix_str)
        renamed = pd_mat.update_output_cols(
            index_col_1_out_name="Origin", index_col_2_out_name="Destination"
        )
        result = translation.pandas_long_matrix_zone_translation(
            **pd_mat.input_kwargs(
                check_totals=check_totals,
                index_col_1_out_name="Origin",
                index_col_2_out_name="Destination",
            )
        )

        # Dtypes are checked in TestPandasMatrixParams.test_correct_results test. Ignore here.
        pd.testing.assert_series_equal(result, renamed, check_dtype=False)

    def test_additional_cols(
        self,
        pd_matrix_str: str,
        check_totals: bool,
        request,
    ):
        """Test a warning is raised when there are additional columns."""
        pd_mat: PandasLongMatrixResults = request.getfixturevalue(pd_matrix_str)
        new_mat = pd_mat.df.copy().to_frame().reset_index()
        new_mat.columns = ["production", "attraction", "values"]
        new_mat["extra_col"] = 0

        msg = "Extra columns found in matrix"
        with pytest.warns(UserWarning, match=msg):
            result = translation.pandas_long_matrix_zone_translation(
                **pd_mat.input_kwargs(check_totals=check_totals) | {"matrix": new_mat}
            )

        # Dtypes are checked in TestPandasMatrixParams.test_correct_results test. Ignore here.
        pd.testing.assert_series_equal(result, pd_mat.expected_result, check_dtype=False)


# ## READ FILE TRANSLATION FIXTURES & TESTS ## #
@dataclasses.dataclass
class PandasFileVectorResults:
    """Parameters and results for vector file translation tests."""

    vector_path: pathlib.Path
    vector_zone_column: str
    translation_path: pathlib.Path
    translation_from_column: str
    translation_to_column: str
    translation_factors_column: str
    expected: pd.DataFrame


@dataclasses.dataclass
class PandasFileMatrixResults:
    """Parameters and results for matrix file translation tests."""

    matrix_path: pathlib.Path
    matrix_zone_columns: str
    matrix_value_column: str
    translation_path: pathlib.Path
    translation_from_column: str
    translation_to_column: str
    translation_factors_column: str
    expected: pd.DataFrame


@pytest.fixture(name="vector_file_translation")
def fix_vector_file_translation(
    tmp_path: pathlib.Path,
    pd_multi_vector_multiindex: PandasMultiVectorResults,
) -> PandasFileVectorResults:
    """Write vector translation test files to temp directory."""
    data = pd_multi_vector_multiindex.vector
    data.index.name = "zone_id"
    trans_data = pd_multi_vector_multiindex.translation

    data_path = tmp_path / "test_vector_data.csv"
    assert not data_path.is_file(), "test vector file already exists"
    data.to_csv(data_path)
    assert data_path.is_file(), "test vector file not created"

    translation_path = tmp_path / "test_translation.csv"
    assert not translation_path.is_file(), "test translation file already exists"
    trans_data.df.to_csv(translation_path, index=False)
    assert translation_path.is_file(), "test translation not created"

    return PandasFileVectorResults(
        vector_path=data_path,
        vector_zone_column=data.index.name,
        translation_path=translation_path,
        translation_from_column=trans_data.translation_from_col,
        translation_to_column=trans_data.translation_to_col,
        translation_factors_column=trans_data.translation_factors_col,
        expected=pd_multi_vector_multiindex.expected_result,
    )


@pytest.fixture(name="matrix_file_translation")
def fix_matrix_file_translation(
    tmp_path: pathlib.Path,
    pd_matrix_split: PandasMatrixResults,
) -> PandasFileMatrixResults:
    """Write matrix translation test files to temp directory."""
    # Convert square to long format
    data = pd_matrix_split.mat.stack().to_frame()
    data.index.names = ["origin", "destination"]
    data.columns = ["value"]
    trans_data = pd_matrix_split.translation

    expected = pd_matrix_split.expected_result
    expected.index.name = data.index.names[0]
    expected.columns.name = data.index.names[1]

    data_path = tmp_path / "test_data_matrix_long.csv"
    assert not data_path.is_file(), "test vector file already exists"
    data.to_csv(data_path)
    assert data_path.is_file(), "test vector file not created"

    translation_path = tmp_path / "test_translation.csv"
    assert not translation_path.is_file(), "test translation file already exists"
    trans_data.df.to_csv(translation_path, index=False)
    assert translation_path.is_file(), "test translation not created"

    return PandasFileMatrixResults(
        matrix_path=data_path,
        matrix_zone_columns=data.index.names,
        matrix_value_column=data.columns[0],
        translation_path=translation_path,
        translation_from_column=trans_data.translation_from_col,
        translation_to_column=trans_data.translation_to_col,
        translation_factors_column=trans_data.translation_factors_col,
        expected=expected,
    )


class TestVectorTranslationFromFile:
    """Tests for the `vector_translation_from_file` function."""

    def test_simple(self, vector_file_translation: PandasFileVectorResults) -> None:
        """Test standard translation of vector file with all arguments."""
        output_path = vector_file_translation.vector_path.parent / "test_result.csv"
        translation.vector_translation_from_file(
            vector_path=vector_file_translation.vector_path,
            translation_path=vector_file_translation.translation_path,
            output_path=output_path,
            vector_zone_column=vector_file_translation.vector_zone_column,
            translation_from_column=vector_file_translation.translation_from_column,
            translation_to_column=vector_file_translation.translation_to_column,
            translation_factors_column=vector_file_translation.translation_factors_column,
        )

        assert output_path.is_file(), "translated vector not created"
        result = io.read_csv(
            output_path, index_col=vector_file_translation.translation_to_column
        )
        result.index = pd.to_numeric(result.index, downcast="unsigned")
        result.columns = pd.to_numeric(result.columns, downcast="unsigned")
        pd.testing.assert_frame_equal(
            result,
            vector_file_translation.expected,
            check_dtype=False,
            check_column_type=False,
            check_index_type=False,
        )


class TestMatrixTranslationFromFile:
    """Tests for `matrix_translation_from_file` function."""

    def test_long_matrix(self, matrix_file_translation: PandasFileMatrixResults) -> None:
        """Test translation of matrix file in long format."""
        output_path = matrix_file_translation.matrix_path.parent / "test_result.csv"
        translation.matrix_translation_from_file(
            matrix_path=matrix_file_translation.matrix_path,
            translation_path=matrix_file_translation.translation_path,
            output_path=output_path,
            matrix_zone_columns=matrix_file_translation.matrix_zone_columns,
            matrix_values_column=matrix_file_translation.matrix_value_column,
            translation_from_column=matrix_file_translation.translation_from_column,
            translation_to_column=matrix_file_translation.translation_to_column,
            translation_factors_column=matrix_file_translation.translation_factors_column,
        )

        assert output_path.is_file(), "translated matrix not created"
        result = io.read_csv_matrix(output_path, format_="long")
        result.index = pd.to_numeric(result.index, downcast="unsigned")
        result.columns = pd.to_numeric(result.columns, downcast="unsigned")
        pd.testing.assert_frame_equal(
            result,
            matrix_file_translation.expected,
            check_dtype=False,
            check_column_type=False,
            check_index_type=False,
        )
