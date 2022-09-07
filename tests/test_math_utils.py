# -*- coding: utf-8 -*-
"""Tests for the caf.toolkit.math_utils module"""
# Built-Ins
import math
from typing import Union

# Third Party
import pytest


# Local Imports
# pylint: disable=import-error,wrong-import-position
from caf.toolkit import math_utils
# pylint: enable=import-error,wrong-import-position

# # # CONSTANTS # # #


# # # FIXTURES # # #


# # # TESTS # # #
class TestIsAlmostEqual:
    """Tests for caf.toolkit.math_utils.is_almost_equal"""

    @pytest.mark.parametrize("val1", [0, 0.5, 1])
    @pytest.mark.parametrize("val2", [0, 0.5, 1])
    @pytest.mark.parametrize("rel_tol", [0.0001, 0.05, 1.5])
    @pytest.mark.parametrize("abs_tol", [0, 0.5, 10])
    def test_equal_to_builtin(
        self,
        val1: Union[int, float],
        val2: Union[int, float],
        rel_tol: float,
        abs_tol: float,
    ):
        """Test it works exactly like math.isclose"""
        expected = math.isclose(val1, val2, rel_tol=rel_tol, abs_tol=abs_tol)
        got = math_utils.is_almost_equal(val1=val1, val2=val2, rel_tol=rel_tol, abs_tol=abs_tol,)
        assert expected == got
