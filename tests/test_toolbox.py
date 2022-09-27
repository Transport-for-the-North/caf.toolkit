# -*- coding: utf-8 -*-
"""Tests for the caf.toolkit.toolbox module"""
# Built-Ins
from typing import Any
from typing import Iterable

# Third Party
import pytest

# Local Imports
# pylint: disable=import-error,wrong-import-position
from caf.toolkit import toolbox

# pylint: enable=import-error,wrong-import-position

# # # CONSTANTS # # #


# # # FIXTURES # # #


# # # TESTS # # #
class TestListSafeRemove:
    """Tests for caf.toolkit.toolbox.list_safe_remove"""

    @pytest.fixture(name="base_list", scope="class")
    def fixture_base_list(self):
        """Basic list for testing"""
        return [1, 2, 3, 4, 5, 6, 7, 8, 9]

    @pytest.mark.parametrize("remove", [[1], [1, 2], [20], [1, 20]])
    @pytest.mark.parametrize("throw_error", [True, False])
    def test_error_and_removal(
        self,
        base_list: list[Any],
        remove: list[Any],
        throw_error: bool,
    ):
        """Test that errors are thrown and items removed correctly"""
        # Check if an error should be thrown
        diff = set(remove) - set(base_list)
        all_items_in_list = len(diff) == 0

        # Build the expected return value
        expected_list = base_list.copy()
        for item in remove:
            if item in expected_list:
                expected_list.remove(item)

        # Check if an error is raised when it should be
        if throw_error and not all_items_in_list:
            with pytest.raises(ValueError):
                toolbox.list_safe_remove(
                    lst=base_list,
                    remove=remove,
                    throw_error=throw_error,
                )

        else:
            # Should work as expected
            new_lst = toolbox.list_safe_remove(
                lst=base_list,
                remove=remove,
                throw_error=throw_error,
            )
            assert new_lst == expected_list


class TestIsNoneLike:
    """Tests for caf.toolkit.toolbox.is_none_like"""

    @pytest.mark.parametrize("obj", [None, "none", "NONE", " None   "])
    def test_true_none_items(self, obj: Any):
        """Test single items are identified as None"""
        assert toolbox.is_none_like(obj)

    @pytest.mark.parametrize("obj", [0, "not none", "string"])
    def test_false_none_items(self, obj: Any):
        """Test single items are not identified as None"""
        assert not toolbox.is_none_like(obj)

    @pytest.mark.parametrize("obj", [[], [None], [None, None], [None, "none"]])
    def test_true_list_items(self, obj: list[Any]):
        """Test lists of items are identified as None"""
        assert toolbox.is_none_like(obj)

    @pytest.mark.parametrize("obj", [[0], [None, 0], [None, None, "not none"]])
    def test_false_list_items(self, obj: list[Any]):
        """Test lists of items are not identified as None"""
        assert not toolbox.is_none_like(obj)


class TestEqualIgnoreOrder:
    """Tests for caf.toolkit.toolbox.equal_ignore_order"""

    def test_order_match(self):
        """Test when both iterables are the same"""
        lst = [1, 2, 3]
        assert toolbox.equal_ignore_order(lst, lst)

    def test_out_of_order_match(self):
        """Test when both iterables are the same, but in different order"""
        lst = [1, 2, 3]
        lst2 = [3, 1, 2]
        assert toolbox.equal_ignore_order(lst, lst2)

    @pytest.mark.parametrize("one", [[], [1], [1, 2]])
    @pytest.mark.parametrize("two", [[2], [3, 4]])
    def test_not_match(self, one: Iterable[Any], two: Iterable[Any]):
        """Test when iterables do not match at all"""
        assert not toolbox.equal_ignore_order(one, two)
