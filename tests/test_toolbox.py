# -*- coding: utf-8 -*-
"""Tests for the caf.toolkit.toolbox module"""
from __future__ import annotations

# Built-Ins
import dataclasses
import operator
from typing import Any, Iterable

# Third Party
import numpy as np
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


class TestSetComparison:
    """Tests for set/list comparison functions.

    covers:
        caf.toolkit.toolbox.get_missing_items()
        caf.toolkit.toolbox.compare_sets()
    """

    @dataclasses.dataclass
    class Results:
        """Hold input and expected output to functions"""

        # Inputs
        item1: list
        item2: list

        # Returns
        item1_not_2: list
        item2_not_1: list
        equal: bool

    @pytest.fixture(name="equal_items", scope="class")
    def fixture_equal_items(self) -> Results:
        """Object of two equal items"""
        return self.Results(
            item1=[1, 2, 3, 4, 5],
            item2=[1, 2, 3, 4, 5],
            item1_not_2=list(),
            item2_not_1=list(),
            equal=True,
        )

    @pytest.fixture(name="similar_items", scope="class")
    def fixture_similar_items(self) -> Results:
        """Object of two similar items"""
        return self.Results(
            item1=[1, 2, 3, 4, 5],
            item2=[3, 4, 5, 6, 7],
            item1_not_2=[1, 2],
            item2_not_1=[6, 7],
            equal=False,
        )

    @pytest.fixture(name="different_items", scope="class")
    def fixture_different_items(self) -> Results:
        """Object of two different items"""
        item1 = [1, 2, 3, 4, 5]
        item2 = [6, 7, 8, 9, 10]
        return self.Results(
            item1=item1,
            item2=item2,
            item1_not_2=item1,
            item2_not_1=item2,
            equal=False,
        )

    @pytest.mark.parametrize(
        "item_results_str",
        ["equal_items", "similar_items", "different_items"],
    )
    def test_correct_lists(self, item_results_str: str, request):
        """Check that the list function returns the correct result"""
        item_results = request.getfixturevalue(item_results_str)
        result = toolbox.get_missing_items(item_results.item1, item_results.item2)
        assert item_results.item1_not_2 == result[0]
        assert item_results.item2_not_1 == result[1]

    @pytest.mark.parametrize(
        "item_results_str",
        ["equal_items", "similar_items", "different_items"],
    )
    def test_correct_sets(self, item_results_str: str, request):
        """Check that the list function returns the correct result"""
        item_results = request.getfixturevalue(item_results_str)
        result = toolbox.compare_sets(set(item_results.item1), set(item_results.item2))
        assert item_results.equal == result[0]
        assert set(item_results.item1_not_2) == set(result[1])
        assert set(item_results.item2_not_1) == set(result[2])

    @pytest.mark.parametrize(
        "item_results_str",
        ["equal_items", "similar_items", "different_items"],
    )
    def test_non_unique_list1(self, item_results_str: str, request):
        """Check that an error is raised when items are not unique"""
        item_results = request.getfixturevalue(item_results_str)
        new_item1 = item_results.item1.copy()
        new_item1 += new_item1
        msg = "only works on lists of unique items"
        with pytest.raises(ValueError, match=msg):
            toolbox.get_missing_items(new_item1, item_results.item2)

    @pytest.mark.parametrize(
        "item_results_str",
        ["equal_items", "similar_items", "different_items"],
    )
    def test_non_unique_list2(self, item_results_str: str, request):
        """Check that an error is raised when items are not unique"""
        item_results = request.getfixturevalue(item_results_str)
        new_item2 = item_results.item2.copy()
        new_item2 += new_item2
        msg = "only works on lists of unique items"
        with pytest.raises(ValueError, match=msg):
            toolbox.get_missing_items(item_results.item1, new_item2)


class TestDictList:
    """Tests for caf.toolkit.toolbox.dict_list"""

    @pytest.fixture(name="list_of_dicts", scope="function")
    def fix_list_of_dicts(self) -> list[dict[str, Any]]:
        """List of dicts for testing"""
        return [
            {"a": 1, "b": 2, "c": 3},
            {"a": 4, "b": 5, "c": 6},
            {"a": 7, "b": 8, "c": 9},
        ]

    @pytest.fixture(name="expected_add", scope="class")
    def fix_expected_add(self) -> dict[str, list[Any]]:
        """Expected output from list_of_dicts"""
        return {"a": 12, "b": 15, "c": 18}

    @pytest.fixture(name="expected_mul", scope="class")
    def fix_expected_mul(self) -> dict[str, list[Any]]:
        """Expected output from list_of_dicts"""
        return {"a": 28, "b": 80, "c": 162}

    @pytest.fixture(name="expected_sub", scope="class")
    def fix_expected_sub(self) -> dict[str, list[Any]]:
        """Expected output from list_of_dicts"""
        return {"a": -10, "b": -11, "c": -12}

    @pytest.mark.parametrize(
        "expected_str,op",
        [
            ("expected_add", operator.add),
            ("expected_mul", operator.mul),
            ("expected_sub", operator.sub),
        ],
    )
    def test_dict_list(
        self, list_of_dicts: list[dict[str, Any]], expected_str: str, op, request
    ):
        """Test that dict_list works as expected"""
        expected = request.getfixturevalue(expected_str)
        assert toolbox.combine_dict_list(list_of_dicts, op) == expected
