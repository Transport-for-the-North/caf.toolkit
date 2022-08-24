# -*- coding: utf-8 -*-
"""Tests for the caf.toolkit.toolbox module"""
# Built-Ins
from typing import Any

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
