# -*- coding: utf-8 -*-
"""Tests for the caf.toolkit.concurrency.multiprocessing module"""
# Built-Ins
from typing import Any
from typing import Iterable
from typing import Callable
from typing import NamedTuple

# Third Party
import pytest
import numpy as np


# Local Imports
# pylint: disable=import-error,wrong-import-position
from caf.toolkit import concurrency
# pylint: enable=import-error,wrong-import-position

# # # CONSTANTS # # #


# # # FIXTURES # # #
class FunctionAndArguments(NamedTuple):
    """Collection of args and kwargs for a function call"""
    fn: Callable
    arg_list: list[Iterable[np.ndarray]]
    kwarg_list: list[dict[str, Any]]


@pytest.fixture(name="callee", scope="module")
def fixture_callee():
    """Build the function and arguments to call it with"""
    # Init
    n_repeats = 10
    n_vals = 10

    # Create the args and kwargs
    arg_list = list()
    kwarg_list = list()
    for i in range(n_repeats):
        # Create the arg value
        arg_val = list(np.random.randint(10, size=n_vals))
        arg_list.append((arg_val,))

        # Create kwarg value - alternate
        if i % 2 == 1:
            kwarg_list.append(dict())
        else:
            kwarg_list.append({"reverse": True})

    return FunctionAndArguments(
        fn=sorted,
        arg_list=arg_list,
        kwarg_list=kwarg_list,
    )


# # # TESTS # # #
class TestMultiprocessOrder:
    """Tests caf.toolkit.concurrency.multiprocess in_order arguments"""

    def test_in_order(self, callee: FunctionAndArguments):
        """Test running multiprocess in order"""
        # Generate baseline to compare
        expected_results = list()
        for args, kwargs in zip(callee.arg_list, callee.kwarg_list):
            expected_results.append(callee.fn(*args, **kwargs))

        # Run and check
        results = concurrency.multiprocess(
            fn=callee.fn,
            args=callee.arg_list,
            kwargs=callee.kwarg_list,
        )

        assert results == expected_results

    def test_out_order(self, callee: FunctionAndArguments):
        """Test running multiprocess out of order"""
        pass


    # Test running in order

    # Test running out of order

    # Test different process counts

    # Test no args / kwargs given

    # Test args and kwargs different length

    # Test errors are captured


    # Test timeout
