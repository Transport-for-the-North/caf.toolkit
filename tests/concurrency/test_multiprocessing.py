"""Tests for the caf.toolkit.concurrency.multiprocessing module"""
# Built-Ins
import multiprocessing as mp
import os
import time
from collections.abc import Callable, Iterable
from typing import Any, NamedTuple

# Third Party
import numpy as np
import pytest

# Local Imports
# pylint: disable=import-error,wrong-import-position
from caf.toolkit import concurrency, toolbox

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
    n_vals = 5

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
            arg_list=callee.arg_list,
            kwarg_list=callee.kwarg_list,
            in_order=True,
        )

        assert results == expected_results

    def test_out_order(self, callee: FunctionAndArguments):
        """Test running multiprocess out of order"""
        # Generate baseline to compare
        expected_results = list()
        for args, kwargs in zip(callee.arg_list, callee.kwarg_list):
            expected_results.append(callee.fn(*args, **kwargs))

        # Run and check
        results = concurrency.multiprocess(
            fn=callee.fn,
            arg_list=callee.arg_list,
            kwarg_list=callee.kwarg_list,
            in_order=False,
        )

        assert toolbox.equal_ignore_order(results, expected_results)


class TestMultiprocessArgsKwargs:
    """Tests caf.toolkit.concurrency.multiprocess args and kwargs"""

    @staticmethod
    def kwargs_only_function(iterator: Iterable, reverse: bool = False):
        """Wrapper of sorted to accept kwargs only"""
        return sorted(iterator, reverse=reverse)

    def test_args_only(self, callee: FunctionAndArguments):
        """Test running multiprocess with only args"""
        # Generate baseline to compare
        expected_results = list()
        for args in callee.arg_list:
            expected_results.append(callee.fn(*args))

        # Run and check
        results = concurrency.multiprocess(
            fn=callee.fn,
            arg_list=callee.arg_list,
            in_order=True,
        )

        assert results == expected_results

    def test_kwargs_only(self, callee: FunctionAndArguments):
        """Test running multiprocess with only kwargs"""
        # Generate baseline to compare
        kwarg_list = list()
        expected_results = list()
        for args, kwargs in zip(callee.arg_list, callee.kwarg_list):
            new_kwargs = kwargs.copy()
            new_kwargs.update({"iterator": list(args)[0]})
            kwarg_list.append(new_kwargs)
            expected_results.append(self.kwargs_only_function(**new_kwargs))

        # Run and check
        results = concurrency.multiprocess(
            fn=self.kwargs_only_function,
            kwarg_list=kwarg_list,
            in_order=True,
        )

        assert results == expected_results


class TestMultiprocessErrors:
    """Tests caf.toolkit.concurrency.multiprocess error production"""

    @staticmethod
    def error_throw_function(*args, **kwargs):
        """Throw an error"""
        raise OSError

    @staticmethod
    def wait_function(*args, **kwargs):
        """Wait for timeout error"""
        del args
        del kwargs
        time.sleep(100)

    def test_no_args_kwargs(self, callee: FunctionAndArguments):
        """Test running multiprocess with no args or kwargs"""
        with pytest.raises(ValueError):
            concurrency.multiprocess(fn=callee.fn)

    def test_broken_args_kwargs(self, callee: FunctionAndArguments):
        """Test running multiprocess different length args and kwargs"""
        arg_list = callee.arg_list.copy()
        del arg_list[0]
        with pytest.raises(ValueError):
            concurrency.multiprocess(
                fn=callee.fn,
                arg_list=arg_list,
                kwarg_list=callee.kwarg_list,
            )

    def test_error_catching(self, callee: FunctionAndArguments):
        """Test errors are correctly caught and handled"""
        with pytest.raises(mp.ProcessError):
            concurrency.multiprocess(
                fn=self.error_throw_function,
                arg_list=callee.arg_list,
                kwarg_list=callee.kwarg_list,
            )

    def test_timeout(self, callee: FunctionAndArguments):
        """Test that timeouts are thrown correctly"""
        with pytest.raises(TimeoutError):
            concurrency.multiprocess(
                fn=self.wait_function,
                arg_list=callee.arg_list,
                kwarg_list=callee.kwarg_list,
                result_timeout=1,
            )


class TestMultiprocessProcessCount:
    """Tests caf.toolkit.concurrency.multiprocess process counts"""

    @pytest.mark.parametrize("process_count", [1, 2, 4, 8])
    def test_fine_process_counts(
        self,
        callee: FunctionAndArguments,
        process_count: int,
    ):
        """Make sure process counts work as they should"""
        # Don't run if it's not an OK number
        cpu_count = os.cpu_count()
        assert cpu_count is not None
        if process_count > cpu_count - 1:
            return

        # Otherwise generate baseline to compare
        expected_results = list()
        for args in callee.arg_list:
            expected_results.append(callee.fn(*args))

        # Run and check
        results = concurrency.multiprocess(
            fn=callee.fn, arg_list=callee.arg_list, in_order=True, process_count=process_count
        )

        assert results == expected_results

    def test_too_big_process_count(self, callee: FunctionAndArguments):
        """Make sure user in warned when process count too big"""
        # Generate a number that should throw a warning
        cpu_count = os.cpu_count()
        assert cpu_count is not None
        process_count = cpu_count + 1

        # Run and check
        with pytest.warns(UserWarning):
            concurrency.multiprocess(
                fn=callee.fn, arg_list=callee.arg_list, process_count=process_count
            )

    def test_too_small_process_count(self, callee: FunctionAndArguments):
        """Make sure error is raised when process count too small"""
        # Generate a number that should throw a warning
        cpu_count = os.cpu_count()
        assert cpu_count is not None
        process_count = -cpu_count * 2

        # Run and check
        with pytest.raises(ValueError):
            concurrency.multiprocess(
                fn=callee.fn, arg_list=callee.arg_list, process_count=process_count
            )
