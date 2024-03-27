# -*- coding: utf-8 -*-
"""Library of multiprocessing functionality."""
from __future__ import annotations

# Built-Ins
import multiprocessing as mp

# Built-ins
import os
import time
import traceback
import warnings
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Collection,
    Iterable,
    Mapping,
    Optional,
    TypeVar,
)

# Third Party
import tqdm

# Local Imports
from caf.toolkit import tqdm_utils

# Need for type-hints
if TYPE_CHECKING:
    # Built-Ins
    from multiprocessing.pool import ApplyResult, Pool
    from multiprocessing.synchronize import Event

# # # CONSTANTS # # #
_T = TypeVar("_T")


# # # FUNCTIONS # # #
def create_kill_pool_fn(
    pool: Pool,
    terminate_process_event: Event,
):
    """
    Create a Callback function for each function in a Pool.

    This is called whenever an exception is raised inside one of the processes in
    a Pool. This is mostly used to give a clean error output when an error occurs.
    """

    def kill_pool(process_error=None, process_callback=True):
        """Print error and kill pool completely.

        Needs to accept a `process_error` arg to be used as a callback in
        `multiprocessing.Pool.apply_async()`
        """
        if process_callback and process_error is not None:
            traceback.print_exception(
                type(process_error),
                process_error,
                process_error.__traceback__,
            )

        pool.close()
        pool.terminate()
        terminate_process_event.set()

    return kill_pool


def wait_for_pool_results(
    results: list[ApplyResult[_T]],
    terminate_process_event: Event,
    result_timeout: int,
    pbar_kwargs: Optional[dict[str, Any]] = None,
) -> list[_T]:
    """Wait for and grab results from a multiprocessing Pool.

    Aims to return all results once processes have complete.
    Will throw an error if `terminate_process_event` is set, or
    `result_timeout` is reached.

    Parameters
    ----------
    results:
        A list of `multiprocessing.pool.AsyncResult`. The results to wait
        for.

    terminate_process_event:
        A `multiprocessing.Event`. This event should get set if an error occurs
        and the `multiprocessing.Pool` (which is generating the `results`)
        needs to terminate.

    result_timeout:
        How many seconds to wait for all `results` before throwing an error.

    pbar_kwargs:
        A dictionary of keyword arguments to pass into a tqdm progress bar.
        Will be used as `tqdm.tqdm(**pbar_kwargs)`

    Returns
    -------
    results_out:
        A list of the return values collected from `results`.
        May not be in the same order as received `results`.

    Raises
    ------
    multiprocessing.ProcessError:
        Will be raised in the following cases:
        - If `terminate_process_event`
        - If an error is discovered in one of the processes
        - If one or more of the results are lost when retrieving results

    TimeoutError:
        If the seconds spent waiting for `results` is greater than
        `result_timeout`

    """
    # Initialise loop
    start_time = time.time()
    got_all_results = False
    return_results = list()
    n_start_results = len(results)

    # If not given any kwargs, assume no pbar wanted
    if pbar_kwargs is None:
        pbar_kwargs = {"disable": True}

    # Context is meant to keep the pbar tidy
    with tqdm_utils.std_out_err_redirect_tqdm() as orig_stdout:
        # Additional args for context
        pbar_kwargs["file"] = orig_stdout
        pbar_kwargs["dynamic_ncols"] = True

        # If no total given, we can add one!
        if "total" not in pbar_kwargs or pbar_kwargs["total"] == 0:
            pbar_kwargs["total"] = n_start_results

        # Improves time prediction guessing
        pbar_kwargs["smoothing"] = 0

        # Finally, make to pbar!
        pbar = tqdm.tqdm(**pbar_kwargs)

        # Grab all the results as they come in
        while not got_all_results:
            # Wait for a bit to avoid intensive looping
            time.sleep(0.05)

            # Check for an event
            if terminate_process_event.is_set():
                raise mp.ProcessError("While getting results terminate_process_event was set.")

            # Check if we've run out of time
            if (time.time() - start_time) > result_timeout:
                raise TimeoutError("Ran out of time while waiting for results.")

            # Check if we have any results
            res_to_remove = list()
            for i, res in enumerate(results):
                if not res.ready():
                    continue

                if not res.successful():
                    raise mp.ProcessError("An error occurred in one of the processes.")

                # Give a minute to get the result
                # Shouldn't take this long as we know the result is ready
                return_results.append(res.get(60))
                res_to_remove.append(i)

            # Update the progress bar with the number of results we just got
            if len(res_to_remove) > 0:
                pbar.update(len(res_to_remove))

            # Remove results we've got
            for i in sorted(res_to_remove, reverse=True):
                del results[i]

            # Quick sanity check
            if not len(results) + len(return_results) == n_start_results:
                raise mp.ProcessError(
                    "While getting the multiprocessing results an error "
                    "occurred. Lost one or more results. Started with "
                    f"{n_start_results:d}, now have "
                    f"{len(results) + len(return_results):d}."
                )

            # Check if we have all results
            if len(return_results) == n_start_results:
                got_all_results = True

    # Tidy up before we leave
    pbar.close()

    return return_results


def _call_order_wrapper(
    index: int,
    func: Callable[..., _T],
    *args,
    **kwargs,
) -> tuple[int, _T]:
    """Wrap a function return values with a calling index.

    Useful when placing a function into an asynchronous Pool. The index of the
    function is returned alongside the results, allowing for sorting.

    Note
    ----
        Originally tried to implement this as a function decorator, however
        Pools do not like decorated functions as they become un-pickleable.
    """
    return index, func(*args, **kwargs)


def _check_args_kwargs(
    args: Optional[Collection[Iterable[Any]]] = None,
    kwargs: Optional[Collection[Mapping[str, Any]]] = None,
    args_default: Optional[Any] = None,
    kwargs_default: Optional[Any] = None,
    length: Optional[int] = None,
) -> tuple[Collection[Iterable[Any]], Collection[Mapping[str, Any]]]:
    """Format args and kwargs correctly if only one set.

    If args or kwargs are set to None they are filled with their default value
    to match the length of the other.
    If both are None, then they are set to length.
    If neither are None, they are returned as is.
    """
    # Init
    args_default = list() if args_default is None else args_default
    kwargs_default = dict() if kwargs_default is None else kwargs_default

    if args is not None and kwargs is None:
        kwargs = [kwargs_default for _ in range(len(args))]
    elif args is None and kwargs is not None:
        args = [args_default for _ in range(len(kwargs))]
    elif args is None and kwargs is None and length is not None:
        args = [args_default for _ in range(length)]
        kwargs = [kwargs_default for _ in range(length)]
    elif args is None and kwargs is None and length is None:
        raise ValueError(
            "Both args and kwargs are None and length has not "
            "been set. Don't know how to proceed!"
        )
    # If no branch taken, both args and kwargs must have been set

    assert args is not None
    assert kwargs is not None

    return args, kwargs


def _process_pool_wrapper_kwargs_in_order(
    fn: Callable[..., _T],
    arg_list: Collection[Iterable[Any]],
    kwarg_list: Collection[Mapping[str, Any]],
    process_count: int,
    pool_maxtasksperchild: int,
    result_timeout: int,
    pbar_kwargs: Optional[dict[str, Any]] = None,
) -> list[_T]:
    """See `process_pool_wrapper()` for full documentation of this function.

    Sibling function with `_process_pool_wrapper_kwargs_out_order()`.
    Should only be called from `process_pool_wrapper()`.
    """
    # pylint: disable=too-many-locals
    terminate_processes_event = mp.Event()

    with mp.Pool(processes=process_count, maxtasksperchild=pool_maxtasksperchild) as pool:
        kill_pool = create_kill_pool_fn(pool, terminate_processes_event)

        try:
            # Add each function call to the pool with an index identifier
            apply_results: list[ApplyResult[tuple[int, _T]]] = list()
            for i, (args, kwargs) in enumerate(zip(arg_list, kwarg_list)):
                apply_results.append(
                    pool.apply_async(
                        func=_call_order_wrapper,
                        args=(i, fn, *args),
                        kwds=kwargs,
                        error_callback=kill_pool,
                    )
                )

            result_timeout *= max(len(apply_results), 1)
            results = wait_for_pool_results(
                results=apply_results,
                terminate_process_event=terminate_processes_event,
                result_timeout=result_timeout,
                pbar_kwargs=pbar_kwargs,
            )

        except BaseException as exception:
            # If any exception, clean up and re-raise
            kill_pool(process_callback=False)
            traceback.print_exc()
            raise exception
    del pool

    # Order the results, and separate from enumerator
    _, final_results = zip(*sorted(results, key=lambda x: x[0]))
    return list(final_results)


def _process_pool_wrapper_kwargs_out_order(
    fn: Callable[..., _T],
    arg_list: Collection[Iterable[Any]],
    kwarg_list: Collection[Mapping[str, Any]],
    process_count: int,
    pool_maxtasksperchild: int,
    result_timeout: int,
    pbar_kwargs: Optional[dict[str, Any]] = None,
) -> list[_T]:
    """See `process_pool_wrapper()` for full documentation of this function.

    Sibling function with `_process_pool_wrapper_kwargs_in_order()`.
    Should only be called from `process_pool_wrapper()`.
    """
    terminate_process_event = mp.Event()

    with mp.Pool(processes=process_count, maxtasksperchild=pool_maxtasksperchild) as pool:
        kill_pool = create_kill_pool_fn(pool, terminate_process_event)

        try:
            apply_results: list[ApplyResult[_T]] = list()

            # Add each function call to the pool
            for args, kwargs in zip(arg_list, kwarg_list):
                apply_results.append(
                    pool.apply_async(
                        fn,
                        args=args,
                        kwds=kwargs,
                        error_callback=kill_pool,
                    )
                )

            result_timeout *= max(len(apply_results), 1)
            results = wait_for_pool_results(
                results=apply_results,
                terminate_process_event=terminate_process_event,
                result_timeout=result_timeout,
                pbar_kwargs=pbar_kwargs,
            )

        except BaseException as exception:
            # If any exception, clean up and re-raise
            kill_pool(process_callback=False)
            traceback.print_exc()
            raise exception

    del pool
    return results


def multiprocess(
    fn: Callable[..., _T],
    arg_list: Optional[Collection[Iterable[Any]]] = None,
    kwarg_list: Optional[Collection[Mapping[str, Any]]] = None,
    process_count: Optional[int] = None,
    pool_maxtasksperchild: int = 4,
    in_order: bool = False,
    result_timeout: int = 86400,
    pbar_kwargs: Optional[dict[str, Any]] = None,
) -> list[_T]:
    """Run a function and arguments across multiple cores of a CPU.

    Runs the given function with the arguments given in a multiprocessing.Pool,
    returning the function output.

    Deals with various process_count values:
        - If negative, `os.cpu_count() - process_count` processes will be used
        - If 0, no multiprocessing will be used. The code will be run in
          a for loop, using only one process (and therefore CPU).
        - If positive, process_count processes will be used. If process_count
          is greater than `os.cpu_count() - 1`, a warning will be raised.

    Parameters
    ----------
    fn:
        The function to call.

    arg_list:
        A list of iterables e.g. tuples/lists. `len(args)` equals the
        number of times `fn` will be called. If `kwargs` is also provided,
        `args` should directly correspond to it. Each tuple should contain a
        full set of non-keyword arguments to be passed to a single call of fn.

    kwarg_list:
        A list of dictionaries. The keys should be the keyword argument names,
        and the values the keyword argument values. `len(kwargs)` equals the
        number of times `fn` will be called. If `args` is also provided,
        `kwargs` should directly correspond to it. Each dictionary should
        contain a full set of keyword arguments to be passed to a single
        call of `fn`.

    process_count:
        The number of processes to create in the Pool. Typically, this
        should not exceed the number of cores available.
        Defaults to `os.cpu_count() - 1`.

    pool_maxtasksperchild:
        Passed into the created Pool as `maxtaskperchild=pool_maxtaskperchild`.
        It is the number of tasks a worker process can complete before it will
        exit and be replaced with a fresh worker process, to enable unused
        resources to be freed.

    in_order:
        Whether the indexes of the return values need to directly corresspond
        to the input values (`args` and `kwargs`) given. Setting this to `True`
        is slightly slower due to sorting the results.

    result_timeout:
        How long to wait for each process to finish before throwing an
        exception. Defaults to 86400 seconds, (24 hours).

    pbar_kwargs:
        A dictionary of keyword arguments to pass into a progress bar.
        This dictionary is passed into `tqdm.tqdm(**pbar_kwargs)` when
        building the progress bar.

    See Also
    --------
    `tqdm.tqdm()`

    Examples
    --------
    The following three function calls:
    >>> a = sorted(range(10))
    >>> b = sorted(range(100))
    >>> c = sorted(range(20), reverse=True)

    Would be called, using this function, like this:
    >>> # Note the use of a tuple to make sure a single argument is still
    >>> # iterable
    >>> a_args = (range(10), )
    >>> b_args = (range(100), )
    >>> c_args = (range(20 ), )
    >>>
    >>> # Need to use an empty dict where arguments are not given
    >>> a_kwargs = dict()
    >>> b_kwargs = dict()
    >>> c_kwargs = {'reverse': True}

    >>> args_list = [a_args, b_args, c_args]
    >>> kwargs_list = [a_kwargs, b_kwargs, c_kwargs]
    >>> a, b, c = multiprocess(sorted, args_list, kwargs_list)

    # TODO(BT): Add example of how to convert a for-loop into one of these calls.
    """
    # TODO(BT): Maybe add functionality to allow calling a function without
    #  any arguments n times
    # Validate the args and kwargs
    if arg_list is None and kwarg_list is None:
        raise ValueError(
            "Both args and kwargs are set to None. Cannot infer the number of "
            "times to call fn. Please set either args or kwargs."
        )

    if arg_list is not None and kwarg_list is not None:
        if len(arg_list) != len(kwarg_list):
            raise ValueError(
                "Both args and kwargs were given but they are not the same "
                "length. Cannot infer the number of times to call fn.\n"
                f"args length: {len(arg_list)}\n"
                f"kwargs length: {len(kwarg_list)}"
            )

    # Format correctly where not given
    arg_list, kwarg_list = _check_args_kwargs(arg_list, kwarg_list)

    # Validate process_count
    cpu_count = os.cpu_count()
    if cpu_count is None:
        raise OSError("Cannot determine CPU count of system.")

    process_count = cpu_count - 1 if process_count is None else process_count
    if process_count < -cpu_count:
        raise ValueError(
            f"Negative process_count given is too small. Cannot run "
            f"{process_count:d} less processes than cpu count as only "
            f"{os.cpu_count():d} cpus have been found by python."
        )

    if process_count > cpu_count - 1:
        warnings.warn(
            f"Process_count given is too high ({process_count}). It is greater "
            f"than one less than the CPU count found by Python "
            f"{cpu_count - 1:d}. Only do this if you know what you're "
            f"doing otherwise it may intermittently freeze your system."
        )

    # Determine the number of processes to use
    if process_count < 0:
        process_count = cpu_count + process_count

    # Just run a for-loop if the process count is 0
    if process_count == 0:
        if pbar_kwargs is not None:
            # If no total given, we can add one!
            if "total" not in pbar_kwargs or pbar_kwargs["total"] == 0:
                pbar_kwargs["total"] = len(kwarg_list)
            return [
                fn(*a, **k) for a, k in tqdm.tqdm(zip(arg_list, kwarg_list), **pbar_kwargs)
            ]

        return [fn(*a, **k) for a, k in zip(arg_list, kwarg_list)]

    # If we get here, the process count must be > 0 and valid
    # Now either run in order or not
    if in_order:
        return _process_pool_wrapper_kwargs_in_order(
            fn=fn,
            arg_list=arg_list,
            kwarg_list=kwarg_list,
            process_count=process_count,
            pool_maxtasksperchild=pool_maxtasksperchild,
            result_timeout=result_timeout,
            pbar_kwargs=pbar_kwargs,
        )

    return _process_pool_wrapper_kwargs_out_order(
        fn=fn,
        arg_list=arg_list,
        kwarg_list=kwarg_list,
        process_count=process_count,
        pool_maxtasksperchild=pool_maxtasksperchild,
        result_timeout=result_timeout,
        pbar_kwargs=pbar_kwargs,
    )
