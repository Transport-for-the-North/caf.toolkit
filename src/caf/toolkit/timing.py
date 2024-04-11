# -*- coding: utf-8 -*-
"""Common utility functions for handling date and times."""
# Built-Ins
import datetime
import math
import time
from typing import Literal

# # # CONSTANTS # # #
TimePrecision = Literal["millisecond", "microsecond"]

# # # CLASSES # # #


# # # FUNCTIONS # # #
def current_milli_time() -> float:
    """Get the current system time in milliseconds."""
    return time.perf_counter() * 1000


def get_time() -> str:
    """Get the current time with millisecond precision.

    The time is returned in "%H:%M:%S.%f" format
    """
    return get_datetime(time_format="%H:%M:%S.%f", precision="millisecond")


def get_datetime(
    time_format: str = "%d-%m-%Y  %H:%M:%S.%f",
    precision: TimePrecision = "microsecond",
) -> str:
    """Get the current datetime at different precisions.

    Parameters
    ----------
    time_format:
        What format to get the datetime at. Defaults to
        "%d-%m-%Y  %H:%M:%S.%f". See datetime.strftime() for acceptable formats

    precision:
        What precision to get the datetime at.
        Currently, supports "microsecond" and "millisecond".

    Returns
    -------
    datetime_str:
        A string of the current datetime in `time_format`
    """
    # Init
    valid_precision = TimePrecision.__args__  # type: ignore
    precision = precision.strip().lower()

    # Validate
    if precision not in valid_precision:
        raise ValueError(
            f"`{precision}` is not a valid precision. Can only accept one of:\n"
            f"{valid_precision}"
        )

    if time_format[-2:] != "%f":
        raise ValueError(
            "`time_format` is not in the correct format. The precision can only"
            "be set if the format ends in '%f'."
        )

    # Get the correct format - assumes time ends in microseconds
    if precision == "microsecond":
        return datetime.datetime.now().strftime(time_format)

    if precision == "millisecond":
        return datetime.datetime.now().strftime(time_format)[:-3]

    raise ValueError(
        "%s seems to be a valid precision - but I don't know "
        "how to format it. There must be a missing if statement!"
    )


def time_taken(
    start_time: float,
    end_time: float,
) -> str:
    """
    Format the time taken into hours, minutes and seconds.

    Parameters
    ----------
    start_time:
        The start time, in milliseconds. Can be gotten from current_milli_time()

    end_time:
        The end time, in milliseconds. Can be gotten from current_milli_time()

    Returns
    -------
    time_taken:
        The time passed between start and end time in format:
        xx-hrs xxm xx.xxs. Where x is replaced with actual values

    Raises
    ------
    ValueError:
        If end_time - start_time is less than, or equal to, 0
    """
    # Init
    elapsed_time = end_time - start_time
    elapsed_secs = elapsed_time / 1000

    # Validate
    if elapsed_time < 0:
        raise ValueError(
            "Elapsed time is negative! Was the start_time "
            "and end_time given the wrong way around?"
        )

    # Split into minutes and seconds
    seconds = elapsed_secs % 60
    minutes = math.floor(elapsed_secs / 60)

    # If no minutes passed, just return seconds
    if minutes == 0:
        return f"{seconds:.2f}s"

    # Split into hours and minutes
    res_mins = minutes % 60
    hours = math.floor(minutes / 60)

    # If no hours passed, return minutes and seconds
    if hours == 0:
        return f"{res_mins:d}m {seconds:.2f}s"

    # Otherwise return the full format
    return f"{hours:d}hrs {res_mins:d}m {seconds:.2f}s"
