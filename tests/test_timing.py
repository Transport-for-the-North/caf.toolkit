# -*- coding: utf-8 -*-
"""Tests for the {} module"""
# Built-Ins
import dataclasses

# Third Party
import pytest

# Local Imports
# pylint: disable=import-error,wrong-import-position
from caf.toolkit import timing

# pylint: enable=import-error,wrong-import-position

# # # CONSTANTS # # #


# # # FIXTURES # # #


# # # TESTS # # #
class TestTimeTaken:
    """Test the caf.toolkit.timing.time_taken function"""

    @dataclasses.dataclass
    class ConvergenceExample:
        """Collection of data to pass to an RMSE call"""

        start_time: float
        end_time: float
        elapsed_str: str

    @staticmethod
    def time_to_milli(seconds: int = 0, minutes: int = 0, hours: int = 0) -> int:
        """Convert time into milliseconds"""
        minutes += hours * 60
        seconds += minutes * 60
        return seconds * 1000

    @pytest.fixture(name="current_time", scope="class")
    def fixture_current_time(self) -> float:
        """Get the current time in milliseconds"""
        return timing.current_milli_time()

    @pytest.fixture(name="seconds_change", scope="class")
    def fixture_seconds_change(self, current_time: float) -> ConvergenceExample:
        """Create a fixture where the seconds change"""
        seconds = 23
        end_time = current_time + self.time_to_milli(seconds=seconds)
        return self.ConvergenceExample(
            start_time=current_time,
            end_time=end_time,
            elapsed_str=f"{seconds:.2f}s",
        )

    @pytest.fixture(name="minutes_change", scope="class")
    def fixture_minutes_change(self, current_time: float) -> ConvergenceExample:
        """Create a fixture where the seconds change"""
        seconds = 48
        minutes = 28
        end_time = current_time + self.time_to_milli(seconds, minutes)
        return self.ConvergenceExample(
            start_time=current_time,
            end_time=end_time,
            elapsed_str=f"{minutes:d}m {seconds:.2f}s",
        )

    @pytest.fixture(name="hours_change", scope="class")
    def fixture_hours_change(self, current_time: float) -> ConvergenceExample:
        """Create a fixture where the seconds change"""
        seconds = 16
        minutes = 52
        hours = 4
        end_time = current_time + self.time_to_milli(seconds, minutes, hours)
        return self.ConvergenceExample(
            start_time=current_time,
            end_time=end_time,
            elapsed_str=f"{hours:d}hrs {minutes:d}m {seconds:.2f}s",
        )

    @pytest.mark.parametrize(
        "change_str",
        ["seconds_change", "minutes_change", "hours_change"],
    )
    def test_correct(self, change_str: str, request):
        """Test the correct result is gotten"""
        change = request.getfixturevalue(change_str)
        result = timing.time_taken(change.start_time, change.end_time)
        assert result == change.elapsed_str

    @pytest.mark.parametrize(
        "change_str",
        ["seconds_change", "minutes_change", "hours_change"],
    )
    def test_negative_change(self, change_str: str, request):
        """Test an error is raised when time taken is negative"""
        change = request.getfixturevalue(change_str)
        msg = "Elapsed time is negative"
        with pytest.raises(ValueError, match=msg):
            timing.time_taken(change.end_time, change.start_time)
