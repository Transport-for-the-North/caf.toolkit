# -*- coding: utf-8 -*-
"""
Tests for the `log_helpers` module in caf.toolkit
"""
# Built-Ins

# Third Party
import pydantic
import pytest


# Local Imports
from caf.toolkit import LogHelper, TemporaryLogFile, ToolDetails, SystemInformation


# # # Fixture # # #


# # # Tests # # #
class TestToolDetails:
    """Test ToolDetails class validation."""

    @pytest.mark.parametrize(
        "version",
        ["0.0.4", "1.2.3", "10.20.30", "1.1.2-prerelease+meta", "1.1.2+meta", "1.0.0-alpha"],
    )
    @pytest.mark.parametrize(
        "url",
        ["https://www.github.com/", "http://www.github.com/", "http://github.com/", None],
    )
    def test_valid(self, version: str, url: str | None) -> None:
        """Test valid values of version and homepage / source URLs."""
        # Not testing different values for name because there's no validation on it
        # Ignoring mypy type stating str is incorrect type
        ToolDetails("test_name", version, url, url)  # type: ignore

    @pytest.mark.parametrize("url", [None, "http://github.com"])
    def test_str(self, url: str | None) -> None:
        """Test converting to formatted string with / without optional values."""
        name, version = "test1", "1.2.3"
        if url is None:
            # fmt: off
            correct = (
                "Tool Information\n"
                "----------------\n"
                "name    : test1\n"
                "version : 1.2.3"
            )

        else:
            # fmt: off
            correct = (
                "Tool Information\n"
                "----------------\n"
                "name       : test1\n"
                "version    : 1.2.3\n"
                "homepage   : http://github.com\n"
                "source_url : http://github.com"
            )

        assert str(ToolDetails(name, version, url, url)) == correct  # type: ignore

    @pytest.mark.parametrize("version", ["1", "1.2", "1.1.2+.123", "alpha"])
    def test_invalid_versions(self, version: str) -> None:
        """Test correctly raise ValidationError for invalid versions."""
        url = "https://github.com"
        with pytest.raises(pydantic.ValidationError):
            ToolDetails("test_name", version, url, url)  # type: ignore

    @pytest.mark.parametrize("url", ["github.com", "github", "www.github.com"])
    def test_invalid_urls(self, url: str) -> None:
        """Test correctly raise ValidationError for invalid homepage / source URLs."""
        with pytest.raises(pydantic.ValidationError):
            ToolDetails("test_name", "1.2.3", url, url)  # type: ignore
