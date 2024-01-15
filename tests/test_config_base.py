# -*- coding: utf-8 -*-
"""
Tests for the config_base module in caf.toolkit
"""
# Built-Ins
import dataclasses
import datetime
import pathlib
from pathlib import Path
from typing import Optional

# Third Party
import pytest

# pylint: disable=import-error
from pydantic import ValidationError

# Local Imports
from caf.toolkit import BaseConfig

# pylint: enable=import-error

# # # Fixture # # #


@dataclasses.dataclass
class SubClassTest:
    """Subclass to be included as a parameter in ConfigTestClass"""

    whole: int
    decimal: float


# pylint: disable=too-few-public-methods
class ConfigTestClass(BaseConfig):
    """Class created to test BaseConfig"""

    dictionary: dict[str, float]
    path: Path
    list: list[str]
    set: set[int]
    tuple: tuple[Path, Path]
    date_time: datetime.datetime
    sub: Optional[SubClassTest] = None
    default: bool = True
    option: Optional[int] = None


# pylint: enable=too-few-public-methods


@pytest.fixture(name="path", scope="session")
def fixture_dir(tmp_path_factory):
    """
    Temp path for test i/o
    Parameters
    ----------
    tmp_path_factory

    Returns
    -------
    None
    """
    path = tmp_path_factory.mktemp("dir")
    return path


@pytest.fixture(name="basic", scope="session")
def fixture_basic(path) -> ConfigTestClass:
    """
    Basic config for testing
    Parameters
    ----------
    path: Above fixture

    Returns
    -------
    conf (ConfigTestClass): A testing config
    """
    conf_dict = {"foo": 1.3, "bar": 3.6}
    conf_path = path / "basic"
    conf_list = ["far", "baz"]
    conf_set = [1, 2, 3]
    conf_tuple = tuple([(path / "tuple_1"), (path / "tuple_2")])
    conf_date_time = datetime.datetime(2000, 1, 1, 10, 30)
    conf_opt = 4
    conf = ConfigTestClass(
        dictionary=conf_dict,
        path=conf_path,
        list=conf_list,
        set=conf_set,
        tuple=conf_tuple,
        date_time=conf_date_time,
        option=conf_opt,
    )
    return conf


class DummyDatetime(datetime.datetime):
    """Dummy sub-class to return constant value for now."""

    @classmethod
    def now(cls, tz=None):
        "Constant time of 0."
        return cls.fromtimestamp(0, tz)


@pytest.fixture(name="datetime_now")
def fixture_monkeypatch_datetime_now(monkeypatch: pytest.MonkeyPatch) -> DummyDatetime:
    """Monkeypatch datetime.datetime to return a constant value for now method."""
    monkeypatch.setattr(datetime, "datetime", DummyDatetime)
    return DummyDatetime.now()


# # # TESTS # # #


class TestCreateConfig:
    """
    Class for testing basic creation of configs using the BaseConfig
    """

    @pytest.mark.parametrize(
        "param, type_iter",
        [
            ("dictionary", dict),
            ("path", Path),
            ("list", list),
            ("set", set),
            ("tuple", tuple),
            ("date_time", datetime.datetime),
            ("default", bool),
            ("option", int),
        ],
    )
    def test_type(self, basic, param, type_iter):
        """
        Tests that all parameters are of the expected type.
        Parameters
        ----------
        basic: the test config
        param: the config param being tested
        type_iter: the type each param is expected to be

        Returns
        -------
        None
        """
        val = basic.model_dump()[param]
        assert isinstance(val, type_iter)

    @pytest.mark.parametrize("param, type_iter", [("default", True), ("option", None)])
    def test_default(self, basic, param, type_iter):
        """
        Tests default values are correctly written
        Parameters
        ----------
        basic: the test config
        param: the config parameter being tested
        type_iter: the expected value of the given parameter

        Returns
        -------
        None
        """
        config = ConfigTestClass(
            dictionary=basic.dictionary,
            path=basic.path,
            list=basic.list,
            set=basic.set,
            tuple=basic.tuple,
            date_time=basic.date_time,
        )
        val = config.model_dump()[param]
        assert val == type_iter

    def test_wrong_type(self, basic):
        """
        Tests that the correct error is raised when the config is initialised
        with an incorrect type
        Parameters
        ----------
        basic: the test config. In this case the config is altered.

        Returns
        -------
        None
        """
        with pytest.raises(ValidationError, match="validation error for ConfigTestClass"):
            ConfigTestClass(
                dictionary=["a", "list"],
                path=basic.path,
                list=basic.list,
                set=basic.set,
                tuple=basic.tuple,
                date_time=basic.date_time,
            )


class TestYaml:
    """
    Class for testing configs being converted to and from yaml, as well as saved and loaded.
    """

    def test_to_from_yaml(self, basic):
        """
        Test that when a config is converted to yaml and back it remains identical
        Parameters
        ----------
        basic: the test config

        Returns
        -------
        None
        """
        yaml = basic.to_yaml()
        conf = ConfigTestClass.from_yaml(yaml)
        assert conf == basic

    def test_custom_sub(self, basic):
        """
        Test that custom subclasses are recognised and read correctly when
        converted to and from yaml
        Parameters
        ----------
        basic: test config

        Returns
        -------
        None
        """
        conf = basic
        conf.sub = SubClassTest(whole=3, decimal=5.7)
        yam = conf.to_yaml()
        assert isinstance(ConfigTestClass.from_yaml(yam).sub, SubClassTest)

    def test_save_load(self, basic, path):
        """
        Test that when a config is saved to a yaml file then read in again it
        remains identical
        Parameters
        ----------
        basic: the test config
        path: a tmp file path for the config to be saved to and loaded from

        Returns
        -------
        None
        """
        file_path = path / "save_test.yml"
        basic.save_yaml(file_path)
        assert ConfigTestClass.load_yaml(file_path) == basic


class TestExample:
    """Test writing the example file is correct."""

    def write_example(self, path_: pathlib.Path, comment_: Optional[str], /, **kwargs) -> str:
        """Run `ConfigTestClass.write_example` and read output."""
        example_file = path_ / "test_example.yml"
        ConfigTestClass.write_example(example_file, comment_=comment_, **kwargs)

        with open(example_file, "rt", encoding="utf-8") as file:
            return file.read()

    @pytest.mark.parametrize("comment", [None, "# config example comment"])
    def test_default_example(self, path: pathlib.Path, comment: Optional[str]) -> None:
        """Write example without descriptions."""
        example = self.write_example(path, comment)

        expected = (
            "dictionary: REQUIRED\n"
            "path: REQUIRED\n"
            "list: REQUIRED\n"
            "set: REQUIRED\n"
            "tuple: REQUIRED\n"
            "date_time: REQUIRED\n"
            "sub: OPTIONAL\n"
            "default: yes\n"
            "option: OPTIONAL\n"
        )

        if comment is not None:
            expected = comment + "\n" + expected

        assert example == expected, "Write example without descriptions"

    @pytest.mark.parametrize("comment", [None, "# config example comment"])
    def test_example(self, path: pathlib.Path, comment: Optional[str]) -> None:
        """Write example with descriptions."""
        example_values = dict(
            dictionary="This is a dictionary",
            path="This is a path",
            list="This is a list",
            set="This is a set",
            tuple="Two paths to files",
            date_time="This is a data and time",
            sub=SubClassTest("integer value", "decimal value"),
            default="This value defaults to true",
            option="This value is optional",
        )
        example = self.write_example(path, comment, **example_values)

        expected = (
            "dictionary: {dictionary}\n"
            "path: {path}\n"
            "list: {list}\n"
            "set: {set}\n"
            "tuple: {tuple}\n"
            "date_time: {date_time}\n"
            "sub:\n"
            "  whole: integer value\n"
            "  decimal: decimal value\n"
            "default: {default}\n"
            "option: {option}\n"
        ).format(**example_values)

        if comment is not None:
            expected = comment + "\n" + expected

        assert example == expected, "Write example with descriptions"


class TestConfigComments:
    """Test adding datetime and other comments to YAML config."""

    def test_datetime(
        self, tmp_path: pathlib.Path, basic: ConfigTestClass, datetime_now: datetime.datetime
    ) -> None:
        """Test the automatic datetime comment."""
        path = tmp_path / "test_datetime_comment.yml"
        basic.save_yaml(path)

        with open(path, "rt", encoding="utf-8") as file:
            written = file.read()

        yaml = basic.to_yaml()

        datetime_comment = (
            f"# ConfigTestClass config written on {datetime_now:%Y-%m-%d at %H:%M}"
        )
        assert written == datetime_comment + "\n" + yaml, "incorrect datetime comment"

    @pytest.mark.parametrize(
        ["comment", "formatted"],
        [
            (None, None),
            ("# short comment",) * 2,
            ("short comment missing #", "# short comment missing #"),
            (
                "longer comment missing hash without any formatting already done to "
                "it, still more comment getting longer and longer on a single line",
                "# longer comment missing hash without any formatting already done to it,"
                "\n# still more comment getting longer and longer on a single line",
            ),
            (
                "already\nformatted\ncomment\nwith\nnewlines",
                "# already formatted comment with newlines",
            ),
        ],
    )
    @pytest.mark.parametrize("format_", [True, False])
    def test_other_comment(
        self,
        basic: ConfigTestClass,
        tmp_path: pathlib.Path,
        comment: Optional[str],
        formatted: Optional[str],
        format_: bool,
    ) -> None:
        """Test the formatted and unformatted custom comments."""
        path = tmp_path / "test_other_comment.yml"
        basic.save_yaml(
            path, datetime_comment=False, other_comment=comment, format_comment=format_
        )

        with open(path, "rt", encoding="utf-8") as file:
            written = file.read()

        yaml = basic.to_yaml()

        if comment is None:
            assert written == yaml, "incorrect no comment"

        elif formatted is None:
            raise ValueError("formatted should not be None if comment is not None")

        elif format_:
            assert written == formatted + "\n" + yaml, "comment formatting error"

        else:
            comment_lines = [i if i.startswith("#") else f"# {i}" for i in comment.split("\n")]
            comment = "\n".join(comment_lines)

            assert written == comment + "\n" + yaml, "comment no formatting error"


# # # FUNCTIONS # # #
