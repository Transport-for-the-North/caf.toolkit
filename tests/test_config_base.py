# -*- coding: utf-8 -*-
"""
Tests for the config_base module in caf.toolkit
"""
# Built-Ins
from pathlib import Path
import dataclasses

# Third Party
import pytest

# pylint: disable=import-error
from pydantic import ValidationError

# Local Imports
from caf.toolkit import BaseConfig


# pylint: enable=import-error
# # # Fixture # # #


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
def fixture_basic(path):
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
    conf_opt = 4
    conf = ConfigTestClass(
        dictionary=conf_dict,
        path=conf_path,
        list=conf_list,
        set=conf_set,
        tuple=conf_tuple,
        option=conf_opt,
    )
    return conf


# # # CLASSES # # #


@dataclasses.dataclass
class TestSubClass:
    """
    Subclass to be included as a parameter in ConfigTestClass
    """

    whole: int
    decimal: float


# pylint: disable=too-few-public-methods
class ConfigTestClass(BaseConfig):
    """
    Class created to test BaseConfig
    """

    dictionary: dict[str, float]
    path: Path
    list: list[str]
    set: set[int]
    tuple: tuple[Path, Path]
    sub: TestSubClass = None
    default: bool = True
    option: int = None


# pylint: enable=too-few-public-methods


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
        val = basic.dict()[param]
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
        )
        val = config.dict()[param]
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
        conf.sub = TestSubClass(whole=3, decimal=5.7)
        yam = conf.to_yaml()
        assert isinstance(ConfigTestClass.from_yaml(yam).sub, TestSubClass)

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


# # # FUNCTIONS # # #
