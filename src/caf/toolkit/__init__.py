from . import _version

__version__ = _version.get_versions()["version"]

# Alias
from caf.toolkit import pandas_utils
from caf.toolkit.config_base import BaseConfig
