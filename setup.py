from setuptools import setup
import versioneer

PACKAGE_NAME = "mits_utils"

setup(
    name=PACKAGE_NAME,
    url="https://github.com/Transport-for-the-North/MITs-Utils",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
)
