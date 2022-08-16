from setuptools import setup
import versioneer

setup(
    name="caf.toolkit",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
)
