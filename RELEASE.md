# Release Notes

The MITs Utils codebase follows [Semantic Versioning](https://semver.org/); the convention
for most software products. In summary, this means the version numbers should be read in the
following way.

Given a version number MAJOR.MINOR.PATCH (e.g. 1.0.0), increment the:

- MAJOR version when you make incompatible API changes,
- MINOR version when you add functionality in a backwards compatible manner, and
- PATCH version when you make backwards compatible bug fixes.

Note that the main branch of this repository contains a work in progress, and  may **not**
contain a stable version of the codebase. We aim to keep the master branch stable, but for the
most stable versions, please see the
[releases](https://github.com/Transport-for-the-North/caf.toolkit/releases)
page on GitHub. A log of all patches made between versions can also be found
there.

Below, a brief summary of patches made since the previous version can be found.

### v0.1.4
 - Packaging system has been updated
   - Moved versioning to [verisoningit](https://github.com/jwodder/versioningit)
   - Updates to the packaging infrastructure to become compliant with [PEP-517](https://peps.python.org/pep-0517/) and [PEP-518](https://peps.python.org/pep-0518/)
   - Removed explicit references to namespace package CAF ([PEP-420](https://peps.python.org/pep-0420/))
   - Dependencies have been updated where necessary to meet new needs
   - (Nearly) All project config and data is stored in `pyproject.toml`
 - BaseConfig comments when saving config to YAML file
   - Added automatic datetime comment to the top of saved configs (parameter in method to disable)
   - Added optional comment parameter for adding custom comments to the top of the config file
