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

### Next Release Notes

This release updated the long_to_wide/wide_to_long methods in df_handling to work 
more efficiently. It also simplifies some translations.

 - Pandas vector translations now all work natively in pandas, rather than falling back to numpy for single vector translations
 - Various pandas conversion methods also work natively in pandas rather than falling back to numpy. They also use multiindexing and in some cases enforce this.