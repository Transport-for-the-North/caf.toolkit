Quick Start
===========

CAF.toolkit is provided as a Python package and a command-line utility.
The command-line utility aims to make some of the commonly used functionality 
available without needing to use Python code, see :ref:`usage` for details.

CAF.toolkit can be installed from pip, conda-forge or pipx
(when using as a command-line utility).

Pip
---
Installing through pip is easy and can be done in one command:
``pip install caf.toolkit``

conda-forge
-----------
Installing through conda-forge is easy and can be done in one command:
``conda install caf.toolkit -c conda-forge``

Pipx
----

`Pipx <https://pipx.pypa.io/stable/>`__ is the recommended way to use caf.toolkit as a utility.
It handles installing the tool in its own container, and makes it easy to access from a terminal.

First install pipx into your default Python environment using pip or conda, see
`Pipx's installation instructions <https://pipx.pypa.io/stable/installation/>`__ for more details.

Once pipx is installed and setup caf.toolkit can be installed using ``pipx install caf.toolkit``,
this should make it available in command-line anywhere using ``caf.toolkit ...``.


Usage
-----

Using caf.toolkit as a command-line tool can be done in one of two ways:

- Called directly (if installed using pipx) ``caf.toolkit ...``
- Ran as a Python module ``python -m caf.toolkit ...``

Either method provides the same functionality and arguments, details of which
can be found in :ref:`tool usage`.

Python
^^^^^^

When using caf.toolkit functionality within Python the recommended alias is ``ctk``:

.. code:: python

    import caf.toolkit as ctk

The :ref:`user guide` contains :ref:`tutorials` and :ref:`code examples`, which
explain available functionality. For a detailed look at the
package API see :ref:`Module API Documentation`.
