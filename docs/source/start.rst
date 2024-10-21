Quick Start
===========

CAF.toolkit is provided as a Python package and a command-line utility,
which requires no Python code written to use. The command-line utility
only aims to provide some of the commonly used functionality,
see :ref:`usage` for details.

CAF.toolkit can be installed from pip, conda-forge or pipx
(when using as a command-line utility).

``pip install caf.toolkit``

``conda install caf.toolkit -c conda-forge``

Pipx
----

When using caf.toolkit as a command-line utility the recommended method for installation
is pipx, as this handles installing the tool in a self-contained way. To install using pipx
first install pipx into your default Python environment using pip or conda, see
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
explain the functionality available within caf.toolkit. For a detailed look at the
package API see :ref:`Module API Documentation`.
