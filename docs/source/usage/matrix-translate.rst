Matrix Translation
==================

This section explains the CLI functionality for translating matrix
data from one zone system to another.

.. seealso::
    :ref:`translation` for translations of vector based data.

Zone translation can be called using one of the following sub-commands:

- matrix-translate: provide all inputs as arguments; or
- matrix-translate-config: provide the path to a config file containing inputs.

More details on both commands are explained below in :ref:`sub-commands`.

.. note::
    Both commands perform the exact same translation but they allow the
    inputs to be provided in slightly different ways.


Sub-Commands
------------

matrix-translate
^^^^^^^^^^^^^^^^

.. argparse::
    :module: caf.toolkit.__main__
    :func: _create_arg_parser
    :path: matrix-translate

matrix-translate-config
^^^^^^^^^^^^^^^^^^^^^^^

.. argparse::
    :module: caf.toolkit.__main__
    :func: _create_arg_parser
    :path: matrix-translate-config
