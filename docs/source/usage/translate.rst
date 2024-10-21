Translation
===========

This section explains the CLI functionality for translating vector
data from one zone system to another.

.. seealso::
    :ref:`matrix translation` for translations of demand (or other) matrices.

Zone translation can be called using one of the following sub-commands:

- translate: provide all inputs as arguments; or
- translate-config: provide the path to a config file containing inputs.

More details on both commands are explained below in :ref:`sub-commands`.

.. note::
    Both commands perform the exact same translation but they allow the
    inputs to be provided in slightly different ways.


Sub-Commands
------------

translate
^^^^^^^^^

.. argparse::
    :module: caf.toolkit.__main__
    :func: _create_arg_parser
    :path: translate

translate-config
^^^^^^^^^^^^^^^^

.. argparse::
    :module: caf.toolkit.__main__
    :func: _create_arg_parser
    :path: translate-config
