Tool Usage
==========


Command-Line Interface
----------------------

CAF.toolkit provides a command-line interface (CLI) for the zone translation
functionality. The below details the basic usage and arguments for running from the command line,
the details for each function within the tool are outlined in :ref:`sub-commands`.

.. argparse::
    :module: caf.toolkit.__main__
    :func: _create_arg_parser
    :nosubcommands:

Sub-Commands
^^^^^^^^^^^^

The following functionality is accessible from caf.toolkit's CLI, each
piece of functionality provides a separate list of input parameters required
for running.

.. toctree::
    :maxdepth: 1

    translate
    matrix-translate
