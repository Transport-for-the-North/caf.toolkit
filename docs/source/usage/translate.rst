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


Validating Zone Relationships in Python
--------------------------------------

Some workflows require a particular relationship between the source and destination
zones. Use ``get_relationship()`` to inspect a correspondence and
``validate_relationship()`` to require an exact relationship:

.. code-block:: python

    import pandas as pd
    from caf.toolkit.translation import ZoneCorrespondence

    vector = pd.DataFrame({
        "source": ["A", "A", "B"],
        "destination": ["X", "Y", "Z"],
        "factor": [0.4, 0.6, 1.0],
    })
    correspondence = ZoneCorrespondence(vector, "source", "destination", "factor")
    assert correspondence.get_relationship() == "one-to-many"
    correspondence.validate_relationship("one-to-many")
    # correspondence.validate_relationship("one-to-one") would raise ValueError.

The four possible results are ``"one-to-one"``, ``"one-to-many"``,
``"many-to-one"`` and ``"many-to-many"``, in the source-to-destination direction.
Repeated identical pairs are counted once. All stored pairs count, including those
with a zero factor. Classification raises ``ValueError`` for empty correspondences
or missing zone IDs, and does not modify the data.

To validate a CSV when reading it, pass ``expected_relationship`` to
``ZoneCorrespondencePath.read()``, for example
``mapping_path.read(expected_relationship="many-to-one")``. If the actual
relationship differs, the error reports the expected and actual relationships.
Omitting this argument keeps the existing reading behavior without relationship
validation.
