"""Classifying and validating zone mappings must not change their data."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import pandas as pd
import pytest

from caf.toolkit import translation

if TYPE_CHECKING:
    from pathlib import Path

Relationship = Literal["one-to-one", "one-to-many", "many-to-one", "many-to-many"]
ZoneId = str | int | None


def _correspondence(pairs: list[tuple[ZoneId, ZoneId]]) -> translation.ZoneCorrespondence:
    frame = pd.DataFrame(pairs, columns=["source", "destination"])
    frame["factors"] = 0.25
    return translation.ZoneCorrespondence(frame, "source", "destination", "factors")


@pytest.mark.parametrize(
    ("pairs", "expected"),
    [
        ([("A", "X")], "one-to-one"),
        ([("A", "X"), ("B", "Y")], "one-to-one"),
        ([("A", "X"), ("A", "Y")], "one-to-many"),
        ([("A", "X"), ("B", "X")], "many-to-one"),
        ([("A", "X"), ("A", "Y"), ("B", "X")], "many-to-many"),
        # Branching can occur in disconnected parts of the mapping.
        ([("A", "X"), ("A", "Y"), ("B", "Z"), ("C", "Z")], "many-to-many"),
        ([(1, 10), (1, 20)], "one-to-many"),
        ([(1, 10), (2, 10)], "many-to-one"),
        ([("A", "X"), ("A", "X"), ("B", "Y")], "one-to-one"),
        ([("A", "X"), ("A", "X"), ("A", "Y")], "one-to-many"),
    ],
)
def test_get_relationship(pairs: list[tuple[ZoneId, ZoneId]], expected: Relationship) -> None:
    """Classify distinct pairs, regardless of input order or identifier type."""
    correspondence = _correspondence(pairs)
    original = correspondence.vector.copy(deep=True)
    assert correspondence.get_relationship() == expected
    assert _correspondence(list(reversed(pairs))).get_relationship() == expected
    pd.testing.assert_frame_equal(correspondence.vector, original)


@pytest.mark.parametrize(
    ("pairs", "message"),
    [
        ([], "empty"),
        ([(None, "X")], "missing zone IDs"),
        ([("A", None)], "missing zone IDs"),
        ([("A", "X"), (None, "Y")], "missing zone IDs"),
    ],
)
def test_unclassifiable_mapping(pairs: list[tuple[ZoneId, ZoneId]], message: str) -> None:
    """Do not silently treat missing IDs or an empty table as one-to-one."""
    correspondence = _correspondence(pairs)
    with pytest.raises(ValueError, match=message):
        correspondence.get_relationship()
    with pytest.raises(ValueError, match=message):
        correspondence.validate_relationship("one-to-one")


def test_zero_weight_pair_counts_and_relationship_is_not_cached() -> None:
    """Classify stored pairs, and reflect edits to the mutable vector."""
    correspondence = _correspondence([("A", "X"), ("B", "Y")])
    correspondence.vector.loc[1, "factors"] = 0
    assert correspondence.get_relationship() == "one-to-one"
    correspondence.vector.loc[1, "source"] = "A"
    assert correspondence.get_relationship() == "one-to-many"


@pytest.mark.parametrize(
    "relationship", ["one-to-one", "one-to-many", "many-to-one", "many-to-many"]
)
def test_validate_relationship(relationship: Relationship) -> None:
    """Accept the exact relationship and reject the other three."""
    pairs: dict[Relationship, list[tuple[ZoneId, ZoneId]]] = {
        "one-to-one": [("A", "X"), ("B", "Y")],
        "one-to-many": [("A", "X"), ("A", "Y")],
        "many-to-one": [("A", "X"), ("B", "X")],
        "many-to-many": [("A", "X"), ("A", "Y"), ("B", "X")],
    }
    correspondence = _correspondence(pairs[relationship])
    for expected in pairs:
        if expected == relationship:
            correspondence.validate_relationship(expected)
        else:
            with pytest.raises(ValueError, match=f"Expected {expected}.*{relationship}"):
                correspondence.validate_relationship(expected)


def test_validate_relationship_rejects_unknown_name() -> None:
    """A misspelled requirement should report an invalid relationship name."""
    correspondence = _correspondence([("A", "X")])
    with pytest.raises(ValueError, match="Unknown zone relationship"):
        correspondence.validate_relationship("one_to_one")  # type: ignore[arg-type]


@pytest.mark.parametrize("generic_names", [False, True])
@pytest.mark.parametrize("factors_mandatory", [False, True])
def test_read_validates_relationship(
    tmp_path: Path, generic_names: bool, factors_mandatory: bool
) -> None:
    """CSV reading can opt into relationship validation without changing defaults."""
    path = tmp_path / "mapping.csv"
    frame = pd.DataFrame({"source": ["A", "A"], "destination": ["X", "Y"]})
    if factors_mandatory:
        frame["factors"] = 0.5
    frame.to_csv(path, index=False)
    mapping = translation.ZoneCorrespondencePath(
        path,
        "source",
        "destination",
        "factors" if factors_mandatory else None,
    )
    original = mapping.read(
        factors_mandatory=factors_mandatory, generic_column_names=generic_names
    )
    result = mapping.read(
        expected_relationship="one-to-many",
        factors_mandatory=factors_mandatory,
        generic_column_names=generic_names,
    )
    pd.testing.assert_frame_equal(result.vector, original.vector)
    with pytest.raises(ValueError, match=r"Expected one-to-one.*one-to-many"):
        mapping.read(
            expected_relationship="one-to-one",
            factors_mandatory=factors_mandatory,
            generic_column_names=generic_names,
        )


def test_read_without_requirement_keeps_empty_mapping(tmp_path: Path) -> None:
    """Relationship checks remain opt-in, including for empty CSV files."""
    path = tmp_path / "empty.csv"
    pd.DataFrame(columns=["source", "destination"]).to_csv(path, index=False)
    mapping = translation.ZoneCorrespondencePath(path, "source", "destination")
    assert mapping.read(factors_mandatory=False).vector.empty
    with pytest.raises(ValueError, match="empty"):
        mapping.read(factors_mandatory=False, expected_relationship="one-to-one")
