"""Metamorphic tests for canonical mapping and exact incidence evidence."""

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from mapping_evidence import inspect_mapping_evidence  # noqa: E402


FIELDS = [
    "scenario",
    "window_id",
    "generated_unit",
    "owner_id",
    "owner_ids",
    "start",
    "end",
]


def row(
    window_id: str,
    owner: str,
    start: int,
    end: int,
    *,
    generated_unit: str = "window",
) -> dict[str, str]:
    return {
        "scenario": "meta",
        "window_id": window_id,
        "generated_unit": generated_unit,
        "owner_id": owner,
        "owner_ids": owner,
        "start": str(start),
        "end": str(end),
    }


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


class MappingEvidenceMetamorphicTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def inspect(self, name: str, rows: list[dict[str, str]]):
        path = self.root / name
        write_rows(path, rows)
        return inspect_mapping_evidence(path, "meta")

    def test_row_permutation_preserves_selected_digest_and_incidence(self) -> None:
        rows = [
            row("w0", "u0", 0, 2),
            row("w1", "u0", 1, 3),
            row("w2", "u1", 0, 2),
        ]
        first = self.inspect("first.csv", rows)
        second = self.inspect("second.csv", list(reversed(rows)))
        self.assertTrue(first.valid)
        self.assertTrue(second.valid)
        self.assertNotEqual(first.mapping_file_sha256, second.mapping_file_sha256)
        self.assertEqual(
            first.selected_mapping_sha256, second.selected_mapping_sha256
        )
        self.assertEqual(first.event_kappa, second.event_kappa)
        self.assertEqual(first.owner_kappa, second.owner_kappa)

    def test_adding_overlapping_record_increases_event_distance(self) -> None:
        base = self.inspect("base.csv", [row("w0", "u0", 0, 2)])
        overlap = self.inspect(
            "overlap.csv",
            [row("w0", "u0", 0, 2), row("w1", "u0", 0, 2)],
        )
        self.assertEqual(base.event_kappa, 1)
        self.assertEqual(overlap.event_kappa, 2)

    def test_half_open_adjacent_intervals_do_not_overlap(self) -> None:
        evidence = self.inspect(
            "adjacent.csv",
            [row("w0", "u0", 0, 2), row("w1", "u0", 2, 4)],
        )
        self.assertTrue(evidence.valid)
        self.assertEqual(evidence.event_kappa, 1)
        self.assertEqual(evidence.owner_kappa, 2)

    def test_adding_owner_record_increases_owner_distance_only_as_expected(self) -> None:
        base = self.inspect(
            "owner-base.csv",
            [row("w0", "u0", 0, 2), row("w1", "u1", 0, 2)],
        )
        added = self.inspect(
            "owner-added.csv",
            [
                row("w0", "u0", 0, 2),
                row("w1", "u1", 0, 2),
                row("w2", "u0", 2, 4),
            ],
        )
        self.assertEqual(base.owner_kappa, 1)
        self.assertEqual(added.owner_kappa, 2)
        self.assertEqual(base.event_kappa, 1)
        self.assertEqual(added.event_kappa, 1)

    def test_owner_renaming_preserves_multiplicities(self) -> None:
        first = self.inspect(
            "owner-names-a.csv",
            [row("w0", "alice", 0, 2), row("w1", "alice", 1, 3)],
        )
        second = self.inspect(
            "owner-names-b.csv",
            [row("w0", "z9", 0, 2), row("w1", "z9", 1, 3)],
        )
        self.assertEqual(first.event_kappa, second.event_kappa)
        self.assertEqual(first.owner_kappa, second.owner_kappa)
        self.assertNotEqual(
            first.selected_mapping_sha256, second.selected_mapping_sha256
        )

    def test_mixed_generated_units_are_invalid(self) -> None:
        evidence = self.inspect(
            "mixed.csv",
            [
                row("w0", "u0", 0, 2, generated_unit="window"),
                row("w1", "u1", 0, 1, generated_unit="event"),
            ],
        )
        self.assertFalse(evidence.valid)
        self.assertIn(
            "MAPPING_GENERATED_UNIT_MIXED",
            [issue.code for issue in evidence.issues],
        )

    def test_duplicate_record_identity_is_invalid(self) -> None:
        evidence = self.inspect(
            "duplicate.csv",
            [row("w0", "u0", 0, 2), row("w0", "u0", 2, 4)],
        )
        self.assertFalse(evidence.valid)
        self.assertIn(
            "MAPPING_WINDOW_ID_DUPLICATE",
            [issue.code for issue in evidence.issues],
        )

    def test_nonpositive_interval_is_invalid(self) -> None:
        evidence = self.inspect(
            "invalid-interval.csv",
            [row("w0", "u0", 2, 2)],
        )
        self.assertFalse(evidence.valid)
        self.assertIn(
            "MAPPING_INTERVAL_INVALID",
            [issue.code for issue in evidence.issues],
        )


if __name__ == "__main__":
    unittest.main()
