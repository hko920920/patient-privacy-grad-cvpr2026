from __future__ import annotations

import sys
import unittest
from collections import Counter
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import backblaze_v2_evidence as backblaze  # noqa: E402
import sepsis2019_v2_evidence as sepsis  # noqa: E402


class SecondaryEvidenceMathTests(unittest.TestCase):
    def test_contiguous_runs_preserve_calendar_gaps(self) -> None:
        mask = (1 << 0) | (1 << 1) | (1 << 3) | (1 << 4) | (1 << 5)
        self.assertEqual(backblaze.contiguous_run_lengths(mask, 6), (2, 3))

    def test_ordered_and_calendar_window_semantics_diverge(self) -> None:
        profiles = Counter({(2, 3): 1})
        ordered = backblaze.WindowPolicy(
            scenario="ordered",
            semantics="ordered_snapshot",
            window_length=3,
            stride=1,
            policy="dense",
        )
        calendar = backblaze.WindowPolicy(
            scenario="calendar",
            semantics="calendar_contiguous",
            window_length=3,
            stride=1,
            policy="dense",
        )
        ordered_row = backblaze.summarize_policy(profiles, ordered)
        calendar_row = backblaze.summarize_policy(profiles, calendar)
        self.assertEqual(ordered_row["num_windows"], 3)
        self.assertEqual(calendar_row["num_windows"], 1)
        self.assertEqual(ordered_row["observed_event_kappa"], 3)
        self.assertEqual(calendar_row["observed_event_kappa"], 1)

    def test_nonoverlap_has_observed_kappa_one_but_candidate_k_two(self) -> None:
        profiles = Counter({(7,): 1})
        policy = backblaze.WindowPolicy(
            scenario="non_overlap",
            semantics="calendar_contiguous",
            window_length=7,
            stride=7,
            policy="non_overlap",
        )
        row = backblaze.summarize_policy(profiles, policy)
        self.assertEqual(row["observed_event_kappa"], 1)
        self.assertEqual(row["candidate_event_stability_k"], 2)
        self.assertFalse(row["same_number_event_dp_reuse_allowed"])

    def test_half_open_interval_kappa(self) -> None:
        self.assertEqual(
            sepsis.exact_interval_kappa([(0, 6), (6, 12), (12, 18)]),
            1,
        )
        self.assertEqual(
            sepsis.exact_interval_kappa([(0, 6), (3, 9), (6, 12)]),
            2,
        )

    def test_label_balancing_changes_selected_window_identity(self) -> None:
        before = [(0, 6, "control"), (6, 12, "control"), (12, 18, "sepsis")]
        after = [(0, 6, "control"), (6, 12, "control"), (12, 18, "control")]
        selected_before = sepsis.select_windows(
            before,
            "owner_label_balanced_cap",
            2,
        )
        selected_after = sepsis.select_windows(
            after,
            "owner_label_balanced_cap",
            2,
        )
        self.assertEqual(
            [(start, end) for start, end, _ in selected_before],
            [(0, 6), (12, 18)],
        )
        self.assertEqual(
            [(start, end) for start, end, _ in selected_after],
            [(0, 6), (6, 12)],
        )


if __name__ == "__main__":
    unittest.main()
