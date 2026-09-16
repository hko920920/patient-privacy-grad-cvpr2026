"""Contribution policies for owner-window aligned training."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import hashlib

import numpy as np

from unitdp.mapping import WindowMapping, WindowRecord


@dataclass
class ContributionPolicy:
    """Select eligible windows and per-step windows for each owner."""

    policy_type: str = "support_balanced_cap"
    max_windows_per_owner: int | None = None
    windows_per_owner_per_step: int | None = None
    seed: int = 0

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> "ContributionPolicy":
        return cls(
            policy_type=str(raw.get("type", "support_balanced_cap")),
            max_windows_per_owner=(
                int(raw["max_windows_per_owner"])
                if raw.get("max_windows_per_owner") not in {None, ""}
                else None
            ),
            windows_per_owner_per_step=(
                int(raw["windows_per_owner_per_step"])
                if raw.get("windows_per_owner_per_step") not in {None, ""}
                else None
            ),
            seed=int(raw.get("seed", 0)),
        )

    def select_mapping(self, mapping: WindowMapping) -> WindowMapping:
        groups = mapping.by_owner()
        selected: list[WindowRecord] = []
        for owner in sorted(groups):
            records = groups[owner]
            selected.extend(self._select_owner_records(records, owner))
        deduped = {record.window_id: record for record in selected}
        return WindowMapping(list(deduped.values()), mapping.source_path)

    def _select_owner_records(self, records: list[WindowRecord], owner: str) -> list[WindowRecord]:
        cap = self.max_windows_per_owner
        if cap is None or cap >= len(records):
            return list(records)
        if cap <= 0:
            return []

        ordered = sorted(records, key=lambda r: (r.start, r.end, r.window_id))
        if self.policy_type in {"first_k", "chronological_cap"}:
            return ordered[:cap]
        if self.policy_type == "random_cap":
            stable = hashlib.sha256(f"{self.seed}:{owner}".encode("utf-8")).digest()
            seed = int.from_bytes(stable[:8], "big") % (2**32)
            rng = np.random.default_rng(seed)
            positions = sorted(rng.choice(len(ordered), size=cap, replace=False).tolist())
            return [ordered[pos] for pos in positions]
        if self.policy_type == "public_label_balanced_cap":
            return self._label_balanced(ordered, cap)
        if self.policy_type in {"support_balanced_cap", "temporal_balanced_cap"}:
            if cap == 1:
                return [ordered[len(ordered) // 2]]
            positions = np.linspace(0, len(ordered) - 1, cap).round().astype(int)
            return [ordered[int(pos)] for pos in positions]
        raise ValueError(f"Unsupported contribution policy: {self.policy_type}")

    @staticmethod
    def _label_balanced(records: list[WindowRecord], cap: int) -> list[WindowRecord]:
        by_label: dict[str, deque[WindowRecord]] = defaultdict(deque)
        for record in records:
            by_label[record.label or ""] .append(record)
        labels = sorted(by_label)
        selected: list[WindowRecord] = []
        while len(selected) < cap and labels:
            next_labels = []
            for label in labels:
                if by_label[label] and len(selected) < cap:
                    selected.append(by_label[label].popleft())
                if by_label[label]:
                    next_labels.append(label)
            labels = next_labels
        return sorted(selected, key=lambda r: (r.start, r.end, r.window_id))

    def sample_for_step(
        self,
        records: list[WindowRecord],
        rng: np.random.Generator,
    ) -> list[WindowRecord]:
        budget = self.windows_per_owner_per_step
        if budget is None or budget >= len(records):
            return list(records)
        if budget <= 0:
            return []
        ordered = sorted(records, key=lambda r: (r.start, r.end, r.window_id))
        positions = rng.choice(len(ordered), size=budget, replace=False)
        return [ordered[int(pos)] for pos in sorted(positions.tolist())]
