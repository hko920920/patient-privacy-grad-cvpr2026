"""Window-to-owner/event mapping utilities."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class WindowRecord:
    window_id: str
    owner_ids: tuple[str, ...]
    start: int
    end: int
    row_index: int
    label: str | None = None
    scenario: str = "default"

    @property
    def owner_id(self) -> str:
        return self.owner_ids[0]

    def to_canonical_dict(self) -> dict[str, object]:
        return {
            "window_id": self.window_id,
            "owner_ids": list(self.owner_ids),
            "start": self.start,
            "end": self.end,
            "row_index": self.row_index,
            "label": self.label,
            "scenario": self.scenario,
        }


class WindowMapping:
    """Parsed mapping from generated windows to owners and raw support."""

    def __init__(self, records: list[WindowRecord], source_path: Path | None = None):
        self.records = list(records)
        self.source_path = source_path
        if not self.records:
            raise ValueError("WindowMapping requires at least one record")

    @classmethod
    def from_csv(cls, path: str | Path) -> "WindowMapping":
        source = Path(path)
        with source.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                raise ValueError("Mapping CSV has no header")
            rows = list(reader)
        records: list[WindowRecord] = []
        for position, row in enumerate(rows):
            owners_raw = row.get("owner_ids") or row.get("owners") or row.get("owner_id")
            if owners_raw is None:
                raise ValueError("Mapping row missing owner_id/owner_ids")
            owner_ids = tuple(item.strip() for item in str(owners_raw).split(";") if item.strip())
            if not owner_ids:
                raise ValueError("Mapping row has empty owner attribution")
            start = int(float(row.get("start", 0) or 0))
            end = int(float(row.get("end", start + 1) or start + 1))
            if end <= start:
                raise ValueError(f"Invalid interval [{start}, {end})")
            row_index = int(float(row.get("row_index", position) or position))
            records.append(
                WindowRecord(
                    window_id=str(row.get("window_id") or row.get("id") or position),
                    owner_ids=owner_ids,
                    start=start,
                    end=end,
                    row_index=row_index,
                    label=(str(row["label"]) if row.get("label") not in {None, ""} else None),
                    scenario=str(row.get("scenario") or "default"),
                )
            )
        return cls(records, source)

    def subset(self, records: Iterable[WindowRecord]) -> "WindowMapping":
        return WindowMapping(list(records), self.source_path)

    def owners(self) -> list[str]:
        return sorted({owner for record in self.records for owner in record.owner_ids})

    def by_owner(self) -> dict[str, list[WindowRecord]]:
        groups: dict[str, list[WindowRecord]] = defaultdict(list)
        for record in self.records:
            for owner in record.owner_ids:
                groups[owner].append(record)
        return {owner: sorted(items, key=lambda r: (r.start, r.end, r.window_id)) for owner, items in groups.items()}

    def row_indices(self) -> list[int]:
        return [record.row_index for record in self.records]

    def owner_kappa(self) -> int:
        counts = Counter(owner for record in self.records for owner in record.owner_ids)
        return max(counts.values(), default=0)

    def event_kappa(self) -> int:
        counts: Counter[tuple[str, int]] = Counter()
        for record in self.records:
            for owner in record.owner_ids:
                for event_idx in range(record.start, record.end):
                    counts[(owner, event_idx)] += 1
        return max(counts.values(), default=0)

    def stats(self) -> dict[str, object]:
        owner_counts = np.array(
            list(Counter(owner for record in self.records for owner in record.owner_ids).values()),
            dtype=float,
        )
        attribution_counts = np.array([len(record.owner_ids) for record in self.records], dtype=float)
        return {
            "num_windows": len(self.records),
            "num_owners": len(self.owners()),
            "event_kappa": self.event_kappa(),
            "owner_kappa": self.owner_kappa(),
            "owner_windows_mean": float(owner_counts.mean()) if owner_counts.size else 0.0,
            "owner_windows_p95": float(np.percentile(owner_counts, 95)) if owner_counts.size else 0.0,
            "multi_owner_windows": int(sum(len(record.owner_ids) > 1 for record in self.records)),
            "max_owners_per_window": int(attribution_counts.max()) if attribution_counts.size else 0,
            "mean_owners_per_window": float(attribution_counts.mean()) if attribution_counts.size else 0.0,
        }

    def label_counts(self) -> dict[str, int]:
        counts = Counter(record.label or "__missing__" for record in self.records)
        return dict(sorted(counts.items()))

    def label_owner_counts(self) -> dict[str, int]:
        owners_by_label: dict[str, set[str]] = defaultdict(set)
        for record in self.records:
            label = record.label or "__missing__"
            owners_by_label[label].update(record.owner_ids)
        return {label: len(owners) for label, owners in sorted(owners_by_label.items())}

    def selection_diagnostics(self, selected: "WindowMapping") -> dict[str, object]:
        source_labels = self.label_counts()
        selected_labels = selected.label_counts()
        labels = sorted(set(source_labels) | set(selected_labels))
        source_total = max(1, sum(source_labels.values()))
        selected_total = max(1, sum(selected_labels.values()))
        label_tv = 0.5 * sum(
            abs(
                source_labels.get(label, 0) / source_total
                - selected_labels.get(label, 0) / selected_total
            )
            for label in labels
        )

        source_label_owners = self.label_owner_counts()
        selected_label_owners = selected.label_owner_counts()
        coverage_by_label = {
            label: (
                selected_label_owners.get(label, 0) / source_label_owners[label]
                if source_label_owners[label] > 0
                else 0.0
            )
            for label in sorted(source_label_owners)
        }
        return {
            "retained_windows": len(selected.records),
            "source_windows": len(self.records),
            "retained_window_fraction": len(selected.records) / max(1, len(self.records)),
            "retained_owners": len(selected.owners()),
            "source_owners": len(self.owners()),
            "retained_owner_fraction": len(selected.owners()) / max(1, len(self.owners())),
            "source_label_counts": source_labels,
            "selected_label_counts": selected_labels,
            "label_total_variation": float(label_tv),
            "source_label_owner_counts": source_label_owners,
            "selected_label_owner_counts": selected_label_owners,
            "label_owner_coverage": coverage_by_label,
            "min_label_owner_coverage": (
                float(min(coverage_by_label.values())) if coverage_by_label else 0.0
            ),
        }

    def canonical_hash(self) -> str:
        payload = [record.to_canonical_dict() for record in sorted(self.records, key=lambda r: r.window_id)]
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def file_hash(self) -> str:
        if self.source_path is None:
            return ""
        digest = hashlib.sha256()
        with self.source_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def write_csv(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "scenario",
                    "window_id",
                    "owner_id",
                    "owner_ids",
                    "start",
                    "end",
                    "row_index",
                    "label",
                ],
            )
            writer.writeheader()
            for record in self.records:
                writer.writerow(
                    {
                        "scenario": record.scenario,
                        "window_id": record.window_id,
                        "owner_id": record.owner_id,
                        "owner_ids": ";".join(record.owner_ids),
                        "start": record.start,
                        "end": record.end,
                        "row_index": record.row_index,
                        "label": record.label or "",
                    }
                )
