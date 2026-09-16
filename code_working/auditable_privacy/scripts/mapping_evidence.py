"""Canonical mapping evidence utilities for the v2.0 certificate path.

The certificate validator must inspect the actual mapping export.  Hash strings
copied into two reports are not evidence that either report describes the file
being certified.  This module provides one canonical parser, digest, and exact
incidence computation shared by the diagnostic audit and the fail-closed
validator.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


@dataclass(frozen=True)
class MappingEvidenceIssue:
    code: str
    field: str
    message: str


@dataclass
class MappingEvidence:
    path: Path
    scenario: str
    mapping_file_sha256: str = ""
    selected_mapping_sha256: str = ""
    selected_record_count: int = 0
    event_kappa: Optional[int] = None
    owner_kappa: Optional[int] = None
    attribution_arity_max: Optional[int] = None
    generated_unit: str = ""
    issues: List[MappingEvidenceIssue] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.issues


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(64 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_owner_ids(row: Mapping[str, Any]) -> List[str]:
    raw = row.get("owner_ids") or row.get("owners") or row.get("owner_id")
    if raw is None or str(raw).strip() == "":
        raise ValueError("Mapping row missing owner_id/owner_ids")
    text = str(raw).replace("|", ";")
    owners = [item.strip() for item in text.split(";") if item.strip()]
    if not owners:
        raise ValueError("Mapping row has empty owner attribution")
    if len(set(owners)) != len(owners):
        raise ValueError("Mapping row repeats an owner identifier")
    return owners


def canonical_records_sha256(records: Iterable[Mapping[str, Any]]) -> str:
    canonical = []
    for record in records:
        owner_ids = record.get("owner_ids")
        if owner_ids is None:
            owner_ids = [record["owner_id"]]
        canonical.append(
            {
                "scenario": str(record.get("scenario", "")),
                "window_id": str(record.get("window_id", "")),
                "generated_unit": str(record.get("generated_unit", "")),
                "owner_ids": sorted(str(owner) for owner in owner_ids),
                "start": int(record["start"]),
                "end": int(record["end"]),
            }
        )
    canonical.sort(
        key=lambda row: (
            row["scenario"],
            row["window_id"],
            row["generated_unit"],
            ";".join(row["owner_ids"]),
            row["start"],
            row["end"],
        )
    )
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_mapping(path: Path) -> List[Dict[str, Any]]:
    """Read a mapping for diagnostic use.

    This preserves the historical audit defaults for absent scenario/window
    identifiers.  Positive v2.0 certification is stricter and is implemented
    by :func:`inspect_mapping_evidence`.
    """

    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"start", "end"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Mapping CSV missing required columns: {sorted(missing)}")
        if not ({"owner_id", "owner_ids", "owners"} & set(reader.fieldnames or [])):
            raise ValueError(
                "Mapping CSV missing required owner column: owner_id or owner_ids"
            )

        records: List[Dict[str, Any]] = []
        for index, row in enumerate(reader):
            owners = parse_owner_ids(row)
            records.append(
                {
                    "window_id": row.get("window_id") or str(index),
                    "owner_id": owners[0],
                    "owner_ids": owners,
                    "generated_unit": row.get("generated_unit") or "",
                    "start": int(row["start"]),
                    "end": int(row["end"]),
                    "scenario": row.get("scenario") or "mapping",
                    "stride": row.get("stride") or "",
                }
            )
    return records


def _add_issue(
    evidence: MappingEvidence, code: str, field_name: str, message: str
) -> None:
    evidence.issues.append(
        MappingEvidenceIssue(code=code, field=field_name, message=message)
    )


def _exact_event_kappa(records: Iterable[Mapping[str, Any]]) -> int:
    """Compute maximum half-open interval overlap without expanding events."""

    deltas: Dict[str, Counter[int]] = defaultdict(Counter)
    for record in records:
        start = int(record["start"])
        end = int(record["end"])
        for owner in record["owner_ids"]:
            deltas[str(owner)][start] += 1
            deltas[str(owner)][end] -= 1

    maximum = 0
    for owner_deltas in deltas.values():
        active = 0
        for position in sorted(owner_deltas):
            active += owner_deltas[position]
            maximum = max(maximum, active)
    return maximum


def _exact_owner_kappa(records: Iterable[Mapping[str, Any]]) -> int:
    counts: Counter[str] = Counter()
    for record in records:
        for owner in record["owner_ids"]:
            counts[str(owner)] += 1
    return max(counts.values(), default=0)


def inspect_mapping_evidence(path: Path, scenario: str) -> MappingEvidence:
    """Inspect and recompute the actual mapping file used by a certificate.

    Certification requires explicit ``scenario`` and ``window_id`` columns,
    valid nonempty intervals and attribution, unique window identifiers within
    the selected scenario, and at least one selected record.
    """

    resolved = path.resolve()
    evidence = MappingEvidence(path=resolved, scenario=str(scenario))
    if not resolved.is_file():
        _add_issue(
            evidence,
            "MAPPING_EVIDENCE_FILE_MISSING",
            "mapping_file",
            f"Mapping evidence file does not exist: {resolved}",
        )
        return evidence

    try:
        evidence.mapping_file_sha256 = file_sha256(resolved)
        with resolved.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
            if len(fieldnames) != len(set(fieldnames)):
                _add_issue(
                    evidence,
                    "MAPPING_DUPLICATE_COLUMNS",
                    "mapping_file.header",
                    "Mapping CSV contains duplicate column names.",
                )
            required = {
                "scenario",
                "window_id",
                "generated_unit",
                "start",
                "end",
            }
            missing = sorted(required - set(fieldnames))
            if missing:
                _add_issue(
                    evidence,
                    "MAPPING_REQUIRED_COLUMNS_MISSING",
                    "mapping_file.header",
                    f"Missing required columns: {missing}.",
                )
            if not ({"owner_id", "owner_ids", "owners"} & set(fieldnames)):
                _add_issue(
                    evidence,
                    "MAPPING_OWNER_COLUMN_MISSING",
                    "mapping_file.header",
                    "Mapping CSV needs owner_id, owner_ids, or owners.",
                )
            raw_rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        _add_issue(
            evidence,
            "MAPPING_EVIDENCE_READ_FAILED",
            "mapping_file",
            str(exc),
        )
        return evidence

    selected_rows = [row for row in raw_rows if str(row.get("scenario") or "") == scenario]
    if not selected_rows:
        _add_issue(
            evidence,
            "MAPPING_SCENARIO_EMPTY",
            "mapping_file.scenario",
            f"No rows match scenario {scenario!r}.",
        )
        return evidence

    records: List[Dict[str, Any]] = []
    seen_window_ids = set()
    generated_units = set()
    for row_index, row in enumerate(selected_rows, start=1):
        field_prefix = f"mapping_file.rows[{row_index}]"
        window_id = str(row.get("window_id") or "").strip()
        if not window_id:
            _add_issue(
                evidence,
                "MAPPING_WINDOW_ID_MISSING",
                f"{field_prefix}.window_id",
                "Every certifiable mapping row needs an explicit window_id.",
            )
        elif window_id in seen_window_ids:
            _add_issue(
                evidence,
                "MAPPING_WINDOW_ID_DUPLICATE",
                f"{field_prefix}.window_id",
                f"Duplicate window_id within scenario: {window_id!r}.",
            )
        seen_window_ids.add(window_id)

        try:
            start = int(str(row.get("start") or ""))
            end = int(str(row.get("end") or ""))
        except ValueError:
            _add_issue(
                evidence,
                "MAPPING_INTERVAL_NOT_INTEGER",
                field_prefix,
                "start and end must be base-10 integers.",
            )
            continue
        if end <= start:
            _add_issue(
                evidence,
                "MAPPING_INTERVAL_INVALID",
                field_prefix,
                f"Expected end > start, received start={start}, end={end}.",
            )
            continue

        try:
            owners = parse_owner_ids(row)
        except ValueError as exc:
            _add_issue(
                evidence,
                "MAPPING_OWNER_ATTRIBUTION_INVALID",
                field_prefix,
                str(exc),
            )
            continue

        generated_unit = str(row.get("generated_unit") or "").strip()
        if generated_unit not in {"window", "event", "owner"}:
            _add_issue(
                evidence,
                "MAPPING_GENERATED_UNIT_INVALID",
                f"{field_prefix}.generated_unit",
                "generated_unit must be exactly window, event, or owner.",
            )
            continue
        generated_units.add(generated_unit)

        records.append(
            {
                "scenario": scenario,
                "window_id": window_id,
                "generated_unit": generated_unit,
                "owner_id": owners[0],
                "owner_ids": owners,
                "start": start,
                "end": end,
            }
        )

    evidence.selected_record_count = len(selected_rows)
    if len(generated_units) > 1:
        _add_issue(
            evidence,
            "MAPPING_GENERATED_UNIT_MIXED",
            "mapping_file.generated_unit",
            f"Selected scenario mixes generated units: {sorted(generated_units)}.",
        )
    if evidence.issues:
        return evidence

    evidence.selected_mapping_sha256 = canonical_records_sha256(records)
    evidence.event_kappa = _exact_event_kappa(records)
    evidence.owner_kappa = _exact_owner_kappa(records)
    evidence.attribution_arity_max = max(
        (len(record["owner_ids"]) for record in records), default=0
    )
    evidence.generated_unit = next(iter(generated_units))
    return evidence
