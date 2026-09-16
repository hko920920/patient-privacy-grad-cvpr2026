from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from verify_v3_srswor_public_collection import (  # noqa: E402
    PublicSrsworVerificationError,
    load_hashed_payload,
    verify_public_collection,
)


EVIDENCE = (
    ROOT / "reports" / "v3_srswor_registered_5seed_v32_20260724"
)


def test_public_srswor_collection_verifies_without_private_inputs():
    report = verify_public_collection(EVIDENCE)
    assert report["status"] == "verified"
    assert report["dataset_count"] == 3
    assert report["run_count"] == 15
    assert report["model_roundtrips"] == 15
    assert report["public_metric_values"] == 60
    assert "private" not in report["scope"].split(";")[0]


def test_public_srswor_payload_forgery_is_rejected(tmp_path: Path):
    source = EVIDENCE / "public_collection_srswor_v3.json"
    forged = json.loads(source.read_text(encoding="utf-8"))
    forged["release_status"] = "privacy_release"
    target = tmp_path / source.name
    target.write_text(json.dumps(forged), encoding="utf-8")
    with pytest.raises(
        PublicSrsworVerificationError,
        match="Payload hash mismatch",
    ):
        load_hashed_payload(target, "public_collection_sha256")
