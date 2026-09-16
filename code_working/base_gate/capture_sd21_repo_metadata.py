#!/usr/bin/env python3
"""Capture reproducible Hugging Face metadata for the exact SD 2.1 mirror gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi


MIRROR_ID = "Manojb/stable-diffusion-2-1-base"
MIRROR_REVISION = "0094d483a120f3f33dafbd187ea4aa60d10de75c"
UPSTREAM_ID = "stabilityai/stable-diffusion-2-1-base"
CORROBORATING_REPOSITORIES = (
    "sd2-community/stable-diffusion-2-1-base",
    "patrickvonplaten/v2-1-base",
)
CORROBORATION_PATHS = (
    "text_encoder/model.fp16.safetensors",
    "unet/diffusion_pytorch_model.fp16.safetensors",
    "vae/diffusion_pytorch_model.fp16.safetensors",
    "v2-1_512-ema-pruned.safetensors",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def lfs_metadata(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    return {
        "size": getattr(value, "size", None),
        "sha256": getattr(value, "sha256", None),
        "pointer_size": getattr(value, "pointer_size", None),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    api = HfApi()
    info = api.model_info(MIRROR_ID, revision=MIRROR_REVISION, files_metadata=True)
    commits = api.list_repo_commits(MIRROR_ID, repo_type="model")
    try:
        upstream = api.model_info(UPSTREAM_ID)
        upstream_status: dict[str, Any] = {
            "status": "ACCESSIBLE",
            "resolved_id": upstream.id,
            "revision": upstream.sha,
        }
    except Exception as exc:
        response = getattr(exc, "response", None)
        upstream_status = {
            "status": "UNAVAILABLE",
            "exception_type": type(exc).__name__,
            "http_status": getattr(response, "status_code", None),
        }

    corroboration = []
    for repo_id in CORROBORATING_REPOSITORIES:
        comparison = api.model_info(repo_id, files_metadata=True)
        hashes = {
            sibling.rfilename: getattr(getattr(sibling, "lfs", None), "sha256", None)
            for sibling in comparison.siblings
        }
        corroboration.append(
            {
                "id": comparison.id,
                "resolved_revision": comparison.sha,
                "hashes": {path: hashes.get(path) for path in CORROBORATION_PATHS},
            }
        )

    payload = {
        "schema": "sd21-huggingface-repository-metadata/v1",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "mirror": {
            "id": info.id,
            "requested_revision": MIRROR_REVISION,
            "resolved_revision": info.sha,
            "private": info.private,
            "gated": getattr(info, "gated", None),
            "last_modified": str(info.last_modified),
            "license": (info.card_data or {}).get("license") if info.card_data else None,
            "commits": [
                {
                    "commit_id": commit.commit_id,
                    "title": commit.title,
                    "created_at": str(commit.created_at),
                    "authors": commit.authors,
                }
                for commit in commits
            ],
            "files": [
                {
                    "path": sibling.rfilename,
                    "size": getattr(sibling, "size", None),
                    "blob_id": getattr(sibling, "blob_id", None),
                    "lfs": lfs_metadata(getattr(sibling, "lfs", None)),
                }
                for sibling in info.siblings
            ],
        },
        "claimed_upstream": {
            "id": UPSTREAM_ID,
            **upstream_status,
        },
        "cross_repository_hash_corroboration": corroboration,
        "interpretation": (
            "The mirror commit title records a clone from the named Stability AI repository, "
            "but the accessible mirror is still a third-party repository. Exact revision and "
            "file hashes, rather than the mutable model name, are the executable identity."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"REPO_METADATA_SHA256={sha256_file(args.output)}")


if __name__ == "__main__":
    main()
