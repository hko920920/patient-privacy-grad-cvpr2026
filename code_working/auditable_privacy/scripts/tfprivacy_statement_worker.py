#!/usr/bin/env python3
"""Execute the official TensorFlow Privacy 0.9.0 statement function.

Run this worker only with the isolated interpreter created by
setup_tfprivacy_baseline.ps1. It loads the exact analysis module from the
installed wheel without importing TensorFlow Privacy's training-package
initializer, which would unnecessarily require the full TensorFlow stack.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import re
from pathlib import Path
from types import ModuleType
from typing import Any


POISSON_EPSILON_RE = re.compile(
    r"Epsilon assuming Poisson sampling \(\*\):\s+([^\s]+)"
)
ORDERED_EPSILON_RE = re.compile(
    r"Epsilon with each example occurring once per epoch:\s+([^\s]+)"
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def load_official_module(expected_sha256: str) -> tuple[ModuleType, Path, str]:
    distribution = importlib.metadata.distribution("tensorflow-privacy")
    site = Path(distribution.locate_file(""))
    module_path = (
        site
        / "tensorflow_privacy"
        / "privacy"
        / "analysis"
        / "compute_dp_sgd_privacy_lib.py"
    )
    if not module_path.is_file():
        raise RuntimeError(f"Official analysis module not found: {module_path}")
    module_sha256 = sha256_bytes(module_path.read_bytes())
    if module_sha256 != expected_sha256:
        raise RuntimeError(
            "TensorFlow Privacy module digest mismatch: "
            f"expected {expected_sha256}, found {module_sha256}"
        )
    spec = importlib.util.spec_from_file_location(
        "tfprivacy_official_statement_module",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load official module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, module_path.resolve(), module_sha256


def parse_epsilon(token: str) -> float | str:
    token = token.strip()
    if token.lower() == "inf":
        return "inf"
    return float(token)


def parse_statement(statement: str, expects_user: bool) -> dict[str, Any]:
    poisson = [parse_epsilon(value) for value in POISSON_EPSILON_RE.findall(statement)]
    ordered = [parse_epsilon(value) for value in ORDERED_EPSILON_RE.findall(statement)]
    expected_count = 2 if expects_user else 1
    if len(poisson) != expected_count or len(ordered) != expected_count:
        raise RuntimeError(
            "Could not parse the official statement deterministically: "
            f"poisson={poisson}, ordered={ordered}, expects_user={expects_user}"
        )
    return {
        "example_ordered_epsilon_printed": ordered[0],
        "example_poisson_epsilon_printed": poisson[0],
        "user_ordered_epsilon_printed": ordered[1] if expects_user else None,
        "user_poisson_epsilon_printed": poisson[1] if expects_user else None,
        "statement_reports_no_user_bound": (
            "No user-level privacy guarantee is possible without a bound"
            in statement
        ),
        "statement_mentions_add_remove": (
            "add-or-remove-one adjacency" in statement
        ),
        "statement_mentions_poisson_caveat": (
            "Poisson sampling is not usually done in training pipelines"
            in statement
        ),
    }


def validate_request(request: dict[str, Any]) -> None:
    required = {
        "case_id",
        "number_of_examples",
        "batch_size",
        "num_epochs",
        "noise_multiplier",
        "delta",
        "used_microbatching",
        "max_examples_per_user",
    }
    missing = sorted(required - set(request))
    if missing:
        raise ValueError(f"Missing request fields: {missing}")
    number = request["number_of_examples"]
    batch = request["batch_size"]
    if isinstance(number, bool) or not isinstance(number, int) or number <= 0:
        raise ValueError(f"Invalid number_of_examples: {number!r}")
    if isinstance(batch, bool) or not isinstance(batch, int) or not 0 < batch <= number:
        raise ValueError(f"Invalid batch_size: {batch!r}")


def run_request(module: ModuleType, request: dict[str, Any]) -> dict[str, Any]:
    validate_request(request)
    maximum = request["max_examples_per_user"]
    statement = module.compute_dp_sgd_privacy_statement(
        number_of_examples=int(request["number_of_examples"]),
        batch_size=int(request["batch_size"]),
        num_epochs=float(request["num_epochs"]),
        noise_multiplier=float(request["noise_multiplier"]),
        delta=float(request["delta"]),
        used_microbatching=bool(request["used_microbatching"]),
        max_examples_per_user=(None if maximum is None else int(maximum)),
        accountant_type=module.AccountantType.RDP,
    )
    parsed = parse_statement(statement, expects_user=maximum is not None)
    return {
        "case_id": str(request["case_id"]),
        "request": request,
        "statement": statement,
        "statement_sha256": sha256_text(statement),
        **parsed,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    document = json.loads(args.input.read_text(encoding="utf-8"))
    if document.get("schema_version") != "tfprivacy_statement_requests_v2.0":
        raise ValueError("Unsupported input schema_version")
    expected_module_sha256 = str(document["expected_module_sha256"])
    module, module_path, module_sha256 = load_official_module(
        expected_module_sha256
    )
    requests = document.get("requests")
    if not isinstance(requests, list) or not requests:
        raise ValueError("requests must be a non-empty list")
    case_ids = [str(request.get("case_id", "")) for request in requests]
    if len(case_ids) != len(set(case_ids)) or any(not value for value in case_ids):
        raise ValueError("request case_id values must be non-empty and unique")

    result = {
        "schema_version": "tfprivacy_statement_worker_output_v2.0",
        "tensorflow_privacy_version": importlib.metadata.version(
            "tensorflow-privacy"
        ),
        "dp_accounting_version": importlib.metadata.version("dp-accounting"),
        "module_path": (
            "tensorflow_privacy/privacy/analysis/"
            "compute_dp_sgd_privacy_lib.py"
        ),
        "module_sha256": module_sha256,
        "results": [run_request(module, request) for request in requests],
    }
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
