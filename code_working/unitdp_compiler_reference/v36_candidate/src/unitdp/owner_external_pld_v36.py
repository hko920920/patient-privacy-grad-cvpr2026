"""Research executor wrapper for the external-PLD V3.6 candidate."""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

import numpy as np

from unitdp.compiler_external_pld_v36 import (
    CompiledOwnerExternalPldAllocationRouteV36,
    ExternalPldContractV36Error,
)
from unitdp.owner_random_allocation_v4 import (
    OwnerRandomAllocationExecutionResultV4,
    OwnerRandomAllocationExecutionV4Error,
    train_owner_random_allocation_fixed_v4,
)
from unitdp.release_artifacts import assert_public_certificate_redacted
from unitdp.source_bundle_external_pld_v36 import (
    SourceBundleExternalPldV36Error,
    execution_source_bundle_external_pld_v36,
    execution_source_bundle_sha256_external_pld_v36,
)


PUBLIC_EXECUTION_SCHEMA_EXTERNAL_PLD_V36 = (
    "unitdp.owner_external_pld_execution_public.v36"
)
PRIVATE_EXECUTION_SCHEMA_EXTERNAL_PLD_V36 = (
    "unitdp.owner_external_pld_execution_private.v36"
)


class OwnerExternalPldExecutionV36Error(RuntimeError):
    """Raised when H execution cannot preserve both nested seals."""


def _sha256_payload(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class OwnerExternalPldExecutionResultV36:
    """Nested A result plus H-specific public and private records."""

    def __init__(
        self,
        *,
        route: CompiledOwnerExternalPldAllocationRouteV36,
        base_result: OwnerRandomAllocationExecutionResultV4,
        public_record: dict[str, Any],
        private_record: dict[str, Any],
    ):
        self.route = route
        self.base_result = base_result
        self.model = base_result.model
        self._public_record = copy.deepcopy(public_record)
        self._private_record = copy.deepcopy(private_record)

    def assert_model_integrity(self) -> None:
        self.base_result.assert_model_integrity()

    def public_payload(self) -> dict[str, Any]:
        self.assert_model_integrity()
        try:
            self.route.assert_private_execution_integrity()
        except ExternalPldContractV36Error as exc:
            raise OwnerExternalPldExecutionV36Error(
                f"H route integrity failed after execution: {exc}"
            ) from exc
        public = copy.deepcopy(self._public_record)
        reported = str(public.pop("public_execution_sha256"))
        if _sha256_payload(public) != reported:
            raise OwnerExternalPldExecutionV36Error(
                "Public H execution hash is inconsistent"
            )
        public["public_execution_sha256"] = reported
        assert_public_certificate_redacted(public)
        return public

    def private_manifest(self) -> dict[str, Any]:
        self.assert_model_integrity()
        private = copy.deepcopy(self._private_record)
        reported = str(private.pop("private_manifest_sha256"))
        if _sha256_payload(private) != reported:
            raise OwnerExternalPldExecutionV36Error(
                "Private H manifest hash is inconsistent"
            )
        private["private_manifest_sha256"] = reported
        return private


def train_owner_external_pld_fixed_v36(
    *,
    route: CompiledOwnerExternalPldAllocationRouteV36,
    raw_train_x: np.ndarray,
    train_y: np.ndarray,
    raw_test_x: np.ndarray,
    test_y: np.ndarray,
    research_seed: int,
) -> OwnerExternalPldExecutionResultV36:
    """Execute H only by delegating to its sealed frozen A route."""

    if not isinstance(
        route,
        CompiledOwnerExternalPldAllocationRouteV36,
    ):
        raise OwnerExternalPldExecutionV36Error(
            "H executor requires a compiled external-PLD route"
        )
    try:
        route.assert_private_execution_integrity()
    except ExternalPldContractV36Error as exc:
        raise OwnerExternalPldExecutionV36Error(
            f"Compiled H route integrity failed: {exc}"
        ) from exc
    if not route.execution_ready:
        raise OwnerExternalPldExecutionV36Error(
            "Compiled H route is not execution-ready"
        )
    try:
        source_before = dict(execution_source_bundle_external_pld_v36())
        source_sha256 = execution_source_bundle_sha256_external_pld_v36(source_before)
    except SourceBundleExternalPldV36Error as exc:
        raise OwnerExternalPldExecutionV36Error(
            f"Could not bind the H execution source: {exc}"
        ) from exc
    if source_sha256 != route.execution_source_bundle_sha256:
        raise OwnerExternalPldExecutionV36Error(
            "H execution source differs from the compiled route"
        )
    try:
        base_result = train_owner_random_allocation_fixed_v4(
            route=route.base_route,
            raw_train_x=raw_train_x,
            train_y=train_y,
            raw_test_x=raw_test_x,
            test_y=test_y,
            research_seed=research_seed,
        )
    except OwnerRandomAllocationExecutionV4Error as exc:
        raise OwnerExternalPldExecutionV36Error(
            f"Nested A executor rejected H: {exc}"
        ) from exc
    try:
        route.assert_private_execution_integrity()
        source_after = dict(execution_source_bundle_external_pld_v36())
    except (
        ExternalPldContractV36Error,
        SourceBundleExternalPldV36Error,
    ) as exc:
        raise OwnerExternalPldExecutionV36Error(
            f"H integrity failed after nested execution: {exc}"
        ) from exc
    if source_after != source_before:
        raise OwnerExternalPldExecutionV36Error(
            "H execution source changed during execution"
        )

    base_public = base_result.public_payload()
    base_private = base_result.private_manifest()
    public_record: dict[str, Any] = {
        "schema_version": PUBLIC_EXECUTION_SCHEMA_EXTERNAL_PLD_V36,
        "visibility": "public",
        "release_status": "research_non_release",
        "privacy_claim": copy.deepcopy(base_public["privacy_claim"]),
        "route": {
            "route_id": route.contract.route_id,
            "implementation_id": route.contract.implementation_id,
            "public_contract_sha256": (route.contract.public_contract_sha256),
            "public_plan_sha256": route.public_plan_sha256,
        },
        "external_accounting_seal": {
            "request_sha256": route.external_request.sha256,
            "response_sha256": (route.external_response.response_sha256),
            "accepted_bound_type": (route.registry_entry.accepted_bound_type),
            "accepted_epsilon": (route.external_response.epsilon_upper),
            "crosscheck_bound_type": (route.registry_entry.crosscheck_bound_type),
            "crosscheck_epsilon": (route.external_response.epsilon_lower),
            "delta": route.contract.base_contract.delta,
        },
        "nested_allocation_execution": base_public,
        "execution_checks": {
            "compiled_H_object_only": True,
            "H_integrity_before_and_after": True,
            "nested_A_integrity_before_and_after": True,
            "external_response_reused_without_reaccounting": True,
            "executor_delegation_only": True,
            "H_source_bundle_unchanged_during_execution": True,
        },
        "model_output": copy.deepcopy(base_public["model_output"]),
        "execution_source_bundle_sha256": source_sha256,
    }
    public_record["public_execution_sha256"] = _sha256_payload(public_record)
    assert_public_certificate_redacted(public_record)

    private_record: dict[str, Any] = {
        "schema_version": PRIVATE_EXECUTION_SCHEMA_EXTERNAL_PLD_V36,
        "visibility": "private",
        "handling": ("do_not_publish_contains_nested_private_execution_record"),
        "public_execution_sha256": (public_record["public_execution_sha256"]),
        "H_private_execution_binding_sha256": (route.private_execution_binding_sha256),
        "nested_A_private_manifest": base_private,
    }
    private_record["private_manifest_sha256"] = _sha256_payload(private_record)
    return OwnerExternalPldExecutionResultV36(
        route=route,
        base_result=base_result,
        public_record=public_record,
        private_record=private_record,
    )
