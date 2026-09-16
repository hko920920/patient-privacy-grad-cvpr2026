"""Independent declarative oracle for the registered owner-Poisson route.

This package intentionally has no dependency on the production ``unitdp``
package.  It interprets the versioned JSON specification under ``specs/`` so
that differential tests do not reuse compiler predicates.
"""

from .oracle_v1 import (
    OracleResult,
    canonical_payload_sha256,
    load_contract_file,
    load_oracle_spec,
    registered_public_projection,
    validate_accountant_environment,
    validate_mapping_array_binding,
    validate_mapping_file,
    validate_preprocessor_artifact,
    validate_registered_accounting,
    validate_registered_research_contract,
    validate_registered_source_bundle,
)

__all__ = [
    "OracleResult",
    "canonical_payload_sha256",
    "load_contract_file",
    "load_oracle_spec",
    "registered_public_projection",
    "validate_accountant_environment",
    "validate_mapping_array_binding",
    "validate_mapping_file",
    "validate_preprocessor_artifact",
    "validate_registered_accounting",
    "validate_registered_research_contract",
    "validate_registered_source_bundle",
]
