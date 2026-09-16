# Unit-Aware Certificate Schema v2.0

Status: normative input/output schema for `privacy_claim_contract_v2_0.md`  
Date: 2026-07-24

Version 2.0 is intentionally incompatible with legacy privacy-report JSON.
Legacy reports may be inspected, but they cannot authorize positive wording.

## Builder Inputs

The builder consumes:

1. one `unit_audit.csv` row for the requested scenario and normalized claim
   unit;
2. the actual mapping CSV named by that row;
3. one v2.0 privacy-report JSON object;
4. the requested scenario and claim unit.

The audit-row `verdict` is ignored. Observed event/owner kappa values are
diagnostics and inputs to registered stability checkers, not authorization by
themselves.

The builder reopens the mapping CSV and independently recomputes its byte
digest, canonical selected-record digest, selected row count, event incidence,
owner incidence, and attribution arity. Matching digest strings copied into two
files are insufficient.

Certifiable mapping CSVs require:

```text
scenario,window_id,generated_unit,start,end
```

plus at least one of `owner_id`, `owner_ids`, or `owners`.
`generated_unit` is exactly `window`, `event`, or `owner` and must match the
accounting unit. Historical CSVs without this column remain diagnostic-only.

## Top-Level Privacy Report

Required keys:

```json
{
  "schema_version": "2.0",
  "scenario": "example_scenario",
  "mechanism": {},
  "privacy": {},
  "pipeline": {},
  "runtime_trace": {},
  "rng_assurance": "known_attack_hardened_secure",
  "noise_seed_policy": "secure_private_unrecorded",
  "public_certificate": {},
  "claim_contracts": {}
}
```

## Mechanism

Required fields:

```json
{
  "mechanism": {
    "mechanism_id": "secure_systemrandom_dpsgd",
    "mechanism_version": "2.0.0",
    "accountant_id": "opacus_rdp",
    "accountant_version": "1.6.0",
    "accounting_unit": "window",
    "accountant_adjacency": "add_remove",
    "sampling_unit": "window",
    "sampler_law": "poisson",
    "accountant_sampler_law": "poisson",
    "sample_rate": 0.02,
    "sample_rate_numerator": 1,
    "sample_rate_denominator": 50,
    "sampling_implementation": "systemrandom_randrange_bernoulli_v1",
    "steps": 100,
    "noise_multiplier": 2.0,
    "secure_mode": true,
    "secure_rng_backend": "python.secrets.SystemRandom",
    "noise_generation": "normalvariate_discard1_sum4_div2_v1",
    "noise_hardening": "known_fp_reconstruction_mitigation_four_draw_v1",
    "clipping_unit": "window",
    "clipping_norm": 1.0,
    "noising_unit": "window",
    "gradient_aggregation": "clipped_sum_plus_gaussian_noise",
    "update_normalization": "public_constant_step_no_dataset_denominator",
    "optimizer_step_size": 0.006666666666666667,
    "adapter_registry_entry_sha256": "<64 hex>",
    "privacy_convention": "rdp_converted"
  }
}
```

Exact sampler identifiers:

```text
poisson
shuffle
fixed_without_replacement
full_batch
custom_registered
```

The built-in path requires sampling, clipping, noising, and accounting units to
match. A different construction requires a registered mechanism checker.

The current positive registry contains one exact mechanism path:

```text
mechanism_id       = secure_systemrandom_dpsgd
mechanism_version  = 2.0.0
accountant_id      = opacus_rdp
accountant_version = 1.6.0
sampler law        = poisson
sampling impl.     = systemrandom_randrange_bernoulli_v1
noise generation  = normalvariate_discard1_sum4_div2_v1
aggregation        = clipped_sum_plus_gaussian_noise
normalization      = public_constant_step_no_dataset_denominator
privacy convention = rdp_converted
```

The validator verifies the exact rational rate, source/checker hashes, registry
entry, and runtime semantics before recomputing epsilon. The lightweight
checker implements the registered Opacus RDP formula; the independent WISDM
verifier additionally cross-checks with `dp-accounting==0.6.0`. Unknown
identifiers, version drift, source drift, and epsilon mismatch fail closed.
For owner records, the same add/remove-one-record semantics are named
`owner_add_remove`.

The `known_attack_hardened_secure` assurance class requires `secure_mode=true`,
a matched runtime trace, a private seed policy, and the registered discarded-
first four-draw construction. It states mitigation of the named known
floating-point reconstruction class only; it is not a universal release,
side-channel, or faithful-execution guarantee.

## Privacy Parameters

```json
{
  "privacy": {
    "epsilon": 1.4546741482184284,
    "delta": 1e-5,
    "delta_convention": "accountant_delta_only"
  }
}
```

Supported privacy conventions:

```text
approx_dp
rdp_converted
pld_converted
```

Supported delta conventions:

```text
accountant_delta_only
delta_augmented_with_bound_failure
conditional_on_bound_event
```

The initial built-in `fixed_mapping` path normally uses
`accountant_delta_only`. Other conventions require the matching registered
schedule analysis.

## Pipeline and Canonical Digest

```json
{
  "pipeline": {
    "mapping_file_sha256": "<64 hex>",
    "selected_mapping_sha256": "<64 hex>",
    "pipeline_sha256": "<sha256 of canonical manifest JSON>",
    "manifest": {
      "dataset_id": "dataset-version",
      "split_id": "public-split-v1",
      "code_version": "wisdm_v2_hardened_pipeline_v2",
      "code_artifact_id": "scripts/wisdm_v2_gold_pipeline.py",
      "code_sha256": "<actual source SHA-256>",
      "accountant_checker_artifact_id": "scripts/rdp_accountant_lite.py",
      "accountant_checker_sha256": "<actual checker SHA-256>",
      "support_convention": "half_open_integer_intervals",
      "record_identity_policy": "public_fixed_ids",
      "generated_record_unit": "window",
      "cross_owner_dependency": "none",
      "mapping_file_sha256": "<same mapping-file digest>",
      "selected_mapping_sha256": "<same selected-record digest>",
      "selected_record_count": 1000,
      "mechanism_sha256": "<sha256 of canonical mechanism JSON>",
      "batch_trace_sha256": "<64 hex>",
      "released_model_sha256": "<64 hex>",
      "parser_evidence_sha256": "<64 hex>",
      "raw_domain": {
        "id": "fixed_owner_slot_payload_domain_v1",
        "sha256": "<64 hex>",
        "owner_presence": "fixed",
        "event_presence": "fixed",
        "mutable_components": ["payload"]
      },
      "influence_support": {
        "status": "complete",
        "components": [
          "features",
          "labels",
          "weights",
          "selection",
          "preprocessing"
        ]
      },
      "owner_attribution_status": "complete",
      "preprocessing": [
        {
          "id": "public_fixed_scale_v1",
          "classification": "public_fixed",
          "parameters": {
            "mean": [0.0],
            "scale": [1.0]
          },
          "parameters_sha256": "<64 hex>"
        }
      ],
      "contribution_policy": {
        "id": "public_owner_generated_record_cap_v1",
        "classification": "public_fixed",
        "parameters": {
          "cap": 600,
          "enforcement": "deterministic_owner_window_prefix_v1",
          "ordering": "ascending_generated_window_ordinal"
        },
        "parameters_sha256": "<64 hex>"
      },
      "schedule": {
        "id": "selected-mapping-once",
        "mode": "fixed_mapping",
        "evidence_status": "validated",
        "parameters": {
          "selected_record_count": 1000
        },
        "parameters_sha256": "<64 hex>"
      },
      "population_definition": "public training split",
      "population_size_policy": "not_used_in_release_mechanism",
      "model_selection_status": "public_validation"
    }
  }
}
```

Canonical digest computation:

```python
json.dumps(
    manifest,
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=False,
).encode("utf-8")
```

followed by SHA-256.

Required preprocessing classifications:

```text
public_fixed
record_local
owner_local
separately_dp
private_global
private_adaptive
```

The initial built-in path accepts `public_fixed`, `record_local`, and
`owner_local`. `separately_dp` requires a registered composition checker.
Unaccounted private global/adaptive preprocessing is blocking.

The pipeline digest binds the mapping digests, selected count, mechanism
digest, raw domain, source and checker identities, execution-artifact digests,
complete influence components, preprocessing parameters, public contribution
policy, and schedule. Hash agreement remains an integrity/drift check rather
than a proof that an exporter was honest or complete.

The built-in positive path accepts only
`population_size_policy=public_fixed` or
`population_size_policy=not_used_in_release_mechanism`. The schema retains
`accountant_covered` as a negative regression token, but semantic validation
always blocks it. In the hardened WISDM mechanism, neither the update nor the
sampling decision uses a private dataset-size denominator.

## Runtime Trace

Matched trace:

```json
{
  "runtime_trace": {
    "status": "matched",
    "trace_hash": "<64 hex>",
    "mechanism_id": "secure_systemrandom_dpsgd",
    "mechanism_version": "2.0.0",
    "accountant_id": "opacus_rdp",
    "accountant_version": "1.6.0",
    "accounting_unit": "window",
    "accountant_adjacency": "add_remove",
    "sampling_unit": "window",
    "sampler_law": "poisson",
    "accountant_sampler_law": "poisson",
    "sample_rate": 0.02,
    "sample_rate_numerator": 1,
    "sample_rate_denominator": 50,
    "sampling_implementation": "systemrandom_randrange_bernoulli_v1",
    "steps": 100,
    "noise_multiplier": 2.0,
    "secure_mode": true,
    "secure_rng_backend": "python.secrets.SystemRandom",
    "noise_generation": "normalvariate_discard1_sum4_div2_v1",
    "noise_hardening": "known_fp_reconstruction_mitigation_four_draw_v1",
    "clipping_unit": "window",
    "clipping_norm": 1.0,
    "noising_unit": "window",
    "gradient_aggregation": "clipped_sum_plus_gaussian_noise",
    "update_normalization": "public_constant_step_no_dataset_denominator",
    "optimizer_step_size": 0.006666666666666667,
    "privacy_convention": "rdp_converted",
    "adapter_registry_entry_sha256": "<64 hex>",
    "code_artifact_id": "scripts/wisdm_v2_gold_pipeline.py",
    "code_sha256": "<64 hex>",
    "accountant_checker_artifact_id": "scripts/rdp_accountant_lite.py",
    "accountant_checker_sha256": "<64 hex>",
    "selected_mapping_sha256": "<64 hex>",
    "pipeline_sha256": "<64 hex>",
    "epsilon": 0.5368273911667286,
    "delta": 1e-6,
    "delta_convention": "accountant_delta_only",
    "runtime_evidence_sha256": "<64 hex>",
    "evidence": {
      "executed_updates": 100,
      "released_model_sha256": "<64 hex>",
      "batch_trace_sha256": "<64 hex>",
      "secure_sampling_rng": "systemrandom_randrange_bernoulli_v1",
      "secure_noise_rng": "normalvariate_discard1_sum4_div2_v1"
    }
  }
}
```

For `status=matched`, `trace_hash` is SHA-256 over the canonical JSON object
containing all runtime fields above except `status` and `trace_hash`. Every
field is compared to the mechanism, privacy, mapping, and pipeline contract.

Allowed statuses:

```text
matched
missing
mismatched
not_applicable
```

`mismatched` is `BLOCKED_INVALID`. `missing` and `not_applicable` are
`BLOCKED_UNVERIFIED` for post-execution wording. Only `matched`, with every
declared field and execution-artifact digest equal, can support an
`EVIDENCE_VALIDATED` concrete-run statement.

## Randomness

```json
{
  "rng_assurance": "known_attack_hardened_secure",
  "noise_seed_policy": "secure_private_unrecorded"
}
```

RNG assurance:

```text
known_attack_hardened_secure
research_prng
unknown
```

The legacy input token `release_grade_secure` remains syntactically recognized
only so the semantic validator can emit `RNG_ASSURANCE_OVERCLAIM`; it can never
authorize a positive certificate.

Noise-seed policy:

```text
secure_private_unrecorded
secure_private_internal
public_deterministic
unknown
```

`research_prng` yields `RESEARCH_ONLY`. Public deterministic DP-noise seeds and
unknown policies cannot authorize a released-model claim.

## Public Metadata Classification

```json
{
  "public_certificate": {
    "metadata_classification": "public_benchmark"
  }
}
```

Allowed values:

```text
public_benchmark
controller_approved
dp_released
private_internal
```

`private_internal` produces a redacted public certificate without scenario,
mapping digests, pipeline digest, or observed multiplicities.

## Claim Contracts

### Generated-unit identity

```json
{
  "claim_contracts": {
    "window": {
      "claimed_unit": "window",
      "raw_adjacency": "add_remove",
      "stability_status": "validated",
      "stability_method": "identity_generated_unit",
      "stability_bound": 1,
      "support_status": "complete",
      "schedule_mode": "fixed_mapping",
      "schedule_evidence_status": "validated",
      "conversion_method": "identity"
    }
  }
}
```

### Fixed event influence under replacement

```json
{
  "claim_contracts": {
    "event": {
      "claimed_unit": "event",
      "raw_adjacency": "fixed_owner_slot_payload_replace_one_v1",
      "raw_domain_id": "fixed_owner_slot_payload_domain_v1",
      "raw_domain_sha256": "<64 hex>",
      "stability_status": "validated",
      "stability_method": "fixed_influence_replacements",
      "stability_bound": 4,
      "support_status": "complete",
      "schedule_mode": "fixed_mapping",
      "schedule_evidence_status": "validated",
      "conversion_method": "builtin_group"
    }
  }
}
```

This domain fixes owner identity, owner-local slot identity, and event
presence; only payload components may change. Generic `replace_one` does not
state those premises and is blocked for this built-in route. If observed event
kappa is two and the accountant uses add/remove adjacency, the registered
checker computes `K=4`, not two.

### Fixed owner partition under owner removal

```json
{
  "claim_contracts": {
    "owner": {
      "claimed_unit": "owner",
      "raw_adjacency": "owner_add_remove",
      "stability_status": "validated",
      "stability_method": "owner_partition_add_remove",
      "stability_bound": 600,
      "support_status": "complete",
      "schedule_mode": "fixed_mapping",
      "schedule_evidence_status": "validated",
      "conversion_method": "builtin_group"
    }
  }
}
```

The pipeline manifest must use `record_identity_policy=owner_partition_ids`,
`cross_owner_dependency=none`, and the registered public contribution policy
`public_owner_generated_record_cap_v1` with deterministic enforcement. The
contract's `stability_bound` must equal that public cap. An observed maximum
may only confirm that the cap was respected.

## Registered Built-In Stability Methods

| Method | Claim | Requirements | Computed K |
| --- | --- | --- | ---: |
| `identity_generated_unit` | Any exact accounting unit | Same unit and exact adjacency | 1 |
| `fixed_influence_replacements` | Event | `fixed_owner_slot_payload_replace_one_v1`; bound raw domain; public-fixed generated IDs; complete influence support | observed event kappa for replace-one accountant; twice that for add/remove accountant |
| `owner_partition_add_remove` | Owner | Owner add/remove; generated add/remove accountant; fixed owner partition; no cross-owner dependency; registered public enforced cap | public contribution cap (observed maximum is diagnostic only) |

All other method names are unregistered and fail closed.

## Certificate Outputs

For basename `certificate_<scenario>_<claim>` the builder writes:

```text
certificate_<scenario>_<claim>.csv
certificate_<scenario>_<claim>.json
certificate_<scenario>_<claim>.md
certificate_<scenario>_<claim>_internal.csv
certificate_<scenario>_<claim>_internal.json
certificate_<scenario>_<claim>_internal.md
```

The unsuffixed files are the redacted public certificate. Internal files may
contain exact provenance and validation messages.

Core output fields:

```text
unit_path
release_status
assurance_status
stability_bound
base_epsilon
base_delta
reported_epsilon
reported_delta
issue_codes
supported_statement
assurance_boundary
do_not_say
```

A blocked or research-only certificate has an empty `supported_statement`.
Every allowed statement is explicitly conditional on a complete and faithful
mapping export. The validator detects registered semantic violations and
cross-artifact drift; it is not a malicious-execution attestation system.

## CLI Exit Contract

The builder writes diagnostic files and exits:

- `0` only when `release_status=ALLOWED`;
- `2` when the certificate is blocked or research-only.

`--diagnostic-exit-zero` may be used by negative-test artifact generation. It
does not change the certificate verdict or create positive wording.

Minimal invocation:

```powershell
python scripts\build_unit_certificate.py `
  --audit-csv <unit_audit.csv> `
  --mapping-csv <actual_mapping.csv> `
  --privacy-report-json <privacy_report_v2.json> `
  --scenario <scenario> `
  --claim-unit <window|event|owner> `
  --output-dir <certificate_dir>
```

The published JSON Schema is applied before semantic validation. The
`jsonschema` package is therefore a required core dependency; if unavailable,
the validator fails closed.
