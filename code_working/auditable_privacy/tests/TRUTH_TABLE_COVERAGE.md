# v2.0 Truth-Table Test Coverage

Status: P0 regression map  
Suite: `python -m unittest discover -s tests -v`
Current result: 68/68 passing on 2026-07-24

The suite uses real temporary mapping CSVs, the published JSON Schema, the
registered Opacus 1.6.0 RDP checker, and the public certificate CLI. Extension
points with no registered checker are tested for default-deny behavior.

| Truth row | Regression evidence |
| --- | --- |
| T01 | `test_valid_window_direct` |
| T02 | `test_missing_runtime_blocks_post_execution_wording` |
| T03 | `test_sampler_accountant_mismatch_blocks_when_rebound` |
| T04 | `test_nonfinite_epsilon_blocks`, `test_invalid_delta_blocks`, `test_zero_noise_blocks` |
| T05 | `test_observed_kappa_one_without_validated_stability_blocks` |
| T06 | `test_native_event_direct_exact_metric` |
| T07 | `test_valid_event_conversion_recomputed` |
| T08 | `test_valid_event_conversion_recomputed`, `test_owner_group_conversion_nonvacuous` |
| T09 | `test_vacuous_group_conversion_blocks` |
| T10 | `test_native_owner_direct` |
| T11 | `test_owner_to_window_k_one_direct` |
| T12 | `test_owner_group_conversion_nonvacuous` |
| T13 | `test_owner_group_conversion_vacuous` |
| T14 | Built-in partition checker default-deny: `test_owner_partition_rejects_multiowner_records` |
| T15 | `test_owner_partition_rejects_multiowner_records`; completeness remains an explicit exporter trust boundary |
| T16 | `test_private_global_preprocessing_blocks` |
| T17 | No composition checker registered; default-deny covered with T18 |
| T18 | `test_separately_dp_without_registered_composition_blocks` |
| T19 | All valid golden tests use a content-bound `fixed_mapping` |
| T20 | `test_private_adaptive_schedule_blocks` |
| T21 | No union checker registered; default-deny covered with T22 |
| T22 | `test_unregistered_schedule_blocks` |
| T23 | `test_user_stochastic_bound_label_blocks` |
| T24 | No stochastic theorem checker registered; default-deny covered with T23 |
| T25 | `test_external_conversion_attachment_cannot_authorize` |
| T26 | No external conversion checker registered; default-deny covered with T25 |
| T27 | `test_native_event_direct_exact_metric`, `test_native_owner_direct` |
| T28 | `test_mapping_bytes_mutation_blocks`, `test_missing_actual_mapping_blocks` |
| T29 | `test_preprocessing_parameter_digest_tamper_blocks`, `test_incomplete_influence_support_blocks` |
| T30 | `test_runtime_noise_mismatch_blocks`, `test_private_population_policy_blocks`, `test_accountant_covered_population_size_is_not_accepted`, `test_data_dependent_update_normalization_blocks` |
| T31 | `test_uncovered_private_model_selection_blocks` |
| T32 | `test_research_prng_is_not_release_allowed`, `test_secure_assurance_with_secure_mode_false_blocks`, `test_universal_release_grade_rng_wording_is_rejected`, `test_secure_systemrandom_registered_mechanism_direct` |
| T33 | `test_public_deterministic_noise_blocks` |
| T34 | `test_private_metadata_is_redacted_not_leaked` |
| T35 | `test_unknown_accountant_checker_blocks` |
| T36 | `test_accountant_version_drift_blocks`, `test_accountant_epsilon_tamper_blocks` |
| T37 | `test_generated_unit_manifest_mismatch_blocks`, `test_mapping_without_generated_unit_column_blocks` |

Additional structural regressions cover audit-kappa tampering, schema type
attacks, source/registry forgery, exact RNG and Gaussian semantics,
runtime-artifact forgery, rational-rate mismatch, public owner-cap enforcement,
generic-event-adjacency rejection, mapping row-count/hash binding, group
conversion numerics, registered-mechanism version drift, and valid/blocked
end-to-end CLI outputs. Five secondary-evidence math tests additionally enforce
the ordered-snapshot/calendar-day distinction, the replace-one to add/remove
factor of two, half-open intervals, and the instability introduced by
label-balanced private selection.
