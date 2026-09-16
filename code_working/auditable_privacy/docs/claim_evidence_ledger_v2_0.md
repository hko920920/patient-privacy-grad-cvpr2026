# Claim-to-Evidence Ledger v2.0

Status: working submission ledger  
Authority: `privacy_claim_contract_v2_0.md` and
`privacy_claim_validator.py`  
Rule: no manuscript claim may be stronger than the status in this ledger.

## Verdict vocabulary

- `SUPPORTED`: current evidence directly supports the stated claim.
- `CONDITIONAL`: support holds only under the adjacency and execution boundary
  written in the claim.
- `DIAGNOSTIC_ONLY`: the artifact supports a mapping or systems observation,
  not an epsilon/delta privacy statement.
- `BLOCKED`: the proposed wording is false, vacuous, stale, or lacks a required
  premise.
- `PENDING`: current-code regeneration or independent verification is not yet
  complete.

## Central claims

| ID | Candidate manuscript claim | Status | Current evidence | Required wording/action |
| --- | --- | --- | --- | --- |
| C01 | A post-hoc audit must bind generated, accounting, and claimed units together with mapping, schedule, mechanism, adjacency, and accountant evidence. | SUPPORTED | `docs/privacy_claim_contract_v2_0.md`; T01--T37 truth table; authoritative validator | Present as the paper's contract and reporting-layer contribution, not as a new accountant or DP theorem. |
| C02 | Missing, malformed, stale, or inconsistent evidence fails closed without positive DP wording. | SUPPORTED | Expanded regression suite: 83/83; actual-file mutation, source/registry forgery, exact RNG/noise/normalization/rate semantics, runtime artifacts, owner cap, raw domain, schema, accountant, redaction, CLI, retrospective and UCI-verifier tamper tests | Cite 83/83 only for the current hashed source state; rerun and rebuild the frozen ablation after every authority-source change. |
| C03 | The hardened WISDM v2 reference run is end-to-end auditable at the generated-window unit. | CONDITIONAL | `reports/wisdm_v2_hardened_secure_002`; 38/38 dependency-independent checks; base epsilon `0.5368273911667286`, delta `1e-6` | Say “conditional on the complete and faithful bound public-benchmark evidence.” Name the known-attack-hardened CSPRNG boundary; do not claim universal side-channel protection or state-of-the-art utility. |
| C04 | For the hardened WISDM overlap run, a fixed owner-slot payload replacement has generated add/remove distance `K=4`; the converted event statement is finite. | CONDITIONAL | Bound raw domain `fixed_owner_slot_payload_domain_v1`; observed event kappa `2`; event certificate `CONVERT/ALLOWED`; converted epsilon `2.1473095646669145`, delta `1.0641849375406323e-05` | State the exact fixed-owner-slot payload adjacency and generated add/remove accountant metric. Never print the base window number as the event number. |
| C05 | The same WISDM run supports an owner-level DP statement by black-box grouping. | BLOCKED | Public enforced owner cap/stability bound `K=600`; observed incidence `565`; owner certificate `GROUP/BLOCKED_VACUOUS`, diagnostic log10(delta_K) `134.033096586` | Use this as a negative result: observed incidence is only a cap-compliance check. Narrow to window/event wording or require a tighter public cap/native owner mechanism. |
| C06 | On 5,000 Sepsis patients, the legacy window-count and observed-incidence values are reproducible from the bound cache. | DIAGNOSTIC_ONLY | `reports/sepsis2019_v2_mapping_evidence_5k_001`; input manifest SHA-256 `2f66990312f32142ce7c2a58afa006cf5e730a7651f817bdd573501bcbde4b56`; evidence SHA-256 `24da11042dde9e1236c7b4c9a3315219d76f4c7595c8e844b47f0ee53f972dd2`; independent verification 128/128 | Report these as exact mapping diagnostics for the bound public cache, not as DP guarantees. |
| C07 | Sepsis 6/12/24-hour dense windows have observed event kappa `6/12/24`; 6-hour non-overlap has observed kappa `1`. | DIAGNOSTIC_ONLY | Current v2 mappings: 167,206 / 137,430 / 83,608 dense windows; 29,958 6-hour non-overlap windows | Keep “observed kappa” distinct from transformation stability and from epsilon/delta. |
| C08 | Sepsis non-overlap with observed event kappa `1` automatically restores same-number event-level DP under an add/remove accountant. | BLOCKED | Fixed-slot raw replace-one changes one generated window by replacement, which has generated add/remove distance `2` | Remove every unconditional “non-overlap restores DIRECT event wording” sentence. It is direct only with an adjacency-compatible base guarantee; otherwise use the validated conversion. |
| C09 | Sepsis label-balanced owner caps can use observed event incidence as a certified fixed-mapping stability bound. | BLOCKED | v2 generator classifies selection as `private_data_dependent`; verifier reconstructs the selection-identity change | Keep only as a negative/control example unless a registered stability or privacy proof is added. |
| C10 | Backblaze Q1 2025 legacy scale and observed multiplicities reproduce under their actual ordered-available-snapshot semantics. | DIAGNOSTIC_ONLY | `reports/backblaze_v2_mapping_evidence_q1_2025_001`; archive SHA-256 `7a230565f72fdb3882c05fdcfab361c801e9a545ba18543899fc7618174171ec`; 27,799,986 rows, 318,426 drives; independent full rescan 86/86 | Report only as exact mapping diagnostics for the bound public archive. The bundle has no mechanism/accountant and cannot authorize DP wording. |
| C11 | “N-day Backblaze windows” and “N available-snapshot windows” are interchangeable. | BLOCKED | v2 evidence separates `calendar_contiguous` from `ordered_snapshot`: 7-wide dense counts are 25,867,949 versus 25,904,370, a 36,421-window difference | Rename legacy rows or use the calendar-contiguous reconstruction. The maximum event/owner kappas remain 7/84 in this case, but the constructors and scale are not identical. |
| C12 | Mapping-only Sepsis or Backblaze evidence can itself authorize an epsilon/delta claim. | BLOCKED | Both v2 bundles declare `diagnostic_only` and contain no epsilon/delta or supported statement | A positive privacy sentence additionally requires an actual complete mapping, executed mechanism/runtime binding, and registered accountant report. |
| C13 | The full contract catches failures that field-presence, schema-only, multiplicity-only, hash-only, and accountant-only checks miss. | SUPPORTED | Frozen 31-case conformance/repair matrix: six valid and 25 non-allow cases linked to T01--T37 or R01--R08; FULL 0/25 unsafe positives and 31/31 full tuples; independent verification 394/394 | Report this as authored invariant/regression coverage, not error prevalence or a competitor ranking. Probe unsafe positives are 25/25, 24/25, 25/25, 24/25, and 22/25 respectively. |
| C14 | The hardened validator repairs root causes that occurred in preserved claim-bearing development artifacts, not only newly authored mutations. | SUPPORTED | Protocol-fixed retrospective lineage: 10/10 pre-protocol incidents, 16 repaired-control sources, production-import-free verification 176/176, tamper tests 3/3 | Call these authenticated system-development-lineage incidents. Do not call them an external paper sample, independent prevalence study, completeness result, or estimate of real-world error frequency. |
| C15 | The UCI HAR route is end-to-end auditable for one published generated-window add/remove query. | CONDITIONAL | Five protocol-fixed runs; 49/49 full raw-source checks and 38/38 portable checks per run; five distinct model digests; window `ALLOWED/DIRECT/K=1`; epsilon `0.5368273911667286`, delta `1e-6` | State that support is conditional on complete and faithful bound evidence for the published-window route. Treat utility only as an integration diagnostic and do not infer cross-mechanism generality. |
| C16 | The same UCI HAR execution authorizes raw-event or owner privacy wording from observed incidence. | BLOCKED | Every confirmatory run returns event and owner `BLOCKED_UNVERIFIED/UNDERSPECIFIED` without a positive sentence; exact raw-session support and a registered public owner contribution contract are absent | Report the negative boundary explicitly. Do not promote observed event kappa 1 or owner incidence 409 into a stability proof. |

## Stale manuscript statements requiring rewrite

The following pre-repair statement families are prohibited until rewritten:

1. “event kappa equals one, therefore the original window epsilon is directly
   event-level” without adjacency compatibility;
2. “owner kappa equals one, therefore owner DP” without binding the raw owner
   add/remove relation to an add/remove-accounted generated mechanism;
3. `DIRECT`, `CONVERT`, or `GROUP` labels produced by legacy
   `unit_audit.py` or legacy certificate files;
4. label-balanced fixed-mapping claims that omit the private selection
   dependency;
5. Backblaze “day” wording when the constructor compresses missing calendar
   dates;
6. any Sepsis/Backblaze epsilon or delta inferred from mapping multiplicity
   alone.

## Current immutable evidence hashes

### Hardened WISDM gold

- Mapping:
  `0bd250b5c2e144131b23bc76f1fbaf7dd955c91f86e75990925ff78e752eef1b`
- Privacy report:
  `2b1c6f3e58052791b68bd1ee1329bd54196192fcb500b8239c358b39fe9346fa`
- Released model:
  `fea84c3636433aef547afad6618c89e3f8df8d3ce77b5d7fdee8d73eb3296a08`
- Independent verification:
  `9cd3b9463a7d7896836ded587526868bf5de40037d58fd147ee11ef9a27266d0`

### Sepsis v2 mapping evidence

- Generator:
  `b0a96f5031c8f4303cafe4a4f6eb2cd328707e59eaadf73202266c8690250d25`
- Independent verifier:
  `7398889eba69c403e0e299596d40e20fb2d8c016ce173eb4aa3e1ae0a15505ba`
- Policy summary:
  `dec88238506050a65c6c7fe26f957f478abc0e227016282998835f33acc43c17`
- Independent verification record:
  `3caadb167ed29bab77d90ed515340de14691d45b88db02451fe3dc578763dbb8`

### Backblaze v2 mapping evidence

- Generator:
  `30d2ab948a532246cc009fc564cf3881c3b6f44e4f8e94d608628d914e3090b9`
- Independent verifier:
  `a6337edeea647d83f95c8be2c19ad2bf28dd683451b553191773de4ec596042b`
- Owner-profile histogram:
  `3d4e64121b9b99b88ec8af1669ff5aee732e2ee35924ec2983338cca12325627`
- Policy summary:
  `98c125baf36c2e341533c857c541d59eb3e2de4e44d3992b67180dc41fe8a3ae`
- Evidence record:
  `8d3ecbcec18ee5653c3b4cb23e53eb6efb4d7fd3d89dfda787522cb0bdbe1d30`
- Independent verification record:
  `663687d721e604d09ecf3d56a6f2e024aa2a0c45917d61eea00689e34bdd6807`

### Legacy-to-v2 comparison

- Comparison CSV:
  `837002f650bc1f0c7a2f670460ce5b78e66562b85bf7ac7f408fa6e32b92eb3a`
- Summary:
  `9afb24a513c9be0882f6b6fe33cec1e5463fad39ec3793d8f35dd3b59f7b0582`

### Validator baseline/ablation

- Frozen case matrix:
  `02fde1c50e0a69e3ef023621f5b63de130bbba24380c686bb5d0ccf676d3f57f`
- Per-case predictions:
  `c2a4d0a3ad391cffe7db45cff7a131e337dfe563478b06a6619e3b9e6064d676`
- Aggregate metrics:
  `12202e0c36fcd05df681205c0f44e86391c114070bf447160c32d727e98c829c`
- Summary:
  `63fa311a63ea88d1c13ec38531b7ed90593dedb5042a1f44891a1cacb9c8fb4c`
- Independent verification:
  `15f2860aae28739a28bdcd50006579ad103e88ddc30fae3db75690eb3295e9de`

### Retrospective development-failure lineage

- Lineage:
  `a7e69a55520088f1208463ab8ee5e4f41e0944a556e16fe09199bef295fc4b94`
- Incident CSV:
  `c92e14383d78a0fa4d20023635b0392d4b13986176ad84a5219dc57bdd686dc0`
- Independent verification:
  `6617060afd53b06caecf8088dc022ca71b643c8849ff3c6f3767f84cd1ec9de2`

### UCI HAR v5 confirmatory route

- Execution manifest:
  `9221d88de30d062a3816ecd9cef2e0850f0e83c6d08041bb35ba0f06c6f34466`
- Confirmatory summary:
  `e222fa63b672dc2a9810755d492ce0314668aaa758179c92e84dd5e6e95d97d8`
- Confirmatory metrics:
  `9523e1772a0a523b067f6c4b3903ea08c128014bebf46187702468ff528bc2c8`

### Expanded P0 gate

- Machine-readable gate:
  `da42efe1e939778c747aeabbec3828bfebacfe51a0d6df356ec0abf9a34687db`
- Gate result:
  `16/16 PASS`; complete regression `83/83`; fresh FULL `31/31` with
  `0/25` unsafe positives.
