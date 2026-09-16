# U patient membership pilot

This is the 2026-09-14 public-development pilot requested in the U execution plan.
It uses existing NIH PA sources and a **new** locked cohort, without changing thesis K5/K10 runs.

Run from `code_working` with `base_gate/.venv/Scripts/python.exe -X utf8`.
The concrete run root is `_reports/cvpr_u_pilot_v1_001`.

- `prepare_cohort`: verify available public source hashes/pixels, follow-up distinctness and duplicate screens; select 400 evaluation patients and separate reference/quality/background roles.
- `verify_cohort`: independently check identities, role isolation, model inclusion, file hashes and U overlap.
- `build_cache`: verified local SD2.1 VAE/text and DINOv2 features, including target-independent reference matches.
- `test_core`: duplicate/followup selection, padded orthonormal probe basis, bounded optimization and full training coverage.
- `train_pair --stop 50`: completed v1 disposable runtime diagnostic with replacement sampling. Some declared train images were never drawn, so this is not the final participation experiment.
- `benchmark_batch`: zero-update batching comparison. An unscaled backward check initially disagreed; with the actual 1024 gradient scale, batch 4 relative gradient difference was 0.006097 and relative loss error 0.00001586. This is approximate FP16 agreement, not bitwise equality.
- `train_coverage --model model_1 --stop 1000` and model_2: v2 separate checkpoints, same base initialization, four-image batch, full epoch permutations. Every selected training image is used by step 228; coverage and exposure counts are verified.
- `verify_training --model model_1 --step 1000`: CPU checkpoint, optimizer, exposure, trace and membership verification.
- `generate_quality --models first --training-dir training_coverage_v2`: fixed four-prompt visual comparison. Use both for the full pair.
- `quality_loss --model model_1`: paired base/target diffusion losses on 32 independent quality patients.
- `run_probe_smoke --training-dir training_coverage_v2 --step 1000 --count-per-model 8`: actual U response computation on fitting patients only; no performance or novelty claim.
- `run_probe_smoke --training-dir training_coverage_v2 --step 1000 --count-per-model 1 --scenario E`: minimal seen-record execution control, one fitting patient evaluated with both models.
- `verify_probe --scenario U --count 8` and `verify_probe --scenario E --count 1`: separate checks of the realized membership pair, observation roles, fold arithmetic, operation counts and checkpoint binding.
- `verify_quality`: verify the fixed generation outputs and checkpoint binding; visual/denoising checks remain separate from clinical utility.
- `encoder_control`: fit-only public-feature baseline with selection-set diagnostics, without calibration/test target queries.
- `write_status`: regenerate the research execution record from saved reports.
- `analyze_u_smoke`: CPU-only analysis of the existing eight U fitting patients. Fixed score signs, per-model AUROC, paired differences, rank stability and patient-deletion sensitivity; no additional model execution or fitting.
- `verify_u_analysis`: independent sklearn/scipy and raw-score checks of the descriptive results.
- `render_u_analysis`: render the saved analysis with legible patient labels; save PNG/PDF and plot hashes.
- `audit_readiness`: CPU-only design/record audit, actual cached conditioning scale and reference checks, plus algebraic counterexamples to overinterpreting rank invariance or increased variance. Saves `review_20260914/review_facts.json`; does not execute the proposed numerical GPU diagnostic.
- `reverify_frozen`: reruns the original CPU verifiers and core tests into new reports, preserving original evidence and checking eleven protected hashes.
- `probe_verified`: separately versioned instrumentation retaining support objectives, all coefficient/gradient/projection steps, and raw paired losses without changing the frozen probe arithmetic.
- `run_numerical_diagnostic_v2` / `verify_numerical_diagnostic --output-tag numerical_v2`: actual same-input FP16/FP32 finite differences and raw checkpointing/legacy gradient comparisons; v1 is preserved.
- `run_basic_controls` / `verify_basic_controls`: the existing eight patients' E/U raw denoising controls under generic/matched prompts; record batch drift separately from metadata/arithmetic integrity.
- `run_traced_replay` / `verify_traced_replay`: all original U8 model-patient evaluations with complete traces; 324F/72B per patient includes 12 extra final-support forwards.
- `run_endpoint_precision` / `verify_endpoint_precision`: the first original patient's saved coefficients evaluated in FP32 with both targets, without reoptimization; 480F/0B total.
- `run_endpoint_precision_u8` / `verify_endpoint_precision_u8`: after a paired-sign precision change was detected, uniformly reevaluate all original U8 endpoints and four score types in FP32 at frozen FP16 coefficients, with original outputs preserved; 3840F/0B.
- `finalize_verification`: verify stage evidence bindings and protected original hashes again, then write the aggregate status without declaring attack efficacy.
- `run_secmi_diagnostic`: fixed FP32 SecMI-stat screen using the SecMI implementation preserved in the CDI authors' repository. Original fit8 and selection40 stay separate; E/U each use two images per patient and both existing targets. Fixed t=100, step=10, negative L2, patient mean primary; 384 image evaluations, 4608F/0B, no fitting or target training. This requires denoiser/model access, not only a finished-image generation API.
- `verify_secmi_diagnostic`: independently recompute saved raw latent norms, patient aggregation, realized labels/exposures and score statistics; apply the predeclared selection40 screening gate. Outputs are in `baseline_screen_20260914/secmi_v1`.
- `check_secmi_raw_states`: direct CPU cross-check of all saved FP32 norms, exact patient aggregation, realized labels and protected hashes; saves a separate report without changing attack outputs.

Verification outputs are under `verification_20260914`. Existing run directories refuse overwrite; new GPU reruns require distinct output tags. The numerical interpretation is `U_VERIFICATION_RESULTS.md`; the subsequent fixed SecMI E/U screen and continuation decision are in `U_BASELINE_SCREEN.md` in the research folder. Numerical checks do not establish attack efficacy. `run_probe_smoke` and its original FP16 scores are historical reproduction artifacts; future extensions must explicitly apply and report the versioned FP32 endpoint policy instead of treating tiny old FP16 paired differences as stable signals.

The probe differentiates the base and target separately **before switching adapter state**. This prevents checkpoint recomputation using the wrong adapter. Gradient scaling is also used for input derivatives. Model parameters stay frozen; support and query noise are separate. Counts 312 forward/72 backward are model-example evaluations per two-image patient, not API calls.

The 400 patients are partitioned into fit 80, selection 40, calibration 140, test 140.
Each target has 200 member/200 nonmember candidate patients. There are 64 reference,
32 quality and 256 common-background patients. Every target trains on 912 images.
Study labels concern incremental fine-tuning, not unknown public-base pretraining.

No acquisition date/study UID is available here: follow-up number separation plus file/pixel/perceptual checks is an explicit limitation. The sample covers patients with at least four eligible PA records. Seventy nonmember calibration patients per model do not support a 1% rank-calibrated operating point; this remains a 5% pilot with uncertainty.

The original candidate is a hypothesis. Strong CDI/MoFit/gradient comparisons, medical utility,
independent-seed confirmation and DP protection comparisons remain later stages.
