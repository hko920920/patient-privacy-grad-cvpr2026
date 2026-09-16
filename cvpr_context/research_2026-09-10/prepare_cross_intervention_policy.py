"""Freeze the next stage-2 diagnostic before any q-control training/results."""
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

RESEARCH=Path(__file__).resolve().parent
ROOT=RESEARCH.parent.parent/'code_working'
RUN=ROOT/'_reports/cvpr_u_pilot_v1_001'
NEW=RUN/'baseline_screen_20260915'
OLD=RUN/'baseline_screen_20260914'
DEST=RESEARCH/'spec_sources/patient_cross_intervention_analysis_contract_20260915.json'

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    assert not DEST.exists()
    original_policy=RESEARCH/'spec_sources/masked_cdi_fixed_scorer_analysis_contract_20260915.json'
    addendum=RESEARCH/'spec_sources/masked_cdi_leave_patient_out_addendum_20260915.json'
    oldp=read(original_policy);add=read(addendum)
    with (RUN/'cohort/evaluation_images.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    p,q='14393','4092'
    selected=sorted({r['patient_id'] for r in rows if r['eval_role']=='selection'},key=int)
    assert len(selected)==40 and q in selected and p not in selected
    u_order=[p]+selected
    cases=[]
    for scenario,pids in [('U',u_order),('E',[p,q])]:
        for pid in pids:
            rr=sorted([r for r in rows if r['patient_id']==pid and r['record_role']==('U_observed' if scenario=='U' else 'train_candidate')],key=lambda r:r['image_id'])
            assert len(rr)==2
            cases.append(dict(patient_id=pid,scenario=scenario,images=[r['image_id'] for r in rr],role=rr[0]['eval_role'],group=rr[0]['assignment_group']))
    prior_fit=NEW/'cdi_masked_u82_v1/fixed_scorer_analysis_v1'
    assert read(prior_fit/'independent_saved_scorer_audit.json')['status']=='PASS_INDEPENDENT_MASKED_CDI_FIT79_INPUT_SCALER_PREDICTION_AUDIT'
    for par in read(prior_fit/'leave_patient_out79_parameters.json'):
        assert p not in par['fitting_patient_ids'] and q not in par['fitting_patient_ids']
    files=[Path(__file__),original_policy,addendum,RUN/'cohort/lock.json',RUN/'cohort/evaluation_images.csv',
        RUN/'cohort/model_1_train.csv',RUN/'training_coverage_v2/protocol.json',
        NEW/'masked_patient_training_v1/verification.json',NEW/'cdi_masked_u82_v1/verification.json',
        NEW/'cdi_masked_u82_v1/results.json']
    files += [prior_fit/n for n in ['protocol.json','provenance.json','analysis.json','patient_scores.json','original_fitted_parameters.json',
        'leave_patient_out79_parameters.json','leave_patient_out79_fit_inputs.json','independent_saved_scorer_audit.json']]
    files += [OLD/d/n for d in ['cdi_u_cohort_v1','cdi_e_cohort_v1'] for n in ['results.json','verification.json']]
    files += [ROOT/'u_patient_audit'/n for n in ['cdi_adapter.py','train_coverage.py','train_pair.py',
        'run_masked_patient_training.py','analyze_masked_cdi_fixed_scorers.py']]
    c=dict(schema='patient-cross-intervention-analysis/v1',created_utc=datetime.now(timezone.utc).isoformat(),current_step=2,
        focal_patients=dict(p=p,q=q),patient_rows=cases,patient_order_U=u_order,patient_order_E=[p,q],
        q_selection_rule='Numeric first selection-A patient with two E records each exposed four times in the original M1 ledger; no q-intervention scores used',
        selected_q_not_replaced_after_results=True,
        prior_p_is_already_explored_case=True,selection40_is_already_explored_development=True,
        original_directory_U=str((OLD/'cdi_u_cohort_v1').resolve()),original_directory_E=str((OLD/'cdi_e_cohort_v1').resolve()),
        p_control_U_directory=str((NEW/'cdi_masked_u82_v1').resolve()),prior_scorer_directory=str(prior_fit.resolve()),
        new_output_directory=str((NEW/'cdi_cross_patient_intervention_v1').resolve()),
        q_training_directory=str((NEW/'masked_patient_4092_training_v1').resolve()),
        branches=['treatment','p_control','q_control'],new_records=dict(q_control_U=82,q_control_E=4,p_control_E=4,total=90),
        reused_records=dict(treatment_U=82,treatment_E=4,p_control_U=82,total=168),
        training=dict(source_model='model_1',updates=1000,initialization='original initialization; do not resume from step43',
            q_first_contribution_update=37,expected_q_masked_updates=[37,61,271,444,553,632,837,845],
            same_schedule_noise_optimizer_and_original_mean_denominator=True,scheduled_slots=4000,effective_slots=3992,
            p_exposures_in_q_control=8,q_exposures_in_q_control=0,
            conventional_resized_manifest_retrain=False,original_treatment_reused_after_exact_prior_verification=True),
        methods=oldp['methods'],clean_primary_methods=add['clean_primary_methods'],
        arms=['original_fit80_frozen_patient_seen','leave_patient_out79_refit'],
        arm_policy='Reuse the already fitted and independently audited coefficients/scalers/C in both arms; no new fitting',
        primary_arm='leave_patient_out79_refit',original_policy_sha256=sha(original_policy),leave_out_addendum_sha256=sha(addendum),
        DINO='Reuse target-independent original U scores only, exact zero response; no E DINO invented',
        formulae=dict(D_p_i='g(treatment,X_i)-g(p_control,X_i)',D_q_i='g(treatment,X_i)-g(q_control,X_i)',
            h_i='D_p_i-D_q_i',p_own_minus_cross='h_p',q_own_minus_cross='-h_q',
            K='h_p-h_q = D_p_p+D_q_q-D_p_q-D_q_p',
            base_cancelled_K='g(q_control,X_p)-g(p_control,X_p)-g(q_control,X_q)+g(p_control,X_q)'),
        scenarios=['U','E'],E_scorer_interpretation='Same U-fitted scorer applied to E as a response diagnostic, not an E-optimized attack efficacy test',
        contexts=['selection_A','selection_B','selection_all','all41','background39_A','background39_B','background39_all','p_others40','q_others40'],
        context_warning='selection40 includes q; it is not forty nonfocal patients for the q intervention. Shared background39 excludes both p and q (A19/B20).',
        statistics='All fixed method/scenario cells and signed own-minus-cross contrasts; descriptive distributions only',
        decision_policy=dict(no_new_efficacy_cutoff=True,no_significance_test=True,
            joint_positive_requires_both_own_minus_cross_positive='K positive alone does not imply this; signs are descriptive and checked against arithmetic precision',
            positive_existing_U_response='Evidence of a fixed-pair matching response already accessible to these existing scores; not new-method necessity',
            E_only_response='No matching U-transfer evidence in this fixed pair/scorers; not universal U impossibility',
            mixed_or_negative_response='No consistent fixed-pair support for the proposed matching-effect explanation; do not select a favorable method or alter q',
            all_cases='K is an analyst-only paired intervention diagnostic, never an externally available attack score or population TPR/FPR'),
        no_new_fitting=True,no_new_C_search=True,no_R_changes=True,no_new_AUC=True,no_CI=True,no_p_value=True,
        calibration_or_test_used=False,stage2_completion_claim=False,new_attack_claim=False,
        limits=['Two fixed patient interventions in one shared training realization; observations across other patients are not independent intervention repeats.',
            'K removes additive intervention/query offsets but not multiplicative sensitivity, pathology/acquisition similarity, or nonlinear optimizer-path interactions.',
            'The actual target training contains p when producing prior auxiliary79 features; direct auxiliary-fit exclusion is not global independence.',
            'Historical tuned-C selection included p; all such results remain secondary.',
            'Removing effective gradient contributions while retaining scheduled zero-loss inputs is the estimand; base pretraining membership remains unknown.',
            'Influence matrices, mean/reference calibration and common-versus-individual effects have prior art; no novelty follows from this diagnostic alone.'],
        execution_binding_policy='Analysis policy fixed before q training/output; producer code frozen before its execution; checkpoint and completed independent training verification are bound when extraction contract is prepared; verifier code gets its own later verification protocol',
        frozen_sha256={str(f.resolve()):sha(f) for f in files})
    with DEST.open('x',encoding='utf-8') as f:json.dump(c,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(path=str(DEST),sha256=sha(DEST),bound_files=len(files),patient_cases=len(cases),new_records=90)))

if __name__=='__main__':main()
