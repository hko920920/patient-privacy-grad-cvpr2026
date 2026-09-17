"""Prepare one fixed-compute real-support contrast. No torch/model or pixel reads."""
from downstream_utility.common import CODE,RESEARCH,sha,read,rows,save,now,require,seed
from downstream_utility.data_v2 import pools,draw
from pathlib import Path
from collections import Counter
import csv,json
import numpy as np

OUT=CODE/'_reports/real_support_plan_20260917_v1'
PROFILE=CODE/'_reports/downstream_profile_20260917_v3'
OLD=CODE/'_reports/downstream_development_20260917_v1'
AUDIT=CODE/'_reports/patient_usage_audit_20260917_v1'
PROTOCOL=RESEARCH/'TRACK1_REAL_SUPPORT_DIAGNOSTIC_PROTOCOL_20260917.md'
SEEDS=[11,23,37]

def write_csv(path,rr):
    with path.open('x',encoding='utf-8',newline='')as f:
        w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)

def statistics(rr):
    positives=[r for r in rr if int(r['label'])]
    pp={r['patient_id']for r in positives};pn={r['patient_id']for r in rr if not int(r['label'])}
    return dict(images=len(rr),patients=len({r['patient_id']for r in rr}),positive_images=len(positives),
                positive_patients=len(pp),negative_patients=len(pn),mixed_label_patients=len(pp&pn))

def main():
    require(not OUT.exists(),'No overwrite')
    rr=rows(PROFILE/'real_manifest_private.csv')
    public=[r for r in rr if r['role']=='public'];extra=[r for r in rr if r['role']=='classifier_selection']
    dev=[r for r in rr if r['role']=='method_development'];private=[r for r in rr if r['role']=='private']
    require(statistics(public)['patients']==672 and len(public)==813,'Public')
    require(statistics(extra)['patients']==2027 and len(extra)==5097,'Former selection')
    expanded=sorted(public+extra,key=lambda r:r['image_id'])
    require(statistics(expanded)['positive_patients']==120 and len(expanded)==5910,'Expanded support')
    ledger={r['patient_id']:r for r in rows(AUDIT/'patient_usage_private.csv')}
    denied={r['patient_id']for r in ledger.values()if r['locked_thesis_final']=='1' or r['locked_cvpr_calibration_test']=='1'}
    reserved={r['patient_id']for r in rows(AUDIT/'provisional_patient_split_private.csv')if r['provisional_role']=='reserved_confirmation'}
    resources=read(CODE/'_reports/downstream_master_20260917_v1/resources.json')
    expert=rows(Path(resources['source_paths']['expert']))
    expert_patients={r['Patient ID']for r in expert};expert_ids={r['Image Index']for r in expert}
    expanded_patients={r['patient_id']for r in expanded}
    require(not expanded_patients&(denied|reserved|expert_patients),'Locked patient')
    require(not set(r['image_id']for r in expanded)&expert_ids,'Expert image')
    overlap={}
    for name,z in [('method_development',dev),('private80',private)]:
        overlap[name]={key:len({r[key]for r in expanded}&{r[key]for r in z})for key in ['patient_id','image_id','sha256']}
        require(all(v==0 for v in overlap[name].values()),'Study disjointness')
    overlap.update(expert_patients=len(expanded_patients&expert_patients),reserved_patients=len(expanded_patients&reserved),
                   locked_patients=len(expanded_patients&denied))
    require(all(ledger[r['patient_id']]['original_thesis_role']=='private_train' for r in extra),'Preserve original role')
    manifest=[dict(r,planned_namespace=('real/public' if r['role']=='public' else 'diagnostic_real/former_classifier_selection'),
                   original_thesis_role=ledger[r['patient_id']]['original_thesis_role'],
                   planned_role='Rwide_diagnostic_training_pool',status='planned_not_yet_optimizer_consumed')for r in expanded]
    overlay=[]
    for pid in sorted({r['patient_id']for r in extra},key=int):
        overlay.append(dict(patient_id=pid,original_thesis_role=ledger[pid]['original_thesis_role'],
                            previous_cvpr_role='classifier_selection',planned_role='Rwide_diagnostic_training_pool',
                            prior_preprocessing_and_calibration_consumed=1,
                            planned_optimizer_consumed=0,may_be_validation_for_Rwide_after_execution=0,
                            metadata_only=True))
    bank=pools(expanded);oldbank=pools(public);schedule=[];summaries={}
    for s in SEEDS:
        oldtrace=read(OLD/f'runs/R1_{s}/trace.json')['trace']
        for step in range(400):
            past=draw(oldbank,16,seed(s,step,'public-shared'))+draw(oldbank,16,seed(s,step,'public-extra'))
            require([r['image_id']for r in past]==oldtrace[step]['image_ids'],'Unchanged R1 sampling recipe')
            chosen=draw(bank,16,seed(s,step,'public-shared'))+draw(bank,16,seed(s,step,'public-extra'))
            require([int(r['label'])for r in chosen]==[0]*8+[1]*8+[0]*8+[1]*8,'Class and slot balance')
            for slot,r in enumerate(chosen):
                schedule.append(dict(seed=s,step=step+1,slot=slot,image_id=r['image_id'],patient_id=r['patient_id'],
                                     original_role=r['role'],label=int(r['label']),array_index=int(r['index'])))
        planned=[r for r in schedule if r['seed']==s];detail={}
        for role in ['public','classifier_selection']:
            for label in [0,1]:
                subset=[r for r in planned if r['original_role']==role and r['label']==label]
                ip=Counter(r['image_id']for r in subset);pp=Counter(r['patient_id']for r in subset)
                eligible=[r for r in expanded if r['role']==role and int(r['label'])==label]
                pi=[ip[r['image_id']]for r in eligible];pcount=[pp[p]for p in sorted({r['patient_id']for r in eligible})]
                detail[f'{role}/{label}']=dict(draws=len(subset),eligible_images=len(eligible),eligible_patients=len(pcount),
                   seen_images=len(ip),seen_patients=len(pp),image_exposure_min=min(pi),image_exposure_median=float(np.median(pi)),
                   image_exposure_max=max(pi),patient_exposure_min=min(pcount),patient_exposure_median=float(np.median(pcount)),
                   patient_exposure_max=max(pcount))
        summaries[str(s)]=dict(actual_not_executed=True,planned_unique_images=len({r['image_id']for r in planned}),
                              planned_unique_patients=len({r['patient_id']for r in planned}),sources=detail)
    OUT.mkdir()
    write_csv(OUT/'expanded_training_manifest_private.csv',manifest)
    write_csv(OUT/'role_overlay_private.csv',overlay)
    write_csv(OUT/'planned_batches_private.csv',schedule)
    write_csv(OUT/'method_development_unchanged_private.csv',dev)
    save(OUT/'state_before.json',{n:read(RESEARCH/n)for n in ['research_state.json','patient_baseline_spec.json']})
    sources=[PROFILE/'real_manifest_private.csv',PROFILE/'image_audit.json',PROFILE/'data.json',
             AUDIT/'patient_usage_private.csv',AUDIT/'provisional_patient_split_private.csv',Path(resources['source_paths']['expert']),
             CODE/'_reports/downstream_master_20260917_v1/resources.json',
             OLD/'contract.json',OLD/'calibration_choice.json',OLD/'result.json',OLD/'verification.json',OLD/'cluster_bootstrap.npz',
             CODE/'_reports/classifier_generalization_20260917_v1/result.json',
             CODE/'_reports/classifier_generalization_20260917_v1/verification.json',
             CODE/'_reports/lora_transfer_20260917_v1/result.json',
             RESEARCH/'TRACK1_CURRENT_HEAD_BRANCH_CLOSURE_20260917.md',PROTOCOL]
    sources += list(Path(__file__).parent.glob('*.py'))
    sources += [CODE/'downstream_utility'/n for n in ['data_v2.py','train_v2.py','common.py','run_v2.py','development_v1.py']]
    for s in SEEDS:
        sources += [OLD/f'runs/R1_{s}'/n for n in ['trace.json','exposure.json','state_400.pt','complete.json']]
        sources += [OLD/f'evaluation/{a}_{s}.npz'for a in ['R0','R1']]
    sources+=list(OUT.glob('*.csv'))+[OUT/'state_before.json']
    plan=dict(schema='real-support-generalization-plan/v1',created_utc=now(),status='design_complete_no_new_model_execution',
       current_step=2,protocol=PROTOCOL.name,arm='Rwide_real_diagnostic_NONDP',new_model_execution=0,new_pixel_decode=0,
       runtime_implemented=False,final_ready=False,expert_opened=False,reserved_opened=False,DP=False,
       counts=dict(public=statistics(public),former_classifier_selection=statistics(extra),expanded=statistics(expanded),
                   method_development=statistics(dev)),overlaps=overlap,
       additional_source_original_role='private_train',original_role_files_changed=False,
       role_overlay_status='planned_not_yet_optimizer_consumed',
       sampling=dict(class_balanced=True,within_label='uniform patient then uniform image within patient/label',
                     original_role_mixture='emergent from class-specific patient counts, NOT 50:50',
                     half_batch_size=16,labels_per_half=[0]*8+[1]*8,
                     stream_names=['public-shared','public-extra'],same_random_streams_not_same_image_ids=True),
       seeds=SEEDS,steps=400,batch=32,learning_rate=.0001,weight_decay=.0001,
       optimizer=dict(name='AdamW',betas=[.9,.999],eps=1e-8,foreach=False),
       initialization='same per-seed ImageNet initialization as prior R1, fresh optimizer',
       selected_calibration_sha256=sha(OLD/'calibration_choice.json'),
       scheduled_presentations=len(schedule),planned_exposure=summaries,
       primary_comparison='Rwide-R1 mean per-seed AUROC',secondary_comparison='Rwide-R0 AUROC/AP',
       bootstrap=dict(iterations=2000,saved_counts=str(OLD/'cluster_bootstrap.npz'),
                      patient_clusters=2026,interval=.95,conditional_development_only=True),
       improvement_candidate=dict(mean_AUROC_gain_min=.01,positive_seeds_min=2,mean_AP_non_decrease=True,
                                  implies_adequate_absolute_performance=False,implies_DP_entry=False),
       execution_budget=dict(main_updates=1200,independent_probe_updates=4,new_generated_images=0,
                             existing_R1_path_witness_images=64*3,method_development_image_forward=5047*3,
                             training_eval_image_forward=sum(v['planned_unique_images']for v in summaries.values()),
                             DP=0,expert_pixels=0,reserved_pixels=0),
       prior_R1_pure_training_seconds=sum(read(OLD/f'runs/R1_{s}/trace.json')['seconds']for s in SEEDS),
       future_total_work_estimate_minutes=[30,50],
       limitations=['Changes diversity, positive support, per-image repetition, source weights and label/cohort composition together',
                    'Same adaptive weak-label development reused, not a new independent confirmation',
                    'Fixed 400 updates is a compute comparison, not a convergence guarantee',
                    'One additional pool, three classifier seeds, no subset or budget sweep'],
       source_sha256={str(p.resolve()):sha(p)for p in sources})
    save(OUT/'plan.json',plan)
    save(RESEARCH/'spec_sources/real_support_diagnostic_plan_20260917.json',plan)
    print(json.dumps(dict(status=plan['status'],counts=plan['counts'],planned_exposure=summaries,
                         scheduled_presentations=len(schedule),new_model_execution=0),indent=2))

if __name__=='__main__':main()
