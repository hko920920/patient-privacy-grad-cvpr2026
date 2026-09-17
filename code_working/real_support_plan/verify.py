"""Independent metadata/schedule checks. No model loading, pixels or new performance."""
from pathlib import Path
from collections import Counter,defaultdict
import csv,hashlib,json
import numpy as np
from downstream_utility.common import CODE,RESEARCH,sha,read,rows,save,now,require

OUT=CODE/'_reports/real_support_plan_20260917_v1'
OLD=CODE/'_reports/downstream_development_20260917_v1'

def independent_draw(records,run_seed,step,stream):
    by_label={0:defaultdict(list),1:defaultdict(list)}
    for r in sorted(records,key=lambda r:r['image_id']):by_label[int(r['label'])][r['patient_id']].append(r)
    value='downstream-profile-20260917-v1|'+'|'.join(map(str,[run_seed,step,stream]))
    number=int(hashlib.sha256(value.encode()).hexdigest()[:15],16)
    rng=np.random.Generator(np.random.PCG64(number));result=[]
    for label in (0,1):
        patient_ids=sorted(by_label[label])
        for _ in range(8):
            pid=patient_ids[int(rng.integers(len(patient_ids)))]
            imgs=by_label[label][pid]
            result.append(imgs[int(rng.integers(len(imgs)))]['image_id'])
    return result

def main():
    count=0
    def check(v,msg):
        nonlocal count
        count+=1;require(v,msg)
    plan=read(OUT/'plan.json')
    check(plan==read(RESEARCH/'spec_sources/real_support_diagnostic_plan_20260917.json'),'Public exact plan')
    for p,h in plan['source_sha256'].items():check(sha(p)==h,'Frozen metadata/source')
    manifest=rows(OUT/'expanded_training_manifest_private.csv');overlay=rows(OUT/'role_overlay_private.csv')
    allrows=rows(CODE/'_reports/downstream_profile_20260917_v3/real_manifest_private.csv')
    chosen=[r for r in allrows if r['role']in ['public','classifier_selection']]
    lookup={r['image_id']:r for r in chosen}
    check(len(manifest)==5910 and len({r['image_id']for r in manifest})==5910,'Unique all-eligible pool')
    check({r['image_id']for r in manifest}==set(lookup),'No outcome-based subset')
    for r in manifest:
        original=lookup[r['image_id']]
        for k in original:check(r[k]==original[k],'Unchanged original image metadata')
        check(r['status']=='planned_not_yet_optimizer_consumed','Reservation only')
        check(r['planned_namespace']==('real/public' if r['role']=='public' else 'diagnostic_real/former_classifier_selection'),'Original source namespace')
    donors=[r for r in chosen if r['role']=='classifier_selection']
    donor_ids={r['patient_id']for r in donors}
    check({r['patient_id']for r in overlay}==donor_ids and len(overlay)==2027,'Complete role overlay')
    for r in overlay:
        check(r['original_thesis_role']=='private_train' and r['previous_cvpr_role']=='classifier_selection','Original role history')
        check(r['planned_optimizer_consumed']=='0' and r['may_be_validation_for_Rwide_after_execution']=='0','No validation reuse after train')
    dev=rows(OUT/'method_development_unchanged_private.csv')
    check(dev==[r for r in allrows if r['role']=='method_development'],'Evaluation unchanged')
    for key in ['patient_id','image_id','sha256']:
        check(not {r[key]for r in manifest}&{r[key]for r in dev},'Train/dev disjoint '+key)
        priv=[r for r in allrows if r['role']=='private']
        check(not {r[key]for r in manifest}&{r[key]for r in priv},'Private80 excluded '+key)
    audit=CODE/'_reports/patient_usage_audit_20260917_v1'
    reserved={r['patient_id']for r in rows(audit/'provisional_patient_split_private.csv')if r['provisional_role']=='reserved_confirmation'}
    locked={r['patient_id']for r in rows(audit/'patient_usage_private.csv')if r['locked_thesis_final']=='1' or r['locked_cvpr_calibration_test']=='1'}
    resource=read(CODE/'_reports/downstream_master_20260917_v1/resources.json')
    expert=rows(resource['source_paths']['expert'])
    check(not {r['patient_id']for r in manifest}&(reserved|locked|{r['Patient ID']for r in expert}),'Protected populations')
    check(not {r['image_id']for r in manifest}&{r['Image Index']for r in expert},'Expert image disjoint')
    # Count directly rather than import the planner's statistics or sampler.
    for role,key in [('public','public'),('classifier_selection','former_classifier_selection'),(None,'expanded')]:
        z=[r for r in manifest if role is None or r['role']==role]
        pos=[r for r in z if int(r['label'])];pp={r['patient_id']for r in pos};pn={r['patient_id']for r in z if not int(r['label'])}
        expected=dict(images=len(z),patients=len({r['patient_id']for r in z}),positive_images=len(pos),
                      positive_patients=len(pp),negative_patients=len(pn),mixed_label_patients=len(pp&pn))
        check(expected==plan['counts'][key],'Independent support counts')
    schedule=rows(OUT/'planned_batches_private.csv')
    check(len(schedule)==38400,'Three fixed 400x32 schedules')
    batches=defaultdict(list)
    for row in schedule:batches[(int(row['seed']),int(row['step']))].append(row)
    check(len(batches)==1200,'1200 future updates')
    public=[r for r in chosen if r['role']=='public'];oldtraces={s:read(OLD/f'runs/R1_{s}/trace.json')['trace']for s in plan['seeds']}
    for s in plan['seeds']:
        seen=[]
        for step in range(400):
            rr=batches[(s,step+1)]
            check([int(x['slot'])for x in rr]==list(range(32)),'Slot order')
            expected=independent_draw(chosen,s,step,'public-shared')+independent_draw(chosen,s,step,'public-extra')
            check([x['image_id']for x in rr]==expected,'Independent exact expanded sampling')
            past=independent_draw(public,s,step,'public-shared')+independent_draw(public,s,step,'public-extra')
            check(past==oldtraces[s][step]['image_ids'],'Existing R1 recipe unchanged')
            check([int(x['label'])for x in rr]==[0]*8+[1]*8+[0]*8+[1]*8,'Each half balanced')
            for row in rr:
                orig=lookup[row['image_id']]
                check(row['patient_id']==orig['patient_id'] and row['original_role']==orig['role'] and row['label']==orig['label']
                      and row['array_index']==orig['index'],'Scheduled record binding')
            seen+=rr
        expected=plan['planned_exposure'][str(s)]
        check(len({r['image_id']for r in seen})==expected['planned_unique_images'],'Unique scheduled images')
        check(len({r['patient_id']for r in seen})==expected['planned_unique_patients'],'Unique scheduled patients')
        for role in ['public','classifier_selection']:
            for label in (0,1):
                selected=[r for r in seen if r['original_role']==role and int(r['label'])==label]
                pop=[r for r in chosen if r['role']==role and int(r['label'])==label]
                ic=Counter(r['image_id']for r in selected);pc=Counter(r['patient_id']for r in selected)
                ie=[ic[r['image_id']]for r in pop];pe=[pc[i]for i in {r['patient_id']for r in pop}]
                values=dict(draws=len(selected),eligible_images=len(pop),eligible_patients=len(pe),
                  seen_images=len(ic),seen_patients=len(pc),image_exposure_min=min(ie),image_exposure_median=float(np.median(ie)),
                  image_exposure_max=max(ie),patient_exposure_min=min(pe),patient_exposure_median=float(np.median(pe)),patient_exposure_max=max(pe))
                check(values==expected['sources'][f'{role}/{label}'],'Scheduled exposure summary')
    with np.load(OLD/'cluster_bootstrap.npz',allow_pickle=False)as z:
        check(z['patient_counts'].shape==(2000,2026),'Frozen bootstrap shape')
        check(set(z['patient_ids'].tolist())=={int(r['patient_id'])for r in dev},'Frozen bootstrap patients')
        check(np.all(z['patient_counts'].sum(axis=1)==2026),'Cluster count preservation')
    before=read(OUT/'state_before.json')
    for name,content in before.items():check(read(RESEARCH/name)==content,'No current-state mutation during planning')
    check(plan['new_model_execution']==0 and plan['new_pixel_decode']==0 and not plan['runtime_implemented'],'Plan only')
    check(plan['execution_budget']['main_updates']==1200 and plan['execution_budget']['independent_probe_updates']==4,'Future updates vs current zero')
    record=dict(status='PASS_METADATA_ROLE_BOUNDARIES_AND_INDEPENDENT_SCHEDULE',created_utc=now(),checks=count,
       original_R1_batches_reproduced=1200,new_planned_batches=1200,new_planned_slots=38400,
       metadata_only=True,new_model_inference=0,new_training=0,new_pixel_decode=0,new_generation=0,
       expert_reserved_opened=False,original_roles_changed=False,runtime_implemented=False,
       plan_sha256=sha(OUT/'plan.json'),protocol_sha256=sha(RESEARCH/plan['protocol']),
       public_plan_sha256=sha(RESEARCH/'spec_sources/real_support_diagnostic_plan_20260917.json'),
       source_sha256_preserved=True)
    save(OUT/'verification.json',record)
    save(RESEARCH/'spec_sources/real_support_diagnostic_plan_verification_20260917.json',record)
    print(json.dumps(record,indent=2))
if __name__=='__main__':main()
