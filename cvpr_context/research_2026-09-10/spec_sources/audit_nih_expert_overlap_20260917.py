"""Read-only patient-level overlap inventory; no images, model scores or role edits."""
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict, Counter
import csv
import hashlib
import json
import time

R = Path(__file__).resolve().parents[1]
C = R.parent.parent / 'code_working'
OUT = C / '_reports/nih_expert_overlap_20260917_v1'
PATHS = {
    'expert': C/'_reports/nih_expert_label_public_routes_20260917_v1/nih_all14_from_public_mirror_810.csv',
    'recovery': C/'_reports/nih_expert_label_public_routes_20260917_v1/recovery_record.json',
    'ledger': C/'_reports/patient_usage_audit_20260917_v1/patient_usage_private.csv',
    'prior_audit': C/'_reports/patient_usage_audit_20260917_v1/audit.json',
    'evidence': C/'_reports/patient_usage_audit_20260917_v1/patient_evidence_private.json',
    'split': C/'_reports/patient_usage_audit_20260917_v1/provisional_patient_split_private.csv',
    'pad_new': C/'_reports/padchest_validation_20260917_v1/evaluator_exclusions_private.csv',
    'pad_all': C/'_reports/padchest_validation_20260917_v1/selected_images_private.csv',
    'chex_new': C/'_reports/chexzero_validation_20260917_v1/evaluator_exclusions_private.csv',
    'census': C/'_data/derived/nih_cxr14_pa_target_enriched_v1/official_pa_test_census_private.csv',
    'thesis_roles': C/'_data/derived/nih_cxr14_pa_target_enriched_v1/patients_private.csv',
    'metadata': C/'_data/intake/nih_chestxray14_metadata/Data_Entry_2017_v2020.csv',
    'official_test': C/'_data/intake/nih_chestxray14_metadata/test_list.txt',
    'official_trainval': C/'_data/intake/nih_chestxray14_metadata/train_val_list.txt',
    'head': C/'_reports/medical_head_20260916_v1/patients.json',
    'reference': C/'_reports/medical_head_20260916_v1/reference.json',
    'backbone_plan': R/'spec_sources/public_medical_backbone_plan_20260916_v1/patients.csv',
    'backbone_trace': C/'_reports/public_medical_backbone_20260916_v1/trace.jsonl',
    'm1': C/'_reports/cvpr_u_pilot_v1_001/cohort/model_1_train.csv',
    'm2': C/'_reports/cvpr_u_pilot_v1_001/cohort/model_2_train.csv',
    'dryrun': C/'_reports/nih_cxr14_k5_private_research_dryrun_v1_001/restricted_runtime_diagnostics.json',
}

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def read(k):
    return json.loads(PATHS[k].read_text(encoding='utf-8-sig'))

def rows(k):
    with PATHS[k].open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def dump(p, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

def ids(records, key='patient_id'):
    return {int(x[key]) for x in records}

def main():
    started=time.perf_counter()
    if OUT.exists():
        raise FileExistsError('Do not overwrite an existing audit')
    hashes={k:sha(p) for k,p in PATHS.items()}
    OUT.mkdir()
    dump(OUT/'contract.json', {
        'created_utc':datetime.now(timezone.utc).isoformat(),
        'source_inputs':{k:{'path':str(p),'sha256':hashes[k]} for k,p in PATHS.items()},
        'source_code_sha256':sha(Path(__file__)),
        'scope':'Patient metadata only; all810 source-labelled images, no outcome-based sampling.',
        'rules':[
            'A patient is positive for a finding if any acquired expert-labelled image is positive.',
            'Patient any-positive and image-positive counts are reported separately.',
            'Apply old individual usage flags plus both later evaluator consumed cohorts; do not use exclusive audit grade alone.',
            'Missing ledger membership is unresolved, not evidence of never-used status.',
            'Locked thesis/CVPR and reserved confirmation roles stay unchanged; metadata intersections do not authorize new evaluation.',
            'No final test adopted, no model execution, no third-classifier rescue.',
            'All14-negative is not asserted normal: other findings and official normal/abnormal are absent from this derivative.',
        ],
    })
    recovery=read('recovery')
    if hashes['expert']!=recovery['export_sha256']:
        raise ValueError('Recovered CSV binding mismatch')
    previous=read('prior_audit')['outputs_sha256']
    for key in ['ledger','evidence','split']:
        if hashes[key]!=previous[PATHS[key].name]:
            raise ValueError('Old audited source changed: '+key)
    images=rows('expert')
    diseases=[k for k in images[0] if k not in ['Image Index','Patient ID','View Position']]
    if len(images)!=810 or len(diseases)!=14 or len({x['Image Index'] for x in images})!=810:
        raise ValueError('Expert file schema/count mismatch')
    by_patient=defaultdict(list)
    meta={x['Image Index']:x for x in rows('metadata')}
    test=set(PATHS['official_test'].read_text().splitlines())
    trainval=set(PATHS['official_trainval'].read_text().splitlines())
    for x in images:
        image=x['Image Index'];pid=int(x['Patient ID'])
        if image not in meta or int(meta[image]['Patient ID'])!=pid or meta[image]['View Position']!='PA':
            raise ValueError('Official metadata image/patient/view mismatch')
        if int(image.split('_')[0])!=pid or not all(x[d] in ['0','1'] for d in diseases):
            raise ValueError('Expert ID or binary label invalid')
        by_patient[pid].append(x)
    population=set(by_patient)
    ledger_rows=rows('ledger');ledger={int(x['patient_id']):x for x in ledger_rows}
    if len(ledger)!=len(ledger_rows):
        raise ValueError('Duplicate patient in prior usage ledger')
    flags=[k for k in ledger_rows[0] if k.startswith('used_in_') or k.startswith('locked_') or k in ['unresolved_possible_use','cross_patient_exact_duplicate']]
    sets={k:{p for p,row in ledger.items() if row[k]=='1'} for k in flags}
    direct={}
    heads=read('head')
    for split,flag in [('public','used_in_public_head'),('train','used_in_private_head'),('eval','used_in_development_loss')]:
        direct[flag]=ids(x for x in heads if x['split']==split)
    direct['used_in_metric_reference']=ids(read('reference'))
    direct['used_in_prior_M1_M2_training']=ids(rows('m1'))|ids(rows('m2'))
    dry=read('dryrun')
    direct['used_in_thesis_dryrun_training']={int(u['patient_id']) for arm in dry['arms'].values() for step in arm for u in step['units']}
    trace=[json.loads(line) for line in PATHS['backbone_trace'].read_text(encoding='utf-8').splitlines() if line.strip()]
    direct['used_in_backbone_training']={int(im.split('_')[0]) for row in trace if row['committed'] for im in row['image_ids']}
    direct['locked_thesis_final']=ids(rows('census'))
    crosscheck={flag:sets[flag]==pids for flag,pids in direct.items()}
    if not all(crosscheck.values()):
        raise ValueError('Direct source differs from ledger: '+str(crosscheck))
    plan=rows('backbone_plan')
    if ids(x for x in plan if x['backbone_role']=='train')!=sets['used_in_backbone_training']:
        raise ValueError('Backbone plan vs actual trace mismatch')
    for role in ['selection','confirmation','reserve']:
        sets['backbone_'+role]=ids(x for x in plan if x['backbone_role']==role)
    sets['padchest_new80']=ids(rows('pad_new'))
    sets['padchest_all309']=ids(rows('pad_all'))
    sets['chexzero_new80']=ids(rows('chex_new'))
    if sets['padchest_new80'] & sets['chexzero_new80']:
        raise ValueError('Expected separate later evaluator cohorts')
    split_rows=rows('split')
    for role in ['target_development','reserved_confirmation']:
        sets['cvpr_'+role]=ids(x for x in split_rows if x['provisional_role']==role)
    known=population & set(ledger)
    used=set().union(*(sets[k] for k in flags if k.startswith('used_in_')),
                     sets['padchest_all309'],sets['chexzero_new80'])
    trained=sets['used_in_backbone_training']|sets['used_in_public_head']|sets['used_in_private_head']
    locked=sets['locked_thesis_final']|sets['locked_cvpr_calibration_test']
    unresolved=sets['unresolved_possible_use']|sets['cross_patient_exact_duplicate']
    clean_known=known-used-unresolved
    sets.update({
        'all_expert_patients':population,
        'matched_local_ledger':known,
        'not_in_local_ledger':population-set(ledger),
        'current_generator_or_head_training_union':trained,
        'any_recorded_result_consumption_union':used,
        'any_locked_thesis_or_cvpr':locked,
        'known_no_recorded_result_consumption':clean_known,
        'clean_but_locked_thesis':clean_known & sets['locked_thesis_final'],
        'clean_but_locked_cvpr':clean_known & sets['locked_cvpr_calibration_test'],
        'clean_existing_cvpr_development':clean_known & sets['cvpr_target_development'] - locked,
        'clean_existing_cvpr_reserved_confirmation':clean_known & sets['cvpr_reserved_confirmation'] - locked,
        'clean_known_nonlocked_nonreserved':clean_known-locked-sets['cvpr_reserved_confirmation'],
    })
    positives={d:{p for p,ims in by_patient.items() if any(x[d]=='1' for x in ims)} for d in diseases}
    def describe(members):
        group=population & members
        selected=[x for p in group for x in by_patient[p]]
        return {
            'patients':len(group),'expert_images':len(selected),
            'positive_patients':{d:len(group & positives[d]) for d in diseases},
            'positive_images':{d:sum(int(x[d]) for x in selected) for d in diseases},
            'patients_positive_E_and_P':len(group & positives['Emphysema'] & positives['Pneumothorax']),
        }
    overlaps={k:{'source_population_patients':len(v),**describe(v)} for k,v in sets.items()}
    matrix=[]
    for p in sorted(population):
        row={'patient_id':p,'expert_images':len(by_patient[p]),'ledger_present':int(p in ledger),
             'original_thesis_role':ledger.get(p,{}).get('original_thesis_role','UNRESOLVED_OUTSIDE_LOCAL_LEDGER'),
             'legacy_audit_grade':ledger.get(p,{}).get('audit_grade','UNKNOWN')}
        row.update({d:int(p in positives[d]) for d in diseases})
        row.update({k:int(p in v) for k,v in sets.items()})
        row['role_reassignment_authorized_by_this_audit']=0
        matrix.append(row)
    with (OUT/'expert_patient_usage_private.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(matrix[0]));writer.writeheader();writer.writerows(matrix)
    source_unchanged={k:sha(p)==hashes[k] for k,p in PATHS.items()}
    if not all(source_unchanged.values()):
        raise ValueError('Input changed during read-only audit')
    result={
        'status':'PATIENT_OVERLAP_AUDIT_COMPLETE_NO_ROLE_REASSIGNMENT',
        'completed_utc':datetime.now(timezone.utc).isoformat(),'seconds':time.perf_counter()-started,
        'expert_source':'publicly_mirrored_derivative_not_official_canonical_copy',
        'all_expert':describe(population), 'overlaps':overlaps,
        'original_thesis_role_counts':dict(Counter(row['original_thesis_role'] for row in matrix)),
        'legacy_audit_grade_counts':dict(Counter(row['legacy_audit_grade'] for row in matrix)),
        'official_image_split_counts':dict(Counter('test' if x['Image Index'] in test else 'train_val' if x['Image Index'] in trainval else 'UNKNOWN' for x in images)),
        'official_metadata_all810_matched':True,'direct_source_ledger_set_equality':crosscheck,
        'all_inputs_unchanged':all(source_unchanged.values()),
        'new_inference':0,'new_generation':0,'new_training':0,'new_DP':0,'patient_pixels_opened':0,
        'existing_roles_modified':False,'final_test_adopted':False,
        'normal_label_inferred_from_all14_negative':False,
        'limitations':['Past ledger covers audited local records, not proof of no external undocumented execution.',
                       'Population and disease counts are descriptive; no model results or performance-based subgroup selection.',
                       'A locked role is an allocation constraint, not evidence of training contamination.'],
        'outputs_sha256':{'expert_patient_usage_private.csv':sha(OUT/'expert_patient_usage_private.csv')},
    }
    dump(OUT/'audit.json',result)
    print(json.dumps({'status':result['status'],'seconds':result['seconds'],
        'role_counts':result['original_thesis_role_counts'],'official_splits':result['official_image_split_counts'],
        'key_groups':{k:{'patients':overlaps[k]['patients'],'images':overlaps[k]['expert_images'],
            'E':overlaps[k]['positive_patients']['Emphysema'],'P':overlaps[k]['positive_patients']['Pneumothorax']}
          for k in ['all_expert_patients','not_in_local_ledger','current_generator_or_head_training_union',
                    'any_recorded_result_consumption_union','clean_but_locked_thesis','clean_but_locked_cvpr',
                    'clean_existing_cvpr_development','clean_existing_cvpr_reserved_confirmation',
                    'clean_known_nonlocked_nonreserved']}},ensure_ascii=False))

if __name__=='__main__':
    main()
