"""Close the failed branch using saved metadata; no model or patient-pixel access."""
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parent
THESIS = ROOT.parents[1]
CODE = THESIS / 'code_working'
RUN = CODE / '_reports/downstream_development_20260917_v1'
MEDICAL = CODE / '_reports/medical_head_20260916_v1'
REPORT = 'TRACK1_CURRENT_HEAD_BRANCH_CLOSURE_20260917.md'
ACTUAL = 'TRACK1_DOWNSTREAM_DEVELOPMENT_RESULTS_20260917.md'
RECORD = ROOT / 'spec_sources/current_head_branch_closure_20260917.json'
VERIFY = ROOT / 'spec_sources/current_head_branch_closure_verification_20260917.json'
STATES = ['research_state.json', 'patient_baseline_spec.json']


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def object_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()


def save_new(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def now():
    return datetime.now(timezone.utc).isoformat()


def collect():
    public_csv = CODE / '_reports/downstream_master_20260917_v1/public_real_images_private.csv'
    with public_csv.open(encoding='utf-8-sig', newline='') as stream:
        public = list(csv.DictReader(stream))
    head = read(MEDICAL / 'images.json')
    public_head = [row for row in head if row['split'] == 'public']
    private_head = [row for row in head if row['split'] == 'train']
    public_ids = {row['image_id'] for row in public_head}
    public_map = {row['image_id']: row for row in public}
    require(public_ids <= set(public_map), 'Head-public must be in downstream public union')
    require(len(public_map) == len(public), 'Duplicate downstream public image')
    backbone = [row for row in public if row['image_id'] not in public_ids]

    def counts(rows):
        positive = [row for row in rows if int(row['weak_P']) == 1]
        return dict(patients=len({int(row['patient_id']) for row in rows}), images=len(rows),
                    P_images=len(positive), P_patients=len({int(row['patient_id']) for row in positive}))

    roles = dict(backbone=counts(backbone), public_head=counts([public_map[i] for i in public_ids]),
                 downstream_public=counts(public))
    require(not ({int(x['patient_id']) for x in backbone} &
                 {int(x['patient_id']) for x in public_head}), 'Public patient roles overlap')
    cohort = read(CODE / '_reports/private_signal_20260917_v1/analysis.json')['cohorts']['train']
    roles['private_head'] = dict(patients=len({int(row['patient_id']) for row in private_head}),
                                images=len(private_head), P_images=cohort['weak_label_image_counts']['Pneumothorax'],
                                P_patients=cohort['weak_label_patient_any_counts']['Pneumothorax'])
    require(roles == dict(backbone=dict(patients=640,images=749,P_images=5,P_patients=5),
                         public_head=dict(patients=32,images=64,P_images=1,P_patients=1),
                         downstream_public=dict(patients=672,images=813,P_images=6,P_patients=6),
                         private_head=dict(patients=80,images=320,P_images=33,P_patients=17)), 'Role counts changed')
    fit = read(MEDICAL / 'fit.json')
    require(fit['patient_counts'] == dict(public=32, train=80, eval=40), 'Fit patient counts')
    objective = read(MEDICAL / 'contract.json')['objective']
    require('r=epsilon-guided_base; X=7.5*phi_conditional' in objective, 'Guided objective binding')
    result = read(RUN / 'result.json')
    verification = read(RUN / 'verification.json')
    require(result['engineering_gate_passed'] is False, 'Failed gate must stay failed')
    require(verification['status'] == 'PASS_INTEGRITY_NOT_EFFICACY', 'Historical integrity status')
    for key in ['expert_pixels', 'reserved_pixels', 'new_DP', 'final_ready']:
        require(not result[key], 'Closed resource flag ' + key)
    means = result['means']
    deltas = {a+'-'+b: {metric: means[a][metric]-means[b][metric] for metric in ['AUROC','AP']}
              for a,b in [('Dreal','R1'),('S2','S1'),('S2','R1'),('S2','R0'),('S2','S0'),('S3','S1'),('S3','R1')]}
    sources = [RUN / name for name in ['contract.json','result.json','verification.json','calibration_choice.json','generation_manifest.json']]
    sources += [MEDICAL / name for name in ['contract.json','fit.json','images.json','patients.json']]
    sources += [public_csv, CODE / '_reports/private_signal_20260917_v1/analysis.json',
                CODE / 'frozen_residual_head/run_medical_head.py', CODE / 'downstream_utility/data_v2.py',
                CODE / 'downstream_utility/train_v2.py']
    sources += [ROOT / name for name in [ACTUAL,'TRACK1_DOWNSTREAM_DEVELOPMENT_PROTOCOL_20260917.md',
                'TRACK1_MEDICAL_HEAD_RESULTS_20260916.md','TRACK1_PRIVATE_SIGNAL_RESULTS_20260917.md',
                'TRACK1_PADCHEST_VALIDATION_RESULTS_20260917.md','TRACK1_CHEXZERO_VALIDATION_RESULTS_20260917.md',
                'NIH_EXPERT_PATIENT_OVERLAP_RESULTS_20260917.md',
                'spec_sources/downstream_development_record_20260917.json',
                'spec_sources/downstream_development_report_verification_20260917.json']]
    require(digest(ROOT / ACTUAL) == 'c82c915d04273a97a37f7e0601e765fa75bab0839a25860eebf0a3baeb4e67a7', 'Original result report changed')
    require(digest(RUN / 'result.json') == 'c6c2bad3a729050910361edd681a12a6cc3cd11c34e8216ef36d661bce5fb038', 'Original result packet changed')
    return roles, objective, deltas, {str(path.relative_to(THESIS)): digest(path) for path in sources}


def publish():
    require(not RECORD.exists() and not VERIFY.exists(), 'Closure already published; use verify only')
    roles, objective, deltas, sources = collect()
    record = dict(schema='current-head-branch-closure/v1', created_utc=now(), report=REPORT,
                  report_sha256=digest(ROOT/REPORT), decision='closed_negative_development_gate',
                  actual_result_report=ACTUAL, closed_scope='E4/full64/current-objective/128-per-method/one-bank/fixed-compute-ResNet18',
                  role_counts=roles, pooled_patient_weights=dict(public=32/112, private=80/112),
                  guided_objective=objective, recomputed_mean_deltas=deltas,
                  head_capacity_isolated_as_cause=False, pooled_dilution_mechanism_established=False,
                  DP_expansion=False, final_ready=False, next_experiment_selected=False,
                  new_model_forwards=0, new_training=0, new_generation=0, new_patient_pixel_reads=0,
                  expert_or_reserved_opened=False, remote_main_checked=False, remote_push=False,
                  source_sha256=sources, preserved_completed_object_sha256={})
    states = {name: read(ROOT/name) for name in STATES}
    for name,d in states.items():
        record['preserved_completed_object_sha256'][name] = {
            key:object_digest(value) for key,value in d.items() if key.startswith('completed_')}
    save_new(RECORD, record)
    for name,d in states.items():
        d['pre_current_head_closure_20260917'] = {key:d.get(key) for key in [
            'next_task','next_task_output','current_planning_report','next_task_plan_report',
            'step2_active_task','research_question_status','next_package_total_work_hours']}
        d['current_decision_report'] = REPORT
        d['current_planning_report'] = REPORT
        d['next_task_plan_report'] = REPORT
        d['next_task'] = 'Current full64 branch is closed. No experiment is scheduled. Any continuation requires a separately justified prospective design with a specific failure hypothesis and fair strong-method comparison; no seed/prompt/scale/bank/gate rescue.'
        d['next_task_output'] = 'A justified research-direction decision if work resumes; existing negative results and locked final/reserved remain preserved.'
        d['next_package_total_work_hours'] = None
        d['step2_active_task'] = 'Current E4/full64 downstream branch closed after negative non-DP development gate'
        d['research_question_status'] = 'current_configuration_closed; head-alone causation and broad private-adaptation impossibility not established'
        d['latest_user_execution_scope_20260917'] = 'Review and close the failed current method; do not expand DP or open expert final; no new model experiment in closure review.'
        d['current_head_branch_closure_20260917'] = dict(report=REPORT,record=str(RECORD.relative_to(ROOT)),
             record_sha256=digest(RECORD),status='closed_negative_development_gate',DP_expansion=False,final_ready=False)
        a=d['active_execution']
        a.update(status='no_running_execution', current_scope='Current full64 branch closed; no new execution contract',
                 latest_design_review=REPORT, latest_planning_status='current_full64_branch_closed',
                 additional_execution_contract_fixed=False, next_execution_not_started=True,
                 next_substep_estimated_work_minutes=None,
                 next_substep_timing_condition='No experiment or duration is scheduled by the closure decision.',
                 last_completed_output=str(RUN))
        a['new_two_track_experiment_status'] = 'nonDP_downstream_complete_private_synthetic_gate_failed_current_branch_closed'
        require(d['current_result_report'] == ACTUAL and d['last_actual_model_result_report'] == ACTUAL, 'Keep actual result pointer')
        (ROOT/name).write_text(json.dumps(d, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print('Closure decision published; no model or pixel execution.')


class Links(HTMLParser):
    def __init__(self):
        super().__init__(); self.links=[]
    def handle_starttag(self, tag, attrs):
        for key,value in attrs:
            if key in ['href','src'] and value:
                self.links.append(value)


def verify():
    require(not VERIFY.exists(), 'Verification already recorded; preserve it')
    record=read(RECORD)
    roles, objective, deltas, sources=collect()
    require(sources == record['source_sha256'], 'Frozen evidence changed during closure')
    require(roles == record['role_counts'] and objective == record['guided_objective'] and deltas == record['recomputed_mean_deltas'], 'Closure factual bindings')
    require(digest(ROOT/REPORT) == record['report_sha256'], 'Closure report changed')
    state_hashes={}
    for name in STATES:
        state=read(ROOT/name)
        for key,expected in record['preserved_completed_object_sha256'][name].items():
            require(object_digest(state[key]) == expected, name+': completed evidence changed '+key)
        require(state['current_decision_report'] == REPORT, 'Decision pointer')
        require(state['current_result_report'] == ACTUAL and state['last_efficacy_result_report'] == ACTUAL, 'Actual evidence pointer')
        require(state['current_head_branch_closure_20260917']['record_sha256'] == digest(RECORD), 'Decision hash')
        require(state['active_execution']['status'] == 'no_running_execution', 'Execution status')
        require(state['active_execution']['additional_execution_contract_fixed'] is False, 'No new contract')
        state_hashes[name]=digest(ROOT/name)
    first,second=[read(ROOT/name) for name in STATES]
    for key in ['current_decision_report','next_task','next_task_output','step2_active_task','research_question_status','current_head_branch_closure_20260917']:
        require(first[key] == second[key], 'State synchronization '+key)
    checked=0
    pages=['track1_current_head_branch_closure.html','index.html','research_framework.html']
    for name in pages:
        parser=Links(); parser.feed((ROOT/name).read_text(encoding='utf-8'))
        for link in parser.links:
            parts=urlsplit(link)
            if parts.scheme or parts.netloc or not parts.path:
                continue
            target=(ROOT/unquote(parts.path)).resolve()
            if target == VERIFY:
                continue  # This immutable verification is written immediately below.
            require(target.exists(), 'Missing local link '+link)
            checked+=1
    build=read(ROOT/'build_summary.json')
    require(build['records']==79 and build['local_pdfs']==75, 'Literature inventory preserved')
    save_new(VERIFY, dict(status='PASS_DOCUMENT_METADATA_AND_STATE_BINDING',created_utc=now(),
         scope='Saved metadata/means/contract/source review; no raw-image decode, model replay or new efficacy evidence',
         immutable_source_files=len(sources),source_sha256=sources,
         role_counts=roles,recomputed_mean_deltas=deltas,local_links_checked=checked,broken_local_links=0,
         state_sha256=state_hashes,report_sha256=digest(ROOT/REPORT),record_sha256=digest(RECORD),
         new_model_forwards=0,new_training=0,new_generation=0,new_patient_pixel_reads=0,
         old_completed_objects_preserved=True,expert_or_reserved_opened=False,remote_main_checked=False,remote_push=False))
    print(json.dumps(dict(status='PASS',immutable_source_files=len(sources),local_links_checked=checked,
                         new_model_or_pixel_work=0,branch='closed'),indent=2))


if __name__ == '__main__':
    require(len(sys.argv)==2 and sys.argv[1] in ['publish','verify'], 'Use publish or verify')
    globals()[sys.argv[1]]()
