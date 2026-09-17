"""Verify the shareable bundle only; this does not independently inspect raw pixels/models."""
from pathlib import Path
import json,hashlib,math
R=Path(__file__).resolve().parent;S=R/'spec_sources';COUNT=0
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def need(v,m):
    global COUNT
    COUNT+=1
    if not v:raise AssertionError(m)
def main():
    report=R/'TRACK1_REAL_SUPPORT_DIAGNOSTIC_RESULTS_20260917.md'
    x=read(S/'real_support_diagnostic_record_20260917.json')
    v=read(S/'real_support_diagnostic_execution_verification_20260917.json')
    p=read(S/'real_support_diagnostic_probe_verification_20260917.json')
    b=read(S/'real_support_report_verification_20260917.json')
    plan=read(S/'real_support_diagnostic_plan_20260917.json')
    need(sha(S/'real_support_diagnostic_record_20260917.json')==v['result_sha256']==b['result_sha256'],'Result copy SHA')
    need(sha(S/'real_support_diagnostic_execution_verification_20260917.json')==b['runtime_verification_sha256'],'Runtime verification binding')
    need(sha(report)==b['report_sha256'],'Report binding')
    need(sha(R/'track1_real_support_diagnostic_results.html')==b['html_sha256'],'HTML binding')
    need(plan['counts']['expanded']['patients']==2699 and plan['counts']['expanded']['images']==5910 and plan['counts']['expanded']['positive_patients']==120,'Prospective pool')
    need(x['main_updates']==1200 and x['probe_updates']==4 and p['updates']==4,'Fixed execution budget')
    need(x['new_generation']==x['new_DP']==x['expert_pixels']==x['reserved_pixels']==0,'Preserved boundaries')
    need(v['independent_bootstrap_metric_pairs']==18000 and v['raw_files_redecoded']==10957,'Recorded independent verification scope')
    need(p['exact_input'] and p['exact_state'],'Recorded runtime wiring')
    for arm in ('R0','R1','Rwide'):
        need([z['seed'] for z in x['scores'][arm]]==[11,23,37],'All seeds')
        for m in ('AUROC','AP'):
            need(abs(sum(z[m] for z in x['scores'][arm])/3-x['means'][arm][m])<1e-12,'Mean seed metric')
            need(f'{x["means"][arm][m]:.6f}' in report.read_text(encoding='utf-8'),'Reported number')
    for base in ('R1','R0'):
        for m in ('AUROC','AP'):
            z=x['comparisons']['Rwide-'+base][m]
            d=[x['scores']['Rwide'][i][m]-x['scores'][base][i][m] for i in range(3)]
            need(max(abs(a-c) for a,c in zip(d,z['seed_deltas']))<1e-12,'Per-seed differences')
            need(abs(sum(d)/3-z['mean_delta'])<1e-12 and sum(a>0 for a in d)==z['positive_seeds'],'Mean and direction')
            need(len(z['percentile95'])==2 and z['percentile95'][0]<=z['percentile95'][1],'Reported interval bounds')
    z=x['comparisons']['Rwide-R1'];gate=z['AUROC']['mean_delta']>=.01 and z['AUROC']['positive_seeds']>=2 and z['AP']['mean_delta']>=0
    need(gate==x['improvement_candidate']==v['improvement_candidate']==b['improvement_candidate'],'Frozen improvement rule')
    for s in ('11','23','37'):
        a=x['actual_exposure'][s];e=plan['planned_exposure'][s]
        need(a['actual_unique_images']==e['planned_unique_images'] and a['actual_unique_patients']==e['planned_unique_patients'],'Exposure coverage')
        need(a['sources']==e['sources'] and a['presentations']==12800,'Planned/actual source exposures')
    for name in ('research_state.json','patient_baseline_spec.json'):
        state=read(R/name);record=state['completed_real_support_diagnostic_20260917']
        need(state['current_result_report']==report.name and state['last_actual_model_result_report']==report.name,'Latest actual result')
        need(state['last_efficacy_result_report']=='TRACK1_LORA_TRANSFER_COMPARISON_RESULTS_20260917.md','Prior private synthetic efficacy remains failed')
        need(record['means']==x['means'] and record['comparisons']==x['comparisons'],'State numbers')
        need(not record['private_synthetic_efficacy_established'] and not record['final_ready'],'Claim boundary')
        need(record['role_overlay_status']=='consumption_recorded_not_validation','Donor consumption recorded')
    print(json.dumps(dict(status='PASS_PUBLIC_REAL_SUPPORT_RESULT_BUNDLE',checks=COUNT,improvement_candidate=gate,
       scope='Document and recorded numeric/hash consistency only; no raw image, checkpoint inference or GPU re-execution',
       means=x['means'],new_model_execution=0),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
