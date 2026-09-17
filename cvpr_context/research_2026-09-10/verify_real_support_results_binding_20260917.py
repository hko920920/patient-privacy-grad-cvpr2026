"""Report/public/local/state binding; no new images, models or optimizer work."""
from pathlib import Path
import hashlib,json,re
from html.parser import HTMLParser
from urllib.parse import unquote,urlsplit
import numpy as np
ROOT=Path(__file__).resolve().parent
THESIS=ROOT.parents[1];OUT=THESIS/'code_working/_reports/real_support_20260917_v1'
REPORT='TRACK1_REAL_SUPPORT_DIAGNOSTIC_RESULTS_20260917.md'
SPEC=ROOT/'spec_sources'
COUNT=0
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def need(value,message):
    global COUNT
    COUNT+=1
    if not bool(value):raise AssertionError(message)
class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[]
    def handle_starttag(self,tag,attrs):
        for k,v in attrs:
            if k in ('href','src') and v:self.links.append(v)
def main():
    x=read(OUT/'result.json');v=read(OUT/'verification.json');c=read(OUT/'contract.json');before=read(OUT/'state_before.json')
    need(sha(OUT/'result.json')==v['result_sha256'] and sha(OUT/'contract.json')==v['contract_sha256'],'Local result/contract integrity')
    for path,h in c['source_sha256'].items():need(sha(Path(path))==h,'Frozen input/code unchanged')
    for path,h in v['files'].items():need(sha(OUT/path)==h,'Completed execution unchanged')
    for a,b in [('result.json','real_support_diagnostic_record_20260917.json'),('verification.json','real_support_diagnostic_execution_verification_20260917.json'),('probe_verification.json','real_support_diagnostic_probe_verification_20260917.json')]:
        need(sha(OUT/a)==sha(SPEC/b),'Exact public artifact copy')
    need(x['main_updates']==1200 and x['probe_updates']==4,'Update counts')
    need(x['new_generation']==x['new_DP']==x['expert_pixels']==x['reserved_pixels']==0 and not x['final_ready'],'Closed boundary counts')
    need(v['independent_bootstrap_metric_pairs']==18000 and v['raw_files_redecoded']==10957,'Independent verification scope')
    need(v['max_metric_difference']<1e-12 and v['max_bootstrap_difference']<1e-12,'Arithmetic tolerances')
    train_time=[];train_forward=dev_forward=witness_forward=0
    for s in (11,23,37):
        t=read(OUT/'runs'/f'Rwide_{s}'/'trace.json');train_time.append(t['seconds']);need(len(t['trace'])==400,'400-step checkpoint')
        for phase in ('training','development'):
            with np.load(OUT/'evaluation'/f'Rwide_{s}_{phase}.npz',allow_pickle=False) as p:
                if phase=='training':train_forward+=len(p['labels'])
                else:dev_forward+=len(p['labels'])
        with np.load(OUT/'evaluation'/f'R1_{s}_witness.npz',allow_pickle=False) as p:witness_forward+=len(p['labels'])
    need((train_forward,dev_forward,witness_forward)==(10374,15141,192),'Actual prediction budgets')
    need(abs(sum(train_time)-x['training_seconds'])<1e-9,'Actual time accounting')
    preserved=0
    for name,old in before.items():
        a=read(ROOT/name);record=a['completed_real_support_diagnostic_20260917']
        for k,val in old.items():
            if k.startswith('completed_'):need(a[k]==val,'Historical completed object preserved');preserved+=1
        for k,val in old['active_execution'].items():
            if k.startswith('completed_'):need(a['active_execution'][k]==val,'Historical completed execution object preserved');preserved+=1
        for k in ('current_result_report','last_actual_model_result_report','current_decision_report','last_real_generalization_result_report'):need(a[k]==REPORT,'Current pointer')
        need(a['last_efficacy_result_report']==old['last_efficacy_result_report']=='TRACK1_LORA_TRANSFER_COMPARISON_RESULTS_20260917.md','Private generation result not superseded')
        need(a['current_step']==2 and a['active_execution']['status']=='no_running_execution','Stage/running state')
        need(record['means']==x['means'] and record['comparisons']==x['comparisons'],'State numbers')
        need(record['improvement_candidate']==x['improvement_candidate'] and not record['private_synthetic_efficacy_established'],'State interpretation')
        need(record['result_sha256']==sha(OUT/'result.json') and record['verification_sha256']==sha(OUT/'verification.json'),'State hashes')
        need(not record['DP_executed'] and not record['expert_opened'] and not record['reserved_opened'] and not record['final_ready'],'State gates')
    md=(ROOT/REPORT).read_text(encoding='utf-8');html=(ROOT/'track1_real_support_diagnostic_results.html').read_text(encoding='utf-8')
    for arm in ('R0','R1','Rwide'):
        for m in ('AUROC','AP'):need(f'{x["means"][arm][m]:.6f}' in md,'Report mean metric')
    for b in ('R1','R0'):
        for m in ('AUROC','AP'):
            val=x['comparisons']['Rwide-'+b][m];need(f'{val["mean_delta"]:+.6f}' in md,'Report effect')
            for z in val['percentile95']:need(f'{z:+.6f}' in md,'Report interval')
    for term in ('비DP 일반화 진단','공개 baseline','original private_train','독립 confirmation이 아니다','final_ready=false','원인 하나를 확정하지 않는다','실제 한 번 이상 학습한'):
        need(term in md,'Claim limitation '+term)
    need('track1_real_support_diagnostic_results.html' in (ROOT/'index.html').read_text(encoding='utf-8'),'Index results link')
    links=0
    for name in ('track1_real_support_diagnostic_results.html','index.html'):
        p=ROOT/name;parser=Links();parser.feed(p.read_text(encoding='utf-8'))
        for link in parser.links:
            q=urlsplit(link)
            if q.scheme or q.netloc or not q.path:continue
            target=(p.parent/unquote(q.path)).resolve();need(target.exists(),'Broken local link: '+link);links+=1
    report=dict(status='PASS_REAL_SUPPORT_RESULT_REPORT_STATE_BINDING',checks=COUNT,local_links=links,prior_completed_objects_preserved=preserved,
      report_sha256=sha(ROOT/REPORT),html_sha256=sha(ROOT/'track1_real_support_diagnostic_results.html'),result_sha256=sha(OUT/'result.json'),
      runtime_verification_sha256=sha(OUT/'verification.json'),public_record_sha256=sha(SPEC/'real_support_diagnostic_record_20260917.json'),
      improvement_candidate=x['improvement_candidate'],new_model_runs_in_this_verifier=0,final_ready=False,
      files={str(p.relative_to(THESIS)):sha(p) for p in [ROOT/REPORT,ROOT/'track1_real_support_diagnostic_results.html',ROOT/'research_state.json',ROOT/'patient_baseline_spec.json',ROOT/'RESEARCH_FRAMEWORK.md',THESIS/'AGENTS.md',THESIS/'CURRENT_STATUS.md',THESIS/'WORKLOG.md']})
    with (SPEC/'real_support_report_verification_20260917.json').open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in report.items() if k!='files'},ensure_ascii=False))
if __name__=='__main__':main()
