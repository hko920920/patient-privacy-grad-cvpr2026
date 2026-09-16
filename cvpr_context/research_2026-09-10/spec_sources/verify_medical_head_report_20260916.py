"""Check final report/state against the frozen medical-head execution."""
import hashlib,json,sys
from pathlib import Path
from datetime import datetime,timezone
from urllib.parse import urlsplit,unquote
import numpy as np
from PIL import Image
from bs4 import BeautifulSoup

R=Path(__file__).resolve().parent.parent
P=R.parent.parent/'code_working/_reports/medical_head_20260916_v1'
sys.path.insert(0,str(R))
from verify_review import static_check

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def require(ok,msg):
    if not ok:raise ValueError(msg)

def main():
    static=static_check()
    source_expected={}
    for name in ['contract.json','generation_contract.json','evaluation_contract.json']:
        for p,h in read(P/name)['source_sha256'].items():
            require(p not in source_expected or source_expected[p]==h,'Conflicting frozen source '+p)
            source_expected[p]=h
    for p,h in source_expected.items():require(sha(p)==h,'Frozen source changed '+p)
    plan=read(R/'spec_sources/public_medical_backbone_plan_20260916_v1/plan_lock.json')
    for p,h in plan['files'].items():require(sha(R/p)==h,'Old plan changed '+p)
    old=P.parent/'public_medical_backbone_20260916_v1/adoption_status.json'
    require(sha(old)=='d08ecc95f0a1150c6eaaaacdd53b8406518e64d1027b3495da32b88d447cc4bd','Original failed adoption changed')
    s=read(R/'research_state.json');t=read(R/'patient_baseline_spec.json')
    fields=['current_step','next_task','next_task_output','step2_open_items','current_result_report','current_planning_report','active_execution']
    for key in fields:require(s[key]==t[key],'State disagreement '+key)
    require(s['current_step']==2 and s['active_execution']['status']=='no_running_execution','Stage/execution')
    require(s['current_result_report']=='TRACK1_MEDICAL_HEAD_RESULTS_20260916.md','Actual result pointer')
    decision=read(P/'research_decision.json')
    require(s['active_execution']['medical_backbone_head']==decision,'Decision/state agreement')
    require(not decision['DP'] and not decision['private_incremental_utility_demonstrated'],'No invented DP/utility claim')
    for f,h in decision['source_sha256'].items():require(sha(P/f)==h,'Decision artifact changed '+f)
    ev=read(P/'evaluation.json');fit=read(P/'fit.json');gen=read(P/'generation.json')
    for name in ['head_verification.json','generation_verification.json','evaluation_verification.json']:
        v=read(P/name);require(v['complete'] and v['status'].startswith('PASS'),'Execution verification '+name)
    require(read(P/'evaluation_verification.json')['evaluation_sha256']==sha(P/'evaluation.json'),'Metric report binding')
    require(read(P/'generation_verification.json')['manifest_sha256']==sha(P/'generation_manifest.json'),'Generation binding')
    review=read(P/'visual_review_root.json')
    require(review['images_reviewed']==192 and review['method_labels_hidden'] and review['reviewer_count']==1,'Actual review scope')
    for f,h in review['reviewed_grid_sha256'].items():require(sha(P/f)==h,'Blind sheet changed')
    manifest=read(P/'generation_manifest.json');index={(x['seed_id'],x['prompt_id'],x['method']):x for x in manifest}
    c=read(P/'contract.json');pres=read(P/'presentation.json');count=0
    require(len(index)==192 and len(pres['exported'])==8,'All output cells')
    for item in pres['exported']:
        sheet=R/item['path'];require(sha(sheet)==item['sha256'],'Presentation hash')
        with Image.open(sheet) as im:a=np.array(im)
        for row,(sid,pid) in enumerate(item['cells']):
            for col,method in enumerate(c['generation_methods']):
                entry=index[(sid,pid,method)];p=P/entry['image_path']
                require(sha(p)==entry['image_sha256'],'Original image hash')
                with Image.open(p) as im:b=np.array(im)
                x=col*256;y=32+row*280+24
                require(np.array_equal(a[y:y+256,x:x+256],b),'Modified presentation pixels')
                count+=1
    require(count==192,'All192 actually presented')
    pages=['track1_medical_head_protocol.html','track1_medical_head_evaluation_note.html','track1_medical_head_results.html']
    links=0
    for name in pages:
        body=(R/name).read_text(encoding='utf-8');require('\ufffd' not in body,'Encoding '+name)
        soup=BeautifulSoup(body,'html.parser')
        for node in soup.find_all(['a','img']):
            href=node.get('href',node.get('src',''));u=urlsplit(href)
            if u.scheme or u.netloc or not u.path:continue
            require((R/unquote(u.path)).resolve().exists(),'Missing report link '+href);links+=1
    report=(R/'TRACK1_MEDICAL_HEAD_RESULTS_20260916.md').read_text(encoding='utf-8')
    for method in ['backbone','public','pooled']:
        for k in ['KID_condition_mean','KID_mixed_descriptive']:
            require(format(ev['methods'][method][k],'.9f') in report,'Reported metric '+method+k)
        require(format(fit['development_MSE'][method],'.10f') in report,'Reported MSE '+method)
    require('미통과' in (R/'index.html').read_text(encoding='utf-8'),'Index decision')
    totals=dict(unet_calls=read(P/'extraction.json')['unet_calls']+gen['unet_calls'],
                unet_examples=read(P/'extraction.json')['unet_examples']+gen['unet_examples'])
    require(totals==dict(unet_calls=10215,unet_examples=20430),'Cost totals')
    result=dict(status='PASS_MEDICAL_HEAD_REPORT_BINDINGS',created_utc=datetime.now(timezone.utc).isoformat(),
        existing_html_check=static,new_page_local_links_and_images=links,new_broken_links=0,
        frozen_source_files=len(source_expected),prior_plan_files=len(plan['files']),old_failed_adoption_unchanged=True,
        state_fields=fields,original_image_pixels_verified=count,reviewer_count=1,clinical_claim=False,DP=False,
        totals=totals,report_sha256=sha(R/'TRACK1_MEDICAL_HEAD_RESULTS_20260916.md'),
        decision_sha256=sha(P/'research_decision.json'),verifier_sha256=sha(__file__))
    (R/'spec_sources/medical_head_report_verification_20260916.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()

