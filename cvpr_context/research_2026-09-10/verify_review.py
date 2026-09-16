"""Check review artifacts and real offline Chrome filtering without external writes."""
import json,subprocess,time,hashlib
from pathlib import Path
from urllib.parse import urlsplit,unquote
from collections import Counter
import requests,websocket,fitz,markdown
from bs4 import BeautifulSoup

ROOT=Path(__file__).resolve().parent

def static_check():
    rows=json.loads((ROOT/'review_ledger.json').read_text(encoding='utf-8'))
    assert len(rows)==79 and len({r['id'] for r in rows})==79
    assert sum(r['legacy35'] for r in rows)==35
    assert sum(r['new'] for r in rows)==12
    assert Counter(r['review_status'] for r in rows)==Counter(selected=75,partial=3,abstract=1)
    assert len(list((ROOT/'reviews').glob('*.md')))==79
    paths=list((ROOT/'reviews').glob('*.md'))+[ROOT/'index.html',ROOT/'report.html',ROOT/'baseline_preparation.html',ROOT/'rationale.html',ROOT/'purpose.html',ROOT/'design.html',ROOT/'thesis_alignment.html',ROOT/'rationale_sources'/'R01.html',ROOT/'rationale_sources'/'R02.html']
    paths.append(ROOT/'runtime_review.html')
    paths.append(ROOT/'auditor_framing_review.html')
    paths.append(ROOT/'baseline_scope_review.html')
    paths.append(ROOT/'u_execution_plan.html')
    paths.append(ROOT/'u_execution_status.html')
    paths.append(ROOT/'u_eight_patient_analysis.html')
    paths.append(ROOT/'u_pilot_review.html')
    paths.append(ROOT/'u_verification_results.html')
    paths.append(ROOT/'u_baseline_screen.html')
    paths.append(ROOT/'method_premise_audit.html')
    paths.append(ROOT/'pfami_comparison_protocol.html')
    paths.append(ROOT/'pfami_comparison_results.html')
    paths.append(ROOT/'paired_score_audit.html')
    paths.append(ROOT/'research_framework.html')
    paths.append(ROOT/'patient_baseline_spec.html')
    paths.append(ROOT/'cdi_kernel_results.html')
    paths.append(ROOT/'cdi_u_comparison_results.html')
    paths.append(ROOT/'cdi_eu_comparison_results.html')
    paths.append(ROOT/'prompt_condition_results.html')
    paths.append(ROOT/'mofit_medical_results.html')
    paths.append(ROOT/'stage2_measurement_audit.html')
    paths.append(ROOT/'stage2_cross_patient_intervention.html')
    paths.append(ROOT/'stage2_design_gap_reset.html')
    paths.append(ROOT/'paper_level_redesign.html')
    paths.append(ROOT/'protection_observation_claim.html')
    paths.append(ROOT/'clipping_observation_local_results.html')
    paths.append(ROOT/'patient_gap_work_inventory.html')
    paths.append(ROOT/'two_track_paper_designs.html')
    paths.append(ROOT/'two_track_deeper_check.html')
    paths.append(ROOT/'two_track_operation_redesign.html')
    paths.append(ROOT/'track1_capacity_protocol.html')
    paths.append(ROOT/'track1_capacity_results.html')
    paths.append(ROOT/'track1_patient_dp_comparison_plan.html')
    paths.append(ROOT/'track1_patient_dp_protocol.html')
    paths.append(ROOT/'track1_patient_dp_results.html')
    paths.append(ROOT/'realistic_research_plan.html')
    paths.extend(ROOT/'spec_sources'/name for name in ['public_assisted_deep_priors_20260916.md','research_gate_review_20260916.md','generation_evaluator_readiness_20260916.md'])
    paths.extend(ROOT/'spec_sources'/name for name in ['track1_problem_framing_review_20260916.md','track1_replan_prior_review_20260916.md','track1_replan_feasibility_20260916.md','track1_replan_independent_20260916.md'])
    paths.append(ROOT/'spec_sources'/'track1_public_reference_alternatives_20260916.md')
    paths.extend(ROOT/'spec_sources'/name for name in ['track1_operation_redesign_20260916.md','track2_operation_redesign_20260916.md','operation_redesign_independent_20260916.md'])
    paths.extend(ROOT/'spec_sources'/folder/'해석.md' for folder in ['track1_cpu_allocation_20260916','track2_cpu_audit_20260916'])
    paths.append(ROOT/'spec_sources'/'two_track_deeper_novelty_20260916.md')
    paths.extend(ROOT/'spec_sources'/name for name in ['protection_unit_eu_novelty_audit_20260915.md','uls_els_clipping_mechanism_review_20260915.md','attack_gradient_connection_20260915.md'])
    bad=[];links=0;pagechecks=0
    for p in paths:
        body=p.read_text(encoding='utf-8')
        assert '\ufffd' not in body,p.name
        soup=BeautifulSoup(markdown.markdown(body) if p.suffix=='.md' else body,'html.parser')
        for a in soup.find_all('a',href=True):
            u=urlsplit(a['href'])
            if u.scheme or u.netloc or not u.path:continue
            dest=(p.parent/unquote(u.path)).resolve();links+=1
            if not dest.exists():bad.append((str(p.relative_to(ROOT)),a['href']))
            elif dest.suffix.lower()=='.pdf' and u.fragment.startswith('page='):
                page=int(u.fragment[5:]);pagechecks+=1
                with fitz.open(dest) as doc:
                    if not 1<=page<=len(doc):bad.append((str(p),a['href']))
    assert not bad,bad
    for r in rows:
        assert hashlib.sha256((ROOT/r['review_file']).read_bytes()).hexdigest()==r['memo_sha256']
        if r['local_pdf']:
            assert (ROOT/r['local_pdf']).read_bytes().lstrip().startswith(b'%PDF')
    return dict(rows=79,checked_local_links=links,checked_pdf_page_anchors=pagechecks,broken_links=bad)

def browser_check():
    chrome=Path('C:/Program Files/Google/Chrome/Application/chrome.exe')
    profile=ROOT/'browser_check_profile';profile.mkdir(exist_ok=True)
    portfile=profile/'DevToolsActivePort'
    portfile.unlink(missing_ok=True)
    log=(ROOT/'browser_check.log').open('w',encoding='utf-8')
    proc=subprocess.Popen([str(chrome),'--headless=new','--disable-gpu','--no-first-run',
        '--no-default-browser-check','--remote-allow-origins=*','--remote-debugging-port=0',
        '--user-data-dir='+str(profile),'about:blank'],stdout=log,stderr=log,
        creationflags=subprocess.CREATE_NO_WINDOW)
    sock=None;events=[];seq=0
    try:
        for _ in range(100):
            if portfile.exists():break
            time.sleep(.1)
        port=int(portfile.read_text().splitlines()[0])
        targets=requests.get(f'http://127.0.0.1:{port}/json',timeout=5).json()
        page=next(x for x in targets if x['type']=='page')
        sock=websocket.create_connection(page['webSocketDebuggerUrl'],timeout=15)
        def call(method,params=None):
            nonlocal seq
            seq+=1;myseq=seq;sock.send(json.dumps(dict(id=myseq,method=method,params=params or {})))
            while True:
                msg=json.loads(sock.recv())
                if msg.get('method')=='Runtime.exceptionThrown':events.append(msg)
                if msg.get('id')==myseq:
                    if 'error' in msg:raise RuntimeError(msg)
                    return msg['result']
        def js(code):
            out=call('Runtime.evaluate',dict(expression=code,returnByValue=True,awaitPromise=True))
            if 'exceptionDetails' in out:raise RuntimeError(out)
            return out['result'].get('value')
        def navigate(path):
            call('Page.navigate',dict(url=path.resolve().as_uri()))
            for _ in range(60):
                if js("document.readyState==='complete'"):break
                time.sleep(.1)
        call('Page.enable');call('Runtime.enable')
        navigate(ROOT/'index.html')
        result=js('''(()=>{const checks=[];const assert=(ok,msg)=>{if(!ok)throw Error(msg)};const test=(field,value,expected)=>{reviewReset();document.getElementById(field).value=value;const n=reviewFilter();assert(n===expected,field+':'+value+' '+n);assert(document.querySelectorAll('.card:not([hidden])').length===n,'DOM differs');checks.push({field,value,count:n})};for(const [v,n] of Object.entries({all:79,legacy:67,legacy35:35,new:12,direct:26}))test('scope',v,n);for(const [v,n] of Object.entries({attack:13,training:13,unit:7,medical:20,federated:4,alternative:16,thesis:3,background:3}))test('role',v,n);for(const [v,n] of Object.entries({selected:75,partial:3,abstract:1,replay:2}))test('status',v,n);reviewReset();document.getElementById('scope').value='new';document.getElementById('role').value='attack';assert(reviewFilter()===7,'combined');test('search','NO_SUCH_PAPER_12345',0);assert(!document.getElementById('empty').hidden,'empty state');reviewReset();document.getElementById('search').value='mOfIt';assert(reviewFilter()>0&&!document.getElementById('X12').hidden,'case-insensitive search');document.getElementById('expand').click();assert([...document.querySelectorAll('.card:not([hidden])')].every(x=>x.open),'expand');document.getElementById('collapse').click();assert([...document.querySelectorAll('.card')].every(x=>!x.open),'collapse');reviewReset();return {checks,combined:7,search:true,expand:true,collapse:true,reset:reviewFilter()}})()''')
        old=ROOT.parent/'reading_list_2026-09-09'
        oldchecks=[]
        for filename,total in [('index.html',71),('cvpr_patient_dp.html',35)]:
            navigate(old/filename)
            check=js('''(()=>{const s=document.getElementById('scope');return {cards:document.querySelectorAll('article').length,banner:!!document.querySelector('a[href="../research_2026-09-10/index.html"]'),options:[...s.options].map(o=>{s.value=o.value;s.dispatchEvent(new Event('change'));return {value:o.value,count:document.querySelectorAll('article:not([hidden])').length}})}})()''')
            assert check['cards']==total and check['banner'],check
            oldchecks.append(dict(file=filename,**check))
        navigate(ROOT/'rationale.html')
        rationale=js('''(()=>({paragraphs:document.querySelectorAll('p').length,tables:document.querySelectorAll('table').length,footnotes:document.querySelectorAll('.footnote li').length,rawFootnotes:document.body.innerText.includes('[^1]'),additionalSources:document.querySelectorAll('a[href^="rationale_sources/R"]').length}))()''')
        assert rationale['paragraphs']>40 and rationale['tables']>=5 and rationale['footnotes']==13 and not rationale['rawFootnotes'] and rationale['additionalSources']==2,rationale
        assert not events,events
        return dict(new=result,legacy=oldchecks,rationale=rationale,javascript_exceptions=0)
    finally:
        if sock:sock.close()
        proc.terminate()
        try:proc.wait(timeout=8)
        except subprocess.TimeoutExpired:proc.kill()
        log.close()

if __name__=='__main__':
    report=dict(static=static_check(),browser=browser_check())
    (ROOT/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))
