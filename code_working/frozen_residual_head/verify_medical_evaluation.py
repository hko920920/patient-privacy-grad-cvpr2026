"""Recount all rows and recompute metrics from saved encoder outputs."""
import argparse,time
from pathlib import Path
from collections import Counter
import numpy as np
from public_medical_backbone.verify_pilot import Check,sha,read,save

ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'_reports/medical_head_20260916_v1'

def kid(real,gen):
    # Independent ordered-pair summation; no production metric import.
    def k(x,y):return (np.einsum('id,jd->ij',x,y,optimize=False)/x.shape[1]+1)**3
    xx,yy,xy=k(real,real),k(gen,gen),k(real,gen)
    return float(xx[~np.eye(len(real),dtype=bool)].mean()+yy[~np.eye(len(gen),dtype=bool)].mean()-2*xy.mean())

def prdc(real,gen):
    rr=np.linalg.norm(real[:,None,:]-real[None,:,:],axis=2);gg=np.linalg.norm(gen[:,None,:]-gen[None,:,:],axis=2)
    rg=np.linalg.norm(real[:,None,:]-gen[None,:,:],axis=2)
    np.fill_diagonal(rr,np.inf);np.fill_diagonal(gg,np.inf)
    ar=np.sort(rr,axis=1)[:,4];ag=np.sort(gg,axis=1)[:,4]
    return dict(precision=float(np.mean((rg<ar[:,None]).any(0))),recall=float(np.mean((rg<ag[None,:]).any(1))),
        density=float(np.mean((rg<ar[:,None]).sum(0)/5)),coverage=float(np.mean(rg.min(1)<ar)))

def verify(out):
    start=time.perf_counter();ch=Check();c=read(out/'contract.json');ec=read(out/'evaluation_contract.json');enc=read(out/'encoding.json');result=read(out/'evaluation.json')
    for p,h in ec['source_sha256'].items():ch.require(sha(p)==h,'evaluator source '+p)
    for fn,key in [('evaluation_features.npz','features_sha256'),('encoding.json','encoding_sha256'),('per_image_scores.json','per_image_sha256'),('evaluation_contract.json','evaluation_contract_sha256')]:ch.require(sha(out/fn)==result[key],'metric binding '+fn)
    ch.require(enc['rows_sha256']==sha(out/'evaluation_rows.json') and enc['generation_manifest_sha256']==sha(out/'generation_manifest.json'),'encoded row bindings')
    rows=read(out/'evaluation_rows.json');ref=read(out/'reference.json');manifest=read(out/'generation_manifest.json')
    ch.require(len(rows)==232 and len({r['row_id'] for r in rows})==232,'all232 unique rows')
    lookup={r['row_id']:i for i,r in enumerate(rows)}
    for r in ref:
        e=rows[lookup['real|'+r['image_id']]];ch.require(e['kind']=='real' and all(e[k]==r[k] for k in r),'real metadata')
    for r in manifest:
        e=rows[lookup[r['image_id']]]
        ch.require(e['kind']=='generated' and e['method']==r['method'] and e['prompt_id']==r['prompt_id'] and e['seed_id']==r['seed_id'] and e['sha256']==r['image_sha256'],'generated metadata')
    for r in rows:ch.require(sha(r['path'])==r['sha256'],'PNG hash')
    with np.load(out/'evaluation_features.npz') as f:
        rad=f['rad'].astype(np.float64);bio=f['bio'].astype(np.float64);text=f['text'].astype(np.float64)
        ch.require(rad.shape==(232,768) and bio.shape==(232,128) and text.shape==(4,128),'feature shapes')
        ch.require(np.isfinite(rad).all() and np.isfinite(bio).all() and np.isfinite(text).all(),'finite features')
        ch.close(f['rad_replay'],f['rad'][:8],'RAD fixed-input replay',atol=1e-5,rtol=0)
        ch.close(f['bio_replay'],f['bio'][:8],'BioViL fixed-input replay',atol=1e-5,rtol=0)
    ch.close(np.linalg.norm(bio,axis=1),np.ones(232),'image embedding norms',atol=1e-5,rtol=0)
    pids=list(c['prompts']);ch.require(enc['prompt_ids']==pids,'text order')
    real=[i for i,r in enumerate(rows) if r['kind']=='real'];specific=ec['specific_prompts'];scores=read(out/'per_image_scores.json');scoremap={r['image_id']:r for r in scores}
    ch.require(len(scores)==len(scoremap)==192,'all per-image scores')
    matrix=np.einsum('nd,pd->np',bio,text,optimize=False)
    for i,r in enumerate(rows):
        if r['kind']=='real':continue
        p=r['prompt_id'];s=scoremap[r['row_id']];ch.require(all(s[k]==r[k] for k in ['method','prompt_id','seed_id']),'score metadata')
        ch.close(np.array(s['matched_cosine']),np.array(matrix[i,pids.index(p)]),'matched cosine',atol=1e-12,rtol=1e-12)
        if p in specific:
            margin=matrix[i,pids.index(p)]-np.mean([matrix[i,pids.index(q)] for q in specific if q!=p])
            ch.close(np.array(s['specific_margin']),np.array(margin),'specific margin',atol=1e-12,rtol=1e-12)
        else:ch.require(s['specific_margin'] is None,'generic not a disease margin')
    for method in ec['methods']:
        allidx=[i for i,r in enumerate(rows) if r.get('method')==method];gi=[i for i in allidx if rows[i]['prompt_id'] in ec['matched_prompts']]
        ch.require(len(allidx)==64 and len(gi)==32,'generated metric counts')
        m=result['methods'][method];ch.close(np.array(m['KID_mixed_descriptive']),np.array(kid(rad[real],rad[gi])),'mixed KID '+method,atol=2e-10,rtol=2e-10)
        cond=[]
        for prompt,condition in zip(ec['matched_prompts'],ec['reference_conditions']):
            ri=[i for i in real if rows[i]['matched_condition']==condition];ii=[i for i in allidx if rows[i]['prompt_id']==prompt]
            ch.require(len(ri)==20 and len(ii)==16,'balanced condition counts');value=kid(rad[ri],rad[ii]);cond.append(value)
            ch.close(np.array(m['KID_by_condition'][prompt]),np.array(value),'condition KID '+method+prompt,atol=2e-10,rtol=2e-10)
        ch.close(np.array(m['KID_condition_mean']),np.array(np.mean(cond)),'condition mean '+method,atol=2e-10,rtol=2e-10)
        expected=prdc(rad[real],rad[gi])
        for key,value in expected.items():ch.close(np.array(m['PRDC'][key]),np.array(value),'PRDC '+method+key,atol=1e-12,rtol=0)
        centered=rad[allidx]-rad[allidx].mean(0);singular=np.linalg.svd(centered,compute_uv=False);prob=singular**2/np.sum(singular**2);prob=prob[prob>0]
        rank=float(np.exp(-np.sum(prob*np.log(prob))));ch.close(np.array(m['effective_rank64']),np.array(rank),'rank SVD '+method,atol=2e-9,rtol=2e-9)
    rng=np.random.default_rng(26091631);draw=rng.integers(0,16,size=(4000,16));keys=list(c['generation_seeds'])
    mapscore={(s['method'],s['seed_id'],s['prompt_id']):s for s in scores}
    for a,b in [('public','backbone'),('pooled','public'),('pooled','backbone')]:
        expected=result['paired'][a+'_minus_'+b]
        for metric in ['matched_cosine','specific_margin']:
            values=np.array([sum(mapscore[a,sid,p][metric]-mapscore[b,sid,p][metric] for p in specific)/3 for sid in keys])
            target=expected[metric]
            ch.close(np.array(target['blocks']),values,'paired blocks '+a+b+metric,atol=1e-12,rtol=0)
            ch.close(np.array(target['descriptive_95_interval']),np.percentile(values[draw].mean(1),[2.5,97.5]),'paired descriptive interval '+a+b+metric,atol=1e-12,rtol=0)
            ch.require(target['positive_blocks']==int(np.sum(values>0)) and target['negative_blocks']==int(np.sum(values<0)),'block sign counts')
    report=dict(status='PASS_MEDICAL_GENERATION_METRICS_SAVED_FEATURES',complete=True,checks=ch.count,seconds=time.perf_counter()-start,
        generated=192,references=40,seed_blocks=16,encoder_full_replay=False,encoder_saved_replay8_checked=True,
        evaluation_sha256=sha(out/'evaluation.json'),evaluation_contract_sha256=sha(out/'evaluation_contract.json'),code_sha256=sha(__file__),errors=ch.errors,clinical_utility_proven=False)
    save(out/'evaluation_verification.json',report);print({k:report[k] for k in ['status','checks','seconds']},flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=DEFAULT);a=p.parse_args();verify(a.out)
