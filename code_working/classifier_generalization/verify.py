"""Independent raw pixel, exposure and metric checks; does not import the production metric/predictor."""
from .common import CODE, RESEARCH, OUT, PROFILE, sha, read, save, now, require, tensor_sha, setup, state_hash
from collections import Counter
from pathlib import Path
import time
import numpy as np
from PIL import Image
import torch

def independent_stats(y,z,w=None):
    y=np.asarray(y,dtype=np.int64);z=np.asarray(z,dtype=np.float64)
    pos=z[y==1];neg=z[y==0]
    auc=float(np.mean((pos[:,None]>neg[None,:])+.5*(pos[:,None]==neg[None,:])))
    order=np.argsort(-z,kind='stable');ys=y[order];zs=z[order]
    end=np.r_[np.flatnonzero(zs[:-1]!=zs[1:]),len(zs)-1]
    tp=np.cumsum(ys)[end];r=tp/ys.sum();precision=tp/(end+1)
    ap=float(np.sum(np.diff(np.r_[0.,r])*precision))
    loss=np.maximum(z,0)+np.log1p(np.exp(-np.abs(z)))-y*z
    ans=dict(AUROC=auc,AP=ap,BCE=float(loss.mean()),
             balanced_BCE=float((loss[y==0].mean()+loss[y==1].mean())/2))
    if w is not None:ans['exposure_weighted_BCE']=float(np.dot(loss,w)/sum(w))
    return ans

def main():
    t=time.perf_counter();count=0;maxdiff=0.
    def check(v,msg):
        nonlocal count
        count+=1;require(v,msg)
    c=read(OUT/'contract.json');result=read(OUT/'result.json')
    check(not c['training_allowed'] and not c['generation_allowed'] and not c['expert_allowed'] and not c['reserved_allowed'],'Forbidden scope')
    for p,h in c['source_sha256'].items():check(sha(p)==h,'Frozen source')
    complete=read(OUT/'inference_complete.json')
    for p,h in complete['files'].items():check(sha(OUT/p)==h,'Sealed inference')
    check(sha(OUT/'result.json')==complete['result_sha256'],'Result sealed')
    manifest=read(OUT/'manifest_private.json');lookup={r['image_id']:r for r in manifest}
    real={r['image_id']:r for r in __import__('downstream_utility.common',fromlist=['rows']).rows(PROFILE/'real_manifest_private.csv')}
    allowed=set(real)
    audit_root=CODE/'_reports/patient_usage_audit_20260917_v1'
    from downstream_utility.common import rows
    denied={int(r['patient_id']) for r in rows(audit_root/'patient_usage_private.csv') if r['locked_thesis_final']=='1' or r['locked_cvpr_calibration_test']=='1'}
    denied|={int(r['patient_id']) for r in rows(audit_root/'provisional_patient_split_private.csv') if r['provisional_role']=='reserved_confirmation'}
    # Independently reproduce every allowed raw/PNG letterbox.
    pixels={}
    for r in manifest:
        if r['namespace'].startswith(('real/','witness/')):
            check(r['image_id'] in allowed and int(r['patient_or_cell_id']) not in denied,'Patient allowed')
            orig=real[r['image_id']]
            check(int(orig['label'])==r['label'] and str(orig['patient_id'])==r['patient_or_cell_id'],'Real label/patient')
            check(Path(orig['path']).resolve()==Path(r['path']).resolve(),'Real path')
        check(sha(r['path'])==r['file_sha256'],'Independent pixel hash')
        with Image.open(r['path']) as im:
            gray=im.convert('L');w,h=gray.size
            nw=max(1,round(224*w/max(w,h)));nh=max(1,round(224*h/max(w,h)))
            resized=np.asarray(gray.resize((nw,nh),Image.Resampling.BILINEAR),dtype=np.uint8)
            a=np.zeros((224,224),dtype=np.uint8);x=(224-nw)//2;y=(224-nh)//2
            a[y:y+nh,x:x+nw]=resized
        check(tensor_sha(a)==r['pixel_sha256'],'Independent letterbox pixels')
        pixels[r['image_id']]=a
    setup();model_checks=[];presentations=0
    runs=read(OUT/'runs_private.json')
    for spec,observed in zip(runs,result['model_results']):
        name=f"{spec['arm']}_{spec['seed']}"
        check(observed['arm']==spec['arm'] and observed['seed']==spec['seed'],'Run alignment')
        with np.load(OUT/'predictions'/f'{name}.npz',allow_pickle=False) as z:packet={k:z[k] for k in z.files}
        trace=read(spec['trace'])['trace'];counts=Counter()
        for b in trace:
            for j,iid in enumerate(b['image_ids']):
                counts[iid]+=1;r=lookup[iid]
                check(r['label']==int(b['labels'][j]) and r['namespace']==b['source_namespaces'][j]
                      and r['patient_or_cell_id']==str(b['patient_ids'][j]),'Trace identities')
                check(r['file_sha256']==b['source_file_sha256'][j] and r['pixel_sha256']==b['source_pixel_sha256'][j],'Trace pixels')
        check(packet['image_ids'].tolist()==sorted(counts),'Only actually seen unique images')
        check(packet['exposures'].tolist()==[counts[i] for i in packet['image_ids']],'Exact exposure count')
        check(sum(counts.values())==12800,'Fixed exposure budget')
        for j,iid in enumerate(packet['image_ids']):
            r=lookup[iid]
            check(int(packet['labels'][j])==r['label'] and str(packet['namespaces'][j])==r['namespace']
                  and str(packet['patient_or_cell_ids'][j])==r['patient_or_cell_id'],'Prediction binding')
        for ns in sorted(set(packet['namespaces'])):
            mask=packet['namespaces']==ns
            actual=independent_stats(packet['labels'][mask],packet['logits'][mask],packet['exposures'][mask])
            ref=observed['components'][ns]
            for k,v in actual.items():
                d=abs(v-ref[k]);maxdiff=max(maxdiff,d);check(d<1e-12,'Independent component metric '+k)
            for label in (0,1):
                m=mask&(packet['labels']==label)
                zz=packet['logits'][m];yy=packet['labels'][m]
                loss=np.maximum(zz,0)+np.log1p(np.exp(-np.abs(zz)))-yy*zz
                check(abs(float(loss.mean())-ref['by_label'][str(label)]['BCE'])<1e-12,'Class BCE')
                check(np.array_equal(np.quantile(zz,[0,.05,.25,.5,.75,.95,1]),ref['by_label'][str(label)]['logit_quantiles']),'Logit quantiles')
        with np.load(spec['stored_development'],allow_pickle=False) as p:old={k:p[k] for k in p.files}
        for k,v in independent_stats(old['labels'],old['logits']).items():
            d=abs(v-observed['development'][k]);maxdiff=max(maxdiff,d);check(d<1e-12,'Independent dev metric')
        difference=float(np.max(np.abs(packet['witness_logits']-old['logits'][:64])))
        check(difference<=c['witness_absolute_tolerance'],'Fixed development replay')
        execution=read(OUT/'predictions'/f'{name}_execution.json')
        checkpoint=torch.load(spec['checkpoint'],map_location='cpu',weights_only=True)
        h=state_hash(checkpoint)
        check(h==execution['full_state_before']==execution['full_state_after'],'State before/after checkpoint identity')
        check(execution['all_parameters_exact'] and execution['all_buffers_exact'] and execution['BN_running_buffers']==60,'BN state preserved')
        check(execution['eval_all_modules'] and execution['inference_mode'],'Frozen inference behavior')
        for b in execution['batches']:
            a=np.stack([pixels[i] for i in b['image_ids']])
            x=torch.tensor(a.copy(),device='cuda',dtype=torch.float32)[:,None].repeat(1,3,1,1)/255.
            mu=torch.tensor([.485,.456,.406],device='cuda',dtype=torch.float32)[None,:,None,None]
            sd=torch.tensor([.229,.224,.225],device='cuda',dtype=torch.float32)[None,:,None,None]
            independent=(x-mu)/sd
            check(tensor_sha(independent)==b['input_sha256'],'Independent GPU input exact')
        for window,key in [(50,'50'),(400,'400'),(50,'first50')]:
            part=trace[:window] if key=='first50' else trace[-window:]
            check(abs(float(np.mean([b['loss'] for b in part]))-observed['training_log'][key]['mean_logged_BCE'])<1e-15,'Logged loss aggregation')
            per_source={}
            for b in part:
                z=np.array(b['logits']);y=np.array(b['labels'])
                loss=np.maximum(z,0)+np.log1p(np.exp(-np.abs(z)))-y*z
                for ns,v in zip(b['source_namespaces'],loss):per_source.setdefault(ns,[]).append(float(v))
            for ns,values in per_source.items():
                check(abs(float(np.mean(values))-observed['training_log'][key]['source_BCE'][ns])<1e-12,'Source log BCE')
        presentations+=len(packet['image_ids'])
        model_checks.append(dict(run=name,seen_images=len(packet['image_ids']),state_unchanged=True,witness_max_absolute_difference=difference))
    check(presentations==c['budget']['training_image_presentations']==result['actual_training_presentations'],'Training inference budget')
    check(result['actual_witness_presentations']==768 and len(manifest)==1453,'Witness and decode scope')
    check(all(result[k]==0 for k in ['optimizer_updates','new_generation','DP','expert_pixels','reserved_pixels']),'Zero forbidden work')
    for p,h in c['source_sha256'].items():check(sha(p)==h,'Source remains unchanged after independent checks')
    save(OUT/'verification.json',dict(status='PASS_FIXED_INFERENCE_DIAGNOSTIC_INTEGRITY',created_utc=now(),
         checks=count,maximum_metric_difference=maxdiff,models=model_checks,
         independent_raw_PNG_decodes=len(manifest),independent_GPU_input_arithmetic=True,new_model_forward=0,
         no_production_metric_import=True,seconds=time.perf_counter()-t,method_efficacy_verified=False,
         result_sha256=sha(OUT/'result.json'),contract_sha256=sha(OUT/'contract.json')))
    print('PASS',count,'checks','metric max difference',maxdiff,'seconds',time.perf_counter()-t,flush=True)

if __name__=='__main__':main()
