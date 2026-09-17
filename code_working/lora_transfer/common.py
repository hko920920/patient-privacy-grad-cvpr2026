from downstream_utility.common import CODE,RESEARCH,RAW,MODEL_FILE,MODEL_SHA,require,sha,read,rows,save,now,seed,tensor_sha,setup
from pathlib import Path
import os,sys,json,time
from collections import Counter

OUT=CODE/'_reports/lora_transfer_20260917_v1'
PLAN_DIR=CODE/'_reports/lora_transfer_plan_20260917_v1'
PLAN=RESEARCH/'spec_sources/lora_transfer_comparison_plan_20260917.json'
OLD=CODE/'_reports/downstream_development_20260917_v1'
PROFILE=CODE/'_reports/downstream_profile_20260917_v3'
ARMS=('L_public','L_pooled')
SEEDS=(11,23,37)
METHODS={'L_public':'lora_public','L_pooled':'lora_pooled'}
ACCESS=[]

def log(**x):print(json.dumps(dict(utc=now(),**x)),flush=True)
def tsave(obj,path):
    import torch
    with Path(path).open('xb') as f:torch.save(obj,f)
def npz(path,**x):
    import numpy as np
    with Path(path).open('xb') as f:np.savez_compressed(f,**x)
def completed(folder):
    if not (folder/'complete.json').exists():return False
    for p,h in read(folder/'complete.json')['files'].items():require(sha(folder/p)==h,'Completed artifact changed: '+str(folder/p))
    return True
def fresh(folder):
    if completed(folder):return False
    require(not folder.exists(),'Partial output preserved; explicit technical amendment needed: '+str(folder))
    folder.mkdir(parents=True);return True
def seal(folder,files,**extra):
    save(folder/'complete.json',dict(created_utc=now(),files={os.path.relpath(p,folder):sha(p) for p in files},**extra))
def model_state_hash(state):
    import hashlib
    h=hashlib.sha256()
    for n,p in sorted(state.items()):
        h.update(n.encode());h.update(str(p.dtype).encode());h.update(str(tuple(p.shape)).encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def modules():
    require(not {'downstream_utility.data','downstream_utility.train'} & set(sys.modules),'Forbidden legacy module')
    require(any(type(x).__name__=='RejectLegacy' for x in sys.meta_path),'Safe import guard required')
    return {n:str(m.__file__) for n,m in sys.modules.items() if n.startswith(('lora_transfer','downstream_utility','u_patient_audit.models')) and getattr(m,'__file__',None)}
def validate_sources():
    c=read(OUT/'contract.json');modules()
    for p,h in c['source_sha256'].items():require(sha(p)==h,'Frozen source changed: '+p)
    return c
def access_guard(phase):
    """Raw pixels are allowlisted per phase; broad historic cache is never allowed."""
    manifest=read(PLAN_DIR/'training_images_private.json')
    role=phase.removeprefix('cache_')
    allowed={os.path.normcase(os.path.abspath(r['path'])) for r in manifest if phase.startswith('cache_') and r['split']==('public' if role=='public' else 'train')}
    # Classifier replay independently reopens only its selected public files.
    if phase=='classifier_replay':
        allowed={os.path.normcase(os.path.abspath(r['path'])) for r in rows(PROFILE/'real_manifest_private.csv') if r['role']=='public'}
    rawroot=os.path.normcase(os.path.abspath(RAW))+os.sep
    privatecache=os.path.normcase(os.path.abspath(OUT/'cache_private/cache.pt'))
    broad=os.path.normcase(os.path.abspath(CODE/'_reports/cvpr_u_pilot_v1_001/cache/cache.pt'))
    def audit(event,args):
        if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=os.path.normcase(os.path.abspath(os.fsdecode(args[0])))
        if p==broad:raise RuntimeError('Broad historic mixed cache forbidden')
        if phase in ('train_public','cache_public','e4_replay') and p==privatecache:raise RuntimeError('Public-only private cache read forbidden')
        if p.startswith(rawroot):
            require(p in allowed,'Non-allowlisted raw pixel open: '+p)
            ACCESS.append(dict(phase=phase,kind='raw',path=p))
        elif p.endswith('cache.pt') and str(OUT).lower() in p.lower():ACCESS.append(dict(phase=phase,kind='cache',path=p))
    sys.addaudithook(audit)
def audit_record():
    return dict(raw_files=sorted({r['path'] for r in ACCESS if r['kind']=='raw'}),cache_files=sorted({r['path'] for r in ACCESS if r['kind']=='cache'}),expert_raw_access=0,reserved_raw_access=0)
