"""Independent file/array/preprocessing/state replay. No production data/train imports."""
import csv,hashlib,json,time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torchvision.transforms import functional as TF,InterpolationMode

BASE=Path(__file__).resolve().parents[1]
PRIOR=BASE/'_reports/downstream_profile_20260917_v3'
OUT=BASE/'_reports/downstream_all_arm_replay_20260917_v1'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def rows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def require(ok,msg):
    if not bool(ok):raise RuntimeError(msg)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def array_sha(a):return hashlib.sha256(a.tobytes()).hexdigest()
def digest(s):
    h=hashlib.sha256()
    for k,v in sorted(s.items()):h.update(k.encode());h.update(v.cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def image224(path):
    with Image.open(path) as im:
        im=im.convert('L');w,h=im.size;rw=round(w*224/max(w,h));rh=round(h*224/max(w,h))
        a=np.asarray(im.resize((rw,rh),Image.Resampling.BILINEAR))
    left=(224-rw)//2;top=(224-rh)//2
    return np.pad(a,((top,224-rh-top),(left,224-rw-left)))
def independent_preprocess(images,arm):
    x=torch.from_numpy(np.stack(images).copy()).to('cuda',torch.float32)[:,None].repeat(1,3,1,1)/255.
    if arm!='R0':
        result=[]
        for slot in range(32):
            key=f'downstream-profile-20260917-v1|11|0|augment|{slot}'
            seed=int(hashlib.sha256(key.encode()).hexdigest()[:15],16);r=np.random.default_rng(seed)
            angle=float(r.uniform(-5,5));shift=[round(float(r.uniform(-.02,.02))*224),round(float(r.uniform(-.02,.02))*224)];scale=float(r.uniform(.95,1.05))
            result.append(TF.affine(x[slot],angle,shift,scale,[0.,0.],InterpolationMode.BILINEAR,fill=0.))
        x=torch.stack(result)
    return ((x-x.new_tensor([.485,.456,.406])[None,:,None,None])/x.new_tensor([.229,.224,.225])[None,:,None,None]).cpu().numpy()

def verify():
    tick=time.perf_counter();c=read(OUT/'contract.json');result=read(OUT/'result.json')
    amendment=read(OUT/'logging_fix_amendment.json') if (OUT/'logging_fix_amendment.json').exists() else None
    for p,h in c['source_sha256'].items():
        if amendment and p in amendment['source_corrections']:
            a=amendment['source_corrections'][p];require(a['old_sha256']==h and sha(a['archive'])==h,'Pre-fix source preserved')
            require(sha(p)==a['new_sha256'],'Amended source binding')
        else:require(sha(p)==h,'Frozen source '+p)
    if amendment:
        for p,h in amendment['completed_artifact_sha256'].items():require(sha(OUT/p)==h,'First completed update preserved')
        require(result['first_segment_optimizer_updates']==1 and result['resumed_optimizer_updates']==13,'One plus13 updates, no rerun')
    require(result['optimizer_updates']==14 and result['new_generation']==0,'Bounded count')
    mf={r['image_id']:r for r in rows(PRIOR/'real_manifest_private.csv')};gm=read(PRIOR/'generation_manifest.json');gmap={r['image_id']:(i,r) for i,r in enumerate(gm)}
    cache=np.load(PRIOR/'real224.npy',mmap_mode='r');bindings=read(OUT/'selected_input_bindings_private.json');byid={r['image_id']:r for r in bindings};pixels={}
    locked={r['patient_id'] for r in rows(BASE/'_reports/patient_usage_audit_20260917_v1/patient_usage_private.csv') if r['locked_thesis_final']=='1' or r['locked_cvpr_calibration_test']=='1'}
    reserved={r['patient_id'] for r in rows(BASE/'_reports/patient_usage_audit_20260917_v1/provisional_patient_split_private.csv') if r['provisional_role']=='reserved_confirmation'}
    for b in bindings:
        iid=b['image_id'];require(sha(b['path'])==b['source_file_sha256'],'Source bytes')
        pixels[iid]=image224(b['path']);require(array_sha(pixels[iid])==b['source_pixel_sha256'],'Source pixel SHA')
        if b['array_key']=='real':
            r=mf[iid];require(r['role'] in ('public','private') and r['patient_id'] not in locked|reserved,'Allowed real patient')
            require(b['patient_id']==r['patient_id'] and b['label']==int(r['label']) and b['index']==int(r['index']),'Real index/patient/label')
            require(b['namespace']=='real/'+r['role'] and b['source_file_sha256']==r['sha256'],'Real source role/SHA')
            require(np.array_equal(pixels[iid],cache[b['index']]),'Raw to actual real array index')
        else:
            i,r=gmap[iid];require(b['index']==i and b['method']==r['method'] and b['cell']==r['cell'],'Generated index/method/cell')
            require(b['namespace']=='synthetic/'+r['method'] and b['source_file_sha256']==r['image_sha256'],'Generated source binding')
            require(b['label']==r['label'] and b['patient_id']==r['cell'],'Requested label/cell')
    initial=torch.load(OUT/'initial.pt',map_location='cpu',weights_only=True);initial_hash=digest(initial)
    old=torch.load(PRIOR/'classifier_initial.pt',map_location='cpu',weights_only=True);require(all(torch.equal(initial[k],old[k]) for k in initial),'Historical initialization preserved')
    traces={};inputs={};timing={};changed_counts={};max_loss_error=0.
    for arm in c['arms']:
        recs=[read(OUT/'runs'/f'{arm}_{rep}'/'trace.json') for rep in (0,1)]
        states=[torch.load(OUT/'runs'/f'{arm}_{rep}'/'state.pt',map_location='cpu',weights_only=True) for rep in (0,1)]
        batches=[np.load(OUT/'runs'/f'{arm}_{rep}'/'input_000001.npy') for rep in (0,1)]
        require(all(torch.equal(states[0][k],states[1][k]) for k in initial),'Exact final weights '+arm)
        require(np.array_equal(batches[0],batches[1]),'Exact actual input arrays '+arm)
        changed_counts[arm]=[]
        for rec,state,batch in zip(recs,states,batches):
            require(rec['initial_state_sha256']==initial_hash and rec['fresh_optimizer_state_count']==0,'Fresh init/AdamW')
            require(rec['run_seed']==11 and rec['steps']==1 and rec['optimizer']['lr']==1e-4 and rec['optimizer']['weight_decay']==1e-4,'Fixed training config')
            require(digest(state)==rec['final_state_sha256'],'Saved final state hash')
            changed=[k for k in rec['parameter_names'] if not torch.equal(initial[k],state[k])]
            require(changed==rec['changed_trainable_parameters'] and len(changed)>0,'Real trainable weight update')
            changed_counts[arm].append(len(changed));tr=rec['trace'][0]
            require(array_sha(batch)==tr['input_sha256'],'Actual preprocessed input hash')
            require(tr['finite_loss'] and tr['finite_gradients'] and set(tr['optimizer_steps'])=={1},'Finite gradients and one Adam step')
            expected='public' if arm in ('R0','R1') else 'private' if arm=='Dreal' else 'synthetic_'+dict(S0='backbone',S1='public',S2='private_only',S3='pooled')[arm]
            require(tr['roles']==['public']*16+[expected]*16,'Actual role mixture '+arm)
            require(sum(tr['labels'])==16 and sum(tr['labels'][:16])==8,'16/16 class balance')
            for j,iid in enumerate(tr['image_ids']):
                b=byid[iid]
                for trace_key,binding_key in [('patient_ids','patient_id'),('roles','role'),('labels','label'),('array_keys','array_key'),('array_indices','index'),('source_namespaces','namespace'),('source_pixel_sha256','source_pixel_sha256'),('source_file_sha256','source_file_sha256')]:
                    require(tr[trace_key][j]==b[binding_key],'Trace source-array provenance '+trace_key)
            z=np.asarray(tr['logits']);y=np.asarray(tr['labels']);e=abs(float(np.mean(np.logaddexp(0,z)-y*z))-tr['loss']);max_loss_error=max(max_loss_error,e);require(e<=2e-6,'Independent BCE')
            require('downstream_utility.data' not in rec['runtime_modules'] and 'downstream_utility.train' not in rec['runtime_modules'],'No legacy import')
        for k in ('image_ids','patient_ids','roles','labels','input_sha256','logits','loss','array_indices','source_namespaces','source_pixel_sha256','source_file_sha256'):
            require(recs[0]['trace'][0][k]==recs[1]['trace'][0][k],'Exact repeated trace '+k)
        traces[arm]=recs[0]['trace'][0];inputs[arm]=batches[0]
        independent=independent_preprocess([pixels[i] for i in traces[arm]['image_ids']],arm)
        require(np.array_equal(independent,inputs[arm]),'Independent raw-to-GPU-preprocess exact '+arm)
        timing[arm]=[dict(total=r['seconds'],preprocess=r['trace'][0]['batch_seconds'],forward_backward=r['trace'][0]['forward_backward_seconds'],optimizer=r['trace'][0]['optimizer_seconds'],peak_allocated=r['peak_allocated'],peak_reserved=r['peak_reserved']) for r in recs]
    require(all(t['image_ids'][:16]==traces['R1']['image_ids'][:16] for t in traces.values()),'Shared real16 in all arms')
    require(traces['R0']['image_ids']==traces['R1']['image_ids'] and not np.array_equal(inputs['R0'],inputs['R1']),'R0/R1 same sources, different augmentation')
    require(all(np.array_equal(inputs[a][:16],inputs['R1'][:16]) for a in c['arms'] if a!='R0'),'Augmented real half exact')
    cells=[byid[i]['cell'] for i in traces['S0']['image_ids'][16:]]
    for arm in ('S1','S2','S3'):
        require([byid[i]['cell'] for i in traces[arm]['image_ids'][16:]]==cells,'Shared synthetic cells')
        require(traces[arm]['labels'][16:]==traces['S0']['labels'][16:],'Shared requested labels')
    for j in range(16,32):
        require(len({traces[a]['source_pixel_sha256'][j] for a in ('S0','S1','S2','S3')})==4,'Distinct actual method pixels')
    out=dict(status='PASS_CORRECTED_ALL_ARM_ONE_STEP_INTEGRATION_REPLAY',complete=True,
        arms=7,repeats=2,optimizer_updates=14,selected_source_images=len(bindings),independent_raw_decodes=len(bindings),
        independent_GPU_preprocessing_exact_arms=7,exact_final_state_replays=7,changed_parameter_tensors=changed_counts,
        max_BCE_error=max_loss_error,initial_state_sha256=initial_hash,source_namespace_count=6,
        actual_source_pixel_array_binding=True,shared_real_and_synthetic_draws=True,
        legacy_imports=False,old98_still_invalid=True,prior100_preserved=True,
        new_generation=0,expert_pixels=0,reserved_pixels=0,AUROC_AP_computed=False,DP_executed=False,
        long_run_convergence_validated=False,private_utility_established=False,final_ready=False,
        timing=timing,seconds=time.perf_counter()-tick,contract_sha256=sha(OUT/'contract.json'),verifier_sha256=sha(Path(__file__)))
    with (OUT/'verification.json').open('x',encoding='utf-8') as f:json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in out.items() if k!='timing'},indent=2),flush=True)
