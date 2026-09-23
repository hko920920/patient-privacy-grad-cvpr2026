"""P-only DINO projection, then separate P/Q condition signals; no utility evaluation."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from prrd_v3.contracts import CODE, OUT, save, sha, setup, require, now, source_hashes
from prrd_v3.datasets import manifests, PixelAccess, TRAIN, DEV
from prrd_v3.encoders import pil_tensor, state_hash, fit_projection
from prrd_v3.bank_runtime import digest
from .signals import conditions, heads, per_image_gradient, patient_class_sums, pool_patient_sums
from .second_source import DinoEncoder, ENCODER_ID, WEIGHTS, PREPROCESS, descriptor, load_combined
from .prepare_a1 import independent


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--a1-handoff',type=Path,required=True)
    ap.add_argument('--campaign-contract',type=Path,required=True)
    args=ap.parse_args()
    start=time.monotonic()
    args.output.mkdir(parents=True,exist_ok=False)
    private=args.output/'private'; private.mkdir()
    public=args.output/'public'; public.mkdir()
    group=manifests()
    dependencies=[WEIGHTS,TRAIN,DEV,args.a1_handoff,args.campaign_contract]+list(Path(__file__).parent.glob('*.py'))
    contract={'created':now(),'scope':'DINO_PUBLIC_PROJECTION_AND_PQ_CONDITION_TARGETS_NONDP',
              'checkpoint':str(WEIGHTS),'checkpoint_sha256':sha(WEIGHTS),'preprocess':PREPROCESS,
              'public_fit_images':813,'P_patients':672,'Q_patients':2027,
              'P_Q_images':5910,'K':4,'dimension':16,'microbatch':16,
              'class_patient_weights':'unchanged; 0.5 per class; pooled sums/counts',
              'forward_images_expected':813+23640,'backward_images':0,
              'private_targets_and_features':True,'Q_V_performance_selection':False,
              'V_recipient_DP_final':False,'independent_atol':1e-12,
              'bindings':{str(p):sha(p) for p in dependencies},'legacy_sources':source_hashes()}
    save(args.output/'contract.json',contract)
    setup(); access=PixelAccess(group,('P','Q')).install()
    encoder=DinoEncoder(); initial=state_hash(encoder.model)
    counts={'forward_images':0,'backward_images':0}
    def hook(module,inputs,output):
        counts['forward_images']+=len(inputs[0])
        require(not output.requires_grad,'Extraction unexpectedly retained graph')
    handle=encoder.model.register_forward_hook(hook)
    torch.cuda.reset_peak_memory_stats()
    features=[]
    for i in range(0,len(group['P']),16):
        rows=group['P'][i:i+16]
        native=torch.stack([pil_tensor(access.image(r)) for r in rows]).cuda()
        with torch.no_grad(): features.append(encoder.raw(native).cpu().numpy())
        if i%256==0: print(json.dumps({'phase':'DINO_public_projection','done':i+len(rows)}),flush=True)
    clean=np.concatenate(features)
    projection=public/'dinov2_P_projection.npz'
    info=fit_projection(clean,group['P'],16,ENCODER_ID,projection)
    np.savez(public/'dinov2_P_clean_features.npz',features=clean,
             image_ids=np.array([r['image_id'] for r in group['P']]))
    encoder.use_projection(projection)
    rule=descriptor(projection)
    save(private/'condition_rule.json',rule)
    cs, hs=conditions(),heads(ENCODER_ID,16)
    paths,stats,times={},{},{}
    for role in ('P','Q'):
        tick=time.monotonic(); rows=group[role]
        require(all(int(r['width'])==int(r['height'])==1024 for r in rows),'Geometry changed')
        z=np.empty((4,len(rows),16),dtype=np.float32)
        for i in range(0,len(rows),16):
            part=rows[i:i+16]
            native=torch.stack([pil_tensor(access.image(r)) for r in part]).cuda()
            with torch.no_grad():
                for k,condition in enumerate(cs):
                    value=encoder(condition.apply(native))
                    require(torch.isfinite(value).all() and value.norm(dim=-1).max()<=1+1e-6,
                            'Invalid DINO bounded feature')
                    z[k,i:i+len(part)]=value.cpu().numpy()
            if i%256==0 or i+len(part)==len(rows):
                print(json.dumps({'phase':'DINO_conditions','role':role,'done':i+len(part),
                                  'total':len(rows),'seconds':time.monotonic()-tick}),flush=True)
        path=private/(role+'_condition_features.npz')
        labels=[int(r['label']) for r in rows]; pids=[r['patient_id'] for r in rows]
        np.savez(path,z=z,condition_ids=np.arange(4),image_ids=np.array([r['image_id'] for r in rows]),
                 patient_ids=np.array(pids),labels=np.array(labels),roles=np.full(len(rows),role),
                 rule_sha256=np.array(digest(rule)))
        gradient=torch.stack([per_image_gradient(torch.from_numpy(z[k]),labels,h) for k,h in enumerate(hs)])
        stats[role]=patient_class_sums(gradient,labels,pids)
        paths[role]=path; times[role]=time.monotonic()-tick
    stats['pooled']=pool_patient_sums(stats['P'],stats['Q'])
    arrays={'schema':np.array('receiver.separate-condition-target/v1'),
            'rule_sha256':np.array(digest(rule)),'condition_ids':np.arange(4)}
    for name,s in stats.items():
        arrays.update({name+'_sums':s.sums.numpy(),name+'_counts':s.counts.numpy(),
                       name+'_gradient':s.balanced().numpy()})
    target=private/'DINO_condition_targets.npz'; np.savez(target,**arrays)
    handle.remove()
    require(counts=={'forward_images':24453,'backward_images':0},'DINO extraction count')
    require(state_hash(encoder.model)==initial,'DINO model changed')
    verification=independent(paths,target,rule)
    save(args.output/'independent_verification.json',verification)
    handoff={'schema':'receiver.target-handoff/v1','target_file':str(target),'target_sha256':sha(target),
             'rule_file':str(private/'condition_rule.json'),'rule_file_sha256':sha(private/'condition_rule.json'),
             'rule_sha256':digest(rule),'contract_file':str(args.output/'contract.json'),
             'contract_sha256':sha(args.output/'contract.json'),'population':'pooled',
             'privacy':'NONDP_INTERNAL_Q_DERIVED'}
    save(private/'target_handoff.json',handoff)
    objective,_=load_combined(args.a1_handoff,handoff,(0,)*64+(1,)*64)
    require(torch.equal(objective.targets[ENCODER_ID],stats['pooled'].balanced()),'Combined loader changed signal')
    require(all(sha(p)==h for p,h in contract['bindings'].items()),'Bound implementation changed')
    require(source_hashes()==contract['legacy_sources'],'Legacy code changed')
    result={'status':'PASS_DINO_P_ONLY_PROJECTION_PQ_CONDITION_TARGETS','created':now(),
            'seconds':time.monotonic()-start,'per_role_seconds':times,'counts':counts,
            'projection':info,'verification':verification,'model_fixed':True,'A1_target_unchanged':True,
            'access':access.report(),'target_sha256':sha(target),
            'handoff_sha256':sha(private/'target_handoff.json'),
            'peak_allocated_MiB':torch.cuda.max_memory_allocated()/2**20,
            'peak_reserved_MiB':torch.cuda.max_memory_reserved()/2**20,
            'V_recipient_DP_final':False}
    save(args.output/'result.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__': main()
