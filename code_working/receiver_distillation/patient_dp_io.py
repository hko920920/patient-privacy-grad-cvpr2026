"""Cached-feature adapter and a separately gated, single-release target writer.

The verification entry point never calls release_from_contract. A future
execution contract must explicitly name one output and one release. Neither
metadata from Q nor RNG state is copied into the protected handoffs.
"""
from pathlib import Path
import json,os
import numpy as np
from prrd_v3.contracts import read,save,sha,require,now
from prrd_v3.bank_runtime import digest
from .patient_dp import (ENCODERS,K,G,Limits,aggregate_image_signals,
                         clipped_contributions,privatize_patients,decode_target)

HANDOFF_KEYS=('a1_handoff','second_handoff')


def cached_paths(job_path):
    job=read(job_path);result={}
    for name,key in zip(ENCODERS,HANDOFF_KEYS):
        hf=Path(job[key]);h=read(hf)
        require(sha(hf)==job[key+'_sha256'],'Historical handoff changed')
        require(sha(h['target_file'])==h['target_sha256'],'Historical target changed')
        require(sha(h['rule_file'])==h['rule_file_sha256'],'Historical rule changed')
        rule=read(h['rule_file']);require(digest(rule)==h['rule_sha256'],'Rule content changed')
        require(rule['encoder_id']==name and rule['conditions']==K and rule['dimension']==16
                and rule['public_seed']==101,'Current fixed recipe mismatch')
        require(sha(rule['checkpoint'])==rule['checkpoint_sha256'],'Checkpoint changed')
        require(sha(rule['projection'])==rule['projection_sha256'],'Public projection changed')
        folder=Path(h['target_file']).parent
        result[name]={'handoff':str(hf),'target':h['target_file'],'rule_file':h['rule_file'],
                      'rule':rule,'P':str(folder/'P_condition_features.npz'),
                      'Q':str(folder/'Q_condition_features.npz')}
    first,second=(result[n]['rule'] for n in ENCODERS)
    for a,b in zip(first['tuples'],second['tuples']):
        require(a['theta']==b['theta'] and a['brightness']==b['brightness'],'Shared conditions differ')
    return result


def read_cached_population(paths,role):
    require(role in ('P','Q'),'Only preparation populations allowed')
    features=[];metadata=None;expected_sums=[];expected_counts=None;expected_targets=[];bindings={}
    for name in ENCODERS:
        item=paths[name];path=Path(item[role]);rule=item['rule']
        with np.load(path,allow_pickle=False) as z:
            require(np.array_equal(z['condition_ids'],np.arange(K)),'Condition axis changed')
            require(str(z['rule_sha256'])==digest(rule),'Cached feature rule differs')
            require(np.all(z['roles']==role),'Wrong cache population')
            cur={k:z[k].copy() for k in ('patient_ids','labels','image_ids')}
            x=z['z'].astype(np.float64)
        if metadata is None:metadata=cur
        else:
            for key in metadata:require(np.array_equal(cur[key],metadata[key]),'Encoder cache order differs')
        require(len(np.unique(cur['image_ids']))==len(cur['image_ids']),'Duplicate image cache row')
        require(x.shape==(K,len(cur['labels']),16) and np.isfinite(x).all(),'Cached features invalid')
        # Same FP64 CE formula as the existing source, independent NumPy path.
        pieces=[]
        for k,t in enumerate(rule['tuples']):
            w,b=np.array(t['weight']),np.array(t['bias'])
            logits=x[k]@w.T+b
            ex=np.exp(logits-logits.max(axis=1,keepdims=True));residual=ex/ex.sum(axis=1,keepdims=True)
            residual[np.arange(len(x[k])),cur['labels'].astype(int)]-=1.
            pieces.append(np.concatenate([(residual[:,:,None]*x[k,:,None,:]).reshape(len(x[k]),32),residual],axis=1))
        features.append(np.stack(pieces,axis=1))
        with np.load(item['target'],allow_pickle=False) as old:
            expected_sums.append(old[role+'_sums'].transpose(1,0,2).copy())
            counts=old[role+'_counts'].copy();expected_targets.append(old[role+'_gradient'].copy())
        if expected_counts is None:expected_counts=counts
        else:require(np.array_equal(counts,expected_counts),'Class counts differ between sources')
        bindings[str(path)]=sha(path)
    require(np.isin(metadata['labels'],[0,1]).all(),'Bad cached labels')
    image_signals=np.stack(features,axis=1)
    patients=aggregate_image_signals(image_signals,metadata['labels'],metadata['patient_ids'])
    return patients,{'sums':np.stack(expected_sums,axis=1),'counts':expected_counts,
                     'targets':np.stack(expected_targets),'metadata':metadata,'feature_hashes':bindings}


def write_target_handoffs(directory,targets,paths,metadata,*,privacy):
    """Save only postprocessed target values and public configuration."""
    require(privacy in ('PUBLIC_SIMULATION_ONLY','PATIENT_DP_SINGLE_QUERY_INTERNAL'),'Explicit target scope')
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    require(set(targets)==set(ENCODERS),'Encoder set differs')
    # Strictly reconstruct public metadata; never dump caller private diagnostics.
    allowed=('schema','epsilon','delta','adjacency','sensitivity','dimension','class_bounds','sigma',
             'calibration','query','private_queries','subsampling','count_noise_std','class_gradient_sum_noise_std')
    require(set(metadata)==set(allowed),'Unexpected mechanism metadata')
    mechanism=directory/'mechanism.json';save(mechanism,{k:metadata[k] for k in allowed})
    handoffs={}
    for name,key in zip(ENCODERS,HANDOFF_KEYS):
        folder=directory/name;folder.mkdir()
        rule=paths[name]['rule'];rulefile=folder/'condition_rule.json';save(rulefile,rule)
        value=np.asarray(targets[name],dtype=np.float64)
        require(value.shape==(K,G) and np.isfinite(value).all() and (np.linalg.norm(value,axis=-1)>1e-12).all(),'Invalid target')
        target=folder/'condition_targets.npz'
        np.savez(target,schema=np.array('receiver.separate-condition-target/v1'),
                 rule_sha256=np.array(digest(rule)),condition_ids=np.arange(K),pooled_gradient=value)
        handoff={'schema':'receiver.target-handoff/v1','target_file':str(target),'target_sha256':sha(target),
                 'rule_file':str(rulefile),'rule_file_sha256':sha(rulefile),'rule_sha256':digest(rule),
                 'contract_file':str(mechanism),'contract_sha256':sha(mechanism),'population':'pooled',
                 'input_population':'P + protected Q' if privacy.startswith('PATIENT') else 'P + public simulated Q',
                 'privacy':privacy,'not_an_execution_authorization':True}
        file=folder/'target_handoff.json';save(file,handoff);handoffs[key]=str(file)
    return handoffs


def release_from_contract(contract_path):
    """Not executed by the design verifier; future explicit one-release entry."""
    contract_path=Path(contract_path);c=read(contract_path)
    require(c.get('schema')=='receiver.patient-dp-release-plan/v1' and c.get('status')=='FROZEN_FOR_EXECUTION',
            'A frozen one-release execution scope is required')
    require(c.get('max_private_releases')==1 and c.get('synthesis_authorized') is False,
            'This command prepares exactly one target, not synthesis')
    require(all(sha(p)==h for p,h in c['input_code_bindings'].items()),'Bound input/code changed')
    calibration=read(c['public_calibration'])
    require(sha(c['public_calibration'])==c['public_calibration_sha256'],'Calibration changed')
    limits=Limits(tuple(calibration['class_bounds']))
    paths=cached_paths(c['reference_job'])
    p,_=read_cached_population(paths,'P');q,_=read_cached_population(paths,'Q')
    require(not(set(p.ids)&set(q.ids)),'P/Q patient overlap')
    ps,pc=p.sums_counts()
    root=Path(c['output_directory']);root.mkdir(parents=True,exist_ok=False)
    # Durable reservation before randomization. A crash never silently resamples.
    marker=root/'RELEASE_RESERVED.json'
    fd=os.open(marker,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'w',encoding='utf-8') as f:
        json.dump({'created':now(),'contract_sha256':sha(contract_path),'status':'RESERVED_ONE_QUERY'},f)
        f.flush();os.fsync(f.fileno())
    noisy,metadata=privatize_patients(q,limits,epsilon=c['epsilon'],delta=c['delta'],adjacency=c['adjacency'])
    targets,post=decode_target(noisy,limits,ps,pc)
    # Q-only diagnostics, raw counts, cache hashes, seed/noise are not in this payload.
    handoffs=write_target_handoffs(root/'protected_targets',targets,paths,metadata,
                                   privacy='PATIENT_DP_SINGLE_QUERY_INTERNAL')
    save(root/'COMPLETED.json',{'created':now(),'status':'ONE_PROTECTED_TARGET_READY',
                              'handoffs':handoffs,'private_queries':1,'synthesis_executed':False})
    return handoffs


def main():
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--execution-contract',type=Path,required=True)
    a=ap.parse_args();print(json.dumps(release_from_contract(a.execution_contract)),flush=True)


if __name__=='__main__':main()

