"""Additional saved-input verification; no model inference or training."""
import sys,json,hashlib
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
CODE=ROOT.parents[1]/'code_working'
sys.path.insert(0,str(CODE))
from downstream_utility.run_v2 import install_guard
install_guard()
from downstream_utility.verify_all_arm_replay import image224,independent_preprocess,array_sha,read,rows,sha,require
from lora_transfer.common import OUT,PROFILE,ARMS,setup,access_guard,audit_record

def main():
    setup();access_guard('classifier_replay')
    folder=OUT/'classifier_replay';require((folder/'report.json').exists(),'Completed actual replay required')
    real={r['image_id']:r for r in rows(PROFILE/'real_manifest_private.csv') if r['role']=='public'}
    generated={r['image_id']:r for r in read(OUT/'generation_manifest.json')}
    pixels={};bindings={};records=[]
    for arm in ARMS:
        for rep in (0,1):
            sub=folder/f'{arm}_{rep}';trace=read(sub/'trace.json')['trace'][0]
            for iid,file_sha,pixel_sha in zip(trace['image_ids'],trace['source_file_sha256'],trace['source_pixel_sha256']):
                r=real.get(iid,generated.get(iid));require(r is not None,'Known selected source')
                p=Path(r['path']) if iid in real else OUT/r['image_path']
                require(sha(p)==file_sha,'Physical file binding')
                if iid not in pixels:pixels[iid]=image224(p)
                require(array_sha(pixels[iid])==pixel_sha,'Independent pixel/array hash')
                bindings[str(p)]=file_sha
            actual=np.load(sub/'input_000001.npy',allow_pickle=False)
            expected=independent_preprocess([pixels[i] for i in trace['image_ids']],arm)
            require(np.array_equal(actual,expected),'Independent GPU augmentation/normalization exact')
            require(array_sha(actual)==trace['input_sha256'],'Saved GPU input trace SHA')
            records.append(dict(arm=arm,repeat=rep,exact=True,input_sha256=trace['input_sha256'],input_file_sha256=sha(sub/'input_000001.npy'),trace_sha256=sha(sub/'trace.json')))
    trainings={a:[json.loads(line) for line in (OUT/'training'/a/'trace.jsonl').read_text().splitlines() if json.loads(line)['committed']] for a in ARMS}
    paired_noise=0;shared_public=0
    for left,right in zip(trainings['L_public'],trainings['L_pooled']):
        require(left['noise_sha256']==right['noise_sha256'] and left['noise_seed']==right['noise_seed'] and left['timesteps']==right['timesteps'],'Actual training noise/timestep pairing')
        paired_noise+=1
        for slot,role in enumerate(right['roles']):
            if role=='public':
                require(left['image_ids'][slot]==right['image_ids'][slot] and left['patient_ids'][slot]==right['patient_ids'][slot],'Actual shared public slot')
                shared_public+=1
    require(paired_noise==448 and shared_public==512,'Actual full training pairing count')
    # Check the first ten prospectively indexed patient bootstrap draws against
    # sklearn's weighted metrics; the main verifier computes all2000 draws.
    from sklearn.metrics import roc_auc_score,average_precision_score
    bootstrap=np.load(OUT/'cluster_bootstrap.npz');arms=bootstrap['arms'].tolist();packets={}
    for arm in arms:
        base=OUT if arm in ARMS else CODE/'_reports/downstream_development_20260917_v1'
        packets[arm]=[dict(np.load(base/'evaluation'/f'{arm}_{seed}.npz')) for seed in (11,23,37)]
    ref=packets[arms[0]][0];ids,inverse=np.unique(ref['patients'],return_inverse=True)
    rng=np.random.default_rng(read(OUT/'contract.json')['plan']['evaluation']['bootstrap_seed']);bootstrap_error=0.
    for draw_index in range(10):
        weights=np.bincount(rng.integers(len(ids),size=len(ids)),minlength=len(ids))[inverse]
        for ai,arm in enumerate(arms):
            for si,packet in enumerate(packets[arm]):
                metrics=[roc_auc_score(packet['labels'],packet['logits'],sample_weight=weights),average_precision_score(packet['labels'],packet['logits'],sample_weight=weights)]
                error=float(np.max(np.abs(np.array(metrics)-bootstrap['values'][draw_index,ai,si])))
                bootstrap_error=max(bootstrap_error,error);require(error<1e-12,'Independent weighted bootstrap metrics')
    result=dict(status='PASS_INDEPENDENT_NEW_LORA_RAW_PNG_TO_GPU_INPUT',records=records,selected_files=len(bindings),source_file_sha256=bindings,
        actual_paired_training_noise_batches=paired_noise,actual_shared_public_slots=shared_public,
        bootstrap_crosscheck_fixed_first_draws=10,bootstrap_crosscheck_models=len(arms)*3,bootstrap_max_abs_error=bootstrap_error,
        script_sha256=sha(Path(__file__)),independent_preprocessor_sha256=sha(CODE/'downstream_utility/verify_all_arm_replay.py'),contract_sha256=sha(OUT/'contract.json'),
        new_model_forwards=0,new_training=0,new_generation=0,access=audit_record())
    p=OUT/'additional_input_verification.json'
    with p.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('source_file_sha256','access')},ensure_ascii=True))

if __name__=='__main__':main()
