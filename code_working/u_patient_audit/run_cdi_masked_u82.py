"""Frozen literal CDI extraction on the verified masked-control U82 only."""
import argparse
from collections import Counter, defaultdict
import gc
from importlib.metadata import version
from pathlib import Path
import time
import traceback
import json

import torch

from .common import ROOT,RUN,PROMPT,digest,read_csv,write_json
from .models import setup,load_unet,load_cache,scheduler,adapter_state
from .cdi_adapter import CDIAdapter,FEATURE_NAMES,source_bindings,tensor_digest
from .run_cdi_base_selection import new_json,now,state_fingerprint,hashes_match
from .run_masked_slot_endpoints_v2 import training_evidence

BASE=RUN/'baseline_screen_20260915'
TRAINING=BASE/'masked_patient_training_v1'
TARGET=RUN/'baseline_screen_20260914/cdi_u_cohort_v1'
OUTPUT=BASE/'cdi_masked_u82_v1'
CONTRACT=BASE/'cdi_masked_u82_contract_v1.json'
RESEARCH=next(ROOT.parent.glob('*/research_2026-09-10'))
POLICIES=[RESEARCH/'spec_sources/masked_cdi_fixed_scorer_analysis_contract_20260915.json',
          RESEARCH/'spec_sources/masked_cdi_leave_patient_out_addendum_20260915.json']
ADAPTER_SHA='70d91b15397394d1405c8e930d2ec94ff4dcb0d3bbddd7ace6c8bdd58b0da8ce'
CHECKPOINTS={'treatment':RUN/'training_coverage_v2/model_1/step_1000.pt',
             'control':TRAINING/'control/step_1000.pt'}


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def scope():
    grouped=defaultdict(list)
    for r in read_csv(RUN/'cohort/evaluation_images.csv'):
        if r['eval_role']=='selection' or (r['eval_role']=='fit' and r['patient_id']=='14393'):
            grouped[r['patient_id']].append(r)
    order=['14393']+sorted((p for p in grouped if p!='14393'),key=int)
    assert len(order)==41
    rows=[]
    for pid in order:
        four=grouped[pid]
        assert len(four)==4 and Counter(r['record_role'] for r in four)=={'train_candidate':2,'U_observed':2}
        rows += sorted((r for r in four if r['record_role']=='U_observed'),key=lambda r:r['image_id'])
    assert len(rows)==82 and [r['image_id'] for r in rows[:2]]==['00014393_001.png','00014393_004.png']
    assert Counter(r['assignment_group'] for r in rows[2:])=={'A':40,'B':40}
    return rows,order


def source_and_training():
    states,training_paths,evidence=training_evidence(TRAINING/'verification.json')
    execution,protocol,verified=read(TARGET/'execution.json'),read(TARGET/'protocol.json'),read(TARGET/'verification.json')
    assert execution['complete'] and execution['records']==480
    assert verified['status']=='PASS_SAVED_U_COHORT_ARITHMETIC_AND_INTEGRITY'
    assert verified['analysis_verification']['status']=='PASS_SAVED_ANALYSIS_PREDICTIONS_AND_STATISTICS'
    for field,name in [('results_sha256','results.json'),('protocol_sha256','protocol.json')]:
        assert digest(TARGET/name)==execution[field]
        if field in verified:assert verified[field]==execution[field]
    assert source_bindings()==protocol['source_bindings']
    assert digest(ROOT/'u_patient_audit/cdi_adapter.py')==ADAPTER_SHA
    rows,order=scope();old={r['image_id']:r for r in read(TARGET/'results.json') if r['model']=='model_1'}
    manifest=read_csv(RUN/'cohort/model_1_train.csv')
    metadata=[];sources=[]
    for r in rows:
        iid,pid=r['image_id'],r['patient_id'];prior=old[iid]
        assert all(prior[k]==r[k] for k in ['image_id','patient_id','eval_role','assignment_group','record_role'])
        assert prior['scenario']=='U' and prior['image_member']==prior['actual_training_exposures']==0
        patient_images=[m['image_id'] for m in manifest if m['patient_id']==pid]
        effective=sum(states['control']['exposures'].get(i,0) for i in patient_images)
        before=sum(states['treatment']['exposures'].get(i,0) for i in patient_images)
        assert before==prior['patient_training_exposures']
        assert states['control']['exposures'].get(iid,0)==0
        assert effective==(0 if pid=='14393' else before)
        raw_path=TARGET/prior['raw_path'];row_path=raw_path.with_suffix('.json')
        assert digest(raw_path)==prior['raw_sha256'] and read(row_path)==prior
        sources.append(dict(image_id=iid,raw_path=str(raw_path.resolve()),raw_sha256=digest(raw_path),
            row_path=str(row_path.resolve()),row_sha256=digest(row_path)))
        metadata.append(dict(model='masked_control',branch='control',scenario='U',patient_id=pid,image_id=iid,
            eval_role=r['eval_role'],assignment_group=r['assignment_group'],record_role='U_observed',
            member=int(effective>0),image_member=0,actual_training_exposures=0,
            patient_training_exposures=effective,effective_patient_member=int(effective>0),
            effective_image_member=0,effective_image_exposures=0,effective_patient_exposures=effective,
            treatment_member=prior['member'],treatment_patient_training_exposures=before,
            pretrained_membership='unknown',membership_label_scope='NIH effective loss contributions only',
            checkpoint_sha256=digest(CHECKPOINTS['control'])))
    assert metadata[0]['member']==metadata[1]['member']==0
    return states,training_paths,evidence,protocol,metadata,sources,order


def prepare(path):
    assert not path.exists() and not OUTPUT.exists()
    states,training_paths,evidence,tp,rows,sources,order=source_and_training()
    frozen=dict(tp['contract']['frozen_sha256'])
    hashes_match(frozen)
    paths=[Path(__file__),ROOT/'u_patient_audit/run_masked_slot_endpoints_v2.py',
        ROOT/'u_patient_audit/run_cdi_base_selection.py',ROOT/'u_patient_audit/verify_masked_patient_training.py',
        *POLICIES,*training_paths,TARGET/'protocol.json',TARGET/'results.json',TARGET/'execution.json',TARGET/'verification.json']
    paths += [TARGET/'analysis_v1'/n for n in ['analysis.json','fit_parameters.json','cv_results.json',
        'predictions.json','provenance.json','contract_copy.json']]
    for s in sources:paths += [Path(s['raw_path']),Path(s['row_path'])]
    for p in paths:frozen[str(p.resolve())]=digest(p)
    c=dict(schema='cdi-masked-u82-contract/v1',current_step=2,created_utc=now(),
        model='masked_control',branch='control',scenario='U',eval_roles=['fit','selection'],
        patient_order=order,image_order=[r['image_id'] for r in rows],measurement_plan=rows,
        expected_patients=41,expected_records=82,expected_unique_images=82,
        master_seed=260914,stream='primary',batch_size=1,prediction_type='epsilon',dtype='float32',
        prompt=PROMPT,code_policy='released_code_literal',feature_names=FEATURE_NAMES,
        treatment_directory=str(TARGET),treatment_sources=sources,
        control_checkpoint=str(CHECKPOINTS['control']),checkpoint_paths={b:str(p) for b,p in CHECKPOINTS.items()},
        checkpoint_sha256={b:digest(p) for b,p in CHECKPOINTS.items()},
        training_evidence=evidence,training_verification=str(TRAINING/'verification.json'),
        cache_sha256=digest(RUN/'cache/cache.pt'),cohort_lock_sha256=digest(RUN/'cohort/lock.json'),
        source_bindings=source_bindings(),analysis_policies={str(p):digest(p) for p in POLICIES},
        output_directory=str(OUTPUT),frozen_sha256=frozen,
        logical_cost='Per image 51+q UNet forward,10+q epsilon-backward; q=actual NO objective evaluations',
        no_control_module_reuse=True,original_treatment_raw_reused_without_changes=True,
        perform_fitting=False,perform_performance_evaluation=False,new_target_training=False,
        no_new_patients=True,no_E_expansion=True,
        limits=['One fixed patient-contribution intervention; no population MIA or privacy guarantee',
            'Original treatment scorers may include this fit patient; separately frozen U79 analysis addresses that distinction',
            'Noise optimization and input gradients are the original CDI kernel, not new target training',
            'Model identity does not enter noise seeds; original image/module/draw primary stream is retained'])
    new_json(path,c);return c


def validate(c):
    assert c['schema']=='cdi-masked-u82-contract/v1' and c['expected_records']==82
    assert c['master_seed']==260914 and c['stream']=='primary' and c['feature_names']==FEATURE_NAMES
    assert c['dtype']=='float32' and c['prediction_type']=='epsilon' and c['batch_size']==1
    assert c['perform_fitting'] is c['perform_performance_evaluation'] is c['new_target_training'] is False
    assert Path(c['output_directory']).resolve()==OUTPUT.resolve()
    hashes_match(c['frozen_sha256'])
    assert c['frozen_sha256'][str(Path(__file__).resolve())]==digest(Path(__file__))
    states,_,evidence,tp,rows,sources,order=source_and_training()
    assert evidence==c['training_evidence'] and rows==c['measurement_plan']
    assert sources==c['treatment_sources'] and order==c['patient_order']
    assert [r['image_id'] for r in rows]==c['image_order']
    for b,p in CHECKPOINTS.items():assert digest(p)==c['checkpoint_sha256'][b]
    for key in ['master_seed','stream','batch_size','prediction_type','dtype','prompt','code_policy']:
        assert c[key]==tp['contract'][key]
    cache=load_cache();sched=scheduler();hidden=cache['hidden'][PROMPT].float()
    assert cache['audit_prompt']==PROMPT and sched.config.prediction_type=='epsilon'
    for s in sources:
        raw=torch.load(s['raw_path'],map_location='cpu',weights_only=True)
        assert raw['image_id']==s['image_id'] and raw['stream']=='primary'
        assert torch.equal(raw['latent'],cache['latents'][s['image_id']].float())
        assert torch.equal(raw['alphas'],sched.alphas_cumprod.float())
        assert raw['hidden_sha256']==tensor_digest(hidden)
    return states,cache,sched,hidden


def same_random_inputs(raw,old):
    assert torch.equal(raw['latent'],old['latent']) and torch.equal(raw['alphas'],old['alphas'])
    assert raw['hidden_sha256']==old['hidden_sha256'] and raw['stream']==old['stream']=='primary'
    assert raw['noise_draws'].keys()==old['noise_draws'].keys()
    for module,draws in raw['noise_draws'].items():
        assert len(draws)==len(old['noise_draws'][module])
        for a,b in zip(draws,old['noise_draws'][module]):
            for key in ['seed','sha256','draw']:assert a[key]==b[key]
            assert torch.equal(a['noise'],b['noise'])


def execute(args,c,states,cache,sched,hidden):
    OUTPUT.mkdir(parents=True,exist_ok=False)
    (OUTPUT/'control').mkdir()
    start=time.perf_counter();contract_hash=digest(args.contract)
    new_json(OUTPUT/'protocol.json',dict(schema='cdi-masked-u82-execution/v1',started_utc=now(),
        contract_path=str(args.contract.resolve()),contract_sha256=contract_hash,contract=c,
        source_bindings=source_bindings(),measurement_plan=c['measurement_plan'],
        environment={n:version(n) for n in ['torch','diffusers','peft','numpy','scipy','safetensors']}))
    rows=[];unet=adapter=None
    try:
        setup()
        unet=load_unet(CHECKPOINTS['control'],training=False).float()
        unet.enable_adapters();unet.eval().requires_grad_(False)
        loaded=adapter_state(unet);expected=states['control']['adapter']
        assert loaded.keys()==expected.keys() and all(torch.equal(loaded[k],expected[k].float()) for k in loaded)
        adapter=CDIAdapter(unet,sched,hidden,master_seed=260914)
        before=state_fingerprint(unet);versions={n:p._version for n,p in unet.named_parameters()}
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
        source={s['image_id']:s for s in c['treatment_sources']}
        for metadata in c['measurement_plan']:
            iid=metadata['image_id']
            measured=adapter.score(cache['latents'][iid],iid,stream='primary')
            raw=measured.pop('raw')
            old=torch.load(source[iid]['raw_path'],map_location='cpu',weights_only=True)
            same_random_inputs(raw,old)
            assert measured['features'] and measured['feature_names']==FEATURE_NAMES
            q=raw['no']['optimizer']['objective_evaluations']
            assert measured['forward']==51+q and measured['backward']==10+q
            path=OUTPUT/'control'/(Path(iid).stem+'.pt')
            assert not path.exists();torch.save(raw,path)
            for key in set(metadata)&set(measured):assert metadata[key]==measured[key]
            row={**measured,**metadata,'raw_path':path.relative_to(OUTPUT).as_posix(),
                'raw_sha256':digest(path),'contract_sha256':contract_hash,
                'cache_sha256':c['cache_sha256'],'cohort_lock_sha256':c['cohort_lock_sha256'],
                'treatment_source':source[iid],'source_noise_and_input_exact':True,
                'reused_record':False,'reused_from_kernel':False}
            new_json(path.with_suffix('.json'),row);rows.append(row)
            assert adapter.assert_weights_unchanged()
            del old,raw,measured
            if len(rows)%10==0 or len(rows)==82:
                progress=dict(event='masked_cdi_progress',records=len(rows),expected_records=82,
                    seconds=time.perf_counter()-start,forward=adapter.forward,backward=adapter.backward)
                write_json(OUTPUT/'progress.json',progress);print(json.dumps(progress),flush=True)
        assert len(rows)==82 and [r['image_id'] for r in rows]==c['image_order']
        assert adapter.assert_weights_unchanged()
        after=state_fingerprint(unet)
        assert before==after
        assert all(p._version==versions[n] and p.grad is None and not p.requires_grad for n,p in unet.named_parameters())
        final_adapter=adapter_state(unet)
        assert all(torch.equal(loaded[k],final_adapter[k]) for k in loaded)
        forward=sum(r['forward'] for r in rows);backward=sum(r['backward'] for r in rows)
        assert forward==adapter.forward and backward==adapter.backward
        assert forward-backward==82*41
        hashes_match(c['frozen_sha256']);assert digest(args.contract)==contract_hash
        for row in rows:assert digest(OUTPUT/row['raw_path'])==row['raw_sha256']
        new_json(OUTPUT/'results.json',rows)
        new_json(OUTPUT/'execution.json',dict(schema='cdi-masked-u82-result/v1',
            status='PASS_MASKED_CDI_U82_EXTRACTION_PENDING_INDEPENDENT_VERIFICATION',complete=True,
            current_step=2,records=82,unique_images=82,unique_patients=41,branch='control',model='masked_control',
            forward=forward,backward=backward,vae_forward=0,runner_seconds=time.perf_counter()-start,
            score_seconds=sum(r['seconds'] for r in rows),initial_state=before,final_state=after,
            all_parameters_frozen=True,no_parameter_gradients=True,parameter_versions_unchanged=True,
            actual_checkpoint_adapter_exact=True,adapter_tensors_unchanged=True,
            checkpoint_sha256=c['checkpoint_sha256']['control'],source_noise_and_input_exact_all82=True,
            NO_objective_evaluations=sum(r['modules']['noise_optim']['optimizer']['objective_evaluations'] for r in rows),
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
            results_sha256=digest(OUTPUT/'results.json'),protocol_sha256=digest(OUTPUT/'protocol.json'),
            contract_sha256=contract_hash,frozen_inputs_unchanged=True,
            new_target_training=False,perform_fitting=False,perform_performance_evaluation=False))
    except BaseException:
        new_json(OUTPUT/'failure.json',dict(status='FAILED_PRESERVED_MASKED_CDI_U82',
            traceback=traceback.format_exc(),completed_records=len(rows),seconds=time.perf_counter()-start))
        raise
    finally:
        adapter=unet=None;gc.collect()
        if torch.cuda.is_initialized():torch.cuda.empty_cache()


def main():
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--prepare-contract',action='store_true');g.add_argument('--dry-run',action='store_true');g.add_argument('--run',action='store_true')
    p.add_argument('--contract',type=Path,default=CONTRACT);p.add_argument('--expected-code-sha256')
    a=p.parse_args()
    if a.expected_code_sha256:assert digest(Path(__file__))==a.expected_code_sha256
    if a.prepare_contract:
        c=prepare(a.contract);print(json.dumps(dict(status='PASS_MASKED_CDI_U82_CONTRACT_PREPARED',
            contract_sha256=digest(a.contract),frozen_files=len(c['frozen_sha256']),CUDA_initialized=torch.cuda.is_initialized())));return
    c=read(a.contract);context=validate(c)
    if a.dry_run:
        assert not torch.cuda.is_initialized()
        print(json.dumps(dict(status='PASS_CPU_MASKED_CDI_U82_PREFLIGHT',records=82,patients=41,new_GPU_calls=0)));return
    execute(a,c,*context)


if __name__=='__main__':main()
