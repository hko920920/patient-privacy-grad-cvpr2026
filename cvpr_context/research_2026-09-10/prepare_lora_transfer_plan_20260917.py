"""Saved-output diagnostics and metadata-only LoRA comparison specification."""
import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
THESIS = ROOT.parents[1]
CODE = THESIS / 'code_working'
OLD = CODE / '_reports/downstream_development_20260917_v1'
OUT = CODE / '_reports/lora_transfer_plan_20260917_v1'
REVIEW = ROOT / 'spec_sources/lora_transfer_stored_review_20260917.json'
PLAN = ROOT / 'spec_sources/lora_transfer_comparison_plan_20260917.json'
REPORT = ROOT / 'TRACK1_LORA_TRANSFER_COMPARISON_PROTOCOL_20260917.md'
ARMS = ['R0','R1','S0','S1','S2','S3','Dreal']
SEEDS = [11,23,37]
SALT = 'lora-transfer-comparison-20260917-v1'


def require(value, message):
    if not value:
        raise RuntimeError(message)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    with path.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')


def key(*parts):
    return hashlib.sha256((SALT+'|'+'|'.join(map(str,parts))).encode()).hexdigest()


def integer(*parts):
    return int(key(*parts)[:15],16)


def diagnostics():
    require(not REVIEW.exists(),'Preserve prior review')
    sources={}; runs={}; curves=[]
    scores=read(OLD/'result.json')['scores']
    for arm in ARMS:
        for seed in SEEDS:
            name=f'{arm}_{seed}'
            paths=[OLD/'runs'/name/'trace.json',OLD/'runs'/name/'exposure.json',OLD/'evaluation'/(name+'.npz')]
            for p in paths:sources[str(p.relative_to(THESIS))]=sha(p)
            tr=read(paths[0]); ex=read(paths[1]); losses=np.asarray([row['loss'] for row in tr['trace']])
            require(len(losses)==400 and np.isfinite(losses).all(),'400 finite saved losses')
            with np.load(paths[2],allow_pickle=False) as packet:
                z=packet['logits'].copy(); y=packet['labels'].copy()
            require(len(z)==5047 and int(y.sum())==223 and np.isfinite(z).all(),'Development-only predictions')
            probabilities=np.exp(-np.logaddexp(0,-z))
            dist={}
            for label in [0,1]:
                v=probabilities[y==label]
                dist[str(label)]=dict(n=len(v),mean=float(v.mean()),q10_q50_q90=np.quantile(v,[.1,.5,.9]).tolist())
            positive=[int(row['exposures']) for row in ex['per_image'] if row['namespace']=='real/public' and int(row['label'])==1]
            require(len(positive)==6,'Same six public positive images')
            blocks=[float(losses[i:i+50].mean()) for i in range(0,400,50)]
            runs[name]=dict(loss_first50=blocks[0],loss_last50=blocks[-1],loss_blocks50=blocks,
                 positive_public_image_exposure=dict(min=min(positive),median=float(np.median(positive)),max=max(positive),total=sum(positive)),
                 saved_development_probability_by_weak_label=dist,
                 development_fraction_below001_or_above099=float(np.mean((probabilities<.01)|(probabilities>.99))))
            curves += [dict(arm=arm,seed=seed,step_end=(i+1)*50,mean_training_BCE=value) for i,value in enumerate(blocks)]
    summary={}
    for arm in ARMS:
        rr=[runs[f'{arm}_{s}'] for s in SEEDS]
        summary[arm]=dict(loss_first50_mean=float(np.mean([v['loss_first50'] for v in rr])),
             loss_last50_mean=float(np.mean([v['loss_last50'] for v in rr])),
             positive_dev_probability_median_across_seeds=float(np.median([v['saved_development_probability_by_weak_label']['1']['q10_q50_q90'][1] for v in rr])),
             negative_dev_probability_median_across_seeds=float(np.median([v['saved_development_probability_by_weak_label']['0']['q10_q50_q90'][1] for v in rr])),
             public_positive_exposure_min=min(v['positive_public_image_exposure']['min'] for v in rr),
             public_positive_exposure_max=max(v['positive_public_image_exposure']['max'] for v in rr))
    calibration=read(OLD/'calibration_choice.json')
    calibration_review=[]
    for lr in [1e-4,3e-4]:
        p=OLD/'calibration'/f'R1_lr{lr}'/'trace.json'; sources[str(p.relative_to(THESIS))]=sha(p)
        tr=read(p)['trace']
        candidates={v['steps']:v for v in calibration['candidates'] if v['lr']==lr}
        calibration_review.append(dict(lr=lr,training_BCE_step351_400=float(np.mean([v['loss'] for v in tr[350:400]])),
             training_BCE_step751_800=float(np.mean([v['loss'] for v in tr[750:800]])),
             selection_AUROC_400=candidates[400]['AUROC'],selection_AUROC_800=candidates[800]['AUROC']))
    for p in [OLD/'result.json',OLD/'verification.json',OLD/'calibration_choice.json']:
        sources[str(p.relative_to(THESIS))]=sha(p)
    result=dict(status='STORED_OUTPUT_REVIEW_ONLY',created_utc=datetime.now(timezone.utc).isoformat(),
                summary=summary,runs=runs,calibration=calibration_review,
                scope='Training losses and existing weak-development predictions; no model rerun, new patient evaluation, threshold selection, or causal overfitting diagnosis.',
                new_model_forwards=0,new_training=0,new_generation=0,new_patient_pixels=0,source_sha256=sources)
    save(REVIEW,result)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,3.5),sharey=True)
    for ax,group,title in zip(axes,[['R0','R1','Dreal'],['S0','S1','S2','S3']],['Real-source arms','Synthetic replacement arms']):
        for arm in group:
            yy=np.asarray([runs[f'{arm}_{seed}']['loss_blocks50'] for seed in SEEDS])
            line=ax.plot(np.arange(50,401,50),yy.mean(0),label=arm)[0]
            ax.fill_between(np.arange(50,401,50),yy.min(0),yy.max(0),color=line.get_color(),alpha=.12)
        ax.set(title=title,xlabel='Optimizer step',ylabel='Training BCE (50-step mean)')
        ax.legend();ax.grid(alpha=.2)
    fig.suptitle('Saved training traces; shading = range of 3 seeds, not a confidence interval',fontsize=10)
    fig.tight_layout();fig.savefig(ROOT/'spec_sources/lora_transfer_saved_loss_20260917.png',dpi=160);plt.close(fig)
    print(json.dumps(dict(summary=summary,calibration=calibration_review),indent=2))


def freeze():
    require(not OUT.exists() and not PLAN.exists(),'No plan overwrite')
    require(REPORT.exists() and REVIEW.exists(),'Protocol and review required')
    old=read(OLD/'contract.json'); metadata_path=CODE/'_reports/medical_head_20260916_v1/images.json'
    image_rows=[x for x in read(metadata_path) if x['split'] in ['public','train']]
    groups={'public':{},'train':{}}
    for row in image_rows:
        groups[row['split']].setdefault(row['patient_id'],[]).append(row)
    require([len(groups[k]) for k in ['public','train']]==[32,80],'Head source patients')
    for sp,n in [('public',2),('train',4)]:
        require(all(len(v)==n for v in groups[sp].values()),'Images per patient')
    require(not(set(groups['public'])&set(groups['train'])),'Patient separation')
    allowed_path=CODE/'_reports/downstream_profile_20260917_v3/real_manifest_private.csv'
    with allowed_path.open(encoding='utf-8-sig',newline='') as f: allowed={x['image_id']:x for x in csv.DictReader(f)}
    for row in image_rows:
        a=allowed[row['image_id']]
        require(int(a['patient_id'])==int(row['patient_id']) and a['role']==('public' if row['split']=='public' else 'private'),'Same patient/source role')
        row['weak_P']=int(a['label'])
        require(a['sha256']==row['sha256'],'Existing image hash binding')
    ordered={sp:{p:sorted(v,key=lambda x:key('image-order',x['image_id'])) for p,v in group.items()} for sp,group in groups.items()}
    rows=[]; extra_cursor=0; public_sequence=[]; pooled_sequence=[]
    for cycle in range(16):
        # Slot layout uses public fixed counts, never private IDs, labels or outcomes.
        layout=sorted([('public',i) for i in range(32)]+[('train',i) for i in range(80)],key=lambda x:key('role-layout',cycle,*x))
        order={sp:sorted(group,key=lambda p:key('patient-order',cycle,p)) for sp,group in groups.items()}
        for sp,rank in layout:
            patient=order[sp][rank]; pool_row=ordered[sp][patient][cycle%len(ordered[sp][patient])]
            if sp=='public':pub_row=pool_row
            else:
                extra_cycle=extra_cursor//32;extra_rank=extra_cursor%32
                extra_order=sorted(groups['public'],key=lambda p:key('public-extra-order',extra_cycle,p))
                p=extra_order[extra_rank];pub_row=ordered['public'][p][extra_cycle%2];extra_cursor+=1
            pooled_sequence.append(pool_row);public_sequence.append(pub_row)
    require(extra_cursor==1280 and len(public_sequence)==len(pooled_sequence)==1792,'448 batches of four')
    for arm,sequence in [('L_public',public_sequence),('L_pooled',pooled_sequence)]:
        for i,row in enumerate(sequence):
            step=i//4+1; slot=i%4
            rows.append(dict(arm=arm,step=step,slot=slot,patient_id=row['patient_id'],image_id=row['image_id'],
                             role='public' if row['split']=='public' else 'private',weak_P=row['weak_P'],
                             timestep=integer('timestep',step,slot)%1000,noise_seed=integer('noise-batch',step)))
    exposure={}
    for arm in ['L_public','L_pooled']:
        subset=[x for x in rows if x['arm']==arm];images=Counter(x['image_id'] for x in subset);patients=Counter(x['patient_id'] for x in subset)
        by_role={sp:Counter(x['image_id'] for x in subset if x['role']==sp) for sp in ['public','private']}
        exposure[arm]=dict(presentations=len(subset),unique_patients=len(patients),unique_images=len(images),
             patient_exposures=sorted(set(patients.values())),weak_P_presentations=sum(x['weak_P'] for x in subset),
             public_image_exposures=sorted(set(by_role['public'].values())),private_image_exposures=sorted(set(by_role['private'].values())))
    require(exposure['L_public']['patient_exposures']==[56] and exposure['L_public']['public_image_exposures']==[28],'Public exposure rule')
    require(exposure['L_pooled']['patient_exposures']==[16] and exposure['L_pooled']['public_image_exposures']==[8] and exposure['L_pooled']['private_image_exposures']==[4],'Pooled patient objective')
    require(exposure['L_public']['weak_P_presentations']==28 and exposure['L_pooled']['weak_P_presentations']==140,'Label exposure accounting')
    OUT.mkdir()
    save(OUT/'training_images_private.json',image_rows)
    with (OUT/'training_schedule_private.csv').open('x',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    save(OUT/'expected_exposure.json',exposure)
    checkpoint=Path(old['checkpoint']);require(sha(checkpoint)=='7bc5ce421cc02091e8c3812676e9c27dc58330afce5e9384abd8edc9f61a3b91','E4 checkpoint binding')
    sources=[REPORT,REVIEW,ROOT/'spec_sources/lora_transfer_saved_loss_20260917.png',Path(__file__),
             metadata_path,allowed_path,checkpoint,OLD/'contract.json',OLD/'calibration_choice.json',OLD/'result.json',
             OLD/'verification.json',OLD/'generation_inputs.pt',OLD/'generation_manifest.json',
             ROOT/'TRACK1_CURRENT_HEAD_BRANCH_CLOSURE_20260917.md',ROOT/'pdfs/B08.pdf',
             OUT/'training_images_private.json',OUT/'training_schedule_private.csv',OUT/'expected_exposure.json']
    sources += [CODE/p for p in ['u_patient_audit/models.py','public_medical_backbone/run_pilot.py','downstream_utility/data_v2.py',
                                'downstream_utility/train_v2.py','downstream_utility/development_v1.py','frozen_residual_head/run_lora_positive_control.py']]
    plan=dict(schema='lora-private-transfer-comparison/v1',status='FROZEN_DESIGN_NOT_EXECUTED',created_utc=datetime.now(timezone.utc).isoformat(),
       question='Can an existing LoRA adaptation transmit incremental private-role downstream utility under the fixed development conditions?',
       current_full64_branch_closed=True,new_method_claim=False,DP_allowed=False,final_ready=False,expert_allowed=False,reserved_allowed=False,
       arms=['L_public','L_pooled'],starting_checkpoint_sha256=sha(checkpoint),
       adaptation=dict(kind='Continue existing E4 LoRA A/B with fresh optimizer; no merge, no new adapter, no head',
            rank=8,alpha=8,target_modules=['to_q','to_k','to_v','to_out.0'],trainable_parameters=1659904,
            scheduler_prediction_type='epsilon_runtime_verified',objective='Conditional epsilon MSE; uniform patient then image; not guided-head regression',
            conditioning_dropout=0.,training_resolution=256,batch_size=4,committed_steps_per_arm=448,
            total_committed_steps=896,checkpoint_primary=448,diagnostic_checkpoint_only=224,
            checkpoint_selection='Fixed terminal step448 in both arms; no metric-based checkpoint selection or fallback',
            optimizer=dict(name='AdamW',lr=1e-4,betas=[.9,.999],eps=1e-8,weight_decay=.01,foreach=False,fused=False),
            gradient_norm_cap=1.,gradient_accumulation=1,precision='FP16 frozen base and autocast; FP32 trainable LoRA and loss',
            scaler_initial=1024.,max_skipped_attempts_per_arm=16,training_repeats=1,training_schedule_salt=SALT,
            cache='Rebuild selected384 only with fixed256 preprocessing/VAE mode/scaling/text prompts; split public64/private320 caches; do not load broad historical mixed cache'),
       expected_exposure=exposure,pooled_patient_mass=dict(public=32/112,private=80/112),
       generation=dict(new_images=256,images_per_arm=128,positive=64,negative=64,negative_prompts=dict(normal=22,effusion=21,cardiomegaly=21),
            cells=old['cells'],existing_actual_initials_sha256=sha(OLD/'generation_inputs.pt'),
            CFG=7.5,steps=30,eta=0.,precision='FP32',resolution=256,reuse_is_development_not_independent_confirmation=True,
            branches='Both conditional and unconditional UNet branches use the adapted LoRA; no full64 head'),
       classifier=dict(new_runs=6,seeds=SEEDS,architecture='ImageNet ResNet18',lr=1e-4,steps=400,batch=32,
            weight_decay=1e-4,real_public=16,synthetic=16,positive_per_half=8,negative_per_half=8,
            reuse_initial_states_real_draws_and_synthetic_cells=True,recalibration=False,
            sampling='Within each requested class patient/cell uniform then image uniform',
            new_source_namespaces=['real/public','synthetic/lora_public','synthetic/lora_pooled']),
       evaluation=dict(role='method_development',patients=2026,images=5047,weak_positive_images=223,
            seed_metrics_then_mean=True,all_six_runs_before_outcomes=True,
            primary='L_pooled minus L_public',gate_comparators=['L_public','R1','R0'],
            mean_AUROC_min_delta=.01,positive_seed_minimum=2,mean_AP_min_delta=0.,
            gate_purpose='Development investment heuristic only; no automatic DP or final',
            reference_arms=ARMS,cluster_bootstrap=2000,bootstrap_seed=integer('paired-cluster-bootstrap'),
            descriptive_comparisons=['L_pooled-L_public','L_pooled-R1','L_pooled-R0','L_pooled-S0','L_public-S1','L_pooled-S3',
                                     '(L_pooled-L_public)-(S3-S1)'],
            causal_head_capacity_identification=False),
       execution_readiness=dict(runner_implemented=False,patient_sampler_runtime_verified=False,E4_continuation_runtime_verified=False,
             downstream_new_source_binding_verified=False,full_execution_started=False,
             required_preflight='New runner, source guards and per-phase allowlists, E4 initializer parity, patient schedule/exposure check, new batch-provider with unchanged classifier math replay; source SHA freeze before models'),
       future_budget=dict(generator_optimizer_successes=896,generation_primary_images=256,classifier_optimizer_successes=2400,
            classifier_preflight_update_cap=8,extra_generation_decode_cap=2,
            implementation_and_full_execution_estimate_minutes=[65,100],estimate_is_not_measurement=True),
       new_model_forwards=0,new_training=0,new_generation=0,new_patient_pixels=0,
       source_sha256={str(p.relative_to(THESIS)):sha(p) for p in sources})
    save(PLAN,plan)
    print(json.dumps(dict(status=plan['status'],exposure=exposure,new_models=0),indent=2))


if __name__=='__main__':
    require(len(sys.argv)==2 and sys.argv[1] in ['diagnostics','freeze'],'Use diagnostics or freeze')
    globals()[sys.argv[1]]()
