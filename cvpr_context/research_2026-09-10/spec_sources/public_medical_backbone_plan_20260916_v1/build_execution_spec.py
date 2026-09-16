"""Build a prospective CPU-only schedule and generation-input specification."""
from pathlib import Path
from collections import Counter
import csv, hashlib, json

HERE=Path(__file__).resolve().parent
SALT='public-medical-backbone-execution-20260916-v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def key(*parts):return hashlib.sha256('|'.join(map(str,(SALT,)+parts)).encode()).hexdigest()
def seed(*parts):return int(key(*parts)[:15],16)
def read_csv(p):
    with p.open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def write_csv(p,rows):
    with p.open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def write_json(p,data):
    with p.open('x',encoding='utf-8') as f:json.dump(data,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
rows=read_csv(HERE/'images.csv');patients=read_csv(HERE/'patients.csv')
train=sorted((r for r in rows if r['backbone_role']=='train'),key=lambda r:r['image_id'])
n=len(train)
if n!=749 or len({r['patient_id'] for r in train})!=640:raise RuntimeError('Unexpected finalized public training cohort')
sequence=[]
for epoch in range(8):
    for row in sorted(train,key=lambda r:key('epoch',epoch,r['image_id'])):
        pos=len(sequence);step=pos//4
        sequence.append(dict(step=step+1,batch_position=pos%4,epoch=epoch+1,image_id=row['image_id'],patient_id=row['patient_id'],
            noise_seed=seed('noise',step),timestep=seed('timestep',step,pos%4)%1000))
if len(sequence)!=8*n or len(sequence)%4:raise RuntimeError('Exposure stream mismatch')
checkpoints=[]
for exposure in (4,8):
    prefix=sequence[:exposure*n];counts=Counter(r['image_id'] for r in prefix)
    if len(counts)!=n or set(counts.values())!={exposure}:raise RuntimeError('Per-image exposure error')
    checkpoints.append(dict(name='E'+str(exposure),successful_optimizer_steps=exposure*n//4,
        planned_image_presentations=exposure*n,per_image_committed_exposure=exposure))
write_csv(HERE/'training_schedule.csv',sequence)
prompts=[('generic','a frontal chest radiograph'),
    ('normal','a frontal posteroanterior chest radiograph with no labeled finding'),
    ('effusion','a frontal posteroanterior chest radiograph with radiographic findings of pleural effusion'),
    ('cardiomegaly','a frontal posteroanterior chest radiograph with radiographic findings of cardiomegaly')]
tasks=[]
for phase in ('selection','confirmation'):
    for pid,prompt in prompts:
        for j in range(4):
            tasks.append(dict(task_id=f'{phase}_{pid}_{j}',phase=phase,prompt_id=pid,prompt=prompt,
                seed=seed('generation',phase,pid,j),generator_device='cpu',initial_dtype='float32',
                guidance=1.,steps=30,width=256,height=256))
if len({r['seed'] for r in tasks})!=32:raise RuntimeError('Generation seeds collide')
write_csv(HERE/'generation_tasks.csv',tasks)
history=[r for r in patients if r['known_diagnostic_history']=='1']
if any(r['backbone_role'] in ('selection','confirmation') for r in history):raise RuntimeError('Known diagnostic history in reserved evaluation cohort')
spec=dict(schema='public-medical-backbone-execution-plan/v1',status='PLANNED_NOT_EXECUTED',
    current_research_step=2,track=1,one_new_training_run=True,
    data=dict(train_patients=640,train_images=n,selection_patients=128,selection_images=128,
        confirmation_patients=128,confirmation_images=128,reserve_patients=9,reserve_images=11,
        source='existing NIH PA public_development; same-source controlled simulation',
        selection_confirmation_patient_cohorts='Reserved for later real-reference/held-out evaluation; first gate assesses generated inputs, not these patients.',
        protected_and_old_auxiliary_patient_overlap=0,known_diagnostic_training_patients=len([r for r in history if r['backbone_role']=='train'])),
    model=dict(model_id='Manojb/stable-diffusion-2-1-base',revision='0094d483a120f3f33dafbd187ea4aa60d10de75c',
        fresh_initialization=True,private_M1_M2_loaded=False,LoRA_rank=8,LoRA_alpha=8,targets=['to_q','to_k','to_v','to_out.0'],
        frozen_base_dtype='float16',trainable_LoRA_dtype='float32',training_autocast='float16',init_seed=260914,
        VAE_text_encoder_frozen=True),
    training=dict(batch=4,learning_rate=1e-4,optimizer='AdamW',betas=[.9,.999],eps=1e-8,weight_decay=.01,
        gradient_norm_cap=1.,initial_gradient_scale=1024.,foreach=False,fused=False,conditional_dropout=0.,
        cache='New public train-only VAE latent mode x scaling factor and weak-label text embeddings; no old mixed-role cache load.',
        sampler='Concatenate 8 independent epoch permutations ranked by SHA256 of fixed salt,epoch,image_id; batch globally by4 across boundaries.',
        sampler_seed_rule=SALT,steps=2*n,checkpoints=checkpoints,
        objective='mean epsilon-prediction MSE',scheduler='read exact frozen SD2.1 config; require epsilon prediction',
        overflow_policy='Retry same successful step/noise/timesteps after scaler backoff; count every attempted batch, forward/backward and image presentation. Abort after100 failed attempts in total.',
        exposure_interpretation='4/8 are committed successful-update exposures; actual attempted exposures and wall time include overflow retries.',
        image_uniform_not_patient_uniform=True),
    generation=dict(primary_precision='FP32 no autocast',guidance=1.,scheduler='DDIM',steps=30,eta=0.,width=256,height=256,
        conditioning='one frozen cached embedding per fixed prompt shared by checkpoints',
        initial_noise='CPU FP32 fixed seed; save actual initial tensors before generation and share across checkpoints',
        old_bridge_inputs='Two old fixed inputs/checkpoint diagnostic only; excluded from checkpoint selection and confirmation counts.',
        planned_selection_images=32,conditional_confirmation_images=16,diagnostic_images=4,max_images=52,
        max_UNet_forward_calls=1560,new_small_head_applied=False),
    gate=dict(unit='generated image/input; not a patient or clinical endpoint',
        per_image_pass='frontal chest-radiograph-like overall structure and no gross nonmedical pattern, severe rotation/superposition or gross structural distortion; nonfinite is failure',
        uncertain_or_reviewer_disagreement='fail',review='two assistant reviewers, checkpoint labels blinded and task order fixed by hash; not clinical experts',
        selection='Both checkpoints use same16 new tasks. Pass requires >=12/16 overall AND>=2/4 within each prompt. Choose earliest passing E4 then E8; never rank by maximum pass count.',
        confirmation='Selected checkpoint only on16 disjoint new seeds, same thresholds, exactly once. Failure does not permit switching checkpoint or tuning on the same confirmation.',
        thresholds='Prospective heuristic resource-allocation gate, not a statistically validated quality threshold.',
        no_pass='Report all outcomes; no head fit, DP search or automatic training-budget expansion.',
        public_reference_patients_used_in_first_gate=False),
    next_after_pass='Freeze chosen public checkpoint. Re-extract backbone-dependent features/predictions/residuals/A/B/Q and fit fresh W0/public-only/pooled non-DP heads. Test private incremental generation utility before DP.',
    projection_policy='The existing public seeded Gaussian-QR P may remain fixed if the320-channel tap and feature convention are unchanged. It is not a learned backbone statistic; no projection-seed search.',
    evidence_limit='No generated efficacy, training convergence, clinical utility, private incremental value or DP advantage established by this plan.',
    cost=dict(reference_1000_steps_seconds=496.731,linear_training_estimate_seconds=2*n*.496731,
        training_estimate_minutes=[10,16],cache_estimate_minutes=[1,3],generation_estimate_minutes=[2,4],
        verification_and_recording_minutes=[10,20],new_runner_implementation_review_minutes=[25,40],
        next_package_estimate_minutes=[50,85],all_estimates_not_new_profiles=True,
        shared_public_training_cost='Report once and include in from-scratch totals; separately show amortized private-adaptation cost for all baselines using same public checkpoint.'),
    input_sha256={name:sha(HERE/name) for name in ('patients.csv','images.csv','diagnostic_history.csv','exclusion_calculation.json','verification.json')},
    output_sha256={name:sha(HERE/name) for name in ('training_schedule.csv','generation_tasks.csv')},
    builder_sha256=sha(__file__),new_GPU_runs=0)
write_json(HERE/'execution_spec.json',spec)
print(json.dumps(dict(status=spec['status'],train_images=n,checkpoints=checkpoints,training_seconds_estimate=spec['cost']['linear_training_estimate_seconds'],max_images=52)))
