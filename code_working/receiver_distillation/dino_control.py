"""One authorized DINO-only control: verify changed selection, synthesize, seal, evaluate."""
import argparse
import difflib
import json
from pathlib import Path
import shutil
import time
import traceback
import torch
from torchvision.transforms import functional as TF
from prrd_v3.contracts import CODE, RESEARCH, read, save, sha, require, now, setup
from prrd_v3.datasets import manifests, PixelAccess
from prrd_v3.encoders import pil_tensor, state_hash
from prrd_v3.bank_runtime import atomic_json, verify_artifact
from .runtime import seed_all
from .schedule import ScheduledRenderer
from .second_source import ENCODER_ID, DinoEncoder, load_combined
from .signals import bind_objective, objective_digest
from .objectives import one_pass, two_pass
from .verify_public import fixture
from .run_feasibility import worker
from .dino_control_runtime import load_dino, code_bindings
from .dino_control_evaluate import dependencies, PREVIOUS

ROOT=CODE/'_reports/receiver_dino_only_s200_20260923_v1'
BANK_ID='dev_DINO_condk4_101_s200'


def initialize(root):
    old=read(PREVIOUS/'execution_bindings.json')
    require(all(sha(p)==h for p,h in old['runtime'].items()),'Previously verified runtime changed')
    reviewed=[]
    for name,expected in old['implementation'].items():
        p=Path(name)
        if sha(p)==expected:continue
        snapshot=PREVIOUS/'implementation_snapshot'/p.name
        require(p.name=='test_target_runtime.py' and expected=='92985e95598e09b05ed47539ef3c640eec3be2b75afa980cae5d4c3b5319f774'
                and sha(p)=='4c070cdf5b2835d9bde593999055dda08805339f3d85770a4633415c8c833e48'
                and sha(snapshot)==expected,'Unreviewed implementation change')
        reviewed.append({'file':str(p),'historical_sha256':expected,'current_sha256':sha(p),
            'diff':''.join(difflib.unified_diff(snapshot.read_text(encoding='utf-8').splitlines(True),
                        p.read_text(encoding='utf-8').splitlines(True),fromfile='frozen',tofile='current')),
            'assessment':'Two-line skip for missing optional test handoff; not imported by runtime or evaluation; preserved user change'})
    root.mkdir(exist_ok=False)
    before=root/'records_before'; before.mkdir()
    for p in [CODE.parent/n for n in ('AGENTS.md','CURRENT_STATUS.md','WORKLOG.md')]+[
            RESEARCH/n for n in ('research_state.json','RESEARCH_FRAMEWORK.md')]:
        shutil.copy2(p,before/p.name)
    paths=[PREVIOUS/'jobs/A2.json',PREVIOUS/'two_bank_seal.json',
           PREVIOUS/'changed_path_verification/result.json',PREVIOUS/'schedule_verification.json',
           PREVIOUS/'second_targets/result.json',PREVIOUS/'second_targets/private/target_handoff.json']
    for arm in ('A1','A2'):
        bank=PREVIOUS/'banks'/('dev_'+arm+'_condk4_101_s200')
        for name in ('run_spec.json','COMPLETED.json','artifact/artifact_seal.json'):paths.append(bank/name)
        spec=read(bank/'run_spec.json'); done=read(bank/'COMPLETED.json')
        require(done['completed_updates']==200,'Historical final not complete')
        verify_artifact(bank/'artifact',spec,done['signature'])
    a2bank=PREVIOUS/'banks/dev_A2_condk4_101_s200'
    receipts=list((a2bank/'checkpoints').glob('step_000000_*.pt.json'))
    require(len(receipts)==1,'Initial checkpoint ambiguity')
    paths.append(receipts[0])
    cp=read(receipts[0]); paths.append(a2bank/cp['path'])
    contract={
        'schema':'receiver.dino-only-control-campaign/v1','created':now(),
        'authorization':'Latest user: add one DINOv2-only 128-image,200-update,seed101 control, then compare cached A1/A2',
        'bank_id':BANK_ID,'images':128,'updates':200,'seed':101,'microbatch':16,
        'maximum_seconds':3600,'time_cap_scope':'cumulative synthesis worker including startup/save/export; check/evaluation separate',
        'estimated_minutes':{'synthesis':[35,45],'whole_task':[50,75]},
        'objective':'exact A2 DINO block at full single-source weight; K4 mean; prior once',
        'reuse':'A2 DINO checkpoint/P-only projection/head/conditions/PQ target; exact A2 templates,optimizer,schedule',
        'raw_pixel_scope':'public P for check and initialization only; no Q/V pixel extraction',
        'evaluation':'final200 PNG seal first; existing BioViL/DenseNet V features and 2000 patient draws',
        'comparison':'DINO-A1 and A2-DINO AUROC/AP; BioViL retention and synthesis cost; one fixed seed developmental',
        'decision':'No new superiority/equivalence threshold; no automatic followup from any result',
        'historical_dependencies':{str(p):sha(p) for p in paths},
        'historical_test_only_change_review':reviewed,
        'first_attempt':'Stopped before GPU and output creation on broad test-file hash check; narrowed to exact reviewed test-only diff; all runtime hashes unchanged',
        'legacy_runtime_bindings':old['runtime'],
        'reused_checks':'target preparation, schedule, optimizer resume and PNG infrastructure remain valid',
        'new_check':{'P_images':4,'conditions':4,'microbatch':4,'optimizer_updates':0,
                     'loss_atol':2e-6,'gradient_atol':2e-7,'gradient_rtol_peak':5e-4,
                     'test':'single DINO retained-graph vs production two-pass; exact A2 DINO objective selection'},
        'storage':'prune only this bank redundant payloads; keep step0,two latest/final and all receipts',
        'private_status':'NONDP_INTERNAL_Q_DERIVED; no DP release claim',
        'automatic_followup':False,'DP_Expert_Reserved_final_receiver':False,
        'free_bytes':shutil.disk_usage(CODE).free,
        'A2_initial_receipt':str(receipts[0]),
    }
    save(root/'campaign_contract.json',contract)
    state_path=RESEARCH/'research_state.json'; state=read(state_path)
    state['receiver_DINO_only_s200_control']={'status':'PREPARING','started':now(),
        'contract':str(root/'campaign_contract.json'),'bank_id':BANK_ID,'automatic_followup':False}
    state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'phase':'scope_recorded','contract_sha256':sha(root/'campaign_contract.json'),
                      'free_bytes':contract['free_bytes']}),flush=True)


def verify(root):
    out=root/'changed_path_verification'; out.mkdir(exist_ok=False)
    contract=read(root/'campaign_contract.json')
    save(out/'contract.json',dict(contract['new_check'],created=now(),
         source_bindings=code_bindings(),campaign_contract_sha256=sha(root/'campaign_contract.json')))
    setup(); seed_all(101); started=time.monotonic()
    job=read(PREVIOUS/'jobs/A2.json'); labels=[0,0,1,1]
    combined,rules=load_combined(job['a1_handoff'],job['second_handoff'],labels,'P','cuda')
    selected=bind_objective(combined.conditions,{ENCODER_ID:combined.heads[ENCODER_ID]},
                            {ENCODER_ID:combined.targets[ENCODER_ID]},labels)
    objective,own_rules=load_dino(job['second_handoff'],labels,'P','cuda')
    require(objective_digest(selected)==objective_digest(objective),'DINO objective differs from A2 block')
    require(own_rules[ENCODER_ID]==rules[ENCODER_ID],'DINO feature contract differs')
    groups=manifests(); access=PixelAccess(groups,('P',)).install()
    rows=fixture(groups['P'])
    templates=torch.stack([TF.resize(pil_tensor(access.image(r)),[224,224],antialias=True) for r in rows]).cuda()
    encoder=DinoEncoder().use_projection(own_rules[ENCODER_ID]['projection'])
    fixed=state_hash(encoder); frozen=objective_digest(objective)
    counts={'forward':0,'backward':0}
    def hook(module,inputs,output):
        n=len(inputs[0]);counts['forward']+=n
        if output.requires_grad:
            def backward(g):counts['backward']+=n;return g
            output.register_hook(backward)
    handle=encoder.register_forward_hook(hook)
    renderer=ScheduledRenderer(templates,'B',101); renderer.set_step(200)
    reference=one_pass(renderer,{ENCODER_ID:encoder},objective,4)
    require(torch.equal(reference['matching'],reference['blocks'][ENCODER_ID]),'Single source is not full weight')
    reference['loss'].backward()
    ref=[p.grad.detach().clone() for p in renderer.parameters() if p.requires_grad]
    ref_loss=float(reference['loss'].detach().cpu()); del reference
    actual=two_pass(renderer,{ENCODER_ID:encoder},objective,4)
    got=[p.grad.detach().clone() for p in renderer.parameters() if p.requires_grad]
    err=max(float((a-b).abs().max()) for a,b in zip(ref,got)); peak=max(float(a.abs().max()) for a in ref)
    relative=float(torch.sqrt(sum((a-b).square().sum() for a,b in zip(ref,got)))/
                   torch.sqrt(sum(a.square().sum() for a in ref)).clamp_min(1e-30))
    loss_err=abs(ref_loss-actual['loss'])
    require(loss_err<=2e-6 and err<=2e-7+5e-4*peak,'DINO-only gradient mismatch')
    require(peak>0 and all(torch.isfinite(g).all() for g in got),'Missing/invalid gradients')
    require(set(actual['blocks'])=={ENCODER_ID} and actual['matching']==actual['blocks'][ENCODER_ID],
            'Wrong source weights')
    handle.remove()
    require(counts=={'forward':48,'backward':32},'Unexpected check compute')
    require(state_hash(encoder)==fixed and objective_digest(objective)==frozen,'Fixed model/target changed')
    require(all(not p.requires_grad and p.grad is None for p in encoder.parameters()),'Encoder unfrozen')
    result={'status':'PASS_DINO_ONLY_SELECTION_AND_ACTUAL_GRADIENT',
            'loss_abs':loss_err,'gradient_max_abs':err,'gradient_relative_L2':relative,
            'reference_peak':peak,'A2_DINO_objective_exact':True,'full_single_source_weight':True,
            'only_model_executed':ENCODER_ID,'prior_once':True,'counts':counts,'optimizer_updates':0,
            'models_and_targets_fixed':True,'access':access.report(),
            'seconds':time.monotonic()-started}
    save(out/'result.json',result); print(json.dumps(result),flush=True)


def bind(root):
    require(read(root/'changed_path_verification/result.json')['status']==
            'PASS_DINO_ONLY_SELECTION_AND_ACTUAL_GRADIENT','Missing changed-path verification')
    old=read(PREVIOUS/'jobs/A2.json'); contract=read(root/'campaign_contract.json')
    job=dict(old,schema='receiver.dino-only-s200-bank/v1',id=BANK_ID,maximum_seconds=3600,
             campaign_contract=str(root/'campaign_contract.json'),
             campaign_contract_sha256=sha(root/'campaign_contract.json'),
             code_bindings=code_bindings(),
             A2_job=str(PREVIOUS/'jobs/A2.json'),
             A2_bank=str(PREVIOUS/'banks/dev_A2_condk4_101_s200'),
             A2_initial_receipt=contract['A2_initial_receipt'],
             A2_template_tensor_digest=read(PREVIOUS/'banks/dev_A2_condk4_101_s200/run_spec.json')['template_tensor_digest'])
    job.pop('a1_handoff');job.pop('a1_handoff_sha256')
    save(root/'job.json',job)
    sources={str(p):sha(p) for p in Path(__file__).parent.glob('*.py')}
    bindings={'created':now(),'job':str(root/'job.json'),'job_sha256':sha(root/'job.json'),
              'campaign_contract_sha256':sha(root/'campaign_contract.json'),
              'implementation':sources,'evaluation_dependencies':dependencies(),'runtime':code_bindings()}
    save(root/'execution_bindings.json',bindings)
    snapshot=root/'implementation_snapshot';snapshot.mkdir()
    for p in Path(__file__).parent.glob('*.py'):shutil.copy2(p,snapshot/p.name)
    return bindings


def run(root,resume=False):
    started=time.monotonic()
    try:
        if not resume:
            initialize(root)
            worker(root,'verify_DINO',['-m','receiver_distillation.dino_control','--phase','verify','--root',str(root)])
            binding=bind(root)
        else:binding=read(root/'execution_bindings.json')
        require(all(sha(p)==h for p,h in binding['implementation'].items()),'Implementation changed')
        require(sha(binding['job'])==binding['job_sha256'],'Bound job changed')
        result_path=root/'DINO_result.json';bank=root/'banks'/BANK_ID
        if not result_path.exists():
            args=['-m','receiver_distillation.dino_control_runtime','--job',binding['job'],
                  '--output',str(bank),'--result',str(result_path)]
            if bank.exists():args.append('--resume')
            worker(root,'DINO_main',args)
        result=read(result_path)
        require(result['status']=='COMPLETE' and result['completed_updates']==200,'Bank paused/incomplete; no evaluation')
        done=read(bank/'COMPLETED.json');spec=read(bank/'run_spec.json')
        verify_artifact(bank/'artifact',spec,done['signature'])
        if not(root/'bank_seal.json').exists():
            save(root/'bank_seal.json',{'created':now(),'bank_id':BANK_ID,
                 'completion_sha256':sha(bank/'COMPLETED.json'),'artifact_sha256':sha(bank/'artifact/artifact_seal.json'),
                 'DINO_result_sha256':sha(result_path),'historical_two_bank_seal_sha256':sha(PREVIOUS/'two_bank_seal.json')})
        if not(root/'evaluation/result.json').exists():
            worker(root,'evaluate_DINO',['-m','receiver_distillation.dino_control_evaluate','--root',str(root)])
        evaluation=read(root/'evaluation/result.json')
        contract=read(root/'campaign_contract.json')
        require(all(sha(p)==h for p,h in contract['historical_dependencies'].items()),'Historical evidence changed')
        require(all(sha(p)==h for p,h in contract['legacy_runtime_bindings'].items()),'Old runtime changed')
        finish={'status':'COMPLETE','finished':now(),'controller_seconds':time.monotonic()-started,
                'DINO_result_sha256':sha(result_path),'evaluation_sha256':sha(root/'evaluation/result.json'),
                'historical_evidence_unchanged':True,'free_bytes':shutil.disk_usage(CODE).free,'automatic_followup':False}
        save(root/'campaign_result.json',finish)
        atomic_json(root/'campaign_status.json',finish,replace=True)
        print(json.dumps(finish),flush=True)
    except Exception as exc:
        if root.exists():
            atomic_json(root/'campaign_status.json',{'status':'STOPPED','updated':now(),'error':str(exc)},replace=True)
        traceback.print_exc();raise


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=ROOT)
    ap.add_argument('--phase',choices=['run','verify'],default='run');ap.add_argument('--resume',action='store_true')
    args=ap.parse_args()
    if args.phase=='verify':verify(args.root)
    else:run(args.root,args.resume)


if __name__=='__main__':main()
