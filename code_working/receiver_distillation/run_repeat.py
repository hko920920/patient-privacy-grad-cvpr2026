"""Bounded seed202 replication: DINO then A2, both sealed before evaluation."""
import argparse,json,shutil,time,traceback,tempfile
from pathlib import Path
import torch
from prrd_v3.contracts import CODE,OUT,RESEARCH,read,save,sha,require,now
from prrd_v3.datasets import manifests,public_templates
from prrd_v3.bank_runtime import atomic_json,digest,tree_digest,verify_artifact,save_checkpoint,load_checkpoint
from .runtime import OPTIMIZER,seed_all
from .schedule import ScheduledRenderer,ACTIVATIONS
from .signals import objective_digest
from .second_source import load_combined
from .dino_control_runtime import load_dino
from .repeat_runtime import code_bindings,initialize_parameters,initial_fingerprint,load_objective
from .run_feasibility import worker
from .evaluate_repeat import dependencies,OLD,FIRST

ROOT=CODE/'_reports/receiver_repeat202_s200_20260923_v1'
IDS=['dev_DINO_condk4_202_s200','dev_A2_condk4_202_s200']


def initialize(root):
    old_binding=read(OLD/'execution_bindings.json')
    require(all(sha(p)==h for p,h in old_binding['implementation'].items()),'Previously verified implementation changed')
    root.mkdir(exist_ok=False);before=root/'records_before';before.mkdir()
    for p in [CODE.parent/n for n in ('AGENTS.md','CURRENT_STATUS.md','WORKLOG.md')]+[
            RESEARCH/n for n in ('research_state.json','RESEARCH_FRAMEWORK.md')]:
        shutil.copy2(p,before/p.name)
    groups=manifests();templates=public_templates(groups['P'],202)
    normalized=[{'image_id':r['image_id'],'sha256':r['sha256']} for r in templates]
    old_public=read(OUT/'templates_P_private.json')
    require('202' in old_public,'Existing seed202 public template list missing')
    require(normalized==[{'image_id':r['image_id'],'sha256':r['sha256']} for r in old_public['202']],
            'Existing public template policy changed')
    job=read(FIRST/'jobs/A2.json')
    require(normalized!=job['templates'],'Template initialization did not change')
    paths=[FIRST/'jobs/A2.json',OLD/'job.json',OLD/'campaign_result.json',OLD/'bank_seal.json',
           FIRST/'two_bank_seal.json',OUT/'templates_P_private.json']
    for handoff in (job['a1_handoff'],job['second_handoff']):
        h=read(handoff);r=read(h['rule_file'])
        paths.extend([Path(handoff),Path(h['target_file']),Path(h['rule_file']),Path(r['projection']),Path(r['checkpoint'])])
    for base,rid in ((OLD,'dev_DINO_condk4_101_s200'),(FIRST,'dev_A2_condk4_101_s200')):
        bank=base/'banks'/rid;spec=read(bank/'run_spec.json');done=read(bank/'COMPLETED.json')
        verify_artifact(bank/'artifact',spec,done['signature'])
        paths.extend([bank/'run_spec.json',bank/'COMPLETED.json',bank/'artifact/artifact_seal.json'])
    contract={'schema':'receiver.seed202-repeat-campaign/v1','created':now(),
        'authorization':'Latest user requests paired DINO-only and A2 seed202, 128 images/200 updates; evaluate after both final PNGs',
        'stage':'2: developmental replication of observed combination gain',
        'bank_ids':IDS,'seed':202,'template_seed':202,'condition_seed':101,
        'templates':normalized,'bank_caps_seconds':{'DINO':3600,'A2':7200},
        'estimated_minutes':{'synthesis_total':109,'whole_task':[120,150]},
        'time_cap_scope':'per-bank cumulative worker wall time, including initialization/save/export; prep/eval separate',
        'images':128,'updates':200,'microbatch':16,'activation_updates':list(ACTIVATIONS),'optimizer':OPTIMIZER,
        'a1_handoff':job['a1_handoff'],'a1_handoff_sha256':sha(job['a1_handoff']),
        'second_handoff':job['second_handoff'],'second_handoff_sha256':sha(job['second_handoff']),
        'target_reuse':'exact seed101 head/transform/projection and non-DP patient/class-weighted P/Q signals; no extraction',
        'initialization':'same predefined public seed202 templates and renderer/optimizer/RNG initial state within DINO/A2 pair',
        'comparison':'A2-DINO DenseNet AUROC/AP and BioViL retention, per-seed101/202 results and simple mean; no new tuned threshold',
        'evaluation':'both final200 PNG artifacts sealed first; original V caches and 2000 paired patient draws',
        'replication_scope':'synthetic initialization including public template selection and residuals; fixed target conditions/heads',
        'old_checks_reused':'gradient/renderer/schedule/checkpoint/PNG code unchanged except seed dispatch and provenance seed',
        'new_checks':'CPU seed101 parity; seed202 changes init but not objective; matched init; wrong-seed checkpoint rejection',
        'pixel_scope':'P initialization only, no Q/V raw pixels; final synthetic PNG readout only',
        'storage':'prune only this campaign redundant checkpoints; retain step0 and two latest/final, all receipts',
        'historical_dependencies':{str(p.resolve()):sha(p) for p in paths},
        'old_runtime_bindings':old_binding['runtime'],
        'automatic_followup':False,'DP_Expert_Reserved_final_receiver':False,
        'nonDP_private_internal':True,'free_bytes':shutil.disk_usage(CODE).free}
    save(root/'campaign_contract.json',contract)
    state_path=RESEARCH/'research_state.json';state=read(state_path)
    state['receiver_seed202_replication']={'status':'PREPARING','started':now(),
        'contract':str(root/'campaign_contract.json'),'bank_ids':IDS,'automatic_followup':False}
    state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'phase':'scope_recorded','contract_sha256':sha(root/'campaign_contract.json'),
                      'public_templates':len(normalized),'free_bytes':contract['free_bytes']}),flush=True)


def verify(root):
    tick=time.monotonic();out=root/'seed_verification';out.mkdir(exist_ok=False)
    contract=read(root/'campaign_contract.json')
    save(out/'contract.json',{'created':now(),'inputs':'constructed four-image tensors; no patient/model forwards',
        'seeds':[101,202],'checkpoint':'CPU seeded initialization and wrong-seed rejection',
        'code_bindings':code_bindings(),'recipe':sha(root/'campaign_contract.json')})
    templates=torch.linspace(.05,.95,4*224*224).reshape(4,1,224,224)
    models={};results={}
    for seed in (101,202):
        r,o=initialize_parameters(templates,seed);r.set_step(1);first=initial_fingerprint(r,o)
        old=ScheduledRenderer(templates,'B',seed)
        options={k:v for k,v in OPTIMIZER.items() if k!='name'};options['betas']=tuple(options['betas'])
        old_o=torch.optim.AdamW(old.parameters(),**options);seed_all(seed);old.set_step(1)
        require(first==initial_fingerprint(old,old_o),'Old seed plumbing differs')
        for step in (1,33,65,97,129,161,200):
            r.set_step(step);old.set_step(step)
            with torch.no_grad():require(torch.equal(r(),old()),'Seed parity forward differs')
        r.set_step(1);models[seed]=(r,o)
        results[str(seed)]={'old_initialization_exact':True,'all_activation_outputs_exact':True}
    require(any(not torch.equal(a,b) for a,b in zip(models[101][0].residuals,models[202][0].residuals)),
            'Seed202 residuals not changed')
    # Same templates/seed give the same parameter, optimizer and RNG state for either arm.
    a,ao=initialize_parameters(templates,202);a.set_step(1);fa=initial_fingerprint(a,ao)
    b,bo=initialize_parameters(templates,202);b.set_step(1);fb=initial_fingerprint(b,bo)
    require(fa==fb,'Paired initial states differ')
    with tempfile.TemporaryDirectory(prefix='receiver_seed_check_') as td:
        path=Path(td);signature=digest({'seed':202})
        save_checkpoint(path,a,ao,0,signature)
        restored,receipt=load_checkpoint(path,b,bo,signature);require(restored==0,'Wrong restored step')
        try:load_checkpoint(path,b,bo,digest({'seed':101}))
        except RuntimeError:rejected=True
        else:rejected=False
        require(rejected,'Wrong seed accepted at resume')
    historical=read(FIRST/'jobs/A2.json');labels=(0,0,1,1);objective_checks={}
    for arm in ('DINO','A2'):
        j=dict(historical,seed=202,template_seed=202,condition_seed=101,source_set=arm,population='P')
        got,_=load_objective(j,labels)
        if arm=='DINO':ref,_=load_dino(j['second_handoff'],labels,'P')
        else:ref,_=load_combined(j['a1_handoff'],j['second_handoff'],labels,'P')
        require(objective_digest(got)==objective_digest(ref),'Seed202 changed condition objective')
        objective_checks[arm]=objective_digest(got)
    result={'status':'PASS_EXPLICIT_SYNTHESIS_SEED_AND_TARGET_SEPARATION','checks':results,
        'seed202_residuals_different':True,'paired_initial_states_exact':True,'wrong_seed_resume_rejected':True,
        'prepared_objective_unchanged':objective_checks,'new_target_extractions':0,'patient_pixels':0,
        'encoder_forwards':0,'encoder_backwards':0,'optimizer_updates':0,'seconds':time.monotonic()-tick}
    save(out/'result.json',result);print(json.dumps(result),flush=True)


def bind(root):
    require(read(root/'seed_verification/result.json')['status']=='PASS_EXPLICIT_SYNTHESIS_SEED_AND_TARGET_SEPARATION',
            'Missing seed verification')
    contract=read(root/'campaign_contract.json');historical=read(FIRST/'jobs/A2.json')
    jobs={};(root/'jobs').mkdir()
    for arm,rid in zip(('DINO','A2'),IDS):
        job=dict(historical,schema='receiver.seed-repeat-s200-bank/v1',id=rid,seed=202,template_seed=202,
            condition_seed=101,source_set=arm,maximum_seconds=contract['bank_caps_seconds'][arm],
            campaign_contract=str(root/'campaign_contract.json'),campaign_contract_sha256=sha(root/'campaign_contract.json'),
            templates=contract['templates'],code_bindings=code_bindings(),previous_A2_job=str(FIRST/'jobs/A2.json'),
            paired_initialization=str(root/'banks'/IDS[0]/'initialization.json') if arm=='A2' else None)
        file=root/'jobs'/(arm+'.json');save(file,job);jobs[arm]=str(file)
    sources={str(p):sha(p) for p in Path(__file__).parent.glob('*.py')}
    binding={'created':now(),'jobs':jobs,'job_sha256':{a:sha(p) for a,p in jobs.items()},
        'implementation':sources,'runtime':code_bindings(),'evaluation_dependencies':dependencies(),
        'contract_sha256':sha(root/'campaign_contract.json')}
    save(root/'execution_bindings.json',binding);snapshot=root/'implementation_snapshot';snapshot.mkdir()
    for p in Path(__file__).parent.glob('*.py'):shutil.copy2(p,snapshot/p.name)
    return binding


def run(root,resume):
    started=time.monotonic()
    try:
        if not resume:
            initialize(root);verify(root);binding=bind(root)
        else:binding=read(root/'execution_bindings.json')
        require(all(sha(p)==h for p,h in binding['implementation'].items()),'Implementation changed')
        for arm,rid in zip(('DINO','A2'),IDS):
            require(sha(binding['jobs'][arm])==binding['job_sha256'][arm],'Job changed')
            bank=root/'banks'/rid;resultfile=root/(arm+'_result.json')
            if not resultfile.exists():
                arguments=['-m','receiver_distillation.repeat_runtime','--job',binding['jobs'][arm],
                    '--seed','202','--output',str(bank),'--result',str(resultfile)]
                if bank.exists():arguments.append('--resume')
                worker(root,arm+'_202_main',arguments)
            result=read(resultfile)
            require(result['status']=='COMPLETE' and result['completed_updates']==200,'Bank incomplete; evaluation closed')
        seals={}
        for rid in IDS:
            bank=root/'banks'/rid;done=read(bank/'COMPLETED.json');spec=read(bank/'run_spec.json')
            verify_artifact(bank/'artifact',spec,done['signature'])
            require(read(bank/'artifact/provenance_public.json')['seed']==202,'Provenance seed mismatch')
            seals[rid]={'completion_sha256':sha(bank/'COMPLETED.json'),'artifact_sha256':sha(bank/'artifact/artifact_seal.json')}
        dinit=read(root/'banks'/IDS[0]/'initialization.json');ainit=read(root/'banks'/IDS[1]/'initialization.json')
        require(dinit['fields']==ainit['fields'] and ainit['paired_initialization_sha256']==sha(root/'banks'/IDS[0]/'initialization.json'),
                'Actual pair initialization differs')
        if not(root/'two_bank_seal.json').exists():save(root/'two_bank_seal.json',{'created':now(),'banks':seals})
        if not(root/'evaluation/result.json').exists():
            worker(root,'evaluate_both202',['-m','receiver_distillation.evaluate_repeat','--root',str(root)])
        contract=read(root/'campaign_contract.json')
        require(all(sha(p)==h for p,h in contract['historical_dependencies'].items()),'Historical evidence changed')
        require(all(sha(p)==h for p,h in contract['old_runtime_bindings'].items()),'Old runtime changed')
        result={'status':'COMPLETE','finished':now(),'controller_seconds':time.monotonic()-started,
            'evaluation_sha256':sha(root/'evaluation/result.json'),'paired_initialization_exact':True,
            'historical_evidence_unchanged':True,'free_bytes':shutil.disk_usage(CODE).free,'automatic_followup':False}
        save(root/'campaign_result.json',result);atomic_json(root/'campaign_status.json',result,replace=True)
        print(json.dumps(result),flush=True)
    except Exception as exc:
        if root.exists():atomic_json(root/'campaign_status.json',{'status':'STOPPED','updated':now(),'error':str(exc)},replace=True)
        traceback.print_exc();raise


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=ROOT)
    ap.add_argument('--resume',action='store_true');args=ap.parse_args();run(args.root,args.resume)


if __name__=='__main__':main()

