"""Authorized P-only A2 seed202 bank, then both seeds' Q-addition report."""
import argparse,json,shutil,time,traceback
from pathlib import Path
import numpy as np
from prrd_v3.contracts import CODE,RESEARCH,read,save,sha,require,now
from prrd_v3.bank_runtime import atomic_json,verify_artifact
from .second_source import load_combined
from .signals import objective_digest
from .public_repeat_runtime import code_bindings,validate_job
from .run_feasibility import worker
from .evaluate_public_repeat import dependencies,PUBLIC,REPEAT

ROOT=CODE/'_reports/receiver_a2_public202_s200_20260923_v1'
BANK_ID='dev_A2_P_condk4_202_s200'


def initialize(root):
    started=time.monotonic();prior=read(PUBLIC/'execution_bindings.json')
    require(all(sha(p)==h for p,h in prior['implementation'].items()),'Verified previous code changed')
    root.mkdir(exist_ok=False);before=root/'records_before';before.mkdir()
    for p in [CODE.parent/n for n in ('AGENTS.md','CURRENT_STATUS.md','WORKLOG.md')]+[
            RESEARCH/n for n in ('research_state.json','RESEARCH_FRAMEWORK.md')]:shutil.copy2(p,before/p.name)
    save(root/'preparation_contract.json',{'created':now(),
        'authorization':'Latest user: one P-only A2 seed202 bank through evaluation, then both seeds Q-addition effects',
        'scope':'reuse exact P targets; same PQ202 initial state; 128/200/microbatch16',
        'checks':'existing target hashes, counts and objective digest exact; initialization fingerprints exact before update1',
        'comparison':'existing PQ202 minus new P202, plus existing seed101 comparison; same2000 patient draws',
        'evaluation':'final200 PNG sealed first; development only; no tuning or intermediate efficacy',
        'estimated_minutes':{'preparation':[5,15],'synthesis':75,'whole_task':[80,90]},
        'maximum_worker_seconds':7200,'new_feature_extraction':False,
        'extra_seeds_DP_Expert_Reserved_final':False,'code_bindings':code_bindings()})
    old=read(REPEAT/'jobs/A2.json');public_job=read(PUBLIC/'job.json')
    previous_check=read(PUBLIC/'target_verification.json');handoffs={};history={};checks={}
    for p in (REPEAT/'jobs/A2.json',PUBLIC/'job.json',PUBLIC/'target_verification.json'):
        history[str(p)]=sha(p)
    for key,name in (('a1_handoff','biovil'),('second_handoff','dinov2')):
        hf=public_job[key];h=read(hf);rule=read(h['rule_file']);handoffs[key]=hf
        require(sha(hf)==public_job[key+'_sha256'],'Previous P handoff changed')
        require(h['population']=='P' and h['privacy']=='PUBLIC_P_ONLY','P target selection')
        require(sha(h['target_file'])==h['target_sha256']==previous_check['checks'][name]['target_sha256'],'P target changed')
        require(sha(h['rule_file'])==h['rule_file_sha256'],'Rule changed')
        pq=read(old[key]);require(h['rule_file_sha256']==pq['rule_file_sha256'],'PQ202 condition rules differ')
        with np.load(h['target_file'],allow_pickle=False) as d:
            require(not any(k.startswith(('Q_','pooled_')) for k in d.files),'Wrong population in target')
            require(np.array_equal(d['P_counts'],[669,6]),'P class-present count changed')
            require(d['P_gradient'].shape==(4,34) and np.isfinite(d['P_gradient']).all(),'Target invalid')
        for p in (hf,h['target_file'],h['rule_file'],rule['checkpoint'],rule['projection']):
            history[str(p)]=sha(p)
        checks[name]={'exact_existing_target':True,'counts':[669,6],'target_sha256':h['target_sha256']}
    objective,rules=load_combined(handoffs['a1_handoff'],handoffs['second_handoff'],(0,)*64+(1,)*64,'P')
    require(objective_digest(objective)==previous_check['objective_digest'],'P objective changed')
    require(all(r['public_seed']==101 for r in rules.values()),'Condition seed changed')
    reference=REPEAT/'banks'/old['id'];spec=read(reference/'run_spec.json');done=read(reference/'COMPLETED.json')
    verify_artifact(reference/'artifact',spec,done['signature'])
    initfile=reference/'initialization.json';init=read(initfile)
    require(init['seed']==202 and init['condition_seed']==101,'Wrong paired initialization')
    require(init['template_tensor_digest']==spec['template_tensor_digest'],'Template witness mismatch')
    require(old['templates']==spec['templates'] and old['seed']==202,'Reference recipe mismatch')
    for p in (initfile,reference/'run_spec.json',reference/'COMPLETED.json',reference/'artifact/artifact_seal.json'):
        history[str(p)]=sha(p)
    verified={'status':'PASS_EXACT_REUSE','created':now(),'checks':checks,
        'objective_digest':objective_digest(objective),'new_target_extractions':0,'patient_pixels':0,
        'encoder_forwards':0,'encoder_backwards':0,'optimizer_updates':0,
        'reference_initialization':str(initfile),'seconds':time.monotonic()-started}
    save(root/'target_verification.json',verified)
    contract={'schema':'receiver.public-a2-seed202/v1','created':now(),'stage':'2 development Q-addition repetition',
        'authorization':'One P-only A2 seed202 bank and paired development comparison only',
        'bank_id':BANK_ID,'reference_job':str(REPEAT/'jobs/A2.json'),'reference_bank':str(reference),
        'paired_initialization':str(initfile),'initialization_sha256':sha(initfile),
        'seed':202,'template_seed':202,'condition_seed':101,'images':128,'updates':200,'microbatch':16,
        'maximum_seconds':7200,'target_population':'P','P_images':813,'P_patients':672,'P_class_present_patients':[669,6],
        'patient_weight':'mean visits within patient/class, then present-patient mean, then0.5 each class',
        'historical_dependencies':history,'old_runtime_bindings':prior['runtime'],
        **handoffs,**{k+'_sha256':sha(v) for k,v in handoffs.items()},
        'comparison':'A2(PQ)-A2(P) separately for101 and202; reuse all historical banks/predictions',
        'evaluation':'same V caches/readout and2000 paired patient draws after new final PNG sealing',
        'summary':'mean of two seed-specific metric deltas, not prediction ensemble and not seed-population CI',
        'estimated_minutes':{'synthesis':75,'whole_task':[80,90]},
        'time_cap_scope':'cumulative synthesis worker including initialization/save/export; prep/eval separate',
        'storage':'retain initial plus last two/final checkpoints and all receipts; prune this bank only',
        'automatic_followup':False,'DP_Expert_Reserved_final_receiver':False}
    save(root/'campaign_contract.json',contract)
    job=dict(old,schema='receiver.public-a2-seed202-bank/v1',id=BANK_ID,population='P',
        **handoffs,**{k+'_sha256':sha(v) for k,v in handoffs.items()},
        campaign_contract=str(root/'campaign_contract.json'),campaign_contract_sha256=sha(root/'campaign_contract.json'),
        code_bindings=code_bindings(),paired_initialization=str(initfile))
    save(root/'job.json',job);validate_job(job)
    sources={str(p):sha(p) for p in Path(__file__).parent.glob('*.py')}
    binding={'created':now(),'job':str(root/'job.json'),'job_sha256':sha(root/'job.json'),
        'runtime':code_bindings(),'implementation':sources,'evaluation_dependencies':dependencies(),
        'contract_sha256':sha(root/'campaign_contract.json'),'target_verification_sha256':sha(root/'target_verification.json')}
    save(root/'execution_bindings.json',binding);snapshot=root/'implementation_snapshot';snapshot.mkdir()
    for p in Path(__file__).parent.glob('*.py'):shutil.copy2(p,snapshot/p.name)
    report=RESEARCH/'RECEIVER_A2_PUBLIC_SEED202_CONTRACT_20260923.md'
    require(not report.exists(),'Existing contract cannot be overwritten')
    report.write_text('# A2 public-only seed202: frozen execution contract\n\n'
        'Authorized: one new P-only A2 bank, 128 images / 200 updates / microbatch16. '
        'Synthesis and template seed202; condition/head seed101 unchanged. Exact P targets from the completed seed101 control are reused. '
        'P class-present denominators remain669/6. No Q target/count participates in the new objective.\n\n'
        'Renderer, optimizer, RNG, active levels and template tensor must exactly match the completed PQ202 initialization before update1. '
        'Use the verified repeat_runtime._run unchanged. No new gradient/replay/profile runs. '
        'Final PNG only, followed by cached-V BioViL/DenseNet readout and the original2000 paired patient draws. '
        'Report PQ minus P for101 and202 separately; the mean delta is descriptive and does not estimate synthesis-population uncertainty.\n\n'
        'No new P/Q feature extraction, historical retraining, additional seed, training extension, DP, Expert, Reserved or final receiver. '
        'Estimated synthesis75 minutes, whole task80-90 minutes; cumulative worker cap120 minutes. '
        'Normal preparation/training/export/evaluation runs continuously. Stop for invalid data/numerics, unsafe resume or cap.\n\n'
        'Machine-readable contract: '+str(root/'campaign_contract.json')+'\nSHA256: '+sha(root/'campaign_contract.json')+'\n',encoding='utf-8')
    state_path=RESEARCH/'research_state.json';state=read(state_path)
    state['receiver_public_a2_seed202']={'status':'PREPARED','started':now(),'contract':report.name,
        'machine_contract':str(root/'campaign_contract.json'),'bank_id':BANK_ID,'new_target_extractions':0,
        'development_evaluation_opened':False,'automatic_followup':False}
    state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(verified),flush=True);return binding


def run(root,resume):
    started=time.monotonic()
    try:
        binding=read(root/'execution_bindings.json') if resume else initialize(root)
        require(all(sha(p)==h for p,h in binding['implementation'].items()),'Implementation changed')
        require(sha(binding['job'])==binding['job_sha256'],'Job changed')
        contract=read(root/'campaign_contract.json');bank=root/'banks'/BANK_ID;resultfile=root/'A2_P_result.json'
        if not resultfile.exists():
            arguments=['-m','receiver_distillation.public_repeat_runtime','--job',binding['job'],
                       '--output',str(bank),'--result',str(resultfile)]
            if bank.exists():arguments.append('--resume')
            worker(root,'A2_P_202_main',arguments)
        result=read(resultfile);require(result['status']=='COMPLETE' and result['completed_updates']==200,'Incomplete bank; evaluation closed')
        done=read(bank/'COMPLETED.json');spec=read(bank/'run_spec.json')
        verify_artifact(bank/'artifact',spec,done['signature'])
        init=read(bank/'initialization.json');reference=read(contract['paired_initialization'])
        require(all(init[k]==reference[k] for k in ('seed','condition_seed','fields','template_tensor_digest')),'Initial state mismatch')
        require(init['paired_initialization_sha256']==contract['initialization_sha256'],'Initial witness binding')
        if not(root/'bank_seal.json').exists():save(root/'bank_seal.json',{'created':now(),
            'bank_id':BANK_ID,'completion_sha256':sha(bank/'COMPLETED.json'),'artifact_sha256':sha(bank/'artifact/artifact_seal.json')})
        if not(root/'evaluation/result.json').exists():
            worker(root,'evaluate_two_seed_Q_addition',['-m','receiver_distillation.evaluate_public_repeat','--root',str(root)])
        for key in ('historical_dependencies','old_runtime_bindings'):
            require(all(sha(p)==h for p,h in contract[key].items()),'Historical evidence/code changed')
        result={'status':'COMPLETE','finished':now(),'controller_seconds':time.monotonic()-started,
            'evaluation_sha256':sha(root/'evaluation/result.json'),'paired_initialization_exact':True,
            'historical_evidence_unchanged':True,'free_bytes':shutil.disk_usage(CODE).free,'automatic_followup':False}
        save(root/'campaign_result.json',result);atomic_json(root/'campaign_status.json',result,replace=True)
        print(json.dumps(result),flush=True)
    except Exception as exc:
        if root.exists():atomic_json(root/'campaign_status.json',{'status':'STOPPED','updated':now(),'error':str(exc)},replace=True)
        traceback.print_exc();raise


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=ROOT);ap.add_argument('--resume',action='store_true')
    args=ap.parse_args();run(args.root,args.resume)


if __name__=='__main__':main()

