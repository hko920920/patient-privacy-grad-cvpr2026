"""One A2 public-only bank, then its fixed P+Q minus P development comparison."""
import argparse,json,shutil,time,traceback
from pathlib import Path
import numpy as np
import torch
from prrd_v3.contracts import CODE,RESEARCH,read,save,sha,require,now
from prrd_v3.datasets import manifests,public_templates
from prrd_v3.bank_runtime import atomic_json,digest,tree_digest,verify_artifact
from .second_source import load_combined
from .signals import objective_digest
from .public_runtime import code_bindings,validate_job
from .run_feasibility import worker
from .evaluate_public import dependencies,FIRST

ROOT=CODE/'_reports/receiver_a2_public_s200_20260923_v1'
REPEAT=CODE/'_reports/receiver_repeat202_s200_20260923_v1'
BANK_ID='dev_A2_P_condk4_101_s200'


def initialize(root):
    started=time.monotonic();prior=read(REPEAT/'execution_bindings.json')
    require(all(sha(p)==h for p,h in prior['implementation'].items()),'Verified previous code changed')
    root.mkdir(exist_ok=False);before=root/'records_before';before.mkdir()
    for p in [CODE.parent/n for n in ('AGENTS.md','CURRENT_STATUS.md','WORKLOG.md')]+[
            RESEARCH/n for n in ('research_state.json','RESEARCH_FRAMEWORK.md')]:shutil.copy2(p,before/p.name)
    save(root/'preparation_contract.json',{'created':now(),'authorization':'Latest user: one public-only A2 seed101 bank through evaluation',
        'scope':'P target selection only, unchanged 128/200/seed101/microbatch16 and existing recipe',
        'target_check':'P sum/count recomputation and exact original P objective; atol1e-12; no Q/pooled blocks in selected files',
        'comparison':'existing A2(P+Q) minus new A2(P), DenseNet AUROC/AP and BioViL, same2000 patient draws',
        'evaluation':'last200 PNG sealed first; development only; no threshold/HPO/new checkpoints',
        'estimated_minutes':{'preparation':[10,20],'synthesis':75},'maximum_worker_seconds':7200,
        'extra_seeds_DP_Expert_Reserved_final':False,'code_bindings':code_bindings()})
    old=read(FIRST/'jobs/A2.json');groups=manifests();ps=groups['P']
    pts=[{'image_id':r['image_id'],'sha256':r['sha256']} for r in public_templates(ps,101)]
    require(pts==old['templates'],'Historical template selection differs')
    counts=np.array([len({r['patient_id'] for r in ps if int(r['label'])==c}) for c in (0,1)])
    require(len(ps)==813 and len({r['patient_id'] for r in ps})==672 and counts.tolist()==[669,6],'P role/count changed')
    history={str(FIRST/'jobs/A2.json'):sha(FIRST/'jobs/A2.json')};handoffs={};checks={}
    for key,name in (('a1_handoff','biovil'),('second_handoff','dinov2')):
        original=read(old[key]);rule=read(original['rule_file'])
        for p in (old[key],original['target_file'],original['rule_file'],rule['checkpoint'],rule['projection']):
            path=Path(p).resolve();history[str(path)]=sha(path)
        require(sha(original['target_file'])==original['target_sha256'],'Original target changed')
        require(sha(original['rule_file'])==original['rule_file_sha256'],'Original rule changed')
        with np.load(original['target_file'],allow_pickle=False) as d:
            selected={k:d[k].copy() for k in ('schema','rule_sha256','condition_ids','P_sums','P_counts','P_gradient')}
        require(np.array_equal(selected['P_counts'],counts),'P count does not match manifest')
        recomputed=.5*(selected['P_sums']/counts[None,:,None]).sum(1)
        error=float(np.max(np.abs(recomputed-selected['P_gradient'])))
        require(error<=1e-12 and np.isfinite(recomputed).all(),'Public class/patient mean mismatch')
        folder=root/'targets'/name;folder.mkdir(parents=True)
        target=folder/'P_condition_targets.npz';np.savez(target,**selected)
        rulefile=folder/'condition_rule.json';shutil.copy2(original['rule_file'],rulefile)
        handoff=dict(original,target_file=str(target),target_sha256=sha(target),rule_file=str(rulefile),
                     rule_file_sha256=sha(rulefile),population='P',privacy='PUBLIC_P_ONLY',
                     contract_file=str(root/'preparation_contract.json'),contract_sha256=sha(root/'preparation_contract.json'))
        hf=folder/'target_handoff.json';save(hf,handoff);handoffs[key]=str(hf)
        with np.load(target,allow_pickle=False) as d:
            require(not any(k.startswith(('Q_','pooled_')) for k in d.files),'Nonpublic block copied')
            require(np.array_equal(d['P_gradient'],selected['P_gradient']),'P signal modified')
        checks[name]={'counts':counts.tolist(),'max_recomputed_error':error,'exact_selected_P':True,
                      'Q_or_pooled_blocks':False,'target_sha256':sha(target)}
    labels=(0,)*64+(1,)*64
    objective,rules=load_combined(handoffs['a1_handoff'],handoffs['second_handoff'],labels,'P')
    original,original_rules=load_combined(old['a1_handoff'],old['second_handoff'],labels,'P')
    require(objective_digest(objective)==objective_digest(original) and rules==original_rules,'P selection changes objective')
    try:load_combined(handoffs['a1_handoff'],handoffs['second_handoff'],labels,'pooled')
    except KeyError:rejected=True
    else:rejected=False
    require(rejected,'P-only target permits pooled selection')
    reference=FIRST/'banks'/old['id'];spec=read(reference/'run_spec.json');done=read(reference/'COMPLETED.json')
    verify_artifact(reference/'artifact',spec,done['signature'])
    checkpoints=list((reference/'checkpoints').glob('step_000000_*.pt'));require(len(checkpoints)==1,'Missing unique old initialization')
    cp=checkpoints[0];record=read(str(cp)+'.json');require(sha(cp)==record['sha256'],'Original checkpoint changed')
    payload=torch.load(cp,map_location='cpu',weights_only=True)
    require(tree_digest(payload)==record['payload_digest'] and payload['completed_updates']==0,'Invalid old initial state')
    witness={'seed':101,'condition_seed':101,'template_tensor_digest':spec['template_tensor_digest'],
        'fields':{k:tree_digest(payload[k]) for k in ('renderer','optimizer','rng','active_levels','requires_grad')},
        'checkpoint_sha256':sha(cp),'source_checkpoint':str(cp)}
    save(root/'reference_initialization.json',witness)
    for p in (cp,Path(str(cp)+'.json'),reference/'run_spec.json',reference/'COMPLETED.json',reference/'artifact/artifact_seal.json'):
        history[str(p)]=sha(p)
    verified={'status':'PASS_EXACT_P_TARGET_SELECTION','created':now(),'checks':checks,
        'objective_digest':objective_digest(objective),'same_heads_transforms_projection':True,
        'pooled_selection_rejected':True,'reference_initialization_sha256':sha(root/'reference_initialization.json'),
        'new_target_extractions':0,'patient_pixels':0,'encoder_forwards':0,'encoder_backwards':0,
        'optimizer_updates':0,'seconds':time.monotonic()-started}
    save(root/'target_verification.json',verified)
    contract={'schema':'receiver.public-a2-control/v1','created':now(),'stage':'2 developmental private-data-value control',
        'authorization':'One A2 public-only seed101 bank, preparation through sealed final PNG and V evaluation',
        'bank_id':BANK_ID,'reference_job':str(FIRST/'jobs/A2.json'),'reference_bank':str(reference),
        'templates':pts,'seed':101,'images':128,'updates':200,'microbatch':16,'maximum_seconds':7200,
        'target_population':'P','P_images':813,'P_patients':672,'P_class_present_patients':counts.tolist(),
        'patient_weight':'mean visits within patient/class, then present-patient mean, then0.5 each class',
        'initialization_sha256':sha(root/'reference_initialization.json'),
        'historical_dependencies':history,'old_runtime_bindings':prior['runtime'],
        **{k:v for k,v in handoffs.items()},**{k+'_sha256':sha(v) for k,v in handoffs.items()},
        'comparison':'A2(P+Q)-A2(P); existing seed101 baseline reused, both use same public initial state',
        'evaluation':'same V caches/readout and2000 paired patient draws after final PNG sealing',
        'estimated_minutes':{'synthesis':75,'whole_task':[85,95]},
        'time_cap_scope':'cumulative synthesis worker including initialization/save/export; prep/eval separate',
        'storage':'retain initial plus last two/final checkpoints and all receipts, prune this bank only',
        'automatic_followup':False,'DP_Expert_Reserved_final_receiver':False}
    save(root/'campaign_contract.json',contract)
    job=dict(old,schema='receiver.public-a2-s200-bank/v1',id=BANK_ID,population='P',
        template_seed=101,condition_seed=101,source_set='A2',**handoffs,
        **{k+'_sha256':sha(v) for k,v in handoffs.items()},
        campaign_contract=str(root/'campaign_contract.json'),campaign_contract_sha256=sha(root/'campaign_contract.json'),
        code_bindings=code_bindings(),paired_initialization=str(root/'reference_initialization.json'))
    save(root/'job.json',job);validate_job(job)
    sources={str(p):sha(p) for p in Path(__file__).parent.glob('*.py')}
    binding={'created':now(),'job':str(root/'job.json'),'job_sha256':sha(root/'job.json'),
        'runtime':code_bindings(),'implementation':sources,'evaluation_dependencies':dependencies(),
        'contract_sha256':sha(root/'campaign_contract.json'),'target_verification_sha256':sha(root/'target_verification.json')}
    save(root/'execution_bindings.json',binding);snapshot=root/'implementation_snapshot';snapshot.mkdir()
    for p in Path(__file__).parent.glob('*.py'):shutil.copy2(p,snapshot/p.name)
    state_path=RESEARCH/'research_state.json';state=read(state_path)
    state['receiver_public_a2_control']={'status':'PREPARED','started':now(),'contract':str(root/'campaign_contract.json'),
        'bank_id':BANK_ID,'new_target_extractions':0,'development_evaluation_opened':False,'automatic_followup':False}
    state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(verified),flush=True);return binding


def run(root,resume):
    started=time.monotonic()
    try:
        binding=read(root/'execution_bindings.json') if resume else initialize(root)
        require(all(sha(p)==h for p,h in binding['implementation'].items()),'Implementation changed')
        require(sha(binding['job'])==binding['job_sha256'],'Job changed')
        bank=root/'banks'/BANK_ID;resultfile=root/'A2_P_result.json'
        if not resultfile.exists():
            arguments=['-m','receiver_distillation.public_runtime','--job',binding['job'],
                       '--output',str(bank),'--result',str(resultfile)]
            if bank.exists():arguments.append('--resume')
            worker(root,'A2_P_101_main',arguments)
        result=read(resultfile);require(result['status']=='COMPLETE' and result['completed_updates']==200,'Incomplete bank; evaluation closed')
        done=read(bank/'COMPLETED.json');spec=read(bank/'run_spec.json')
        verify_artifact(bank/'artifact',spec,done['signature'])
        init=read(bank/'initialization.json');reference=read(root/'reference_initialization.json')
        require(all(init[k]==reference[k] for k in ('seed','condition_seed','fields','template_tensor_digest')),'Initial state mismatch')
        require(init['paired_initialization_sha256']==sha(root/'reference_initialization.json'),'Initial witness binding')
        if not(root/'bank_seal.json').exists():save(root/'bank_seal.json',{'created':now(),
            'bank_id':BANK_ID,'completion_sha256':sha(bank/'COMPLETED.json'),'artifact_sha256':sha(bank/'artifact/artifact_seal.json')})
        if not(root/'evaluation/result.json').exists():worker(root,'evaluate_PQ_minus_P',['-m','receiver_distillation.evaluate_public','--root',str(root)])
        contract=read(root/'campaign_contract.json')
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
