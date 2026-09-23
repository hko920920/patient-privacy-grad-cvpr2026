"""P-only A2 adapter using the already verified training/save/replay loop."""
import argparse,json,time,uuid
from pathlib import Path
from prrd_v3.contracts import read,require,sha
from prrd_v3.bank_runtime import atomic_json,file_lock
from .repeat_runtime import _run,code_bindings as previous_bindings


def code_bindings():
    result=previous_bindings();result[str(Path(__file__))]=sha(__file__)
    return result


def validate_job(job):
    require(job['schema']=='receiver.public-a2-s200-bank/v1','Wrong P-only job')
    require(job['population']=='P' and job['source_set']=='A2','P-only two-source objective required')
    require(job['seed']==job['template_seed']==job['condition_seed']==101,'Seed changed')
    require(job['artifact_kind']=='MAIN_BANK','Only the authorized final bank')
    require(all(sha(p)==h for p,h in job['code_bindings'].items()),'Frozen runtime changed')
    require(sha(job['campaign_contract'])==job['campaign_contract_sha256'],'Contract changed')
    contract=read(job['campaign_contract']);old=read(contract['reference_job'])
    for key in ('images','updates','microbatch','activation_updates','optimizer','templates','maximum_seconds'):
        require(job[key]==old[key],'Changed matched recipe: '+key)
    require(job['id']==contract['bank_id'],'Unauthorized bank')
    require(sha(job['paired_initialization'])==contract['initialization_sha256'],'Reference initialization changed')
    for key in ('a1_handoff','second_handoff'):
        require(sha(job[key])==job[key+'_sha256']==contract[key+'_sha256'],'P-only target binding changed')
        h=read(job[key]);require(h['population']=='P' and h['privacy']=='PUBLIC_P_ONLY','Wrong target population')
    return (0,)*64+(1,)*64


def execute(job,directory,*,resume=False):
    started=time.monotonic();wall_started=time.time();labels=validate_job(job)
    directory=Path(directory)
    if resume:require(directory.is_dir(),'Missing resume directory')
    else:directory.mkdir(parents=True,exist_ok=False)
    ledger_path=directory/'time_budget.json'
    with file_lock(directory/'.process.lock'):
        require(not(directory/'COMPLETED.json').exists(),'Completed bank cannot rerun')
        charged=0.
        if resume:
            ledger=read(ledger_path);charged=ledger['charged_seconds']
            if ledger['active']:charged+=max(0.,wall_started-ledger['attempt_wall_started'])
        cap=job['maximum_seconds'];require(charged<cap,'Cumulative bank time cap exhausted')
        atomic_json(ledger_path,{'charged_seconds':charged,'active':True,
            'attempt_wall_started':wall_started,'cap_seconds':cap},replace=resume)
        try:
            result=_run(job,directory,labels,False,resume,started,started+cap-charged)
            result['worker_seconds']=time.monotonic()-started
            result['cumulative_worker_seconds']=charged+result['worker_seconds']
            require(result['cumulative_worker_seconds']<=cap,'Bank time cap exceeded')
            atomic_json(directory/('attempt_'+uuid.uuid4().hex+'.json'),result)
            return result
        finally:
            atomic_json(ledger_path,{'charged_seconds':charged+time.monotonic()-started,'active':False,
                'attempt_wall_started':wall_started,'cap_seconds':cap},replace=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--job',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--result',type=Path,required=True)
    ap.add_argument('--resume',action='store_true');a=ap.parse_args()
    result=execute(read(a.job),a.output,resume=a.resume);atomic_json(a.result,result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
