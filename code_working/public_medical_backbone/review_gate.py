"""Aggregate two blinded, non-clinical morphology reviews using the fixed gate."""
import argparse,csv,hashlib,json
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'_reports/public_medical_backbone_20260916_v1'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def require(ok,msg):
    if not ok:raise RuntimeError(msg)
def main(out,phase):
    contract=read(out/'contract.json');plan=Path(contract['plan_directory'])
    source=str(Path(__file__).resolve())
    require(contract['source_sha256'][source]==sha(Path(__file__)),'Gate source changed')
    verified=out/(phase+'_verification.json');v=read(verified)
    require(v['complete'] and v['status'].startswith('PASS'),'Numerical verification required')
    manifest_path=out/('manifest_'+phase+'.json');manifest=read(manifest_path)
    selected=[r for r in manifest if r['phase']==phase]
    ids={r['image_id'] for r in selected}
    require(len(ids)==len(selected),'Duplicate generated IDs')
    with (plan/'generation_tasks.csv').open(encoding='utf-8-sig',newline='') as f:
        tasks={r['task_id']:r for r in csv.DictReader(f)}
    ballots=[];review_files={}
    for reviewer in ('root','independent'):
        path=out/('review_'+phase+'_'+reviewer+'.json');review=read(path)
        require(review['checkpoint_blinded'] is True,'Review must be blinded')
        require(review['reviewer']==reviewer and review['phase']==phase,'Review identity')
        votes=review['images'];require(len(votes)==len(ids),'Incomplete image review')
        vote_map={r['image_id']:r for r in votes}
        require(set(vote_map)==ids and len(vote_map)==len(votes),'Review IDs mismatch')
        require(all(type(r['pass']) is bool and r['note'].strip() for r in votes),'Explicit binary decision and reason required')
        ballots.append(vote_map);review_files[path.name]=sha(path)
    joint={iid:all(b[iid]['pass'] for b in ballots) for iid in ids}
    checkpoints=sorted({r['checkpoint'] for r in selected})
    results={}
    for cp in checkpoints:
        subset=[r for r in selected if r['checkpoint']==cp]
        require(len(subset)==16,'Expected 16 images/checkpoint')
        counts={k:sum(joint[r['image_id']] for r in subset if tasks[r['task_id']]['prompt_id']==k)
                for k in ('generic','normal','effusion','cardiomegaly')}
        total=sum(joint[r['image_id']] for r in subset)
        results[cp]=dict(total_pass=total,per_prompt_pass=counts,passes_gate=total>=12 and min(counts.values())>=2)
    decision=dict(complete=True,created_utc=datetime.now(timezone.utc).isoformat(),phase=phase,
        manifest_sha256=sha(manifest_path),review_sha256=review_files,
        checkpoint_results=results,joint_pass_by_image=joint,
        reviewer_disagreements=sum(ballots[0][i]['pass']!=ballots[1][i]['pass'] for i in ids),
        nonclinical_heuristic_only=True,private_utility_or_DP_claim=False)
    if phase=='selection':
        require(checkpoints==['E4','E8'],'Both checkpoints required')
        decision['selected_checkpoint']=next((cp for cp in ('E4','E8') if results[cp]['passes_gate']),None)
        decision['selection_verification_sha256']=sha(verified)
    else:
        selection=read(out/'selection_decision.json')
        require(checkpoints==[selection['selected_checkpoint']],'Confirmation checkpoint fixed')
        decision.update(selected_checkpoint=checkpoints[0],passes_gate=results[checkpoints[0]]['passes_gate'],
            confirmation_verification_sha256=sha(verified),selection_decision_sha256=sha(out/'selection_decision.json'))
    path=out/(phase+'_decision.json')
    with path.open('x',encoding='utf-8') as f:json.dump(decision,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:v for k,v in decision.items() if k not in ('joint_pass_by_image',)},ensure_ascii=False))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('selection','confirmation'));p.add_argument('--out',type=Path,default=DEFAULT)
    args=p.parse_args();main(args.out,args.phase)
