"""Independent CPU scan of saved SecMI latent states and realized data labels."""
import json, math, time
from collections import defaultdict
from pathlib import Path
import torch
from .common import ROOT, RUN, digest, read_csv, write_json

def main():
    start=time.perf_counter()
    out=RUN/'baseline_screen_20260914/secmi_v1'
    def read(name): return json.loads((out/name).read_text(encoding='utf-8'))
    protocol,report=read('protocol.json'),read('report.json')
    assert digest(out/'protocol.json')==report['protocol_sha256']
    for name,sha in protocol['code_sha256'].items(): assert digest(Path(__file__).with_name(name))==sha
    for name,sha in protocol['input_sha256'].items(): assert digest(ROOT/name)==sha
    for name,sha in protocol['original_protected_sha256'].items(): assert digest(ROOT/name)==sha
    meta={r['image_id']:r for r in read_csv(RUN/'cohort/evaluation_images.csv')}
    exposure={}
    for model,sha in protocol['checkpoint_sha256'].items():
        checkpoint=RUN/'training_coverage_v2'/model/'step_1000.pt'
        assert digest(checkpoint)==sha
        state=torch.load(checkpoint,map_location='cpu',weights_only=True)
        assert state['step']==1000 and sum(state['exposures'].values())==4000
        exposure[model]=state['exposures']
    images,patients=read('image_scores.json'),read('patient_scores.json')
    assert len(images)==384 and len(patients)==192
    assert digest(out/'image_scores.json')==report['image_scores_sha256']
    assert digest(out/'patient_scores.json')==report['patient_scores_sha256']
    groups=defaultdict(list); residuals=[]; seen=set()
    for row in images:
        model,image_id=row['model'],row['image_id']; source=meta[image_id]
        assert (model,image_id) not in seen; seen.add((model,image_id))
        for key in ('patient_id','eval_role','assignment_group','record_role'): assert row[key]==source[key]
        member=int(source['assignment_group']==('A' if model=='model_1' else 'B'))
        actual=exposure[model].get(image_id,0)
        assert row['member']==member and row['actual_training_exposures']==actual
        assert row['image_member']==int(actual>0)==int(member and row['scenario']=='E')
        assert source['record_role']==('train_candidate' if row['scenario']=='E' else 'U_observed')
        path=out/row['tensor_path']; assert path.resolve().is_relative_to(out.resolve())
        assert digest(path)==row['tensor_sha256']
        tensors=torch.load(path,map_location='cpu',weights_only=True)
        assert tensors['image_id']==image_id and tensors['model']==model
        a,b=tensors['z_det'],tensors['z_recon']
        assert a.shape==b.shape==(1,4,32,32) and a.dtype==b.dtype==torch.float32
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
        fp32=float(torch.linalg.vector_norm((a-b).flatten()).item())
        assert fp32==row['l2'] and row['score']==-fp32
        fp64=float(torch.linalg.vector_norm((a.double()-b.double()).flatten()).item())
        residuals.append(abs(fp32-fp64))
        assert row['forward']==len(row['stages'])==12 and row['backward']==0
        groups[model,row['scenario'],row['eval_role'],row['patient_id']].append(row)
    for p in patients:
        rows=groups[p['model'],p['scenario'],p['eval_role'],p['patient_id']]
        assert len(rows)==2 and sorted(r['image_id'] for r in rows)==p['image_ids']
        assert p['score_mean']==math.fsum(r['score'] for r in rows)/2
        assert p['score_max']==max(r['score'] for r in rows)
    result=dict(status='PASS_INDEPENDENT_RAW_STATES_LABELS_AND_AGGREGATION',images=384,patient_rows=192,
        fp32_norms_exactly_reproduced=384,max_abs_fp32_norm_minus_fp64_norm=max(residuals),
        protected_files_verified=len(protocol['original_protected_sha256']),total_forward=4608,total_backward=0,
        checkpoint_and_input_hashes_verified=True,elapsed_seconds=time.perf_counter()-start,
        code_sha256=digest(Path(__file__)),protocol_sha256=digest(out/'protocol.json'))
    write_json(out/'independent_raw_states_check.json',result)
    print(json.dumps(result))

if __name__=='__main__': main()
