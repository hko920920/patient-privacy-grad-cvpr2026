import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='4'
import csv,hashlib,json
from datetime import datetime,timezone
from pathlib import Path

CODE=Path(__file__).resolve().parents[1]
RESEARCH=next((CODE.parent/'CVPR \uc8fc\uc81c \ud0d0\uc0c9').glob('research_2026-09-10'))
OUT=CODE/'_reports/downstream_profile_20260917_v3'
MASTER=CODE/'_reports/downstream_master_20260917_v1'
MEDICAL=CODE/'_reports/medical_head_20260916_v1'
SIGNAL=CODE/'_reports/private_signal_20260917_v1'
RAW=CODE/'_data/raw/nih_cxr14_pa_k10_plus_census_v1/images'
ARMS=('R0','R1','S0','S1','S2','S3','Dreal')
METHODS=('backbone','public','private_only','pooled')
MODEL_FILE=Path.home()/'.cache/torch/hub/checkpoints/resnet18-f37072fd.pth'
MODEL_SHA='f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec'

def require(ok,msg):
    if not bool(ok): raise RuntimeError(msg)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def rows(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def save(p,obj):
    with Path(p).open('x',encoding='utf-8') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def save_rows(p,items):
    with Path(p).open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(items[0]));w.writeheader();w.writerows(items)
def now():return datetime.now(timezone.utc).isoformat()
def seed(*parts):return int(hashlib.sha256(('downstream-profile-20260917-v1|'+'|'.join(map(str,parts))).encode()).hexdigest()[:15],16)
def tensor_sha(x):
    a=x.detach().cpu().contiguous().numpy() if hasattr(x,'detach') else x
    return hashlib.sha256(a.tobytes()).hexdigest()
def check(out=OUT):
    c=read(out/'contract.json')
    for p,h in c['source_sha256'].items():require(sha(p)==h,'Frozen source changed '+p)
    require(not c['expert_final_allowed'] and not c['reserved_confirmation_allowed'],'No final access')
    return c
def setup():
    import torch
    torch.set_num_threads(4);require(torch.cuda.is_available(),'CUDA required')
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
