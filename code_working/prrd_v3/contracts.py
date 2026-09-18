"""Immutable input bindings, phase budgets and legacy import rejection."""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '4'
import csv, hashlib, importlib.abc, json, sys
from datetime import datetime, timezone
from pathlib import Path
import yaml

CODE = Path(__file__).resolve().parents[1]
PACKAGE = Path(__file__).resolve().parent
RESEARCH = next(CODE.parent.glob('CVPR */research_2026-09-10'))
SUPPLIED = RESEARCH/'spec_sources/prrd_v3_supplied_20260918/prrd_execution_plan_v3'
OUT = CODE/'_reports/prrd_v3_20260918_w01'
ARMS = ('A', 'B', 'C', 'D', 'R_joint')
SEEDS = (101, 202, 303)
FORBIDDEN = {'downstream_utility.data', 'downstream_utility.train'}

def now(): return datetime.now(timezone.utc).isoformat()
def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024), b''): h.update(chunk)
    return h.hexdigest()

def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')

def rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f: return list(csv.DictReader(f))

def seed(*parts): return int(hashlib.sha256('|'.join(map(str, parts)).encode()).hexdigest()[:15],16)
def require(ok, message):
    if not bool(ok): raise RuntimeError(message)

class RejectLegacy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in FORBIDDEN: raise RuntimeError('Blocked legacy module: '+fullname)

def install_legacy_guard():
    require(not FORBIDDEN.intersection(sys.modules), 'Legacy module already loaded')
    if not any(isinstance(x,RejectLegacy) for x in sys.meta_path): sys.meta_path.insert(0,RejectLegacy())

def setup():
    install_legacy_guard()
    import torch
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    require(torch.cuda.is_available(), 'CUDA required for runtime checks')

def contract():
    with (OUT/'contract.yaml').open(encoding='utf-8') as f: return yaml.safe_load(f)

def check_bindings(cfg=None):
    cfg = contract() if cfg is None else cfg
    for path, expected in cfg['bindings'].items():
        require(sha(path)==expected, 'Changed bound input: '+path)
    for name in ('dp_authorized','expert_authorized','reserved_authorized'):
        require(cfg['execution'][name] is False, 'Protected phase not part of this runner')
    return cfg

def source_hashes(): return {str(p):sha(p) for p in sorted(PACKAGE.glob('*.py'))}

def require_w2_authority():
    cfg=check_bindings()
    report=read(OUT/'w1_verification.json')
    require(report['status']=='PASS_PUBLIC_RUNTIME_ONLY', 'W1 has not passed')
    require(report['source_sha256']==source_hashes(), 'Runtime changed since W1 verification')
    budget_path=OUT/'w2_time_authorization.json'
    require(budget_path.exists(), 'W2 numeric time authorization is unbound; report measured cost first')
    budget=read(budget_path)
    require(budget['contract_sha256']==sha(OUT/'contract.yaml'), 'Budget bound to another contract')
    require(budget.get('user_authorization') and budget['maximum_seconds']>0, 'Invalid W2 authority')
    return cfg,budget

