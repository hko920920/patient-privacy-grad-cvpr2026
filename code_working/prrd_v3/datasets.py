"""Exact manifest allow-list: W1 opens P pixels only; locked cohorts fail closed."""
from __future__ import annotations
import io, os, sys
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image
from .contracts import CODE, OUT, rows, sha, require, seed, save

TRAIN = CODE/'_reports/real_support_plan_20260917_v1/expanded_training_manifest_private.csv'
DEV = CODE/'_reports/real_support_plan_20260917_v1/method_development_unchanged_private.csv'
RAW = (CODE/'_data/raw').resolve()
EXPECTED = {'P':(672,813,6,3),'Q':(2027,5097,114,102),'V':(2026,5047,114,104)}

def manifests():
    train, dev = rows(TRAIN), rows(DEV)
    groups={'P':[r for r in train if r['planned_namespace']=='real/public'],
            'Q':[r for r in train if r['planned_namespace']=='diagnostic_real/former_classifier_selection'],
            'V':dev}
    require(len(groups['P'])+len(groups['Q'])==len(train), 'Unknown training namespace')
    require({r['original_thesis_role'] for r in groups['Q']}=={'private_train'}, 'Q origin changed')
    for role, group in groups.items():
        by=defaultdict(set)
        for r in group: by[r['patient_id']].add(int(r['label']))
        actual=(len(by),len(group),sum(1 in y for y in by.values()),sum(y=={0,1} for y in by.values()))
        require(actual==EXPECTED[role], f'Role counts differ: {role}: {actual}')
        require(len({r['image_id'] for r in group})==len(group), 'Repeated physical image')
    for key in ('patient_id','image_id','sha256'):
        a,b,c=[{r[key] for r in groups[k]} for k in ('P','Q','V')]
        require(not (a&b or a&c or b&c), 'Role overlap: '+key)
    return groups

class PixelAccess:
    """Audit hook covers ordinary Python opens in raw tree, including PIL/loaders.

    Unknown image paths are denied without reading final/reserved rosters. This
    is a process execution guard, not an OS sandbox against malicious code.
    """
    def __init__(self, groups, allowed_roles):
        require(set(allowed_roles)<= {'P','Q','V'}, 'Locked role requested')
        self.allowed_roles=frozenset(allowed_roles)
        self.allowed={str(Path(r['path']).resolve()).casefold():(role,r)
                      for role in allowed_roles for r in groups[role]}
        self.successes=defaultdict(int); self.denials=0; self.verified=set()
        self.raw_prefix=str(RAW).casefold()+os.sep
    def check(self,path):
        p=str(Path(path).resolve()).casefold()
        if p not in self.allowed:
            self.denials+=1
            raise PermissionError('PRRD pixel outside current phase allow-list')
        return self.allowed[p]
    def audit(self,event,args):
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            p=str(Path(os.fsdecode(args[0])).resolve()).casefold()
            if p.startswith(self.raw_prefix): self.check(p)
    def install(self): sys.addaudithook(self.audit); return self
    def image(self,row):
        role, trusted=self.check(row['path'])
        require(row==trusted, 'Row metadata is not the allow-listed record')
        p=Path(row['path'])
        # Reuse the verified manifest SHA; hash every newly opened source once.
        if str(p) not in self.verified:
            require(sha(p)==row['sha256'], 'Raw image SHA changed')
            self.verified.add(str(p))
        with Image.open(p) as im:
            require(im.size==(int(row['width']),int(row['height'])), 'Raw geometry mismatch')
            out=im.convert('L').copy()
        self.successes[role]+=1
        return out
    def report(self): return {'roles':sorted(self.allowed_roles),'decoded':dict(self.successes),
                              'unique_sha_verified':len(self.verified),'denials':self.denials,
                              'expert_or_reserved_pixels':0,'guard':'exact path and metadata allow-list'}

def public_templates(group, bank_seed):
    by=defaultdict(list)
    for r in group: by[r['patient_id']].append(r)
    order=sorted(by,key=lambda p:seed('prrd-v3-template-patient',bank_seed,p))[:64]
    chosen=[min(by[p],key=lambda r:seed('prrd-v3-template-image',bank_seed,r['image_id'])) for p in order]
    return chosen+chosen

def derangement(n, bank_seed):
    require(n>=2,'Cannot derange fewer than two mixed patients')
    rng=np.random.default_rng(seed('prrd-v3-Q-shuffle',bank_seed))
    for _ in range(10000):
        a=rng.permutation(n)
        if np.all(a!=np.arange(n)): return a
    raise RuntimeError('Derangement construction failed')

def png_roundtrip(x):
    a=np.rint(np.clip(x,0,1)*255).astype(np.uint8)
    buffer=io.BytesIO(); Image.fromarray(a,mode='L').save(buffer,format='PNG')
    buffer.seek(0)
    with Image.open(buffer) as im: decoded=np.asarray(im).copy()
    require(np.array_equal(decoded,a),'PNG export/decode mismatch')
    return decoded,buffer.getvalue()
