from downstream_utility.common import CODE, RESEARCH, RAW, sha, read, rows, save, now, require, tensor_sha, setup
from pathlib import Path
import hashlib, json
import numpy as np

OUT = CODE / '_reports/classifier_generalization_20260917_v1'
OLD = CODE / '_reports/downstream_development_20260917_v1'
LORA = CODE / '_reports/lora_transfer_20260917_v1'
PROFILE = CODE / '_reports/downstream_profile_20260917_v3'
ARMS = ('R1', 'Dreal', 'L_public', 'L_pooled')
SEEDS = (11, 23, 37)
PROTOCOL = RESEARCH / 'TRACK1_CLASSIFIER_GENERALIZATION_PROTOCOL_20260917.md'

def state_hash(state):
    h = hashlib.sha256()
    for k, v in sorted(state.items()):
        a = v.detach().cpu().contiguous().numpy()
        h.update(json.dumps([k, str(a.dtype), list(a.shape)], separators=(',', ':')).encode())
        h.update(a.tobytes())
    return h.hexdigest()

def npz(path, **values):
    with Path(path).open('xb') as f:
        np.savez_compressed(f, **values)

def validate_sources(c):
    for p, h in c['source_sha256'].items():
        require(sha(p) == h, 'Frozen source changed: ' + p)

def log(**values):
    print(json.dumps(dict(utc=now(), **values)), flush=True)
