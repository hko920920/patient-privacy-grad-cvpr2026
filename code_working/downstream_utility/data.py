"""One explicit full-frame preprocessing path and patient-uniform batch draws."""
import hashlib,time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
import numpy as np
from PIL import Image
from .common import *

def letterbox(im,size=224):
    im=im.convert('L');w,h=im.size
    scale=size/max(w,h);nw=max(1,round(w*scale));nh=max(1,round(h*scale))
    small=im.resize((nw,nh),Image.Resampling.BILINEAR)
    canvas=Image.new('L',(size,size),0)
    canvas.paste(small,((size-nw)//2,(size-nh)//2))
    return np.asarray(canvas,dtype=np.uint8).copy()

def prepare_cache(out=OUT):
    c=check(out);manifest=rows(out/'real_manifest_private.csv')
    require(not (out/'real224.npy').exists(),'No cache overwrite')
    start=time.perf_counter();cache=np.lib.format.open_memmap(out/'real224.npy',mode='w+',dtype=np.uint8,shape=(len(manifest),224,224))
    def one(row):
        tick=time.perf_counter();p=Path(row['path']).resolve()
        require(p.is_relative_to(RAW.resolve()),'Outside image root')
        data=p.read_bytes();require(hashlib.sha256(data).hexdigest()==row['sha256'],'Raw SHA '+row['image_id'])
        require(len(data)==int(row['bytes']),'Byte count')
        with Image.open(BytesIO(data)) as im:
            im.load();require(im.size==(int(row['width']),int(row['height'])),'Dimensions')
            require(im.mode==row['mode'],'Image mode')
            x=letterbox(im)
        return x,dict(index=int(row['index']),image_id=row['image_id'],patient_id=row['patient_id'],role=row['role'],
                      raw_sha256=row['sha256'],tensor_sha256=tensor_sha(x),seconds=time.perf_counter()-tick)
    audits=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,(x,a) in enumerate(pool.map(one,manifest)):
            cache[i]=x;audits.append(a)
            if (i+1)%2000==0:print(json.dumps({'phase':'data','done':i+1,'total':len(manifest)}),flush=True)
    cache.flush();del cache
    save(out/'image_audit.json',audits)
    save(out/'data.json',dict(complete=True,images=len(manifest),patients=len({r['patient_id'] for r in manifest}),
        seconds=time.perf_counter()-start,total_bytes=sum(int(r['bytes']) for r in manifest),
        preprocessing='PIL L; full-frame bilinear letterbox224; round nearest-even; centered black pad',
        raw_SHA_recomputed=True,pixels_decoded=True,expert_pixels=0,reserved_pixels=0,
        manifest_sha256=sha(out/'real_manifest_private.csv'),cache_sha256=sha(out/'real224.npy'),audit_sha256=sha(out/'image_audit.json')))

def pools(records):
    # Deduplicate exact repeated rows: a physical real-repeat CSV is the same population.
    unique={r['image_id']:r for r in records};result={0:{},1:{}}
    for r in sorted(unique.values(),key=lambda z:z['image_id']):
        result[int(r['label'])].setdefault(r['patient_id'],[]).append(r)
    return result

def draw(pool,n,rng_seed):
    rng=np.random.default_rng(rng_seed);chosen=[]
    for label in (0,1):
        patients=sorted(pool[label]);require(bool(patients),'Empty class')
        for _ in range(n//2):
            p=patients[int(rng.integers(len(patients)))];rr=pool[label][p]
            chosen.append(rr[int(rng.integers(len(rr)))])
    return chosen

def batch_records(source_pools,arm,run_seed,step):
    shared=draw(source_pools['public'],16,seed(run_seed,step,'public-shared'))
    if arm in ('R0','R1'):
        other=draw(source_pools['public'],16,seed(run_seed,step,'public-extra'))
    elif arm=='Dreal':other=draw(source_pools['private'],16,seed(run_seed,step,'private-extra'))
    else:
        method=dict(S0='backbone',S1='public',S2='private_only',S3='pooled')[arm]
        # Identical synthetic cell indices across methods because pseudo-patient keys/cells match.
        other=draw(source_pools[method],16,seed(run_seed,step,'synthetic-extra'))
    return shared+other

def preprocess_batch(arrays,arm,run_seed,step,device):
    import torch
    from torchvision.transforms import functional as TF,InterpolationMode
    x=torch.from_numpy(np.stack(arrays).copy()).to(device=device,dtype=torch.float32)[:,None].repeat(1,3,1,1)/255.
    if arm!='R0':
        augmented=[]
        for slot in range(len(arrays)):
            rng=np.random.default_rng(seed(run_seed,step,'augment',slot))
            angle=float(rng.uniform(-5,5));tx=int(round(rng.uniform(-.02,.02)*224));ty=int(round(rng.uniform(-.02,.02)*224));scale=float(rng.uniform(.95,1.05))
            augmented.append(TF.affine(x[slot],angle,[tx,ty],scale,[0.,0.],interpolation=InterpolationMode.BILINEAR,fill=0.))
        x=torch.stack(augmented)
    mean=x.new_tensor([.485,.456,.406])[None,:,None,None];std=x.new_tensor([.229,.224,.225])[None,:,None,None]
    return (x-mean)/std
