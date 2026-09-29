"""Select public128 using only public DINO/RN features, before DP target access."""
import argparse,csv,hashlib,time
from pathlib import Path
import numpy as np
from PIL import Image
from carrier_common import Inputs,check,dump,sha,now,write_seal,same_order

def run(out):
    tick=time.monotonic();io=Inputs(out,"select")
    check(not (out/"selection_seal.json").exists(),"Selection already sealed")
    d=io.nz("P_DINO");r=io.nz("P_RN");same_order(d,r)
    check(set(d["roles"])==set(r["roles"])=={"P"},"Public roles")
    check(d["z"].shape==(4,813,16) and r["z"].shape==(813,128),"Public feature schema")
    check(np.array_equal(d["condition_ids"],np.arange(4)),"DINO conditions")
    with io.path("P_roster").open(encoding="utf-8-sig",newline="") as f:roster=list(csv.DictReader(f))
    byid={x["image_id"]:x for x in roster}
    ids=d["image_ids"].astype(str);pids=d["patient_ids"].astype(str);y=d["labels"].astype(int)
    check(set(byid)==set(ids) and len(roster)==813,"Public roster mismatch")
    for i in range(813):
        check(str(byid[ids[i]]["patient_id"])==pids[i],"P roster patient mismatch")
        check(int(byid[ids[i]]["weak_P"])==y[i],"P roster class mismatch")
    salt=io.contract["selection"]["salt"]
    keys=[hashlib.sha256((salt+"|"+p+"|"+im).encode()).hexdigest() for p,im in zip(pids,ids)]
    candidates=[]
    for p in sorted(set(pids)):
        ix=np.flatnonzero(pids==p)
        pos=ix[y[ix]==1]
        eligible=pos if len(pos) else ix
        candidates.append(min(eligible,key=lambda i:keys[i]))
    candidates=sorted(candidates,key=lambda i:keys[i])
    check(len(candidates)==672,"P patient count")
    zd=d["z"].transpose(1,0,2).reshape(813,64).astype(float)
    zs=[zd[candidates],r["z"][candidates].astype(float)]
    scales=[float(np.sqrt(np.mean(np.sum(z*z,axis=1)))) for z in zs]
    check(min(scales)>1e-12,"Selection block scale")
    z=np.concatenate([a/(s*np.sqrt(2)) for a,s in zip(zs,scales)],axis=1)
    chosen=[j for j,i in enumerate(candidates) if y[i]==1]
    check(len(chosen)==6,"P positive patients")
    # Candidate order already fixes hash-based tie breaking.
    dist=np.min(np.stack([np.sum((z-z[j])**2,axis=1) for j in chosen]),axis=0)
    trace=[]
    while len(chosen)<128:
        dist[chosen]=-np.inf
        j=int(np.argmax(dist))
        check(np.isfinite(dist[j]) and j not in chosen,"Selection duplicate")
        trace.append({"step":len(chosen)+1,"candidate_index":j,"nearest_squared_distance":float(dist[j])})
        chosen.append(j)
        dist=np.minimum(dist,np.sum((z-z[j])**2,axis=1))
    ix=np.array([candidates[j] for j in chosen],dtype=np.int64)
    check(len(set(pids[ix]))==len(set(ids[ix]))==128 and int(y[ix].sum())==6,"Selected public128 scope")
    # Independent full-distance formulation validates all greedy choices.
    dmat=np.maximum(np.sum(z*z,axis=1)[:,None]+np.sum(z*z,axis=1)[None,:]-2*z@z.T,0)
    second=[j for j,i in enumerate(candidates) if y[i]==1]
    for item in trace:
        nearest=dmat[:,second].min(axis=1);nearest[second]=-np.inf
        check(int(np.argmax(nearest))==item["candidate_index"],"Independent greedy selection differs")
        second.append(item["candidate_index"])
    check(second==chosen,"Selection verifier")
    root=Path(io.contract["public_image_root"]).resolve()
    rows=[];total=0
    for i in ix:
        io.check_time();p=(root/ids[i]).resolve()
        check(p.parent==root and p.is_file(),"Public image path changed")
        with Image.open(p) as im:
            check(im.size==(1024,1024),"Public native image geometry changed")
            mode=im.mode
        h=sha(p);total+=p.stat().st_size
        rows.append({"path":str(p),"image_id":ids[i],"patient_id":pids[i],"label":int(y[i]),
                     "sha256":h,"cache_row":int(i),"native_width":1024,"native_height":1024,"mode":mode})
    np.savez_compressed(out/"public128_selection.npz",indices=ix,DINO=zd[ix],
        ResNet18=r["z"][ix].astype(float),image_ids=ids[ix],patient_ids=pids[ix],labels=y[ix])
    dump(out/"public128_manifest.json",{"status":"PUBLIC128_FROZEN","created":now(),"selection":io.contract["selection"],
        "rows":rows,"public_patients":128,"positive_patients":6,"block_RMS":scales,"trace":trace,
        "selection_uses_DP":False,"selection_uses_DenseNet":False,"selection_uses_V":False,
        "native_source_bytes_referenced":total,"new_image_copies":0,"new_model_forwards":0,
        "preprocessing":"Original native P images; same pinned tensor/condition paths as P caches and existing PNGs, not resized/exported anew",
        "independent_full_distance_selection_identical":True,"seconds":time.monotonic()-tick})
    io.verify();dump(out/"selection_inputs.json",io.receipts)
    write_seal(out,"selection_seal.json","PUBLIC128_SEALED_BEFORE_TARGET_ACCESS",
        ["public128_selection.npz","public128_manifest.json","selection_inputs.json"])
    print('{"phase":"PUBLIC128_SEALED","patients":128,"positive_patients":6,"model_forwards":0}',flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,required=True);run(p.parse_args().out)

