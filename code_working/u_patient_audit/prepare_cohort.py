"""Read only public-development source images; freeze a separate U pilot cohort."""
import itertools, json, time
from collections import defaultdict, Counter
import numpy as np
from scipy.fft import dctn
from PIL import Image
from .common import *
from data_pipeline.nih_cxr14_model_input import load_native_grayscale

def fingerprint(gray):
    pixels=np.asarray(gray,dtype=np.uint8)
    tiny=np.asarray(gray.resize((64,64),Image.Resampling.LANCZOS),dtype=np.float32)/255
    dct=dctn(np.asarray(gray.resize((32,32),Image.Resampling.LANCZOS),dtype=np.float32),norm="ortho")[:8,:8].flatten()
    bits=dct[1:]>np.median(dct[1:])
    phash=sum(int(b)<<i for i,b in enumerate(bits))
    return hashlib.sha256(pixels.tobytes()).hexdigest(),phash,tiny

def near_duplicate(a,b):
    if a["pixel_sha256"]==b["pixel_sha256"]:
        return True
    if (a["_phash"]^b["_phash"]).bit_count()>6:
        return False
    x=a["_tiny"].ravel(); y=b["_tiny"].ravel()
    xc=x-x.mean(); yc=y-y.mean()
    corr=float(np.dot(xc,yc)/max(float(np.linalg.norm(xc)*np.linalg.norm(yc)),1e-12))
    return corr>=0.995 and float(np.abs(x-y).mean())<=0.02

def choose_distinct(rows, n):
    chosen=[]
    for row in sorted(rows,key=lambda r:stable("image-choice",r["patient_id"],r["image_id"])):
        if any(row["followup_no"]==q["followup_no"] or near_duplicate(row,q) for q in chosen):
            continue
        chosen.append(row)
        if len(chosen)==n:
            return chosen
    return []

def strip(row,role,group="",record_role=""):
    return {k:row[k] for k in ["image_id","patient_id","image_filename","followup_no",
            "patient_age","patient_sex","finding_labels","target_patient","sha256","pixel_sha256"]} | {
            "source_partition":"public_development","eval_role":role,
            "assignment_group":group,"record_role":record_role}

def main():
    output=RUN/"cohort"
    if (output/"lock.json").exists():
        raise RuntimeError("cohort already frozen; verify, do not overwrite")
    start=time.perf_counter()
    source=read_csv(SOURCE)
    public=[r for r in source if r["partition"]=="public_development"]
    inventory={r["image_id"]:r for r in read_csv(INVENTORY)}
    inv_patients=defaultdict(set)
    for r in inventory.values():
        inv_patients[r["sha256"].lower()].add(r["patient_id"])
    bypatient=defaultdict(list); rejected=Counter(); audited=[]
    for i,row in enumerate(public):
        path=IMAGES/row["image_filename"]
        if not path.is_file():
            rejected["not_acquired"]+=1;continue
        assert path.resolve().parent==IMAGES.resolve()
        assert row["view"]=="PA" and row["followup_no"].isdigit()
        got=digest(path)
        assert row["image_id"] in inventory
        assert got==inventory[row["image_id"]]["sha256"].lower(),"source image hash mismatch"
        if len(inv_patients[got])>1:
            rejected["file_duplicate_cross_patient"]+=1;continue
        gray,_=load_native_grayscale(path)
        pixel,phash,tiny=fingerprint(gray)
        r=dict(row,sha256=got,pixel_sha256=pixel,_phash=phash,_tiny=tiny)
        bypatient[r["patient_id"]].append(r);audited.append(r)
        if (i+1)%500==0:
            print(json.dumps({"phase":"source_verification","processed":i+1,"total":len(public)}),flush=True)
    pixel_patients=defaultdict(set)
    for r in audited:
        pixel_patients[r["pixel_sha256"]].add(r["patient_id"])
    for patient,rows in list(bypatient.items()):
        clean=[r for r in rows if len(pixel_patients[r["pixel_sha256"]])==1]
        rejected["pixel_duplicate_cross_patient"]+=len(rows)-len(clean)
        bypatient[patient]=clean
    eligible={p:choose_distinct(rows,4) for p,rows in bypatient.items()}
    eligible={p:rows for p,rows in eligible.items() if rows}
    patient_order=sorted(eligible,key=lambda p:stable("patient-cohort",p))
    n=min(400,len(patient_order)//20*20)
    assert n>=300, f"insufficient U cohort {n}; revise design before training"
    cohort_patients=patient_order[:n]
    counts={"fit":n//5,"selection":n//10,"calibration":int(n*.35)//2*2}
    counts["test"]=n-sum(counts.values())
    roles={}
    position=0
    for role,num in counts.items():
        assert num%2==0
        selected=cohort_patients[position:position+num];position+=num
        # Pair within source target/sex strata, then coin-flip inclusion within pairs.
        strata=defaultdict(list)
        for p in selected:
            strata[(eligible[p][0]["target_patient"],eligible[p][0]["patient_sex"])].append(p)
        pairs=[]; leftovers=[]
        for key,group in sorted(strata.items()):
            group=sorted(group,key=lambda p:stable("assignment-order",role,p))
            pairs.extend(zip(group[0:len(group)//2*2:2],group[1:len(group)//2*2:2]))
            if len(group)%2:leftovers.append(group[-1])
        leftovers.sort(key=lambda p:stable("leftover",role,p))
        pairs.extend(zip(leftovers[::2],leftovers[1::2]))
        for p,q in pairs:
            if seed("assignment-coin",role,p,q)%2:p,q=q,p
            roles[p]=(role,"A");roles[q]=(role,"B")
    eval_rows=[]
    for p in cohort_patients:
        role,group=roles[p]
        for j,row in enumerate(eligible[p]):
            eval_rows.append(strip(row,role,group,"train_candidate" if j<2 else "U_observed"))
    pool={p:choose_distinct(rows,2) for p,rows in bypatient.items() if p not in roles}
    pool={p:rows for p,rows in pool.items() if rows}
    pool_order=sorted(pool,key=lambda p:stable("auxiliary-role",p))
    assert len(pool_order)>=352, f"insufficient auxiliary patients {len(pool_order)}"
    auxiliary=[]
    for role,patients in [("reference",pool_order[:64]),("quality",pool_order[64:96]),("background",pool_order[96:352])]:
        for p in patients:
            auxiliary.extend(strip(r,role,record_role=role) for r in pool[p])
    all_rows=eval_rows+auxiliary
    # Screen cross-patient perceptual matches across ALL selected roles.
    selected_by_id={r["image_id"]:r for r in audited}
    selected=[selected_by_id[r["image_id"]] for r in all_rows]
    cross_near=[]
    for i,a in enumerate(selected):
        for b in selected[i+1:]:
            if a["patient_id"]!=b["patient_id"] and near_duplicate(a,b):
                cross_near.append((a["image_id"],b["image_id"]))
    assert not cross_near, f"cross-patient near duplicates require review: {len(cross_near)}"
    output.mkdir(parents=True,exist_ok=True)
    write_csv(output/"evaluation_images.csv",eval_rows)
    write_csv(output/"auxiliary_images.csv",auxiliary)
    for model,group in [("model_1","A"),("model_2","B")]:
        rows=[r for r in eval_rows if r["record_role"]=="train_candidate" and r["assignment_group"]==group]
        rows += [r for r in auxiliary if r["eval_role"]=="background"]
        write_csv(output/(model+"_train.csv"),rows)
    audit_rows=[{k:r[k] for k in ["image_id","patient_id","image_filename","sha256","pixel_sha256"]} for r in audited]
    write_csv(output/"source_audit.csv",audit_rows)
    summary={
        "schema":"cvpr-u-cohort/v1","scope":"public-development controlled incremental-finetuning pilot",
        "source_public_patients":len({r["patient_id"] for r in public}),
        "source_public_images":len(public),"verified_images":len(audited),
        "rejected_images":dict(rejected),"eligible_four_distinct_records":len(eligible),
        "evaluation_patients":n,"evaluation_images":len(eval_rows),
        "role_patient_counts":counts,"per_model_member_patients":n//2,
        "reference_patients":64,"quality_patients":32,"background_patients":256,
        "per_model_training_images":n+512,"k_train":2,"m_observed":2,"overlap_U":0,
        "cross_patient_near_duplicate_matches":len(cross_near),
        "duplicate_rule":{"phash_hamming_max":6,"correlation_min":0.995,"MAE_max":0.02},
        "acquisition_limitation":"Different NIH followup_no values; no acquisition timestamps or study UID to certify independence.",
        "sampling_limitation":"Patients with >=4 verified distinct available PA records only; not all patients.",
        "thesis_contract_modified":False,"seconds":time.perf_counter()-start,
        "medical_training_started":False
    }
    write_json(output/"summary.json",summary)
    files={p.name:digest(p) for p in output.glob("*.csv")}
    files["summary.json"]=digest(output/"summary.json")
    source_paths=[SOURCE,INVENTORY]+[SOURCE.parent/(k+"_private.csv") for k in ["k2","k5","k10"]]
    lock={"files":files,"source_files":{p.relative_to(ROOT).as_posix():digest(p) for p in source_paths},
          "selection_code_sha256":digest(Path(__file__)),"selection_salt":SALT}
    write_json(output/"lock.json",lock)
    # Diagnostic grid contains public sources only, anonymized labels.
    chosen_examples=cohort_patients[:4]
    canvas=Image.new("RGB",(4*256,4*256))
    for y,p in enumerate(chosen_examples):
        for x,r in enumerate(eligible[p]):
            gray,_=load_native_grayscale(IMAGES/r["image_filename"])
            canvas.paste(gray.resize((256,256),Image.Resampling.LANCZOS).convert("RGB"),(x*256,y*256))
    canvas.save(output/"selection_grid.png")
    print(json.dumps(summary),flush=True)

if __name__=="__main__":main()

