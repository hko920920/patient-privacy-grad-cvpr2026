"""Eight zero-anchored label solves; no V or DenseNet features accessed."""
import argparse,csv,time,sys
from pathlib import Path
import numpy as np
from scipy.optimize import lsq_linear
from carrier_common import Inputs,check,dump,sha,now,packet,check_seal,write_seal,same_order,export_packet

def solve(xp,xs,w,targets,models):
    blocks=[];responses=[];scales={};ops={};op_error=0.
    for m in models:
        z=xs[m];check(z.shape[0]==128,"Carrier size")
        op=np.linalg.solve(z.T@z/128+.1*np.eye(z.shape[1]),z.T/128)
        dual=z.T@np.linalg.solve(z@z.T+12.8*np.eye(128),np.eye(128))
        op_error=max(op_error,float(abs(op-dual).max()))
        check(op_error<1e-10,"Ridge operator primal/dual mismatch")
        ops[m]=op;t=xp[m]@targets[m]
        s=float(np.sqrt(np.sum(w*t*t)));check(s>1e-12,"Degenerate public target RMS")
        scales[m]=s
        blocks.append(np.sqrt(w[:,None]/len(models))*(xp[m]@op)/s)
        responses.append(np.sqrt(w/len(models))*t/s)
    A=np.vstack(blocks);t=np.concatenate(responses)
    eta=.001*np.sum(A*A)/128
    check(eta>0 and np.isfinite(eta),"Invalid regularization")
    design=np.vstack([A,np.sqrt(eta)*np.eye(128)])
    response=np.concatenate([t,np.zeros(128)])
    fit=lsq_linear(design,response,bounds=(-1,1),tol=1e-12,max_iter=500)
    check(fit.success and np.isfinite(fit.x).all(),"Label solve failed")
    l=fit.x;check(abs(l).max()<=1,"Label outside box")
    gradient=design.T@(design@l-response)
    kkt=float(abs(l-np.clip(l-gradient,-1,1)).max())
    check(kkt<=1e-7,"Independent projected-gradient optimality")
    initial=float(t@t);final=float(np.sum((design@l-response)**2))
    check(final<=initial+1e-12,"Label fit worsened objective")
    detail={"models":models,"target_RMS":scales,"eta":float(eta),"anchor":"zero",
        "initial_objective_zero_label":initial,"final_objective":final,
        "relative_target_RMSE":{m:float(np.linalg.norm(b@l-v)*np.sqrt(len(models))) for m,b,v in zip(models,blocks,responses)},
        "iterations":int(fit.nit),"solver_status":int(fit.status),"solver_optimality":float(fit.optimality),
        "independent_projected_gradient_max_abs":kkt,"ridge_primal_dual_max_abs":op_error,
        "labels_at_bound":int(np.sum(abs(l)>1-1e-6)),"labels_positive":int(np.sum(l>0))}
    return l,detail

def run(out):
    tick=time.monotonic();io=Inputs(out,"construct")
    check(not (out/"label_seal.json").exists(),"Labels already sealed")
    check_seal(out,"selection_seal.json","PUBLIC128_SEALED_BEFORE_TARGET_ACCESS")
    sel=packet(out/"public128_selection.npz")
    pubmanifest=__import__("json").loads((out/"public128_manifest.json").read_text(encoding="utf-8"))
    ps=io.nz("P_DINO");pr=io.nz("P_RN");same_order(ps,pr)
    check(set(ps["roles"])==set(pr["roles"])=={"P"},"Construction only public features")
    xp={"DINO":ps["z"].transpose(1,0,2).reshape(813,64).astype(float),"ResNet18":pr["z"].astype(float)}
    ix=sel["indices"]
    for m in xp:check(np.array_equal(xp[m][ix],sel[m]),"Frozen public features changed")
    sys.path.insert(0,io.contract["historical_code_dir"])
    from fixed_image_compilation import pw
    w=pw(ps);check(abs(w.sum()-1)<1e-12 and np.all(w>0),"Public patient/class weights")
    check(np.array_equal(w,pw(pr)),"Encoder weighting mismatch")
    targetweights=io.nz("prior_target_weights")
    priorreceipt=io.js("prior_label_seal")
    check(sha(io.path("prior_target_weights"))==priorreceipt["files"]["target_weights.npz"],"Step1 target seal")
    synD=io.nz("synthetic_DINO")
    labels={};details={};packetfiles=[];target_error=0.;rms_by_draw={}
    (out/"label_packets").mkdir(exist_ok=False)
    for draw in ("DP1","DP2"):
        io.check_time()
        mu=io.nz(draw+"_target")["target"].reshape(2,64).astype(float)
        wd=np.linalg.solve(xp["DINO"].T@(w[:,None]*xp["DINO"])+.1*np.eye(64),.5*(mu[1]-mu[0]))
        target_error=max(target_error,float(abs(wd-targetweights[draw+"_DINO"]).max()))
        check(target_error<1e-10,"Frozen target mismatch")
        targets={"DINO":targetweights[draw+"_DINO"],"ResNet18":targetweights[draw+"_MT_RN"]}
        sr=io.nz(draw+"_synthetic_RN")
        check(np.array_equal(sr["labels"],[0]*64+[1]*64),"Synthetic label row order")
        synthetic={"DINO":synD[draw].astype(float),"ResNet18":sr["z"].astype(float)}
        with io.path(draw+"_PNG_csv").open(encoding="utf-8",newline="") as f:rows=list(csv.DictReader(f))
        manifest=io.js(draw+"_PNG_manifest");bank=Path(io.contract["banks"][draw]).resolve()
        check(len(rows)==128 and [int(r["label"]) for r in rows]==[0]*64+[1]*64,"Original bank order")
        image_rows=[]
        for r in rows:
            p=(bank/r["file"]).resolve();check(p.is_relative_to(bank),"PNG outside bank")
            h=sha(p);check(h==manifest["files_sha256"][r["file"]],"Synthetic image hash mismatch")
            image_rows.append({"path":str(p),"image_id":r["file"],"label":int(r["label"]),"sha256":h})
        for carrier,xs,imrows in (("SYN",synthetic,image_rows),("PUBLIC",sel,pubmanifest["rows"])):
            for recipe,models in (("RN_ONLY",["ResNet18"]),("JOINT",["DINO","ResNet18"])):
                name=draw+"_"+carrier+"_"+recipe
                l,fit=solve(xp,xs,w,targets,models)
                labels[name]=l;details[name]=fit
                file="label_packets/"+name+".csv"
                export_packet(imrows,l,out/file);packetfiles.append(file)
                print('{"phase":"LABELS_CONSTRUCTED","cell":"'+name+'"}',flush=True)
        for recipe in ("RN_ONLY","JOINT"):
            check(details[draw+"_SYN_"+recipe]["target_RMS"]==details[draw+"_PUBLIC_"+recipe]["target_RMS"],
                  "Carrier targets differ")
    check(len(labels)==8,"Unexpected label count")
    np.savez_compressed(out/"soft_labels.npz",**labels)
    io.verify()
    dump(out/"construction_inputs.json",io.receipts)
    dump(out/"construction_results.json",{"scope":"Step3 carrier comparison with zero anchor, targets frozen from Step1",
        "fits":details,"existing_DINO_target_max_abs":target_error,"new_label_packets":8,
        "new_pixel_updates":0,"new_Q_access":0,"new_DP_releases":0,"V_access":False,"DenseNet_access":False,
        "new_model_forwards":0,"seconds":time.monotonic()-tick})
    files=["soft_labels.npz","construction_inputs.json","construction_results.json","public128_selection.npz","public128_manifest.json","selection_seal.json"]+packetfiles
    write_seal(out,"label_seal.json","ALL_EIGHT_LABELS_SEALED_BEFORE_EVALUATION",files,
        selection_sha256=sha(out/"selection_seal.json"),label_packets=8)
    print('{"phase":"ALL_EIGHT_LABELS_SEALED","V_access":false}',flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,required=True);run(p.parse_args().out)

