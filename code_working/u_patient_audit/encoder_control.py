"""Target-independent encoder diagnostic on fit/selection patients only."""
import json,time
from collections import defaultdict
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from contextlib import nullcontext
from .common import *
from .models import load_cache

def main():
    verify_inputs();cache=load_cache()
    out=RUN/"encoder_control"
    if (out/"report.json").exists():raise RuntimeError("encoder diagnostic exists")
    rows=read_csv(RUN/"cohort/evaluation_images.csv");by=defaultdict(list)
    for r in rows:
        if r["eval_role"] in {"fit","selection"} and r["record_role"]=="U_observed":
            by[r["patient_id"]].append(r)
    features=[];patients=[]
    for p,rs in sorted(by.items()):
        vectors=[cache["encoder_features"][r["image_id"]].numpy() for r in rs]
        a,b=vectors
        features.append(np.concatenate([(a+b)/2,np.abs(a-b),[float(a@b)]]))
        patients.append({"patient_id":p,"role":rs[0]["eval_role"],"group":rs[0]["assignment_group"]})
    x=np.asarray(features)
    fit=np.array([p["role"]=="fit" for p in patients])
    scores=[];summary=[]
    start=time.perf_counter()
    with nullcontext():
        scaler=StandardScaler().fit(x[fit]);xs=scaler.transform(x)
        for model,group in [("model_1","A"),("model_2","B")]:
            y=np.array([int(p["group"]==group) for p in patients])
            clf=LogisticRegression(C=.1,max_iter=2000,solver="lbfgs").fit(xs[fit],y[fit])
            pred=clf.decision_function(xs)
            summary.append({"model":model,"fit_patients":int(fit.sum()),"selection_patients":int((~fit).sum()),
                "fit_AUC":float(roc_auc_score(y[fit],pred[fit])),
                "selection_AUC":float(roc_auc_score(y[~fit],pred[~fit]))})
            scores.extend(dict(p,model=model,member=int(label),score=float(value)) for p,label,value in zip(patients,y,pred))
    write_json(out/"scores.json",scores)
    report={"status":"COMPUTED_DEVELOPMENT_DIAGNOSTIC","features":"mean DINO, absolute inter-image difference, cosine",
        "regularization_C":.1,"normalization_fit_patients_only":True,
        "results":summary,"seconds_fitting_only":time.perf_counter()-start,
        "target_queries":0,"feature_extraction_cost":"shared DINO cache; not zero total computation",
        "calibration_test_used":False,"limitation":"Small selection set; random inclusion diagnostic does not eliminate all confounding.",
        "environment_note":"Optional threadpoolctl limiter failed on host OpenBLAS introspection; no packages changed; rerun without that optional limiter."}
    write_json(out/"report.json",report);print(json.dumps(report))
if __name__=="__main__":main()
