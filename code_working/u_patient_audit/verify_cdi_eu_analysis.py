"""Independent four-cell saved-feature, fitting and rank/statistical verification."""
from __future__ import annotations
import argparse, hashlib, json, time, warnings
from pathlib import Path
import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from .common import ROOT, RUN

ANALYZER_SHA="271061072d98e736a34030533ca00d22b40055045327563c83720418c56e08bc"
CONTRACT_SHA="bc2497d17b3110ec8b994eaa541d4eda00891ff5d30215a9aff26b7aa5f1db43"
OUT=RUN/"baseline_screen_20260914/cdi_e_cohort_v1/eu_analysis_v1"
def read(p):return json.loads(Path(p).read_text(encoding="utf-8-sig"))
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def need(ok,label):
    if not ok:raise AssertionError(label)
def tight(a,b,label):
    a,b=np.asarray(a,dtype=np.float64),np.asarray(b,dtype=np.float64)
    need(a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all()
         and np.allclose(a,b,rtol=2e-12,atol=2e-12),label)
def design(x,rep):
    if rep=="image26":return x[:,:,:26].reshape(-1,26)
    if rep=="mean26":return x[:,:,:26].mean(1)
    need(rep in ("meanmax52","meanmax54"),"declared representation")
    z=x[:,:,:26] if rep=="meanmax52" else x
    return np.c_[z.mean(1),z.max(1)]
def pool(prob,spec):
    if spec["representation"]!="image26":return prob
    p=prob.reshape(-1,2)
    return p.mean(1) if spec["patient_probability_pool"]=="mean" else p.max(1)
def rank_auc(y,s):
    pos,neg=s[y==1],s[y==0]
    return float(np.mean((pos[:,None]>neg).astype(float)+.5*(pos[:,None]==neg)))
def fresh_fit(x,y,spec,C,cfg):
    z=design(x,spec["representation"])
    labels=np.repeat(y,2) if spec["representation"]=="image26" else y
    scaler=StandardScaler().fit(z)
    clf=LogisticRegression(C=C,solver=cfg["solver"],penalty="l2",
        max_iter=cfg["max_iter"],random_state=cfg["random_state"],class_weight=None)
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always");clf.fit(scaler.transform(z),labels)
    return scaler,clf,ws
def run(out):
    begun=time.perf_counter();pro=read(out/"provenance.json");protocol=read(out/"protocol.json")
    need(pro["source_sha256"]==ANALYZER_SHA==digest(protocol["source_path"]),"frozen analyzer")
    need(pro["protocol_sha256"]==digest(out/"protocol.json"),"analysis protocol binding")
    need(pro["analysis_contract_sha256"]==CONTRACT_SHA==digest(protocol["analysis_contract_path"]),"frozen pre-E contract")
    c=read(out/"contract_copy.json")
    need(c==read(protocol["analysis_contract_path"]),"contract equality")
    def hashes():
        for p,h in pro["input_sha256"].items():need(digest(p)==h,"input changed "+p)
        for n,h in pro["output_sha256"].items():need(digest(out/n)==h,"output changed "+n)
    hashes()
    roles={r:[p for p in c["patient_rows"] if p["role"]==r] for r in ("fit","selection")}
    patient={p["patient_id"]:p for p in c["patient_rows"]}
    need(len(patient)==120 and len(roles["fit"])==80 and len(roles["selection"])==40,"patient roles")
    fitids=[p["patient_id"] for p in roles["fit"]];fit_index={p:i for i,p in enumerate(fitids)}
    val=[]
    for i,fold in enumerate(c["folds"]):
        tr,va=set(fold["train_patient_ids"]),set(fold["validation_patient_ids"])
        need(fold["fold"]==i and len(tr)==64 and len(va)==16 and not tr&va and tr|va==set(fitids),"patient-disjoint fold")
        need(sum(patient[p]["group"]=="A" for p in va)==8,"balanced folds")
        val+=list(va)
    need(len(val)==len(set(val))==80,"five disjoint validation folds")
    methods={s["id"]:s for s in c["methods"]};need(len(methods)==18,"18 methods")
    models=c["models"];x={};y={}
    for scenario in ("E","U"):
        d=Path(c["inputs"][scenario]["directory"]);v=read(d/c["inputs"][scenario]["verification_file"])
        need(v["status"]==c["inputs"][scenario]["required_status"] and v["results_sha256"]==digest(d/"results.json"),"raw PASS bound")
        rows=read(d/"results.json");ix={(r["model"],r["image_id"]):r for r in rows}
        expected={(m,i) for m in models for p in patient.values() for i in p["images"][scenario]}
        need(len(rows)==len(ix)==480 and set(ix)==expected,"complete E/U features")
        for m in models:
            for role,ps in roles.items():
                labels=[int(p["group"]==("A" if m=="model_1" else "B")) for p in ps]
                for p,label in zip(ps,labels):
                    for image in p["images"][scenario]:
                        row=ix[(m,image)]
                        need(row["member"]==int(row["patient_training_exposures"]>0)==label,"patient label/exposure")
                        need(row["image_member"]==int(row["actual_training_exposures"]>0)==(label if scenario=="E" else 0),"image exposure")
                        need(row["patient_id"]==p["patient_id"] and row["scenario"]==scenario and row["eval_role"]==role,"image metadata")
                x[(scenario,m,role)]=np.asarray([[ix[(m,i)]["features"]+
                    [ix[(m,i)]["modules"]["noise_optim"]["optimizer"]["fun"]] for i in p["images"][scenario]] for p in ps])
                y[(m,role)]=np.asarray(labels)
    parrows=read(out/"fit_parameters.json");cvrows=read(out/"cv_results.json")
    params={(p["fit_scenario"],p["model"],p["method"]):p for p in parrows}
    cvs={(p["fit_scenario"],p["model"],p["method"]):p for p in cvrows}
    lrkeys={(s,m,n) for s in ("E","U") for m in models for n,sp in methods.items() if sp["kind"]=="logistic_regression"}
    need(len(parrows)==len(cvrows)==40 and set(params)==set(cvs)==lrkeys,"40 exact fitted scorers")
    old=Path(c["inputs"]["U"]["directory"])/"analysis_v1"
    oldp={(r["model"],r["method"]):r for r in read(old/"fit_parameters.json")}
    oldcv={(r["model"],r["method"]):r for r in read(old/"cv_results.json")}
    refit_count=0;cv_refit_count=0
    cfg=c["logistic_regression"];eps=np.finfo(float).eps
    for key,entry in params.items():
        fs,m,n=key;sp=methods[n];p=entry["parameters"];z=design(x[(fs,m,"fit")],sp["representation"])
        need(entry["fitting_patient_ids"]==fitids and entry["fitting_rows"]==len(z)==p["scaler_n_samples_seen"],"fit-only scaler rows")
        need(p["classes"]==[0,1],"membership direction")
        tight(p["scaler_mean"],z.mean(0),"fit mean");tight(p["scaler_var"],z.var(0),"fit var")
        var,mean=np.asarray(p["scaler_var"]),np.asarray(p["scaler_mean"])
        constant=var<=len(z)*eps*var+(len(z)*mean*eps)**2
        tight(p["scaler_scale"],np.where(constant,1,np.sqrt(var)),"fit scales")
        cv=cvs[key];need(cv["chosen_C"]==entry["chosen_C"],"CV choice")
        if fs=="U":
            for row,reference in ((entry,oldp[(m,n)]),(cv,oldcv[(m,n)])):
                need(row["reused_original_U"] is True,"U reused")
                need({k:v for k,v in row.items() if k not in ("fit_scenario","reused_original_U")}==reference,"original U params/CV unchanged")
        else:
            need(entry["reused_original_U"] is False and cv["reused_original_U"] is False,"E newly fit")
            sc,cl,ws=fresh_fit(x[(fs,m,"fit")],y[(m,"fit")],sp,entry["chosen_C"],cfg)
            need(not ws and not entry["warnings"],"E no convergence warnings")
            tight(cl.coef_,p["coef"],"independent E all-fit coefficients")
            tight(cl.intercept_,p["intercept"],"independent E all-fit intercept")
            need(cl.n_iter_.tolist()==p["n_iter"],"E fit iterations");refit_count+=1
        if sp["C_policy"]=="fixed":
            need(entry["chosen_C"]==1 and not cv["candidates"],"fixed C1");continue
        need([r["C"] for r in cv["candidates"]]==[.01,.1,1.,10.,100.],"fixed grid")
        for candidate in cv["candidates"]:
            losses=[];need(len(candidate["folds"])==5,"five CV folds")
            for fold,ref in zip(candidate["folds"],c["folds"]):
                need(fold["fold"]==ref["fold"] and fold["validation_patient_ids"]==ref["validation_patient_ids"],"saved CV IDs")
                va=np.asarray([fit_index[p] for p in ref["validation_patient_ids"]])
                prob=np.asarray(fold["validation_probabilities"])
                need(prob.shape==(16,) and np.isfinite(prob).all() and ((prob>=0)&(prob<=1)).all(),"valid fold probabilities")
                if fs=="E":
                    tr=np.asarray([fit_index[p] for p in ref["train_patient_ids"]])
                    sc,cl,ws=fresh_fit(x[(fs,m,"fit")][tr],y[(m,"fit")][tr],sp,candidate["C"],cfg)
                    pred=pool(cl.predict_proba(sc.transform(design(x[(fs,m,"fit")][va],sp["representation"])))[:,1],sp)
                    tight(prob,pred,"independent E CV refit probabilities")
                    need(not ws and not fold["warnings"],"CV convergence");cv_refit_count+=1
                yy=y[(m,"fit")][va];pp=np.clip(prob,eps,1-eps)
                loss=np.mean(-yy*np.log(pp)-(1-yy)*np.log1p(-pp))
                tight(fold["patient_log_loss"],loss,"patient logloss");losses.append(loss)
            tight(candidate["mean_patient_log_loss"],np.mean(losses),"CV mean")
        need(entry["chosen_C"]==min(cv["candidates"],key=lambda a:(a["mean_patient_log_loss"],a["C"]))["C"],"min logloss choice")
    allscores={};checked_rows=0
    for filename,role in (("predictions.json","selection"),("fit_predictions.json","fit")):
        rows=read(out/filename)
        expected={(fs,es,m,n,p["patient_id"]) for fs in ("E","U") for es in (("E","U") if role=="selection" else (fs,))
                  for m in models for n in methods for p in roles[role]}
        ix={(r["fit_scenario"],r["evaluation_scenario"],r["model"],r["method"],r["patient_id"]):r for r in rows}
        need(len(rows)==len(ix)==len(expected)==5760 and set(ix)==expected,"complete prediction identities")
        for fs in ("E","U"):
            for es in (("E","U") if role=="selection" else (fs,)):
                for m in models:
                    for n,sp in methods.items():
                        xx=x[(es,m,role)]
                        if sp["kind"]=="scalar":
                            raw=sp["direction"]*xx[:,:,sp["feature_index"]]
                            pred=raw.mean(1) if sp["pool_after_direction"]=="mean" else raw.max(1)
                        else:
                            p=params[(fs,m,n)]["parameters"];z=design(xx,sp["representation"])
                            logit=((z-np.asarray(p["scaler_mean"]))/np.asarray(p["scaler_scale"]))@np.asarray(p["coef"])[0]+p["intercept"][0]
                            pred=pool(expit(logit),sp)
                        stored=[]
                        for pat,label in zip(roles[role],y[(m,role)]):
                            r=ix[(fs,es,m,n,pat["patient_id"])]
                            need(r["group"]==pat["group"] and r["member"]==label and r["role"]==role,"prediction metadata")
                            stored.append(r["score"])
                        tight(stored,pred,"independent saved-param/scalar predictions")
                        allscores[(fs,es,m,n,role)]=np.asarray(stored);checked_rows+=len(stored)
    ps=roles["selection"];aa=np.flatnonzero([p["group"]=="A" for p in ps]);bb=np.flatnonzero([p["group"]=="B" for p in ps])
    rng=np.random.default_rng(260914)
    indices=np.asarray([np.r_[rng.choice(aa,20,replace=True),rng.choice(bb,20,replace=True)] for _ in range(2000)])
    br=read(out/"bootstrap_resamples.json")
    need(br["selection_patient_order"]==[p["patient_id"] for p in ps] and np.array_equal(indices,br["indices"]),"independent paired bootstrap regeneration")
    counts=np.asarray([np.bincount(i,minlength=40) for i in indices],dtype=float)
    analysis=read(out/"analysis.json");points={};boots={}
    resultkeys={(s,e,m,n) for s in ("E","U") for e in ("E","U") for m in models for n in methods}
    need(len(analysis["results"])==144 and {(r["fit_scenario"],r["evaluation_scenario"],r["model"],r["method"]) for r in analysis["results"]}==resultkeys,"144 cells")
    for r in analysis["results"]:
        key=(r["fit_scenario"],r["evaluation_scenario"],r["model"],r["method"]);fs,es,m,n=key
        yy=y[(m,"selection")];s=allscores[(*key,"selection")]
        pos,neg=np.flatnonzero(yy),np.flatnonzero(1-yy)
        credits=(s[pos,None]>s[None,neg]).astype(float)+.5*(s[pos,None]==s[None,neg])
        points[key]=float(credits.mean())
        boots[key]=np.einsum("bi,ij,bj->b",counts[:,pos],credits,counts[:,neg])/400
        tight(r["selection_patient_AUC"],points[key],"independent half-tie AUC")
        tight(r["selection_patient_AUC_CI95"],np.quantile(boots[key],[.025,.975]),"independent weighted-rank bootstrap")
        tight(r["fit_patient_AUC_descriptive_only"],rank_auc(y[(m,"fit")],allscores[(fs,fs,m,n,"fit")]),"descriptive fit AUC")
    for field,contractfield,expected_n in (("within_cell_contrasts","original_within_cell_contrasts",48),("cross_cell_contrasts","cross_cell_contrasts",108)):
        definitions={r["id"]:r for r in c[contractfield]};rows=analysis[field]
        need(len(rows)==expected_n,"contrast counts");seen=set()
        for r in rows:
            d=definitions[r["id"]]
            need(all(r[k]==v for k,v in d.items()),"predeclared contrast identity")
            if field=="within_cell_contrasts":
                prefix=(r["fit_scenario"],r["evaluation_scenario"],r["model"])
                a,b=(*prefix,d["a"]),(*prefix,d["b"]);identifier=(*prefix,r["id"])
            else:
                a=(*[d["a"][k] for k in ("fit_scenario","evaluation_scenario")],r["model"],r["method"])
                b=(*[d["b"][k] for k in ("fit_scenario","evaluation_scenario")],r["model"],r["method"])
                identifier=(r["model"],r["method"],r["id"])
            need(identifier not in seen,"unique contrasts");seen.add(identifier)
            tight(r["selection_delta_AUC"],points[a]-points[b],"paired difference")
            tight(r["paired_CI95"],np.quantile(boots[a]-boots[b],[.025,.975]),"paired contrast CI")
    oldscores=read(old/"predictions.json")
    for r in oldscores:
        if r["method"] not in methods:continue
        pids=[p["patient_id"] for p in roles[r["role"]]]
        need(allscores[("U","U",r["model"],r["method"],r["role"])][pids.index(r["patient_id"])]==r["score"],"U predictions exactly unchanged")
    need(refit_count==20 and cv_refit_count==250 and checked_rows==11520,"verification scope counts")
    need(analysis["overall_stage2_gate"] is None and analysis["new_target_training"] is False,"stage2 comparison scope")
    hashes()
    result=dict(status="PASS_EU_SAVED_PREDICTIONS_FIT_AND_STATISTICS",research_stage=2,
        prediction_rows=checked_rows,AUC_metrics=144,within_cell_contrasts=48,cross_cell_contrasts=108,
        independently_refitted_E_scorers=20,independently_refitted_E_CV_fits=250,
        U_fitting_repeated=False,U_parameters_CV_predictions_exact=True,bootstrap_resamples=2000,
        new_GPU_calls=0,causal_claim=False,conditional_intervals_only=True,
        verification_code_sha256=digest(__file__),analysis_provenance_sha256=digest(out/"provenance.json"),
        analysis_sha256=digest(out/"analysis.json"),seconds_cpu=time.perf_counter()-begun)
    with (out/"verification.json").open("x",encoding="utf-8") as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(result),flush=True)
def self_test():
    rng=np.random.default_rng(47)
    for n in (8,40):
        y=np.arange(n)%2;s=rng.integers(0,5,n).astype(float)
        tight(rank_auc(y,s),roc_auc_score(y,s),"rank ties")
    x=rng.normal(size=(8,2,27));need(design(x,"image26").shape==(16,26),"image design")
    need(design(x,"meanmax52").shape==(8,52) and design(x,"meanmax54").shape==(8,54),"set design")
    import builtins,symtable
    table=symtable.symtable(Path(__file__).read_text(encoding="utf-8"),__file__,"exec")
    missing=[]
    def walk(scope):
        if scope.get_type()=="function":
            for sym in scope.get_symbols():
                if sym.is_referenced() and sym.is_global() and sym.get_name() not in globals() and not hasattr(builtins,sym.get_name()):
                    missing.append((scope.get_name(),sym.get_name()))
        for child in scope.get_children():walk(child)
    walk(table);need(not missing,"undefined globals "+repr(missing))
    print("PASS_INDEPENDENT_EU_VERIFIER_SYNTHETIC_RANK_REPRESENTATION_GLOBALS; no results inspected")
if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--self-test",action="store_true")
    parser.add_argument("--output",type=Path,default=OUT);parser.add_argument("--expected-code-sha256")
    a=parser.parse_args()
    if a.self_test:self_test()
    else:
        need(a.expected_code_sha256==digest(__file__),"externally frozen verifier source")
        run(a.output)

