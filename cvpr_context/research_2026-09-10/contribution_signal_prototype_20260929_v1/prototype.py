"""2-4B public-only implementation. All clinical/private evaluation paths excluded."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, math, sys, time
import numpy as np

OUT=Path(__file__).resolve().parent
ROOT=OUT.parent
THESIS=ROOT.parent.parent
RADIUS=1.0-16*np.finfo(np.float64).eps
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):
    Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
def finite(x,name="array"):
    a=np.asarray(x,dtype=np.float64)
    if not np.isfinite(a).all(): raise ValueError(name+" must be finite")
    return a
def clip_rows(x,cap):
    a=finite(x); c=float(cap)
    if a.ndim!=2 or not math.isfinite(c) or c<=0: raise ValueError("Invalid rows/cap")
    norm=np.linalg.norm(a,axis=1)
    if not np.isfinite(norm).all(): raise ValueError("Norm overflow")
    return a*np.minimum(1,c/np.maximum(norm,np.finfo(float).tiny))[:,None]
def patient_query(values,present,caps):
    v=finite(values); p=np.asarray(present)
    c=finite(caps)
    if v.ndim!=3 or v.shape[1]!=2 or p.shape!=v.shape[:2] or p.dtype!=bool or c.shape!=(2,) or np.any(c<=0):
        raise ValueError("Invalid patient/class schema")
    if np.any(v[~p]!=0): raise ValueError("Absent class must be zero")
    clipped=np.stack([clip_rows(v[:,j],c[j]) for j in range(2)],axis=1)
    q=clipped/c[None,:,None]/np.sqrt(2.)
    norms=np.linalg.norm(q.reshape(len(v),-1),axis=1)
    q*=np.minimum(1,RADIUS/np.maximum(norms,np.finfo(float).tiny))[:,None,None]
    return q.reshape(len(v),-1),clipped
def decode(sums,caps,noisy_counts):
    s=finite(sums); c=finite(caps); n=finite(noisy_counts)
    if s.ndim not in (2,3) or s.shape[-2]!=2 or c.shape!=(2,) or np.any(c<=0) or n.shape!=s.shape[:-1]:
        raise ValueError("Invalid numerator/count schema")
    den=np.maximum(1.,n)
    means=np.sqrt(2.)*s*c[...,None]/den[...,None]
    return .5*(means[...,1,:]-means[...,0,:])
def compile_labels(q,y0,N,A,base):
    q=finite(q)
    if q.shape[-1]!=A.shape[0]: raise ValueError("Wrong target dimension")
    matrix=q if q.ndim==2 else q[None,:]
    v=np.linalg.solve(A,(matrix-base).T).T
    raw=y0[None,:]+v@N.T
    alpha=1/np.maximum(1,np.max(np.abs(raw),axis=1))
    labels=alpha[:,None]*raw
    return raw,labels,alpha

def read_contract():
    c=json.loads((OUT/"contract.json").read_text(encoding="utf-8"))
    seal=json.loads((OUT/"contract_seal.json").read_text(encoding="utf-8"))
    assert sha(OUT/"contract.json")==seal["sha256"],"Contract changed"
    for name,h in c["preserved_evidence"].items(): assert sha(ROOT/name)==h,name
    for dep in c["source_dependencies"].values(): assert sha(dep["path"])==dep["sha256"]
    return c

def load_public(c):
    data={}
    for name,spec in c["inputs"].items():
        p=Path(spec["path"])
        assert sha(p)==spec["sha256"],name
        with np.load(p,allow_pickle=False) as f: data[name]={k:f[k] for k in f.files}
    dp,rp,sel=data["DINO_P"],data["RN_P"],data["selection"]
    assert set(dp["roles"])==set(rp["roles"])=={"P"}
    for k in ("image_ids","patient_ids","labels"): assert np.array_equal(dp[k],rp[k])
    xd=dp["z"].transpose(1,0,2).reshape(813,64).astype(float)
    xr=rp["z"].astype(float)
    assert xd.shape==(813,64) and xr.shape==(813,128)
    assert np.array_equal(xd[sel["indices"]],sel["DINO"])
    assert np.array_equal(xr[sel["indices"]],sel["ResNet18"])
    ids=dp["patient_ids"]; labels=dp["labels"]; pts=np.unique(ids)
    present=np.zeros((len(pts),2),bool)
    rowsR=np.zeros((len(pts),2,128)); rowsD=np.zeros((len(pts),2,64))
    weights=np.zeros(len(ids))
    counts=np.array([len(np.unique(ids[labels==j])) for j in (0,1)])
    for i,p in enumerate(pts):
        for j in (0,1):
            mask=(ids==p)&(labels==j)
            if np.any(mask):
                present[i,j]=True
                rowsR[i,j]=xr[mask].mean(axis=0); rowsD[i,j]=xd[mask].mean(axis=0)
                weights[mask]=.5/counts[j]/int(mask.sum())
    assert tuple(counts)==(669,6) and present.sum()==675 and len(pts)==672
    assert abs(weights.sum()-1)<1e-12
    row_weights=np.zeros_like(present,dtype=float)
    row_weights[present]=np.repeat(1/len(pts)/present.sum(axis=1),present.sum(axis=1))
    assert abs(row_weights.sum()-1)<1e-12
    return {"XD":xd,"XR":xr,"XS":sel["DINO"].astype(float),"RS":sel["ResNet18"].astype(float),
            "weights":weights,"present":present,"rowsR":rowsR,"rowsD":rowsD,"counts":counts,
            "patient_row_weights":row_weights}

def geometry(p):
    w=p["weights"]; xr=p["XR"]; xs=p["XS"]; rs=p["RS"]
    H=xr.T@(w[:,None]*xr)+.1*np.eye(128)
    e,V=np.linalg.eigh(H)
    Hsqrt=(V*np.sqrt(e))@V.T
    Hinv=(V/np.sqrt(e))@V.T
    _,sv,vt=np.linalg.svd(xs.T,full_matrices=True)
    rank=int(np.sum(sv>max(xs.T.shape)*np.finfo(float).eps*sv[0]))
    N=vt[rank:].T
    B=np.linalg.solve(rs.T@rs/128+.1*np.eye(128),rs.T/128)
    M=Hsqrt@B@N
    U,svm,_=np.linalg.svd(M,full_matrices=False)
    assert N.shape==(128,64) and U.shape==(128,64) and svm[-1]>1e-8
    a=p["rowsR"]@Hinv
    center=np.sum(a*p["patient_row_weights"][:,:,None],axis=(0,1))
    resid=np.where(p["present"][:,:,None],a-center,0.)
    flat=resid[p["present"]]
    wr=p["patient_row_weights"][p["present"]]
    cov=flat.T@(wr[:,None]*flat)
    ec,Vc=np.linalg.eigh(cov); PCA=Vc[:,::-1][:,:64]
    truth=.5*(a[p["present"][:,1],1].mean(axis=0)-a[p["present"][:,0],0].mean(axis=0))
    return {"H":H,"Hsqrt":Hsqrt,"Hinv":Hinv,"N":N,"B":B,"M":M,"U":U,
            "A":U.T@M,"a":a,"center":center,"resid":resid,"PCA":PCA,"truth":truth,
            "pca_eigenvalues":ec[::-1],"M_singular":svm}

def main():
    start=time.perf_counter()
    assert not (OUT/"results.json").exists(),"Preserve existing result"
    c=read_contract()
    # Reuse only pure solver/calibrator; their entrypoints are not called.
    sys.path.insert(0,str(ROOT/"contribution_public128_control_20260928_v1"))
    from construct_labels import solve as old_label_solve
    sys.path.insert(0,str(THESIS/"code_working"))
    from relation_distillation.moments import analytic_gaussian_std
    audit=[]
    allowed={str(Path(x["path"]).resolve()).casefold() for x in c["inputs"].values()}
    def access_hook(event,args):
        if event!="open" or not isinstance(args[0],(str,bytes,Path)): return
        try: f=Path(args[0]).resolve()
        except (TypeError,ValueError): return
        if f.suffix.lower() in (".npz",".npy",".csv",".pt",".pth",".png",".jpg",".dcm"):
            if str(f).casefold() not in allowed and not f.is_relative_to(OUT):
                raise RuntimeError("Data read outside frozen public input list: "+str(f))
            audit.append(str(f))
    sys.addaudithook(access_hook)
    p=load_public(c); g=geometry(p)
    dp=p["present"]
    classD=[p["rowsD"][dp[:,j],j].mean(axis=0) for j in (0,1)]
    HD=p["XD"].T@(p["weights"][:,None]*p["XD"])+.1*np.eye(64)
    source_target=np.linalg.solve(HD,.5*(classD[1]-classD[0]))
    y0,y0detail=old_label_solve({"DINO":p["XD"]},{"DINO":p["XS"]},p["weights"],{"DINO":source_target},["DINO"])
    base_full=g["Hsqrt"]@g["B"]@y0
    base=g["U"].T@base_full
    truth_q=g["U"].T@g["truth"]
    true_raw,true_labels,true_alpha=compile_labels(truth_q,y0,g["N"],g["A"],base)
    inaccessible=g["truth"]-base_full-g["U"]@(g["U"].T@(g["truth"]-base_full))
    inaccessible2=float(inaccessible@inaccessible)
    # All models get the same positive-global-rescaling convention.
    unconstrained=np.linalg.lstsq(g["Hsqrt"]@g["B"],g["truth"],rcond=None)[0]
    unc_alpha=1/max(1,float(np.max(abs(unconstrained))))
    Bs=np.linalg.solve(p["XS"].T@p["XS"]/128+.1*np.eye(64),p["XS"].T/128)
    ref={
      "source_public_label_solve":y0detail,"source_preserving_exact_target_alpha":float(true_alpha[0]),
      "source_preserving_exact_target_full_H_error":float(np.sum((true_raw[0]@g["B"].T@g["Hsqrt"]-g["truth"])**2)),
      "inaccessible_constant":inaccessible2,
      "unconstrained_full_H_error":float(np.sum((g["Hsqrt"]@g["B"]@unconstrained-g["truth"])**2)),
      "unconstrained_alpha":unc_alpha,
      "unconstrained_source_head_change_before_scale":float(np.linalg.norm(Bs@(unconstrained-y0))),
      "source_baseline_full_H_error":float(np.sum((base_full-g["truth"])**2)),
      "comparison_note":"Unconstrained reference breaks source preservation; not a competing algorithm satisfying the same constraints."}
    pp=c["illustrative_privacy_only"]
    sigma=analytic_gaussian_std(pp["numerator_epsilon"],pp["numerator_delta"],1.)*(1+1e-10)
    sigma_source=analytic_gaussian_std(pp["shared_source_epsilon"],pp["shared_source_delta"],1.)*(1+1e-10)
    K=c["simulations"]["draws"]
    rng=np.random.default_rng(c["simulations"]["seed"])
    ambient=rng.standard_normal((K,2,128))
    count_coins=rng.standard_normal((K,2))
    noisy_counts=p["counts"][None,:]+2*sigma_source*count_coins
    den=np.maximum(1.,noisy_counts)
    fixed_counts=np.broadcast_to(p["counts"],(K,2))
    methods={"FULL":np.eye(128),"COMPAT":g["U"],"PCA":g["PCA"]}
    fullcaps=np.array([max(c["cap_floor"],float(np.quantile(np.linalg.norm(g["resid"][dp[:,j],j],axis=1),.95,method="linear"))) for j in (0,1)])
    results={}; saved={
      "y0":y0,"source_target":source_target,"N":g["N"],"U":g["U"],"PCA":g["PCA"],
      "H":g["H"],"Hsqrt":g["Hsqrt"],"Hinv":g["Hinv"],"B":g["B"],"A":g["A"],
      "center":g["center"],"truth":g["truth"],"truth_q":truth_q,"base":base,
      "present":dp,"counts":p["counts"],"weights":p["weights"],"ambient_noise_coins":ambient,
      "count_noise_coins":count_coins,"noisy_counts":noisy_counts,
      "true_raw_labels":true_raw,"true_labels":true_labels,"true_alpha":true_alpha,
      "unconstrained_raw_labels":unconstrained,"unconstrained_labels":unc_alpha*unconstrained}
    null_targets={}
    for method,T in methods.items():
        values=g["resid"]@T
        dims=T.shape[1]
        owncaps=np.array([max(c["cap_floor"],float(np.quantile(np.linalg.norm(values[dp[:,j],j],axis=1),.95,method="linear"))) for j in (0,1)])
        D=T.T@g["U"]
        # Same-cap, NO-CLIPPING null, before and after shared count division.
        sums_nc=np.sum(values,axis=0)/fullcaps[:,None]/np.sqrt(2.)
        noise=sigma*(ambient@T)
        nc=decode(sums_nc[None,:,:]+noise,fullcaps,noisy_counts)@D
        null_targets[method]=nc
        for policy,caps in (("shared_full_q95",fullcaps),("own_q95",owncaps)):
            key=method+"__"+policy
            query,clipped=patient_query(values,dp,caps)
            qs=np.sum(query,axis=0).reshape(2,dims)
            qmean=decode(qs,caps,p["counts"])@D
            bias=qmean-truth_q
            multiplier=.5*sigma*sigma*float(np.sum((caps/p["counts"])**2))
            covariance=multiplier*(D.T@D)
            variance=float(np.trace(covariance))
            bias2=float(bias@bias); risk=bias2+variance
            samples=qs[None,:,:]+noise
            qtruecount=decode(samples,caps,fixed_counts)@D
            qnoisy=decode(samples,caps,noisy_counts)@D
            raws,labels,alphas=compile_labels(qnoisy,y0,g["N"],g["A"],base)
            achieved=raws@g["B"].T@g["Hsqrt"]@g["U"]
            fiterr=float(np.max(abs(achieved-qnoisy)))
            assert fiterr<1e-9,"Compiler not solving its declared target"
            source_error=float(np.max(abs((raws-y0)@Bs.T)))
            assert source_error<1e-9
            source_scaled_error=float(np.max(abs(labels@Bs.T-alphas[:,None]*(Bs@y0))))
            assert source_scaled_error<1e-9 and np.max(abs(labels))<=1+1e-15
            conditional_mean=decode(qs,caps,np.array([1.,1.])) # dimensions sanity below
            # For each fixed shared denominator, exact numerator mean/variance.
            means=decode(np.broadcast_to(qs,(K,2,dims)),caps,noisy_counts)@D
            cbias2=np.sum((means-truth_q)**2,axis=1)
            cvariance=.5*sigma*sigma*np.sum((caps[None,:]/den)**2,axis=1)*float(np.sum(D*D))
            expected_given_counts=cbias2+cvariance
            errors=np.sum((qnoisy-truth_q)**2,axis=1)
            fixed_errors=np.sum((qtruecount-truth_q)**2,axis=1)
            perrow_error=np.linalg.norm((clipped-values)@D,axis=2)
            norm=np.linalg.norm(values,axis=2)
            rowcounts=[int(np.sum(norm[dp[:,j],j]>caps[j])) for j in (0,1)]
            fullobjective=np.sum((raws@g["B"].T@g["Hsqrt"]-g["truth"])**2,axis=1)
            assert np.max(abs(fullobjective-(errors+inaccessible2)))<1e-8
            results[key]={
              "signal_dimension_per_class":dims,"patient_numerator_query_dimension":2*dims,
              "caps":caps.tolist(),"clipped_patient_class_counts":rowcounts,
              "max_patient_query_norm":float(np.linalg.norm(query,axis=1).max()),
              "retained_aggregate_bias_squared":bias2,"retained_noise_variance":variance,
              "analytic_retained_target_MSE_true_counts":risk,
              "analytic_full_H_objective_true_counts":risk+inaccessible2,
              "row_error_patient_equal_mean":float(np.sum(perrow_error*p["patient_row_weights"])),
              "fixed_true_count_simulated_MSE":float(fixed_errors.mean()),
              "noisy_count_simulated_MSE":float(errors.mean()),
              "noisy_count_simulated_MSE_SE":float(errors.std(ddof=1)/np.sqrt(K)),
              "noisy_count_conditionally_integrated_MSE":float(expected_given_counts.mean()),
              "noisy_count_MSE_quantiles":np.quantile(errors,[0,.25,.5,.75,.95,1]).tolist(),
              "alpha_quantiles":np.quantile(alphas,[0,.25,.5,.75,1]).tolist(),
              "raw_label_norm_mean":float(np.linalg.norm(raws,axis=1).mean()),
              "compiler_max_abs_error":fiterr,"source_head_max_abs_error":source_error,
              "source_scaled_head_max_abs_error":source_scaled_error,
              "note":"Raw/de-scaled head objective. Scaling guarantees linear score ranks, not target magnitudes or clinical accuracy."}
            for name,arr in {"query":query,"caps":caps,"qmean":qmean,"bias":bias,"covariance":covariance,"q_fixedcounts":qtruecount,
                             "q_noisycounts":qnoisy,"raw_labels":raws,"labels":labels,"alphas":alphas,
                             "errors":errors,"expected_given_counts":expected_given_counts}.items():
                saved[key+"__"+name]=arr
    null_error=float(np.max(abs(null_targets["FULL"]-null_targets["COMPAT"])))
    assert null_error<1e-11,"No-clipping equality violated"
    primary=results["COMPAT__shared_full_q95"]
    practical=(primary["analytic_retained_target_MSE_true_counts"] < min(results[m+"__shared_full_q95"]["analytic_retained_target_MSE_true_counts"] for m in ("FULL","PCA"))
               and primary["noisy_count_conditionally_integrated_MSE"] < min(results[m+"__shared_full_q95"]["noisy_count_conditionally_integrated_MSE"] for m in ("FULL","PCA")))
    result={
      "item":"2-4B","scope":c["scope"],"status":"IMPLEMENTED_PUBLIC_COMPARISON_AWAITING_INDEPENDENT_VERIFICATION",
      "data":{"P_patients":672,"P_images":813,"P_patient_classes":675,"class_patient_counts":p["counts"].tolist(),"both_class_patients":int(np.sum(dp.all(axis=1)))},
      "geometry":{"nullspace_dim":64,"compatible_dim":64,"M_condition":float(g["M_singular"][0]/g["M_singular"][-1]),
        "PCA_retained_covariance_fraction":float(g["pca_eigenvalues"][:64].sum()/g["pca_eigenvalues"].sum()),
        "U_PCA_overlap_trace":float(np.sum((g["U"].T@g["PCA"])**2)),
        "common_public_center_norm":float(np.linalg.norm(g["center"]))},
      "privacy_simulation":{"sigma_numerator":sigma,"sigma_source":sigma_source,"count_noise_SD":2*sigma_source,
          "sensitivity":1.,"budget_specification":pp,"actual_Q_accesses":0,"actual_private_releases":0},
      "simulation":{"draws":K,"seed":c["simulations"]["seed"],"count_floor_hits_by_class":np.sum(noisy_counts<1,axis=0).tolist(),
        "joint_releases":False,"all_label_variants_are_public_numeric_stress_cases_not_clinical_banks":True},
      "reference":ref,"methods":results,
      "no_clipping_same_cap_coupled_target_max_abs":null_error,
      "practical_primary_support_within_P":bool(practical),
      "decision_limit":"No Q utility / unseen transfer / independent-method novelty established.",
      "data_access_log":sorted(set(audit)),"seconds":time.perf_counter()-start,
      "clinical_readouts":0,"model_forwards":0,"pixel_updates":0,
      "contract_sha256":sha(OUT/"contract.json")}
    for spec in c["inputs"].values(): assert sha(spec["path"])==spec["sha256"]
    for n,h in c["preserved_evidence"].items(): assert sha(ROOT/n)==h
    np.savez_compressed(OUT/"public_arrays.npz",**saved)
    dump(OUT/"results.json",result)
    dump(OUT/"execution_seal.json",{"created_utc":datetime.now(timezone.utc).isoformat(),
         "files":{f:sha(OUT/f) for f in ("contract.json","prototype.py","public_arrays.npz","results.json")},
         "phase":"PUBLIC_RESULT_SEALED_BEFORE_INDEPENDENT_VERIFICATION"})
    print(json.dumps({k:result[k] for k in ("status","geometry","privacy_simulation","simulation","reference","methods","no_clipping_same_cap_coupled_target_max_abs","practical_primary_support_within_P","seconds")},ensure_ascii=False,indent=2))
if __name__=="__main__": main()

