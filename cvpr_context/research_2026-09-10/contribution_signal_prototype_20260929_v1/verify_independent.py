"""Independent raw-input / KKT / privacy-profile verification of sealed 2-4B."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, math, time
import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.special import ndtr, log_ndtr

OUT=Path(__file__).resolve().parent
ROOT=OUT.parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
def main():
    start=time.perf_counter()
    assert not (OUT/"verification.json").exists(),"Do not overwrite verifier output"
    seal=json.loads((OUT/"execution_seal.json").read_text(encoding="utf-8"))
    for f,h in seal["files"].items():assert sha(OUT/f)==h,f
    c=json.loads((OUT/"contract.json").read_text(encoding="utf-8"))
    result=json.loads((OUT/"results.json").read_text(encoding="utf-8"))
    with np.load(OUT/"public_arrays.npz",allow_pickle=False) as z:a={k:z[k] for k in z.files}
    inputs={}
    for name,spec in c["inputs"].items():
        assert sha(spec["path"])==spec["sha256"]
        with np.load(spec["path"],allow_pickle=False) as z:inputs[name]={k:z[k] for k in z.files}
    D,R,S=inputs["DINO_P"],inputs["RN_P"],inputs["selection"]
    assert set(D["roles"])==set(R["roles"])=={"P"}
    for k in ("patient_ids","image_ids","labels"):assert np.array_equal(D[k],R[k])
    xd=np.moveaxis(D["z"],1,0).reshape(813,64).astype(float)
    xr=R["z"].astype(float); xs=S["DINO"].astype(float); rs=S["ResNet18"].astype(float)
    ids=D["patient_ids"]; y=D["labels"]; pts=np.unique(ids)
    # Rebuild records independently using explicit patient/class row lists and fsum.
    rows=[]; owner=[]; classes=[]; rw=[]; w=np.zeros(813)
    counts=np.array([len(set(ids[y==j])) for j in (0,1)])
    for i,pt in enumerate(pts):
        cs=np.unique(y[ids==pt])
        for cl in cs:
            ix=np.flatnonzero((ids==pt)&(y==cl))
            row=np.array([math.fsum(xr[ix,j].tolist())/len(ix) for j in range(128)])
            rows.append(row); owner.append(i); classes.append(int(cl)); rw.append(1/672/len(cs))
            w[ix]=.5/counts[cl]/len(ix)
    rows=np.array(rows); owner=np.array(owner); classes=np.array(classes); rw=np.array(rw)
    assert np.allclose(w,a["weights"],rtol=0,atol=1e-17) and tuple(counts)==(669,6)
    H=xr.T@(w[:,None]*xr)+.1*np.eye(128)
    Hcheck=float(np.max(abs(H-a["H"])))
    eig,ev=np.linalg.eigh(H)
    invsqrt=ev@np.diag(1/np.sqrt(eig))@ev.T
    sq=ev@np.diag(np.sqrt(eig))@ev.T
    ar=rows@invsqrt
    center=rw@ar
    residual=ar-center
    source_null_error=float(np.max(abs(xs.T@a["N"])))
    assert source_null_error<1e-12
    independent_B=rs.T@np.linalg.solve(rs@rs.T+12.8*np.eye(128),np.eye(128))
    assert np.max(abs(independent_B-a["B"]))<1e-12
    G=sq@independent_B
    M=G@a["N"]
    proj=M@np.linalg.pinv(M)
    assert np.max(abs(proj-a["U"]@a["U"].T))<1e-11
    # PCA optimal covariance rank and projector without depending on eigenvector signs.
    C=residual.T@(rw[:,None]*residual)
    e,V=np.linalg.eigh(C); P=V[:,-64:]
    assert np.max(abs(P@P.T-a["PCA"]@a["PCA"].T))<1e-10
    t=.5*(ar[classes==1].mean(axis=0)-ar[classes==0].mean(axis=0))
    tq=a["U"].T@t
    assert np.max(abs(t-a["truth"]))<1e-12
    # Test imported pure routines against independently reconstructed values.
    from prototype import patient_query,decode,clip_rows,compile_labels
    methods={"FULL":np.eye(128),"COMPAT":a["U"],"PCA":a["PCA"]}
    sigma=result["privacy_simulation"]["sigma_numerator"]
    eps=c["illustrative_privacy_only"]["numerator_epsilon"]
    delta=c["illustrative_privacy_only"]["numerator_delta"]
    # Independent profile evaluation, not invoking the calibrator.
    profile=float(ndtr(.5/sigma-eps*sigma)-np.exp(eps+log_ndtr(-.5/sigma-eps*sigma)))
    assert profile<=delta and abs(profile-delta)<delta*1e-7
    count_sd=result["privacy_simulation"]["count_noise_SD"]
    assert np.max(abs(a["noisy_counts"]-(counts+count_sd*a["count_noise_coins"])))<1e-14
    countden=np.maximum(1.,a["noisy_counts"])
    K=len(countden); KKT=np.block([[G.T@G,xs],[xs.T,np.zeros((64,64))]])
    lu=lu_factor(KKT)
    Bs={}
    base_scores={}
    for lam in (.01,.1,1.):
        Bs[lam]=xs.T@np.linalg.solve(xs@xs.T+128*lam*np.eye(128),np.eye(128))
        base_scores[lam]=xd@Bs[lam]@a["y0"]
    checks={}; max_addremove=0.; max_clip_inequality=0.; max_scoreerr=0.; max_kkt=0.
    max_rank_violation=0.
    truth_full_labels=a["true_raw_labels"][0]
    assert np.max(abs(xs.T@(truth_full_labels-a["y0"])))<1e-11
    # Independent synthetic saturation and malformed-input tests.
    present=np.array([[True,True],[True,False],[False,False]],dtype=bool)
    toy=np.zeros((3,2,7));toy[0,0,0]=10.;toy[0,1,3]=-100.;toy[1,0,2]=1e6
    qt,_=patient_query(toy,present,np.array([2.,3.]))
    assert np.linalg.norm(qt[0])<=1 and np.linalg.norm(qt[0])>1-1e-12
    assert np.linalg.norm(qt[1])<=1/np.sqrt(2)+1e-14 and np.linalg.norm(qt[2])==0
    failures=0
    invalid=[
      lambda:patient_query(toy,present,np.array([0.,3.])),
      lambda:patient_query(toy,present.astype(int),np.array([2.,3.])),
      lambda:patient_query(np.full_like(toy,np.nan),present,np.array([2.,3.])),
      lambda:patient_query(toy+1,present,np.array([2.,3.])),
      lambda:clip_rows(np.ones((2,3)),np.inf),
      lambda:clip_rows(np.ones(3),1),
      lambda:decode(np.ones((2,3)),np.array([1.,1.]),np.array([np.nan,1.])),
      lambda:decode(np.ones((3,3)),np.array([1.,1.]),np.array([1.,2.,3.])),
      lambda:decode(np.ones((2,3)),np.array([-1.,1.]),np.array([1.,2.]))]
    for f in invalid:
        try:f()
        except ValueError:failures+=1
    assert failures==len(invalid)
    neg=decode(np.ones((2,3)),np.ones(2),np.array([-1e5,0.]))
    assert np.isfinite(neg).all() and np.max(abs(neg))==0.
    for method,T in methods.items():
      values=residual@T
      d=T.shape[1]; transport=T.T@a["U"]
      for policy in ("shared_full_q95","own_q95"):
        key=method+"__"+policy
        caps=a[key+"__caps"]
        norm=np.linalg.norm(values,axis=1)
        z=values*np.minimum(1,caps[classes]/np.maximum(norm,1e-300))[:,None]
        built=np.zeros((672,2,d))
        built[owner,classes]=z/caps[classes,None]/np.sqrt(2)
        bn=np.linalg.norm(built.reshape(672,-1),axis=1)
        built*=np.minimum(1,(1-16*np.finfo(float).eps)/np.maximum(bn,1e-300))[:,None,None]
        independent=built.reshape(672,2*d)
        qerror=float(np.max(abs(independent-a[key+"__query"])))
        assert qerror<1e-12
        sums=np.array([math.fsum(independent[:,j].tolist()) for j in range(2*d)]).reshape(2,d)
        # Reconstruct each neighbor's sum independently; both-class patients included.
        total=sums.reshape(-1)
        local=0.
        for i in range(672):
            minus=np.sum(np.delete(independent,i,axis=0),axis=0)
            local=max(local,float(np.linalg.norm(total-minus)))
        max_addremove=max(max_addremove,local)
        assert local<=1+1e-11
        exactmeans=np.sqrt(2)*sums*caps[:,None]/counts[:,None]
        mean=.5*(exactmeans[1]-exactmeans[0])@transport
        bias=mean-tq
        cov=np.zeros((64,64))
        for j in (0,1):
            L=((1 if j else -1)*caps[j]/(np.sqrt(2)*counts[j]))*transport.T
            cov+=sigma*sigma*L@L.T
        assert np.max(abs(cov-a[key+"__covariance"]))<1e-12
        risk=float(bias@bias+np.trace(cov))
        assert abs(risk-result["methods"][key]["analytic_retained_target_MSE_true_counts"])<1e-10
        numnoise=sigma*(a["ambient_noise_coins"]@T)
        means=(np.sqrt(2)*(sums[None,:,:]+numnoise)*caps[None,:,None]/countden[:,:,None])
        qdraw=(.5*(means[:,1]-means[:,0]))@transport
        assert np.max(abs(qdraw-a[key+"__q_noisycounts"]))<1e-12
        # KKT equality-constrained Label Solve, no use of candidate's nullspace solve.
        target=a["U"]@qdraw.T+(np.eye(128)-proj)@t[:,None]
        rhs=np.vstack([G.T@target,np.broadcast_to((xs.T@a["y0"])[:,None],(64,K))])
        sol=lu_solve(lu,rhs)[:128].T
        kkt_error=float(np.max(abs(sol-a[key+"__raw_labels"])))
        max_kkt=max(max_kkt,kkt_error)
        assert kkt_error<1e-7
        alpha=1/np.maximum(1,np.max(abs(sol),axis=1))
        assert np.max(abs(alpha[:,None]*sol-a[key+"__labels"]))<1e-9
        errors=np.sum((qdraw-tq)**2,axis=1)
        assert abs(errors.mean()-result["methods"][key]["noisy_count_simulated_MSE"])<1e-10
        max_scaled_score=0.; min_step=0.
        for lam in (.01,.1,1.):
            baseline=base_scores[lam]
            actual=a[key+"__labels"]@Bs[lam].T@xd.T
            expected=a[key+"__alphas"][:,None]*baseline[None,:]
            scoreerr=float(np.max(abs(actual-expected)))
            max_scaled_score=max(max_scaled_score,scoreerr)
            # Positive global scale preserves order except numerical ties; reject substantive inversions.
            order=np.argsort(baseline,kind="stable")
            normalized=actual/a[key+"__alphas"][:,None]
            step=float(np.min(np.diff(normalized[:,order],axis=1)))
            min_step=min(min_step,step)
        max_scoreerr=max(max_scoreerr,max_scaled_score)
        max_rank_violation=max(max_rank_violation,-min_step)
        assert max_scaled_score<1e-10 and min_step>-1e-9
        # Full H loss decomposition: the inaccessible component is a constant.
        fullerr=np.sum((sol@G.T-t)**2,axis=1)
        const=result["reference"]["inaccessible_constant"]
        assert np.max(abs(fullerr-errors-const))<1e-7
        checks[key]={"query_independent_max_abs":qerror,"add_remove_l2_max":local,
                    "KKT_label_max_abs":kkt_error,"public_source_score_scale_max_abs":max_scaled_score,
                    "max_numerical_rank_inversion":-min_step,"analytic_MSE":risk,
                    "simulated_MSE":float(errors.mean())}
    # Same-cap full / compatible individual clipping inequality.
    caps=a["FULL__shared_full_q95__caps"]
    fullclip=residual*np.minimum(1,caps[classes]/np.maximum(np.linalg.norm(residual,axis=1),1e-300))[:,None]
    small=residual@a["U"]
    smallclip=small*np.minimum(1,caps[classes]/np.maximum(np.linalg.norm(small,axis=1),1e-300))[:,None]
    diff=np.linalg.norm(smallclip-small,axis=1)-np.linalg.norm(fullclip@a["U"]-small,axis=1)
    max_clip_inequality=float(diff.max())
    assert max_clip_inequality<1e-12
    # No-clipping exact equality with the same Gaussian realization, including shared noisy counts.
    fsum=np.stack([residual[classes==j].sum(axis=0) for j in (0,1)])
    noise_full=np.sqrt(2)*sigma*caps[None,:,None]*a["ambient_noise_coins"]
    ncf=.5*((fsum[1]+noise_full[:,1])/countden[:,1,None]-(fsum[0]+noise_full[:,0])/countden[:,0,None])@a["U"]
    smallsum=fsum@a["U"]; smallnoise=noise_full@a["U"]
    ncs=.5*((smallsum[1]+smallnoise[:,1])/countden[:,1,None]-(smallsum[0]+smallnoise[:,0])/countden[:,0,None])
    nce=float(np.max(abs(ncf-ncs)))
    assert nce<1e-11
    covariance_identity=float(np.max(abs(a["U"].T@a["U"]-np.eye(64))))
    assert covariance_identity<1e-12
    # The generic workload strategy with U and candidate are the identical query/decoder.
    generic=smallclip/caps[classes,None]/np.sqrt(2.)
    gi=np.zeros((672,2,64));gi[owner,classes]=generic
    norms=np.linalg.norm(gi.reshape(672,-1),axis=1)
    gi*=np.minimum(1,(1-16*np.finfo(float).eps)/np.maximum(norms,1e-300))[:,None,None]
    generic_error=float(np.max(abs(gi.reshape(672,-1)-a["COMPAT__shared_full_q95__query"])))
    assert generic_error<1e-12
    for f,h in seal["files"].items():assert sha(OUT/f)==h,f
    for name,h in c["preserved_evidence"].items():assert sha(ROOT/name)==h,name
    verification={
      "item":"2-4B","status":"PASS","created_utc":datetime.now(timezone.utc).isoformat(),
      "independent_formulations":["patient/class fsum aggregation","primal/dual ridge","KKT equality-constrained Label Solve","Gaussian privacy profile","covariance via per-class linear maps"],
      "H_input_max_abs":Hcheck,"source_nullspace_max_abs":source_null_error,
      "analytic_Gaussian_delta_at_sigma":profile,"requested_delta":delta,
      "all_patient_remove_cases":672*6,"max_add_remove_l2":max_addremove,
      "saturated_multi_class_neighbor_pass":True,"invalid_inputs_rejected":failures,
      "negative_count_floor_pass":True,"same_cap_row_clipping_inequality_max_violation":max_clip_inequality,
      "no_clip_coupled_target_max_abs":nce,"no_clip_noise_covariance_identity_max_abs":covariance_identity,
      "generic_workload_U_query_max_abs":generic_error,
      "KKT_all_labels_max_abs":max_kkt,"public_source_score_scale_max_abs":max_scoreerr,
      "public_source_substantive_rank_inversions":0,"max_numeric_rank_inversion":max_rank_violation,
      "checked_ridge_values":[.01,.1,1.],"checked_label_configurations":6*512,
      "per_configuration":checks,
      "input_and_historical_evidence_unchanged":True,
      "Q_V_DP_release_Expert_Reserved_accesses":0,
      "limitations":"Public P only; no Q efficacy, no private output, no general novelty. Public rank checks are algebra validation, not clinical AUROC/AP evaluation.",
      "seconds":time.perf_counter()-start}
    write(OUT/"verification.json",verification)
    print(json.dumps(verification,ensure_ascii=False,indent=2))
if __name__=="__main__":main()

