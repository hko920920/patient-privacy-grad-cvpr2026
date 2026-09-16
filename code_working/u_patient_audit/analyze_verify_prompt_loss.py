"""Predeclared t140 prompt-loss diagnostic: raw verification before scalar analysis.
No fitting, no prompt selection, no patient-causal interpretation.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from .common import ROOT,RUN
from .models import scheduler

OUT=RUN/"baseline_screen_20260914/prompt_loss_t140_v1"
EU=ROOT.parent/"CVPR 주제 탐색/research_2026-09-10/spec_sources/cdi_eu_analysis_contract_20260915.json"
EU_SHA="bc2497d17b3110ec8b994eaa541d4eda00891ff5d30215a9aff26b7aa5f1db43"
PROMPTS=("null","generic","weak_label")
SCORES=("negative_target_loss","base_minus_target_loss")
MODELS=("model_1","model_2")
CONTRASTS=(("weak_label","null"),("generic","null"),("weak_label","generic"))
def read(p):return json.loads(Path(p).read_text(encoding="utf-8-sig"))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def need(v,msg):
    if not v:raise AssertionError(msg)
def new_json(p,v):
    with Path(p).open("x",encoding="utf-8") as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def thash(v):return hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def close(a,b,label,rtol=2e-12,atol=2e-12):
    a,b=np.asarray(a,dtype=float),np.asarray(b,dtype=float)
    need(a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all() and np.allclose(a,b,rtol=rtol,atol=atol),label)
def cpu_tensor(v,label):
    need(isinstance(v,torch.Tensor) and v.dtype==torch.float32 and v.device.type=="cpu"
         and tuple(v.shape)==(1,4,32,32) and torch.isfinite(v).all(),label)
def policy():
    return dict(stage=2,scope="t140 conditioning diagnostic; no optimization",
        prompts=list(PROMPTS),models=list(MODELS),scenarios=["E","U"],image_pool="mean of exactly2 losses",
        scores=list(SCORES),direction="high membership; -target loss or base loss minus target loss",
        primary_role="selection",fit_role="descriptive only; no selection or fitting",
        target_AUC_count=24,base_A_assignment_AUC_count=6,prompt_contrasts=list(CONTRASTS),
        prompt_difference_CIs=24,bootstrap=dict(seed=260914,resamples=2000,stratified="20A20B patients"),
        test_calibration_used=False,new_fitting=False,
        limitations=["one fixed noise per image and one timestep","cached FP16 posterior mode promoted to FP32",
            "MoFit uses fresh posterior samples, so this is not its complete score",
            "U weak-label is a hypothetical training template, not an actually used training prompt",
            "base AUC labels are assignment A/B, not base membership",
            "target-minus-base includes domain-wide adaptation; not individual training influence",
            "shared training initialization, reused development patients and unadjusted conditional intervals",
            "lower loss under a prompt is not by itself evidence of membership",
            "null dropout zero does not imply unconditional output is unaffected by training"],
        overall_stage2_gate=None)
def verify_raw(directory):
    start=time.perf_counter();torch.set_num_threads(2)
    protocol=read(directory/"protocol.json");execution=read(directory/"execution.json")
    c=protocol["contract"]
    need(execution["complete"] is True and execution["records"]==1440,"complete1440records")
    need(execution["results_sha256"]==sha(directory/"results.json"),"results binding")
    need(execution["protocol_sha256"]==sha(directory/"protocol.json"),"protocol binding")
    need(sha(protocol["contract_path"])==protocol["contract_sha256"],"frozen extraction contract")
    need(read(protocol["contract_path"])==c,"embedded contract")
    frozen=c["frozen_sha256"]
    def unchanged():
        for p,h in frozen.items():need(sha(p)==h,"frozen input changed "+p)
    unchanged()
    need(sha(EU)==EU_SHA,"same frozen patient cohorts")
    reference=read(EU);patients=reference["patient_rows"]
    image_meta={image:dict(p,scenario=s) for p in patients for s in ("E","U") for image in p["images"][s]}
    need(len(image_meta)==480,"480 E/U images")
    cache=torch.load(RUN/"cache/cache.pt",map_location="cpu",weights_only=True)
    cache_sha,lock_sha=sha(RUN/"cache/cache.pt"),sha(RUN/"cohort/lock.json")
    alphas=scheduler().alphas_cumprod
    ledgers={}
    for m in MODELS:
        ckpt=RUN/("training_coverage_v2/"+m+"/step_1000.pt")
        state=torch.load(ckpt,map_location="cpu",weights_only=True)
        need(state["step"]==1000 and sum(state["exposures"].values())==4000,"actual step/exposure total")
        ledgers[m]=dict(exposures=state["exposures"],sha=sha(ckpt))
    rows=read(directory/"results.json");index={(r["model"],r["image_id"]):r for r in rows}
    expected={(m,i) for m in (*MODELS,"base") for i in image_meta}
    need(len(rows)==len(index)==1440 and set(index)==expected,"complete exact model/image set")
    checked=0;raw_hashes={};byimage={}
    for r in rows:
        m,i=r["model"],r["image_id"];p=image_meta[i];s=p["scenario"]
        label=int(p["group"]==("A" if m=="model_1" else "B")) if m in MODELS else 0
        ex=ledgers[m]["exposures"] if m in MODELS else {}
        image_ex=int(ex.get(i,0))
        patient_ex=sum(int(ex.get(im,0)) for images in p["images"].values() for im in images)
        need(r["member"]==label==int(patient_ex>0) and r["patient_training_exposures"]==patient_ex,"patient membership/exposure")
        need(r["image_member"]==int(image_ex>0) and r["actual_training_exposures"]==image_ex,"exact image exposure")
        need(int(image_ex>0)==(label if s=="E" else 0),"E/U label relation")
        need(r["patient_id"]==p["patient_id"] and r["eval_role"]==p["role"] and r["assignment_group"]==p["group"]
             and r["scenario"]==s and r["record_role"]==("train_candidate" if s=="E" else "U_observed"),"cohort metadata")
        need(r["pretrained_membership"]=="unknown","pretraining unknown")
        if m in MODELS:need(r["checkpoint_sha256"]==ledgers[m]["sha"],"checkpoint hash")
        need(r["cache_sha256"]==cache_sha and r["cohort_lock_sha256"]==lock_sha,"cache/lock")
        need(r["contract_sha256"]==protocol["contract_sha256"] and r["weights_unchanged"] is True
             and r["unet_training"] is False,"row freeze guards")
        if m=="base":need(r["checkpoint_sha256"]==c["base_weight_sha256"],"actual base artifact")
        rawpath=(directory/r["raw_path"]).resolve()
        need(rawpath.is_relative_to(directory.resolve()),"contained raw path")
        need(sha(rawpath)==r["raw_sha256"],"raw content hash")
        need(read(rawpath.with_suffix(".json"))==r,"per-image JSON equality")
        raw=torch.load(rawpath,map_location="cpu",weights_only=True)
        need(raw["model"]==m and raw["image_id"]==i and raw["timestep"]==140,"raw identity")
        need(torch.equal(raw["alphas"],alphas),"frozen CPU scheduler")
        for n in ("latent","epsilon","noised_latent"):cpu_tensor(raw[n],n)
        latent=cache["latents"][i].float()
        need(torch.equal(raw["latent"],latent),"unchanged cached latent mode")
        seed=int.from_bytes(hashlib.sha256(f"promptloss-v1|260915|{i}|140".encode()).digest()[:8],"big")%(2**63-1)
        epsilon=torch.randn((1,4,32,32),generator=torch.Generator(device="cpu").manual_seed(seed),dtype=torch.float32)
        need(raw["noise_seed"]==seed and torch.equal(raw["epsilon"],epsilon) and raw["noise_sha256"]==thash(epsilon),"independently regenerated noise")
        need(r["noise_seed"]==seed and r["noise_sha256"]==raw["noise_sha256"]
             and r["prompt_hidden_sha256"]==raw["prompt_hidden_sha256"],"row/raw noise and hidden metadata")
        noised=alphas[140].sqrt()*latent+(1-alphas[140]).sqrt()*epsilon
        close(raw["noised_latent"].numpy(),noised.numpy(),"forward noising",rtol=1e-6,atol=1e-6)
        texts={"null":"","generic":cache["audit_prompt"],"weak_label":cache["train_prompts"][i]}
        need(r["prompt_texts"]==texts,"unchanged prompt strings")
        need(set(raw["predictions"])==set(r["losses"])==set(PROMPTS),"three fixed prompt conditions")
        losses={}
        for kind in PROMPTS:
            pred=raw["predictions"][kind];cpu_tensor(pred,"saved prediction")
            need(raw["prompt_hidden_sha256"][kind]==thash(cache["hidden"][texts[kind]].float()),"actual cached prompt hidden")
            residual=pred.double().numpy()-epsilon.double().numpy()
            mse=float(np.mean(residual*residual));close(r["losses"][kind],mse,"independent residual MSE",rtol=2e-6,atol=1e-7)
            losses[kind]=mse;checked+=1
        need(r["forward"]==3 and r["backward"]==0,"perimage counts")
        fingerprint=(thash(raw["latent"]),thash(raw["epsilon"]),thash(raw["noised_latent"]),raw["prompt_hidden_sha256"])
        if i in byimage:need(fingerprint==byimage[i],"identical image/noise/prompt inputs across models")
        else:byimage[i]=fingerprint
        raw_hashes[r["raw_path"]]=r["raw_sha256"]
        index[(m,i)]=dict(r,verified_fp64_losses=losses)
    cells=read(directory/"cells.json")
    cellix={(r["model"],r["image_id"],r["prompt_kind"]):r for r in cells}
    need(len(cells)==len(cellix)==4320 and set(cellix)=={(m,i,k) for m,i in expected for k in PROMPTS},"4320 prompt cells")
    for (m,i,k),r in cellix.items():
        close(r["loss_mse"],index[(m,i)]["losses"][k],"flat cells loss")
        close(r["negative_loss"],-r["loss_mse"],"flat cell polarity")
        source=index[(m,i)]
        need(r["forward"]==1 and r["backward"]==0 and r["prompt_text"]==source["prompt_texts"][k],"flat cell count/prompt")
        need(all(r[n]==source[n] for n in ("patient_id","eval_role","scenario","record_role","assignment_group","member","image_member","actual_training_exposures","patient_training_exposures","raw_path","raw_sha256")),"flat cell metadata")
    need(execution["forward"]==4320 and execution["backward"]==0,"total forward counts")
    need(execution["cells_sha256"]==sha(directory/"cells.json"),"flat-cell binding")
    reports=execution["model_reports"]
    need(len(reports)==3 and {r["model"] for r in reports}==set((*MODELS,"base")),"all model reports")
    for report in reports:
        need(report["initial_state"]["sha256"]==report["final_state"]["sha256"]
             and report["state_values_exactly_unchanged"] is True and report["parameter_versions_and_no_gradients"] is True
             and report["all_parameters_frozen"] is True and report["unet_training"] is False
             and report["forward"]==1440 and report["backward"]==0,"model guard evidence")
        if report["model"]=="base":need(report["initial_state"]["artifact_tensors_exactly_checked"]==686 and report["no_lora_modules"] is True,"actual base state verification")
        else:need(report["adapter_matches_checkpoint"] is True,"loaded target adapter")
    unchanged()
    result=dict(status="PASS_PROMPT_LOSS_RAW_ARITHMETIC_AND_COHORT",records=1440,losses_verified=checked,
        independent_noise_regenerations=1440,distinct_noises=480,new_GPU_calls=0,
        independent_UNet_execution=False,model_state_guards="producer guard evidence; not an independent forward rerun",
        results_sha256=sha(directory/"results.json"),protocol_sha256=sha(directory/"protocol.json"),
        execution_sha256=sha(directory/"execution.json"),verification_code_sha256=sha(__file__),
        raw_sha256=raw_hashes,seconds_cpu=time.perf_counter()-start)
    new_json(directory/"verification.json",result)
    print(json.dumps({k:v for k,v in result.items() if k!="raw_sha256"}),flush=True)
    return reference,index
def analyze(directory,reference,index):
    start=time.perf_counter();out=directory/"analysis_v1";out.mkdir()
    new_json(out/"protocol.json",dict(created_utc=datetime.now(timezone.utc).isoformat(),policy=policy(),
        code_sha256=sha(__file__),raw_verification_sha256=sha(directory/"verification.json"),
        results_sha256=sha(directory/"results.json"),cohort_contract_sha256=EU_SHA))
    roles={r:[p for p in reference["patient_rows"] if p["role"]==r] for r in ("fit","selection")}
    selected=roles["selection"];a=np.flatnonzero([p["group"]=="A" for p in selected]);b=np.flatnonzero([p["group"]=="B" for p in selected])
    rng=np.random.default_rng(260914);ix=np.asarray([np.r_[rng.choice(a,20,replace=True),rng.choice(b,20,replace=True)] for _ in range(2000)])
    cnt=np.asarray([np.bincount(v,minlength=40) for v in ix],dtype=float)
    scores={};rows=[];metrics=[];points={};boots={}
    for s in ("E","U"):
        for m in (*MODELS,"base"):
            for kind in PROMPTS:
                for scoretype in (SCORES if m!="base" else ("negative_base_loss_A_assignment",)):
                    for role,ps in roles.items():
                        values=[];labels=[];high_precision=[]
                        for p in ps:
                            loss=np.mean([index[(m,i)]["losses"][kind] for i in p["images"][s]])
                            base=np.mean([index[("base",i)]["losses"][kind] for i in p["images"][s]])
                            score=float(base-loss if scoretype=="base_minus_target_loss" else -loss)
                            loss64=np.mean([index[(m,i)]["verified_fp64_losses"][kind] for i in p["images"][s]])
                            base64=np.mean([index[("base",i)]["verified_fp64_losses"][kind] for i in p["images"][s]])
                            high_precision.append(float(base64-loss64 if scoretype=="base_minus_target_loss" else -loss64))
                            label=int(p["group"]==("B" if m=="model_2" else "A"))
                            values.append(score);labels.append(label)
                            rows.append(dict(scenario=s,model=m,prompt=kind,score_type=scoretype,role=role,
                                patient_id=p["patient_id"],group=p["group"],label=label,score=score,
                                label_semantics="A assignment, not base membership" if m=="base" else "incremental patient membership"))
                        scores[(s,m,kind,scoretype,role)]=(np.asarray(values),np.asarray(labels))
                        if role=="selection":selection64=np.asarray(high_precision)
                    key=(s,m,kind,scoretype);v,y=scores[(*key,"selection")];pos,neg=np.flatnonzero(y),np.flatnonzero(1-y)
                    credit=(v[pos,None]>v[None,neg]).astype(float)+.5*(v[pos,None]==v[None,neg])
                    auc=float(credit.mean());close(auc,roc_auc_score(y,v),"independent sklearn rank AUC")
                    credit64=(selection64[pos,None]>selection64[None,neg]).astype(float)+.5*(selection64[pos,None]==selection64[None,neg])
                    bs=np.einsum("bi,ij,bj->b",cnt[:,pos],credit,cnt[:,neg])/400
                    vv,yy=v[ix],y[ix];positive=vv[yy==1].reshape(2000,20);negative=vv[yy==0].reshape(2000,20)
                    direct=((positive[:,:,None]>negative[:,None,:]).sum((1,2))+.5*(positive[:,:,None]==negative[:,None,:]).sum((1,2)))/400
                    close(bs,direct,"independent direct-index bootstrap")
                    points[key],boots[key]=auc,bs
                    fv,fy=scores[(*key,"fit")]
                    metrics.append(dict(scenario=s,model=m,prompt=kind,score_type=scoretype,
                        selection_AUC=auc,CI95=np.quantile(bs,[.025,.975]).tolist(),
                        fit_AUC_descriptive_only=float(roc_auc_score(fy,fv)),
                        fp64_saved_prediction_residual_AUC=float(credit64.mean()),
                        GPU_MSE_vs_FP64_residual_AUC_delta=float(auc-credit64.mean()),
                        GPU_MSE_vs_FP64_residual_pair_credit_mismatches=int(np.count_nonzero(credit!=credit64)),
                        selection_statistical_units=40,base_assignment_only=m=="base"))
    contrasts=[]
    for s in ("E","U"):
        for m in MODELS:
            for typ in SCORES:
                for pa,pb in CONTRASTS:
                    ka,kb=(s,m,pa,typ),(s,m,pb,typ)
                    contrasts.append(dict(scenario=s,model=m,score_type=typ,a=pa,b=pb,
                        delta_AUC=points[ka]-points[kb],paired_CI95=np.quantile(boots[ka]-boots[kb],[.025,.975]).tolist()))
    need(len(metrics)==30 and len(contrasts)==24 and len(rows)==3600,"fixed report scope")
    result=dict(status="PASS_SCALAR_ANALYSIS_WITH_TWO_ARITHMETIC_PATHS",policy=policy(),metrics=metrics,
        prompt_contrasts=contrasts,new_fitting=False,new_GPU_calls=0,overall_stage2_gate=None,
        seconds_cpu=time.perf_counter()-start)
    new_json(out/"analysis.json",result);new_json(out/"patient_scores.json",rows)
    new_json(out/"bootstrap.json",dict(patient_order=[p["patient_id"] for p in selected],indices=ix.tolist()))
    new_json(out/"provenance.json",dict(code_sha256=sha(__file__),output_sha256={n:sha(out/n) for n in ("protocol.json","analysis.json","patient_scores.json","bootstrap.json")},
        raw_verification_sha256=sha(directory/"verification.json"),results_sha256=sha(directory/"results.json")))
    print(json.dumps(dict(status=result["status"],metrics=30,paired_contrasts=24,seconds=result["seconds_cpu"])),flush=True)
def self_test():
    y=np.array([0,0,1,1]);s=np.array([1,2,2,3]);need(roc_auc_score(y,s)==.875,"tie handling")
    need(policy()["target_AUC_count"]==24 and policy()["overall_stage2_gate"] is None,"fixed policy")
    import builtins,symtable
    missing=[]
    def walk(scope):
        if scope.get_type()=="function":
            for sym in scope.get_symbols():
                if sym.is_referenced() and sym.is_global() and sym.get_name() not in globals() and not hasattr(builtins,sym.get_name()):
                    missing.append((scope.get_name(),sym.get_name()))
        for child in scope.get_children():walk(child)
    walk(symtable.symtable(Path(__file__).read_text(encoding="utf-8"),__file__,"exec"))
    need(not missing,"global dependencies "+repr(missing));print("PASS_PROMPT_DIAGNOSTIC_STATIC_POLICY_RANK_GLOBALS; no target or results used")
if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--self-test",action="store_true");parser.add_argument("--output",type=Path,default=OUT)
    parser.add_argument("--expected-code-sha256");args=parser.parse_args()
    if args.self_test:self_test()
    else:
        need(args.expected_code_sha256==sha(__file__),"externally frozen analysis/verifier")
        ref,index=verify_raw(args.output);analyze(args.output,ref,index)
