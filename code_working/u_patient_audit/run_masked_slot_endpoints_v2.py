"""Fixed endpoint measurements after a separately verified masked-slot replay.

No fitting, optimization, target training, AUC or seed selection occurs here.
All existing treatment predictions are reused except p14393's 28 new t140 cells.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import gc
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import time
import traceback
import torch
import torch.nn.functional as F
from .common import ROOT, RUN, PROMPT, digest, read_csv
from .models import setup, load_unet, load_cache, scheduler, adapter_state
from .run_prompt_loss_t140 import new_json, now, tensor_sha, guards
from .run_repeat_noise_t140 import epsilon_for as t140_noise
from .run_cdi_base_selection import state_fingerprint

BASE = RUN / "baseline_screen_20260915"
TRAINING = BASE / "masked_patient_training_v1"
OUTPUT = BASE / "masked_slot_endpoints_v2"
CONTRACT = BASE / "masked_slot_endpoints_contract_v2.json"
VERIFIER_BINDING = dict(analysis_policy_frozen_before_gpu=True,
    independent_verifier_code_frozen_before_gpu=False,
    independent_verifier_code_binding="Completed verifier code is bound later in its own independent protocol; not a pre-GPU code freeze")
PROMPT_DIR = RUN / "baseline_screen_20260914/prompt_loss_t140_v1"
REPEAT_DIR = BASE / "repeat_noise_t140_v1"
CROSS_DIR = BASE / "mofit_cross_response_v1"
CDI_DIRS = {s: RUN / "baseline_screen_20260914" / ("cdi_" + s.lower() + "_cohort_v1") for s in ("E", "U")}
PID = "14393"
P_IMAGES = ["00014393_002.png", "00014393_006.png", "00014393_001.png", "00014393_004.png"]
BRANCHES = ("treatment", "control")
FAMILIES = ("generic_t140", "cdi_dl_t100", "fixed_mofit")
OLD_CP = RUN / "training_coverage_v2/model_1/step_1000.pt"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def require(x, message):
    if not x:
        raise AssertionError(message)


def exact_nested(a, b, name="state"):
    if isinstance(a, torch.Tensor):
        require(isinstance(b, torch.Tensor) and a.shape == b.shape and a.dtype == b.dtype
                and torch.equal(a, b), "exact checkpoint tensor " + name)
    elif isinstance(a, dict):
        require(isinstance(b, dict) and a.keys() == b.keys(), "checkpoint dictionary " + name)
        for k in a:
            exact_nested(a[k], b[k], name + "." + str(k))
    elif isinstance(a, (tuple, list)):
        require(type(a) is type(b) and len(a) == len(b), "checkpoint sequence " + name)
        for i, (x, y) in enumerate(zip(a, b)):
            exact_nested(x, y, name + "." + str(i))
    else:
        require(type(a) is type(b) and a == b, "checkpoint scalar " + name)


def scope_rows():
    rows = [r for r in read_csv(RUN / "cohort/evaluation_images.csv")
            if r["eval_role"] == "selection" or (r["eval_role"] == "fit" and r["patient_id"] == PID)]
    order = [PID] + sorted({r["patient_id"] for r in rows if r["eval_role"] == "selection"}, key=int)
    grouped = defaultdict(list)
    for r in rows:
        grouped[r["patient_id"]].append(r)
    selected = []
    for p in order:
        require(len(grouped[p]) == 4, "four fixed E/U images")
        for s, role in (("E", "train_candidate"), ("U", "U_observed")):
            rs = sorted((r for r in grouped[p] if r["record_role"] == role), key=lambda r:r["image_id"])
            require(len(rs) == 2, "two fixed images per scenario")
            selected.extend(dict(r, scenario=s) for r in rs)
    require(len(order) == 41 and len(selected) == 164
            and [r["image_id"] for r in selected[:4]] == P_IMAGES, "p1 plus selection40, no other scope")
    return selected, order


def training_evidence(verification_path):
    v = read(verification_path)
    require(v["status"]=="PASS_MASKED_TRAINING_SAVED_STATE_INPUTS_AND_MASK_ARITHMETIC",
            "completed independent training verification required")
    e, p = read(TRAINING / "execution.json"), read(TRAINING / "protocol.json")
    require(e["status"] == "PASS_REPLAY_AND_MASKED_CONTRIBUTION_EXECUTION_PENDING_INDEPENDENT_VERIFICATION",
            "completed original replay and masked contribution execution")
    require(e["complete"] is True, "training complete")
    require(v["execution_sha256"]==digest(TRAINING/"execution.json")
            and v["protocol_sha256"]==digest(TRAINING/"protocol.json")
            and v["contract_sha256"]==p["contract_sha256"], "actual independently verified training outputs")
    conformance=read(TRAINING/"conformance.json")
    require(conformance["status"]=="PASS_EXACT_TREATMENT_REPLAY_AND_SHARED_PREFIX"
            and len(conformance["gates"])==3 and all(q["exact"] is True for q in conformance["gates"])
            and digest(TRAINING/"conformance.json")==e["conformance_sha256"], "original exact replay gates")
    states = {}
    paths = [Path(verification_path), TRAINING / "execution.json", TRAINING / "protocol.json",
             TRAINING / "conformance.json"]
    for step in (250, 1000):
        original_path = RUN / "training_coverage_v2/model_1" / f"step_{step:04d}.pt"
        replay_path = TRAINING / "treatment" / f"step_{step:04d}.pt"
        original = torch.load(original_path, map_location="cpu", weights_only=True)
        replay = torch.load(replay_path, map_location="cpu", weights_only=True)
        exact_nested(original, replay, f"replay{step}")
        paths.extend([original_path, replay_path])
        if step == 1000:
            states["treatment"] = original
    prefixes = [TRAINING / b / "step_0043.pt" for b in BRANCHES]
    a, b = [torch.load(path, map_location="cpu", weights_only=True) for path in prefixes]
    exact_nested(a,b,"common_prefix43")
    control_path = TRAINING / "control/step_1000.pt"
    control = torch.load(control_path, map_location="cpu", weights_only=True)
    require(control["step"] == control["attempt"] == 1000 and len(control["losses"]) == 1000,
            "control full committed schedule")
    expected = dict(states["treatment"]["exposures"])
    require([expected.pop(i) for i in P_IMAGES[:2]] == [4,4], "original eight E contributions")
    require(control["exposures"] == expected and sum(expected.values()) == 3992, "only p E effective exposures removed")
    states["control"] = control
    for branch in BRANCHES:
        for step in (43,250,1000):
            q=TRAINING/branch/f"step_{step:04d}.pt"
            require(digest(q)==v["checkpoint_sha256"][branch][str(step)],"verified branch checkpoint identity")
            paths.append(q)
    paths += prefixes + [control_path]
    if "contract_path" in p:
        paths.append(Path(p["contract_path"]))
        require(digest(p["contract_path"]) == p["contract_sha256"], "training contract binding")
    return states, paths, dict(status=v["status"], verification_sha256=digest(verification_path),
        original_step250_full_saved_state_exact=True,
        original_step1000_full_saved_state_exact=True,
        common_prefix43_full_saved_state_exact=True,
        exact_comparison_scope="complete checkpoint including original numerical contract; intervention policy lives in separate outer contract",
        control_effective_exposures=3992,
        treatment_effective_exposures=4000, removed_E_contributions=8)


def verified_sources():
    specs = [
        (PROMPT_DIR, "PASS_PROMPT_LOSS_RAW_ARITHMETIC_AND_COHORT", "verification.json"),
        (REPEAT_DIR, "PASS_REPEAT_NOISE_SAVED_TENSORS_AND_REUSE", "verification_v2.json"),
        (CROSS_DIR, "PASS_CROSS_RESPONSE_SAVED_ARITHMETIC_SOURCE_AND_REPLAYS", "verification.json"),
        *[(d, f"PASS_SAVED_{s}_COHORT_ARITHMETIC_AND_INTEGRITY", "verification.json") for s,d in CDI_DIRS.items()]]
    paths = []
    for d, status, verification_file in specs:
        v, e = read(d / verification_file), read(d / "execution.json")
        require(v["status"] == status, "required source verification " + str(d))
        for name, field in (("results.json", "results_sha256"), ("protocol.json", "protocol_sha256")):
            require(digest(d/name) == e[field], "source extraction output binding")
            if field in v:
                require(digest(d/name) == v[field], "source verification binding")
        paths += [d / n for n in ("protocol.json", "execution.json", "results.json", verification_file)]
        p = read(d/"protocol.json")
        paths.append(Path(p["contract_path"]))
        require(digest(p["contract_path"]) == p["contract_sha256"], "source contract binding")
    rows, _ = scope_rows()
    ids = {r["image_id"] for r in rows}
    prompt = {r["image_id"]: r for r in read(PROMPT_DIR/"results.json")
              if r["model"] == "model_1" and r["patient_id"] == PID}
    repeat = {r["image_id"]: r for r in read(REPEAT_DIR/"results.json") if r["model"] == "model_1"}
    dl = {r["image_id"]: r for d in CDI_DIRS.values() for r in read(d/"results.json")
          if r["model"] == "model_1" and r["image_id"] in ids}
    cross = [r for r in read(CROSS_DIR/"results.json") if r["recipient_model"] == "model_1"
             and (r["source_model"] is None or r["source_model"] == "model_1")]
    require(len(prompt)==4 and len(repeat)==160 and len(dl)==164 and len(cross)==100, "exact reused source scope")
    for d, mapping in ((PROMPT_DIR,prompt),(REPEAT_DIR,repeat)):
        for r in mapping.values():
            q = d/r["raw_path"]
            require(digest(q)==r["raw_sha256"], "source raw packet")
            paths.append(q)
    for r in dl.values():
        q = CDI_DIRS[r["scenario"]]/r["raw_path"]
        require(digest(q)==r["raw_sha256"], "CDI source raw")
        paths.append(q)
    paths.append(CROSS_DIR/"raw.pt")
    require(digest(CROSS_DIR/"raw.pt")==read(CROSS_DIR/"execution.json")["raw_sha256"], "MoFit source raw")
    return dict(prompt=prompt, repeat=repeat, dl=dl, cross=cross), paths


def policy():
    return dict(schema="masked-slot-endpoint-policy/v1", patient_id=PID, current_step=2,
        rows=164, patients=41, selection_patients=40, scenario_images={"E":2,"U":2},
        branches=list(BRANCHES), families=list(FAMILIES), model_dtype="float32", batch_size=1,
        generic_t140=dict(repeats=list(range(8)), prompt=PROMPT, prefixes=[1,2,4,8],
            original_selection_reuse=1280, original_p_repeat0_reuse=4, original_p_new=28, control_new=1312,
            score_precision="MSE: saved FP32 runtime loss; L2: CPU FP64 residual norm, matching repeated-noise analysis",
            noising="reuse exact saved input; new p repeats use frozen repeat_noise_t140 epsilon and scheduler.add_noise"),
        cdi_dl_t100=dict(repeats=list(range(5)), prefixes=[1,2,4,5], original_reuse=820, control_new=820,
            noising="exact saved original CDI noised input and CPU noise, not replaced by scheduler.add_noise",
            L2="literal GPU torch.norm epsilon residual; source saved module output on treatment",
            secondary_MSE="CPU reduction on both branches; original CDI did not export an MSE scalar"),
        fixed_mofit=dict(queries=P_IMAGES, conditions=["null"]+P_IMAGES, repeats=list(range(5)),
            original_reuse=100, control_new=100, embeddings="frozen original M1 post-update final embeddings",
            latent="same original posterior samples saved in cross_response",
            noising="exact cross_response saved input", new_optimization=False),
        expected_new_forward=2260, expected_backward=0, expected_vae_forward=0,
        expected_reused_cells=2204, expected_all_cells=4464,
        summary=dict(scores=["negative_MSE","negative_L2"], orientation="negative image loss before mean/max",
            primary_by_family={"generic_t140":"negative GPU MSE; FP64 L2 also reported",
                "cdi_dl_t100":"negative original GPU L2; CPU MSE and FP64 losses are sensitivity diagnostics",
                "fixed_mofit":"each frozen-embedding h=GPU conditioned MSE minus same-cell null MSE; null loss also reported"},
            change="treatment score minus masked-control score; no forced positive result",
            pools=["mean","max"], all_repeats=True, fixed_prefixes=True,
            patient_case="report p14393 separately; one fixed contribution intervention, not population efficacy",
            context="selection40 distributions of paired changes within A/B and all, descriptively",
            mofit="all four frozen source embeddings and null, raw loss and h=conditioned-minus-null, no best condition",
            AUC=False, CI=False, fitting=False, reference_selection=False, individual_population_causal_claim=False),
        limits=["Mask removes eight loss contributions with fixed denominator/slots; nonzero exposures4000 versus3992.",
            "Original M1 data cohort and all other scheduled records are fixed, unlike original M1-versus-M2 comparison.",
            "Local result for this selected patient/training realization is not an average patient causal or attack-performance estimate.",
            "MoFit fixed embeddings are treatment-derived and are not reoptimized on the control model.",
            "Cached VAE mode generic/DL and MoFit posterior samples are separate input families.",
            "No new training, fitting, parameter optimization, seed selection, AUC or automatic follow-up occurs in this runner."])


def prepare(args):
    require(not args.contract.exists() and not args.output_dir.exists(), "new immutable endpoint contract")
    states, training_paths, evidence = training_evidence(args.training_verification)
    sources, paths = verified_sources()
    rows, order = scope_rows()
    paths += training_paths
    paths += [Path(__file__), ROOT/"u_patient_audit/models.py", ROOT/"u_patient_audit/common.py",
        ROOT/"u_patient_audit/run_prompt_loss_t140.py",ROOT/"u_patient_audit/run_repeat_noise_t140.py",
        ROOT/"u_patient_audit/analyze_verify_repeat_noise_t140_v2.py",
        ROOT/"u_patient_audit/verify_fp32_noising.py",
        ROOT/"u_patient_audit/verify_masked_patient_training.py",
        ROOT/"u_patient_audit/cdi_adapter.py",ROOT/"u_patient_audit/run_mofit_cross_response.py",
        ROOT/"u_patient_audit/run_cdi_base_selection.py", RUN/"cache/cache.pt", RUN/"cache/summary.json",
        RUN/"cohort/lock.json",RUN/"cohort/evaluation_images.csv",RUN/"cohort/model_1_train.csv"]
    # Bind the already checked base, scheduler and text artifacts through source frozen manifests.
    frozen = dict(read(PROMPT_DIR/"protocol.json")["contract"]["frozen_sha256"])
    for p in paths:
        frozen[str(p.resolve())] = digest(p)
    c = dict(schema="masked-slot-endpoint-contract/v2", created_utc=now(), policy=policy(), verification_binding=VERIFIER_BINDING,
        training_directory=str(TRAINING), training_verification=str(args.training_verification.resolve()),
        training_evidence=evidence, checkpoint_paths=dict(treatment=str(OLD_CP),
            control=str(TRAINING/"control/step_1000.pt")),
        checkpoint_sha256=dict(treatment=digest(OLD_CP),control=digest(TRAINING/"control/step_1000.pt")),
        source_directories=dict(prompt=str(PROMPT_DIR),repeat=str(REPEAT_DIR),cross=str(CROSS_DIR),
            cdi_E=str(CDI_DIRS["E"]),cdi_U=str(CDI_DIRS["U"])),
        image_order=[r["image_id"] for r in rows],patient_order=order,
        output_directory=str(args.output_dir.resolve()), frozen_sha256=frozen,
        new_target_training=False,new_fitting=False,new_optimization=False)
    new_json(args.contract,c)
    return c


def validate(args):
    c=read(args.contract)
    require(c["schema"]=="masked-slot-endpoint-contract/v2" and c["policy"]==policy(), "fixed endpoint policy")
    require(Path(c["output_directory"]).resolve()==args.output_dir.resolve(), "fixed output directory")
    for path,h in c["frozen_sha256"].items():
        require(digest(path)==h, "frozen endpoint source/input "+path)
    states,_,evidence=training_evidence(Path(c["training_verification"]))
    require(evidence==c["training_evidence"], "same actual replay/exposure proof")
    sources,_=verified_sources()
    rows,order=scope_rows()
    require([r["image_id"] for r in rows]==c["image_order"] and order==c["patient_order"],"fixed query cohort")
    for b in BRANCHES:
        require(digest(c["checkpoint_paths"][b])==c["checkpoint_sha256"][b],"actual endpoint checkpoint")
    return c,states,sources,rows


def t140_inputs(row, sources, cache, sched):
    iid=row["image_id"]
    d, source = (PROMPT_DIR,sources["prompt"][iid]) if row["patient_id"]==PID else (REPEAT_DIR,sources["repeat"][iid])
    raw=torch.load(d/source["raw_path"],map_location="cpu",weights_only=True)
    latent=cache["latents"][iid].float()
    require(torch.equal(latent,raw["latent"]),"same t140 cached latent")
    expected_hidden=raw["prompt_hidden_sha256"]["generic"] if row["patient_id"]==PID else raw["hidden_sha256"]
    require(tensor_sha(cache["hidden"][PROMPT].float())==expected_hidden,"same generic t140 conditioning")
    inputs=[]
    for j in range(8):
        seed,noise=t140_noise(iid,j)
        if row["patient_id"]==PID:
            if j==0:
                require(seed==raw["noise_seed"] and torch.equal(noise,raw["epsilon"]),"same original p noise0")
                noisy,pred,loss=raw["noised_latent"],raw["predictions"]["generic"],source["losses"]["generic"]
            else:
                noisy=sched.add_noise(latent,noise,torch.tensor([140]));pred=loss=None
        else:
            q=raw["draws"][j]
            require(q["seed"]==seed and torch.equal(noise,q["epsilon"]),"same source t140 repeat")
            noisy,pred,loss=q["noised_latent"],q["prediction"],q["loss_mse_fp32"]
        inputs.append(dict(repeat=j,seed=seed,noisy=noisy,noise=noise,prediction=pred,source_mse=loss,
            source_l2=None,source_path=str((d/source["raw_path"]).resolve()),source_raw_sha256=source["raw_sha256"]))
    return latent,inputs


def dl_inputs(row,sources):
    iid=row["image_id"];source=sources["dl"][iid];d=CDI_DIRS[row["scenario"]]
    raw=torch.load(d/source["raw_path"],map_location="cpu",weights_only=True)
    require(float(raw["module_outputs"]["denoising_loss"].mean())==source["features"][0],"exact original DL feature")
    inputs=[]
    for j in range(5):
        q,n=raw["predictions"]["denoising_loss"][j],raw["noise_draws"]["denoising_loss"][j]
        require(q["timestep"]==100,"source DL t100")
        inputs.append(dict(repeat=j,seed=n["seed"],noisy=q["input"],noise=n["noise"],prediction=q["epsilon"],
            source_mse=None,source_l2=float(raw["module_outputs"]["denoising_loss"][0,j,0]),
            source_path=str((d/source["raw_path"]).resolve()),source_raw_sha256=source["raw_sha256"]))
    return raw["latent"],inputs


def measure(unet,inp,hidden,t,branch,family,count):
    reuse=branch=="treatment" and inp["prediction"] is not None
    if reuse:
        pred=inp["prediction"].clone()
    else:
        before=count["forward"]
        with torch.inference_mode():
            gpu_noise=inp["noise"].to("cuda")
            pred=unet(inp["noisy"].to("cuda"),torch.tensor([t],device="cuda"),
                      encoder_hidden_states=hidden.to("cuda"),return_dict=False)[0]
            gpu_mse=float(F.mse_loss(pred,gpu_noise)) if family=="fixed_mofit" else float((pred-gpu_noise).square().mean())
            gpu_l2=float(torch.norm((pred-gpu_noise).reshape(1,-1),p=2,dim=-1)[0])
        require(count["forward"]==before+1 and pred.dtype==torch.float32 and bool(torch.isfinite(pred).all()),"one finite endpoint query")
        pred=pred.cpu()
    residual=pred-inp["noise"]
    # Match source FP32 runtime loss conventions, and save independent-reduction inputs.
    if family=="cdi_dl_t100":
        mse=float((residual*residual).mean())
        l2=float(inp["source_l2"]) if reuse else gpu_l2
    else:
        mse=float(inp["source_mse"]) if reuse else gpu_mse
        l2=float(torch.norm(residual.reshape(1,-1),p=2,dim=-1)[0])
    return dict(repeat=inp["repeat"],seed=inp["seed"],prediction=pred,noised_latent=inp["noisy"],
        epsilon=inp["noise"],loss_mse_fp32=mse,loss_l2_fp32=l2,
        loss_mse_fp64=float((pred.double()-inp["noise"].double()).square().mean()),
        loss_l2_fp64=float(torch.linalg.vector_norm(pred.double()-inp["noise"].double())),
        mse_fp32_reduction="CPU secondary" if family=="cdi_dl_t100" else "source-matched GPU primary",
        l2_fp32_reduction="source-matched GPU primary" if family=="cdi_dl_t100" else "CPU secondary",
        reused=reuse,forward=int(not reuse),backward=0,
        source_path=inp["source_path"],source_raw_sha256=inp["source_raw_sha256"])


def metadata(row,branch,state,c):
    exposures=state["exposures"];iid=row["image_id"]
    allrows=read_csv(RUN/"cohort/model_1_train.csv")
    pex=sum(exposures.get(q["image_id"],0) for q in allrows if q["patient_id"]==row["patient_id"])
    return dict(branch=branch,patient_id=row["patient_id"],image_id=iid,scenario=row["scenario"],
        eval_role=row["eval_role"],assignment_group=row["assignment_group"],record_role=row["record_role"],
        effective_patient_member=int(pex>0),effective_image_member=int(exposures.get(iid,0)>0),
        effective_image_exposures=exposures.get(iid,0),effective_patient_exposures=pex,
        checkpoint_sha256=c["checkpoint_sha256"][branch],base_pretraining_membership="unknown")


def save_packet(out,branch,family,row,latent,hidden,cells,meta,results,source_image=None):
    directory=out/branch/family;directory.mkdir(parents=True,exist_ok=True)
    suffix="" if source_image is None else "_"+source_image
    path=directory/(Path(row["image_id"]).stem+suffix+".pt")
    require(not path.exists(),"immutable per-image endpoint packet")
    torch.save(dict(metadata=meta,family=family,latent=latent,hidden=hidden,cells=cells),path)
    raw_hash=digest(path)
    for index,cell in enumerate(cells):
        info={k:v for k,v in cell.items() if not isinstance(v,torch.Tensor)}
        results.append(dict(**meta,family=family,condition=source_image or "generic",
            raw_path=path.relative_to(out).as_posix(),raw_sha256=raw_hash,cell_index=index,**info))


def execute(args,c,states,sources,rows):
    out=args.output_dir;out.mkdir(parents=True,exist_ok=False)
    contract_hash=digest(args.contract);start=time.perf_counter()
    new_json(out/"protocol.json",dict(schema="masked-slot-endpoint-execution/v2",contract=c,
        contract_path=str(args.contract.resolve()),contract_sha256=contract_hash,code_sha256=digest(__file__),
        verification_binding=VERIFIER_BINDING,
        environment={n:version(n) for n in ("torch","numpy","diffusers","peft")}))
    cache=load_cache();sched=scheduler();require(sched.config.prediction_type=="epsilon","same epsilon backend")
    hidden=cache["hidden"][PROMPT].float()
    crossraw=torch.load(CROSS_DIR/"raw.pt",map_location="cpu",weights_only=True)
    cross_raw_hash=digest(CROSS_DIR/"raw.pt")
    crossindex={(r["query_image_id"],r["source_image_id"],r["noise_index"]):r for r in sources["cross"]}
    results,reports=[],[];unet=None;handles=[]
    try:
        setup()
        for branch in BRANCHES:
            ms=time.perf_counter()
            unet=load_unet(Path(c["checkpoint_paths"][branch]),training=False).float()
            unet.enable_adapters();unet.eval().requires_grad_(False);unet.disable_gradient_checkpointing()
            expected=states[branch]["adapter"];loaded=adapter_state(unet)
            require(loaded.keys()==expected.keys() and all(torch.equal(loaded[k],expected[k].float()) for k in loaded),"actual FP32 adapter values")
            before=state_fingerprint(unet);versions={n:p._version for n,p in unet.named_parameters()}
            guards(unet,versions);count=dict(forward=0,backward=0)
            def fh(module,inputs,output):count["forward"]+=1
            def bh(module,gin,gout):count["backward"]+=1
            handles=[unet.register_forward_hook(fh),unet.register_full_backward_hook(bh)]
            if branch=="treatment":
                previous=next(r for r in read(REPEAT_DIR/"execution.json")["model_reports"] if r["model"]=="model_1")
                require(before["sha256"]==previous["initial_state"]["sha256"],"same original treatment full weights")
            for number,row in enumerate(rows):
                meta=metadata(row,branch,states[branch],c)
                for family,t in (("generic_t140",140),("cdi_dl_t100",100)):
                    latent,inputs=t140_inputs(row,sources,cache,sched) if t==140 else dl_inputs(row,sources)
                    cells=[measure(unet,q,hidden,t,branch,family,count) for q in inputs]
                    save_packet(out,branch,family,row,latent,hidden,cells,meta,results)
                guards(unet,versions)
                if (number+1)%20==0:
                    print(json.dumps(dict(event="masked_endpoint_progress",branch=branch,images=number+1,
                        all_images=164,forward=count["forward"],seconds=time.perf_counter()-ms)),flush=True)
            for row in rows[:4]:
                iid=row["image_id"];meta=metadata(row,branch,states[branch],c)
                for source_iid in [None]+P_IMAGES:
                    h=crossraw["shared"]["null_hidden"] if source_iid is None else crossraw["shared"]["embeddings"]["model_1"][source_iid]
                    cells=[]
                    for j in range(5):
                        oldrow=crossindex[iid,source_iid,j];q=crossraw["cells"][oldrow["cell_id"]]
                        inp=dict(repeat=j,seed=oldrow["noise_seed"],noisy=q["noised_latent"],noise=q["target"],
                            prediction=q["prediction"],source_mse=oldrow["loss"],source_l2=None,
                            source_path=str((CROSS_DIR/"raw.pt").resolve()),source_raw_sha256=cross_raw_hash)
                        cells.append(measure(unet,inp,h,140,branch,"fixed_mofit",count))
                    save_packet(out,branch,"fixed_mofit",row,crossraw["shared"]["latents"][iid],h,cells,meta,results,
                                source_image=source_iid or "null")
            guards(unet,versions);after=state_fingerprint(unet)
            require(before["sha256"]==after["sha256"],"unchanged endpoint weights")
            current=adapter_state(unet)
            require(all(torch.equal(current[k],expected[k].float()) for k in current),"adapter exact after inference")
            require(count==dict(forward=28 if branch=="treatment" else 2232,backward=0),"fixed endpoint query count")
            reports.append(dict(branch=branch,counts=count,initial_state=before,final_state=after,
                all_parameters_frozen=True,no_parameter_gradients=True,parameter_versions_unchanged=True,
                actual_checkpoint_adapter_exact=True,unet_training=False,seconds=time.perf_counter()-ms))
            for h in handles:h.remove()
            handles=[];unet=None;del loaded,current
            gc.collect();torch.cuda.empty_cache()
        require(len(results)==4464 and sum(r["forward"] for r in results)==2260
                and sum(r["reused"] for r in results)==2204,"complete stored/new/reused cells")
        for p,h in c["frozen_sha256"].items():
            require(digest(p)==h,"unchanged endpoint input/code "+p)
        require(digest(args.contract)==contract_hash,"unchanged endpoint contract")
        new_json(out/"results.json",results)
        new_json(out/"execution.json",dict(status="PASS_MASKED_ENDPOINT_EXTRACTION_PENDING_INDEPENDENT_VERIFICATION",
            complete=True,records=4464,new_forward=2260,backward=0,vae_forward=0,reused_cells=2204,
            patients=41,unique_images=164,model_reports=reports,seconds=time.perf_counter()-start,
            results_sha256=digest(out/"results.json"),protocol_sha256=digest(out/"protocol.json"),
            contract_sha256=contract_hash,frozen_inputs_unchanged=True,verification_binding=VERIFIER_BINDING,
            new_target_training=False,new_optimization=False,new_fitting=False,AUC_computed=False,
            stage2_completion_claim=False))
        print(json.dumps(dict(status="PASS_MASKED_ENDPOINT_EXTRACTION_PENDING_INDEPENDENT_VERIFICATION",
            records=4464,new_forward=2260,seconds=time.perf_counter()-start)),flush=True)
    except BaseException:
        new_json(out/"failure.json",dict(status="FAILED_PRESERVED_MASKED_ENDPOINTS",traceback=traceback.format_exc(),
            completed_cells=len(results),seconds=time.perf_counter()-start))
        raise
    finally:
        for h in handles:h.remove()
        unet=None
        gc.collect()
        if torch.cuda.is_initialized():torch.cuda.empty_cache()


def main():
    p=argparse.ArgumentParser()
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--create-contract",action="store_true")
    mode.add_argument("--dry-run",action="store_true")
    mode.add_argument("--run",action="store_true")
    p.add_argument("--contract",type=Path,default=CONTRACT)
    p.add_argument("--output-dir",type=Path,default=OUTPUT)
    p.add_argument("--training-verification",type=Path,default=TRAINING/"verification.json")
    p.add_argument("--expected-code-sha256")
    args=p.parse_args()
    if args.expected_code_sha256:require(digest(__file__)==args.expected_code_sha256,"frozen endpoint producer")
    if args.create_contract:
        c=prepare(args)
        print(json.dumps(dict(status="PASS_ENDPOINT_CONTRACT_PREPARED_NO_GPU",sha256=digest(args.contract),
                             bound_files=len(c["frozen_sha256"]),new_forward=2260)))
        return
    c,states,sources,rows=validate(args)
    if args.dry_run:
        print(json.dumps(dict(status="PASS_MASKED_ENDPOINT_CPU_PREFLIGHT",images=len(rows),
            new_forward=2260,CUDA_initialized=torch.cuda.is_initialized(),contract_sha256=digest(args.contract))))
        return
    execute(args,c,states,sources,rows)


if __name__=="__main__":
    main()
