"""One explicitly post-hoc positive control; fixed pre-existing minimum grid C."""
import copy,gc,json,shutil,time
from pathlib import Path
import torch
from .common import ROOT,RUN,digest,write_json
from .models import setup,load_unet
from .run_clipping_observation_local import norm,dot,cosine,clip,flatten,restore,optimizer_step

def main():
    source=RUN/"protection_observation_20260915/clipping_observation_local_v1"
    out=source.parent/"clipping_activation_control_v1"
    if out.exists():raise FileExistsError(out)
    prior=json.loads((source/"results.json").read_text())
    pp=json.loads((source/"protocol.json").read_text())
    calibration=ROOT/"_reports/nih_cxr14_public_clip_calibration_v1_001/report.json"
    cc=json.loads(calibration.read_text())
    C=min(cc["selection_rule"]["grid"])
    assert C==.01
    assert all(r["both_gradients_unclipped"] and r["delta_A_B_norm"]==0 for r in prior)
    protocol=copy.deepcopy(pp)
    protocol.update(schema="clipping-activation-control/v1",clip_norm=C,
        scope="single post-hoc activation positive control; not operational C selection or DP/MIA efficacy",
        outcome_policy="triggered by all-inactive primary result; use minimum already-existing public calibration grid C=.01 once, never select E/U sign or sweep",
        source_run=str(source),source_results_sha256=digest(source/"results.json"),
        source_protocol_sha256=digest(source/"protocol.json"),calibration_path=str(calibration),calibration_sha256=digest(calibration),
        primary_forward=96,primary_backward=0,expected_forward=114,expected_backward=0,extra_VAE_images=0,
        numerical_controls="first triplet per state: baseline replay3F plus half actual displacement6F",
        code_sha256={str(Path(__file__).relative_to(ROOT)):digest(Path(__file__)),
            "u_patient_audit/run_clipping_observation_local.py":digest(Path(__file__).with_name("run_clipping_observation_local.py"))},
        gradient_source="unchanged saved FP32 gradients from primary; no new backward",
        expected_parameter_optimizer_steps=32)
    out.mkdir(parents=True,exist_ok=False)
    write_json(out/"protocol.json",protocol)
    for name in ("inputs.pt","state_250.pt","state_1000.pt"):
        shutil.copyfile(source/name,out/name)
    setup();started=time.perf_counter()
    inputs=torch.load(source/"inputs.pt",map_location="cpu",weights_only=True)
    forward=0;results=[]
    for binding in pp["checkpoint_bindings"]:
        step=binding["step"]
        assert digest(binding["path"])==binding["sha256"]
        state=torch.load(source/f"state_{step}.pt",map_location="cpu",weights_only=True)
        unet=load_unet(Path(binding["path"]),training=True).float()
        named=[(n,p) for n,p in unet.named_parameters() if p.requires_grad]
        params=[p for _,p in named]
        assert [n for n,_ in named]==state["parameter_names"]
        theta=state["theta"]
        assert torch.equal(flatten(params),theta)
        opt=torch.optim.AdamW(params,lr=1e-4,foreach=False,fused=False)
        tt=torch.tensor([pp["timestep"]],device="cuda")
        hh=inputs["hidden"].to("cuda")
        def predict(records):
            nonlocal forward
            values=[]
            with torch.no_grad():
                for row in records:
                    pred=unet(inputs["queries"][row["image_id"]]["noisy"].to("cuda"),tt,hh).sample
                    assert torch.isfinite(pred).all()
                    values.append(pred.cpu());forward+=1
            return values
        def contrast(pa,pb,records):
            return [float(((y.double()-x.double())*(y.double()+x.double()-
                2*inputs["queries"][row["image_id"]]["target"].double())).mean())
                for x,y,row in zip(pa,pb,records)]
        for index,pr in enumerate(r for r in prior if r["step"]==step):
            start=time.perf_counter()
            assert digest(source/pr["raw_file"])==pr["raw_sha256"]
            raw=torch.load(source/pr["raw_file"],map_location="cpu",weights_only=True)
            records=raw["records"]
            grads=raw["gradients"]
            g1,g2,gu=grads
            gm=(g1+g2)/2
            c1,a1=clip(g1,C);c2,a2=clip(g2,C)
            va=(c1+c2)/2;vb,b=clip(gm,C)
            ct=((a1-(a1+a2)/2)*(g1-gm)+(a2-(a1+a2)/2)*(g2-gm))/2
            cp=ct-gm*(dot(ct,gm)/max(dot(gm,gm),1e-300))
            ta=optimizer_step(params,opt,state["optimizer"],theta,va);pa=predict(records)
            tb=optimizer_step(params,opt,state["optimizer"],theta,vb);pb=predict(records)
            da=ta-theta;db=tb-theta;dab=ta-tb
            actual=contrast(pa,pb,records)
            linear=[-dot(g,dab) for g in grads]
            la=[float((p.double()-inputs["queries"][r["image_id"]]["target"].double()).square().mean()) for p,r in zip(pa,records)]
            lb=[float((p.double()-inputs["queries"][r["image_id"]]["target"].double()).square().mean()) for p,r in zip(pb,records)]
            controls={}
            if index==0:
                restore(params,theta);replay=predict(records)
                controls["baseline_replay_exact"]=[torch.equal(x,y) for x,y in zip(replay,raw["prediction_base"])]
                assert all(controls["baseline_replay_exact"])
                controls["baseline_replay_predictions"]=replay
                halves={}
                for label,d in (("A",da),("B",db)):
                    restore(params,theta+d*.5)
                    halves[label]=predict(records)
                controls["half_predictions"]=halves
                controls["half_displacement_score_A_minus_B"]=contrast(halves["A"],halves["B"],records)
                controls["half_displacement_note"]="actual FP32 displacement interpolation, not a second Adam step"
            path=out/pr["raw_file"]
            torch.save(dict(patient_id=pr["patient_id"],step=step,records=records,gradients=grads,
                delta_A=da,delta_B=db,prediction_base=raw["prediction_base"],prediction_A=pa,prediction_B=pb,
                controls=controls,source_raw_file=str(source/pr["raw_file"]),source_raw_sha256=pr["raw_sha256"]),path)
            r=copy.deepcopy(pr)
            r.update(clip_factors=[a1,a2],mean_clip_factor=b,both_gradients_unclipped=False,
                clipped_A_B_cosine=cosine(va,vb),clipped_A_B_norm_difference=norm(va-vb),
                c_perpendicular_norm=norm(cp),c_algebra_l2_error=norm(va-((a1+a2)/2*gm+ct)),
                delta_A_norm=norm(da),delta_B_norm=norm(db),delta_A_B_norm=norm(dab),delta_A_B_cosine=cosine(da,db),
                loss_A=la,loss_B=lb,score_A_minus_B=actual,linear_prediction_A_minus_B=linear,
                raw_SGD_gradient_contrast=[dot(g,va-vb) for g in grads],
                Emean_score_A_minus_B=sum(actual[:2])/2,U_score_A_minus_B=actual[2],
                Emean_linear_prediction=sum(linear[:2])/2,U_linear_prediction=linear[2],
                E_U_raw_sign_opposition=(sum(actual[:2])/2)*actual[2]<0,
                E_U_linear_sign_opposition=(sum(linear[:2])/2)*linear[2]<0,
                controls={k:v for k,v in controls.items() if k not in ("half_predictions","baseline_replay_predictions")},
                raw_sha256=digest(path),seconds=time.perf_counter()-start)
            results.append(r)
            print(json.dumps({k:r[k] for k in ("patient_id","step","clip_factors","Emean_score_A_minus_B","U_score_A_minus_B","seconds")}),flush=True)
        del unet,opt,named,params,hh
        gc.collect();torch.cuda.empty_cache()
    assert forward==114,forward
    write_json(out/"results.json",results)
    report=dict(status="COMPLETED_ACTIVATION_CONTROL_PENDING_INDEPENDENT_VERIFICATION",
        patients=8,state_conditions=16,forward=forward,backward=0,vae_images=0,
        elapsed_seconds=time.perf_counter()-started,results_sha256=digest(out/"results.json"),
        protocol_sha256=digest(out/"protocol.json"),inputs_sha256=digest(out/"inputs.pt"),
        max_cuda_GiB=torch.cuda.max_memory_allocated()/2**30,
        source_checkpoints_unchanged=all(digest(x["path"])==x["sha256"] for x in pp["checkpoint_bindings"]),
        scope=protocol["scope"],clip_norm=C)
    assert report["source_checkpoints_unchanged"]
    write_json(out/"report.json",report)
    print(json.dumps(report),flush=True)
if __name__=="__main__":main()
