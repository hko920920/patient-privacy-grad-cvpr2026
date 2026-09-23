"""CPU cache verification only: never randomize Q or execute synthesis/eval."""
from pathlib import Path
import argparse,json,time,shutil
import numpy as np
import torch
from prrd_v3.contracts import CODE,RESEARCH,read,rows,save,sha,require,now
from prrd_v3.datasets import TRAIN
from . import patient_dp as dp
from .patient_dp_io import cached_paths,read_cached_population,write_target_handoffs
from .second_source import load_combined
from .signals import bind_objective


def compare_population(p,reference,manifest,role):
    s,n=p.sums_counts();meta=reference['metadata']
    group=[r for r in manifest if r['planned_namespace']=={
        'P':'real/public','Q':'diagnostic_real/former_classifier_selection'}[role]]
    for key,expected in [('image_ids',[r['image_id'] for r in group]),
                         ('patient_ids',[r['patient_id'] for r in group]),
                         ('labels',[int(r['label']) for r in group])]:
        require(np.array_equal(meta[key],expected),'Manifest/cache mismatch: '+role+'/'+key)
    require(np.array_equal(n,reference['counts']),'Patient class count mismatch')
    target=dp.raw_balanced_target(s,n,np.zeros_like(s),np.zeros(2))
    errors={'sums':float(abs(s-reference['sums']).max()),
            'targets':float(abs(target-reference['targets']).max())}
    require(max(errors.values())<=1e-12,'Historical target reproduction exceeds1e-12')
    return {'patients':len(p.ids),'images':len(group),'class_counts':n.tolist(),
            'max_abs_errors':errors,'metadata_order_exact':True}


def public_handoff_check(output,paths,p,limits):
    """Actual public cache as a simulated Q; no real Q input is accepted."""
    ps,pc=p.sums_counts();contributions,_=dp.clipped_contributions(p,limits)
    z=np.random.default_rng(823).standard_normal(limits.dimension)
    noisy,metadata=dp.simulate_public(dp.sum_patient_axis(contributions),limits,z)
    target,post=dp.decode_target(noisy,limits,ps,pc)
    handoffs=write_target_handoffs(output/'public_simulation',target,paths,metadata,
                                  privacy='PUBLIC_SIMULATION_ONLY')
    labels=(0,0,1,1)
    loaded,_=load_combined(handoffs['a1_handoff'],handoffs['second_handoff'],labels,device='cpu')
    direct=bind_objective(loaded.conditions,loaded.heads,
                          {n:torch.tensor(target[n]) for n in dp.ENCODERS},labels)
    values={}
    for name in dp.ENCODERS:
        with np.load(paths[name]['P'],allow_pickle=False) as f:
            indices=np.r_[np.flatnonzero(f['labels']==0)[:2],np.flatnonzero(f['labels']==1)[:2]]
            values[name]=torch.tensor(f['z'][:,indices].astype(np.float64),requires_grad=True)
    loss,_=loaded.matching(values);grads=torch.autograd.grad(loss,tuple(values.values()))
    mirror={n:v.detach().clone().requires_grad_(True) for n,v in values.items()}
    expected,_=direct.matching(mirror);refs=torch.autograd.grad(expected,tuple(mirror.values()))
    error=max(float((g-r).abs().max()) for g,r in zip(grads,refs))
    require(torch.isfinite(loss) and all(torch.isfinite(g).all() and g.norm()>0 for g in grads),
            'Loaded target has invalid/zero feature gradient')
    require(error==0 and float(loss)==float(expected),'Serialization changed objective/gradient')
    return {'scope':'PUBLIC_SIMULATION_ONLY','status':'PASS','loss':float(loss),
            'target_bytes_roundtrip_exact':all(np.array_equal(loaded.targets[n].numpy(),target[n]) for n in dp.ENCODERS),
            'feature_gradient_max_error':error,'gradient_norms':[float(g.norm()) for g in grads],
            'encoder_forwards':0,'image_gradient_rerun':False,'Q_noise_draws':0,
            'postprocessing':post}


def verify(output):
    started=time.monotonic();output=output.resolve()
    require(read(output/'unit_tests_first.json')['status']=='PASS','Constructed-array tests missing')
    contract=read(output/'work_contract.json')
    require(not contract['Q_noise_release_authorized'] and not contract['synthesis_authorized'],
            'Wrong verifier work scope')
    reference_job=CODE/'_reports/receiver_repeat202_s200_20260923_v1/jobs/A2.json'
    job=read(reference_job)
    require(all(sha(p)==h for p,h in job['code_bindings'].items()),'Historical A2 code changed')
    paths=cached_paths(reference_job)
    own=list(Path(__file__).parent.glob('patient_dp*.py'))+[Path(__file__),Path(__file__).with_name('test_patient_dp.py')]
    inputs={str(p):sha(p) for p in own+[reference_job,TRAIN]}
    for item in paths.values():
        for key in ('handoff','target','rule_file','P','Q'):
            inputs[item[key]]=sha(item[key])
        for key in ('checkpoint','projection'):inputs[item['rule'][key]]=sha(item['rule'][key])
    before={'created':now(),'input_code_bindings':inputs,'existing_runtime_code':job['code_bindings'],
            'new_encoder_forwards':0,'Q_noise_draws':0,'target_reproduction_atol':1e-12}
    save(output/'verification_contract.json',before)
    snapshot=output/'implementation_snapshot';snapshot.mkdir(exist_ok=False)
    for p in own:shutil.copy2(p,snapshot/p.name)
    shutil.copy2(RESEARCH/'RECEIVER_PATIENT_DP_TARGET_SPEC_20260923.md',snapshot/'prospective_spec.md')

    # P-only selection is sealed before loading Q values. Q cache hashes above
    # are integrity checks, never inputs to thresholds or noise calibration.
    p,pref=read_cached_population(paths,'P');limits,details=dp.public_limits(p)
    calibration={'schema':'receiver.public-patient-gradient-calibration/v1','created':now(),
        'population':'P_ONLY','class_bounds':list(limits.class_bounds),'rule':'class-present norm q95 linear; floor1e-6',
        'details':details,'public_feature_hashes':pref['feature_hashes'],
        'public_rule_hashes':{n:sha(paths[n]['rule_file']) for n in dp.ENCODERS},
        'chosen_before_loading_Q_values':True,'Q_or_V_used_for_choice':False}
    calibration_path=output/'public_calibration.json';save(calibration_path,calibration)
    calibration_hash=sha(calibration_path)
    save(output/'proposed_mechanism.json',dp.mechanism_metadata(limits))
    print(json.dumps({'phase':'P_CALIBRATION_SEALED','class_bounds':limits.class_bounds}),flush=True)

    q,qref=read_cached_population(paths,'Q')
    require(not(set(p.ids)&set(q.ids)),'P/Q patients overlap')
    manifest=rows(TRAIN)
    populations={role:compare_population(data,ref,manifest,role)
                 for role,data,ref in [('P',p,pref),('Q',q,qref)]}
    ps,pc=p.sums_counts();qs,qc=q.sums_counts()
    raw=dp.raw_balanced_target(ps,pc,qs,qc)
    pooled_errors={}
    for i,name in enumerate(dp.ENCODERS):
        with np.load(paths[name]['target'],allow_pickle=False) as f:
            pooled_errors[name]=float(abs(raw[i]-f['pooled_gradient']).max())
    require(max(pooled_errors.values())<=1e-12,'Pooled target failed reproduction')

    # Internal clipping-only arithmetic. No noise or utility computed on Q.
    contrib,diag=dp.clipped_contributions(q,limits)
    bounded_sum=dp.sum_patient_axis(contrib)
    target,post=dp.decode_target(bounded_sum,limits,ps,pc)
    clipped=np.stack([target[n] for n in dp.ENCODERS])
    expected_sums=np.zeros_like(qs);expected_counts=np.zeros(2)
    for j in range(len(q.ids)):
        for c in (0,1):
            if not q.present[j,c]:continue
            mean=q.means[j,c];length=float(np.sqrt(np.sum(mean*mean)))
            factor=min(1.,limits.class_bounds[c]/max(length,np.finfo(float).tiny))
            rounding=diag['joint_factors'][j]
            expected_sums[c]+=mean*factor*rounding;expected_counts[c]+=rounding
    expected=dp.raw_balanced_target(ps,pc,expected_sums,expected_counts)
    clipped_error=float(abs(clipped-expected).max())
    require(clipped_error<=1e-12,'Clipping-only independent target mismatch')
    require(np.linalg.norm(contrib,axis=1).max()<=1.,'Actual Q bound exceeded')
    require(np.max(abs(2*bounded_sum[-2:]-qc))<=1e-10,'Unexpected count rescaling')
    require(sha(calibration_path)==calibration_hash,'Calibration changed after Q was loaded')
    clipping={'scope':'INTERNAL_NONPRIVATE_NOT_A_RELEASE','patients':len(q.ids),
        'class_counts':qc.tolist(),'class_clipped_patients':diag['class_clipped_patients'].tolist(),
        'class_clipped_fraction':(diag['class_clipped_patients']/qc).tolist(),
        'joint_roundoff_limited_patients':diag['joint_roundoff_limited_patients'],
        'max_contribution_norm':float(np.linalg.norm(contrib,axis=1).max()),
        'independent_target_max_abs':clipped_error,
        'count_reconstruction_max_abs':float(abs(2*bounded_sum[-2:]-qc).max()),
        'target_absolute_change_by_encoder_condition':np.linalg.norm(clipped-raw,axis=-1).tolist(),
        'target_relative_change_by_encoder_condition':(np.linalg.norm(clipped-raw,axis=-1)/np.linalg.norm(raw,axis=-1)).tolist(),
        'raw_target_norms':np.linalg.norm(raw,axis=-1).tolist(),
        'consistency_projection_factors':post['consistency_projection_factor'],
        'public_fallback_conditions':post['public_fallback_conditions'],'Q_noise_draws':0}
    save(output/'INTERNAL_clipping_verification.json',clipping)
    public=public_handoff_check(output,paths,p,limits)
    require(all(sha(f)==h for f,h in inputs.items()),'Inputs changed during verification')
    require(all(sha(f)==h for f,h in job['code_bindings'].items()),'Legacy code changed')

    proposal={'schema':'receiver.patient-dp-release-plan/v1','status':'DRAFT_NOT_AUTHORIZED',
        'created':now(),'reference_job':str(reference_job),'public_calibration':str(calibration_path),
        'public_calibration_sha256':sha(calibration_path),'input_code_bindings':inputs,
        'output_directory':str(CODE/'_reports/receiver_patient_dp_first_release_20260923_v1'),
        'epsilon':8.,'delta':1e-5,'adjacency':'add_remove_one_patient','max_private_releases':1,
        'synthesis_authorized':False,'new_banks_proposed':['A2_Q_CLIPPED_NO_NOISE_101','A2_Q_DP8_101'],
        'bank_recipe':{'images':128,'updates':200,'seed':101,'condition_seed':101,'microbatch':16,
                       'optimizer':job['optimizer'],'activation_updates':job['activation_updates']},
        'reuse_controls':['existing A2 P-only101','existing A2 nonDP P+Q101'],
        'evaluation':'Freeze both new final PNGs before existing V BioViL/DenseNet paired2000 patient draws',
        'synthesis_reference_minutes':148,'synthesis_estimate_range_minutes':[140,160],
        'end_to_end_planning_minutes':[155,180],
        'proposed_synthesis_time_cap_seconds_per_bank':7200,
        'proposed_package_time_cap_seconds':18000,
        'mandatory_before_execution':['freeze exact one-release and two-bank scope with time cap',
            'attach protected target hashes to isolated jobs; metadata must distinguish clipping-only and DP',
            'keep raw-Q diagnostics and nonDP checkpoints outside protected artifact package'],
        'no_new_encoder_extraction_required':True,'Expert_Reserved_final_receiver':False,
        'no_automatic_noise_redraw_or_expansion':True}
    save(output/'first_dp_comparison_DRAFT.json',proposal)
    result={'created':now(),'status':'PASS_TARGET_PATH_NO_Q_RELEASE_OR_TRAINING',
        'unit_tests':read(output/'unit_tests_first.json')['tests'],'public_calibration':calibration,'mechanism':dp.mechanism_metadata(limits),
        'populations':populations,'pooled_target_errors':pooled_errors,'clipping':clipping,
        'public_handoff_objective':public,'legacy_code_and_inputs_unchanged':True,
        'private_release_executed':False,'Q_noise_draws':0,'new_feature_extractions':0,
        'encoder_forwards':0,'encoder_backwards':0,'optimizer_updates':0,'V_utility_evaluations':0,
        'Expert_Reserved_final_receiver_access':False,'seconds':time.monotonic()-started,
        'proposal':str(output/'first_dp_comparison_DRAFT.json')}
    save(output/'verification.json',result)
    print(json.dumps({k:result[k] for k in ('status','populations','pooled_target_errors','seconds')},ensure_ascii=False),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();verify(args.output)


if __name__=='__main__':main()
