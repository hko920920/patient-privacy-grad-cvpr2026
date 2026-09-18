"""Step2 only: extract P raw/point paths and bind patient relation statistics.

No classifier solve, synthesis updates, loss integration, DP or Q/V pixels.
"""
from __future__ import annotations
import argparse,json,math,time
from pathlib import Path
import numpy as np
import torch
from torchvision.transforms import functional as TF
from .contracts import setup,OUT,sha,save,read,source_hashes,require,now
from .datasets import manifests,PixelAccess,TRAIN,DEV
from .encoders import Encoder,state_hash,preprocess,pil_tensor,BIO_PATH,BIO_SHA
from .patient_moments import aggregate,patient_contributions
from .relation_paths import (patient_uniform_center,patient_paths,public_scales,
                              relation_target,paired_relation)


def run(output):
    require(not output.exists(),'Step2 output must be new; do not overwrite completed evidence')
    output.mkdir(parents=True);setup();started=time.perf_counter()
    groups=manifests();group=groups['P'];access=PixelAccess(groups,('P',)).install()
    old=read(OUT/'source_P_projection.json')
    require(sha(OUT/'source_P_projection.npz')==old['projection']['sha256'],'Point PCA changed')
    require(sha(OUT/'source_P_features.npz')==old['feature_sha256'],'Legacy cache changed')
    legacy=np.load(OUT/'source_P_features.npz',allow_pickle=False)
    ids=np.array([row['image_id'] for row in group]);pids=np.array([row['patient_id'] for row in group])
    labels=np.array([int(row['label']) for row in group]);roles=np.array(['P']*len(group))
    require(np.array_equal(legacy['image_ids'],ids),'Legacy P cache order differs')
    model_source=Path(__import__('health_multimodal.image.model.model',fromlist=['ImageModel']).__file__)
    contract={'schema':'prrd.step2.public-feature-paths/v1','created':now(),'scope':'P_ONLY_FEATURE_AND_PATIENT_STATISTICS',
        'source_sha256':source_hashes(),'inputs':{str(p):sha(p) for p in (TRAIN,DEV,OUT/'source_P_projection.npz',OUT/'source_P_features.npz',model_source)},
        'checkpoint_path':str(BIO_PATH),'checkpoint_sha256':BIO_SHA,'images':len(group),'patients':len(set(pids)),
        'encoder':'BioViL-T image model; same checkpoint and projected_global_embedding',
        'h_location':'ImageModel.forward_post_encoder: mean(projected_patch_embeddings,dim=(2,3)); before wrapper F.normalize',
        'raw_center':'equal patient mass; uniform images inside patient; P only; label-free',
        'basis':'unchanged source_P_projection.npz basis; NOT refitted on raw features',
        'point':'existing F.normalize(h), public point center/PCA, fixed point norm cap',
        'relation':'a=(h-public_raw_center)@same_point_basis; patient class means; contrast/2; then bound',
        'scales':'P mixed patients; inverse empirical CDF .95; equal patient mass; no interpolation; floor1e-12',
        'feature_microbatch':4,'cached_projection_batch':len(group),'dtype':'model/projection FP32; patient aggregation FP64',
        'criteria':{'same_partition_legacy_point_exact':True,'legacy_normalized_cache_exact':True,
                    'independent_feature_atol':2e-6,'independent_feature_rtol':1e-5,'FP64_statistics_atol':1e-12},
        'guard_or_function_loss':False,'rho_eta_selection':False,'Q_V_pixels':False,
        'full_bank_profile':False,'W2':False,'DP':False,'expert_reserved':False,'remote_upload':False}
    save(output/'feature_path_contract.json',contract)
    torch.manual_seed(918);encoder=Encoder('source').use_projection(OUT/'source_P_projection.npz')
    before=state_hash(encoder.model);require(before==old['encoder_state_sha256'],'Source state changed')
    counts={'forward_calls':0,'forward_images':0,'backward_calls':0,'backward_images':0};captured={}
    def hook(module,inputs,result):
        n=len(inputs[0]);counts['forward_calls']+=1;counts['forward_images']+=n
        captured['h']=result.projected_global_embedding
        captured['spatial_mean']=result.projected_patch_embeddings.mean(dim=(2,3))
        if result.projected_global_embedding.requires_grad:
            def backward(g):
                counts['backward_calls']+=1;counts['backward_images']+=n
                return g
            result.projected_global_embedding.register_hook(backward)
    handle=encoder.model.register_forward_hook(hook)
    features=[];point_inputs=[];phase=time.perf_counter()
    for start in range(0,len(group),4):
        batch=torch.stack([preprocess(pil_tensor(access.image(row)),'source') for row in group[start:start+4]]).cuda()
        with torch.no_grad():
            dual=encoder.raw_and_point(batch)
            require(torch.equal(dual['h'],captured['h']) and torch.equal(dual['h'],captured['spatial_mean']),
                    'Raw extraction location differs from projected patch spatial mean')
            require(not dual['h'].requires_grad,'Unexpected private extraction graph')
        features.append(dual['h'].detach().cpu().numpy());point_inputs.append(dual['point_input'].detach().cpu().numpy())
        captured.clear();del dual,batch
        if start%128==0:print(json.dumps({'phase':'P_raw_features','images':start+len(features[-1])}),flush=True)
    extraction_seconds=time.perf_counter()-phase
    h=np.concatenate(features);point_input=np.concatenate(point_inputs)
    np.savez(output/'source_P_pre_external_norm_features.npz',h=h,point_input=point_input,
             image_ids=ids,patient_ids=pids,labels=labels,roles=roles)
    save(output/'legacy_cache_comparison.json',{'exact':bool(np.array_equal(point_input,legacy['features'])),
         'maximum_abs':float(np.max(np.abs(point_input.astype(float)-legacy['features'].astype(float))))})
    require(np.array_equal(point_input,legacy['features']),'Same-partition normalized cache changed')
    raw_center,weights=patient_uniform_center(h,pids)
    with np.load(OUT/'source_P_projection.npz',allow_pickle=False) as old_projection:
        basis=old_projection['basis'].copy()
        require(np.array_equal(weights,old_projection['patient_weights']),'Public patient weighting changed')
    projection=output/'source_P_relation_projection.npz'
    np.savez(projection,raw_center=raw_center,basis=basis,role=np.array('source'),
             point_projection_sha256=np.array(sha(OUT/'source_P_projection.npz')),
             feature_sha256=np.array(sha(output/'source_P_pre_external_norm_features.npz')),patient_weights=weights,image_ids=ids)
    encoder.use_relation_projection(projection)
    with torch.no_grad():
        z=encoder.project(torch.from_numpy(point_input).cuda()).cpu().numpy()
        old_z=encoder.project(torch.from_numpy(legacy['features']).cuda()).cpu().numpy()
        a=encoder.project_relation(torch.from_numpy(h).cuda()).cpu().numpy()
    require(np.array_equal(z,old_z),'Full-P legacy point output changed')
    np.savez(output/'source_P_dual_features.npz',z=z,a=a,image_ids=ids,patient_ids=pids,labels=labels,roles=roles)
    patients=patient_paths(z,a,labels,pids,roles);scales,calibration=public_scales(patients)
    require(calibration['mixed_patients']==3,'Public mixed population changed')
    save(output/'source_P_relation_scales.json',{'scales':scales,'calibration':calibration,
          'raw_features_sha256':sha(output/'source_P_pre_external_norm_features.npz'),
          'dual_features_sha256':sha(output/'source_P_dual_features.npz'),'projection_sha256':sha(projection)})
    np.savez(output/'patient_paths_P.npz',patient_ids=np.array([p.point.patient_id for p in patients]),
             present=np.stack([p.point.present for p in patients]),point_means=np.stack([p.point.means for p in patients]),
             point_seconds=np.stack([p.point.seconds for p in patients]),raw_affine_means=np.stack([p.raw_means for p in patients]))
    targets={};saved_targets={}
    legacy_target=aggregate(patient_contributions(old_z,labels,pids),'point')
    for arm in ('E','E_R','C','D','R_joint'):
        target,rows=relation_target(patients,arm,scales)
        for key in ('m','A'):require(np.array_equal(target[key],legacy_target[key]),'Patient point moments changed')
        require(np.array_equal(target['counts'][:2],legacy_target['counts']),'Patient class masses changed')
        for key,value in target.items():saved_targets[arm+'_'+key]=value
        for key in ('r','raw_delta','u'):saved_targets[arm+'_'+key+'_rows']=rows[key]
        targets[arm]={'counts':target['counts'].tolist(),'relation_max_norm':float(np.linalg.norm(rows['r'],axis=1).max()),
                      'post_bound_relation_mean':target['delta'].tolist()}
    np.savez(output/'source_P_relation_targets.npz',**saved_targets)
    # A fixed public mixed patient's two labeled images; input-gradient check only.
    mixed=[p.point.patient_id for p in patients if p.point.mixed];pid=mixed[0]
    witness=[next(row for row in group if row['patient_id']==pid and int(row['label'])==label) for label in (0,1)]
    x=torch.stack([TF.resize(pil_tensor(access.image(row)),[224,224],antialias=True) for row in witness]).cuda().requires_grad_(True)
    weight=torch.linspace(-.5,.5,16,device='cuda')
    old_point=encoder(x);old_grad,=torch.autograd.grad((old_point*weight).sum(),x)
    paths=encoder.forward_paths(x);point_grad,=torch.autograd.grad((paths['z']*weight).sum(),x,retain_graph=True)
    require(torch.equal(paths['z'],old_point) and torch.equal(point_grad,old_grad),'Point witness value/gradient changed')
    relation=paired_relation(paths['z'][1],paths['z'][0],paths['a'][1],paths['a'][0],'C',scales)['r']
    raw_gradient,=torch.autograd.grad((relation*weight).sum(),x)
    require(torch.isfinite(raw_gradient).all() and raw_gradient.abs().max()>0,'Raw relation path lacks a finite image gradient')
    witness_record={'images':[row['image_id'] for row in witness],'point_output_exact':True,'point_gradient_exact':True,
                    'point_gradient_norm':float(point_grad.norm()),'raw_relation_gradient_norm':float(raw_gradient.norm()),
                    'raw_relation_gradient_max_abs':float(raw_gradient.abs().max()),
                    'scalar_is_probe_not_training_loss':True,'new_loss_or_optimizer_updates':0}
    handle.remove();captured.clear()
    require(state_hash(encoder.model)==before,'Frozen model parameters/buffers changed')
    require(not any(module.training for module in encoder.model.modules()),'Model left eval mode')
    require(not any(p.grad is not None for p in encoder.model.parameters()),'Model accumulated parameter gradients')
    require(sha(OUT/'source_P_projection.npz')==old['projection']['sha256'] and
            sha(OUT/'source_P_features.npz')==old['feature_sha256'],'Old point artifacts mutated')
    norms=np.linalg.norm(h.astype(float),axis=1)
    summary={'status':'PUBLIC_PATHS_BUILT_INDEPENDENT_SAVED_FEATURE_CHECK_PENDING','source_state_unchanged':True,
             'old_normalized_cache_exact':True,'old_full_P_point_output_exact':True,'point_patient_moments_exact':True,
             'raw_norm_min_median_max':[float(norms.min()),float(np.median(norms)),float(norms.max())],
             'point_z_max_norm':float(np.linalg.norm(z.astype(float),axis=1).max()),'scales':scales,'targets':targets,
             'actual_gradient_witness':witness_record,'counts':counts,'access':access.report(),
             'extraction_seconds':extraction_seconds,'total_seconds':time.perf_counter()-started,
             'feature_path_contract_sha256':sha(output/'feature_path_contract.json'),
             'artifacts':{f.name:sha(f) for f in sorted(output.glob('*.npz'))},
             'new_guard_function_loss_integrated':False,'rho_eta_selected':False,'full_W1_passed':False,
             'new_synthesis_updates':0,'Q_V_pixel_access':0,'DP_releases':0,'expert_reserved_access':False}
    save(output/'public_feature_paths_result.json',summary)
    print(json.dumps({k:summary[k] for k in ('status','scales','counts','access','extraction_seconds','total_seconds')}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
