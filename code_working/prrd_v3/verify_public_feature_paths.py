"""Independent CPU-only reconstruction from saved P features.

Intentionally does not import the producer's encoder, relation/statistic helpers,
Torch or any image loader. It is not a patient utility evaluation.
"""
import argparse,json,math,hashlib
from pathlib import Path
import numpy as np


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def run(directory):
    contract=read(directory/'feature_path_contract.json');result=read(directory/'public_feature_paths_result.json')
    errors={};checks={}
    def check(name,ok):
        checks[name]=bool(ok)
        if not ok:raise AssertionError(name)
    def close(name,actual,expected,atol=1e-12,rtol=0):
        actual=np.asarray(actual);expected=np.asarray(expected)
        err=float(np.max(abs(actual-expected))) if actual.size else 0.
        errors[name]=err
        check(name,actual.shape==expected.shape and np.allclose(actual,expected,atol=atol,rtol=rtol))
    for name,h in result['artifacts'].items():check('artifact_hash_'+name,sha(directory/name)==h)
    for filename,h in contract['source_sha256'].items():check('source_hash_'+Path(filename).name,sha(filename)==h)
    for filename,h in contract['inputs'].items():check('input_hash_'+Path(filename).name,sha(filename)==h)
    raw=np.load(directory/'source_P_pre_external_norm_features.npz',allow_pickle=False)
    dual=np.load(directory/'source_P_dual_features.npz',allow_pickle=False)
    projection=np.load(directory/'source_P_relation_projection.npz',allow_pickle=False)
    ppath=next(Path(k) for k in contract['inputs'] if Path(k).name=='source_P_projection.npz')
    point_projection=np.load(ppath,allow_pickle=False)
    legacy_path=next(Path(k) for k in contract['inputs'] if Path(k).name=='source_P_features.npz')
    legacy=np.load(legacy_path,allow_pickle=False)
    saved=np.load(directory/'patient_paths_P.npz',allow_pickle=False)
    targets=np.load(directory/'source_P_relation_targets.npz',allow_pickle=False)
    scales_record=read(directory/'source_P_relation_scales.json');scales=scales_record['scales']
    for key in ('image_ids','patient_ids','labels','roles'):check('metadata_'+key,np.array_equal(raw[key],dual[key]))
    check('P_only',set(raw['roles'])=={'P'})
    check('legacy_normalized_cache_exact',np.array_equal(raw['point_input'],legacy['features']))
    check('image_order_exact',np.array_equal(raw['image_ids'],legacy['image_ids']))
    check('PCA_axes_unmodified',np.array_equal(projection['basis'],point_projection['basis']))
    check('projection_bound',str(projection['point_projection_sha256'])==sha(ppath))
    ids=raw['patient_ids'];y=raw['labels'];h=raw['h'].astype(np.float64)
    ps=sorted(set(ids));check('P_counts',len(ps)==672 and len(h)==813)
    center=np.array([h[ids==pid].mean(0) for pid in ps]).mean(0)
    close('raw_patient_uniform_center',projection['raw_center'],center)
    # Reconstruct the declared FP32 constants, but use independent FP64 arithmetic.
    basis=point_projection['basis'].astype(np.float32).astype(np.float64)
    norm=np.sqrt(np.sum(h*h,axis=1,keepdims=True))
    normal=h/np.maximum(norm,1e-12)
    rz=(normal-point_projection['center'].astype(np.float32).astype(np.float64))@basis
    rz=rz/np.maximum(np.sqrt(np.sum(rz*rz,axis=1,keepdims=True)),float(point_projection['scale'].astype(np.float32)))
    ra=(h-projection['raw_center'].astype(np.float32).astype(np.float64))@basis
    crit=contract['criteria']
    close('independent_point_z',dual['z'],rz,crit['independent_feature_atol'],crit['independent_feature_rtol'])
    close('independent_raw_affine_a',dual['a'],ra,crit['independent_feature_atol'],crit['independent_feature_rtol'])
    z=dual['z'].astype(np.float64);a=dual['a'].astype(np.float64);d=z.shape[1]
    present=np.zeros((len(ps),2),int);means=np.zeros((len(ps),2,d));second=np.zeros((len(ps),2,d,d));am=np.zeros_like(means)
    for i,pid in enumerate(ps):
        for c in (0,1):
            ix=np.flatnonzero((ids==pid)&(y==c))
            if len(ix):
                present[i,c]=1
                means[i,c]=sum((z[k] for k in ix),np.zeros(d))/len(ix)
                second[i,c]=sum((np.outer(z[k],z[k]) for k in ix),np.zeros((d,d)))/len(ix)
                am[i,c]=sum((a[k] for k in ix),np.zeros(d))/len(ix)
    check('patient_order',np.array_equal(saved['patient_ids'],np.array(ps)))
    for key,value in [('present',present),('point_means',means),('point_seconds',second),('raw_affine_means',am)]:
        close(key,saved[key],value)
    mix=np.flatnonzero(present.prod(1));check('mixed_population',len(mix)==3)
    deltas=(am[mix,1]-am[mix,0])/2
    endpoints=(means[mix,1]-means[mix,0])/2
    joint=np.concatenate([am[mix,1],am[mix,0]],axis=1)/math.sqrt(2)
    for key,value in [('C',deltas),('E_R',endpoints),('R_joint',joint)]:
        norms=sorted(math.sqrt(float(row@row)) for row in value)
        expected=max(norms[math.ceil(.95*len(norms))-1] if norms else 0.,1e-12)
        close('public_scale_'+key,scales[key],expected)
    point_m=np.zeros((2,d));point_A=np.zeros((2,d,d));counts=present.sum(0)
    for c in (0,1):
        ix=np.flatnonzero(present[:,c])
        point_m[c]=sum((means[k,c] for k in ix),np.zeros(d))/max(len(ix),1)
        point_A[c]=sum((second[k,c] for k in ix),np.zeros((d,d)))/max(len(ix),1)
    def cap_rows(array,radius):
        return np.array([row/max(float(radius),math.sqrt(float(row@row))) for row in array])
    for arm in ('E','E_R','C','D','R_joint'):
        if arm=='E':rr=endpoints
        elif arm=='E_R':rr=cap_rows(endpoints,scales['E_R'])
        elif arm in ('C','D'):rr=cap_rows(deltas,scales['C']) # P fixed; no Q read.
        else:
            u=cap_rows(joint,scales['R_joint']);rr=(u[:,:d]-u[:,d:])/math.sqrt(2)
            close(arm+'_u',targets[arm+'_u'],u.mean(0))
            close(arm+'_U',targets[arm+'_U'],sum((np.outer(x,x) for x in u),np.zeros((2*d,2*d)))/len(u))
        expected={'counts':np.append(counts,len(mix)),'m':point_m,'A':point_A,
                  'delta':rr.mean(0),'C':sum((np.outer(x,x) for x in rr),np.zeros((d,d)))/len(rr),
                  'r_rows':rr,'raw_delta_rows':deltas}
        for key,val in expected.items():close(arm+'_'+key,targets[arm+'_'+key],val)
        check('unit_relation_'+arm,max(np.linalg.norm(rr,axis=1))<=1+1e-12)
    check('old_point_path_preserved',result['old_normalized_cache_exact'] and result['old_full_P_point_output_exact'] and result['point_patient_moments_exact'])
    check('actual_image_gradient_probe',result['actual_gradient_witness']['point_output_exact'] and result['actual_gradient_witness']['point_gradient_exact'] and result['actual_gradient_witness']['raw_relation_gradient_norm']>0)
    check('frozen_model',result['source_state_unchanged'])
    check('no_disallowed_execution',result['new_synthesis_updates']==result['Q_V_pixel_access']==result['DP_releases']==0 and not result['new_guard_function_loss_integrated'])
    verification={'status':'PASS_PUBLIC_RAW_POINT_PATHS_AND_PATIENT_STATISTICS_ONLY','checks':checks,'errors':errors,
                  'source':'Independent NumPy recomputation from freshly extracted saved raw and dual P features; no producer math imports',
                  'raw_center_label_free':True,'scale_based_on_public_mixed':3,'point_projection_unchanged':True,
                  'result_sha256':sha(directory/'public_feature_paths_result.json'),
                  'feature_contract_sha256':sha(directory/'feature_path_contract.json'),
                  'GPU_forward':0,'new_pixel_access':0,'new_learner_or_function_loss':False,'full_W1_passed':False,
                  'patient_utility_evaluated':False}
    dest=directory/'independent_feature_statistics_verification.json'
    with dest.open('x',encoding='utf-8') as f:json.dump(verification,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({'status':verification['status'],'checks':len(checks),
                      'point_z_error':errors['independent_point_z'],'raw_a_error':errors['independent_raw_affine_a'],
                      'max_statistic_error':max(v for k,v in errors.items() if not k.startswith('independent_'))}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--directory',type=Path,required=True)
    run(parser.parse_args().directory)
