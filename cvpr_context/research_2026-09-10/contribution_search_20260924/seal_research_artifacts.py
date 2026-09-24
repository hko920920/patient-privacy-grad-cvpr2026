"""Seal local label-only packets, validate independent ridge solves and summarize."""
import argparse,csv,hashlib,json,platform,sys,time
from pathlib import Path
import numpy as np
import scipy
from sklearn.metrics import roc_auc_score,average_precision_score
from geometry_diagnostic import packet
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def run(root,out):
    r=root/'_reports';labels=packet(out/'joint_compiler_soft_labels.npz')
    pred=packet(out/'joint_compiler_predictions_private.npz');boot=packet(out/'joint_compiler_bootstrap.npz')
    sourceboot=packet(r/'downstream_development_20260917_v1/cluster_bootstrap.npz')
    result=json.loads((out/'joint_compiler_results.json').read_text(encoding='utf-8'))
    vfiles={'DenseNet':'prrd_pilot_abcd101_20260921_v1/evaluation/recipient_V_features.npz',
            'ResNet18':'receiver_resnet18_reuse_20260924_v1/V_features.npz'}
    y=pred['labels'];pids=pred['patient_ids'];_,inverse=np.unique(pids,return_inverse=True)
    readout_error=0.;bootstrap_error=0.;manifest={'status':'EXPLORATORY_LOCAL_PACKET_READY',
      'not_a_confirmatory_result':True,'construction_models':['DINOv2','ResNet18'],'DenseNet_in_label_objective':False,
      'new_Q_access':0,'new_DP_release':0,'pixel_updates':0,'Expert_Reserved':False,
      'artifact':'Existing DP PNG images plus a new 128-vector of bounded soft labels; no pixel changes.',
      'privacy_scope':'For the fixed public maps/algorithm, each packet is post-processing of its original feature-DP release. This does not retroactively privatize algorithm selection, V, or prior non-DP development. No reduction of the existing joint transcript bound is claimed.',
      'readout':'Uniform 1/128 sample weights; frozen receiver features; solve (Z.T Z/128+0.1 I)w=Z.T regression_target/128; no intercept; do not hard-threshold labels or reweight by their argmax.',
      'software':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__},
      'packets':{},'file_hashes':{}}
    art=out/'label_packets';art.mkdir(exist_ok=True)
    for draw,folder,targetfolder in [
        ('DP1','receiver_feature_dp8_20260924_v1/protected_png','receiver_feature_dp8_20260924_v1/one_release/protected_target'),
        ('DP2','receiver_feature_followup_20260924_v1/protected_png','receiver_feature_followup_20260924_v1/one_release/protected_target')]:
        bank=r/folder;l=labels[draw];assert l.shape==(128,) and np.isfinite(l).all() and abs(l).max()<=1.
        with (bank/'images.csv').open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
        sealed=json.loads((bank/'release_manifest.json').read_text(encoding='utf-8'))
        csvpath=art/(draw+'_soft_labels.csv')
        with csvpath.open('w',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['file','original_hard_label','regression_target','soft_class1_target','png_sha256'])
            writer.writeheader()
            for row,v in zip(rows,l):
                p=(bank/row['file']).resolve();h=sha(p)
                assert p.is_relative_to(bank.resolve())
                assert h==sealed['files_sha256'][row['file']]
                writer.writerow({'file':row['file'],'original_hard_label':row['label'],
                                 'regression_target':format(float(v),'.17g'),
                                 'soft_class1_target':format(float((v+1)/2),'.17g'),'png_sha256':h})
        with csvpath.open(encoding='utf-8',newline='') as f:
            roundtrip=np.array([float(row['regression_target']) for row in csv.DictReader(f)])
        assert np.array_equal(l,roundtrip)
        for rec,vfile in vfiles.items():
            path=r/'receiver_feature_dp8_20260924_v1/evaluation'/(rec+'_PNG_features.npz') if draw=='DP1' else r/'receiver_feature_followup_20260924_v1/evaluation'/(rec+'_FEATURE_DP2_PNG_features.npz')
            z=packet(path)['z'].astype(np.float64);v=packet(r/vfile)
            # Dual 128x128 kernel solve, independent of primal implementation.
            w=z.T@np.linalg.solve(z@z.T+12.8*np.eye(128),roundtrip)
            s=v['z'].astype(np.float64)@w
            n=draw+'_JOINT_eval_'+rec;j=list(pred['names']).index(n)
            readout_error=max(readout_error,float(abs(s-pred['scores'][:,j]).max()))
            assert abs(roc_auc_score(y,s)-result['metrics'][n]['AUROC'])<1e-12
            for k in (0,17,1999):
                sw=sourceboot['patient_counts'][k,inverse]
                ref=np.array([roc_auc_score(y,s,sample_weight=sw),average_precision_score(y,s,sample_weight=sw)])
                bootstrap_error=max(bootstrap_error,float(abs(ref-boot['metrics'][k,j,:]).max()))
            manifest['file_hashes'][str(path)]=sha(path)
        mechanism=json.loads((r/targetfolder/'mechanism.json').read_text(encoding='utf-8'))
        manifest['packets'][draw]={'images_directory':str(bank),'soft_labels_csv':str(csvpath),'soft_labels_sha256':sha(csvpath),
            'target_sha256':sha(r/targetfolder/'target.npz'),'target_mechanism':mechanism,
            'original_pixel_hashes_verified':128,'labels_changed_sign':int(np.sum((l>0)!=(np.arange(128)>=64)))}
    assert readout_error<1e-12 and bootstrap_error<1e-12
    manifest['verification']={'primal_dual_score_max_abs_error':readout_error,'independent_bootstrap_max_abs_error':bootstrap_error,
                              'CSV_roundtrip_exact':True,'PNG_hashes_verified':256}
    for p in out.iterdir():
        if p.suffix in ('.py','.md','.json') and p.name!='research_artifact_manifest.json':
            manifest['file_hashes'][p.name]=sha(p)
    (out/'research_artifact_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'verification':manifest['verification'],'packets':{k:{a:b for a,b in v.items() if a!='target_mechanism'} for k,v in manifest['packets'].items()}},ensure_ascii=False,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();run(a.root,a.out)

