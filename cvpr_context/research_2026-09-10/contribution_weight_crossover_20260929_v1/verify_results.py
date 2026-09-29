"""Independent artifact/contrast reconstruction after the one permitted evaluation."""
import csv,datetime,hashlib,json
from pathlib import Path
import numpy as np
out=Path(__file__).resolve().parent;rr=out.parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def nz(p):
 with np.load(p,allow_pickle=False) as f:return {k:f[k] for k in f.files}
c=read(out/'contract.json');r=read(out/'results.json');f=read(out/'construction_results.json')
for v in c['inputs'].values():assert sha(v['path'])==v['sha256']
for p,h in c['implementation'].items():assert sha(p)==h
for p,h in c['historical_result_hashes'].items():assert sha(p)==h
assert sha(out/'EXECUTION_CONTRACT.md')==c['human_contract_sha256']
assert sha(rr/c['design_report'])==c['design_report_sha256']
assert sha(Path(c['design_dir'])/'NEXT_2_2R2_DESIGN.json')==c['design_json_sha256']
for name in ['label_seal.json','completion.json']:
 seal=read(out/name);assert seal['contract_sha256']==sha(out/'contract.json')
 for filename,h in seal['files'].items():assert sha(out/filename)==h
labels=nz(out/'soft_labels.npz');assert len(labels)==4
for name,l in labels.items():
 assert l.shape==(128,) and np.isfinite(l).all() and np.max(abs(l))<=1
 with (out/'label_packets'/f'{name}.csv').open(encoding='utf-8',newline='') as ff:rows=list(csv.DictReader(ff))
 assert np.array_equal(l,[float(x['regression_target']) for x in rows])
p=nz(out/'predictions_private.npz');b=nz(out/'paired_bootstrap.npz');d=nz(out/'contrast_bootstrap.npz')
op=nz(c['inputs']['comparison_predictions']['path']);ob=nz(c['inputs']['comparison_bootstrap']['path'])
pn={str(n):i for i,n in enumerate(p['names'])};bn={str(n):i for i,n in enumerate(b['names'])};dn={str(n):i for i,n in enumerate(d['names'])}
on={str(n):i for i,n in enumerate(op['names'])};obn={str(n):i for i,n in enumerate(ob['names'])}
wanted=[draw+'_'+method+'_'+rec for draw in ['DP1','DP2'] for method in ['HR_RN_ONLY','ACKD_RN_ONLY','DINO_ONLY'] for rec in ['DenseNet','ResNet18']]
assert len(pn)==20 and len(wanted)==12
for n in wanted:
 assert np.array_equal(p['scores'][:,pn[n]],op['scores'][:,on[n]])
 assert np.array_equal(b['metrics'][:,bn[n]],ob['metrics'][:,obn[n]])
error=0.
for rec in ['DenseNet','ResNet18']:
 for stem,terms in r['contrast_definitions'].items():
  arrays=[];points=[]
  for draw in ['DP1','DP2']:
   key=draw+'_'+stem+'_'+rec
   arr=sum(coef*b['metrics'][:,bn[draw+'_'+name+'_'+rec]] for name,coef in terms.items())
   pt=sum(coef*np.array([r['metrics'][draw+'_'+name+'_'+rec][m]['point'] for m in ['AUROC','AP']]) for name,coef in terms.items())
   assert np.array_equal(arr,d['differences'][:,dn[key]])
   for k,m in enumerate(['AUROC','AP']):
    v=r['per_release_contrasts'][key][m]
    error=max(error,abs(pt[k]-v['delta']),float(abs(np.quantile(arr[:,k],[.025,.975])-v['patient_cluster_95']).max()))
   arrays.append(arr);points.append(pt)
  key='MEAN_'+stem+'_'+rec;arr=(arrays[0]+arrays[1])/2;pt=(points[0]+points[1])/2
  assert np.array_equal(arr,d['differences'][:,dn[key]])
  for k,m in enumerate(['AUROC','AP']):
   v=r['fixed_two_release_mean_contrasts'][key][m]
   error=max(error,abs(pt[k]-v['mean_delta']),float(abs(np.quantile(arr[:,k],[.025,.975])-v['fixed_two_release_patient_cluster_95']).max()))
assert error<1e-15
sealed=datetime.datetime.fromisoformat(read(out/'label_seal.json')['utc'])
for line in (out/'access_log.jsonl').read_text(encoding='utf-8').splitlines():
 row=json.loads(line);assert row['phase'] in c['inputs'][row['key']]['phases']
 if row['phase']=='evaluate':assert datetime.datetime.fromisoformat(row['utc'])>=sealed
 else:assert row['key'] not in ['V_DenseNet','V_ResNet18','comparison_predictions','comparison_bootstrap','patient_bootstrap']
expected=nz(c['inputs']['bridge_targets']['path']);targets=nz(out/'crossover_target_weights.npz')
for draw in ['DP1','DP2']:
 assert np.array_equal(targets[draw+'_HR_B_RN_ONLY'],expected['HR_B_'+draw])
 assert np.array_equal(targets[draw+'_ACKD_U_RN_ONLY'],expected['ACKD_U_'+draw])
primary=[r['fixed_two_release_mean_contrasts']['MEAN_'+x+'_DenseNet'] for x in ['HR_U_minus_HR_B','ACKD_U_minus_ACKD_B']]
assert r['decisions']['shared_weight_separation_hypothesis_passes']==all(x['passes_development_hurdle'] for x in primary)
size=sum(x.stat().st_size for x in out.rglob('*') if x.is_file())
assert size<50*1024**2
checks={'status':'PASS','inputs_unchanged':len(c['inputs']),'implementation_unchanged':len(c['implementation']),
 'historical_reports_unchanged':len(c['historical_result_hashes']),'reused_predictions_exact':12,'reused_bootstraps_exact':12,
 'new_targets_exact_to_design':4,'new_labels':4,'new_readouts':8,'all_contrast_reconstruction_max_abs':error,
 'evaluation_after_label_seal':True,'label_KKT_max_abs':max(x['independent_projected_gradient_max_abs'] for x in f['fits'].values()),
 **r['verification'],'output_MiB':size/1024**2}
(out/'verification.json').write_text(json.dumps(checks,indent=2)+'\n',encoding='utf-8')
print(json.dumps(checks,indent=2))
