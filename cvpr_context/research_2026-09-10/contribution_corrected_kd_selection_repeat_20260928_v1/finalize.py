import csv,datetime,hashlib,json
from pathlib import Path
import numpy as np
out=Path(__file__).resolve().parent;rr=out.parent;tr=rr.parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def nz(p):
 with np.load(p,allow_pickle=False) as f:return {k:f[k] for k in f.files}
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def pre(p,s):
 t=p.read_text(encoding='utf-8');h,_,r=t.partition('\n');p.write_text(h+'\n\n'+s+'\n\n'+r,encoding='utf-8')
def ci(x):return '['+', '.join(format(v,'+.6f') for v in x)+']'
assert not (out/'finalization.json').exists()
c=read(out/'contract.json');r=read(out/'results.json');f=read(out/'construction_results.json')
for v in c['inputs'].values():assert sha(v['path'])==v['sha256']
for p,h in c['implementation'].items():assert sha(p)==h
for p,h in c['historical_result_hashes'].items():assert sha(p)==h
assert sha(out/'EXECUTION_CONTRACT.md')==c['human_contract_sha256']
for n in ['selection_seal.json','label_seal.json','completion.json']:
 s=read(out/n);assert s['contract_sha256']==sha(out/'contract.json')
 for p,h in s['files'].items():assert sha(out/p)==h
labs=nz(out/'soft_labels.npz');assert len(labs)==10
for n,x in labs.items():
 assert x.shape==(128,) and np.isfinite(x).all() and abs(x).max()<=1
 with (out/'label_packets'/f'{n}.csv').open(encoding='utf-8',newline='') as ff:rows=list(csv.DictReader(ff))
 assert np.array_equal(x,[float(z['regression_target']) for z in rows])
used=nz(out/'used_target_weights.npz');hr=nz(c['inputs']['HR_targets']['path']);corr=nz(c['inputs']['correction_targets']['path']);src=nz(c['inputs']['prior_target_weights']['path'])
for name,x in used.items():
 draw=name.split('_')[0]
 y=src[draw+'_DINO'] if name.endswith('DINO_ONLY') else hr[name] if name.endswith('HR_RN_ONLY') else corr[name]
 assert np.array_equal(x,y)
b=nz(out/'paired_bootstrap.npz');d=nz(out/'contrast_bootstrap.npz');p=nz(out/'predictions_private.npz')
ob=nz(c['inputs']['comparison_bootstrap']['path']);op=nz(c['inputs']['comparison_predictions']['path'])
bn={str(n):i for i,n in enumerate(b['names'])};dn={str(n):i for i,n in enumerate(d['names'])};pn={str(n):i for i,n in enumerate(p['names'])}
assert len(pn)==56
for i,n in enumerate(ob['names']):assert np.array_equal(b['metrics'][:,bn[str(n)]],ob['metrics'][:,i])
for i,n in enumerate(op['names']):assert np.array_equal(p['scores'][:,pn[str(n)]],op['scores'][:,i])
err=0.
for rec in ['DenseNet','ResNet18']:
 for control in ['ACKD_RN_ONLY','AKD_RN_ONLY','CKD_RN_ONLY','DINO_ONLY']:
  ds=[];pts=[]
  for draw in ['DP1','DP2']:
   aa='S202_'+draw+'_HR_RN_ONLY_'+rec;bb='S202_'+draw+'_'+control+'_'+rec;key='S202_'+draw+'_HR_RN_ONLY_minus_'+control+'_'+rec
   dd=b['metrics'][:,bn[aa]]-b['metrics'][:,bn[bb]];pt=np.array([r['metrics'][aa][m]['point']-r['metrics'][bb][m]['point'] for m in ['AUROC','AP']])
   ds.append(dd);pts.append(pt);assert np.array_equal(dd,d['differences'][:,dn[key]])
   for k,m in enumerate(['AUROC','AP']):
    z=r['per_release_contrasts'][key][m];err=max(err,abs(pt[k]-z['delta']),float(abs(np.quantile(dd[:,k],[.025,.975])-z['patient_cluster_95']).max()))
  key='MEAN_S202_HR_RN_ONLY_minus_'+control+'_'+rec;dd=(ds[0]+ds[1])/2;pt=(pts[0]+pts[1])/2
  assert np.array_equal(dd,d['differences'][:,dn[key]])
  for k,m in enumerate(['AUROC','AP']):
   z=r['fixed_two_release_mean_contrasts'][key][m];err=max(err,abs(pt[k]-z['mean_delta']),float(abs(np.quantile(dd[:,k],[.025,.975])-z['fixed_two_release_patient_cluster_95']).max()))
assert err<1e-15
select_time=datetime.datetime.fromisoformat(read(out/'selection_seal.json')['utc']);label_time=datetime.datetime.fromisoformat(read(out/'label_seal.json')['utc'])
for line in (out/'access_log.jsonl').read_text(encoding='utf-8').splitlines():
 q=json.loads(line);assert q['phase'] in c['inputs'][q['key']]['phases']
 if q['phase']=='construct':assert datetime.datetime.fromisoformat(q['utc'])>=select_time
 if q['phase']=='evaluate':assert datetime.datetime.fromisoformat(q['utc'])>=label_time
tm=datetime.datetime.now(datetime.timezone.utc);mins=(tm-datetime.datetime.fromisoformat(c['started_utc'])).total_seconds()/60
size=sum(p.stat().st_size for p in out.rglob('*') if p.is_file());assert size<50*1024**2
v={'status':'PASS','inputs_unchanged':len(c['inputs']),'implementation_unchanged':len(c['implementation']),'historical_reports_unchanged':len(c['historical_result_hashes']),
 'old_predictions_exact':36,'old_bootstraps_exact':36,'frozen_targets_exact':10,'contrast_max_abs':err,'evaluation_after_label_seal':True,'construction_after_selection_seal':True,
 'elapsed_minutes':mins,'label_KKT_max_abs':max(x['independent_projected_gradient_max_abs'] for x in f['fits'].values()),**r['verification'],'output_MiB_before_report':size/1024**2}
dump(out/'verification.json',v)
lines=['# 2-3 보완(2-3R1): 강화 KD 대조의 공개128 선정 반복','',
 '**결론: 주비교 HR−두 보정 결합 KD(ACKD)의 개선은 새 공개128에서도 반복됐다. 공분산 보정만 한 CKD 대비도 반복됐다. 반면 공개 기준값 보존만 한 AKD 대비 평균 구간은0을 포함하므로, 세 강화 대안 모두에 대한 우위가 반복됐다는 전체 기준은 통과하지 못했다.**','',
 '이번 선정에서는 HR−DINO-only도 두release 양수이고 개발 hurdle을 통과했다. 다만 기존 선정의DP2 음수 결과를 지우거나, 두 선정 전체에서 source-only 우위가 완전히 확인됐다고 해석하지 않는다.','',
 '## 실행 범위와 반복의 의미','',
 '이전과 동일한 공개patient별1후보→양성6포함→DINO/RN equal-RMS farthest-first128 알고리즘에서 salt만 public-carrier-20260928-v1-repeat202로 고정했다. 첫선정과 환자97/128, 영상95/128이 겹친다. 새로운P/Q/V나새DP noise가아니다.',
 '기존HR/AKD/CKD/ACKD/DINO목표를bitwise그대로사용하고label10개를새carrier에계산했다. 모든label봉인후DenseNet/RN20readouts,기존36결과보존. 선정을결과에맞춰다시하지않았다.','',
 '## DenseNet 개발 결과','',
 '| 방법 | DP1 AUROC | DP2 AUROC | DP1 AP | DP2 AP |','|---|---:|---:|---:|---:|']
for m in ['DINO_ONLY','AKD_RN_ONLY','CKD_RN_ONLY','ACKD_RN_ONLY','HR_RN_ONLY']:
 a=r['metrics']['S202_DP1_'+m+'_DenseNet'];z=r['metrics']['S202_DP2_'+m+'_DenseNet']
 lines.append(f"| {m} | {a['AUROC']['point']:.6f} | {z['AUROC']['point']:.6f} | {a['AP']['point']:.6f} | {z['AP']['point']:.6f} |")
lines+=['','| HR 대비대상 | 두release 평균AUROC차이 | 조건부95%CI | 평균AP차이 | 조건부95%CI | 개발hurdle |','|---|---:|---|---:|---|---|']
for control in ['ACKD_RN_ONLY','AKD_RN_ONLY','CKD_RN_ONLY','DINO_ONLY']:
 x=r['fixed_two_release_mean_contrasts']['MEAN_S202_HR_RN_ONLY_minus_'+control+'_DenseNet'];a=x['AUROC'];z=x['AP']
 lines.append(f"| {control} | {a['mean_delta']:+.6f} | {ci(a['fixed_two_release_patient_cluster_95'])} | {z['mean_delta']:+.6f} | {ci(z['fixed_two_release_patient_cluster_95'])} | {x['passes_development_hurdle']} |")
lines+=['','## 해석','',
 '- **주비교는 긍정 재현이다.** HR−ACKD AUROC 평균+0.033434,CI[+0.016395,+0.051698],AP+0.008374,CI[+0.002700,+0.014679]. 첫선정의AUROC+0.035346과같은방향이다. 점수가좋은반복만골라낸결과가아니다.',
 '- **CKD 대비 추가효용도 남았다.** AUROC+0.021664,CI[+0.008625,+0.035256],AP+0.006011,CI[+0.001675,+0.010631]. 공분산보정이단순KD를개선했더라도현재HR의모든추가효용을대체하지는않았다.',
 '- **AKD 대비는 미확정이다.** AUROC점차이는DP1/DP2모두양수이고평균+0.030019이지만CI[−0.010837,+0.067796]이다. 따라서사전all-controls반복gate는False다. 다른두대조가긍정이라는이유로이를통과처리하지않는다.',
 '- **DINO-only 대비는 이번 선정에서 긍정적이다.** 평균AUROC+0.025617,CI[+0.003314,+0.048818]; AP구간은0포함이다. 첫선정HR−DINO평균+0.022776/양의구간과합치면전달방향을계속설계할근거는있다. 그러나첫선정DP2에서는−0.002719였고동일DP두개를재사용했으므로독립4회성공으로세지않는다.',
 '- **현재 근거의 정확한 범위:** 같은보호head·공개자료에서특정public-transfer연산이보정KD대안보다나은개발효용을줄수있다는근거가강해졌다. patient/class평균·clipping·중심화·가중중어느것이핵심인지와조건부평균전달선행대비새로운지식은아직닫히지않았다.','',
 '영상AUROC/AP,동일2000환자cluster paired bootstrap이다. 구간은현재bank/두DP실현에조건부. DenseNet은재사용개발수신자,RN은구성모델이다. 모든개별contrast/RN결과는results.json에보존했다.','',
 '## 검산과 실측','',
 f'- 실행 전 예상20~35분, 계약 작성부터 검산·기록까지 **{mins:.2f}분**.',
 f'- 공개선정 {read(out/"public128_manifest.json")["seconds"]:.3f}초,10labels {f["seconds"]:.3f}초,20readouts/bootstrap {r["seconds"]:.3f}초.',
 f'- labelKKT {v["label_KKT_max_abs"]:.3e}; ridge primal/dual {v["ridge_primal_dual_max_abs"]:.3e}; bootstrap witness {v["bootstrap_sklearn_max_abs"]:.3e}.',
 f'- 입력{v["inputs_unchanged"]}/구현{v["implementation_unchanged"]}/기존보고서{v["historical_reports_unchanged"]} hash불변. 기존36예측/bootstrap,10목표bitwise일치. 모든CI재구성오차{err:.1e}.',
 f'- 추가출력 {size/1024**2:.2f}MiB. 새Q/DP release/noise/encoder forward/pixel optimization/Expert/Reserved0. 재시도나결과기반설정변경없음.','',
 '## 번호·다음 작업','',
 '| 번호 | 상태 |','|---|---|',
 '| 1-1~1-5 | 정한 기본대조범위 완료, 과거혼합결과유지 |',
 '| 2-1 / 2-2 | 기존 연산비교·head정보대조 완료 |',
 '| 2-1R1 | 공개 기준값/공분산 전달 분해·선행·보정대조설계 완료 |',
 '| 2-2R1 | 세보정KD대조 긍정통과 |',
 '| 2-3R1 | **주대조·CKD대조는반복,전체세대안우위gate는미충족** |',
 '| 원래2-3 | 기존source-only조건의보류상태보존 |',
 '| 3-1~3-3 | 아직기여/독립평가로자동진행하지않음 |','',
 '다음은 **2-1 보완(2-1R2): 전달 지도의 남은 차이를 기여 가설로 좁히는 설계**다. 공개자료의 patient/class 평균과 clipping 순서가 image-level KD와 만드는 차이를수식·공개자료·직접선행으로비교하고,어떤보존조건을위반할때실패해야하는지하나의반증가능한예측으로정리한다. 예상30~45분. 이설계에서새V효용/label/Q/DP release/추가receiver/Expert/Reserved는없다.',
 '반복을더늘려AKD구간이양수가될때까지기다리지않는다. 이번긍정결과를없애지도않고,충분한신규성이없는현재연산을새이름으로방법기여로올리지도않는다. 기전과직접선행의차이가정해져야그차이에대한2-2후속을설계할수있다.','',
 '[강화대조결과](PUBLIC_TRANSFER_2_2R1_KD_CORRECTIONS_RESULTS_20260928.md) · [연산/선행](PUBLIC_TRANSFER_2_1R1_ANCHOR_GEOMETRY_DESIGN_20260928.md)','']
report=rr/'PUBLIC_TRANSFER_2_3R1_SELECTION_REPEAT_RESULTS_20260928.md';assert not report.exists();report.write_text('\n'.join(lines),encoding='utf-8')
block='**2-3R1 COMPLETE / PRIMARY REPEATED, ALL-CONTROL GATE MIXED — 2026-09-28**\n\n새public128 HR−ACKD DenseNet평균AUROC+0.033434 CI[+0.016395,+0.051698],HR−CKD+0.021664 CI양수. HR−AKD+0.030019이나CI0포함으로전체반복gate미충족. HR−DINO이번선정+0.025617 CI양수/두draw양수,기존선정DP2음수유지. 환자97/영상95겹침,독립cohort/noise아님. 새10labels/20readouts,기존36재사용. 다음2-1R2는patient/class/clipping연산차이와선행대조설계30~45분,추가성능반복아님. Stage2유지.'
for pp in [tr/'AGENTS.md',tr/'CURRENT_STATUS.md',tr/'WORKLOG.md',rr/'RESEARCH_FRAMEWORK.md']:
 link=report.name if pp.parent==rr else 'CVPR 주제 탐색/research_2026-09-10/'+report.name
 pre(pp,block+'\n\n[Results](<'+link+'>)')
plan=rr/'PUBLIC_TRANSFER_NUMBERED_PLAN_20260928.md'
pre(plan,'**2-3R1완료 — 2026-09-28:** 주대조HR−ACKD와CKD반복긍정,AKD구간0포함으로전체gate혼합. 다음2-1R2는patient/class/clipping연산차이와선행대조설계(30~45분). 원래2-3보류·3번미착수유지. [결과]('+report.name+')')
s=read(rr/'research_state.json');s['current_result_report']=report.name;s['last_efficacy_result_report']=report.name
s['step2_active_task']='2-3R1 complete: primary and CKD gains repeat; AKD superiority unconfirmed. Next2-1R2 concrete privacy-unit/clipping transport design.'
s['active_execution']={'status':'no_running_execution','last_completed':'2-3R1','completed_utc':tm.isoformat(),'report':report.name,'next_item':'2-1R2'}
s['next_task']='2-1R2 design only: derive patient/class aggregation and clipping alignment gap versus image KD/CME; closest primary priors and one falsifiable hypothesis. ETA30-45min before starting; no new labels/V/Q/releases/Expert/Reserved.'
s['numbered_execution_plan_20260928']['current_item']='2-3R1 complete mixed overall;next2-1R2 design';s['numbered_execution_plan_20260928']['report_sha256']=sha(plan)
s['numbered_execution_plan_20260928'].setdefault('upper_2_supplements',{})['2-3R1']={'status':'primary_positive_all_controls_mixed','report':report.name,'elapsed_minutes':mins}
dump(rr/'research_state.json',s)
dump(out/'finalization.json',{'status':'COMPLETE','elapsed_minutes':mins,'report':str(report),'report_sha256':sha(report),'verification_sha256':sha(out/'verification.json')})
print(json.dumps({'item':'2-3R1','status':'RECORDED','elapsed_minutes':mins,'verification':v},ensure_ascii=True))

