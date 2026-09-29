import csv,datetime,hashlib,json
from pathlib import Path
import numpy as np
out=Path(__file__).resolve().parent;rr=out.parent;tr=rr.parent.parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def packet(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def check(x,s):
 if not x:raise RuntimeError(s)
def prepend(p,s):
 b=p.read_bytes();i=b.index(b'\n')+1;p.write_bytes(b[:i]+('\n'+s+'\n\n').encode('utf-8')+b[i:])
def ci(x):return '['+','.join(f'{v:+.6f}' for v in x)+']'
c=read(out/'contract.json');r=read(out/'results.json');f=read(out/'construction_results.json')
for p,h in c['implementation'].items():check(sha(p)==h,'Implementation changed')
for x in c['inputs'].values():check(sha(x['path'])==x['sha256'],'Input changed')
for p,h in c['historical_result_hashes'].items():check(sha(p)==h,'Historical result changed')
check(sha(out/'EXECUTION_CONTRACT.md')==c['human_contract_sha256'],'Human contract changed')
for n in ['label_seal.json','completion.json']:
 s=read(out/n);check(s['contract_sha256']==sha(out/'contract.json'),'Contract seal')
 for p,h in s['files'].items():check(sha(out/p)==h,'Output seal')
labels=packet(out/'soft_labels.npz');check(len(labels)==2,'Expected two new labels')
for n,l in labels.items():
 check(l.shape==(128,) and np.isfinite(l).all() and abs(l).max()<=1,'Label validity')
 with (out/'label_packets'/f'{n}.csv').open(encoding='utf-8',newline='') as z:rows=list(csv.DictReader(z))
 check(np.array_equal(l,[float(x['regression_target']) for x in rows]),'CSV roundtrip')
b=packet(out/'paired_bootstrap.npz');d=packet(out/'contrast_bootstrap.npz');p=packet(out/'predictions_private.npz')
oldb=packet(c['inputs']['comparison_bootstrap']['path']);oldp=packet(c['inputs']['comparison_predictions']['path'])
bn={str(n):i for i,n in enumerate(b['names'])};dn={str(n):i for i,n in enumerate(d['names'])};pn={str(n):i for i,n in enumerate(p['names'])}
for i,n in enumerate(oldb['names']):check(np.array_equal(b['metrics'][:,bn[str(n)]],oldb['metrics'][:,i]),'Old bootstrap changed')
for i,n in enumerate(oldp['names']):check(np.array_equal(p['scores'][:,pn[str(n)]],oldp['scores'][:,i]),'Old predictions changed')
maxerr=0.
for rec in ['DenseNet','ResNet18']:
 for method in ['MT_RN_ONLY','MT_JOINT','KD_RN_ONLY','KD_JOINT']:
  dif=[];pt=[]
  for draw in ['DP1','DP2']:
   a=f'{draw}_{method}_{rec}';z=f'{draw}_DINO_ONLY_{rec}';key=f'{draw}_{method}_minus_DINO_ONLY_{rec}'
   dd=b['metrics'][:,bn[a]]-b['metrics'][:,bn[z]];dif.append(dd)
   pp=np.array([r['metrics'][a][m]['point']-r['metrics'][z][m]['point'] for m in ['AUROC','AP']]);pt.append(pp)
   check(np.array_equal(dd,d['differences'][:,dn[key]]),'Paired difference')
   for j,m in enumerate(['AUROC','AP']):
    x=r['per_release_contrasts'][key][m];maxerr=max(maxerr,abs(float(pp[j])-x['delta']),float(abs(np.quantile(dd[:,j],[.025,.975])-x['patient_cluster_95']).max()))
  key=f'MEAN_{method}_minus_DINO_ONLY_{rec}';md=(dif[0]+dif[1])/2;mp=(pt[0]+pt[1])/2
  check(np.array_equal(md,d['differences'][:,dn[key]]),'Same-row release average')
  for j,m in enumerate(['AUROC','AP']):
   x=r['fixed_two_release_mean_contrasts'][key][m];maxerr=max(maxerr,abs(float(mp[j])-x['mean_delta']),float(abs(np.quantile(md[:,j],[.025,.975])-x['fixed_two_release_patient_cluster_95']).max()))
check(maxerr<1e-15,'Saved contrast mismatch')
ts=datetime.datetime.fromisoformat(read(out/'label_seal.json')['utc'])
for ln in (out/'access_log.jsonl').read_text(encoding='utf-8').splitlines():
 a=json.loads(ln);check(a['phase'] in c['inputs'][a['key']]['phases'],'Forbidden input phase')
 if a['phase']=='evaluate':check(datetime.datetime.fromisoformat(a['utc'])>=ts,'Evaluation before labels')
tm=datetime.datetime.now(datetime.timezone.utc);mins=(tm-datetime.datetime.fromisoformat(c['started_utc'])).total_seconds()/60
size=sum(x.stat().st_size for x in out.rglob('*') if x.is_file());check(size<50*1024**2,'Storage limit')
v={'status':'PASS','inputs_unchanged':len(c['inputs']),'implementation_unchanged':len(c['implementation']),
   'historical_reports_unchanged':len(c['historical_result_hashes']),'reused_predictions_exact':16,'reused_bootstrap_exact':16,
   'contrast_max_abs':maxerr,'evaluation_after_label_seal':True,'new_label_KKT_max_abs':max(x['independent_projected_gradient_max_abs'] for x in f['fits'].values()),
   **r['verification'],'elapsed_minutes':mins,'output_MiB_before_report':size/1024**2}
dump(out/'verification.json',v)
lines=['# 1-5 결과: 공개128에서 DINO-only 라벨 대조','',
 '**판정: 혼합. 공개 carrier에서 MT의 DINO-only 대비 일관된 추가 가치는 아직 확인하지 못했다.**',
 '1-5는 실행·검산·평가를 완료했다. 상위1의 현재 범위는 완료이며, 원안의 보류된 합성4칸/nullspace를 완료로 표시하지 않는다. 연구 전체 단계2를 유지한다.','',
 '## 무엇을 실행했나','',
 '고정 공개128과 기존 DP1/DP2 목표를 재사용해 DINO-only zero-anchor 라벨2개를 계산했다. 기존 MT RN-only/Joint와 KD RN-only/Joint의16개 readout·bootstrap은 그대로 두고, 새 DINO-only의 DenseNet/RN readout4개만 추가했다. 동일 ridge0.1·bounds[-1,1]·목표RMS/eta 규칙·P class별 환자 균등 가중·BVLS·평가를 사용했다. 두 라벨을 봉인한 뒤 V를 평가했다.','',
 '## DenseNet 개발 결과','',
 '| 방법 | DP1 AUROC | DP2 AUROC | DP1 AP | DP2 AP |','|---|---:|---:|---:|---:|']
for method in ['DINO_ONLY','MT_RN_ONLY','MT_JOINT','KD_RN_ONLY','KD_JOINT']:
 a=r['metrics'][f'DP1_{method}_DenseNet'];b0=r['metrics'][f'DP2_{method}_DenseNet']
 lines.append(f"| {method} | {a['AUROC']['point']:.6f} | {b0['AUROC']['point']:.6f} | {a['AP']['point']:.6f} | {b0['AP']['point']:.6f} |")
lines+=['','| 주 비교 | DP 요약 | AUROC 차이 | 환자95%CI | AP 차이 | 환자95%CI |','|---|---|---:|---|---:|---|']
for method in ['MT_RN_ONLY','MT_JOINT']:
 for draw in ['DP1','DP2']:
  x=r['per_release_contrasts'][f'{draw}_{method}_minus_DINO_ONLY_DenseNet'];a=x['AUROC'];b0=x['AP']
  lines.append(f"| {method}−DINO-only | {draw} | {a['delta']:+.6f} | {ci(a['patient_cluster_95'])} | {b0['delta']:+.6f} | {ci(b0['patient_cluster_95'])} |")
lines+=['','| 두 고정 요약의 평균 차이 | AUROC | 조건부95%CI | AP | 조건부95%CI |','|---|---:|---|---:|---|']
for method in ['MT_RN_ONLY','MT_JOINT','KD_RN_ONLY','KD_JOINT']:
 x=r['fixed_two_release_mean_contrasts'][f'MEAN_{method}_minus_DINO_ONLY_DenseNet'];a=x['AUROC'];b0=x['AP']
 lines.append(f"| {method}−DINO-only | {a['mean_delta']:+.6f} | {ci(a['fixed_two_release_patient_cluster_95'])} | {b0['mean_delta']:+.6f} | {ci(b0['fixed_two_release_patient_cluster_95'])} |")
lines+=['',
 'MT RN-only의 평균 AUROC 차이 +0.020693과 구간 하한 +0.000008883은 양수지만 하한이 거의0이다. DP2 점차이는 음수이고 AP 평균 구간은0 포함이다. 따라서 평균 구간 하나만으로 사전 기준을 통과한 것으로 바꾸지 않는다.',
 'MT Joint는 DP1에서 유리하지만 DP2에서 AUROC가 −0.021647이며 해당 구간도 음수다. 평균 AUROC 차이는 +0.003851, 구간은0 포함이다. 두 MT 구성 모두 사전의 ‘두 요약 AUROC 점차이 양수’ 기준을 충족하지 않았다.',
 '정의한 KD는 DINO-only보다 평균 AUROC/AP가 낮았고 그 조건부 구간은 음수다. 1-4에서 MT가 KD보다 좋았다는 사실은 유지되지만, 그 사실만으로 목표 전달 자체의 필요성을 설명할 수 없다. KD를 거치며 생긴 손실을 피한 것인지, 추가 정보를 유용하게 보존한 것인지는 별도 연산 분석이 필요하다.',
 '', '## 해석 경계','',
 '지표는 영상 AUROC/AP이며 2,000회 같은 환자-cluster를 재표집했다. 평균은 지표 차이 평균이고 앙상블이 아니다. 현재 공개128·라벨·DP 요약에 조건부이며 adaptive V 선택이나 noise 모집단 불확실성을 포함하지 않는다. 동등성·DINO의 일반 우위·MT의 안정성은 입증하지 않았다.',
 'ResNet18 참고 결과와 모든 대비는 results.json에 보존했다. RN은 목표 구성에 사용되어 독립 전이 증거로 추가 계산하지 않는다. 새 Q/noise/release/pixel/encoder forward/Expert/Reserved는0이다.',
 '', '## 검산·실제 비용','',
 f"- 예상15~30분. 기록 시작부터 검산·보고 생성까지 실측 **{mins:.2f}분**.",
 f"- 라벨2개 구성·검산 {f['seconds']:.3f}초, 새 평가·bootstrap {r['seconds']:.3f}초.",
 f"- 새 label KKT 최대 {v['new_label_KKT_max_abs']:.3e}; readout primal/dual {v['ridge_primal_dual_max_abs']:.3e}; 독립 bootstrap witness {v['bootstrap_sklearn_max_abs']:.3e}.",
 '- 기존16개 예측·bootstrap 정확히 일치. 입력14개·구현21개·과거 보고서5개 해시 보존. 저장 대비/평균/CI 재구성 오차0. 라벨 CSV·bounds·유한성·평가 전 봉인 통과.',
 f'- 보고 전 추가 저장 {size/1024**2:.2f}MiB, 이미지·모델 복제 없음. 실패·재시도·설정 변경 없음.',
 '', '## 상위1 종료 판단과 다음 번호','',
 '1-1/1-4의 MT−KD 양성 결과는 유효하다. 1-2/1-5에서 강한 단독 대안 대비 추가 필요성은 일관되지 않았고,1-3에서 합성 제작 추가 가치는 미확인이다. 상위1은 현재 정한 질문에 답한 것으로 완료한다. 이는 현재 복합 구성을 CVPR 기여로 승인했다는 뜻이 아니다.',
 '**다음은2-1: 연산·선행 비교.** 같은 보호 평균·공개자료에 접근하는 teacher-score KD, source-only label solve, MT/signed-public-weight 구현을 수식·코드·원문으로 대응한다. 비교표 하나와2-2에서 실제로 구별할 예측/필요 여부가 산출물이다. 예상1~2시간의 설계·분석이며 새 bank·요약·평가 없이 진행한다. 이미 동일한 연산이면 그것을 별도 baseline으로 만들어 다시 돌리지 않는다.',
 '', '실행 폴더: `'+out.name+'`. 모든 새 결과와원본은 로컬 보존. 사용자 지시에 따라 완료 결과와 다음2-1의 규모/예상 시간을 먼저 보고한 뒤 이어간다.','']
report=rr/'PUBLIC_TRANSFER_1_5_SOURCE_CONTROL_RESULTS_20260928.md';check(not report.exists(),'Existing report')
report.write_text('\n'.join(lines),encoding='utf-8')
block='**1-5 COMPLETE / UPPER1 SCOPED CONTROLS CLOSED — 2026-09-28**\n\n1-5 혼합: 공개128 DINO-only DenseNet AUROC DP1 0.648920/DP2 0.695071. MT RN-only−DINO 평균 +0.020693, CI[+0.000009,+0.042601]이나 DP2 점차이 음수. MT Joint 평균 +0.003851, CI0 포함; DP2는 유의하게 낮음. 두 MT 모두 사전 반복 투자 기준 미충족. KD 대비 기존 양성은 보존. 새 라벨2·평가4, 기존16재사용, Q/release/pixel/forward/Expert/Reserved0. 상위1 현재범위 완료; 원안 보류항목은 미완료 유지. 다음2-1 연산·근접연구 대조. 완료 및 다음설계/ETA 보고 후 이어간다. Stage2 유지.'
for path in [tr/'AGENTS.md',tr/'CURRENT_STATUS.md',tr/'WORKLOG.md',rr/'RESEARCH_FRAMEWORK.md']:
 link=report.name if path.parent==rr else 'CVPR 주제 탐색/research_2026-09-10/'+report.name
 prepend(path,block+'\n\n[Results](<'+link+'>)')
plan=rr/'PUBLIC_TRANSFER_NUMBERED_PLAN_20260928.md'
prepend(plan,'**진행 상태 갱신 — 2026-09-28:** 1-5 완료·혼합, 상위1의 현재 범위 완료. 다음2-1은 연산·선행 비교(예상1~2시간). 아래 계획 수립 당시 미실행 표시는 역사적 상태이며 현재는 본 갱신을 따른다. '+f'[1-5 결과]({report.name})')
s=read(rr/'research_state.json');s['current_result_report']=report.name
s['step2_active_task']='1-5 completed mixed; upper1 scoped controls complete. Next2-1 operation/prior analysis.'
s['active_execution']={'status':'no_running_execution','last_completed':'1-5','report':report.name,'completed_utc':tm.isoformat(),'next_execution_started':False}
s['next_task']='Announce design/scale/ETA1-2hours, then execute2-1 operation/closest-prior comparison. No new synthesis, private release, or V efficacy run.'
s['numbered_execution_plan_20260928']['current_item']='2-1 pending'
s['numbered_execution_plan_20260928']['upper_1']['status']='scoped_controls_complete_mixed'
s['numbered_execution_plan_20260928']['upper_1']['completed'].append('1-5 public128 DINO-only two releases: mixed; MT necessity not consistently supported')
s['numbered_execution_plan_20260928']['upper_1']['pending']=[]
s['numbered_execution_plan_20260928']['report_sha256']=sha(plan)
s['public_transfer_1_5_20260928']={'status':'complete_mixed','report':report.name,'report_sha256':sha(report),'out_dir':str(out),'decisions':r['development_decisions'],'means':r['fixed_two_release_mean_contrasts'],'verification':v,'new_releases':0}
dump(rr/'research_state.json',s)
dump(out/'finalization.json',{'status':'COMPLETE','item':'1-5','report':str(report),'report_sha256':sha(report),'verification':v,'next_item':'2-1','finalizer_sha256':sha(__file__)})
print(json.dumps({'status':'1-5_RECORDED','elapsed_minutes':mins,'verification':v,'next_item':'2-1'},ensure_ascii=True,indent=2))
