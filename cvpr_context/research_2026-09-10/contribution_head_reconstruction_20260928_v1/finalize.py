import csv,datetime,hashlib,json
from pathlib import Path
import numpy as np
out=Path(__file__).resolve().parent;rr=out.parent;tr=rr.parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def nz(p):
 with np.load(p,allow_pickle=False) as f:return {k:f[k] for k in f.files}
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def prepend(p,s):
 t=p.read_text(encoding='utf-8');h,_,r=t.partition('\n');p.write_text(h+'\n\n'+s+'\n\n'+r,encoding='utf-8')
def ci(v):return '['+', '.join(f'{x:+.6f}' for x in v)+']'
assert not (out/'finalization.json').exists()
c=read(out/'contract.json');r=read(out/'results.json');f=read(out/'construction_results.json')
for v in c['inputs'].values():assert sha(v['path'])==v['sha256']
for p,h in c['implementation'].items():assert sha(p)==h
for p,h in c['historical_result_hashes'].items():assert sha(p)==h
assert sha(out/'EXECUTION_CONTRACT.md')==c['human_contract_sha256']
for n in ['label_seal.json','completion.json']:
 s=read(out/n);assert s['contract_sha256']==sha(out/'contract.json')
 for p,h in s['files'].items():assert sha(out/p)==h
l=nz(out/'soft_labels.npz');assert len(l)==2
for n,x in l.items():
 assert x.shape==(128,) and np.isfinite(x).all() and abs(x).max()<=1
 with (out/'label_packets'/f'{n}.csv').open(encoding='utf-8',newline='') as ff:rows=list(csv.DictReader(ff))
 assert np.array_equal(x,[float(z['regression_target']) for z in rows])
b=nz(out/'paired_bootstrap.npz');d=nz(out/'contrast_bootstrap.npz');p=nz(out/'predictions_private.npz')
ob=nz(c['inputs']['comparison_bootstrap']['path']);op=nz(c['inputs']['comparison_predictions']['path'])
bn={str(n):i for i,n in enumerate(b['names'])};dn={str(n):i for i,n in enumerate(d['names'])};pn={str(n):i for i,n in enumerate(p['names'])}
for i,n in enumerate(ob['names']):assert np.array_equal(b['metrics'][:,bn[str(n)]],ob['metrics'][:,i])
for i,n in enumerate(op['names']):assert np.array_equal(p['scores'][:,pn[str(n)]],op['scores'][:,i])
err=0.
for rec in ['DenseNet','ResNet18']:
 for aa,bb in [('MT_RN_ONLY','HR_RN_ONLY'),('HR_RN_ONLY','KD_RN_ONLY'),('HR_RN_ONLY','DINO_ONLY')]:
  points=[];diff=[]
  for draw in ['DP1','DP2']:
   a=draw+'_'+aa+'_'+rec;z=draw+'_'+bb+'_'+rec;key=draw+'_'+aa+'_minus_'+bb+'_'+rec
   dd=b['metrics'][:,bn[a]]-b['metrics'][:,bn[z]];diff.append(dd)
   pt=np.array([r['metrics'][a][m]['point']-r['metrics'][z][m]['point'] for m in ['AUROC','AP']]);points.append(pt)
   assert np.array_equal(dd,d['differences'][:,dn[key]])
   for k,m in enumerate(['AUROC','AP']):
    v=r['per_release_contrasts'][key][m];err=max(err,abs(float(pt[k])-v['delta']),float(abs(np.quantile(dd[:,k],[.025,.975])-v['patient_cluster_95']).max()))
  key='MEAN_'+aa+'_minus_'+bb+'_'+rec;md=(diff[0]+diff[1])/2;mp=(points[0]+points[1])/2
  assert np.array_equal(md,d['differences'][:,dn[key]])
  for k,m in enumerate(['AUROC','AP']):
   v=r['fixed_two_release_mean_contrasts'][key][m];err=max(err,abs(float(mp[k])-v['mean_delta']),float(abs(np.quantile(md[:,k],[.025,.975])-v['fixed_two_release_patient_cluster_95']).max()))
assert err<1e-15
sealed=datetime.datetime.fromisoformat(read(out/'label_seal.json')['utc'])
for line in (out/'access_log.jsonl').read_text(encoding='utf-8').splitlines():
 q=json.loads(line);assert q['phase'] in c['inputs'][q['key']]['phases']
 if q['phase']=='evaluate':assert datetime.datetime.fromisoformat(q['utc'])>=sealed
assert not r['decisions']['proceed_2_3']
tm=datetime.datetime.now(datetime.timezone.utc);mins=(tm-datetime.datetime.fromisoformat(c['started_utc'])).total_seconds()/60
size=sum(x.stat().st_size for x in out.rglob('*') if x.is_file());assert size<50*1024**2
v={'status':'PASS','inputs_unchanged':len(c['inputs']),'implementation_unchanged':len(c['implementation']),'historical_reports_unchanged':len(c['historical_result_hashes']),
 'old_predictions_exact':20,'old_bootstraps_exact':20,'contrast_max_abs':err,'evaluation_after_label_seal':True,'elapsed_minutes':mins,'time_basis':c['time_basis'],
 'new_label_KKT_max_abs':max(x['independent_projected_gradient_max_abs'] for x in f['fits'].values()),**r['verification'],'output_MiB_before_report':size/1024**2}
dump(out/'verification.json',v)
lines=['# 2-2 결과: teacher head만으로 재구성한 공개 전달 대조','',
 '**결론: 현재 MT가 teacher head에 없는 class 공통 평균을 추가로 사용해야 얻는 효용은 확인되지 않았다. 같은 head 정보와 공개 연산만 사용하는 HR가 일반 KD보다 좋았고, MT보다도 AUROC 점추정이 조금 높았다. 그러나 강한 DINO-only 대비 두 release 모두 개선하는 사전 기준은 미충족이다. 다음2-3은 보류한다.**','',
 '이는 내부 연산 대조의 결과다. HR를 새 방법으로 이름 바꿔 CVPR 기여로 선언하지 않는다. 현재 연구 단계2를 유지한다. 상위2 전체 완료도 아니다.','',
 '## 실행 범위','',
 '고정 공개128/기존DINO head DP1·DP2를 재사용했다. 공개 H_D로 class 차이 d를 복구하고 class 공통 평균만 P-only s_P로 대체한 뒤, 같은 class별 MT 지도/RN ridge/zero-anchor bounded solver를 사용했다. RN-only 라벨2개를 봉인한 후 새 DenseNet/RN readout4개와 기존20개 결과를 함께 비교했다. 새Q/noise/release/pixel/encoder forward/수신자탐색/Expert/Reserved는0이다.',
 'HR construction은 실제 보호 class 평균을 읽지 않는다. 기존 source head와 공개 P target/maps만 사용한다. class 공통 성분이 target에 주는 영향을 제거한 대조이며, class clipping이나 지도 ridge를 새로 고르지 않았다.','',
 '## DenseNet 개발 결과','',
 '| 구성 | DP1 AUROC | DP2 AUROC | DP1 AP | DP2 AP |','|---|---:|---:|---:|---:|']
for method in ['DINO_ONLY','KD_RN_ONLY','MT_RN_ONLY','HR_RN_ONLY']:
 a=r['metrics']['DP1_'+method+'_DenseNet'];b0=r['metrics']['DP2_'+method+'_DenseNet']
 lines.append(f"| {method} | {a['AUROC']['point']:.6f} | {b0['AUROC']['point']:.6f} | {a['AP']['point']:.6f} | {b0['AP']['point']:.6f} |")
lines+=['','| 비교 | release | AUROC 차이 | 환자95%CI | AP 차이 | 환자95%CI |','|---|---|---:|---|---:|---|']
for a,b0 in [('MT_RN_ONLY','HR_RN_ONLY'),('HR_RN_ONLY','KD_RN_ONLY'),('HR_RN_ONLY','DINO_ONLY')]:
 for draw in ['DP1','DP2']:
  x=r['per_release_contrasts'][draw+'_'+a+'_minus_'+b0+'_DenseNet'];au=x['AUROC'];ap=x['AP']
  lines.append(f"| {a}−{b0} | {draw} | {au['delta']:+.6f} | {ci(au['patient_cluster_95'])} | {ap['delta']:+.6f} | {ci(ap['patient_cluster_95'])} |")
lines+=['','| 두 고정 release의 평균 차이 | AUROC | 조건부95%CI | AP | 조건부95%CI |','|---|---:|---|---:|---|']
for a,b0 in [('MT_RN_ONLY','HR_RN_ONLY'),('HR_RN_ONLY','KD_RN_ONLY'),('HR_RN_ONLY','DINO_ONLY')]:
 x=r['fixed_two_release_mean_contrasts']['MEAN_'+a+'_minus_'+b0+'_DenseNet'];au=x['AUROC'];ap=x['AP']
 lines.append(f"| {a}−{b0} | {au['mean_delta']:+.6f} | {ci(au['fixed_two_release_patient_cluster_95'])} | {ap['mean_delta']:+.6f} | {ci(ap['fixed_two_release_patient_cluster_95'])} |")
lines+=['','## 결과가 답한 질문','',
 '1. **보호 class 공통 평균의 필요성: 미확인.** MT−HR의 평균 AUROC는−0.002083, 구간[−0.003922,−0.000328]. 현재 두 bank에서는 오히려 HR가 소폭 높다. 사적 정보가 일반적으로 해롭다거나 public midpoint가 noise에 강하다는 모집단 주장은 하지 않는다.',
 '2. **현재 teacher-score KD의 한계: 더 구체화.** HR−KD 평균 AUROC +0.055366, AP +0.014459이며 둘의 조건부 구간이 양수다. 같은 DINO head 정보를 공개 연산으로 다르게 처리해도 기존 KD 대비 이득이 남았다. 따라서 MT−KD 양성을 teacher가 보존하지 못한 추가 class 평균 정보의 효과라고 설명할 수 없다. 어떤 공개 처리 요소 하나가 유일한 원인인지까지 분리한 실험은 아니다.',
 '3. **강한 source-only 대비: 혼합.** HR−DINO-only는 DP1+0.048270, DP2−0.002719. 평균AUROC+0.022776, CI[+0.001641,+0.045153]은 긍정적이다. 이 근거를 지우지 않는다. 다만 두 release 모두 양수라는 사전 투자 기준은 실패했다. 엄격한 이 기준은 반복 비용을 정한 개발 규칙이며, 방법이 무용하다는 통계적 결론이 아니다.',
 '4. **선행 대비 기여: 여전히 미확정.** MT의 공개 가중표현 동치, 기존 Label Solve와의 관계는2-1에 남겼다. 이번 HR를 표준 대안 대신 새 발명으로 승격하지 않는다. 기존 방식의 유용한 구현과 논문 독자적 기여를 구분한다.','',
 '지표는 영상AUROC/AP이며 동일2000회 환자-cluster를 재표집했다. 두 release 평균은 지표 차이의 산술 평균이고 앙상블이 아니다. 현재 DP 출력·carrier·학습라벨에 조건부이며 noise 모집단/개발 선택의 불확실성은 포함하지 않는다. RN은 구성 모델 참고치이며 모든 수치는 results.json에 보존했다.','',
 '## 검산·비용','',
 f"- 실행 전 예상20~40분. 계약 작성 시작(이후 adapter 준비 포함)부터 최종검산/기록까지 **{mins:.2f}분**.",
 f"- 새 라벨2개 구성 {f['seconds']:.3f}초, 평가/bootstrap {r['seconds']:.3f}초. GPU/영상 추출 없음.",
 f"- P-only target 재현 {f['P_target_reconstruction_max_abs']:.3e}; source head 복구 {f['source_head_recovery_max_abs']:.3e}; 독립 목표 분해식 {f['independent_target_formula_max_abs']:.3e}.",
 f"- 라벨KKT {v['new_label_KKT_max_abs']:.3e}; ridge primal/dual {v['ridge_primal_dual_max_abs']:.3e}; bootstrap 독립 witness {v['bootstrap_sklearn_max_abs']:.3e}.",
 f"- 기존20개 예측/bootstrap 그대로 일치. 입력{v['inputs_unchanged']}개·구현{v['implementation_unchanged']}개·과거 보고서{v['historical_reports_unchanged']}개 불변. 대비/평균/CI 재구성 오차{err:.3e}. 평가 전 라벨 봉인 확인.",
 f'- 보고 전 추가 저장 {size/1024**2:.2f}MiB. 실패·재시도·성능을 보고 설정 변경 없음.','',
 '## 정확한 번호와 다음 상태','',
 '| 번호 | 상태 | 의미 |','|---|---|---|',
 '| 1-1~1-5 | 현재 정한 범위 완료 | KD 대비 이득은 있음. source-only/결합/합성 제작 추가 가치는 혼합 또는 미확인 |',
 '| 2-1 | 완료 | 정확한 선행 겹침과 class 공통 성분이라는 연산 차이 정리 |',
 '| 2-2 | 완료 | 추가 공통 성분 필요성 미확인. 같은 teacher head의 공개 처리로 KD 개선 유지 |',
 '| **2-3** | **보류·미실행** | MT/HR 모두DINO-only 대비 사전 반복 투자 기준 미충족 |',
 '| 3-1~3-3 | 미착수 | 독자적 기여 후보/강한 대조를 동결할 근거가 아직 부족. Expert/Reserved 유지 |','',
 '**다음 순서상의 작업은2-3**이며, 원래 규모는 새 공개128 선정1회(salt202), 기존DP1/DP2×고정후보/대조, 예상20~40분이다. 이번 사전 조건에서 실행 대상이 성립하지 않아 시작하지 않았다. 좋은 선정에서 이기는지 보기 위해 기준을 사후 완화하지 않는다.',
 '상위2에서 남은 것은 새로운 선정 반복 자체가 아니라, 표준 공개 가중표현/label solve에도 같은 자원을 허용했을 때 추가 가치가 남는 구체적인 사용 조건과 변경 연산의 설계다. 현재 이득을 버리지는 않지만 이 일이 완료됐다고도 하지 않는다. 그 설계가 정해지기 전에는2-3이나독립평가로 기여 문제를 대신하지 않는다. 새로운 대형 실험·계수탐색·추가 release를 자동 추가하지 않았다.','',
 '[2-1 연산·선행 비교](PUBLIC_TRANSFER_2_1_OPERATION_COMPARISON_20260928.md). 실행 계약/입출력은 `'+out.name+'`에 보존.','']
report=rr/'PUBLIC_TRANSFER_2_2_HEAD_RECONSTRUCTION_RESULTS_20260928.md';assert not report.exists();report.write_text('\n'.join(lines),encoding='utf-8')
block='**2-2 COMPLETE / 2-3 DEFERRED BY PREDECLARED GATE — 2026-09-28**\n\n2-2 HR RN-only(head차이유지·공통평균P) DenseNetAUROC0.697190/0.692352. MT−HR 평균−0.002083 CI[−0.003922,−0.000328]: 추가보호공통성분 필요성미확인. HR−KD 평균+0.055366 CI양수; 같은head 공개처리로기존KD개선유지. HR−DINO 평균+0.022776 CI양수이나DP2−0.002719로두release모두개선 기준미충족. 사전계약대로2-3보류,3번미착수. 1-5/2-1/2-2완료를논문기여확정으로바꾸지않음. 새라벨2/readout4·기존20재사용,새Q/release/Expert/Reserved0. Stage2유지. 현재실행없음.'
for pp in [tr/'AGENTS.md',tr/'CURRENT_STATUS.md',tr/'WORKLOG.md',rr/'RESEARCH_FRAMEWORK.md']:
 link=report.name if pp.parent==rr else 'CVPR 주제 탐색/research_2026-09-10/'+report.name
 prepend(pp,block+'\n\n[Results](<'+link+'>)')
plan=rr/'PUBLIC_TRANSFER_NUMBERED_PLAN_20260928.md'
prepend(plan,'**현재 번호/상태 — 2026-09-28:** 1-1~1-5 현재범위 완료,2-1완료,2-2완료. **2-3은사전투자기준미충족으로보류·미실행**,3-1~3-3미착수. HR−KD이득과HR−DINO평균양성은보존하되두release모두개선은실패했다. 아래이전상태는기록당시내용이며최신상태는본표시를따른다. [2-2결과]('+report.name+')')
s=read(rr/'research_state.json');s['current_result_report']=report.name;s['last_efficacy_result_report']=report.name
s['step2_active_task']='2-2 completed: head-only public reconstruction preserves KD gains; extra class midpoint not useful in fixed draws.2-3 deferred by preregistered source-only consistency gate.'
s['active_execution']={'status':'no_running_execution','last_completed':'2-2','completed_utc':tm.isoformat(),'report':report.name,'next_item':'2-3','next_item_status':'deferred_gate_unmet'}
s['next_task']='2-3 is the next numbered experiment but not authorized by its frozen scientific gate: MT/HR did not improve both releases over DINO-only. Retain Stage2; specify a distinct use-case/operation beyond standard KME/Label Solve before revising the plan. Do not auto-run new carrier, noise or Expert/Reserved.'
s['numbered_execution_plan_20260928']['current_item']='2-2 complete;2-3 deferred by predeclared gate'
s['numbered_execution_plan_20260928']['report_sha256']=sha(plan)
s['numbered_execution_plan_20260928'].setdefault('upper_2',{}).update({'status':'partial_complete_gate_unmet','completed':['2-1 operation/prior review','2-2 head reconstruction control'],
 'pending':['2-3 public128 selection repeat, deferred gate unmet'],'contribution_status':'not_established'})
s['head_reconstruction_2_2_20260928']={'status':'complete_mixed','report':report.name,'report_sha256':sha(report),'out_dir':str(out),'decisions':r['decisions'],'means':r['fixed_two_release_mean_contrasts'],'verification':v,'new_releases':0}
dump(rr/'research_state.json',s)
dump(out/'finalization.json',{'item':'2-2','status':'COMPLETE','report_sha256':sha(report),'verification':v,'next_item':'2-3','next_item_status':'deferred_gate_unmet','finalizer_sha256':sha(__file__)})
print(json.dumps({'item':'2-2','status':'RECORDED','elapsed_minutes':mins,'verification':v,'next_item':'2-3','next_item_status':'deferred_gate_unmet'},ensure_ascii=True,indent=2))
