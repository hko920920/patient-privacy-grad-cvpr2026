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
assert sha(rr/c['design_report'])==c['design_report_sha256']
for n in ['label_seal.json','completion.json']:
 s=read(out/n);assert s['contract_sha256']==sha(out/'contract.json')
 for p,h in s['files'].items():assert sha(out/p)==h
l=nz(out/'soft_labels.npz');assert len(l)==6
for n,x in l.items():
 assert x.shape==(128,) and np.isfinite(x).all() and abs(x).max()<=1
 with (out/'label_packets'/f'{n}.csv').open(encoding='utf-8',newline='') as ff:rows=list(csv.DictReader(ff))
 assert np.array_equal(x,[float(z['regression_target']) for z in rows])
b=nz(out/'paired_bootstrap.npz');d=nz(out/'contrast_bootstrap.npz');p=nz(out/'predictions_private.npz')
ob=nz(c['inputs']['comparison_bootstrap']['path']);op=nz(c['inputs']['comparison_predictions']['path'])
bn={str(n):i for i,n in enumerate(b['names'])};dn={str(n):i for i,n in enumerate(d['names'])};pn={str(n):i for i,n in enumerate(p['names'])}
for i,n in enumerate(ob['names']):assert np.array_equal(b['metrics'][:,bn[str(n)]],ob['metrics'][:,i])
for i,n in enumerate(op['names']):assert np.array_equal(p['scores'][:,pn[str(n)]],op['scores'][:,i])
assert len(pn)==36
err=0.
for key,v in r['per_release_contrasts'].items():
 draw,rest=key.split('_',1);rec='DenseNet' if key.endswith('DenseNet') else 'ResNet18';stem=rest[:-(len(rec)+1)]
 if stem=='ANCHOR_BY_CORRECTION_INTERACTION':terms={'ACKD_RN_ONLY':1.,'AKD_RN_ONLY':-1.,'CKD_RN_ONLY':-1.,'KD_RN_ONLY':1.}
 else:
  aa,bb=stem.split('_minus_');terms={aa:1.,bb:-1.}
 dd=sum(val*b['metrics'][:,bn[draw+'_'+name+'_'+rec]] for name,val in terms.items())
 pt=sum(val*np.array([r['metrics'][draw+'_'+name+'_'+rec][m]['point'] for m in ['AUROC','AP']]) for name,val in terms.items())
 assert np.array_equal(dd,d['differences'][:,dn[key]])
 for k,m in enumerate(['AUROC','AP']):err=max(err,abs(pt[k]-v[m]['delta']),float(abs(np.quantile(dd[:,k],[.025,.975])-v[m]['patient_cluster_95']).max()))
for key,v in r['fixed_two_release_mean_contrasts'].items():
 stem=key[5:];k1='DP1_'+stem;k2='DP2_'+stem;dd=(d['differences'][:,dn[k1]]+d['differences'][:,dn[k2]])/2
 assert np.array_equal(dd,d['differences'][:,dn[key]])
 for k,m in enumerate(['AUROC','AP']):
  pt=(r['per_release_contrasts'][k1][m]['delta']+r['per_release_contrasts'][k2][m]['delta'])/2
  err=max(err,abs(pt-v[m]['mean_delta']),float(abs(np.quantile(dd[:,k],[.025,.975])-v[m]['fixed_two_release_patient_cluster_95']).max()))
assert err<1e-15
sealed=datetime.datetime.fromisoformat(read(out/'label_seal.json')['utc'])
for line in (out/'access_log.jsonl').read_text(encoding='utf-8').splitlines():
 q=json.loads(line);assert q['phase'] in c['inputs'][q['key']]['phases']
 if q['phase']=='evaluate':assert datetime.datetime.fromisoformat(q['utc'])>=sealed
tm=datetime.datetime.now(datetime.timezone.utc);mins=(tm-datetime.datetime.fromisoformat(c['started_utc'])).total_seconds()/60
size=sum(x.stat().st_size for x in out.rglob('*') if x.is_file());assert size<50*1024**2
v={'status':'PASS','inputs_unchanged':len(c['inputs']),'implementation_unchanged':len(c['implementation']),'historical_reports_unchanged':len(c['historical_result_hashes']),
 'old_predictions_exact':24,'old_bootstraps_exact':24,'contrast_max_abs':err,'evaluation_after_label_seal':True,'elapsed_minutes':mins,
 'label_KKT_max_abs':max(x['independent_projected_gradient_max_abs'] for x in f['fits'].values()),**r['verification'],'output_MiB_before_report':size/1024**2}
dump(out/'verification.json',v)
lines=['# 2-2 보완(2-2R1): 공개 기준값 보존·공분산 보정 KD 대조 결과','',
 '**결론: HR는 이번에 정의한 세 보정 대안 모두보다 DenseNet AUROC가 높았고, 세 사전 개발 hurdle을 모두 통과했다. 공개 공분산 보정으로 KD 자체는 개선됐지만, 이 보정만으로 HR의 전체 이득이 설명되지는 않았다.**','',
 '원래 DINO-only 대비 두release 모두 개선 gate는 바뀌지 않는다. 이번 성공은 더 강한 KD 연산 대조에 대한 추가 개발 근거이며, source-only 일반 우위·POST 전체 재현 대비 우위·새 원리·논문기여 완료가 아니다.','',
 '## 실행과 DenseNet 결과','',
 '고정 public128에서 기존 DP1/DP2를 재사용했다. AKD=공개RN supervised 기준값+일반KD로 옮긴 source 변화, CKD=공개 공분산 보정 KD, ACKD=둘다. 새6labels·12readouts를 계산했고 기존24readouts는 그대로 보존했다. 모든 labels를 봉인한 후 같은V/2000환자cluster bootstrap으로 평가했다.','',
 '| 목표 구성 | DP1 AUROC | DP2 AUROC | DP1 AP | DP2 AP |','|---|---:|---:|---:|---:|']
for method in ['DINO_ONLY','KD_RN_ONLY','AKD_RN_ONLY','CKD_RN_ONLY','ACKD_RN_ONLY','MT_RN_ONLY','HR_RN_ONLY']:
 a=r['metrics']['DP1_'+method+'_DenseNet'];z=r['metrics']['DP2_'+method+'_DenseNet']
 lines.append(f"| {method} | {a['AUROC']['point']:.6f} | {z['AUROC']['point']:.6f} | {a['AP']['point']:.6f} | {z['AP']['point']:.6f} |")
lines+=['','| 두 고정release 평균 대조 | AUROC차이 | 조건부95%CI | AP차이 | 조건부95%CI |','|---|---:|---|---:|---|']
stems=['HR_RN_ONLY_minus_ACKD_RN_ONLY','HR_RN_ONLY_minus_AKD_RN_ONLY','HR_RN_ONLY_minus_CKD_RN_ONLY','AKD_RN_ONLY_minus_KD_RN_ONLY','CKD_RN_ONLY_minus_KD_RN_ONLY','ACKD_RN_ONLY_minus_AKD_RN_ONLY','ACKD_RN_ONLY_minus_CKD_RN_ONLY','ANCHOR_BY_CORRECTION_INTERACTION']
for stem in stems:
 x=r['fixed_two_release_mean_contrasts']['MEAN_'+stem+'_DenseNet'];a=x['AUROC'];z=x['AP']
 lines.append(f"| {stem} | {a['mean_delta']:+.6f} | {ci(a['fixed_two_release_patient_cluster_95'])} | {z['mean_delta']:+.6f} | {ci(z['fixed_two_release_patient_cluster_95'])} |")
lines+=['','## 무엇이 확인됐나','',
 '1. **정의한 보정KD 대비 추가효용이 남았다.** 주대조 HR−ACKD 평균 AUROC+0.035346,CI[+0.018043,+0.052953]. 양쪽release 점차이 양수이고 AP평균도 양의구간이다. HR−CKD는+0.028844,CI[+0.014973,+0.043368], AP구간도양수. HR−AKD도AUROC구간양수이나AP구간은0포함이다.',
 '2. **공분산 보정은 KD의 일부 한계를 완화했다.** CKD−KD 평균AUROC+0.026522,CI[+0.003930,+0.049895]. 따라서 “KD는 정보가 없어서 안된다”가 아니라 같은source head에 대한 연산 선택이 결과에 영향을 줬다. AP구간은0포함. 이 효과를 새로운 정규화 이론으로 주장하지 않는다.',
 '3. **공개 기준값 보존만으로는 해결되지 않았다.** AKD−KD와ACKD−CKD는유의한평균AUROC개선을보이지않았다. 두고정release에서확인한결과이며 공개기준값보존원리자체가무용하다는뜻은아니다.',
 '4. **기여의 핵심 원인은 아직 하나로 분리되지 않았다.** HR의patient/class지도,clip,중심화,공개가중과새대조의image-level지도는여전히다르다. 현재public-geometry recipe의추가효용은남았지만,특정요소한개나선행전체대비신규성을이결과로확정하지않는다.',
 '5. **source-only는 계속 강한 대조다.** HR의기존평균+0.022776/양의조건부CI는보존하지만DP2−0.002719로원래2-3gate미충족이다. 새세보정도source-only개발hurdle을통과하지못했다. 이불편한비교를주표에서빼지않았다.','',
 '## 검산과 비용','',
 f'- 실행 전 예상20~35분. 계약 작성부터 최종 검산·기록까지 **{mins:.2f}분**. 목표/6labels {f["seconds"]:.3f}초, 12readouts/bootstrap {r["seconds"]:.3f}초.',
 f'- 독립 augmented least-squares 목표 일치 {f["verification"]["target_pseudoresponse_ridge_max_abs"]:.3e}; 공개 기준값 불변식 오차0.',
 f'- 라벨KKT {v["label_KKT_max_abs"]:.3e},ridge primal/dual {v["ridge_primal_dual_max_abs"]:.3e},bootstrap sklearn witness {v["bootstrap_sklearn_max_abs"]:.3e}.',
 f'- 입력{v["inputs_unchanged"]}개/구현{v["implementation_unchanged"]}개/과거보고서{v["historical_reports_unchanged"]}개불변. 기존24개예측/bootstrap bitwise동일. 모든contrast/CI재구성오차{err:.1e}.',
 f'- 추가출력 {size/1024**2:.2f}MiB. 새Q/release/DP noise/pixel optimization/model forward/Expert/Reserved0. 실패·재시도·성능기반설정변경없음.','',
 '영상 AUROC/AP이고 불확실성은환자cluster재표집이다. 현재두DP실현과carrier/labels를조건으로한구간이며noise모집단이나반복개발선택의불확실성은포함하지않는다. 보조비교는탐색적이고전체가족의95%확증적주장으로쓰지않는다. RN참고결과와모든개별release대비는 results.json에있다.','',
 '## 다음 번호와 범위','',
 '**다음은 2-3 보완(2-3R1): 새 공개128 선정에서 강화KD 대비 차이를 확인하는 한 번의 반복이다.** 이번에통과한것은2-2R1의강화KD대조gate이며,과거source-onlygate를사후통과로바꾸지않는다. 원래2-3의더넓은진행조건은여전히보류다.',
 '같은선정알고리즘에서hash salt의반복표지만202로고정한다. 기존DP두개×HR/AKD/CKD/ACKD/DINO-only=10labels와20readouts를한묶음으로계산한다. 세대안전부와DINO를유지해좋은대안만골라버리지않는다. 새목표계산은없고기존targetweights그대로사용한다. DenseNet을보고carrier를고르지않으며반복은한번만한다. 예상20~35분,새DP비용0.',
 '확인할주질문은HR−ACKD의조건부AUROC개선이새carrier에서도남는가이고,HR−나머지두보정및HR−DINO를함께보고한다. 미래의독립평가나전체선행우위가아니며,원인분리/신규성문제를반복성으로대신하지않는다.',
 '논문 기여 후보는 아직 동결하지 않는다. 이 반복 뒤에는 공개 지도의 patient/class 단위·clipping 정합성이 어떤 차이를 만드는지와 기존 조건부 평균 전달에 귀속되는 부분을 정리해야 한다. 새receiver/계수/DP출력탐색을자동추가하지않는다.','',
 '[설계와 선행](PUBLIC_TRANSFER_2_1R1_ANCHOR_GEOMETRY_DESIGN_20260928.md). 실행계약과상세출력: `'+out.name+'`.','']
report=rr/'PUBLIC_TRANSFER_2_2R1_KD_CORRECTIONS_RESULTS_20260928.md';assert not report.exists();report.write_text('\n'.join(lines),encoding='utf-8')
block='**2-2R1 COMPLETE / STRENGTHENED-KD CONTROL POSITIVE — 2026-09-28**\n\nHR−ACKD DenseNet평균AUROC+0.035346 CI[+0.018043,+0.052953],HR−CKD+0.028844 CI양수,HR−AKD+0.059421 CI양수. HR가세새보정모두개발hurdle통과. CKD−KD+0.026522로표준보정도일부개선. 특정원인/POST전체우위/논문기여완료아님. 원래2-3source-onlygate는계속미충족. 다음2-3R1은강화KD차이의공개carrier반복1회(10labels/20readouts,예상20~35분),과거gate변경없음. 새Q/release/Expert/Reserved0.'
for pp in [tr/'AGENTS.md',tr/'CURRENT_STATUS.md',tr/'WORKLOG.md',rr/'RESEARCH_FRAMEWORK.md']:
 link=report.name if pp.parent==rr else 'CVPR 주제 탐색/research_2026-09-10/'+report.name
 pre(pp,block+'\n\n[Results](<'+link+'>)')
plan=rr/'PUBLIC_TRANSFER_NUMBERED_PLAN_20260928.md'
pre(plan,'**2-2R1완료 / 다음2-3R1 — 2026-09-28:** 강화KD대조모두긍정통과. 별도보완2-3R1에서새공개128×기존DP1/DP2×HR/AKD/CKD/ACKD/DINO-only한번반복(10labels/20readouts,20~35분). 원래2-3source-onlygate는보류를보존한다. [결과]('+report.name+')')
s=read(rr/'research_state.json');s['current_result_report']=report.name;s['last_efficacy_result_report']=report.name
s['step2_active_task']='2-2R1 complete: HR retains gain versus all three corrected KD controls. Next2-3R1 limited carrier repetition; original2-3 source-only gate unchanged.'
s['active_execution']={'status':'no_running_execution','last_completed':'2-2R1','completed_utc':tm.isoformat(),'report':report.name,'next_item':'2-3R1'}
s['next_task']='Announce2-3R1 selection-repeat design10labels/20readouts and ETA20-35min; one fixed public selection salt202, frozen targets and all strengthened KD/source controls. No new DP release or final evaluation.'
s['numbered_execution_plan_20260928']['current_item']='2-2R1 complete;next2-3R1;original2-3 deferred';s['numbered_execution_plan_20260928']['report_sha256']=sha(plan)
s['numbered_execution_plan_20260928'].setdefault('upper_2_supplements',{})['2-2R1']={'status':'complete_positive_controls','report':report.name,'elapsed_minutes':mins}
dump(rr/'research_state.json',s)
dump(out/'finalization.json',{'status':'COMPLETE','elapsed_minutes':mins,'report':str(report),'report_sha256':sha(report),'verification_sha256':sha(out/'verification.json')})
print(json.dumps({'item':'2-2R1','status':'RECORDED','elapsed_minutes':mins,'verification':v},ensure_ascii=True))

