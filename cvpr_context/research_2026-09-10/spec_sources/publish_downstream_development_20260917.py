"""Publish complete local development evidence; no model execution or data selection."""
import json,hashlib
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
ROOT=Path(__file__).resolve().parents[1];CODE=ROOT.parents[1]/'code_working';OUT=CODE/'_reports/downstream_development_20260917_v1'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
r=read(OUT/'result.json');v=read(OUT/'verification.json');c=read(OUT/'contract.json');cal=read(OUT/'calibration_choice.json');gm=read(OUT/'generation_manifest.json');gen=read(OUT/'generation.json')
if v['status']!='PASS_INTEGRITY_NOT_EFFICACY' or v['result_sha256']!=sha(OUT/'result.json'):raise RuntimeError('Unverified')
passed=r['engineering_gate_passed'];judgment='positive_development_private_utility_candidate' if passed else 'negative_private_synthetic_development_gate'
import torch
from collections import defaultdict
chosen=cal['chosen'];a=torch.load(OUT/'calibration'/f"R1_lr{chosen['lr']:g}"/f"state_{chosen['steps']}.pt",map_location='cpu',weights_only=True)
b=torch.load(OUT/'runs/R1_11'/f"state_{chosen['steps']}.pt",map_location='cpu',weights_only=True)
if set(a)!=set(b) or not all(torch.equal(a[k],b[k]) for k in a):raise RuntimeError('Calibration and main R1 checkpoint mismatch')
label_exposure={}
for folder in sorted((OUT/'runs').iterdir()):
 e=read(folder/'exposure.json');counts=defaultdict(int)
 for x in e['per_image']:counts[(x['namespace'],x['patient_id'],x['label'])]+=x['exposures']
 label_exposure[folder.name]=[dict(namespace=k[0],patient_id=k[1],label=k[2],exposures=n) for k,n in sorted(counts.items())]
 if sum(counts.values())!=32*chosen['steps']:raise RuntimeError('Exposure accounting')
with (OUT/'patient_label_exposure.json').open('x',encoding='utf-8') as f:json.dump(label_exposure,f,indent=2)
with (OUT/'report_additional_checks.json').open('x',encoding='utf-8') as f:
 json.dump(dict(calibration_selected_vs_main_R1_seed11_all_tensors_exact=True,
  patient_label_exposure_sha256=sha(OUT/'patient_label_exposure.json'),extra_model_forward=0,extra_training=0),f,indent=2)
arms=c['arms'];names=dict(R0='공개 real 기본',R1='공개 real + 일반 증강',S0='backbone 합성',S1='public-head 합성',S2='private-only 합성',S3='pooled 합성',Dreal='private real 비DP 진단')
lines=['# 비DP downstream 본 개발 결과 — 2026-09-17','',
('**판정: 좋은 개발 결과다. 사전 지정한 사적 합성자료 효용 관문을 통과했다.**' if passed else '**판정: 사적 합성자료 효용에는 나쁜 결과다. 사전 지정한 개발 관문을 통과하지 못했다.**'),
'실행·수치 검산의 통과와 방법의 효용 판정은 구분한다. 이번에는 준비만 한 것이 아니라 512장 생성, R1 calibration 두 trajectory, 7군×3seed의 실제 학습과 개발 평가를 완료했다. Expert final·reserved와 DP는 실행하지 않았다.','',
'[실행 전 명세](TRACK1_DOWNSTREAM_DEVELOPMENT_PROTOCOL_20260917.md)와 [master protocol](TRACK1_DOWNSTREAM_MASTER_PROTOCOL_20260917.md)을 적용했다. 큰 단계2·방향1을 유지한다.','',
'## 실제 성능','',
'영상별 NIH weak label을 사용하는 method-development 2,026명·5,047장(양성223장)의 결과다. AUROC/AP는 seed11/23/37 각각 계산한 뒤 평균했다. 음성은 기흉 label0이며 정상 영상만을 뜻하지 않는다.','',
'| Arm | 역할 | AUROC 평균 | AP 평균 | AUROC seed11 /23 /37 |','|---|---|---:|---:|---|']
for a in arms:lines.append(f"| {a} | {names[a]} | {r['means'][a]['AUROC']:.6f} | {r['means'][a]['AP']:.6f} | "+' / '.join(f"{x['AUROC']:.6f}" for x in r['scores'][a])+' |')
lines+=['','| 비교 | 평균 ΔAUROC | 평균 ΔAP | 양의 AUROC seed | 관문 |','|---|---:|---:|---:|---|']
for a in ['S2','S3']:
 for b,d in r['gate'][a]['comparisons'].items():
  lines.append(f"| {a}−{b} | {d['AUROC_delta']:+.6f} | {d['AP_delta']:+.6f} | {d['positive_seeds']}/3 | {'전체 통과' if r['gate'][a]['passed'] else '전체 미통과'} |")
lines+=['','S2 또는 S3가 S1과 R1 **각각**에 대해 평균AUROC +.01 이상, 양의 차이 최소2/3seed, 평균AP 비감소를 모두 만족해야 통과다. 이 .01은 추가 DP 개발 투자 기준이며 임상적 유의성 또는 final 검정 기준이 아니다.','',
'## 환자 군집을 고려한 보조 불확실성','',
'환자 단위2000회 paired cluster bootstrap에 같은 draw를 모든 arm/seed에 적용했다. 매 draw에서 seed별 metric을 계산한 뒤 평균했다. 아래95% percentile 구간은 개발용이며, 한 생성bank·세 classifier에 조건부다. 반복촬영 영상을 독립 환자로 취급하지 않았다.','',
'| 비교 | ΔAUROC 95% 구간 | ΔAP 95% 구간 |','|---|---|---|']
for comp,d in v['cluster_bootstrap'].items():
 a=d['AUROC']['percentile95'];p=d['AP']['percentile95'];lines.append(f'| {comp} | [{a[0]:+.6f}, {a[1]:+.6f}] | [{p[0]:+.6f}, {p[1]:+.6f}] |')
dreal=r['means']['Dreal']['AUROC']-r['means']['R1']['AUROC']
lines+=['','## 해석과 결정','',f"Dreal−R1의 평균AUROC는 {dreal:+.6f}, AP는 {r['means']['Dreal']['AP']-r['means']['R1']['AP']:+.6f}다. Dreal은 private real을 직접 사용한 비DP 진단 비교이며 수학적 상한이나 보호 방법이 아니다.",
('사적 합성자료의 개발 효용 후보가 생겼다. 그러나 독립 생성bank, 강한 real augmentation, 다른 classifier 및 patient-DP·DP-LoRA 비교는 아직 없다. 이 개발 결과만으로 CVPR 기여 또는 DP 효용을 확정하지 않는다.' if passed else
 '현재 고정 head·생성bank·classifier 설정은 DP 개발 투자 기준에 미달했다. DP로 확대하지 않는다. 기준·seed·prompt·head scale을 결과 후 바꾸어 이 실행을 성공으로 만들지 않는다. 이 결과를 private data 전체의 무용성이나 모든 생성 방법의 불가능성으로 확대하지 않는다.'),
'',
'## R1 calibration과 실행량','',
'Classifier-selection 2,027명·5,097장만 사용했다. 공개 R1 seed11을 LR1e-4/3e-4 각각800step까지 연속 학습하고400/800 checkpoint를 비교했다. 중간 AdamW 초기화는 없다.','',
'| LR | step | Selection AUROC | AP |','|---:|---:|---:|---:|']
for x in cal['candidates']:lines.append(f"| {x['lr']:g} | {x['steps']} | {x['AUROC']:.6f} | {x['AP']:.6f} |")
x=cal['chosen'];lines +=['',f"최고AUROC와 .002 이내 후보 중 적은 step, 작은 LR 우선 규칙으로 **LR{x['lr']:g}, {x['steps']}step**을 선택했다. 이후21run 모두에 동일하게 적용했다. Method-development 성능은21run이 끝난 뒤에만 계산했다.",
f"실제 update는 calibration1600 + development{r['development_optimizer_updates']} + 별도 관찰기 parity4 = **{1600+r['development_optimizer_updates']+4}회**다. 과거 잘못된98회와 이번 학습을 섞지 않았다.",'',
'## 입력·노출량·비용','',
'새 latent64개에 기흉64/비기흉64 요청을 대응시켜 방법별128장, 총512장을 만들었다. 음성 요청은 No Finding22/Effusion21/Cardiomegaly21이다. 모든 방법의 latent·prompt mapping은 같고 생성 후 filtering은 없었다. 요청 라벨은 생성영상의 전문가 판독 정답이 아니다.',
'',
'R0/R1은 public real32장, S0–S3은 public real16+synthetic16장, Dreal은 public16+private real16장이다. 따라서 결과는 **동일 계산량에서 공개 real batch 절반을 대체**한 효과이며, 동일한 real 학습량에 합성을 단순 추가한 효과가 아니다.','',
'| Arm | 공개 양성6장 최대 노출 (seed11/23/37) | 공개 양성6장 중앙 노출 | 총 학습초(3seed) |','|---|---|---|---:|']
train_seconds=0.
for a in arms:
 ex=[read(OUT/'runs'/f'{a}_{s}'/'exposure.json')['groups']['real/public/1'] for s in c['classifier_seeds']]
 tr=[read(OUT/'runs'/f'{a}_{s}'/'trace.json') for s in c['classifier_seeds']];sec=sum(x['seconds'] for x in tr);train_seconds+=sec
 lines.append(f"| {a} | "+' / '.join(str(x['image_exposure_max']) for x in ex)+' | '+' / '.join(f"{x['image_exposure_median']:g}" for x in ex)+f" | {sec:.1f} |")
cal_sec=sum(read(p)['seconds'] for p in (OUT/'calibration').glob('*/trace.json'))
lines+=['',f"512장 생성·저장·모델 적재 총 {gen['seconds']:.1f}초, calibration 학습 {cal_sec:.1f}초,21run 학습 {train_seconds:.1f}초다. 평가 추론·검산·원본 재결속·구현 시간은 별도다.",
'',
'| 생성 방법 | 평균 연산초/장 | 평균 저장초/장 | 최대 allocated GiB |','|---|---:|---:|---:|']
for m in c['methods']:
 rr=[x for x in gm if x['method']==m];lines.append(f"| {m} | {np.mean([x['seconds'] for x in rr]):.3f} | {np.mean([x['output_IO_seconds'] for x in rr]):.3f} | {max(x['peak_allocated'] for x in rr)/2**30:.3f} |")
lines+=['','각 run의 exposure.json에 모든 public/private/synthetic image 및 patient/cell별 횟수, unique coverage, 최대·중앙 노출을 남겼다. Calibration의400/800 exposure도 별도 보존했다. 이 짧은 단일 장비 측정을 patient-DP LoRA 대비 비용 우위로 주장하지 않는다.','',
'## 실행·독립 검산','',
f"검증된 data_v2/train_v2는 변경하지 않고 별도 runner에 SHA를 결속했다. 중간 저장용 관찰 hook의2step plain/observed final state·입력·logit·loss가 exact였고,1step 저장state도 과거 검증과 일치했다. 원본11,277장의 SHA·decode·cache를 다시 대조했다. Legacy import와 허용 명부 밖 raw pixel 접근을 차단했다.",
f"독립 검산 {v['checks']:,}항목은 성능 표본 수가 아니다.512PNG/trajectory,15,360DDIM 전이, 모든 학습 source draw·label·array binding·BCE·exposure, 실제 최종state, calibration 선택과21개 AUROC/AP를 다시 계산했다. 최대 metric 차이는 {v['max_metric_difference']:.3g}, 최대 DDIM 차이는 {v['max_DDIM_absolute_difference']:.3g}였다. 독립 검산기는 생산 data/train 수치 함수를 import하지 않는다.",
'',
'완료 cell/run은 SHA marker로만 재사용한다. Expert final532명·810장과 reserved4213명은 pixel/prediction을 열지 않았다. 기존192장 pooled 미통과, PadChest·CheXzero 실패, 과거98회 무효 학습은 그대로 보존했다. Final-ready=false다.','',
'## 근거','',
'- [계약](../../code_working/_reports/downstream_development_20260917_v1/contract.json), [R1 선택](../../code_working/_reports/downstream_development_20260917_v1/calibration_choice.json).',
'- [실제 성능 전체](../../code_working/_reports/downstream_development_20260917_v1/result.json), [독립 검산·환자 cluster 구간](../../code_working/_reports/downstream_development_20260917_v1/verification.json).',
'- [전체 생성 manifest](../../code_working/_reports/downstream_development_20260917_v1/generation_manifest.json), [안전 본실험 runner](../../code_working/downstream_utility/run_development_v1.py).','',
'이 문서는 로컬 실행·검산 결과다. 원격 main SHA 확인이나 push 완료를 뜻하지 않는다.']
report='TRACK1_DOWNSTREAM_DEVELOPMENT_RESULTS_20260917.md'
with (ROOT/report).open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
sources={'protocol':ROOT/'TRACK1_DOWNSTREAM_DEVELOPMENT_PROTOCOL_20260917.md','report':ROOT/report,
 'contract':OUT/'contract.json','calibration':OUT/'calibration_choice.json','result':OUT/'result.json','verification':OUT/'verification.json'}
record=dict(judgment=judgment,report=report,completed_utc=datetime.now(timezone.utc).isoformat(),engineering_gate_passed=passed,
 generated_images=512,calibration_updates=1600,development_updates=r['development_optimizer_updates'],instrumentation_updates=4,
 development_runs=21,means=r['means'],gate=r['gate'],expert_pixels=0,reserved_pixels=0,new_DP=0,final_ready=False,
 source_paths={k:str(p) for k,p in sources.items()},source_sha256={k:sha(p) for k,p in sources.items()},
 remote_main_checked=False,remote_push=False)
with (ROOT/'spec_sources/downstream_development_record_20260917.json').open('x',encoding='utf-8') as f:json.dump(record,f,ensure_ascii=False,indent=2)
for name in ['research_state.json','patient_baseline_spec.json']:
 p=ROOT/name;s=read(p);s['current_step']=2;s['current_result_report']=s['last_actual_model_result_report']=s['last_efficacy_result_report']=report
 if 'steps' in s:
  s['steps'][2]['status']='nonDP_downstream_pipeline_implemented_and_verified'
  s['steps'][2]['meaning']='Frozen medical head and safe downstream pipeline implemented; new medical DP remains pending.'
  s['steps'][3]['status']='nonDP_downstream_development_complete_gate_'+('passed' if passed else 'failed')
  s['steps'][3]['meaning']='Full512 bank, public calibration and21developmentruns complete; expert final remains closed.'
 s['current_planning_report']='TRACK1_DOWNSTREAM_MASTER_PROTOCOL_20260917.md';s['completed_downstream_development_20260917']=record
 s['step2_active_task']='Non-DP downstream development complete; private synthetic engineering gate '+('passed' if passed else 'failed')
 s['research_question_status']=judgment+'; no expert-final or patient-DP claim'
 s['next_task']=('Design bounded development robustness and patient-DP baseline implementation with all final cohorts closed; preserve the fixed successful non-DP result.' if passed else
 'Keep patient-DP expansion stopped. Review the observed real-private versus synthetic-private transmission gap using saved development outcomes; no automatic seed/prompt/head/threshold rescue or new GPU study.')
 s['next_task_output']='Concrete interpretation and next justified design; all locked final/reserved cohorts preserved.'
 s['step2_open_items']=['Private incremental utility is '+('a development candidate, not independently confirmed.' if passed else 'not established by this frozen synthetic package.'),
  'One synthetic bank, one classifier architecture and weak-label development limit generality.',
  'Medical patient-DP, accountant, patient-DP LoRA and complete cost-utility comparison remain unexecuted.',
  'All final non-DP/DP models and analysis must be frozen before expert-final predictions.']
 s['active_execution'].update(status='no_running_execution',last_completed_report=report,last_completed_output=str(OUT),
  next_execution_not_started=True,latest_planning_status=judgment,current_scope='Completed non-DP development; no running GPU work',
  next_substep_estimated_work_minutes=([45,75] if passed else [20,30]),
  next_substep_timing_condition='Design/review estimate only; no new GPU experiment or final access automatically scheduled.')
 s['updated_utc']=record['completed_utc'];p.write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(record,ensure_ascii=False))
