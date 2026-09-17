"""Publish a completed diagnostic without changing any previous completed record."""
from pathlib import Path
import json,hashlib,copy,csv
from datetime import datetime,timezone
import numpy as np
ROOT=Path(__file__).resolve().parent
THESIS=ROOT.parents[1]
OUT=THESIS/'code_working/_reports/real_support_20260917_v1'
REPORT='TRACK1_REAL_SUPPORT_DIAGNOSTIC_RESULTS_20260917.md'
PROTOCOL='TRACK1_REAL_SUPPORT_DIAGNOSTIC_PROTOCOL_20260917.md'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write_new(p,text):
    with p.open('x',encoding='utf-8',newline='') as f:f.write(text)
def jnew(p,value):write_new(p,json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def prepend(path,text):
    old=path.read_text(encoding='utf-8');head,rest=old.split('\n',1);path.write_text(head+'\n\n'+text.strip()+'\n'+rest,encoding='utf-8')
def main():
    x=read(OUT/'result.json');v=read(OUT/'verification.json');probe=read(OUT/'probe_verification.json');before=read(OUT/'state_before.json')
    assert v['status']=='PASS_EXECUTION_AND_INDEPENDENT_NUMERICS_NOT_CAUSAL_ATTRIBUTION'
    assert v['result_sha256']==sha(OUT/'result.json') and v['improvement_candidate']==x['improvement_candidate']
    assert not (ROOT/REPORT).exists()
    for n,old in before.items():assert read(ROOT/n)==old
    spec=ROOT/'spec_sources'
    for src,dst in [('result.json','real_support_diagnostic_record_20260917.json'),('verification.json','real_support_diagnostic_execution_verification_20260917.json'),('probe_verification.json','real_support_diagnostic_probe_verification_20260917.json')]:
        with (spec/dst).open('xb') as f:f.write((OUT/src).read_bytes())
    with (spec/'real_support_diagnostic_scores_20260917.csv').open('x',encoding='utf-8',newline='') as f:
        w=csv.writer(f);w.writerow(['arm','seed','AUROC','AP','BCE','balanced_BCE'])
        for arm in ('R0','R1','Rwide'):
            for z in x['scores'][arm]:w.writerow([arm,z['seed']]+[z[m] for m in ('AUROC','AP','BCE','balanced_BCE')])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10.2,3.7),constrained_layout=True)
    arms=('R0','R1','Rwide');colors=('#537290','#158c9e','#da842a')
    for ax,m in zip(axes,('AUROC','AP')):
        for i,s in enumerate((11,23,37)):
            ax.plot(range(3),[x['scores'][a][i][m] for a in arms],color=colors[i],marker='o',lw=1.2,label=f'Seed {s}')
        ax.scatter(range(3),[x['means'][a][m] for a in arms],color='black',marker='_',s=350,lw=2.5,label='Seed mean',zorder=5)
        ax.set_xticks(range(3),['R0\nPublic real','R1\nPublic + affine','Rwide\nExpanded real']);ax.set_ylabel(m);ax.grid(axis='y',alpha=.2)
    axes[0].legend(fontsize=8);fig.suptitle('Fixed 400-step real-data diagnostic | same weak-label development patients',fontsize=11)
    assets=ROOT/'assets';assets.mkdir(exist_ok=True);fig.savefig(assets/'real_support_diagnostic_20260917.png',dpi=170);plt.close(fig)
    d=x['comparisons']['Rwide-R1'];d0=x['comparisons']['Rwide-R0'];passed=x['improvement_candidate']
    judgment='실자료 범위 대조에는 긍정적이다' if passed else '실자료 범위 대조의 사전 개선 기준은 미통과다'
    verdict='개선 후보 기준을 통과했다' if passed else '개선 후보 기준을 통과하지 못했다'
    mean=x['means'];fmt=lambda z:f'{z:.6f}';ci=lambda z:'['+', '.join(f'{a:+.6f}' for a in z)+']'
    traces=[read(OUT/'runs'/f'Rwide_{s}'/'trace.json') for s in (11,23,37)]
    peak_allocated=max(t['peak_allocated'] for t in traces)/2**30
    peak_reserved=max(t['peak_reserved'] for t in traces)/2**30
    summary_rows='\n'.join(f'| {a} | {fmt(mean[a]["AUROC"])} | {fmt(mean[a]["AP"])} | {fmt(mean[a]["BCE"])} |' for a in ('R0','R1','Rwide'))
    seed_rows='\n'.join(f'| {s} | {fmt(x["scores"]["R1"][i]["AUROC"])} | {fmt(x["scores"]["Rwide"][i]["AUROC"])} | {d["AUROC"]["seed_deltas"][i]:+.6f} | {fmt(x["scores"]["Rwide"][i]["AP"])} |' for i,s in enumerate((11,23,37)))
    train_rows='\n'.join(f'| {s} | {role} | {z["images"]} | {fmt(z["AUROC"])} | {fmt(z["AP"])} | {fmt(z["BCE"])} |' for s,a in x['training_eval'].items() for role,z in a.items())
    exp_rows='\n'.join(f'| {s} | {a["actual_unique_patients"]} | {a["actual_unique_images"]} | {a["sources"]["public/1"]["draws"]} | {a["sources"]["public/1"]["image_exposure_min"]}–{a["sources"]["public/1"]["image_exposure_max"]} | {a["sources"]["public/1"]["image_exposure_median"]:g} |' for s,a in x['actual_exposure'].items())
    devclass='\n'.join(f'| {a} | {np.mean([z["by_label"]["0"]["logit_mean"] for z in x["scores"][a]]):.6f} | {np.mean([z["by_label"]["1"]["logit_mean"] for z in x["scores"][a]]):.6f} | {np.mean([z["by_label"]["1"]["BCE"] for z in x["scores"][a]]):.6f} |' for a in ('R1','Rwide'))
    body=f'''# 실환자 학습 범위 확대: 고정 분류기 개발 대조 결과

2026-09-17. **{judgment}.** Rwide는 같은 ResNet18·400 step에서 R1 대비 평균 AUROC {d['AUROC']['mean_delta']:+.6f}, AP {d['AP']['mean_delta']:+.6f}, AUROC 우세 {d['AUROC']['positive_seeds']}/3 seed로 {verdict}. 이는 추가 실자료에 직접 접근한 **비DP 일반화 진단**이며, 기존 private synthetic 효용 실패를 취소하거나 DP 진입을 허용하는 결과가 아니다.

[실행 전 명세]({PROTOCOL}) · [계획 검토](TRACK1_REAL_SUPPORT_DIAGNOSTIC_PLAN_REVIEW_20260917.md) · [전체 수치](spec_sources/real_support_diagnostic_record_20260917.json) · [seed별 CSV](spec_sources/real_support_diagnostic_scores_20260917.csv) · [실행 검산](spec_sources/real_support_diagnostic_execution_verification_20260917.json) · [4-update 독립 검산](spec_sources/real_support_diagnostic_probe_verification_20260917.json).

## 무엇을 실제로 바꿨는가

기존 public672명·813장(기흉 양성6명·6장)에 former-classifier-selection2027명·5097장(양성114명·230장) 전체를 더했다. 허용 pool은2699명·5910장, 양성120명·236장이다. Private80은 포함하지 않았다. 추가 자료의 original private_train 출처와 과거 R1 calibration 소비를 유지한다. 이 모델은 확대 공개 baseline이나 privacy-preserving arm이 아니다.

두 half-batch 모두 합친 pool에서 image label별 patient-uniform → image-uniform로 추출했다. 앞16장을 public으로 고정하지 않았다. ImageNet 초기화·ResNet18·224letterbox·affine·FP32·AdamW(lr1e-4,wd1e-4)·400step·batch32·seed11/23/37은 동일하다. 각 seed의 전체 제시12,800회, 양성6,400회도 같다. 변경된 pool은 환자/양성 다양성, 반복, 동반소견, 출처 구성 등을 함께 바꾼다. 반복 감소나 양성6명만의 인과 효과를 분리한 실험이 아니다.

## 같은 개발 환자에서의 결과

평가는 기존 method-development2026명·5047장(양성223장)의 영상별 weak label이다. 반복 사용한 개발자료이며 독립 confirmation이 아니다. 아래는 **seed별 metric을 계산한 뒤 평균**이며 점수 ensemble 성능이 아니다.

| Arm | 평균 AUROC | 평균 AP | 평균 BCE |
|---|---:|---:|---:|
{summary_rows}

| Seed | R1 AUROC | Rwide AUROC | 차이 | Rwide AP |
|---|---:|---:|---:|---:|
{seed_rows}

![세 학습 seed별 AUROC와 AP](assets/real_support_diagnostic_20260917.png)

- Primary Rwide−R1: AUROC {d['AUROC']['mean_delta']:+.6f}, 95% paired patient-cluster 구간 {ci(d['AUROC']['percentile95'])}; AP {d['AP']['mean_delta']:+.6f}, 구간 {ci(d['AP']['percentile95'])}.
- Secondary Rwide−R0: AUROC {d0['AUROC']['mean_delta']:+.6f}, 구간 {ci(d0['AUROC']['percentile95'])}; AP {d0['AP']['mean_delta']:+.6f}, 구간 {ci(d0['AP']['percentile95'])}.
- 사전 기준은 R1 대비 평균 AUROC ≥+0.01, 양의 차이 ≥2/3seed, 평균 AP 비감소다. 판정은 **{verdict}**. 절대 성능의 충분성이나 임상 유의성에 관한 기준은 아니다. R0는 사전 secondary이며 유리한 비교만 선택하지 않았다.
- 기존 Dreal 평균 AUROC0.596172/AP0.078445는 다른 자료 접근의 참고값이며, 이번 primary 대조나 수학적 상한이 아니다.

기존 저장 patient_counts2000개를 그대로 재사용했다. 동일 환자의 모든 영상과 재표집 배수를 모든 arm/seed에 같이 적용했다. 구간은 현재 학습된 모델과 자료에 조건부인 개발 기술 통계다. Patient bootstrap은 새로운 환자 cohort나 새로운 학습 seed를 만드는 절차가 아니다.

## 학습자료와 개발자료의 차이

해당 seed에서 실제 한 번 이상 학습한 고유영상만, 증강 없이 eval()/inference_mode()에서 측정했다. 학습하지 않은 허용 pool 영상은 아래 학습 metric에서 제외했다. Parameter와 BatchNorm buffer는 추론 전후 불변이며 다시 맞추지 않았다.

| Seed | 실자료 출처 | 실제 본 고유영상 | 학습 AUROC | 학습 AP | 평가 모드 BCE |
|---|---|---:|---:|---:|---:|
{train_rows}

| Arm | 개발 음성 평균 logit | 개발 양성 평균 logit | 개발 양성 BCE |
|---|---:|---:|---:|
{devclass}

학습 로그의 마지막50step BCE, 고유영상 BCE, 노출 가중 BCE는 입력·증강·가중치가 다른 값이다. 서로 같아야 하는 것은 아니다. 학습 성능만 낮아져도 train–development 차이는 줄 수 있으므로, 판정은 개발 성능의 실제 개선에 근거한다. Threshold를 변경하거나 개발영상으로 BN을 보정하지 않았다.

Rwide 마지막50step의 평균 학습 BCE는 seed11/23/37에서 각각 {x['actual_exposure']['11']['last50_training_BCE']:.6f} / {x['actual_exposure']['23']['last50_training_BCE']:.6f} / {x['actual_exposure']['37']['last50_training_BCE']:.6f}다. 전체400step 로그와 각 step의 실제 추출 이력은 checkpoint와 함께 보존했다.

이번 Rwide의 실제 본 학습영상 AUROC는 {min(a['all_seen']['AUROC'] for a in x['training_eval'].values()):.6f}–{max(a['all_seen']['AUROC'] for a in x['training_eval'].values()):.6f}이고, 개발 AUROC는 seed별 {min(a['AUROC'] for a in x['scores']['Rwide']):.6f}–{max(a['AUROC'] for a in x['scores']['Rwide']):.6f}다. 학습자료 판별력은 여전히 매우 높으며 개발 일반화 차이는 남아 있다. 개발 양성 점수도 평균적으로 음의 logit에 머문다. 따라서 이번 개선을 일반화 문제의 완전한 해결이나 임상적 사용 가능성으로 확대하지 않는다.

## 실제 노출과 역할 소비

| Seed | 실제 환자 | 실제 고유영상 | 기존 public 양성6장 제시 합 | 해당 영상당 최소–최대 | 중앙값 |
|---|---:|---:|---:|---:|---:|
{exp_rows}

각 seed는 양성120명·236장 전체를 사용했다. 허용5910장 전체를 각 run에서 다 학습했다는 뜻은 아니다. 원 R1의 양성6장 반복은 영상당1025–1113회였고, 이번 실제 범위는45–62회다. 같은 양성6400회가 더 많은 환자·영상에 배분됐다. Class별·source별·환자별·영상별 count는 로컬 trace와 exposure CSV, 공개 JSON 집계에 결속했다.

2027명 전체를 이 분기의 학습 자원으로 관리하고, 그중 실제 optimizer에 등장한 환자는 세 seed 합집합 {x['actually_trained_former_selection_patients']}명이다. 별도 consumed overlay에 실제 횟수를 기록했다. 원 역할표와 과거 calibration 기록은 수정하지 않았다. 해당 환자를 이 분기의 미사용 validation으로 재사용하지 않는다.

## 실행과 검산

계약과 소스를 모델 실행 전에 동결했다. 기존 data_v2/train_v2는 변경하지 않았고 legacy import를 차단했다. 공급자는 연결 구간 밖에서 원래 함수로 복원된다. 원본·bytes·decode·224cache10957장 전체를 검사한 뒤 다음 순서로 실행했다.

1. 원 R1 1step와 public-only를 넣은 새 공급자1step가 선택ID·실제GPU입력·logit·loss·최종state까지 exact.
2. Rwide1step를 같은 초기상태에서 두 번 실행해 동일 항목 exact. 네 run 모두 실제 trainable tensor62개 변경, finite loss/gradient, 새 AdamW.
3. 독립 raw→GPU 입력 재계산 통과 뒤, probe 가중치를 버리고 각 seed의 과거 동일 ImageNet 초기화에서400step씩, 총1200update 실행. 최종 모델3개를 모두 고정한 뒤 평가.
4. 원 R1 평가 경로는 고정 개발64장×3모델=192건을 재추론해 기존 raw logit과 exact. 192개 고유영상이 아니다.
5. 실제 trace1200batch·38400slot을 독립 sampler와 사전 일정에 대조. 원본10957장 전체를 독립 디코딩해 cache와 다시 일치 확인. AUROC/AP와2000draw×3arm×3seed=18000개 bootstrap metric 쌍을 별도 sklearn 계산으로 재검산.

네 probe 검산 {probe['checks']:,}항목, 본 독립 검산 {v['checks']:,}항목 통과. 최대 metric 오차 {v['max_metric_difference']:.3g}, bootstrap 오차 {v['max_bootstrap_difference']:.3g}. 이 수는 무결성 검사 수이며 성능 표본 수가 아니다. 새1200본update+4연결update 이외의 학습, 새 생성, DP, expert532/reserved4213 pixel·prediction 접근은0이다.

원본 최초 검산 {x['raw_audit_seconds']:.3f}초, 세 본학습 합계 {x['training_seconds']:.3f}초, 최종 독립 검산 {v['seconds']:.3f}초다. 본학습 peak allocated/reserved는 각각 {peak_allocated:.3f}/{peak_reserved:.3f}GiB다. 코드 구현·문서화 시간을 포함한 총 작업 시간과 구분한다. 실행은 완료됐으며 현재 실행 중인 GPU 작업은 없다.

## 연구 판단의 범위

이번에 확인한 것은 **이 확대된 실자료 pool과 같은 고정 분류기 학습 절차의 개발 결과**다. 기존 양성 지원 부족, 반복 암기, 라벨/분포 차이 중 원인 하나를 확정하지 않는다. 더 넓은 pool의400step도 수렴 보장은 아니다.

이번처럼 개발 AUROC와 AP가 세 seed에서 함께 개선되면, 같은 ResNet18·224 전처리·고정400step 절차가 모든 자료 범위에서 기흉 정보를 구별하지 못하는 것은 아니라는 근거가 된다. 다만 생성기를 사용하지 않은 대조이므로, 이전 합성영상의 내용·요청 라벨·학습 배합이 적절했는지는 여기서 확인하지 않았다. 원래 제한된 공개자료와 보호 대상 사적자료라는 접근 조건에서 같은 이득을 얻을 수 있는지도 별도 문제다.

기존 [full64 종료](TRACK1_CURRENT_HEAD_BRANCH_CLOSURE_20260917.md), [LoRA 사적 효용 미통과](TRACK1_LORA_TRANSFER_COMPARISON_RESULTS_20260917.md), [고정 분류기 일반화 진단](TRACK1_CLASSIFIER_GENERALIZATION_RESULTS_20260917.md)은 그대로 유지한다. 현재 결과를 새 공개 baseline, 사적 합성 효용 또는 DP 성공으로 바꾸지 않는다. Expert final과 reserved는 계속 보존하고 final_ready=false, 큰 단계2를 유지한다.

다음은 이 결과가 허용하는 자료 접근 조건과 아직 남은 전달 경로 질문을 구분해 다음 대조 하나의 필요성을 판단하는 일이다. 새 adapter·학습량·seed·생성 bank·DP를 자동으로 추가하지 않았다.
'''
    write_new(ROOT/REPORT,body)
    record=dict(report=REPORT,protocol=PROTOCOL,status='completed_real_support_diagnostic',output_directory=str(OUT),
      means=x['means'],comparisons=x['comparisons'],improvement_candidate=passed,new_training_updates=1200,probe_updates=4,
      training_pool_patients=2699,training_pool_images=5910,training_positive_patients=120,
      former_selection_patients=2027,source_roles_unchanged=True,role_overlay_status='consumption_recorded_not_validation',
      raw_files=10957,verification_checks=v['checks'],new_generation=0,DP_executed=False,expert_opened=False,reserved_opened=False,
      final_ready=False,private_synthetic_efficacy_established=False,causal_failure_source_established=False,
      contract_sha256=sha(OUT/'contract.json'),result_sha256=sha(OUT/'result.json'),verification_sha256=sha(OUT/'verification.json'),
      public_record='spec_sources/real_support_diagnostic_record_20260917.json')
    for name,old in before.items():
        a=copy.deepcopy(old);assert 'completed_real_support_diagnostic_20260917' not in a
        a['completed_real_support_diagnostic_20260917']=copy.deepcopy(record)
        for k in ('current_result_report','last_actual_model_result_report','current_decision_report'):a[k]=REPORT
        a['last_real_generalization_result_report']=REPORT
        a['current_planning_report']=PROTOCOL
        # Efficacy pointer means private-generation efficacy; real diagnostic has its own explicit pointer.
        assert a['last_efficacy_result_report']=='TRACK1_LORA_TRANSFER_COMPARISON_RESULTS_20260917.md'
        a['step2_active_task']='Fixed real-support diagnostic completed; expanded-pool development improvement flag='+str(passed)+'; no private synthetic or causal claim'
        a['next_task']='Interpret the fixed real-support contrast and separate permitted data access from the still-unresolved private synthetic transfer question. Select at most one justified prospective contrast; no automatic DP, generator, seed, step or final/reserved execution.'
        a['next_task_output']='Bounded next-comparison decision grounded in the observed support contrast; no rescue of prior frozen failures.'
        a['step2_open_items']=['Expanded real-data diagnostic is complete; it is NONDP and not a replacement public baseline',
          'Support, repetition, cohort composition and weak labels are not causally separated',
          'Original head and fixed LoRA private synthetic utility gates remain failed',
          'Former selection2027 is consumed as training for this branch, not reusable independent validation',
          'Expert532/reserved4213 remain closed, DP paused, full stage2 not complete']
        a['active_execution'].update(status='no_running_execution',last_completed_output=str(OUT),last_completed_report=REPORT,
          output_directory=str(OUT),next_execution_not_started=True,next_execution_scheduled=False,
          latest_planning_status='fixed_real_support_diagnostic_completed',stage2_completion_claim=False,
          completed_real_support_diagnostic=copy.deepcopy(record))
        a['step2_completed_diagnostics'].append('Rwide: same classifier400steps x3; expanded5910real-image pool; development AUROC delta '+f'{d["AUROC"]["mean_delta"]:+.6f}'+'; independent raw/sampler/metric verification; original failures and final resources preserved')
        a['updated_utc']=datetime.now(timezone.utc).isoformat()
        for k,val in old.items():
            if k.startswith('completed_'):assert a[k]==val
        for k,val in old['active_execution'].items():
            if k.startswith('completed_'):assert a['active_execution'][k]==val
        (ROOT/name).write_text(json.dumps(a,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    short=f'**2026-09-17 실환자 범위 대조 실제 완료 — {judgment}:** [고정 Rwide 3seed 결과](track1_real_support_diagnostic_results.html). 개발 AUROC R1 {mean["R1"]["AUROC"]:.6f} → Rwide {mean["Rwide"]["AUROC"]:.6f}, 차이{d["AUROC"]["mean_delta"]:+.6f}, 우세{d["AUROC"]["positive_seeds"]}/3; AP {mean["R1"]["AP"]:.6f} → {mean["Rwide"]["AP"]:.6f}. 추가2027명을 직접 학습한 비DP 진단이며 공개 baseline/사적 합성 효용이 아니다. 같은400step·초기화·학습코드,4probe와1200본update 및{v["checks"]:,}개 독립검산을 완료했다. 기존head/LoRA실패·원역할·expert532/reserved4213·DP중단 유지. Former-selection은 해당분기 학습으로 소비됐으며 다시validation으로 쓰지 않는다. 큰단계2·final_ready=false,실행중없음. 아래 최신/다음은 과거 이력이다.'
    prepend(ROOT/'RESEARCH_FRAMEWORK.md',short.replace('track1_real_support_diagnostic_results.html',REPORT))
    prepend(THESIS/'CURRENT_STATUS.md',short.replace('track1_real_support_diagnostic_results.html','CVPR%20주제%20탐색/research_2026-09-10/track1_real_support_diagnostic_results.html'))
    prepend(THESIS/'AGENTS.md',f'- Latest REAL-SUPPORT DIAGNOSTIC RESULT (2026-09-17): read {REPORT}. Rwide fixed NONDP real-pool diagnostic completed: AUROC {mean["Rwide"]["AUROC"]:.6f}, R1 delta{d["AUROC"]["mean_delta"]:+.6f}, AP {mean["Rwide"]["AP"]:.6f}, positive seeds{d["AUROC"]["positive_seeds"]}/3, prospective improvement flag={passed}. Both halves expanded2699patients5910images/P120patients; same classifier/initialization/lr/400steps/seeds. Raw10957, four independent wiring updates,1200mainupdates, unchanged corrected kernels, raw-cache/actual draws/18000bootstrap metric pairs checked. Former-selection2027 retain original private_train and prior calibration history but are now diagnostic training resources; never call them untouched validation. Not a public baseline, synthetic utility or causal attribution. Preserve every old completed result and head/LoRA failures. DP0, expert532/reserved4213 closed,stage2/final_ready=false. Last actual result is this real diagnostic; last private-generation efficacy remains LoRA failure. No running execution or automatic next experiment. Earlier latest/next entries are history.')
    prepend(THESIS/'WORKLOG.md',f'''## 150-REAL-PATIENT-SUPPORT-DIAGNOSTIC-EXECUTION (2026-09-17 KST)

- 사용자 승인 명세대로 구현·실행·검산했다. 시작 예상30–50분, 실제 원본검사와4update를 통과한 뒤 3seed×400 본학습을 수행했다. 과거data_v2/train_v2 수정0, 독립 새provider만 연결했다.
- {judgment}. Rwide AUROC{mean['Rwide']['AUROC']:.6f}/AP{mean['Rwide']['AP']:.6f}, R1차이{d['AUROC']['mean_delta']:+.6f}, seed{d['AUROC']['positive_seeds']}/3, 구간{ci(d['AUROC']['percentile95'])}, 후보기준={passed}. R0 비교와 실제학습자료 성능·노출도 전부보고했다.
- Raw10957장 SHA/decode/cache 결속 및 독립전체재디코딩, probe입력/가중치exact,38400실제slot/노출재계산,bootstrap18000metric쌍 재계산. 독립본검산{v['checks']}항목,최대오차{v['max_bootstrap_difference']:.3g}. 검사수는 성능표본수가 아니다.
- Additional2027명 출처private_train·calibration이력·원역할을 보존하고 실제학습소비overlay작성. Expert532/reserved4213/DP/새생성0. 기존head/LoRA음성결과불변. 자료범위 확장의 진단이지 private synthetic효용·공개baseline·단일원인증명이 아니다.
- {REPORT}/HTML/공개JSON/CSV/그림/두state/framework/AGENTS/CURRENT_STATUS 연결. 전체실험완료,실행중없음. 다음 대조나 DP를 자동정하지 않았다.
''')
    print(json.dumps(dict(report=REPORT,means=x['means'],improvement_candidate=passed),ensure_ascii=False))
if __name__=='__main__':main()
