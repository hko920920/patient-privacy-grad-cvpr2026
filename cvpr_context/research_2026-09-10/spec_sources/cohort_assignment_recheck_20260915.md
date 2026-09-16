# 2번 — 기존 cohort의 A/B 배정 재확인

2026-09-15 KST. 기존 코드와 manifest를 읽고, 저장된 400명의 배정을 독립 SHA 계산으로 재생성했다. **저장 배정과의 불일치는 0명이며, 현재 CDI feature에 A/B 정답을 직접 넣은 경로는 확인되지 않았다.** 이것은 label 누출이 전혀 없다는 증명이나 현재 성능의 원인 판정이 아니다.

이번 범위는 코드·hash·역할·배정의 결정적 확인이다. 새 GPU 호출, 원본 영상 추론, 공격 fitting, AUC 계산, 성능에 따른 분할 선택은 하지 않았다. calibration/test의 공격 특징이나 점수를 읽지 않았으며, 이미 잠긴 manifest의 역할·배정 일치만 확인했다. 모든 원본·실행 코드·계약은 수정하지 않았다.

## 1. 실제 배정 규칙

아래 코드 경로는 `code_working/` 기준이다.

1. `u_patient_audit/prepare_cohort.py:28–35,80–87`: 서로 다른 follow-up 번호와 중복 검사 조건을 만족하는 PA 영상 4장이 있는 환자만 적격이다. 영상 선택은 `stable('image-choice', patient_id, image_id)` 순서이며 A/B를 정하기 전에 수행한다. 적격 환자를 `stable('patient-cohort', patient_id)`로 정렬해 최대400명을 선택한다. 그 순서의 연속 구간으로 fit80, selection40, calibration140, test140을 정한다.
2. `prepare_cohort.py:93–105`: **각 역할 안에서** `(target_patient, patient_sex)` strata를 만든다. strata 내부 환자를 `stable('assignment-order', role, patient_id)`로 정렬하여 이웃끼리 짝짓는다. strata별 홀수 잔여는 `stable('leftover', role, patient_id)`로 다시 정렬하여 짝짓는다.
3. 각 쌍 `(p,q)`에 `seed('assignment-coin', role, p, q) % 2`를 적용해 한 명을 A, 한 명을 B로 정한다. A는 model_1의 추가 학습 참여, B는 model_2의 추가 학습 참여다. 공통 background 환자는 별도다.
4. `u_patient_audit/common.py:10,23–28`: salt는 `cvpr-u-pilot-20260914-v1`, stable은 salt와 인자를 `|`로 연결한 SHA256 hex다. seed는 앞15개 hex를 정수로 바꾼 값이다.

`target_patient`는 membership label이 아니다. `data_pipeline/build_nih_cxr14_manifests.py:146–152,377–378`에서 환자의 PA 기록 중 적어도 하나에 Pneumothorax, Pneumonia, Consolidation, Effusion, Mass, Nodule 중 하나가 있는지를 뜻하는 source enrichment flag다. 개별 U 영상에 반드시 그 finding이 있다는 뜻도 아니다.

fit과 selection은 assignment 정렬·coin의 hash 입력에 **다른 role 문자열**을 넣는다. 환자 선택, 영상 선택, 배정 정렬, 배정 coin도 서로 다른 prefix다. 따라서 단일 age/sex/ID threshold나 동일한 hash 순서의 앞 절반을 A로 두는 식의 공통 feature→label 규칙은 이 코드에 없다. 입력 구분이 통계적 독립성이나 모든 우연한 연관의 부재를 증명하는 것은 아니다.

## 2. 저장 배정의 독립 재생성 결과

코드 hash가 cohort lock의 `selection_code_sha256`와 같고, evaluation manifest의 hash도 잠긴 값과 같은지 먼저 확인했다. 이후 helper를 import하지 않고 SHA 문자열 구성·정렬·짝짓기·coin을 새로 계산했다.

| 역할 | 환자 | 원래 쌍 | strata별 홀수 잔여 환자 | 저장 A/B와 불일치 |
|---|---:|---:|---:|---:|
| fit | 80 | 40 | 2 | 0 |
| selection | 40 | 20 | 4 | 0 |
| calibration | 140 | 70 | 2 | 0 |
| test | 140 | 70 | 2 | 0 |

400명의 manifest 순서는 `patient-cohort` hash 정렬과 일치했다. 환자 내부 sex 값 불일치0명, source target_patient flag 불일치0명이었다. 이 작업에서 모든 source 영상의 적격성·near-duplicate 계산을 다시 수행한 것은 아니다. 선택된400명 이후 단계의 배정 재현과 저장된 코드·입력 결속을 확인한 것이다.

fit/selection의 실제 strata별 A/B 환자 수는 다음과 같다.

| target_patient / sex | fit A / B | selection A / B |
|---|---:|---:|
| 0 / F | 3 / 4 | 1 / 2 |
| 0 / M | 2 / 2 | 1 / 2 |
| 1 / F | 12 / 11 | 5 / 4 |
| 1 / M | 23 / 23 | 13 / 12 |

성별 총수는 fit의 양쪽이 F15/M25, selection의 양쪽이 F6/M14로 같다. 그러나 source target_patient=1은 fit A35/B34, selection A18/B16이다. **짝짓기와 잔여 처리로 대략 균형을 맞췄지만 모든 strata를 완전히 같게 만들지는 않았다.** 이 작은 공통 방향의 불균형이 실제 점수 차이를 얼마나 설명하는지는 계산하지 않았다.

나이, 전체 finding 조합, 질환 중증도, 환자별 획득·가용 follow-up 개수는 A/B 층화 변수에 포함되지 않는다. 4장 적격성/획득 범위 및 source enrichment 자체의 선택 편향도 있다. follow-up 번호가 다르다는 조건은 촬영 날짜·study UID에 의한 독립성을 보장하지 않는다. 이런 요소는 **가능한 교란 요인**이지 이번 확인으로 특정된 성능 원인이 아니다.

## 3. 확인한 feature 경계와 남는 한계

`u_patient_audit/analyze_cdi_u_cohort.py:93–139`는 `features`의 26개 scalar와 별도 NO objective만 x로 만들고, ID는 정렬/조회, A/B는 y와 검증에 쓴다. 환자 mean/meanmax는 그 숫자 배열에만 적용한다. age/sex/finding/ID/assignment 문자열을 feature 열에 넣는 코드는 이 경로에 없다.

CDI noise는 `cdi_adapter.py:74–76,110–117`의 `cdi-v1|master_seed|image_id|stream|module|draw`에서 정하며 A/B·membership·role을 입력하지 않는다. 이 seed 형식은 cohort assignment coin과 별도다. 영상 선택도 A/B 배정 전에 이루어지므로 직접 A/B에 따라 E/U 영상을 고르는 규칙은 없다.

다만 이미지에는 임상적·촬영·환자 특성이 있을 수 있고, ID로 고정한 noise도 이미지별 수치 차이를 만든다. 앞 코드 확인만으로 이미지 특성과 우연한 배정 연관, shared pretrained model의 반응, 반복 개발자료 사용의 영향을 배제하지 못한다.

전체 patient roster, role, strata 정보, 공개 salt/코드를 모두 가진 사람은 이 **인공 추가학습 배정**을 모델 없이 재구성할 수 있다. 이번 독립 재생성이 그 예다. 이는 labels를 가진 연구자의 manifest 감사와 같다. 현재 scorer에 그러한 ID/배정 재구성 정보를 제공했다는 관측은 아니며, 여기서 계산한 AUC를 그 우회 규칙으로 얻었다고 말하면 안 된다.

## 4. 현재 해석과 한 가지 가능한 후속 대조

**판정: 저장 배정 오류나 feature 열의 직접 label 누출은 이 확인에서 발견하지 못했다. 가능한 cohort 연관·분포 요인은 남아 있으며 원인이 확정된 것은 아니다.**

Base 대조에서도 일부 분류가 보일 수 있다는 논리와, 관측된 target 집계 개선이 base/배정 특성만으로 전부 설명된다는 결론은 다르다. 현재 base 대조에는 일부 target의 집계 이득 차이도 남아 있으므로 “base에서도 그대로 남았으니 cohort bias가 원인”이라고 쓰지 않는다. 해당 성능·CI는 별도의 검산된 base 결과 보고서에서 해석하며 이 메모에서 새로 계산하지 않았다.

배정 연관이 실제로 남는다는 의문을 더 확인해야 할 때의 **한 가지 CPU 대조 후보**는 원래 selection20쌍 안에서만 A/B를 뒤집는 label-permutation null 진단이다. 이미 고정된 scorer·base 점수·사전 주통계는 그대로 두고 모든 비교에 동일한 쌍별 뒤집기를 적용한다. 이는 단순 전역 permutation으로 원래 짝짓기 구조를 버리는 것을 피한다. 새 target 학습·GPU·환자 확대는 필요하지 않다.

이 후속은 아직 실행하지 않았다. deterministic hash coin을 random assignment처럼 취급하는 해석 가정과 selection40의 반복 사용을 명시해야 하며, 확증 검정·개인 인과효과·pretraining membership 판정으로 부르지 않는다. 결과를 본 뒤 유리한 scorer를 고르는 규칙도 추가하지 않는다. 이 제안이 현재 원인 확인을 완료했다는 뜻은 아니다.

## 5. 이번에 확인한 입력 SHA256

경로는 `code_working/` 기준이다.

| 파일 | SHA256 |
|---|---|
| `u_patient_audit/prepare_cohort.py` | `d20e7f19f692a394e2402838b3d06f8f5def06d90bce4b071ac0257247090f7f` |
| `u_patient_audit/common.py` | `22ff06578ab7431ff9ac670eaef112371c800b13e9a3a4bf9cc3a5ba11b768d7` |
| `u_patient_audit/analyze_cdi_u_cohort.py` | `752be6cdf1db57b2309bd897fee8f43f53f1807d23fa92cb07973b4720d14268` |
| `u_patient_audit/cdi_adapter.py` | `70d91b15397394d1405c8e930d2ec94ff4dcb0d3bbddd7ace6c8bdd58b0da8ce` |
| `data_pipeline/build_nih_cxr14_manifests.py` | `98ea10573d276d6b7ca4a7bf3f32ae6746c833b585bbcdc8e19f45ccc0def1e7` |
| `_reports/cvpr_u_pilot_v1_001/cohort/lock.json` | `187fdbcc6ea8b883e6b82d97f293d7a746df5408632e6f8824a9769866e1dfc4` |
| `_reports/cvpr_u_pilot_v1_001/cohort/evaluation_images.csv` | `08efab0eafeeca7cb6b8ff191c2eca2954e092a9c282a69785ac84a1cac35a7e` |
| `_data/derived/nih_cxr14_pa_target_enriched_v1/all_selected_pa_images_private.csv` | `547e0c1773d0f3681c22674124ff665cb730abf5153745ae32d110b0cab01bed` |

## 6. 재현용 CPU 명령

`code_working`에서 아래 PowerShell 명령을 실행한다. 기존 결과를 쓰거나 변경하지 않으며 aggregate count만 출력한다. source helper를 import하지 않는다.

```powershell
@'
import csv, hashlib, json
from pathlib import Path
from collections import defaultdict, Counter
r=Path.cwd(); c=r/'_reports/cvpr_u_pilot_v1_001/cohort'
sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
lock=json.loads((c/'lock.json').read_text(encoding='utf8'))
assert sha(r/'u_patient_audit/prepare_cohort.py')==lock['selection_code_sha256']
assert sha(c/'evaluation_images.csv')==lock['files']['evaluation_images.csv']
def h(*v):
    return hashlib.sha256('|'.join(map(str,(lock['selection_salt'],)+v)).encode()).hexdigest()
def seed(*v): return int(h(*v)[:15],16)
with (c/'evaluation_images.csv').open(encoding='utf-8-sig',newline='') as f:
    rows=list(csv.DictReader(f))
patients=defaultdict(list)
for row in rows: patients[row['patient_id']].append(row)
assert list(patients)==sorted(patients,key=lambda p:h('patient-cohort',p))
assert all(len({r['patient_sex'] for r in v})==1 for v in patients.values())
assert all(len({r['target_patient'] for r in v})==1 for v in patients.values())
report=[]
for role in ('fit','selection','calibration','test'):
    selected={p:v for p,v in patients.items() if v[0]['eval_role']==role}
    strata=defaultdict(list)
    for p,v in selected.items():
        strata[(v[0]['target_patient'],v[0]['patient_sex'])].append(p)
    pairs=[]; leftovers=[]
    for key,group in sorted(strata.items()):
        group=sorted(group,key=lambda p:h('assignment-order',role,p))
        pairs.extend(zip(group[:len(group)//2*2:2],group[1:len(group)//2*2:2]))
        if len(group)%2: leftovers.append(group[-1])
    leftover_count=len(leftovers)
    leftovers.sort(key=lambda p:h('leftover',role,p))
    pairs.extend(zip(leftovers[::2],leftovers[1::2]))
    expected={}
    for p,q in pairs:
        if seed('assignment-coin',role,p,q)%2: p,q=q,p
        expected[p]='A'; expected[q]='B'
    assert len(expected)==len(selected)
    mismatches=sum(any(r['assignment_group']!=expected[p] for r in v)
                   for p,v in selected.items())
    assert mismatches==0
    counts=Counter((v[0]['target_patient'],v[0]['patient_sex'],v[0]['assignment_group'])
                   for v in selected.values())
    report.append(dict(role=role,patients=len(selected),pairs=len(pairs),
        leftovers=leftover_count,mismatches=mismatches,
        strata=[dict(target_patient=s[0],sex=s[1],A=counts[s+('A',)],B=counts[s+('B',)])
                for s in sorted(strata)]))
print(json.dumps(dict(status='PASS_DETERMINISTIC_ASSIGNMENT_RECONSTRUCTION_ONLY',
    report=report,new_fitting=False,new_target_calls=0),indent=2))
'@ | & .\base_gate\.venv\Scripts\python.exe -X utf8 -
```
