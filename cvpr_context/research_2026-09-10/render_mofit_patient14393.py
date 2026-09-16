"""Render completed, verified one-patient measurements without choosing a score."""
from pathlib import Path
import csv
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent.parent / 'code_working/_reports/cvpr_u_pilot_v1_001/baseline_screen_20260915'
ANALYSIS = SOURCE / 'mofit_patient14393_two_image_analysis_v1'
ARTIFACT = ROOT / 'mofit_two_image_results_artifacts/v1'
LABELS = {'h': 'h: MoFit core', 'negative_u': '−u: null 원시 loss',
          'negative_v': '−v: 식8의 보조 차이', 'negative_u_plus_v': '−(u+v): VLM 조건부 원시 loss'}

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))

def main():
    data, provenance = read(ANALYSIS / 'analysis.json'), read(ANALYSIS / 'provenance.json')
    assert data['status'] == 'PASS_FIXED_SCALAR_AND_POOLING_ARITHMETIC'
    assert data['records'] == 8 and data['independent_patients'] == 1
    assert not data['AUC_computed'] and not data['gamma_or_final_score_selected']
    for name, expected in provenance['output_sha256'].items():
        assert sha(ANALYSIS / name) == expected
    for name, expected in provenance['input_sha256'].items():
        assert sha(Path(name)) == expected
    dirs = [SOURCE / n for n in ['mofit_medical_full_v1', 'mofit_medical_target2_full_v1', 'mofit_medical_remaining_pair_v1']]
    executions = [read(p / 'execution.json') for p in dirs]
    for p in dirs:
        v = read(p / 'verification.json')
        assert v['status'].startswith('PASS_')
        assert v['results_sha256'] == sha(p / 'results.json')
    ARTIFACT.mkdir(parents=True, exist_ok=False)
    for name in ['analysis.json', 'protocol.json', 'provenance.json', 'image_scores.csv', 'aggregates.csv', 'target_differences.csv']:
        shutil.copyfile(ANALYSIS / name, ARTIFACT / name)
    clarification = ROOT / 'spec_sources/mofit_score_naming_clarification_20260915.md'
    rows = data['original_huv']
    order = {i: k for k, i in enumerate(['00014393_002.png', '00014393_006.png', '00014393_001.png', '00014393_004.png'])}
    rows = sorted(rows, key=lambda r: (r['model'], order[r['image_id']]))
    lines = ['## 6. 같은 환자의 E2/U2 집계까지 실제 완료', '',
        '최초 E002/U001에 나머지 E006/U004를 추가해 **환자1명·영상4장·두 target의8 records**를 완료했다. 각 영상·target에서 동일1,000+300회로 독립 최적화한 뒤, 사전 고정한 네 점수의 mean/max를 모두 계산했다. 환자군 AUC·오탐률·개인 인과효과의 추정은 아니다.', '',
        '| target | 영상 | 역할 | h | u | v |', '|---|---|---|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['model']} | {r['image_id']} | {r['scenario']} | {r['h']:.10f} | {r['u']:.10f} | {r['v']:.10f} |")
    lines += ['', '표의 차이는 **참여 target M1 − 비참여 target M2**다. 양수는 이 고정 사례의 점수 방향이며 참여의 인과효과나 검출 성공 판정이 아니다. `max`는 각 영상에 고정 부호를 적용한 뒤 계산했다.', '',
        '| 점수 | 사진 조건·집계 | M1 | M2 | M1−M2 |', '|---|---|---:|---:|---:|']
    differences = data['target_differences']
    assert len(differences) == 32
    for score, label in LABELS.items():
        for scenario in ['E', 'U']:
            for pool in ['mean', 'max']:
                r = next(x for x in differences if x['score'] == score and x['scenario'] == scenario and x['pool'] == pool)
                lines.append(f"| {label} | {scenario} {pool} | {r['participant_score']:.10f} | {r['nonparticipant_score']:.10f} | {r['participant_minus_nonparticipant']:+.10f} |")
    latest = executions[-1]
    counts = {key: sum(e['counts'][key] for e in executions) for key in executions[0]['counts']}
    lines += ['',
        f"추가4건 실행기는 **{latest['total_seconds']:.3f}초**, 계측 구간은{latest['measurement_seconds']:.3f}초였다. 최초두실행까지 full8건의 실행기 합은{sum(e['total_seconds'] for e in executions):.3f}초, 총{counts['unet_forward_examples']:,}F/{counts['unet_backward_calls']:,}B이며 benchmark·FD는 별도다.", '',
        '각 원시 실행의 저장산술 검산 PASS 뒤 기존4건과 추가4건을 결속했다. 이미지 점수32행·집계32행·target 차이32행을 별도 산술 항등식으로 대조했다. 이 집계 검산은 원시 신경망 역전파의 재실행이 아니다.', '',
        '[사진별 점수 CSV](mofit_two_image_results_artifacts/v1/image_scores.csv) · [전체 집계 CSV](mofit_two_image_results_artifacts/v1/aggregates.csv) · [사진·집계의 전체 target 차이 CSV](mofit_two_image_results_artifacts/v1/target_differences.csv) · [출처 manifest](mofit_two_image_results_artifacts/v1/manifest.json)', '',
        '**명칭 정정:** 동결 분석 계약은 `L_VLM`을 원시 조건부 loss의 이름으로 사용해 논문 식8의 기호와 혼동되는 설명을 포함했다. 실제 `−v`와 `−(u+v)` 계산은 사전 선언대로다. 위 표에서 `−v`는 식8의 보조 차이, `−(u+v)`는 별도 원시 조건부 loss로 구분했다. [원계약을 보존한 명칭 정정](spec_sources/mofit_score_naming_clarification_20260915.md).', '',
        '**남은 범위:** 의료 Eq9의 fusion·보정, 정책 학습에서 분리한 환자군 평가, 개인 참여에 특이적인 실패 원인 설명은 미완료다. 한 환자의 E2/U2 집계를 마친 사실을2번 전체 완료나 새 설계의 근거로 올리지 않는다.', '']
    section = '\n'.join(lines)
    standalone = section.replace('(mofit_two_image_results_artifacts/v1/', '(').replace('(spec_sources/', '(../../spec_sources/')
    (ARTIFACT / 'comparison_section.md').write_text(standalone, encoding='utf-8')
    manifest = {'schema': 'mofit-one-patient-display-artifacts/v1', 'renderer_sha256': sha(Path(__file__)),
                'source_analysis_sha256': sha(ANALYSIS / 'analysis.json'),
                'files': {p.name: sha(p) for p in ARTIFACT.iterdir() if p.is_file()},
                'score_naming_clarification_source': str(clarification),
                'score_naming_clarification_sha256': sha(clarification),
                'records': 8, 'independent_patients': 1, 'new_score_selection': False}
    (ARTIFACT / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    report = ROOT / 'MOFIT_MEDICAL_RESULTS.md'
    text = report.read_text(encoding='utf-8')
    marker = '## 6. 이번에 완료한 범위와 다음 한 작업'
    assert marker in text
    report.write_text(text.split(marker)[0] + section, encoding='utf-8')
    print(json.dumps({'status': 'VERIFIED_EIGHT_RECORD_DISPLAY_RENDERED', 'image_rows': 8, 'pool_rows': 16,
                      'files': len(manifest['files']), 'new_score_selection': False}, ensure_ascii=False))

if __name__ == '__main__':
    main()
