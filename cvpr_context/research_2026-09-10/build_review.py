"""Build the audited review ledger and offline HTML from authored review notes.

No inferred full-paper/completed-replication flag; reading scopes remain explicit.
"""
from pathlib import Path
from collections import Counter
import csv,html,json,re,hashlib
import markdown
from bs4 import BeautifulSoup

ROOT=Path(__file__).resolve().parent
PLAN=json.loads((ROOT/'comparison_plan.json').read_text(encoding='utf-8'))
ATTACK={r['id'] for r in PLAN['attack_candidates']}
TRAIN={r['id'] for r in PLAN['training_candidates']}
UNITS=set(PLAN['unit_baselines'])
ROLES={'attack':'공격 비교 후보','training':'생성 학습 비교 후보','unit':'보호 단위·회계',
       'medical':'의료·identity·복제·방어','federated':'연합·개인화 확장',
       'alternative':'과거 생성·효율 방향','thesis':'졸논 증빙·워터마크','background':'벤치마크·종설'}
STATUSES={'selected':'PDF 선택 절 검토','partial':'공식 본문 일부만 확인','abstract':'저자 초록만 확인'}
CSS='''body{margin:0;background:#f5f6f8;color:#202a38;font:16px/1.65 system-ui,"Malgun Gothic",sans-serif}main{max-width:1120px;margin:auto;padding:28px 20px 80px}h1{font-size:29px;line-height:1.3}h2{font-size:23px;margin-top:36px}a{color:#174ca1;overflow-wrap:anywhere}header,.panel{background:white;border:1px solid #d7dce4;border-radius:12px;padding:22px;margin-bottom:18px}header p{margin:10px 0}.tag{display:inline-block;border-radius:5px;background:#eaf0f9;padding:2px 7px;margin:3px 5px 3px 0;font-size:13px}label{display:block;font-size:14px;font-weight:600}select,input,button{font:inherit;padding:9px;border:1px solid #aeb8c6;border-radius:6px;background:white;max-width:100%;box-sizing:border-box}select,input{width:100%}.filters{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.search{grid-column:1/-1}button{cursor:pointer;margin-right:8px;color:#173d75}.card{background:white;border:1px solid #d7dce4;border-radius:9px;margin:12px 0;padding:16px}.card[hidden]{display:none}summary{cursor:pointer;overflow-wrap:anywhere}.title{font-weight:650}.meta{font-size:14px;color:#536176}.review{border-top:1px solid #e4e7ed;margin-top:12px;padding-top:10px}.review p{margin:12px 0}.links{display:flex;gap:16px;flex-wrap:wrap;margin-top:10px}table{border-collapse:collapse;display:block;overflow-x:auto;font-size:14px}th,td{border:1px solid #ccd3dd;padding:9px;text-align:left;vertical-align:top}th{background:#eef2f7}code,pre{background:#eef1f5;border-radius:3px;overflow-wrap:anywhere}pre{padding:12px;white-space:pre-wrap}#count{font-weight:650}.notice{border-left:4px solid #355f99;padding-left:14px}.empty{padding:24px;text-align:center}@media(max-width:800px){.filters{grid-template-columns:1fr 1fr}}@media(max-width:480px){.filters{grid-template-columns:1fr}main{padding:16px 12px}h1{font-size:24px}}'''

CSS += '\nimg{max-width:100%;height:auto}'

def role(pid):
    if pid in ATTACK:return 'attack'
    if pid in TRAIN:return 'training'
    if pid in UNITS:return 'unit'
    if pid.startswith('A'):return 'medical'
    return {'C':'federated','D':'alternative','E':'thesis'}.get(pid[0],'background')

def write_csv(path,rows,fields):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)

def html_doc(title,body):
    return '<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+html.escape(title)+'</title><style>'+CSS+'</style></head><body><main>'+body+'</main></body></html>'

def convert(md,from_review=False):
    soup=BeautifulSoup(markdown.markdown(md,extensions=['tables','fenced_code','footnotes']),'html.parser')
    for a in soup.find_all('a',href=True):
        href=a['href']
        if from_review and href.startswith('../'):href=href[3:]
        if re.fullmatch(r'reviews/[A-Z]\d\d\.md',href):href='index.html#'+Path(href).stem
        href={'REVIEW_REPORT.md':'report.html','BASELINE_PREPARATION.md':'baseline_preparation.html','PATIENT_AUDIT_RATIONALE.md':'rationale.html','RESEARCH_PURPOSE.md':'purpose.html','EXPERIMENT_DESIGN.md':'design.html','THESIS_CVPR_ALIGNMENT.md':'thesis_alignment.html','rationale_sources/R01.md':'rationale_sources/R01.html','rationale_sources/R02.md':'rationale_sources/R02.html'}.get(href,href)
        if href=='AUDITOR_FRAMING_REVIEW.md':href='auditor_framing_review.html'
        if href=='BASELINE_SCOPE_REVIEW.md':href='baseline_scope_review.html'
        if href=='U_EXECUTION_PLAN.md':href='u_execution_plan.html'
        if href=='U_EXECUTION_STATUS.md':href='u_execution_status.html'
        if href=='U_EIGHT_PATIENT_ANALYSIS.md':href='u_eight_patient_analysis.html'
        if href=='U_PILOT_REVIEW.md':href='u_pilot_review.html'
        if href=='U_VERIFICATION_RESULTS.md':href='u_verification_results.html'
        if href=='U_BASELINE_SCREEN.md':href='u_baseline_screen.html'
        if href=='METHOD_PREMISE_AUDIT_2026-09-14.md':href='method_premise_audit.html'
        if href=='PFAMI_COMPARISON_PROTOCOL.md':href='pfami_comparison_protocol.html'
        if href=='PFAMI_COMPARISON_RESULTS.md':href='pfami_comparison_results.html'
        if href=='PAIRED_SCORE_AUDIT.md':href='paired_score_audit.html'
        if href=='RESEARCH_FRAMEWORK.md':href='research_framework.html'
        if href=='PATIENT_BASELINE_SPEC.md':href='patient_baseline_spec.html'
        if href=='CDI_KERNEL_RESULTS.md':href='cdi_kernel_results.html'
        if href=='CDI_U_COMPARISON_RESULTS.md':href='cdi_u_comparison_results.html'
        if href=='CDI_EU_COMPARISON_RESULTS.md':href='cdi_eu_comparison_results.html'
        if href=='PROMPT_CONDITION_RESULTS.md':href='prompt_condition_results.html'
        if href=='MOFIT_MEDICAL_RESULTS.md':href='mofit_medical_results.html'
        if href=='STAGE2_MEASUREMENT_AUDIT_20260915.md':href='stage2_measurement_audit.html'
        if href=='STAGE2_CROSS_PATIENT_INTERVENTION_20260915.md':href='stage2_cross_patient_intervention.html'
        if href=='STAGE2_DESIGN_GAP_RESET_20260915.md':href='stage2_design_gap_reset.html'
        if href=='PAPER_LEVEL_REDESIGN_20260915.md':href='paper_level_redesign.html'
        if href=='PROTECTION_OBSERVATION_CLAIM_20260915.md':href='protection_observation_claim.html'
        if href=='CLIPPING_OBSERVATION_LOCAL_RESULTS_20260915.md':href='clipping_observation_local_results.html'
        if href=='PATIENT_GAP_WORK_INVENTORY_20260916.md':href='patient_gap_work_inventory.html'
        if href=='TWO_TRACK_PAPER_DESIGNS_20260916.md':href='two_track_paper_designs.html'
        if href=='TWO_TRACK_DEEPER_CHECK_20260916.md':href='two_track_deeper_check.html'
        if href=='TWO_TRACK_OPERATION_REDESIGN_20260916.md':href='two_track_operation_redesign.html'
        if href=='TRACK1_CAPACITY_PROTOCOL_20260916.md':href='track1_capacity_protocol.html'
        if href=='TRACK1_CAPACITY_RESULTS_20260916.md':href='track1_capacity_results.html'
        if href=='TRACK1_PATIENT_DP_COMPARISON_PLAN_20260916.md':href='track1_patient_dp_comparison_plan.html'
        if href=='TRACK1_PATIENT_DP_PROTOCOL_20260916.md':href='track1_patient_dp_protocol.html'
        if href=='TRACK1_PATIENT_DP_RESULTS_20260916.md':href='track1_patient_dp_results.html'
        if href=='REALISTIC_RESEARCH_PLAN_20260916.md':href='realistic_research_plan.html'
        a['href']=href
    return str(soup)

def stage_notice(name):
    corrected = {'METHOD_PREMISE_AUDIT_2026-09-14.md','PFAMI_COMPARISON_PROTOCOL.md','PFAMI_COMPARISON_RESULTS.md','PAIRED_SCORE_AUDIT.md'}
    if name not in corrected:
        return ''
    return '<div class="panel notice"><strong>단계 번호 정정: 이 보고서의 작업은 전체 계획의 2번입니다.</strong><br>사용자가 고정한 네 단계에서 1번은 문제·목표 고정으로 완료됐고, 현재는 2번의 기존 방법 실패 조건·원인 검토 중입니다. 아래 과거 원문의 “1번/1단계” 표기는 assistant의 번호 오류입니다. 부분 진단 완료를 2번 전체 완료나 방향 폐기 판단으로 읽지 않습니다. <a href="research_framework.html">고정 연구틀·완료/미완료·다음 작업</a></div>'

def main():
    registry=json.loads((ROOT/'registry.json').read_text(encoding='utf-8'))
    codes={r['id']:r for r in json.loads((ROOT/'code_acquisition.json').read_text(encoding='utf-8'))}
    records=[]
    for r in registry:
        pid=r['id']
        if pid.startswith('F'):continue
        path=ROOT/'reviews'/(pid+'.md')
        assert path.exists(),pid
        memo=path.read_text(encoding='utf-8')
        status='abstract' if pid=='C02' else 'partial' if pid in {'A20','A21','B17'} else 'selected'
        venue=r['venue'] if pid!='X11' else '공개 preprint · 게재 venue 미확인'
        v=venue.lower()
        main_venue=any(x in v for x in ['cvpr','iccv','iclr','neurips','icml','eccv','aaai','usenix','ndss','ccs','colm','aistats','miccai','wacv']) and not any(x in v for x in ['workshop','tpdp','short paper','dgm4miccai'])
        scope=' / '.join(p for p in memo.split('\n\n') if any(x in p for x in ['검토:','검토 범위','검토 수준','검토자의']))
        rec=dict(id=pid,title=r['title'],alias=r.get('alias',pid),venue=venue,
                 role=role(pid),role_label=ROLES[role(pid)],review_status=status,review_status_label=STATUSES[status],
                 reading_scope=scope or '개별 메모의 페이지별 확인 범위 참조',source_url=r['source_url'],
                 pdf_url=r.get('pdf_url',''),local_pdf=('pdfs/'+pid+'.pdf') if (ROOT/'pdfs'/(pid+'.pdf')).exists() else '',
                 legacy35=bool(r.get('latest18')),new=pid.startswith('X'),main_venue=bool(main_venue),
                 review_file='reviews/'+pid+'.md',memo_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                 code_status=codes.get(pid,{}).get('status','not_acquired'),code_commit=codes.get(pid,{}).get('commit',''),
                 replay_status='public_packet_only' if pid in {'A12','X02'} else 'not_reproduced',
                 whole_paper_read_claim=False,full_experiment_reproduced=False,memo=memo)
        records.append(rec)
    assert len(records)==79 and sum(x['legacy35'] for x in records)==35
    assert Counter(x['review_status'] for x in records)==Counter(selected=75,partial=3,abstract=1)
    (ROOT/'review_ledger.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
    fields=[k for k in records[0] if k!='memo']
    write_csv(ROOT/'review_ledger.csv',records,fields)
    lookup={r['id']:r for r in records}
    comparisons=[]
    for kind,entries in [('attack',PLAN['attack_candidates']),('training',PLAN['training_candidates']),('unit',[dict(id=i,method=lookup[i]['alias'],priority='privacy-unit algorithm prior') for i in PLAN['unit_baselines']])]:
        for x in entries:
            rec={**lookup[x['id']],**x,'comparison_type':kind};comparisons.append(rec)
    write_csv(ROOT/'comparison_matrix.csv',comparisons,['id','method','comparison_type','venue','priority','access','auxiliary','cost','adaptation','control','code_status','code_commit','replay_status','source_url','local_pdf','review_file'])
    esc=html.escape
    stats=Counter(x['role'] for x in records)
    head='''<header><h1>CVPR 환자 DP 의료영상 · 직접 검토 장부</h1><p><strong>79편 개별 기록</strong> · 기존 67 + 추가 12 · 2026-09-10</p><p class="notice">현재 방향은 강한 기존 방법과 대조할 연구 후보다. CVPR 신규성·성공 가능성이 입증된 상태는 아니다.</p><p>PDF 선택 절 검토 75편 · 공식 본문 일부 3편 · 저자 초록만 1편. 모든 페이지 정독이나 전체 실험 재현으로 표시하지 않는다.</p><p><a href="report.html">기여도 재판정·필수 비교 읽기</a> · <a href="baseline_preparation.html">구현 준비와 실제 실행 결과</a> · <a href="comparison_matrix.csv" download>비교군 CSV</a> · <a href="review_ledger.csv" download>79편 장부 CSV</a></p><p>공개 점수 재계산: CLiD와 Tracing the Roots. 의료 환자 공격 성공 검증은 아직 아니다.</p></header>'''
    head = '<section class="panel"><h2>현재2번: 환자 감사의 간극을 위한 조사·설계의 도달점</h2><p><a href="patient_gap_work_inventory.html"><strong>2026-09-16: 무엇이 됐고 무엇이 남았는가</strong></a> · <a href="paper_level_redesign.html">졸논·CVPR 큰 구조</a> · <a href="research_framework.html">고정 연구틀</a></p><p>가까운 선행의 가정·기여 경계와 기존 공격의 E/U 비교, U 적응의 관측까지 확보했습니다. 보유 자료의 대표성·보조정보·보호 평가 적용 범위는 후보 질문이며, 중요한 빈틈과 해결 설계를 한 주장으로 연결하는 작업은 미완료입니다.</p><p><a href="clipping_observation_local_results.html">최근 clipping 보조 관측</a>은 보존합니다. 이 관측을 환자 위험 평가의 핵심 검증이나 자동 후속 주제로 두지 않습니다. 이번 현황 정리의 새 실험은0입니다.</p></section>' + head
    head = '<section class="panel"><h2>현재 두 방향: 보호 설계·효율 / 보호 평가 개선</h2><p><a href="two_track_deeper_check.html"><strong>최신 심화 검토: 선행 환원·두 v0의 CPU 비교·실패 원인</strong></a> · <a href="two_track_paper_designs.html">최초 설계</a> · <a href="research_framework.html">현재 연구틀</a></p><p>두 목표는 유지합니다. 현재 v0는 방향1의 직접 선행 중복과 계산 효율 문제, 방향2의 배분 비용·보류 문제를 확인해 의료·본학습 확대를 보류했습니다. 우리 초안의 검증 결과이며 기존 SOTA 전체의 실패나 두 연구 목표의 불가능성을 뜻하지 않습니다.</p><p>현재2번의 문헌·설계 검토입니다. 사전 고정 CPU 비교와 독립 검산을 마쳤고 새 GPU·의료학습은0입니다. 아래 환자 E/U·clipping 중심 안내는 이전 작업 이력입니다.</p></section>' + head
    head = '<section class="panel"><h2>현재2번: 보호 학습 연산과 보호 평가의 재설계</h2><p><a href="two_track_operation_redesign.html"><strong>2026-09-16 최신: 환자별 보호 통계로 생성 보정층 학습 / 증류 보호의 수리 선택 평가</strong></a> · <a href="research_framework.html">현재 연구틀</a></p><p>두 방향을 유지합니다. 방향1은 동결 모델의 작은 잔차층과 환자별 충분통계 보호로 구체화했습니다. 반복 역전파 제거와 표현력의 대가를 구분하고, 같은 head의 DP-SGD도 필수 비교로 둡니다. 방향2는 teacher·student 데이터 경로에 맞는 보호 수리 선택이 질문이며, 가까운 선행과 겹쳐 추가 기여가 미확정입니다.</p><p>이번 원문·수식·독립 검토에서 새 GPU·학습·공격은0입니다. 성능·신규성 검증 완료가 아닙니다. 아래 v0 CPU 결과와 E/U·clipping 안내는 이전 작업 이력입니다.</p></section>' + head
    head = '<section class="panel"><h2>방향1 첫 실제 결과: 작은 보정층의 적응 능력 확인</h2><p><a href="track1_capacity_results.html"><strong>2026-09-16 결과·의미·독립 검산</strong></a> · <a href="track1_capacity_protocol.html">고정 실행 조건</a> · <a href="research_framework.html">현재 연구틀</a></p><p>학습80명/평가40명·각4장. 비DP 시간·공간64차원 보정층의 개발평가 MSE가 base대비3.1384%, 단순16차원대비 .4467% 감소했습니다. 두 비교40/40이며 원시·통계 독립 검산PASS입니다. 단순층이 전체 감소의86.15%를 이미 얻었습니다.</p><p>본 추출5분15초, 총3850F/0B. 작은 함수족의 적응 근거이며 DP 적용 후 효용·생성 품질·새 기여는 미확인입니다. 다음은 같은 특징의 환자DP 통계 보호와 DP-SGD를 공정하게 비교하는 조건 구체화입니다.</p></section>' + head
    head = '<section class="panel"><h2>방향1 최신 실제 결과: 환자DP 비교와 손실 원인</h2><p><a href="track1_patient_dp_results.html"><strong>64개 DP 보정층·20개 대조·독립 검산</strong></a> · <a href="track1_patient_dp_protocol.html">공개 보정과 실행 조건</a></p><p>ε8/δ1e-5에서 full64 DP-SGD는 기반 모델보다 MSE2.9258%, 일회 SSP는0.4181% 감소했습니다. 현재 SSP는 같은 head의 SGD보다 낮은 성능이며, 공개32명만의 회귀도 넘지 못했습니다. 선택된 floor가 SSP의 개선을 크게 깎는 것은 잡음 없는 대조로 확인했습니다.</p><p>본 비교20.540초·새backbone0F/0B, 독립 검산41,812항목PASS. 다음은 공개 기준을 활용하는 잔차 보호와 같은 공개 초기화를 받는 강한 대안의 대조입니다. 새 기여·생성 품질은 미검증입니다. 아래 준비·미실행 설명은 이전 기록 당시 상태입니다.</p></section>' + head
    head = '<section class="panel"><h2>현재 계획: 기존 원리를 활용한 생성 적응의 기여 검토</h2><p><a href="realistic_research_plan.html"><strong>2026-09-16 현실적인 계획·판정 기준·예상 시간</strong></a> · <a href="research_framework.html">현재 연구틀</a></p><p>원리의 선행 존재만으로 의료 diffusion·환자DP 적용 기여를 배제하지 않습니다. 생성 적응 구성과 추가 보호 solver의 기여를 구분합니다. 다음은 공개+사적 비DP 대조와 현재 보정층의 실제 생성 연결, 기존 RAD-DINO/BioViL-T 평가기와 환자 분리 참조를 연결하며 구현·검토·분석 포함 총2.5–5시간으로 조정했습니다.</p><p>이번은 계획 재수립으로 새 실험0입니다. 실제 생성 효용·전체 비용·최종 기여는 미확정이며 작은 이미지 grid만으로 확정하지 않습니다. 아래 최신/다음 안내는 이전 시점의 이력입니다.</p></section>' + head
    head += '<p class="notice"><a href="purpose.html"><strong>이전 연구 목적: 환자 보호·보장 환산·재사용 비용과 공격의 역할</strong></a><br>현재 범위와 다음 작업은 위 두 방향 설계와 연구틀을 따릅니다.</p>'
    head += '<p class="notice"><a href="rationale.html"><strong>환자 집합 감사법 후보의 상세 근거와 반증 조건</strong></a><br>상관·사진 수·기존 방법·DP 보장의 관계. 추가 대조 2편은 본문에서 별도로 확인할 수 있습니다.</p>'
    head += '<p class="notice"><a href="design.html"><strong>설계 v0.1: 교차 사진 반응 감사 → DP 보호·효용·비용 비교</strong></a><br>첫 알고리즘 후보와 비교 계획의 원문입니다. 후속 구현·실행·검증 상태는 아래 최신 실행 기록을 따릅니다.</p>'
    head += '<p><a href="thesis_alignment.html"><strong>CVPR 추진 시 졸업논문의 중심·권장 목차·공통 실증</strong></a></p>'
    head += '<p><a href="runtime_review.html"><strong>실험별 시간 검토: Spark 4대·baseline·공통 학습·U 추가 실험</strong></a></p>'
    head += '<p><a href="auditor_framing_review.html"><strong>제한 접근 감사자 검토: ICO·HIPAA 근거, white-box 조건과 E/U 구분</strong></a></p>'
    head += '<p><a href="baseline_scope_review.html"><strong>U 조건 중심 비교군 검토: 직접 비교·변형·신규성 경계와 공정한 평가</strong></a></p>'
    head += '<p class="notice"><a href="u_execution_plan.html"><strong>실행 계획 원문: U 분할 → 최소 구현·두 모델 진단 → 강한 비교 → DP 보호 비교</strong></a><br>2026-09-14 계획 이력입니다. 분할과 첫 학습·감사 실행은 후속 기록에서 완료됐으며, 원문 미실행 표기는 작성 당시 상태입니다.</p>'
    head += '<p class="notice"><a href="u_execution_status.html"><strong>실행 현황: 실제 사진 선정·분할 검증·U 파일럿 학습과 감사 코드</strong></a><br>2026-09-14 사용자 승인 후 실행 기록. 계획안의 미실행 표기보다 이 현황을 우선하며, 성능·기여 판정은 별도로 표시합니다.</p>'
    head += '<p class="notice"><a href="u_verification_results.html"><strong>실제 재검증: 입력 미분·전체 U8 재현·E/U 대조·정밀도</strong></a><br>빠진 검사를 실제 GPU에서 수행한 결과와 수정, 예상·실측 시간을 확인할 수 있습니다. 계산 검증과 공격 성능을 구분합니다.</p>'
    head += '<p class="notice"><a href="u_baseline_screen.html"><strong>앞선 실행: 기존 SecMI 방법의 E/U 48명 진단</strong></a><br>기존 개발 8명과 방법 선택 40명을 분리하고, 두 모델에서 고정된 기존 공격을 실행했습니다. 원시 결과 재검산과 당시 사전 기준에 따른 판단입니다.</p>'
    head += '<p class="notice"><a href="method_premise_audit.html"><strong>후속 원문·설계 검토: MoFit·CDI·PFAMI와 현재 U 후보</strong></a><br>기존 방법의 실제 작동 조건과 미검증 전제를 대조했습니다. 교차 사진 반응의 관측과 membership 추가 정보의 입증을 구분하며, 현재 R 확대 보류를 유지합니다.</p>'
    head += '<p class="notice"><a href="pfami_comparison_results.html"><strong>2번의 최근 GPU 진단: PFAMI 고정 비교·독립 검산</strong></a><br>선택 40명·480건·9,600F를 완료했습니다. U AUC 0.3550/0.6575이며 정규화의 추가 이득은 미확인입니다. 이 부분 결과를 2번 전체 완료나 방향 폐기의 근거로 확대하지 않습니다. <a href="pfami_comparison_protocol.html">실행 전 계약</a></p>'
    head += '<p class="notice"><a href="paired_score_audit.html"><strong>최신 분석: 두 모델의 공통 순위와 참여 방향 변화</strong></a><br>기존 40명의 저장 점수로 반대 AUC의 구조를 확인했습니다. 새 GPU 실행 없이 공통 성분·점수 변화·환자 간 순위 변화를 구분하며, 개인 인과 효과나 새 공격의 성능으로 해석하지 않습니다.</p>'
    controls='''<section class="panel"><div class="filters"><label>목록 범위<select id="scope"><option value="all">전체 79편</option><option value="legacy">기존 관련 67편</option><option value="legacy35">기존 최신 판정 35편</option><option value="new">이번 추가 12편</option><option value="direct">공격·생성 비교 후보 26편</option></select></label><label>역할<select id="role"><option value="all">모든 역할</option>'''
    controls+=''.join('<option value="'+k+'">'+esc(v)+' ('+str(stats[k])+')</option>' for k,v in ROLES.items())
    controls+='''</select></label><label>확인 수준<select id="status"><option value="all">모든 확인 수준</option><option value="selected">PDF 선택 절 (75)</option><option value="partial">공식 본문 일부 (3)</option><option value="abstract">초록만 (1)</option><option value="replay">공개 packet 재계산 (2)</option></select></label><label>발표 형식<select id="venue"><option value="all">모든 발표 형식</option><option value="main">메인 학회 표기</option><option value="other">저널·워크샵·공개본 등</option></select></label><label class="search">논문명·방법·검토 내용 검색<input id="search" type="search" placeholder="예: MoFit, CLiD, 환자 평균, FPR" autocomplete="off"></label></div><p id="count" aria-live="polite"></p><button id="reset" type="button">필터 초기화</button><button id="expand" type="button">표시된 검토 펼치기</button><button id="collapse" type="button">모두 접기</button><p class="meta">비교 후보 수는 전부 실행할 의무 목록이나 같은 문제의 SOTA 순위가 아니다. 접근 권한과 주장에 맞는 비교군을 선택한다.</p></section>'''
    cards=[]
    for r in records:
        body='\n'.join(r['memo'].splitlines()[1:])
        links='<a href="'+esc(r['source_url'],quote=True)+'">공식 원문</a>'
        if r['local_pdf']:links+=' <a href="'+r['local_pdf']+'" download="'+r['id']+'.pdf">저장된 PDF 다운로드</a> <a href="'+r['local_pdf']+'">PDF 열기</a>'
        elif r['pdf_url']:links+=' <a href="'+esc(r['pdf_url'],quote=True)+'">외부 PDF · 접근 제한 가능</a>'
        links+=' <a href="'+r['review_file']+'">검토 Markdown</a>'
        tags=''.join('<span class="tag">'+esc(x)+'</span>' for x in [r['role_label'],r['review_status_label'],r['venue']])
        cards.append('<details class="card" id="'+r['id']+'"><summary><span class="title">'+r['id']+' · '+esc(r['title'])+'</span><br>'+tags+'</summary><div class="links">'+links+'</div><div class="review">'+convert(body,True)+'</div></details>')
    data=[{k:r[k] for k in ['id','role','review_status','legacy35','new','main_venue','replay_status']}|{'search':(r['title']+' '+r['alias']+' '+r['memo']).lower()} for r in records]
    script='''<script>const records=DATA;const fields=['scope','role','status','venue','search'];const el=id=>document.getElementById(id);function filter(){const f=Object.fromEntries(fields.map(k=>[k,el(k).value]));let n=0;for(const r of records){const a=f.scope==='all'||f.scope==='legacy'&&!r.new||f.scope==='legacy35'&&r.legacy35||f.scope==='new'&&r.new||f.scope==='direct'&&['attack','training'].includes(r.role);const b=f.role==='all'||r.role===f.role;const c=f.status==='all'||r.review_status===f.status||f.status==='replay'&&r.replay_status==='public_packet_only';const d=f.venue==='all'||f.venue==='main'&&r.main_venue||f.venue==='other'&&!r.main_venue;const show=a&&b&&c&&d&&r.search.includes(f.search.trim().toLowerCase());el(r.id).hidden=!show;if(show)n++}el('count').textContent=n+' / 79편 표시';el('empty').hidden=n!==0;return n}function reset(){fields.forEach(k=>el(k).value=k==='search'?'':'all');filter()}fields.forEach(k=>el(k).addEventListener(k==='search'?'input':'change',filter));el('reset').onclick=reset;el('expand').onclick=()=>records.forEach(r=>{if(!el(r.id).hidden)el(r.id).open=true});el('collapse').onclick=()=>records.forEach(r=>el(r.id).open=false);filter();function anchor(){const id=decodeURIComponent(location.hash.slice(1));if(records.some(r=>r.id===id)){reset();el(id).open=true;el(id).scrollIntoView()}}window.addEventListener('hashchange',anchor);anchor();window.reviewFilter=filter;window.reviewReset=reset;</script>'''.replace('DATA',json.dumps(data,ensure_ascii=False).replace('<','\\u003c'))
    (ROOT/'index.html').write_text(html_doc('CVPR 환자 DP 원문 검토 79편',head+controls+''.join(cards)+'<p id="empty" class="empty" hidden>조건에 맞는 문헌이 없습니다.</p>'+script),encoding='utf-8')
    for name,out in [('REVIEW_REPORT.md','report.html'),('BASELINE_PREPARATION.md','baseline_preparation.html'),('PATIENT_AUDIT_RATIONALE.md','rationale.html'),('RESEARCH_PURPOSE.md','purpose.html'),('EXPERIMENT_DESIGN.md','design.html'),('THESIS_CVPR_ALIGNMENT.md','thesis_alignment.html'),('AUDITOR_FRAMING_REVIEW.md','auditor_framing_review.html'),('BASELINE_SCOPE_REVIEW.md','baseline_scope_review.html'),('U_EXECUTION_PLAN.md','u_execution_plan.html'),('U_EXECUTION_STATUS.md','u_execution_status.html'),('U_EIGHT_PATIENT_ANALYSIS.md','u_eight_patient_analysis.html'),('U_PILOT_REVIEW.md','u_pilot_review.html'),('U_VERIFICATION_RESULTS.md','u_verification_results.html'),('U_BASELINE_SCREEN.md','u_baseline_screen.html')]:
        body='<p><a href="index.html">← 전체 검토 장부</a></p>'+convert((ROOT/name).read_text(encoding='utf-8'))
        doc=html_doc(name,body)
        if out=='rationale.html':
            doc=doc.replace('</style>','body{background:white;color:#222}main{max-width:940px}p{margin:18px 0}h1{margin-bottom:30px}table{font-size:15px}.footnote{font-size:14px}@media print{main{max-width:none;padding:0}a{color:inherit}table{font-size:11px}}'+'</style>')
        (ROOT/out).write_text(doc,encoding='utf-8')
    audit_name='METHOD_PREMISE_AUDIT_2026-09-14.md'
    audit_body='<p><a href="index.html">← 전체 검토 장부</a></p>'+stage_notice(audit_name)+convert((ROOT/audit_name).read_text(encoding='utf-8'))
    (ROOT/'method_premise_audit.html').write_text(html_doc('기존 방법·현재 U 설계의 근거 대조',audit_body),encoding='utf-8')
    for name,out in [('PFAMI_COMPARISON_PROTOCOL.md','pfami_comparison_protocol.html'),('PFAMI_COMPARISON_RESULTS.md','pfami_comparison_results.html'),('PAIRED_SCORE_AUDIT.md','paired_score_audit.html'),('RESEARCH_FRAMEWORK.md','research_framework.html'),('PATIENT_BASELINE_SPEC.md','patient_baseline_spec.html'),('CDI_KERNEL_RESULTS.md','cdi_kernel_results.html'),('CDI_U_COMPARISON_RESULTS.md','cdi_u_comparison_results.html'),('CDI_EU_COMPARISON_RESULTS.md','cdi_eu_comparison_results.html'),('PROMPT_CONDITION_RESULTS.md','prompt_condition_results.html'),('MOFIT_MEDICAL_RESULTS.md','mofit_medical_results.html'),('STAGE2_MEASUREMENT_AUDIT_20260915.md','stage2_measurement_audit.html'),('STAGE2_CROSS_PATIENT_INTERVENTION_20260915.md','stage2_cross_patient_intervention.html'),('STAGE2_DESIGN_GAP_RESET_20260915.md','stage2_design_gap_reset.html')]:
        body='<p><a href="index.html">← 전체 검토 장부</a></p>'+stage_notice(name)+convert((ROOT/name).read_text(encoding='utf-8'))
        (ROOT/out).write_text(html_doc(name,body),encoding='utf-8')
    redesign_body='<p><a href="index.html">← 전체 검토 장부</a></p>'+convert((ROOT/'PAPER_LEVEL_REDESIGN_20260915.md').read_text(encoding='utf-8'))
    (ROOT/'paper_level_redesign.html').write_text(html_doc('졸논에서 CVPR까지: 목적 중심 연구 재설계',redesign_body),encoding='utf-8')
    claim_body='<p><a href="index.html">← 전체 검토 장부</a> · <a href="paper_level_redesign.html">전체 연구 구조</a></p>'+convert((ROOT/'PROTECTION_OBSERVATION_CLAIM_20260915.md').read_text(encoding='utf-8'))
    (ROOT/'protection_observation_claim.html').write_text(html_doc('보호 방식과 E/U 평가: 구체적 후보와 검증 설계',claim_body),encoding='utf-8')
    local_body='<p><a href="index.html">← 전체 검토 장부</a> · <a href="protection_observation_claim.html">후보 설계</a></p>'+convert((ROOT/'CLIPPING_OBSERVATION_LOCAL_RESULTS_20260915.md').read_text(encoding='utf-8'))
    (ROOT/'clipping_observation_local_results.html').write_text(html_doc('클리핑·Adam·E/U 최소 검증 결과',local_body),encoding='utf-8')
    inventory_body='<p><a href="index.html">← 전체 검토 장부</a> · <a href="research_framework.html">고정 연구틀</a></p>'+convert((ROOT/'PATIENT_GAP_WORK_INVENTORY_20260916.md').read_text(encoding='utf-8'))
    (ROOT/'patient_gap_work_inventory.html').write_text(html_doc('환자 감사 질문을 위한 조사·설계의 현재 도달점',inventory_body),encoding='utf-8')
    tracks_body='<p><a href="index.html">← 전체 검토 장부</a> · <a href="research_framework.html">현재 연구틀</a></p>'+convert((ROOT/'TWO_TRACK_PAPER_DESIGNS_20260916.md').read_text(encoding='utf-8'))
    tracks_html=html_doc('보호 설계와 보호 평가의 두 논문 후보',tracks_body)
    tracks_html=tracks_html.replace('</style>','body{background:white;color:#222}main{max-width:1020px}h2{border-top:1px solid #ddd;padding-top:22px}th{background:#f0f0f0}code,pre{border-radius:0;background:#f4f4f4}</style>')
    (ROOT/'two_track_paper_designs.html').write_text(tracks_html,encoding='utf-8')
    deeper_body='<p><a href="index.html">← 전체 검토 장부</a> · <a href="two_track_paper_designs.html">최초 설계</a></p>'+convert((ROOT/'TWO_TRACK_DEEPER_CHECK_20260916.md').read_text(encoding='utf-8'))
    (ROOT/'two_track_deeper_check.html').write_text(html_doc('두 방향 심화 검토: 선행·비용·CPU 결과',deeper_body),encoding='utf-8')
    operation_body='<p><a href="index.html">← 전체 검토 장부</a> · <a href="research_framework.html">현재 연구틀</a></p>'+convert((ROOT/'TWO_TRACK_OPERATION_REDESIGN_20260916.md').read_text(encoding='utf-8'))
    (ROOT/'two_track_operation_redesign.html').write_text(html_doc('보호 학습과 보호 평가: 연산부터 다시 만든 후보',operation_body),encoding='utf-8')
    for source, target, title in [
        ('TRACK1_CAPACITY_PROTOCOL_20260916.md','track1_capacity_protocol.html','방향1: 작은 잔차층의 실제 확인 계획'),
        ('TRACK1_CAPACITY_RESULTS_20260916.md','track1_capacity_results.html','방향1: 환자 분리 잔차층의 실제 결과'),
        ('TRACK1_PATIENT_DP_COMPARISON_PLAN_20260916.md','track1_patient_dp_comparison_plan.html','방향1: 다음 환자DP 비교 조건'),
        ('TRACK1_PATIENT_DP_PROTOCOL_20260916.md','track1_patient_dp_protocol.html','방향1: 공개 보정과 환자DP 실행 조건'),
        ('TRACK1_PATIENT_DP_RESULTS_20260916.md','track1_patient_dp_results.html','방향1: 실제 환자DP 결과와 손실 원인'),
        ('REALISTIC_RESEARCH_PLAN_20260916.md','realistic_research_plan.html','의료 생성모델 환자 보호: 현실적인 다음 계획')]:
        if (ROOT/source).exists():
            body='<p><a href="index.html">← 전체 검토 장부</a> · <a href="two_track_operation_redesign.html">설계와 선행 대조</a></p>'+convert((ROOT/source).read_text(encoding='utf-8'))
            (ROOT/target).write_text(html_doc(title,body),encoding='utf-8')
    for pid in ['R01','R02']:
        body='<p><a href="../rationale.html">← 상세 판단 근거</a></p>'+convert((ROOT/'rationale_sources'/(pid+'.md')).read_text(encoding='utf-8'))
        (ROOT/'rationale_sources'/(pid+'.html')).write_text(html_doc(pid,body),encoding='utf-8')
    summary=dict(records=len(records),local_pdfs=sum(bool(r['local_pdf']) for r in records),statuses=dict(Counter(r['review_status'] for r in records)),roles=dict(stats),comparison_rows=len(comparisons),public_packet_replays=2)
    (ROOT/'build_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False))

if __name__=='__main__':main()
