"""Audit saved outputs, generate result tables, and close the local task."""
import csv,datetime,hashlib,json,sys
from pathlib import Path
import numpy as np

out=Path(__file__).resolve().parent;rr=out.parent;tr=rr.parent.parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def npz(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def check(v,msg):
    if not v:raise RuntimeError(msg)
def prepend(p,s):
    b=p.read_bytes();i=b.index(b'\n')+1;p.write_bytes(b[:i]+('\n'+s+'\n\n').encode('utf-8')+b[i:])
def ci(v):return '['+','.join(f'{x:+.6f}' for x in v)+']'

c=read(out/'contract.json');r=read(out/'results.json');f=read(out/'construction_results.json')
check(sha(out/'EXECUTION_CONTRACT.md')==c['human_contract_sha256'],'Human contract changed')
for p,h in c['implementation'].items():check(sha(p)==h,'Implementation changed')
for p in c['inputs'].values():check(sha(p['path'])==p['sha256'],'Input changed')
for p,h in c['historical_result_hashes'].items():check(sha(p)==h,'Historical report changed')
for name in ('label_seal.json','completion.json'):
    s=read(out/name);check(s['contract_sha256']==sha(out/'contract.json'),'Contract/seal mismatch')
    for p,h in s['files'].items():check(sha(out/p)==h,'Sealed output changed')
labels=npz(out/'soft_labels.npz');oldlabels=npz(c['inputs']['old_labels']['path']);aliases=read(out/'label_aliases.json')
for name,l in labels.items():
    check(l.shape==(128,) and np.isfinite(l).all() and abs(l).max()<=1,'Invalid labels')
    if name in aliases:check(np.array_equal(l,oldlabels[aliases[name]['key']]),'Reused label changed')
    else:
        with (out/'label_packets'/f'{name}.csv').open(encoding='utf-8',newline='') as h:rows=list(csv.DictReader(h))
        check(np.array_equal(l,[float(x['regression_target']) for x in rows]),'CSV roundtrip')
        check(all(abs(float(x['soft_class1_target'])-(float(x['regression_target'])+1)/2)<1e-15 for x in rows),'Soft target mapping')
check(len(aliases)==4 and len(list((out/'label_packets').glob('*.csv')))==4,'Packet count')
b=npz(out/'paired_bootstrap.npz');d=npz(out/'contrast_bootstrap.npz')
bn={str(k):i for i,k in enumerate(b['names'])};dn={str(k):i for i,k in enumerate(d['names'])}
maxerr=0.
for rec in ('DenseNet','ResNet18'):
    for recipe in ('RN_ONLY','JOINT'):
        bs=[];points=[]
        for draw in ('DP1','DP2'):
            a=f'{draw}_MT_{recipe}_{rec}';z=f'{draw}_KD_{recipe}_{rec}';key=f'{draw}_{recipe}_MT_minus_KD_{rec}'
            dist=b['metrics'][:,bn[a],:]-b['metrics'][:,bn[z],:];bs.append(dist)
            check(np.array_equal(dist,d['differences'][:,dn[key],:]),'Paired bootstrap contrast')
            point=np.array([r['metrics'][a][m]['point']-r['metrics'][z][m]['point'] for m in ('AUROC','AP')]);points.append(point)
            for j,m in enumerate(('AUROC','AP')):
                rr0=r['per_release_contrasts'][key][m]
                maxerr=max(maxerr,abs(float(point[j])-rr0['delta']),float(abs(np.quantile(dist[:,j],[.025,.975])-rr0['patient_cluster_95']).max()))
        md=(bs[0]+bs[1])/2;mp=(points[0]+points[1])/2;key=f'MEAN_{recipe}_MT_minus_KD_{rec}'
        check(np.array_equal(md,d['differences'][:,dn[key],:]),'Release mean bootstrap pairing')
        for j,m in enumerate(('AUROC','AP')):
            rr0=r['fixed_two_release_mean_contrasts'][key][m]
            maxerr=max(maxerr,abs(float(mp[j])-rr0['mean_delta']),float(abs(np.quantile(md[:,j],[.025,.975])-rr0['fixed_two_release_patient_cluster_95']).max()))
check(maxerr<1e-15,'Reported contrast mismatch')
access=[json.loads(x) for x in (out/'access_log.jsonl').read_text(encoding='utf-8').splitlines()]
sealed=datetime.datetime.fromisoformat(read(out/'label_seal.json')['utc'])
for a in access:
    check(a['phase'] in c['inputs'][a['key']]['phases'],'Forbidden phase access')
    if a['phase']=='evaluate':check(datetime.datetime.fromisoformat(a['utc'])>=sealed,'Evaluation before labels')
size=sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
check(size<50*1024**2,'Storage bound exceeded')
stamp=datetime.datetime.now(datetime.timezone.utc)
duration=(stamp-datetime.datetime.fromisoformat(c['started_utc'])).total_seconds()/60
verification={'status':'PASS','inputs_unchanged':len(c['inputs']),'implementation_unchanged':len(c['implementation']),
    'historical_reports_unchanged':len(c['historical_result_hashes']),'new_label_packets':4,'reused_MT_labels_exact':4,
    'contrast_reconstruction_max_abs':maxerr,'evaluation_after_label_seal':True,
    'new_label_KKT_max_abs':max(x['independent_projected_gradient_max_abs'] for x in f['fits'].values() if not x['reused']),
    **r['verification'],'contract_to_finalization_minutes':duration,'scope_size_MiB_before_report':size/1024**2}
dump(out/'verification.json',verification)

lines=['# 공개128에서 일반 KD와 현재 평균 전달 비교 결과','',
       '작성일: 2026-09-28  ',
       '**연구 판정: 긍정적. 정의한 일반 teacher-score KD 대비 현재 평균 전달의 개발 효용 이득이 공개128에서도 확인됐다.**  ',
       'RN-only와 Joint 모두 두 기존 DP 요약에서 DenseNet AUROC가 개선됐다. 합성영상에만 나타난 이득은 아니다. 두 모델 결합·합성영상 제작의 필요성과 독자적 논문 기여가 확보됐다는 뜻은 아니다. 큰 연구 단계2를 유지한다.','',
       '## DenseNet 주평가','',
       '| 라벨 목표 | 보호 요약 | 일반 KD AUROC | 현재 MT AUROC | MT−KD | 대응 환자95%CI |',
       '|---|---|---:|---:|---:|---|']
for recipe in ('RN_ONLY','JOINT'):
    for draw in ('DP1','DP2'):
        kd=r['metrics'][f'{draw}_KD_{recipe}_DenseNet']['AUROC']['point'];mt=r['metrics'][f'{draw}_MT_{recipe}_DenseNet']['AUROC']['point']
        x=r['per_release_contrasts'][f'{draw}_{recipe}_MT_minus_KD_DenseNet']['AUROC']
        lines.append(f"| {recipe} | {draw} | {kd:.6f} | {mt:.6f} | {x['delta']:+.6f} | {ci(x['patient_cluster_95'])} |")
lines+=['','| 라벨 목표 | 보호 요약 | 일반 KD AP | 현재 MT AP | MT−KD | 대응 환자95%CI |','|---|---|---:|---:|---:|---|']
for recipe in ('RN_ONLY','JOINT'):
    for draw in ('DP1','DP2'):
        kd=r['metrics'][f'{draw}_KD_{recipe}_DenseNet']['AP']['point'];mt=r['metrics'][f'{draw}_MT_{recipe}_DenseNet']['AP']['point']
        x=r['per_release_contrasts'][f'{draw}_{recipe}_MT_minus_KD_DenseNet']['AP']
        lines.append(f"| {recipe} | {draw} | {kd:.6f} | {mt:.6f} | {x['delta']:+.6f} | {ci(x['patient_cluster_95'])} |")
lines+=['','| 라벨 목표 | 두 요약 평균 AUROC 차이 | 조건부95%CI | 평균 AP 차이 | 조건부95%CI |','|---|---:|---|---:|---|']
for recipe in ('RN_ONLY','JOINT'):
    x=r['fixed_two_release_mean_contrasts'][f'MEAN_{recipe}_MT_minus_KD_DenseNet'];a=x['AUROC'];p=x['AP']
    lines.append(f"| {recipe} | {a['mean_delta']:+.6f} | {ci(a['fixed_two_release_patient_cluster_95'])} | {p['mean_delta']:+.6f} | {ci(p['fixed_two_release_patient_cluster_95'])} |")
lines+=['','네 AUROC 비교의 구간이 모두 양수다. AP 점차이도 네 비교 모두 양수이나 RN-only DP2의 AP 구간은0을 포함한다. 두 recipe 각각의 고정 두-release 평균에서는 AUROC/AP 구간이 모두 양수다. 사전에 고정한 개발 투자 기준은 두 recipe 모두 충족했다.',
        '', '지표는 영상별 AUROC/AP다. 2,000회 같은 환자-cluster 재표집으로 비교했다. 평균은 두 release의 지표 차이의 산술 평균이며 예측 ensemble이 아니다. 구간은 고정된 두 DP 요약·공개128·라벨에 조건부다. 네 비교를 독립 재현4회로 세지 않는다. DenseNet/V는 재사용된 개발 평가다.',
        '', '## 비교의 공정성과 범위','',
        '공개128 선정·순서·이미지/특징·기존 DP1/DP2·공개 환자/class 가중치·DINO 목표·라벨 제약·RMS 정규화 규칙·ridge·평가를 고정했다. 차이는 RN 목표 생성 방식이다. RN-only와 DINO+RN Joint를 모두 보고하며 유리한 하나만 고르지 않았다.',
        '', r'KD는 $q=X_Dw_D$, $w_R^{KD}=(X_R^TWX_R+0.1I)^{-1}X_R^TWq$다. sigmoid/temperature/clipping을 추가하지 않았다. 1번의 봉인된 목표 계수를 재사용하고 다른 least-squares 분해로 검산했다. MT도 1번의 공개 class별 affine mean map 목표를 그대로 썼다.',
        '', r'두 방식 모두 $\min_{\ell\in[-1,1]^{128}}\|A\ell-t\|^2+\eta\|\ell\|^2$, $\eta=10^{-3}\|A\|_F^2/128$의 zero-anchor다. 모델별 P 가중 목표 RMS를 사용하며 Joint는1/2씩이다. 규칙이 같아도 목표 RMS와 eta의 수치는 다르다. BVLS tol1e−12/max_iter500으로 같은 bounded 문제를 풀었다.',
        '', '기존 MT 라벨4개·점수·bootstrap을 정확히 재사용했다. 새 KD 라벨4개를 모두 고정한 후 DenseNet/RN readout8개만 계산했다. 새 pixel 학습·Q 조회·DP 요약·noise·모델 forward·공개영상 재선정·새 수신자·Expert/Reserved 접근은0이다. 기존 여섯 release 공동 공개 기본 상한48/6e−5는 변하지 않는다. 공개128+DP 라벨은 완전한 public-only 대조가 아니다.',
        '', '## 목표 최적화와 검산','',
        'RN-only KD의 공개 목표 relative RMSE는 DP1 0.008355, DP2 0.008976로 작았다. 따라서 해당 KD 목표를 거의 맞추지 못한 실패 실행을 대조로 쓴 것은 아니다. Joint KD의 DINO/RN residual은 더 크지만 이를 성능 차이의 입증된 원인으로 단정하지 않는다. 계산 성공과 개발 효용은 별개로 보고한다.',
        '', '| 검산 | 최대 오차 |','|---|---:|',
        f"| KD 목표의 독립 least-squares 재구성 | {f['KD_independent_lstsq_max_abs']:.3e} |",
        f"| 새 라벨 projected-gradient KKT | {verification['new_label_KKT_max_abs']:.3e} |",
        f"| 새 ridge primal/dual 예측 | {r['verification']['new_ridge_primal_dual_max_abs']:.3e} |",
        f"| 기존 MT 예측 재현 | {r['verification']['reused_MT_prediction_max_abs']:.3e} |",
        f"| 독립 sklearn 가중 bootstrap witness | {r['verification']['bootstrap_sklearn_max_abs']:.3e} |",
        f"| 저장된 paired contrast/평균/CI 재구성 | {maxerr:.3e} |",' ',
        '입력17개·구현19개·과거 결과보고서4개의 해시를 보존했다. 모든 라벨 bounds/유한성·CSV roundtrip·재사용MT 일치·평가 전 봉인·phase별 입력 경계를 확인했다. 이번 실행에서 수치 실패·재시도·계수 변경은 없었다.',
        '', '## ResNet18 참고 평가','',
        'RN은 목표 구성에 사용된 모델이다. 아래 결과를 별도의 독립 receiver 전이 성공으로 계산하지 않는다.','',
        '| 라벨 목표 | 평균 AUROC 차이 | 조건부95%CI | 평균 AP 차이 | 조건부95%CI |','|---|---:|---|---:|---|']
for recipe in ('RN_ONLY','JOINT'):
    x=r['fixed_two_release_mean_contrasts'][f'MEAN_{recipe}_MT_minus_KD_ResNet18'];a=x['AUROC'];p=x['AP']
    lines.append(f"| {recipe} | {a['mean_delta']:+.6f} | {ci(a['fixed_two_release_patient_cluster_95'])} | {p['mean_delta']:+.6f} | {ci(p['fixed_two_release_patient_cluster_95'])} |")
lines+=['','## 지금까지의 전략 판단','',
        '1번의 합성영상·hard-anchor 설정에서 확인한 MT−KD 이득에 더해, 이번에는 고정 공개128·zero-anchor에서 두 recipe 모두 AUROC 이득을 확인했다. 기존 보호 정보를 다른 모델의 목표로 전달하는 구성의 개발 가치가 더 구체화됐다.',
        '', '2번의 결합 필요성 미확인과 3번의 합성영상 추가 비용 미정당화는 그대로다. 이번 성공으로 그 결론을 뒤집지 않는다. 공개 carrier에서도 현재 목표 구성을 사용할 이유가 생겼지만, 공개 재라벨링 자체가 새 발명이 된 것은 아니다.',
        '', '이번 대조는 정의한 teacher-score ridge KD다. 모든 KD variant/POST/KIP/DP KME의 공식 전체 구현을 이겼다는 결과가 아니다. 기존 전략에 기록된 signed-public-label ridge 동치와 알려진 label solve의 중복을 유지한다. 같은 정보 접근에서 class-conditional mean/signed-weight 표현을 허용한 기존 대안과의 차이는 남은 기여 문제다.',
        '', '다음으로는 현재 목표 구성의 어느 차이가 필요한지 수식·기존 구현에서 좁히고, 이미 동치인 연산은 신규성에서 제외해야 한다. 공개128 재선정 반복은 그 이후 필요한 재현 과제이며 이번에 자동 실행하지 않았다. 새 private release·encoder·loss·Expert/Reserved를 추가하지 않는다. 연구 단계2와 CVPR 기여 미확정 상태를 유지한다.',
        '', '## 비용과 산출물','',
        f"- 네 새 라벨 계산·구성 검산: {f['seconds']:.3f}초.",
        f"- 새 readout/대응 bootstrap/기존 점수 연결: {r['seconds']:.3f}초.",
        f'- 계약 동결 후 본 검산·보고 생성까지: {duration:.2f}분. 계약 이전 adapter 작성·조회 시간은 이 수치에 포함하지 않는다.',
        f'- 보고 생성 전 추가 폴더 크기: {size/1024**2:.2f}MiB. 모델·이미지 복제 없음.',
        '', '실행 폴더: `'+out.name+'`.',
        '주요 파일: `EXECUTION_CONTRACT.md`, `contract.json`, `run_comparison.py`, `label_packets/`, `label_seal.json`, `results.json`, `paired_bootstrap.npz`, `contrast_bootstrap.npz`, `verification.json`.',
        '환자별 prediction과 DP 파생 라벨은 로컬 artifact로 유지하며 원격으로 업로드하지 않았다.','']
report=rr/'PUBLIC_TRANSFER_PUBLIC128_KD_RESULTS_20260928.md'
check(not report.exists(),'Do not overwrite an existing report')
report.write_text('\n'.join(lines),encoding='utf-8')
block='**PUBLIC128 KD CONTROL COMPLETE — 2026-09-28**\n\n'+(
    '공개128 KD 대조는 긍정적. 같은 기존DP1/DP2·공개128·zero-anchor에서 MT−KD DenseNet 평균 AUROC: '
    'RN-only +0.053284, CI[+0.030382,+0.076880]; Joint +0.068250, CI[+0.045708,+0.092791]. '
    '두 recipe 평균 AP 구간도 양수. 각release AUROC4개 모두 양의 구간이나 RN-only DP2 AP는0 포함. '
    '정의한 teacher-score KD 대비 공개 carrier의 추가 개발 효용. 결합/합성영상 필요성·선행 전체 우위·CVPR 기여 확정 아님. '
    '새 KD라벨4개, MT4개재사용; Q/release/pixel/forward/Expert/Reserved0. 모든 검산 통과. Stage2 유지, 실행 종료, 자동 후속 없음.')
for p in [tr/'AGENTS.md',tr/'CURRENT_STATUS.md',tr/'WORKLOG.md',rr/'RESEARCH_FRAMEWORK.md']:
    link=(report.name if p.parent==rr else 'CVPR 주제 탐색/research_2026-09-10/'+report.name)
    prepend(p,block+'\n\n[Results](<'+link+'>)')
state=read(rr/'research_state.json')
state['current_step']=2;state['current_result_report']=report.name
state['step2_active_task']='Public128 KD comparison complete and positive for RN-only/Joint; no running task. Method novelty remains unresolved.'
state['active_execution']={'status':'no_running_execution','last_completed':'public128_kd_control_20260928','report':report.name,'completed_utc':stamp.isoformat(),'next_execution_started':False}
state['next_task']='Narrow the target-construction difference against equally informed class-conditional/public weighted-label transfer. Preserve positive KD comparisons and existing algebraic equivalence; no automatic new release or final evaluation.'
state['public_transfer_public128_kd_20260928']={'status':'complete_positive_development','report':report.name,'report_sha256':sha(report),'out_dir':str(out),
    'contract_sha256':sha(out/'contract.json'),'decisions':r['development_decisions'],'means':r['fixed_two_release_mean_contrasts'],
    'new_KD_labels':4,'reused_MT_labels':4,'new_private_releases':0,'verification':verification,'completed_utc':stamp.isoformat()}
dump(rr/'research_state.json',state)
dump(out/'finalization.json',{'status':'COMPLETE','report':str(report),'report_sha256':sha(report),'completed_utc':stamp.isoformat(),
    'current_step':2,'active_execution':'none','verification':verification,'finalizer_sha256':sha(__file__)})
print(json.dumps({'status':'COMPLETE','report_name':report.name,'verification':verification},ensure_ascii=True,indent=2))
