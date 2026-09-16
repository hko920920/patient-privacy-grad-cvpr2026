"""Render verified frozen contrasts; no new analysis choices or fitting."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
CODE = ROOT.parent.parent / 'code_working'
RUN = CODE / '_reports/cvpr_u_pilot_v1_001/baseline_screen_20260915'
SOURCE = RUN / 'cdi_cross_patient_intervention_v1/existing_scorer_analysis_v2'
OUT = ROOT / 'cross_intervention_artifacts/v1'
METHODS = ['cdi_image_mean_fixed','cdi_image_max_fixed','patient_mean26_fixed','patient_meanmax52_fixed']
NAMES = ['Image LR, mean','Image LR, max','Patient mean26 LR','Patient mean+max52 LR']
ARM = 'leave_patient_out79_refit'


def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for part in iter(lambda:f.read(8*1024*1024), b''):
            h.update(part)
    return h.hexdigest()


def savecsv(path, rows):
    with path.open('x', encoding='utf-8-sig', newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    a = read(SOURCE/'analysis.json')
    audit = read(SOURCE/'independent_fixed_scorer_audit_v1/verification.json')
    assert a['complete'] and audit['complete']
    assert audit['status']=='PASS_INDEPENDENT_CROSS_PATIENT_FIXED_SCORER_ARITHMETIC_AND_PROVENANCE'
    assert audit['analysis_sha256']==sha(SOURCE/'analysis.json')
    rows=read(SOURCE/'patient_scores.json');contrasts=read(SOURCE/'cross_contrasts.json')
    contexts=read(SOURCE/'patient_contexts.json')
    index={(r['arm'],r['method'],r['scenario']):r for r in contrasts}
    primary=[index[ARM,m,s] for s in ['U','E'] for m in METHODS]
    assert len(primary)==8 and all(r['clean_fixed_C_primary'] for r in primary)
    OUT.mkdir(parents=True,exist_ok=False)
    flat=[]
    for r in contrasts:
        flat.append(dict(arm=r['arm'],method=r['method'],scenario=r['scenario'],
            clean_primary=r['clean_fixed_C_primary'],D_p_p=r['D'][0][0],D_p_q=r['D'][0][1],
            D_q_p=r['D'][1][0],D_q_q=r['D'][1][1],R_p=r['R_p']['value'],R_q=r['R_q']['value'],K=r['K']['value'],
            both_positive_arithmetic=r['both_own_minus_cross_positive_outside_arithmetic_envelope']))
    savecsv(OUT/'all_existing_contrasts.csv',flat)
    plotrows=[r for r in rows if r['arm']==ARM and r['method'] in METHODS and r['scenario']=='U']
    savecsv(OUT/'primary_U_response_points.csv',[
        dict(method=r['method'],patient_id=r['patient_id'],group=r['group'],
             D_p_i=r['D_p_i'],D_q_i=r['D_q_i'],h_i=r['h_i']) for r in plotrows])
    fig, axes=plt.subplots(2,2,figsize=(11,9),constrained_layout=True)
    for ax,method,title in zip(axes.flat,METHODS,NAMES):
        rr=[r for r in plotrows if r['method']==method]
        assert len(rr)==41
        background=[r for r in rr if r['patient_id'] not in ['14393','4092']]
        ax.scatter([r['D_p_i'] for r in background],[r['D_q_i'] for r in background],
                   color='#929aa5',s=24,alpha=.7,label='Other 39 patients')
        for pid,color,marker in [('14393','#c43e35','o'),('4092','#176aa8','s')]:
            r=next(r for r in rr if r['patient_id']==pid)
            name='p' if pid=='14393' else 'q'
            ax.scatter(r['D_p_i'],r['D_q_i'],color=color,s=78,marker=marker,zorder=4,label=name+' (patient '+pid+')')
            ax.annotate(name,(r['D_p_i'],r['D_q_i']),xytext=(6,6),textcoords='offset points',color=color,weight='bold')
        vals=[r[f] for r in rr for f in ['D_p_i','D_q_i']]
        lo,hi=min(vals),max(vals);pad=max((hi-lo)*.08,1e-5);lo-=pad;hi+=pad
        ax.plot([lo,hi],[lo,hi],linestyle='--',color='#333333',lw=1,label='Equal response to p/q removal')
        ax.set(xlim=(lo,hi),ylim=(lo,hi),xlabel='D[p,i]: score change from removing p',ylabel='D[q,i]: score change from removing q',title=title)
        ax.axhline(0,color='#cccccc',lw=.6);ax.axvline(0,color='#cccccc',lw=.6)
        ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.15)
    axes[0,0].legend(fontsize=8,loc='best')
    fig.suptitle('Fixed existing U79 scorers: two patient-removal interventions\nU2 per patient; descriptive fixed-path responses, not population MIA performance',fontsize=12)
    fig.savefig(OUT/'primary_U_response_scatter.png',dpi=170)
    fig.savefig(OUT/'primary_U_response_scatter.svg');plt.close(fig)
    lines=['## 이번 실제 결과','',
           '아래는 새 환자를 점수기 보조학습에서 제외한 기존 U79 고정 C 주4개 전부다. E도 같은 U-fit 점수기로 계산했다. 양수/음수는 계산 오차를 넘는 산술 부호이며 통계적 유의성이나 오탐률 판정이 아니다.','',
           '| 기존 방법 | U R_p | U R_q | U K | E R_p | E R_q | E K |','|---|---:|---:|---:|---:|---:|---:|']
    for method,name in zip(METHODS,NAMES):
        u,e=index[ARM,method,'U'],index[ARM,method,'E']
        v=[r[k]['value'] for r in [u,e] for k in ['R_p','R_q','K']]
        lines.append('| '+name+' | '+' | '.join(f'{x:+.8f}' for x in v)+' |')
    lines+=['','![기존 주4방법의 두 개입 U 반응](cross_intervention_artifacts/v1/primary_U_response_scatter.png)','',
            '회색 점은 공통 배경39명이다. p가 대각선 아래면 R_p>0, q가 위면 R_q>0이다. 두 색 점 사이의 관계를 보이며 회색 환자39명을 독립 개입 반복39회로 세지 않는다.','',
            '[모든73개 대비 CSV](cross_intervention_artifacts/v1/all_existing_contrasts.csv) · [주4방법의 U164점 CSV](cross_intervention_artifacts/v1/primary_U_response_points.csv) · [벡터 그림](cross_intervention_artifacts/v1/primary_U_response_scatter.svg)','',
            '| 기존 방법 | p의 h | q의 h | 공통39 h 중앙값 | 공통39 h Q25 | 공통39 h Q75 |','|---|---:|---:|---:|---:|---:|']
    for method,name in zip(METHODS,NAMES):
        rr=[r for r in plotrows if r['method']==method]
        pp=next(r['h_i'] for r in rr if r['patient_id']=='14393')
        qq=next(r['h_i'] for r in rr if r['patient_id']=='4092')
        cx=next(r['h_i'] for r in contexts if r['arm']==ARM and r['method']==method and r['context']=='background39_all')
        vals=[pp,qq,cx['median'],cx['q25'],cx['q75']]
        lines.append('| '+name+' | '+' | '.join(f'{x:+.8f}' for x in vals)+' |')
    (OUT/'verified_results_fragment.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    files=['analysis.json','patient_scores.json','cross_contrasts.json','patient_contexts.json','independent_fixed_scorer_audit_v1/verification.json']
    manifest=dict(schema='cross-intervention-display-artifacts/v1',source_code_sha256=sha(Path(__file__)),
        source_sha256={str(SOURCE/n):sha(SOURCE/n) for n in files},
        output_sha256={p.name:sha(p) for p in OUT.iterdir()},new_analysis_choices=False,new_fitting=False)
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({'status':'PASS_VERIFIED_CROSS_CONTRAST_DISPLAY','primary_rows':8,'scatter_points':len(plotrows),'all_contrasts':len(flat)}))


if __name__=='__main__':
    main()
