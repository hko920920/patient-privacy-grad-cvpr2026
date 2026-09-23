"""Record the user's 'start' authorization and fixed scope before new GPU work."""
from pathlib import Path
import json
import shutil
from prrd_v3.contracts import CODE, OUT, RESEARCH, now, read, save, sha, source_hashes, require
from .second_source import WEIGHTS, PREPROCESS


def main():
    root=CODE/'_reports/receiver_a1a2_s200_20260922_v1'
    root.mkdir(exist_ok=False)
    before=root/'records_before'; before.mkdir()
    for p in [CODE.parent/n for n in ('AGENTS.md','CURRENT_STATUS.md','WORKLOG.md')]+[
            RESEARCH/n for n in ('research_state.json','RESEARCH_FRAMEWORK.md')]:
        shutil.copy2(p,before/p.name)
    budget_path=CODE/'_reports/receiver_feasibility_budget_20260922_v1/budget.json'
    budget=read(budget_path)
    handoff=CODE/'_reports/receiver_a1_targets_runtime_20260922_v1/targets/private/target_handoff.json'
    require(sha(handoff)==budget['A1_target_handoff_sha256'],'Prepared A1 binding changed')
    contract={
        'schema':'receiver.A1-A2-s200-campaign/v1','created':now(),
        'authorization':'User: 시작, after first matched A1/A2 200-update budget decision',
        'scope':'prepare second source, changed-path verification, two banks, seal both, fixed V evaluation',
        'budget_file':str(budget_path),'budget_sha256':sha(budget_path),'budget':budget,
        'A1_handoff':str(handoff),'A1_handoff_sha256':sha(handoff),
        'second_encoder':'public DINOv2 ViT-B/14','second_weights':str(WEIGHTS),
        'second_weights_sha256':sha(WEIGHTS),'second_preprocessing':PREPROCESS,
        'second_public_projection':'P-only patient-equal PCA16/q95, no whitening, no label use',
        'A1_A2_relation':'BioViL objective unchanged; A2 adds separate DINO block at equal encoder mean',
        'second_head_id':'dinov2_vitb14','shared_conditions':'exact A1 condition affine and brightness tensors',
        'raw_pixel_scope':'P/Q target preparation; P-only initialization; V features reused, no V pixels',
        'evaluation':'both final200 PNG banks sealed first; existing point-only ridge0.1 BioViL/DenseNet',
        'bootstrap':'same existing 2000 patient-cluster draws; fixed-bank conditional intervals',
        'initialization':'same 128 public templates, fresh seed101 independent optimizers',
        'A2_profile':'one public128 full-pyramid warm-up plus 3 measured updates, no parameter search',
        'verification':'existing tolerances; new DINO and two-source gradients, CPU schedule/restore',
        'storage':'only new run redundant checkpoint payloads may be pruned; initial and two latest retained; all receipts/traces retained',
        'automatic_followup':False,'DP_Expert_Reserved_final_receiver':False,
        'legacy_sources':source_hashes(),'free_bytes_at_start':shutil.disk_usage(CODE).free,
    }
    save(root/'campaign_contract.json',contract)
    (root/'execution_contract.md').write_text(
        '# A1/A2 첫 200회 개발 비교 실행 계약\n\n'
        '사용자의 시작 지시에 따라 목표 준비부터 두 bank의 최종 PNG 및 개발 평가까지 연속 실행한다.\n'
        'A1 60분, A2 120분의 합성 worker 상한을 유지한다. 준비·검사·평가 시간은 별도 기록한다.\n'
        '공개 P만으로 DINO PCA16/q95를 고정하고 동일 조건의 환자별 class 평균을 사용한다.\n'
        '두 결과 모두 고정되기 전에 효용 평가를 열지 않는다. 추가 seed·DP·final은 실행하지 않는다.\n',
        encoding='utf-8')
    state_path=RESEARCH/'research_state.json'
    state=read(state_path)
    state['receiver_A1_A2_first200_execution']={
        'status':'RUNNING_IMPLEMENTATION_AND_SECOND_TARGET_PREPARATION','started':now(),
        'contract':str(root/'campaign_contract.json'),'bank_ids':budget['bank_ids'],
        'updates':200,'final_evaluation_opened':False,'DP_final_opened':False,
    }
    state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'campaign':str(root),'contract_sha256':sha(root/'campaign_contract.json'),
                      'free_bytes':shutil.disk_usage(CODE).free}),flush=True)


if __name__=='__main__':main()
