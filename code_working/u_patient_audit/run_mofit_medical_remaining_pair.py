"""Remaining E006/U004 of the same patient, frozen MoFit kernel on both targets."""
import argparse
import gc
from importlib.metadata import version
import json
from pathlib import Path
import re
import time
import traceback
import torch

from .common import ROOT, RUN, IMAGES, digest, read_csv, write_json
from .models import setup, snapshot, scheduler, load_unet, adapter_state
from .mofit_medical_adapter import MoFitMedicalAdapter, kernel_policy, tensor_hash, make_draws
from .run_mofit_medical_kernel import INPUT_DIR, load, new_json, now, fingerprint, assert_hashes
from .run_mofit_medical_target2 import COUNTS, KERNEL_ARGUMENTS

BASE = RUN / 'baseline_screen_20260915'
RESEARCH = next(ROOT.parent.glob('*/research_2026-09-10'))
MODEL_ORDER = ['model_1', 'model_2']
IMAGES_ORDER = ['00014393_006.png', '00014393_004.png']
CHECKPOINTS = {m: RUN / 'training_coverage_v2' / m / 'step_1000.pt' for m in MODEL_ORDER}
PRIOR_DIRS = {'model_1': BASE / 'mofit_medical_full_v1',
              'model_2': BASE / 'mofit_medical_target2_full_v1'}
ORIGINAL_CONTRACT = BASE / 'mofit_medical_target2_contract_v1.json'
ANALYSIS_CONTRACT = RESEARCH / 'spec_sources/mofit_patient14393_two_image_analysis_contract_20260915.json'


def actual_exposure():
    states, summary, counts, image_members = {}, {}, {}, {}
    for model in MODEL_ORDER:
        state = torch.load(CHECKPOINTS[model], map_location='cpu', weights_only=True)
        manifest = read_csv(RUN / 'cohort' / (model + '_train.csv'))
        assert state['step'] == 1000 and sum(state['exposures'].values()) == 4000
        assert set(state['exposures']) == {r['image_id'] for r in manifest}
        assert all(v > 0 for v in state['exposures'].values())
        patient = [r for r in manifest if r['patient_id'] == '14393']
        member = int(bool(patient))
        assert member == (model == 'model_1')
        counts[model] = {i: state['exposures'].get(i, 0) for i in IMAGES_ORDER}
        image_members[model] = {i: int(i in state['exposures']) for i in IMAGES_ORDER}
        assert counts[model] == dict(zip(IMAGES_ORDER, [4 if member else 0, 0]))
        summary[model] = dict(patient_training_images=len(patient),
            patient_training_exposures=sum(state['exposures'][r['image_id']] for r in patient),
            manifest_images=len(manifest), training_exposures_total=4000)
        assert summary[model]['patient_training_images'] == (2 if member else 0)
        assert summary[model]['patient_training_exposures'] == (8 if member else 0)
        states[model] = state
    return states, summary, counts, image_members


def create_contract(path):
    if path.exists():
        raise RuntimeError('Immutable remaining-pair contract already exists')
    old = load(ORIGINAL_CONTRACT)
    frozen = dict(old['frozen_sha256'])
    assert_hashes(frozen)
    paths = [Path(__file__), ROOT / 'u_patient_audit/verify_mofit_medical_remaining_pair.py',
             ROOT / 'u_patient_audit/analyze_mofit_patient14393.py', ANALYSIS_CONTRACT, ORIGINAL_CONTRACT]
    for model in MODEL_ORDER:
        paths += [CHECKPOINTS[model], RUN / 'cohort' / (model + '_train.csv')]
        paths += [PRIOR_DIRS[model] / n for n in
                  ('protocol.json', 'results.json', 'raw.pt', 'execution.json',
                   'verification.json', 'verification_protocol.json')]
        prior = load(PRIOR_DIRS[model] / 'verification.json')
        assert prior['status'].startswith('PASS_') and prior['images'] == 2
    for p in paths:
        frozen[str(p.resolve())] = digest(p)
    _, exposure, counts, members = actual_exposure()
    c = dict(schema='mofit-medical-remaining-pair-contract/v1', current_step=2,
        created_utc=now(), patient_id='14393', eval_role='fit', assignment_group='A',
        checkpoint_step=1000, model_order=MODEL_ORDER, image_order=IMAGES_ORDER,
        checkpoints={m: str(p) for m,p in CHECKPOINTS.items()},
        checkpoint_sha256={m: digest(p) for m,p in CHECKPOINTS.items()},
        member_by_model={'model_1': 1, 'model_2': 0}, actual_exposure=exposure,
        actual_training_exposures_by_model=counts, image_member_by_model=members,
        input_directory=str(INPUT_DIR), inputs_sha256=old['inputs_sha256'],
        model_snapshot=old['model_snapshot'], alpha_cumprod_t140=old['alpha_cumprod_t140'],
        vae_scaling_factor=.18215, dtype='float32', kernel_policy=kernel_policy(),
        kernel_arguments=KERNEL_ARGUMENTS, expected_counts_per_image=COUNTS,
        expected_counts_per_model={k: 2*v for k,v in COUNTS.items()},
        expected_total_counts={k: 4*v for k,v in COUNTS.items()},
        prior_directories={m: str(p) for m,p in PRIOR_DIRS.items()},
        analysis_contract=str(ANALYSIS_CONTRACT), analysis_contract_sha256=digest(ANALYSIS_CONTRACT),
        E_meaning='Fixed train_candidate role: member/exposure4 only in target1, absent in target2',
        U_meaning='U_observed role: absent image in both targets; patient participates only in target1',
        comparison_scope='Complete same patient14393 E2/U2 using remaining E006/U004; no new patient',
        initialization='Frozen inputs and all four CPU draws reused across both targets and prior images',
        checkpoint_adapter_loading='Saved FP32 LoRA values exact; base FP16 artifact promoted to FP32',
        causal_limit='Whole A/B training cohorts differ; not an isolated individual-patient intervention',
        analysis_scope='Fixed h,-u,-v,-(u+v), per-image orientation before E/U patient mean/max; no AUC/CI',
        no_new_backward_check='Reuse numerical kernel; prior target1 initial-point FD only',
        frozen_sha256=frozen, new_target_training=False, attack_parameter_tuning=False,
        cohort_expansion=False, perform_membership_performance_evaluation=False,
        stage2_completion_claim=False)
    new_json(path, c)
    return c


def validate(c):
    for key,value in dict(schema='mofit-medical-remaining-pair-contract/v1', current_step=2,
        patient_id='14393', eval_role='fit', assignment_group='A', checkpoint_step=1000,
        dtype='float32', new_target_training=False, attack_parameter_tuning=False,
        cohort_expansion=False, perform_membership_performance_evaluation=False,
        stage2_completion_claim=False).items():
        assert c[key] == value, key
    assert c['model_order'] == MODEL_ORDER and c['image_order'] == IMAGES_ORDER
    assert c['kernel_arguments'] == KERNEL_ARGUMENTS and c['kernel_policy'] == kernel_policy()
    assert c['expected_counts_per_image'] == COUNTS
    assert c['expected_counts_per_model'] == {k: 2*v for k,v in COUNTS.items()}
    assert c['expected_total_counts'] == {k: 4*v for k,v in COUNTS.items()}
    assert c['member_by_model'] == {'model_1': 1, 'model_2': 0}
    assert_hashes(c['frozen_sha256'])
    states, exposure, counts, members = actual_exposure()
    assert exposure == c['actual_exposure'] and counts == c['actual_training_exposures_by_model']
    assert members == c['image_member_by_model']
    for model,p in CHECKPOINTS.items():
        assert Path(c['checkpoints'][model]).resolve() == p.resolve()
        assert digest(p) == c['checkpoint_sha256'][model]
    assert digest(INPUT_DIR / 'inputs.pt') == c['inputs_sha256']
    assert digest(ANALYSIS_CONTRACT) == c['analysis_contract_sha256']
    payload = torch.load(INPUT_DIR / 'inputs.pt', map_location='cpu', weights_only=True)
    assert payload['policy'] == load(INPUT_DIR / 'preparation_protocol.json')
    assert payload['policy']['kernel_policy'] == kernel_policy()
    for key,value in make_draws().items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(payload['draws'][key], value)
        else:
            assert payload['draws'][key] == value
    assert torch.equal(payload['hidden'][''], payload['null_hidden'])
    rows = {m: {} for m in MODEL_ORDER}
    for image in IMAGES_ORDER:
        item = payload['images'][image]
        source = item['metadata']
        assert source['patient_id'] == '14393' and source['eval_role'] == 'fit'
        assert source['assignment_group'] == 'A'
        assert digest(IMAGES / source['image_filename']) == item['original_file_sha256'] == source['sha256']
        assert tensor_hash(item['pixels']) == item['pixels_sha256']
        for model in MODEL_ORDER:
            rows[model][image] = dict(image_id=image, patient_id='14393', eval_role='fit',
                assignment_group='A', scenario='E' if source['record_role'] == 'train_candidate' else 'U',
                record_role=source['record_role'], model=model, member=c['member_by_model'][model],
                image_member=members[model][image], actual_training_exposures=counts[model][image],
                patient_training_exposures=exposure[model]['patient_training_exposures'],
                caption=item['caption'], pixels_sha256=item['pixels_sha256'],
                initial_hidden_sha256=tensor_hash(payload['hidden'][item['caption']]))
    assert [rows['model_1'][i]['scenario'] for i in IMAGES_ORDER] == ['E', 'U']
    return payload, rows, states


def execute(args, c, payload, metadata, states):
    from diffusers import AutoencoderKL
    tag = args.attempt_tag or 'v1'
    assert re.fullmatch('[A-Za-z0-9_-]+', tag)
    out = BASE / ('mofit_medical_remaining_pair_' + tag)
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    contract_sha = digest(args.contract)
    new_json(out / 'protocol.json', dict(schema='mofit-medical-remaining-pair-execution/v1',
        current_step=2, mode='full', started_utc=now(), contract=c,
        contract_path=str(args.contract.resolve()), contract_sha256=contract_sha,
        alpha_cumprod_t140=c['alpha_cumprod_t140'],
        environment={n: version(n) for n in ['torch','diffusers','transformers','peft','numpy']}))
    unet = vae = adapter = None
    results, raw, model_reports = [], {m: {} for m in MODEL_ORDER}, []
    totals = {k: 0 for k in COUNTS}
    try:
        setup()
        vae = AutoencoderKL.from_pretrained(snapshot() / 'vae', torch_dtype=torch.float16,
            variant='fp16', use_safetensors=True, local_files_only=True).float().to('cuda').eval()
        vae.requires_grad_(False)
        assert float(vae.config.scaling_factor) == c['vae_scaling_factor']
        sched = scheduler()
        assert float(sched.alphas_cumprod[140]) == c['alpha_cumprod_t140']
        for model in MODEL_ORDER:
            model_dir = out / model
            model_dir.mkdir()
            unet = load_unet(CHECKPOINTS[model], training=False).float().requires_grad_(False).eval()
            loaded = adapter_state(unet)
            expected = states[model]['adapter']
            assert loaded.keys() == expected.keys()
            assert all(torch.equal(loaded[k], expected[k].float()) for k in loaded)
            adapter = MoFitMedicalAdapter(unet, vae, sched, payload['null_hidden'])
            before = {'unet': fingerprint(unet), 'vae': fingerprint(vae)}
            assert before == load(PRIOR_DIRS[model] / 'execution.json')['before_fingerprints']
            model_counts = {k: 0 for k in COUNTS}
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            measured_start = time.perf_counter()
            for image in IMAGES_ORDER:
                item = payload['images'][image]
                def progress(stage, done, total, counts, loss):
                    if done <= 2 or done % 50 == 0 or done == total:
                        q = dict(event='MoFit_remaining_pair_progress', image_id=image, model=model,
                            stage=stage, done=done, total=total, counts=counts, loss=loss,
                            seconds=time.perf_counter()-measured_start)
                        write_json(out / 'progress.json', q)
                        print(json.dumps(q, allow_nan=False), flush=True)
                measured = adapter.run(item['pixels'], image, payload['hidden'][item['caption']],
                    payload['draws'], **KERNEL_ARGUMENTS, progress=progress)
                report = {**metadata[model][image], **measured['report']}
                assert report['counts'] == COUNTS
                results.append(report)
                raw[model][image] = measured['raw']
                for key in COUNTS:
                    totals[key] += report['counts'][key]
                    model_counts[key] += report['counts'][key]
                torch.save(raw[model][image], model_dir / (Path(image).stem + '.pt'))
                new_json(model_dir / (Path(image).stem + '.json'), report)
                adapter.assert_weights_unchanged()
            torch.cuda.synchronize()
            seconds = time.perf_counter()-measured_start
            after = {'unet': fingerprint(unet), 'vae': fingerprint(vae)}
            assert before == after
            after_adapter = adapter_state(unet)
            assert all(torch.equal(loaded[k], after_adapter[k]) for k in loaded)
            adapter.assert_weights_unchanged()
            assert model_counts == c['expected_counts_per_model']
            mr = dict(model=model, images=2, counts=model_counts, measurement_seconds=seconds,
                before_fingerprints=before, after_fingerprints=after,
                target_and_VAE_fingerprints_equal=True, checkpoint_adapter_values_exact=True,
                adapter_tensors_exact_equal=True, parameter_versions_and_no_weight_gradients=True,
                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
                actual_exposure=c['actual_exposure'][model],
                checkpoint_sha256=c['checkpoint_sha256'][model])
            model_reports.append(mr)
            new_json(model_dir / 'execution.json', mr)
            adapter = unet = None
            del measured, loaded, after_adapter
            gc.collect()
            torch.cuda.empty_cache()
        assert_hashes(c['frozen_sha256'])
        assert digest(args.contract) == contract_sha
        assert totals == c['expected_total_counts']
        new_json(out / 'results.json', results)
        torch.save(raw, out / 'raw.pt')
        execution = dict(schema='mofit-medical-remaining-pair-result/v1',
            status='PASS_REMAINING_PAIR_EXECUTION_PENDING_INDEPENDENT_VERIFICATION',
            complete=True, current_step=2, mode='full', model_order=MODEL_ORDER,
            patients=1, unique_images=2, images=4, counts=totals, ended_utc=now(),
            measurement_seconds=sum(m['measurement_seconds'] for m in model_reports),
            total_seconds=time.perf_counter()-started, model_reports=model_reports,
            target_and_VAE_fingerprints_equal=True, checkpoint_adapter_values_exact=True,
            adapter_tensors_exact_equal=True, parameter_versions_and_no_weight_gradients=True,
            frozen_files_unchanged=True, peak_cuda_allocated_bytes=max(m['peak_cuda_allocated_bytes'] for m in model_reports),
            peak_cuda_reserved_bytes=max(m['peak_cuda_reserved_bytes'] for m in model_reports),
            peak_memory_scope='Maximum of separate model kernel windows; counters reset after model loading',
            cuda_device=torch.cuda.get_device_name(0), actual_exposure=c['actual_exposure'],
            results_sha256=digest(out / 'results.json'), raw_sha256=digest(out / 'raw.pt'),
            protocol_sha256=digest(out / 'protocol.json'),
            perform_membership_performance_evaluation=False, new_target_training=False,
            attack_parameter_tuning=False, cohort_expansion=False, stage2_completion_claim=False)
        new_json(out / 'execution.json', execution)
        print(json.dumps(execution, allow_nan=False), flush=True)
    except BaseException as exc:
        new_json(out / 'failure.json', dict(status='FAILED_INCOMPLETE_REMAINING_PAIR_KERNEL',
            ended_utc=now(), seconds=time.perf_counter()-started, error_type=type(exc).__name__,
            error=str(exc), traceback=traceback.format_exc(),
            completed_records=[{'model':r['model'],'image_id':r['image_id']} for r in results],
            completed_counts=totals, current_adapter_counts=dict(adapter.counts) if adapter else None,
            automatic_fallback=False))
        raise
    finally:
        adapter = unet = vae = None
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--contract', type=Path, required=True)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--create-contract', action='store_true')
    mode.add_argument('--dry-run', action='store_true')
    mode.add_argument('--run', action='store_true')
    p.add_argument('--attempt-tag')
    args = p.parse_args()
    if args.create_contract:
        c = create_contract(args.contract)
        print(json.dumps(dict(status='CPU_REMAINING_PAIR_CONTRACT_CREATED_NO_GPU',
            sha256=digest(args.contract), frozen_files=len(c['frozen_sha256']))), flush=True)
        return
    c = load(args.contract)
    payload, metadata, states = validate(c)
    if args.dry_run:
        print(json.dumps(dict(status='PASS_CPU_REMAINING_PAIR_PREFLIGHT_NO_GPU',
            actual_exposure=c['actual_exposure'], image_members=c['image_member_by_model'],
            counts=c['expected_total_counts'], contract_sha256=digest(args.contract))), flush=True)
        return
    execute(args, c, payload, metadata, states)


if __name__ == '__main__':
    main()

