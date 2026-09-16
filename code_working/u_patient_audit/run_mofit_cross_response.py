"""Inference-only crossing of eight frozen MoFit embeddings, images and noises.

This is a fixed-embedding diagnostic, not a new optimization or patient attack.
"""
import argparse
import gc
import json
from importlib.metadata import version
from pathlib import Path
import time
import traceback

import torch
import torch.nn.functional as F

from .common import ROOT, RUN, digest, read_csv
from .models import setup, scheduler, load_unet, adapter_state
from .mofit_medical_adapter import tensor_hash
from .run_mofit_medical_kernel import load, new_json, now, fingerprint, assert_hashes

BASE = RUN / 'baseline_screen_20260915'
MODELS = ['model_1', 'model_2']
IMAGES = ['00014393_002.png', '00014393_006.png',
          '00014393_001.png', '00014393_004.png']
SCENARIO = {i: ('E' if k < 2 else 'U') for k, i in enumerate(IMAGES)}
CHECKPOINTS = {m: RUN / 'training_coverage_v2' / m / 'step_1000.pt' for m in MODELS}
PRIOR = [BASE / 'mofit_medical_full_v1', BASE / 'mofit_medical_target2_full_v1',
         BASE / 'mofit_medical_remaining_pair_v1']
PRIOR_STATUS = ['PASS_SAVED_MOFIT_ARITHMETIC_AND_PROVENANCE_FD_REVIEW_SEPARATE',
                'PASS_TARGET2_SAVED_MOFIT_ARITHMETIC_AND_ACTUAL_NONMEMBERSHIP',
                'PASS_REMAINING_PAIR_SAVED_MOFIT_ARITHMETIC_AND_ACTUAL_MEMBERSHIP']
CONTRACT = BASE / 'mofit_cross_response_contract_v1.json'
OUTPUT = BASE / 'mofit_cross_response_v1'
SEEDS = [260915, 26091501, 26091502, 26091503, 26091504]
RTOL, ATOL = 1e-5, 1e-6


def policy():
    return dict(schema='mofit-cross-response-policy/v1', current_step=2,
        patient_id='14393', eval_role='fit', assignment_group='A',
        model_order=MODELS, image_order=IMAGES, timestep=140, prediction_type='epsilon',
        dtype='float32', batch_size=1, model_mode='eval',
        embedding_phase='final monitor after Adam update 300; embedding_updates[-1].after',
        latent='Saved original posterior sample, identical across target models for each image',
        noise_seeds=SEEDS, noise_rng='CPU torch.Generator manual_seed; randn FP32 shape [1,4,32,32]',
        noise_policy='One original and four fresh noises; shared across all images, embeddings and models',
        condition_order='null first, then source model order and source image order',
        expected_records=360, expected_optimized_records=320, expected_null_records=40,
        expected_forward=360, expected_backward=0, expected_vae_forward=0,
        score='h = conditioned MSE minus same recipient/image/noise null MSE',
        original_replay=dict(optimized_cells=8, null_cells=8, rtol=RTOL, atol=ATOL,
            report_exact_and_absolute_relative_errors=True,
            meaning='FP32 execution agreement only; not a scientific efficacy threshold'),
        analysis=dict(matrix='All 4x4 query-image/source-image cells, for each recipient/source model and noise',
            diagonal='Same source/query image, retaining all four images and five noises',
            noise='Compare frozen-embedding original-noise response with each fresh noise and their fixed mean',
            model_embedding_decomposition='For same image/noise, A=h(M1,e1), B=h(M1,e2), C=h(M2,e1), D=h(M2,e2); recipient=.5*((A-C)+(B-D)); embedding=.5*((A-B)+(C-D)); their sum=A-D',
            aggregation='Report every image and E/U mean and max after member-high orientation; for max aggregate each 2x2 cell before the same decomposition',
            inference='Descriptive only: no AUC, CI, threshold, direction selection, gamma or best-condition selection'),
        limits=['Only one patient and two whole-cohort target models; not individual causal membership',
                'Only same-patient images: cross-image persistence does not establish patient specificity',
                'New noises evaluate fixed embeddings; this is not Table14 noise-specific reoptimization',
                'A response lost under changed noise is not by itself failure of the original coupled attack',
                'No population membership-performance or stage2-completion claim'],
        new_target_training=False, new_optimization=False, new_patients=False,
        fitting=False, score_tuning=False, membership_performance_evaluation=False)


def exposure():
    states, meta = {}, {}
    for m in MODELS:
        q = torch.load(CHECKPOINTS[m], map_location='cpu', weights_only=True)
        manifest = read_csv(RUN / 'cohort' / (m + '_train.csv'))
        assert q['step'] == 1000 and sum(q['exposures'].values()) == 4000
        assert set(q['exposures']) == {r['image_id'] for r in manifest}
        own = [r for r in manifest if r['patient_id'] == '14393']
        member = int(bool(own))
        assert member == int(m == 'model_1')
        total = sum(q['exposures'][r['image_id']] for r in own)
        assert total == 8 * member and len(own) == 2 * member
        meta[m] = {}
        for i in IMAGES:
            n = q['exposures'].get(i, 0)
            assert n == (4 if member and SCENARIO[i] == 'E' else 0)
            meta[m][i] = dict(patient_id='14393', eval_role='fit', assignment_group='A',
                scenario=SCENARIO[i], member=member, image_member=int(n > 0),
                actual_training_exposures=n, patient_training_exposures=total)
        states[m] = q
    return states, meta


def sources():
    packets, rows, hashes = {}, {}, {}
    for k, p in enumerate(PRIOR):
        v, e, r = load(p / 'verification.json'), load(p / 'execution.json'), load(p / 'results.json')
        assert v['status'] == PRIOR_STATUS[k] and e['complete'] is True
        for name in ['results', 'raw', 'protocol']:
            f = p / (name + ('.pt' if name == 'raw' else '.json'))
            h = digest(f)
            assert v[name + '_sha256'] == e[name + '_sha256'] == h
        raw = torch.load(p / 'raw.pt', map_location='cpu', weights_only=True)
        for row in r:
            m, i = row['model'], row['image_id']
            assert (m, i) not in packets and m in MODELS and i in IMAGES
            q = raw[m][i] if k == 2 else raw[i]
            assert row['stage1_steps'] == 1000 and row['stage2_steps'] == 300
            assert q['embedding_updates'][-1]['iteration'] == 299
            emb = q['final']['embedding']
            assert torch.equal(emb, q['embedding_updates'][-1]['after'])
            assert tensor_hash(emb) == q['traces']['embedding'][-1]['after_sha256']
            assert tuple(emb.shape) == (1, 77, 1024) and emb.dtype == torch.float32
            scores = q['final']['original_scores']
            for key in ['u', 'h', 'v']:
                assert row[key] == scores[key] == q['traces']['embedding'][-1][key]
            latent = q['posterior']['original']['scaled_sample']
            assert tuple(latent.shape) == (1, 4, 32, 32) and latent.dtype == torch.float32
            assert torch.equal(latent, scores['optimized_cell']['scaled_latent'])
            assert torch.equal(latent, scores['null_cell']['scaled_latent'])
            assert torch.equal(q['draws']['diffusion_noise'], scores['optimized_cell']['target'])
            assert torch.equal(q['draws']['diffusion_noise'], scores['null_cell']['target'])
            packets[m, i], rows[m, i] = q, row
        for name in ['verification.json', 'verification_protocol.json', 'execution.json',
                     'protocol.json', 'results.json', 'raw.pt']:
            hashes[str((p / name).resolve())] = digest(p / name)
    assert len(packets) == 8
    null = packets[MODELS[0], IMAGES[0]]['null_hidden']
    noise = torch.randn((1, 4, 32, 32), generator=torch.Generator().manual_seed(SEEDS[0]))
    for m in MODELS:
        for i in IMAGES:
            q = packets[m, i]
            assert torch.equal(q['null_hidden'], null)
            assert torch.equal(q['draws']['diffusion_noise'], noise)
            assert torch.equal(q['posterior']['original']['scaled_sample'],
                               packets[MODELS[0], i]['posterior']['original']['scaled_sample'])
            assert torch.equal(q['initial_hidden'], packets[MODELS[0], i]['initial_hidden'])
            assert torch.equal(q['initial_pixels'], packets[MODELS[0], i]['initial_pixels'])
    return packets, rows, hashes


def prepare(path):
    assert not path.exists(), 'Immutable contract already exists'
    old = load(BASE / 'mofit_medical_remaining_pair_contract_v1.json')
    frozen = dict(old['frozen_sha256'])
    assert_hashes(frozen)
    _, _, bindings = sources()
    states, metadata = exposure()
    frozen.update(bindings)
    for p in [Path(__file__), BASE / 'mofit_medical_remaining_pair_contract_v1.json']:
        frozen[str(p.resolve())] = digest(p)
    c = dict(schema='mofit-cross-response-contract/v1', created_utc=now(),
        policy=policy(), checkpoint_sha256={m: digest(p) for m, p in CHECKPOINTS.items()},
        metadata=metadata, original_source_bindings=bindings,
        alpha_cumprod_t140=old['alpha_cumprod_t140'], frozen_sha256=frozen,
        prior_unet_fingerprints={m: load(PRIOR[k] / 'execution.json')['before_fingerprints']['unet']
                                 for k, m in enumerate(MODELS)})
    new_json(path, c)
    return c


def validate(c):
    assert c['schema'] == 'mofit-cross-response-contract/v1' and c['policy'] == policy()
    assert_hashes(c['frozen_sha256'])
    assert c['frozen_sha256'][str(Path(__file__).resolve())] == digest(Path(__file__))
    packets, rows, hashes = sources()
    assert hashes == c['original_source_bindings']
    states, metadata = exposure()
    assert metadata == c['metadata']
    for m, p in CHECKPOINTS.items():
        assert digest(p) == c['checkpoint_sha256'][m]
        for i in IMAGES:
            for key in ['patient_id', 'eval_role', 'assignment_group', 'scenario', 'member',
                        'image_member', 'actual_training_exposures']:
                assert rows[m, i][key] == metadata[m][i][key]
    s = scheduler()
    assert s.config.prediction_type == 'epsilon'
    assert float(s.alphas_cumprod[140]) == c['alpha_cumprod_t140']
    return packets, rows, states


def agreement(a, b):
    a, b = a.detach().cpu(), b.detach().cpu()
    assert a.shape == b.shape and a.dtype == b.dtype == torch.float32
    delta = (a.double() - b.double()).abs()
    return dict(exact=torch.equal(a, b), close=bool(torch.allclose(a, b, rtol=RTOL, atol=ATOL)),
        max_absolute_error=float(delta.max()),
        relative_l2_error=float(torch.linalg.vector_norm(delta) /
                                torch.linalg.vector_norm(b.double()).clamp_min(1e-30)))


def self_test():
    a = torch.arange(12, dtype=torch.float32).reshape(1, 3, 2, 2)
    assert agreement(a, a)['exact']
    assert not agreement(a, a + 1.)['close']
    # The two-path average is an exact algebraic decomposition, including max pooling.
    x = torch.tensor([[1., 3.], [2., -1.]], dtype=torch.float64)
    A, B, C, D = x.flatten().tolist()
    assert .5*((A-C)+(B-D)) + .5*((A-B)+(C-D)) == A-D
    assert 2*4*(2*4+1)*len(SEEDS) == policy()['expected_forward']
    assert len(set(SEEDS)) == 5
    return dict(status='PASS_CPU_FIXED_DESIGN_AND_ARITHMETIC', CUDA_initialized=torch.cuda.is_initialized())


def execute(args, c, packets, oldrows, states):
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    start, contract_hash = time.perf_counter(), digest(args.contract)
    new_json(out / 'protocol.json', dict(schema='mofit-cross-response-execution/v1',
        started_utc=now(), contract_path=str(args.contract.resolve()), contract_sha256=contract_hash,
        contract=c, code_sha256=digest(Path(__file__)),
        environment={n: version(n) for n in ['torch', 'diffusers', 'transformers', 'peft', 'numpy']}))
    shared = dict(latents={i: packets[MODELS[0], i]['posterior']['original']['scaled_sample'] for i in IMAGES},
        embeddings={m: {i: packets[m, i]['final']['embedding'] for i in IMAGES} for m in MODELS},
        null_hidden=packets[MODELS[0], IMAGES[0]]['null_hidden'],
        noises={str(k): torch.randn((1, 4, 32, 32), generator=torch.Generator().manual_seed(seed))
                for k, seed in enumerate(SEEDS)})
    raw, results, replays, reports = dict(shared=shared, cells={}), [], [], []
    unet = None
    try:
        setup()
        sched = scheduler()
        t = torch.tensor([140], dtype=torch.long, device='cuda')
        assert sched.config.prediction_type == 'epsilon'
        assert float(sched.alphas_cumprod[140]) == c['alpha_cumprod_t140']
        for m in MODELS:
            unet = load_unet(CHECKPOINTS[m], training=False).float().requires_grad_(False).eval()
            before = fingerprint(unet)
            assert before == c['prior_unet_fingerprints'][m]
            loaded = adapter_state(unet)
            assert loaded.keys() == states[m]['adapter'].keys()
            assert all(torch.equal(v, states[m]['adapter'][k].float()) for k, v in loaded.items())
            versions = {n: p._version for n, p in unet.named_parameters()}
            counter = dict(calls=0, examples=0)
            def counted(_module, inputs, _output):
                counter['calls'] += 1
                counter['examples'] += inputs[0].shape[0]
            hook = unet.register_forward_hook(counted)
            torch.cuda.synchronize()
            ms = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                for i in IMAGES:
                    latent = shared['latents'][i].to('cuda')
                    for noise_index, seed in enumerate(SEEDS):
                        epsilon = shared['noises'][str(noise_index)].to('cuda')
                        noisy = sched.add_noise(latent, epsilon, t)
                        conditions = [(None, None)] + [(sm, si) for sm in MODELS for si in IMAGES]
                        null_loss = None
                        for sm, si in conditions:
                            hidden = shared['null_hidden'] if sm is None else shared['embeddings'][sm][si]
                            prediction = unet(noisy, t, encoder_hidden_states=hidden.to('cuda')).sample
                            assert not prediction.requires_grad and bool(torch.isfinite(prediction).all())
                            loss = float(F.mse_loss(prediction.float(), epsilon.float(), reduction='mean'))
                            if sm is None:
                                null_loss = loss
                            cell_id = f'{m}|{i}|{sm or "null"}|{si or "null"}|{noise_index}'
                            cell = dict(prediction=prediction.cpu(), noised_latent=noisy.cpu(),
                                        target=epsilon.cpu(), loss=loss)
                            raw['cells'][cell_id] = cell
                            row = dict(cell_id=cell_id, recipient_model=m, query_image_id=i,
                                query_scenario=SCENARIO[i], source_model=sm, source_image_id=si,
                                source_scenario=SCENARIO[si] if si else None,
                                condition='null' if sm is None else 'optimized',
                                noise_index=noise_index, noise_seed=seed, loss=loss,
                                null_loss=null_loss, h=loss-null_loss, forward=1, backward=0,
                                query_metadata=c['metadata'][m][i],
                                source_metadata=c['metadata'][sm][si] if sm else None)
                            results.append(row)
                            if noise_index == 0 and (sm is None or (sm == m and si == i)):
                                key = 'null_cell' if sm is None else 'optimized_cell'
                                old = packets[m, i]['final']['original_scores'][key]
                                comparisons = {k: agreement(cell[k], old[k])
                                               for k in ['prediction', 'noised_latent', 'target']}
                                comparisons['loss'] = agreement(torch.tensor(loss), torch.tensor(old['loss']))
                                comparisons['h'] = agreement(torch.tensor(row['h']),
                                    torch.tensor(0. if sm is None else oldrows[m, i]['h']))
                                replays.append(dict(cell_id=cell_id, original_cell=key, comparisons=comparisons))
                                assert all(x['close'] for x in comparisons.values()), 'Original diagonal replay mismatch'
                    print(json.dumps(dict(event='cross_response_image_done', model=m, image_id=i,
                        cumulative_records=len(results), seconds=time.perf_counter()-start)), flush=True)
            torch.cuda.synchronize()
            hook.remove()
            assert counter == dict(calls=180, examples=180)
            assert all(not p.requires_grad and p.grad is None and p._version == versions[n]
                       for n, p in unet.named_parameters())
            after = fingerprint(unet)
            assert before == after
            after_adapter = adapter_state(unet)
            assert all(torch.equal(v, after_adapter[k]) for k, v in loaded.items())
            reports.append(dict(model=m, forward_calls=180, forward_examples=180, backward=0,
                before_fingerprint=before, after_fingerprint=after,
                checkpoint_adapter_exact=True, adapter_tensors_unchanged=True,
                parameter_versions_unchanged=True, all_parameters_frozen=True, no_parameter_gradients=True,
                model_training=unet.training, seconds=time.perf_counter()-ms,
                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated()))
            unet = None
            del loaded, after_adapter, prediction
            gc.collect()
            torch.cuda.empty_cache()
        assert len(results) == len(raw['cells']) == 360 and len(replays) == 16
        assert_hashes(c['frozen_sha256'])
        assert digest(args.contract) == contract_hash
        new_json(out / 'results.json', results)
        new_json(out / 'original_replay.json', replays)
        torch.save(raw, out / 'raw.pt')
        new_json(out / 'execution.json', dict(schema='mofit-cross-response-result/v1',
            status='PASS_CROSS_RESPONSE_EXTRACTION_PENDING_INDEPENDENT_VERIFICATION', complete=True,
            current_step=2, records=360, optimized_records=320, null_records=40,
            patients=1, images=4, source_embeddings=8, noises=5,
            forward=360, backward=0, vae_forward=0, original_replay_records=16,
            shared_original_latents_exact_across_models=True, final_embedding_after_update_verified=True,
            model_reports=reports, seconds=time.perf_counter()-start, ended_utc=now(),
            input_bindings_unchanged=True, contract_sha256=contract_hash,
            protocol_sha256=digest(out/'protocol.json'), results_sha256=digest(out/'results.json'),
            raw_sha256=digest(out/'raw.pt'), original_replay_sha256=digest(out/'original_replay.json'),
            scientific_efficacy_pass=False, membership_performance_evaluation=False,
            new_optimization=False, new_target_training=False))
    except BaseException:
        torch.save(raw, out / 'partial_raw.pt')
        new_json(out / 'failure.json', dict(status='FAILED_PRESERVED', traceback=traceback.format_exc(),
            records=len(results), seconds=time.perf_counter()-start))
        raise


def main():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--prepare-contract', action='store_true')
    g.add_argument('--dry-run', action='store_true')
    g.add_argument('--self-test', action='store_true')
    g.add_argument('--run', action='store_true')
    p.add_argument('--contract', type=Path, default=CONTRACT)
    p.add_argument('--output-dir', type=Path, default=OUTPUT)
    p.add_argument('--expected-code-sha256')
    a = p.parse_args()
    if a.expected_code_sha256:
        assert digest(Path(__file__)) == a.expected_code_sha256
    if a.self_test:
        print(json.dumps(self_test()))
        return
    if a.prepare_contract:
        c = prepare(a.contract)
        print(json.dumps(dict(contract=str(a.contract), sha256=digest(a.contract),
            bound_files=len(c['frozen_sha256']), CUDA_initialized=torch.cuda.is_initialized())))
        return
    c = load(a.contract)
    packets, rows, states = validate(c)
    if a.dry_run:
        print(json.dumps(dict(status='PASS_CPU_SOURCE_PHASE_LATENT_EXPOSURE_AND_HASH_PREFLIGHT',
            packets=len(packets), expected_forward=360, CUDA_initialized=torch.cuda.is_initialized())))
        return
    execute(a, c, packets, rows, states)


if __name__ == '__main__':
    main()
