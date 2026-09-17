"""Fixed-checkpoint inference only. No optimizer, backward, generation or BN calibration."""
from downstream_utility.run_v2 import install_guard
install_guard()
from .common import *
from collections import Counter
import argparse, csv, gc, os, sys, time

def prepare():
    require(not OUT.exists(), 'No overwrite or silent restart')
    from downstream_utility.development_v1 import allowed_real
    rr = allowed_real()
    real = {r['image_id']: r for r in rr}
    audit = {r['image_id']: r for r in read(PROFILE / 'image_audit.json')}
    gm = {r['image_id']: r for r in read(LORA / 'generation_manifest.json')}
    selected, runs, sources = {}, [], set()
    for arm in ARMS:
        parent = OLD if arm in ('R1', 'Dreal') else LORA
        for seed in SEEDS:
            folder = parent / 'runs' / f'{arm}_{seed}'
            for n, h in read(folder / 'complete.json')['files'].items():
                require(sha(folder / n) == h, 'Prior completed artifact changed')
            trace = read(folder / 'trace.json')
            require(trace['steps'] == 400 and len(trace['trace']) == 400, 'Fixed 400 steps')
            exposure = Counter()
            for step, b in enumerate(trace['trace'], 1):
                require(b['step'] == step and len(b['image_ids']) == 32, 'Trace batch count')
                for j, iid in enumerate(b['image_ids']):
                    ns = b['source_namespaces'][j]
                    expected_ns = ['real/public']
                    if arm == 'Dreal': expected_ns += ['real/private']
                    if arm.startswith('L_'): expected_ns += ['synthetic/lora_' + arm[2:]]
                    require(ns in expected_ns, 'Arm source boundary')
                    if iid in real:
                        r = real[iid]
                        require(r['role'] in ('public', 'private'), 'Training role boundary')
                        rec = dict(image_id=iid, patient_or_cell_id=str(r['patient_id']), namespace='real/' + r['role'],
                                   label=int(r['label']), path=str(Path(r['path']).resolve()),
                                   file_sha256=r['sha256'], pixel_sha256=audit[iid]['tensor_sha256'],
                                   label_type='NIH_weak_image_label')
                    else:
                        g = gm[iid]
                        require(g['arm'] == arm, 'Synthetic arm binding')
                        rec = dict(image_id=iid, patient_or_cell_id=g['cell'], namespace='synthetic/' + g['method'],
                                   label=int(g['label']), path=str((LORA / g['image_path']).resolve()),
                                   file_sha256=g['image_sha256'], pixel_sha256=b['source_pixel_sha256'][j],
                                   label_type='requested_condition_not_expert_ground_truth')
                    require(rec['namespace'] == ns and rec['label'] == int(b['labels'][j]), 'Trace label/source')
                    require(rec['patient_or_cell_id'] == str(b['patient_ids'][j]), 'Patient/cell binding')
                    require(rec['file_sha256'] == b['source_file_sha256'][j], 'File binding')
                    require(rec['pixel_sha256'] == b['source_pixel_sha256'][j], 'Pixel binding')
                    if iid in selected: require(selected[iid] == rec, 'Unique consistent records')
                    selected[iid] = rec
                    exposure[iid] += 1
            require(sum(exposure.values()) == 12800, '400 x 32 exposures')
            expected_exposure = {r['image_id']: r['exposures'] for r in read(folder / 'exposure.json')['per_image']}
            require(dict(exposure) == expected_exposure, 'Trace vs exposure file')
            prediction = parent / 'evaluation' / f'{arm}_{seed}.npz'
            with np.load(prediction, allow_pickle=False) as packet:
                require(len(packet['logits']) == 5047, 'Stored development count')
                dev_ids = packet['image_ids'].tolist()
                require(dev_ids == [r['image_id'] for r in rr if r['role'] == 'method_development'], 'Dev order')
                witness = dev_ids[:64]
            runs.append(dict(arm=arm, seed=seed, checkpoint=str(folder/'state_400.pt'), trace=str(folder/'trace.json'),
                             exposure_file=str(folder/'exposure.json'), stored_development=str(prediction),
                             image_ids=sorted(exposure), exposure=dict(sorted(exposure.items())), witness_ids=witness))
            sources.update([folder/n for n in ['complete.json','state_400.pt','trace.json','exposure.json']])
            sources.add(prediction)
    witnesses = runs[0]['witness_ids']
    require(all(r['witness_ids'] == witnesses for r in runs), 'Common deterministic witness')
    for iid in witnesses:
        r = real[iid]
        require(r['role'] == 'method_development' and iid not in selected, 'Disjoint witness')
        selected[iid] = dict(image_id=iid, patient_or_cell_id=str(r['patient_id']), namespace='witness/method_development',
                            label=int(r['label']), path=str(Path(r['path']).resolve()), file_sha256=r['sha256'],
                            pixel_sha256=audit[iid]['tensor_sha256'], label_type='NIH_weak_image_label')
    require(len(selected) == 1453 and len(witnesses) == 64, 'Declared pixel scope')
    for parent in (OLD, LORA):
        sources.update(parent/n for n in ['contract.json','result.json','verification.json','generation_manifest.json'])
    sources.update(PROFILE/n for n in ['real_manifest_private.csv','image_audit.json'])
    sources.update(Path(__file__).parent.glob('*.py'))
    sources.update(CODE/'downstream_utility'/n for n in ['common.py','run_v2.py','data_v2.py','train_v2.py','development_v1.py'])
    sources.update([PROTOCOL, CODE/'_reports/patient_usage_audit_20260917_v1/patient_usage_private.csv',
                    CODE/'_reports/patient_usage_audit_20260917_v1/provisional_patient_split_private.csv'])
    sources.update(RESEARCH/n for n in ['TRACK1_CURRENT_HEAD_BRANCH_CLOSURE_20260917.md',
                     'TRACK1_DOWNSTREAM_DEVELOPMENT_RESULTS_20260917.md','TRACK1_LORA_TRANSFER_COMPARISON_RESULTS_20260917.md'])
    OUT.mkdir()
    save(OUT/'manifest_private.json', list(sorted(selected.values(), key=lambda r:r['image_id'])))
    save(OUT/'runs_private.json', runs)
    save(OUT/'state_before.json', {n:read(RESEARCH/n) for n in ['research_state.json','patient_baseline_spec.json']})
    sources.update([OUT/'manifest_private.json', OUT/'runs_private.json', OUT/'state_before.json'])
    budget = dict(unique_training_images=1389, witness_images=64, raw_or_PNG_decode=1453,
                  training_image_presentations=sum(len(r['image_ids']) for r in runs),
                  witness_presentations=64*12, models=12, optimizer_updates=0, new_generation=0,
                  expert_pixels=0, reserved_pixels=0, DP=0)
    save(OUT/'contract.json', dict(schema='fixed-classifier-generalization/v1', created_utc=now(),
         source_sha256={str(p.resolve()):sha(p) for p in sorted(sources)}, budget=budget,
         arms=list(ARMS), seeds=list(SEEDS), batch_size=32, checkpoint_steps=400, witness_absolute_tolerance=1e-5,
         parameter_and_buffer_mutation_allowed=False, calibration_allowed=False, final_ready=False,
         expert_allowed=False, reserved_allowed=False, generation_allowed=False, training_allowed=False,
         scope='Stored models, actually seen unique train images, separate source metrics; old development reused',
         protocol=str(PROTOCOL)))
    log(phase='contract_frozen_before_inference', **budget)

def install_access_guard(records):
    allowed = {os.path.normcase(str(Path(r['path']).resolve())) for r in records}
    pixel_roots = [os.path.normcase(str(RAW.resolve()))+os.sep,
                   os.path.normcase(str((LORA/'generation').resolve()))+os.sep]
    opened = Counter()
    def audit(event, args):
        if event != 'open' or not args or not isinstance(args[0], (str, bytes, os.PathLike)): return
        p = os.path.normcase(os.path.abspath(os.fsdecode(args[0])))
        if any(p.startswith(root) for root in pixel_roots):
            require(p in allowed, 'Blocked non-allowlisted pixel path')
            opened[p] += 1
    sys.addaudithook(audit)
    return opened

def metrics(y, z, counts=None):
    from sklearn.metrics import roc_auc_score, average_precision_score
    y, z = np.asarray(y, dtype=np.int64), np.asarray(z, dtype=np.float64)
    loss = np.logaddexp(0., z) - y*z
    q = [0, .05, .25, .5, .75, .95, 1]
    result = dict(images=len(y), positive_images=int(y.sum()), prevalence=float(y.mean()),
                  AUROC=float(roc_auc_score(y,z)), AP=float(average_precision_score(y,z)),
                  BCE=float(loss.mean()), balanced_BCE=float(np.mean([loss[y==v].mean() for v in (0,1)])),
                  by_label={str(v):dict(images=int(sum(y==v)), BCE=float(loss[y==v].mean()),
                    logit_mean=float(z[y==v].mean()), logit_quantiles=np.quantile(z[y==v],q).tolist()) for v in (0,1)})
    if counts is not None:
        counts=np.asarray(counts,dtype=np.int64)
        result.update(exposure_weighted_BCE=float(np.average(loss,weights=counts)),
                      exposures=int(counts.sum()),exposure_min=int(counts.min()),
                      exposure_median=float(np.median(counts)),exposure_max=int(counts.max()))
    return result

def infer():
    from PIL import Image
    from downstream_utility import data_v2
    from torchvision.models import resnet18
    import torch
    c=read(OUT/'contract.json');validate_sources(c)
    require(not (OUT/'inference_started.json').exists(), 'No implicit rerun; preserve partial output')
    save(OUT/'inference_started.json',dict(utc=now(),contract_sha256=sha(OUT/'contract.json')))
    setup();tick=time.perf_counter()
    records=read(OUT/'manifest_private.json');lookup={r['image_id']:r for r in records}
    runs=read(OUT/'runs_private.json')
    opened=install_access_guard(records)
    arrays={};bindings=[]
    for r in records:
        require(sha(r['path'])==r['file_sha256'],'Pixel file SHA changed')
        with Image.open(r['path']) as im: a=data_v2.letterbox(im)
        require(tensor_sha(a)==r['pixel_sha256'],'Decoded letterbox differs from historical training input')
        arrays[r['image_id']]=a
        bindings.append(dict(image_id=r['image_id'],file_sha256=r['file_sha256'],pixel_sha256=tensor_sha(a)))
    save(OUT/'pixel_bindings_private.json',bindings)
    log(phase='pixels_verified',images=len(arrays),seconds=time.perf_counter()-tick)
    outputs=[];model_records=[];(OUT/'predictions').mkdir()
    for r in runs:
        name=f"{r['arm']}_{r['seed']}"
        state=torch.load(r['checkpoint'],map_location='cpu',weights_only=True)
        full_before=state_hash(state)
        model=resnet18(weights=None);model.fc=torch.nn.Linear(model.fc.in_features,1)
        model.load_state_dict(state,strict=True);model.requires_grad_(False);model.cuda().eval()
        require(all(not m.training for m in model.modules()),'All modules evaluation mode')
        buffers_before={k:v.detach().cpu().clone() for k,v in model.named_buffers()}
        params_before={k:v.detach().cpu().clone() for k,v in model.named_parameters()}
        model_start=time.perf_counter();batches=[]
        def forward(ids, phase):
            z=[]
            with torch.inference_mode():
                for start in range(0,len(ids),32):
                    batch_ids=ids[start:start+32]
                    x=data_v2.preprocess_batch([arrays[i] for i in batch_ids],'R0',0,0,'cuda')
                    require(torch.is_inference_mode_enabled() and not torch.is_grad_enabled(),'Inference-only mode')
                    pred=model(x).flatten()
                    require(bool(torch.isfinite(pred).all()),'Finite prediction')
                    batches.append(dict(phase=phase,image_ids=batch_ids,input_sha256=tensor_sha(x)))
                    z.extend(pred.cpu().tolist())
            return np.asarray(z,dtype=np.float64)
        witness=forward(r['witness_ids'],'development_path_witness')
        with np.load(r['stored_development'],allow_pickle=False) as old:
            packet={k:old[k].copy() for k in old.files}
        diff=float(np.max(np.abs(witness-packet['logits'][:64])))
        require(diff<=c['witness_absolute_tolerance'],'Existing evaluation path parity failed')
        z=forward(r['image_ids'],'actually_seen_training_images')
        y=np.array([lookup[i]['label'] for i in r['image_ids']],dtype=np.int64)
        ns=np.array([lookup[i]['namespace'] for i in r['image_ids']])
        patients=np.array([lookup[i]['patient_or_cell_id'] for i in r['image_ids']])
        counts=np.array([r['exposure'][i] for i in r['image_ids']],dtype=np.int64)
        npz(OUT/'predictions'/f'{name}.npz',image_ids=np.array(r['image_ids']),labels=y,logits=z,namespaces=ns,
            patient_or_cell_ids=patients,exposures=counts,witness_logits=witness)
        after=model.state_dict()
        require(state_hash(after)==full_before,'Full model state mutated')
        require(all(torch.equal(v,dict(model.named_parameters())[k].detach().cpu()) for k,v in params_before.items()),'Parameter mutation')
        require(all(torch.equal(v,dict(model.named_buffers())[k].detach().cpu()) for k,v in buffers_before.items()),'Buffer/BN mutation')
        require(all(p.grad is None for p in model.parameters()),'Unexpected gradient')
        components={}
        for source in sorted(set(ns)):
            mask=ns==source;part=metrics(y[mask],z[mask],counts[mask])
            part.update(unique_patients=len(set(patients[mask])) if source.startswith('real/') else None,
                        unique_synthetic_cells=len(set(patients[mask])) if source.startswith('synthetic/') else None,
                        label_type=lookup[r['image_ids'][int(np.flatnonzero(mask)[0])]]['label_type'])
            components[source]=part
        trace=read(r['trace'])['trace'];training_log={}
        for window in [50,400]:
            b=trace[-window:];source_losses={}
            for row in b:
                losses=np.logaddexp(0.,np.array(row['logits']))-np.array(row['labels'])*np.array(row['logits'])
                for source,v in zip(row['source_namespaces'],losses):source_losses.setdefault(source,[]).append(float(v))
            training_log[str(window)]=dict(mean_logged_BCE=float(np.mean([b['loss'] for b in b])),
                                           source_BCE={k:float(np.mean(v)) for k,v in source_losses.items()})
        # Explicit first50 and last50 summaries without retraining.
        first=trace[:50];first_source={}
        for b in first:
            losses=np.logaddexp(0.,np.array(b['logits']))-np.array(b['labels'])*np.array(b['logits'])
            for source,v in zip(b['source_namespaces'],losses):first_source.setdefault(source,[]).append(float(v))
        training_log['first50']=dict(mean_logged_BCE=float(np.mean([b['loss'] for b in first])),
                                    source_BCE={k:float(np.mean(v)) for k,v in first_source.items()})
        output=dict(arm=r['arm'],seed=r['seed'],components=components,
                    development=metrics(packet['labels'],packet['logits']),training_log=training_log)
        outputs.append(output)
        rec=dict(run=name,checkpoint_sha256=sha(r['checkpoint']),full_state_before=full_before,full_state_after=state_hash(after),
                 parameter_tensors=len(params_before),buffer_tensors=len(buffers_before),
                 BN_running_buffers=sum(k.endswith(('running_mean','running_var','num_batches_tracked')) for k in buffers_before),
                 all_parameters_exact=True,all_buffers_exact=True,eval_all_modules=True,inference_mode=True,
                 witness_max_absolute_difference=diff,witness_exact=bool(np.array_equal(witness,packet['logits'][:64])),
                 training_images=len(y),witness_images=64,seconds=time.perf_counter()-model_start,batches=batches)
        save(OUT/'predictions'/f'{name}_execution.json',rec);model_records.append(rec)
        log(phase='model_done',run=name,train_images=len(y),witness_max_difference=diff,
            source_AUROC={k:v['AUROC'] for k,v in components.items()},dev_AUROC=output['development']['AUROC'])
        del model,state,after,params_before,buffers_before;gc.collect();torch.cuda.empty_cache()
    validate_sources(c)
    save(OUT/'result.json',dict(schema=c['schema'],created_utc=now(),contract_sha256=sha(OUT/'contract.json'),
         model_results=outputs,budget=c['budget'],actual_training_presentations=sum(r['training_images'] for r in model_records),
         actual_witness_presentations=64*len(model_records),all_model_states_unchanged=True,
         maximum_witness_logit_difference=max(r['witness_max_absolute_difference'] for r in model_records),
         runtime_modules={n:str(m.__file__) for n,m in sys.modules.items() if n.startswith(('classifier_generalization','downstream_utility')) and getattr(m,'__file__',None)},
         pixel_open_events=len(opened),seconds=time.perf_counter()-tick,method_efficacy_retested=False,
         causal_failure_source_established=False,optimizer_updates=0,new_generation=0,DP=0,
         expert_pixels=0,reserved_pixels=0,final_ready=False))
    save(OUT/'inference_complete.json',dict(utc=now(),files={str(p.relative_to(OUT)):sha(p) for p in sorted((OUT/'predictions').glob('*'))},
                                         result_sha256=sha(OUT/'result.json')))
    log(phase='inference_complete',models=12,seconds=time.perf_counter()-tick)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','infer'])
    args=p.parse_args()
    (prepare if args.phase=='prepare' else infer)()
