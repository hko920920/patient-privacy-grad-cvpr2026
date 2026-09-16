"""CPU-only, pinned medical-BLIP captions and fresh FP32 CLIP initialization.

Preparation for the fixed fit14393 MoFit numerical/runtime kernel, not fitting.
"""
import gc
import json
import time
from pathlib import Path

from .common import ROOT, RUN, IMAGES, MODEL_ID, MODEL_REVISION, digest, read_csv, verify_inputs
from dp_protocol.calibrate_xray_public_clip_norms import disable_broken_optional_onnx
disable_broken_optional_onnx()
import torch
from huggingface_hub import snapshot_download
from transformers import BlipProcessor, BlipForConditionalGeneration, CLIPTokenizer, CLIPTextModel
from data_pipeline import nih_cxr14_model_input as mi
from .mofit_medical_adapter import make_draws, tensor_hash, kernel_policy

BLIP_ID = 'Siddartha01/blip-medical-captioning-roco'
BLIP_REVISION = 'f8a8378a9013e8e6330a930dc0b405798d326c56'
BLIP_WEIGHT_SHA256 = '815efd21dd1b889afbf56c3b3575e432c03f0468946db340321cfe4ce562a9cd'
OUTPUT = RUN / 'baseline_screen_20260915/mofit_medical_inputs_v1'


def save_json(path, value):
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def main():
    if OUTPUT.exists():
        raise RuntimeError('Immutable input preparation directory already exists')
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    verify_inputs()
    start = time.perf_counter()
    blip_path = Path(snapshot_download(BLIP_ID, revision=BLIP_REVISION, local_files_only=True))
    sd_path = Path(snapshot_download(MODEL_ID, revision=MODEL_REVISION, local_files_only=True))
    if digest(blip_path / 'model.safetensors') != BLIP_WEIGHT_SHA256:
        raise RuntimeError('Medical BLIP weight hash mismatch')
    rows = sorted([r for r in read_csv(RUN / 'cohort/evaluation_images.csv')
                   if r['patient_id'] == '14393'], key=lambda r: r['image_id'])
    if len(rows) != 4 or {r['eval_role'] for r in rows} != {'fit'}:
        raise RuntimeError('Expected fixed fit14393 E2/U2')
    preprocessing_path = Path(mi.__file__)
    files = {str(p): digest(p) for p in sorted(blip_path.iterdir()) if p.is_file()}
    for component in ['tokenizer', 'text_encoder']:
        for p in sorted((sd_path / component).iterdir()):
            if p.is_file():
                files[str(p)] = digest(p)
    for p in [Path(__file__), preprocessing_path, RUN / 'cohort/lock.json',
              RUN / 'cohort/evaluation_images.csv', Path(__file__).with_name('mofit_medical_adapter.py')]:
        files[str(p)] = digest(p)
    policy = {'purpose': 'fixed numerical/runtime preparation, not patient performance',
              'patient_id': '14393', 'image_order': [r['image_id'] for r in rows],
              'BLIP_ID': BLIP_ID, 'BLIP_revision': BLIP_REVISION,
              'caption_generation': {'max_length': 50, 'do_sample': False, 'num_beams': 1},
              'BLIP_input': 'same NIH full-field Lanczos P256 RGB image then pinned BLIP processor',
              'caption_replaces': 'public COCO precomputed BLIP2 captions; this is medical-BLIP initialization',
              'CLIP_policy': 'FP16 artifact loaded then promoted FP32 before fresh CPU encoding; max77 incl padding',
              'null_hidden': 'fresh empty-prompt CLIP FP32, not old FP16 hidden promoted',
              'trust_remote_code': False, 'device': 'cpu', 'kernel_policy': kernel_policy(),
              'source_sha256': files}
    OUTPUT.mkdir(parents=True)
    save_json(OUTPUT / 'preparation_protocol.json', policy)
    processor = BlipProcessor.from_pretrained(blip_path, local_files_only=True, trust_remote_code=False)
    captioner = BlipForConditionalGeneration.from_pretrained(blip_path, use_safetensors=True,
                       torch_dtype=torch.float32, local_files_only=True, trust_remote_code=False).eval()
    captioner.requires_grad_(False)
    images = {}
    for row in rows:
        path = IMAGES / row['image_filename']
        pil, pre_meta = mi.preprocess_path(path, 256)
        pixels = mi.pil_to_normalized_tensor(pil).unsqueeze(0)
        inputs = processor(images=pil, return_tensors='pt')
        with torch.inference_mode():
            tokens = captioner.generate(**inputs, max_length=50, do_sample=False, num_beams=1)
        caption = processor.decode(tokens[0], skip_special_tokens=True).strip()
        if not caption:
            raise RuntimeError('Empty medical BLIP caption')
        images[row['image_id']] = {'metadata': row, 'pixels': pixels,
                  'pixels_sha256': tensor_hash(pixels), 'original_file_sha256': digest(path),
                  'caption': caption, 'caption_tokens': tokens.cpu(),
                  'BLIP_pixel_values_sha256': tensor_hash(inputs['pixel_values'])}
        print(json.dumps({'phase': 'CPU_BLIP_caption', 'image_id': row['image_id'],
                          'completed': len(images), 'total': len(rows)}), flush=True)
    del captioner, processor
    gc.collect()
    tokenizer = CLIPTokenizer.from_pretrained(sd_path / 'tokenizer', local_files_only=True)
    text = CLIPTextModel.from_pretrained(sd_path / 'text_encoder', torch_dtype=torch.float16,
                  variant='fp16', use_safetensors=True, local_files_only=True).float().eval()
    text.requires_grad_(False)
    prompts = sorted({item['caption'] for item in images.values()} | {''})
    hidden = {}
    token_inputs = {}
    with torch.inference_mode():
        for prompt in prompts:
            tokens = tokenizer(prompt, padding='max_length', max_length=77,
                               truncation=True, return_tensors='pt')
            h = text(tokens.input_ids)[0].contiguous()
            if tuple(h.shape) != (1, 77, 1024) or not torch.isfinite(h).all():
                raise RuntimeError('CLIP shape/finite mismatch')
            hidden[prompt] = h.cpu()
            token_inputs[prompt] = {'input_ids': tokens.input_ids.cpu(),
                                    'attention_mask': tokens.attention_mask.cpu()}
    del text
    gc.collect()
    payload = {'images': images, 'hidden': hidden, 'token_inputs': token_inputs,
               'null_hidden': hidden[''], 'draws': make_draws(), 'policy': policy}
    torch.save(payload, OUTPUT / 'inputs.pt')
    summary = {'status': 'CPU_PREPARATION_COMPLETE_NOT_MODEL_ATTACK_EXECUTION',
               'images': {k: {f: v[f] for f in ['metadata', 'caption', 'pixels_sha256',
                                             'original_file_sha256', 'BLIP_pixel_values_sha256']}
                          for k, v in images.items()},
               'inputs_sha256': digest(OUTPUT / 'inputs.pt'),
               'protocol_sha256': digest(OUTPUT / 'preparation_protocol.json'),
               'hidden_sha256': {p: tensor_hash(h) for p, h in hidden.items()},
               'seconds': time.perf_counter() - start, 'new_GPU_inference': 0,
               'torch_version': torch.__version__, 'source_sha256': files}
    save_json(OUTPUT / 'summary.json', summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in ('images', 'hidden_sha256', 'source_sha256')}), flush=True)


if __name__ == '__main__':
    main()
