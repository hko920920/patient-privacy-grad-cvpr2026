"""MoFit public COCO schedule adapted explicitly to NIH P256 / SD2.1.

Frozen model weights; input-only FP32 differentiation. No membership evaluation.
The caller owns model loading, immutable input/source binding, and GPU scheduling.
"""
from __future__ import annotations

import hashlib
import time
from contextlib import nullcontext

import torch
import torch.nn.functional as F


def cpu(x):
    return x.detach().cpu().contiguous().clone()


def tensor_hash(x):
    return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def kernel_policy():
    return {
        'schema': 'mofit-medical-kernel/v1',
        'name': 'Public COCO 1000+300 schedule, medical-BLIP initialization, NIH P256/SD2.1 adaptation',
        'source_commit': '91e4b5edc153bac84b0b4209f70d1b2b94e653b2',
        'precision': 'float32, source fp16 artifact values promoted; frozen UNet/VAE',
        'pixel_shape': [1, 3, 256, 256], 'embedding_shape': [1, 77, 1024],
        'timestep': 140, 'scaling_factor': 0.18215,
        'prediction_type': 'epsilon',
        'stage1': {'steps': 1000, 'initial_uniform_radius': .3,
                   'initial_clamp': False, 'projection': 'clamp [-1,1] after update only; no original-centred ball',
                   'step': '.15-(.15-.0015)*iteration/1000',
                   'objective': 'MSE(unconditional epsilon on VAE posterior mean, fixed epsilon)',
                   'CFG': 'batch2 null+caption, loss from null branch only',
                   'update': 'pixel -= step * sign(gradient)'},
        'stage2': {'steps': 300, 'coordinates': 'entire 77x1024 including padding',
                   'posterior': 'two fixed samples, surrogate then original, once before optimization',
                   'objective': 'MSE(epsilon on surrogate posterior sample and optimized embedding, fixed epsilon)',
                   'Adam': {'lr': .06, 'betas': [.9, .999], 'eps': 1e-8,
                            'weight_decay': 0., 'amsgrad': False, 'foreach': False},
                   'monitor': 'every iteration: original sample under null, current, initial embedding',
                   'score': 'h=L(original,current)-L(original,null); member-high source orientation'},
        'draws': {'backend': 'CPU torch FP32 explicit draws, not original CUDA bitwise replay',
                  'initial_uniform_seed': 0, 'diffusion_seed': 260915,
                  'posterior_seed_reset': 0, 'posterior_order': ['surrogate', 'original'],
                  'diffusion_note': 'new P256 fixed Gaussian; unavailable author rnd_noise_1.npy not reused'},
        'fd': {'h': [.01, .005], 'direction': 'CPU Rademacher RMS1',
               'pixel_seed': 1731, 'embedding_seed': 1732,
               'point': 'first update before optimizer/projection; fixed inputs/noise',
               'decision': 'raw AD/FD and heuristic rounding scale; no universal model-specific tolerance or forced PASS'},
        'snapshots': {'pixel': [0, 1, 99, 499, 999], 'embedding': [0, 1, 29, 149, 299]},
        'counts': {'full_core_UNet_forward_examples': 3200, 'full_core_UNet_backward_calls': 1300,
                   'full_core_VAE_forward': 1002, 'full_core_VAE_backward': 1000,
                   'benchmark_2plus2_with_FD_UNet_forward_examples': 24,
                   'benchmark_2plus2_with_FD_UNet_backward_calls': 4,
                   'benchmark_2plus2_with_FD_VAE_forward': 8,
                   'benchmark_2plus2_with_FD_VAE_backward': 2,
                   'optional_final_surrogate_objectives_extra_UNet_forward_examples': 3,
                   'optional_final_surrogate_objectives_extra_VAE_forward': 1},
        'limits': ['kernel is not patient membership performance',
                   'current backend/input/RNG/text initialization are explicit adaptations',
                   'COCO schedule is public; separate exact medical schedule is not released',
                   'sparse full-run packets do not independently verify all input gradients',
                   'short embedding prefix differs from full because its surrogate differs'],
    }


def make_draws(pixel_shape=(1, 3, 256, 256), latent_shape=(1, 4, 32, 32)):
    g = torch.Generator(device='cpu').manual_seed(0)
    initial = torch.rand(pixel_shape, generator=g, dtype=torch.float32) * .6 - .3
    diffusion = torch.randn(latent_shape, generator=torch.Generator(device='cpu').manual_seed(260915))
    posterior = torch.Generator(device='cpu').manual_seed(0)
    surrogate = torch.randn(latent_shape, generator=posterior, dtype=torch.float32)
    original = torch.randn(latent_shape, generator=posterior, dtype=torch.float32)
    return {'initial_pixel_perturbation': initial, 'diffusion_noise': diffusion,
            'posterior_noise_surrogate': surrogate, 'posterior_noise_original': original,
            'metadata': kernel_policy()['draws']}


class MoFitMedicalAdapter:
    def __init__(self, unet, vae, scheduler, null_hidden):
        self.unet, self.vae, self.scheduler = unet, vae, scheduler
        self.device = next(unet.parameters()).device
        self.null_hidden = null_hidden.detach().to(self.device, torch.float32)
        self.unet.eval().requires_grad_(False)
        self.vae.eval().requires_grad_(False)
        if scheduler.config.prediction_type != 'epsilon':
            raise ValueError('MoFit source objective requires epsilon prediction')
        if scheduler.config.num_train_timesteps != 1000:
            raise ValueError('Expected 1000-step training scheduler')
        for model in (unet, vae):
            for name, parameter in model.named_parameters():
                if parameter.dtype != torch.float32 or parameter.device != self.device:
                    raise ValueError(f'Model parameter must already be FP32 on {self.device}: {name}')
                if parameter.grad is not None:
                    raise ValueError(f'Existing weight gradient: {name}')
            for module in model.modules():
                if isinstance(module, torch.nn.Dropout) and module.p != 0:
                    raise ValueError('Nonzero dropout requires a separately declared policy')
        self._versions = [(name, p, p._version) for prefix, model in [('unet', unet), ('vae', vae)]
                          for name, p in [(prefix + '.' + n, p) for n, p in model.named_parameters()]]
        self.counts = {}
        self._reset_counts()

    def _reset_counts(self):
        self.counts = {'unet_forward_calls': 0, 'unet_forward_examples': 0,
                       'unet_backward_calls': 0, 'vae_forward': 0, 'vae_backward': 0}

    def assert_weights_unchanged(self):
        for name, p, version in self._versions:
            if p._version != version or p.grad is not None or p.requires_grad:
                raise AssertionError(f'Weight mutation/gradient: {name}')
        return True

    def _encode(self, pixels):
        self.counts['vae_forward'] += int(pixels.shape[0])
        return self.vae.encode(pixels).latent_dist

    def _predict(self, latent, hidden):
        self.counts['unet_forward_calls'] += 1
        self.counts['unet_forward_examples'] += int(latent.shape[0])
        t = torch.tensor([140], dtype=torch.long, device=self.device)
        return self.unet(latent, t, encoder_hidden_states=hidden).sample

    def _noised(self, latent, noise):
        return self.scheduler.add_noise(latent, noise,
                                        torch.tensor([140], dtype=torch.long, device=self.device))

    def pixel_cell(self, pixels, initial_hidden, noise, *, gradient=False,
                   mode='literal_cfg2', capture=True):
        x = pixels.detach().to(self.device, torch.float32).requires_grad_(gradient)
        with torch.enable_grad() if gradient else torch.no_grad():
            posterior = self._encode(x)
            z = posterior.mean * .18215
            noisy = self._noised(z, noise)
            if mode == 'literal_cfg2':
                pred_all = self._predict(torch.cat([noisy, noisy]),
                                         torch.cat([self.null_hidden, initial_hidden]))
                prediction = pred_all.chunk(2)[0]
            elif mode == 'unconditional_only_diagnostic':
                prediction = self._predict(noisy, self.null_hidden)
            else:
                raise ValueError(mode)
            loss = F.mse_loss(prediction.float(), noise.float(), reduction='mean')
            grad = torch.autograd.grad(loss, x)[0] if gradient else None
            if gradient:
                self.counts['unet_backward_calls'] += 1
                self.counts['vae_backward'] += 1
        cell = {'loss': float(loss.detach())}
        if capture:
            cell.update(posterior_mean=cpu(posterior.mean), scaled_latent=cpu(z),
                        noised_latent=cpu(noisy), prediction=cpu(prediction), target=cpu(noise))
            if mode == 'literal_cfg2':
                cell['conditional_prediction_unused'] = cpu(pred_all.chunk(2)[1])
        return float(loss.detach()), (cpu(grad) if gradient else None), cell

    def embedding_cell(self, latent, embedding, noise, *, gradient=False, capture=True):
        theta = embedding.detach().to(self.device, torch.float32).requires_grad_(gradient)
        with torch.enable_grad() if gradient else torch.no_grad():
            noisy = self._noised(latent, noise)
            prediction = self._predict(noisy, theta)
            loss = F.mse_loss(prediction.float(), noise.float(), reduction='mean')
            grad = torch.autograd.grad(loss, theta)[0] if gradient else None
            if gradient:
                self.counts['unet_backward_calls'] += 1
        cell = {'loss': float(loss.detach())}
        if capture:
            cell.update(scaled_latent=cpu(latent), noised_latent=cpu(noisy),
                        prediction=cpu(prediction), target=cpu(noise))
        return float(loss.detach()), (cpu(grad) if gradient else None), cell

    def _fd(self, point, gradient, objective, steps, seed):
        g = torch.Generator(device='cpu').manual_seed(seed)
        direction = (torch.randint(0, 2, point.shape, generator=g) * 2 - 1).float()
        point = cpu(point)
        ad = float((gradient.double() * direction.double()).sum())
        rows = []
        for h in steps:
            plus_point = point + float(h) * direction
            minus_point = point - float(h) * direction
            plus, _, plus_cell = objective(plus_point)
            minus, _, minus_cell = objective(minus_point)
            fd = (plus - minus) / (2 * h)
            heuristic = torch.finfo(torch.float32).eps * (abs(plus) + abs(minus)) / (2 * h)
            rows.append({'h': float(h), 'plus_loss': plus, 'minus_loss': minus,
                         'finite_difference': fd, 'heuristic_rounding_scale': heuristic,
                         'absolute_AD_difference': abs(fd - ad),
                         'actual_plus_displacement_l2': float((plus_point - point).double().norm()),
                         'actual_minus_displacement_l2': float((minus_point - point).double().norm()),
                         'plus': plus_cell, 'minus': minus_cell})
        return {'direction': direction, 'point': point, 'ADdot': ad, 'seed': seed,
                'cells': rows, 'decision': 'unclassified; independent verifier interprets numerical evidence'}

    def _posterior(self, pixels, draw):
        with torch.no_grad():
            posterior = self._encode(pixels)
            # Explicit saved CPU-generated draw; source sample() uses the same algebra.
            sample = posterior.mean + posterior.std * draw
            scaled = sample * .18215
        return scaled, {'mean': cpu(posterior.mean), 'std': cpu(posterior.std),
                        'logvar': cpu(posterior.logvar), 'noise': cpu(draw),
                        'sample': cpu(sample), 'scaled_sample': cpu(scaled),
                        'scaling_factor': .18215}

    def run(self, pixels, image_id, initial_hidden, draws, stage1_steps=1000,
            stage2_steps=300, nominal_stage1_steps=1000, mode='literal_cfg2',
            monitor='literal_every_step', fd_steps=(.01, .005),
            capture_pixel=(0, 1, 99, 499, 999), capture_embedding=(0, 1, 29, 149, 299),
            endpoint_diagnostics=False, progress=None):
        if mode != 'literal_cfg2' or monitor != 'literal_every_step':
            raise ValueError('Primary run requires literal CFG2 and all three per-step monitors')
        if not 0 < stage1_steps <= nominal_stage1_steps or stage2_steps <= 0:
            raise ValueError('Invalid step counts')
        if nominal_stage1_steps != 1000:
            raise ValueError('Short run retains original 1000-step learning-rate schedule')
        self.assert_weights_unchanged()
        self._reset_counts()
        started = time.perf_counter()
        pixels = pixels.detach().to(self.device, torch.float32)
        hidden = initial_hidden.detach().to(self.device, torch.float32)
        if tuple(pixels.shape) != (1, 3, 256, 256) or tuple(hidden.shape) != (1, 77, 1024):
            raise ValueError('Expected declared P256 pixels and full SD2.1 embedding')
        if hidden.shape != self.null_hidden.shape or not torch.isfinite(pixels).all():
            raise ValueError('Input mismatch/nonfinite')
        noises = {k: v.to(self.device, torch.float32) for k, v in draws.items() if isinstance(v, torch.Tensor)}
        noise = noises['diffusion_noise']
        raw = {'initial_pixels': cpu(pixels), 'initial_hidden': cpu(hidden),
               'null_hidden': cpu(self.null_hidden),
               'draws': {k: cpu(v) if isinstance(v, torch.Tensor) else v for k, v in draws.items()},
               'pixel_updates': [], 'embedding_updates': [], 'posterior': {},
               'fd': {}, 'traces': {'pixel': [], 'embedding': []}}
        adv = pixels + noises['initial_pixel_perturbation']
        pixel_initial_loss = None
        for iteration in range(stage1_steps):
            capture = iteration in capture_pixel
            before = adv.detach()
            loss, grad_cpu, cell = self.pixel_cell(before, hidden, noise, gradient=True,
                                                  mode=mode, capture=capture)
            if pixel_initial_loss is None:
                pixel_initial_loss = loss
            if iteration == 0 and fd_steps:
                raw['fd']['pixel'] = self._fd(before, grad_cpu,
                    lambda x: self.pixel_cell(x, hidden, noise, mode=mode), fd_steps, 1731)
            grad = grad_cpu.to(self.device)
            step = .15 - (.15 - .0015) / nominal_stage1_steps * iteration
            unclamped = before - grad.sign() * step
            adv = unclamped.clamp(-1, 1).detach()
            trace = {'iteration': iteration, 'loss': loss, 'actual_step_size': step,
                     'gradient_l2': float(grad_cpu.double().norm()),
                     'gradient_absmax': float(grad_cpu.abs().max()),
                     'gradient_finite': bool(torch.isfinite(grad_cpu).all()),
                     'before_sha256': tensor_hash(before), 'gradient_sha256': tensor_hash(grad_cpu),
                     'after_sha256': tensor_hash(adv),
                     'step_l2': float((adv - before).double().norm()),
                     'clamped_count': int((unclamped != adv).sum())}
            if not trace['gradient_finite'] or not torch.isfinite(adv).all():
                raise FloatingPointError('Nonfinite pixel update')
            raw['traces']['pixel'].append(trace)
            if capture:
                raw['pixel_updates'].append(dict(iteration=iteration, before=cpu(before),
                    gradient=grad_cpu, after=cpu(adv), actual_step_size=step, loss=loss, cell=cell))
            if progress:
                progress('pixel', iteration + 1, stage1_steps, dict(self.counts), loss)
        pixel_endpoint = None
        if endpoint_diagnostics:
            loss, _, cell = self.pixel_cell(adv, hidden, noise, mode=mode)
            pixel_endpoint = {'loss': loss, 'cell': cell,
                              'initial_minus_final': pixel_initial_loss - loss}
        latent_sur, raw['posterior']['surrogate'] = self._posterior(adv, noises['posterior_noise_surrogate'])
        latent_ori, raw['posterior']['original'] = self._posterior(pixels, noises['posterior_noise_original'])
        theta = hidden.detach().clone().requires_grad_(True)
        optimizer = torch.optim.Adam([theta], lr=.06, betas=(.9, .999), eps=1e-8,
                                     weight_decay=0., amsgrad=False, foreach=False)
        initial_surrogate_loss = None
        final_monitor = None
        for iteration in range(stage2_steps):
            capture = iteration in capture_embedding
            before = cpu(theta)
            loss, grad_cpu, cell = self.embedding_cell(latent_sur, theta, noise,
                                                       gradient=True, capture=capture)
            if initial_surrogate_loss is None:
                initial_surrogate_loss = loss
            if iteration == 0 and fd_steps:
                raw['fd']['embedding'] = self._fd(theta, grad_cpu,
                    lambda x: self.embedding_cell(latent_sur, x, noise), fd_steps, 1732)
            state = optimizer.state.get(theta, {})
            if capture:
                avg_before = cpu(state['exp_avg']) if state else torch.zeros_like(before)
                sq_before = cpu(state['exp_avg_sq']) if state else torch.zeros_like(before)
                step_before = float(state['step']) if state else 0.
            optimizer.zero_grad(set_to_none=True)
            theta.grad = grad_cpu.to(self.device)
            optimizer.step()
            theta.grad = None
            # Literal source evaluation cadence; final iteration reused as final score.
            u, _, ucell = self.embedding_cell(latent_ori, self.null_hidden, noise,
                                              capture=iteration == stage2_steps - 1)
            h_raw, _, hcell = self.embedding_cell(latent_ori, theta, noise,
                                                  capture=iteration == stage2_steps - 1)
            v_raw, _, vcell = self.embedding_cell(latent_ori, hidden, noise,
                                                  capture=iteration == stage2_steps - 1)
            trace = {'iteration': iteration, 'loss': loss,
                     'gradient_l2': float(grad_cpu.double().norm()),
                     'gradient_absmax': float(grad_cpu.abs().max()),
                     'gradient_finite': bool(torch.isfinite(grad_cpu).all()),
                     'before_sha256': tensor_hash(before), 'gradient_sha256': tensor_hash(grad_cpu),
                     'after_sha256': tensor_hash(theta),
                     'step_l2': float((cpu(theta) - before).double().norm()),
                     'u': u, 'h_raw': h_raw, 'v_raw': v_raw, 'h': h_raw - u, 'v': v_raw - u}
            if not trace['gradient_finite'] or not torch.isfinite(theta).all():
                raise FloatingPointError('Nonfinite embedding update')
            raw['traces']['embedding'].append(trace)
            if capture:
                state = optimizer.state[theta]
                raw['embedding_updates'].append(dict(iteration=iteration, before=before,
                    gradient=grad_cpu, after=cpu(theta), exp_avg_before=avg_before,
                    exp_avg_after=cpu(state['exp_avg']), exp_avg_sq_before=sq_before,
                    exp_avg_sq_after=cpu(state['exp_avg_sq']), step_before=step_before,
                    step_after=float(state['step']), loss=loss, cell=cell))
            if iteration == stage2_steps - 1:
                final_monitor = {'u': u, 'h_raw': h_raw, 'v_raw': v_raw,
                                 'h': h_raw - u, 'v': v_raw - u,
                                 'null_cell': ucell, 'optimized_cell': hcell, 'initial_cell': vcell}
            if progress:
                progress('embedding', iteration + 1, stage2_steps, dict(self.counts), loss)
        embedding_endpoint = None
        if endpoint_diagnostics:
            loss, _, cell = self.embedding_cell(latent_sur, theta, noise)
            embedding_endpoint = {'loss': loss, 'cell': cell,
                                  'initial_minus_final': initial_surrogate_loss - loss}
        raw['final'] = {'surrogate_pixels': cpu(adv), 'embedding': cpu(theta),
                        'original_scores': final_monitor,
                        'surrogate_pixel_endpoint': pixel_endpoint,
                        'surrogate_embedding_endpoint': embedding_endpoint,
                        'initial_pixel_objective': pixel_initial_loss,
                        'initial_embedding_objective': initial_surrogate_loss}
        self.assert_weights_unchanged()
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        report = {'image_id': image_id, 'policy': kernel_policy(),
                  'stage1_steps': stage1_steps, 'stage2_steps': stage2_steps,
                  'nominal_stage1_steps': nominal_stage1_steps, 'mode': mode, 'monitor': monitor,
                  'fd_steps': list(fd_steps), 'endpoint_diagnostics': endpoint_diagnostics,
                  'counts': dict(self.counts), 'seconds': time.perf_counter() - started,
                  'u': final_monitor['u'], 'h': final_monitor['h'], 'v': final_monitor['v'],
                  'pixel_initial_minus_final': pixel_endpoint['initial_minus_final'] if pixel_endpoint else None,
                  'embedding_initial_minus_final': embedding_endpoint['initial_minus_final'] if embedding_endpoint else None,
                  'captured_pixel_updates': [x['iteration'] for x in raw['pixel_updates']],
                  'captured_embedding_updates': [x['iteration'] for x in raw['embedding_updates']],
                  'weight_versions_and_no_gradients': True,
                  'unet_training': self.unet.training, 'vae_training': self.vae.training,
                  'all_embedding_coordinates_optimized': True,
                  'optimizer_parameter_shape': list(theta.shape),
                  'membership_performance_claim': False}
        return {'report': report, 'raw': raw}
