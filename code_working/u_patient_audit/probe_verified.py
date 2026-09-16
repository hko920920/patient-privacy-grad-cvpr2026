"""Versioned numerical diagnostics preserving the frozen U probe objective.

FP16 keeps the original arithmetic, seed bank, basis, norm, update and scale.
FP32 explicitly promotes the supplied UNet before version tracking. Its noisy
inputs and targets are promoted copies of the legacy FP16 tensors. Use a separate
model for this precision control. No membership or performance verdict is made.
"""
import time
import torch
from .common import seed
from .models import diffusion_input
from .probe import SUPPORT, make_basis, projected_step


class DiagnosticProbe:
    def __init__(self, unet, scheduler, cache, precision="fp16"):
        if precision not in {"fp16", "fp32"}:
            raise ValueError("precision must be fp16 or fp32")
        self.unet, self.scheduler, self.cache = unet, scheduler, cache
        self.precision = precision
        self.device = torch.device("cuda")
        self.compute_dtype = torch.float16 if precision == "fp16" else torch.float32
        self._verify_zero_dropout()
        if precision == "fp32":
            self.unet.to(dtype=torch.float32)
        self.hidden = cache["hidden"][cache["audit_prompt"]].clone().to(self.device, torch.float32)
        self.basis = make_basis(self.hidden.cpu(), cache["token_mask"]).to(self.device)
        self.scale = self.hidden.norm().detach()
        self.forward = 0
        self.backward = 0
        self._input_bank = {}
        # Preserve legacy train mode. Caller may toggle gradient checkpointing
        # explicitly through the UNet API without changing this mode.
        self.unet.train()
        self.unet.requires_grad_(False)
        self.initial_versions = {n: p._version for n, p in self.unet.named_parameters()}

    def _verify_zero_dropout(self):
        configured = getattr(self.unet.config, "dropout", 0.0)
        if configured is not None and float(configured) != 0.0:
            raise ValueError("train-mode diagnostics require zero UNet dropout")
        for name, module in self.unet.named_modules():
            if isinstance(module, torch.nn.modules.dropout._DropoutNd) and float(module.p) != 0.0:
                raise ValueError("nonzero dropout in " + name)
        for name, config in getattr(self.unet, "peft_config", {}).items():
            if float(getattr(config, "lora_dropout", 0.0)) != 0.0:
                raise ValueError("nonzero adapter dropout in " + name)

    def _switch(self, base):
        if base:
            self.unet.disable_adapters()
        else:
            self.unet.enable_adapters()
        # Installed PEFT enable_adapters re-enables parameter gradients.
        self.unet.requires_grad_(False)

    def _coefficient(self, a):
        value = torch.as_tensor(a, device=self.device, dtype=torch.float32)
        if value.shape != (8,) or not torch.isfinite(value).all():
            raise ValueError("coefficient must be finite with shape (8,)")
        return value

    def _embedding(self, a):
        hidden = self.hidden + (self.scale * (self.basis @ a)).reshape_as(self.hidden)
        return hidden.to(self.compute_dtype)

    def _fixed_inputs(self, image_id, phase, tval, draw):
        key = (str(image_id), str(phase), int(tval), int(draw))
        if key not in self._input_bank:
            z = self.cache["latents"][image_id].clone().to(self.device, torch.float16)
            generator = torch.Generator(device="cuda").manual_seed(
                seed("audit-noise", phase, image_id, int(tval), int(draw)))
            noise = torch.randn(z.shape, generator=generator, device=self.device, dtype=torch.float16)
            t = torch.tensor([int(tval)], device=self.device)
            # Generate in the legacy FP16 path BEFORE precision-control promotion.
            with torch.no_grad():
                noisy, target = diffusion_input(z, t, noise, self.scheduler)
            if noisy.dtype != torch.float16 or target.dtype != torch.float16:
                raise RuntimeError("pinned legacy diffusion inputs must remain FP16")
            self._input_bank[key] = (noisy.detach(), target.detach(), t)
        noisy, target, t = self._input_bank[key]
        return noisy.to(self.compute_dtype), target.to(self.compute_dtype), t

    def cell(self, image_id, a, base, phase, tval, draw=0, gradient=False):
        """One legacy cell. No-gradient calls return a zero gradient tensor."""
        self._switch(base)
        a = self._coefficient(a)
        noisy, target, t = self._fixed_inputs(image_id, phase, tval, draw)
        grad = torch.zeros_like(a)
        with torch.set_grad_enabled(gradient):
            coeff = a.detach().clone().requires_grad_(gradient)
            hidden = self._embedding(coeff)
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.precision == "fp16"):
                prediction = self.unet(noisy, t, hidden).sample
                value = (prediction.float() - target.float()).square().mean()
            self.forward += 1
            if gradient:
                # Finish differentiation BEFORE switching the adapter state.
                grad = torch.autograd.grad(value * 1024.0, coeff)[0] / 1024.0
                self.backward += 1
                if not torch.isfinite(grad).all():
                    raise RuntimeError("non-finite coefficient gradient")
                grad = grad.detach()
        if not torch.isfinite(value):
            raise RuntimeError("non-finite denoising loss")
        return float(value.detach()), grad

    def delta_cells(self, image_id, a, phase, ts, draws=1, gradient=False):
        """Retain original base/target accumulation order, not per-cell averaging."""
        a = self._coefficient(a)
        ts = tuple(int(t) for t in ts)
        if not ts or not isinstance(draws, int) or draws < 1:
            raise ValueError("nonempty timesteps and positive integer draws required")
        cells = [(t, d) for t in ts for d in range(draws)]
        count = len(cells)
        base_rows, target_rows = [], []
        base_sum, target_sum = 0.0, 0.0
        base_grad, target_grad = torch.zeros_like(a), torch.zeros_like(a)
        for tval, draw in cells:
            loss, grad = self.cell(image_id, a, True, phase, tval, draw, gradient=gradient)
            base_rows.append((loss, grad))
            base_sum += loss / count
            base_grad += grad / count
        for tval, draw in cells:
            loss, grad = self.cell(image_id, a, False, phase, tval, draw, gradient=gradient)
            target_rows.append((loss, grad))
            target_sum += loss / count
            target_grad += grad / count
        records = []
        for (tval, draw), (bl, bg), (tl, tg) in zip(cells, base_rows, target_rows):
            records.append({
                "timestep": tval, "draw": draw, "phase": phase,
                "noise_seed": seed("audit-noise", phase, image_id, tval, draw),
                "base_loss": bl, "target_loss": tl, "value": bl - tl,
                "gradient_computed": bool(gradient),
                "base_gradient": bg.cpu().tolist() if gradient else None,
                "target_gradient": tg.cpu().tolist() if gradient else None,
                "gradient": (bg - tg).cpu().tolist() if gradient else None,
            })
        return {"value": base_sum - target_sum, "gradient": base_grad - target_grad,
                "base_loss": base_sum, "target_loss": target_sum, "cells": records}

    @staticmethod
    def _json_delta(result):
        return {k: v.detach().cpu().tolist() if torch.is_tensor(v) else v for k, v in result.items()}

    def trace_support(self, image_id, steps=6):
        """Six updates cost36F/36B; final support objective adds6F without B."""
        if not isinstance(steps, int) or steps < 1:
            raise ValueError("steps must be a positive integer")
        started = time.perf_counter()
        before_f, before_b = self.forward, self.backward
        a = torch.zeros(8, device=self.device)
        records = []
        for step in range(steps):
            result = self.delta_cells(image_id, a, "support", SUPPORT, draws=1, gradient=True)
            grad = result["gradient"]
            grad_norm = grad.norm()
            requested_step = 0.01 * grad / (grad_norm + 1e-12)
            raw_norm = (a + requested_step).norm()
            updated = projected_step(a, grad).detach()
            actual_step = updated - a
            records.append({
                "step": step,
                "coefficient_before": a.cpu().tolist(),
                "coefficient_norm_before": float(a.norm()),
                "objective_before": result["value"],
                "base_loss_before": result["base_loss"],
                "target_loss_before": result["target_loss"],
                "gradient": grad.cpu().tolist(), "gradient_norm": float(grad_norm),
                "requested_step": requested_step.cpu().tolist(),
                "raw_coefficient_norm": float(raw_norm),
                "projection_applied": bool(raw_norm > 0.05),
                "projection_factor": float(torch.clamp(0.05 / (raw_norm + 1e-12), max=1.0)),
                "coefficient_after": updated.cpu().tolist(),
                "coefficient_norm_after": float(updated.norm()),
                "actual_step": actual_step.cpu().tolist(),
                "actual_step_norm": float(actual_step.norm()),
                "gradient_dot_actual_step": float(torch.dot(grad, actual_step)),
                "quantized_embedding_step": self.embedding_change(a, updated),
                "cells": result["cells"],
            })
            a = updated
        final = self.delta_cells(image_id, a, "support", SUPPORT, draws=1, gradient=False)
        # Reuse the next iteration's objective for each preceding update.
        for i, record in enumerate(records):
            after = records[i + 1]["objective_before"] if i + 1 < len(records) else final["value"]
            record["objective_after"] = after
            record["objective_change"] = after - record["objective_before"]
        torch.cuda.synchronize()
        self.assert_weights_unchanged()
        return {
            "image_id": image_id, "precision": self.precision,
            "support_timesteps": list(SUPPORT), "draws_per_timestep": 1,
            "step_size": 0.01, "radius": 0.05, "gradient_scale": 1024.0,
            "initial_coefficient": [0.0] * 8, "final_coefficient": a.cpu().tolist(),
            "initial_objective": records[0]["objective_before"],
            "final_objective": final["value"],
            "objective_gain": final["value"] - records[0]["objective_before"],
            "steps": records, "final_evaluation": self._json_delta(final),
            "forward": self.forward - before_f, "backward": self.backward - before_b,
            "extra_final_objective_forward": 2 * len(SUPPORT),
            "seconds": time.perf_counter() - started, "weights_unchanged": True,
            "scope": "numerical optimization diagnostic; no membership or performance conclusion",
        }

    def embedding_change(self, a, b):
        """Conditioning change AFTER the actual selected-precision cast."""
        with torch.no_grad():
            first = self._embedding(self._coefficient(a))
            second = self._embedding(self._coefficient(b))
            difference = second.float() - first.float()
            return {"changed_count": int(torch.count_nonzero(first != second)),
                    "element_count": first.numel(), "l2": float(difference.norm()),
                    "max_abs": float(difference.abs().max()), "precision": self.precision}

    def assert_weights_unchanged(self):
        parameters = dict(self.unet.named_parameters())
        if set(parameters) != set(self.initial_versions):
            raise AssertionError("parameter names changed during diagnostic")
        if any(p.grad is not None for p in parameters.values()):
            raise AssertionError("a target parameter acquired a gradient")
        if any(p.requires_grad for p in parameters.values()):
            raise AssertionError("a target parameter is no longer frozen")
        if any(p._version != self.initial_versions[n] for n, p in parameters.items()):
            raise AssertionError("a target parameter version changed")
        return True

