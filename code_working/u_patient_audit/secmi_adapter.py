"""Exact acquired CDI-repository SecMI-stat kernels with a local SD adapter.

Only upstream class ASTs are compiled: no third-party entry point or dependency
installation is executed. The upstream single/multi-step and L2 implementations
are unchanged. Input encoding is explicitly identity on verified cached latents.
"""
import ast
import hashlib
import math
from pathlib import Path
import time
from types import SimpleNamespace
from typing import List, Tuple

import torch

from .common import ROOT, digest

RESEARCH = ROOT.parent / "CVPR 주제 탐색" / "research_2026-09-10"
UPSTREAM = RESEARCH / "code_sources/X01"
COMMIT = "dcd62258b0b3fde05d52aaecfade3b5f4c09507a"
FILES = {
    "src/attacks/features_extraction/secmi.py": "4e4f36d04f0cfe7f6dcab4812d66be4ed65680fbbb92910927d6b5bce85de9c3",
    "src/attacks/scores_computation/secmi.py": "10f1d642639d30f9f14f15a9d7657d819413a9ce077b68ff7aa7bcc4aee74dc2",
    "conf/attack/secmi_stat.yaml": "aa188200f526bc4f2c5c4fac3e92f3541636d8af1f809de9eb882e95d25fb6ad",
}
EXPECTED_STEPS = [(t, t + 10) for t in range(0, 100, 10)] + [(100, 110), (110, 100)]


def source_bindings():
    for name, expected in FILES.items():
        assert digest(UPSTREAM / name) == expected, "acquired upstream source changed: " + name
    config = (UPSTREAM / "conf/attack/secmi_stat.yaml").read_text(encoding="utf-8")
    assert all(line in config.splitlines() for line in ("name: secmi_stat", "members_lower: true", "t0: 0", "t: 100", "step: 10"))
    return {
        "repository": "https://github.com/sprintml/copyrighted_data_identification",
        "commit": COMMIT, "source_root": str(UPSTREAM), "files_sha256": dict(FILES),
        "classes_reused": ["SecMIExtractor", "SecMIStat"],
        "execution": "original class ASTs compiled unchanged; wrapper supplies model/config only",
        "t0": 0, "t": 100, "step": 10, "members_lower": True,
        "raw_discrepancy": "Euclidean L2 norm, not squared sum or mean",
        "encoding_adaptation": "identity on existing verified VAE posterior-mode latents",
    }


def _upstream_class(relative, name):
    source_bindings()
    path = UPSTREAM / relative
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    nodes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name]
    assert len(nodes) == 1
    module = ast.Module(body=nodes, type_ignores=[])
    namespace = {"torch": torch, "T": torch.Tensor, "Tuple": Tuple, "List": List,
                 "sqrt": math.sqrt, "FeatureExtractor": object, "ScoreComputer": object}
    exec(compile(module, filename=str(path), mode="exec"), namespace)
    return namespace[name]


class SecMIAdapter:
    def __init__(self, unet, scheduler, cache):
        self.unet, self.scheduler, self.cache = unet, scheduler, cache
        self.device = next(unet.parameters()).device
        self.unet.to(dtype=torch.float32)
        if hasattr(self.unet, "enable_adapters"):
            self.unet.enable_adapters()
        self.unet.eval().requires_grad_(False)
        self.prediction_type = str(scheduler.config.prediction_type)
        if self.prediction_type not in {"epsilon", "v_prediction"}:
            raise ValueError("unsupported prediction type: " + self.prediction_type)
        self.alphas = scheduler.alphas_cumprod.detach().cpu().to(torch.float32)
        assert self.alphas.ndim == 1 and len(self.alphas) > 110
        assert bool(torch.isfinite(self.alphas).all()) and bool(((self.alphas > 0) & (self.alphas <= 1)).all())
        self.hidden = cache["hidden"][cache["audit_prompt"]].detach().clone().to(self.device, torch.float32)
        self.forward = 0
        self.backward = 0
        self.initial_versions = {n: p._version for n, p in unet.named_parameters()}
        extractor_type = _upstream_class("src/attacks/features_extraction/secmi.py", "SecMIExtractor")
        statistic_type = _upstream_class("src/attacks/scores_computation/secmi.py", "SecMIStat")
        self.extractor = extractor_type()
        self.extractor.model = self
        self.extractor.device = self.device
        self.extractor.attack_cfg = SimpleNamespace(t0=0, t=100, step=10)
        self.statistic = statistic_type()
        # Instrument around the original method; its AST and arithmetic are intact.
        self._original_single_step = self.extractor.single_step
        self.extractor.single_step = self._recorded_single_step
        self.stages = []

    def encode(self, latents):
        assert latents.dtype == torch.float32 and latents.shape == (1, 4, 32, 32)
        assert latents.device == self.device and not latents.requires_grad
        assert bool(torch.isfinite(latents).all())
        return latents

    def get_alpha_cumprod(self, timestep):
        return float(self.alphas[int(timestep)])

    def predict_noise_from_latent(self, latents, classes, timestep):
        assert latents.shape[0] == 1 and latents.dtype == classes.dtype == torch.float32
        assert not torch.is_grad_enabled(), "SecMI measurement must not build gradients"
        t = torch.tensor([int(timestep)], dtype=torch.long, device=self.device)
        with torch.autocast(device_type=self.device.type, enabled=False):
            prediction = self.unet(latents, t, encoder_hidden_states=classes).sample
        assert prediction.dtype == torch.float32 and prediction.shape == latents.shape
        assert bool(torch.isfinite(prediction).all())
        if self.prediction_type == "epsilon":
            epsilon = prediction
        else:
            alpha = self.get_alpha_cumprod(timestep)
            epsilon = math.sqrt(alpha) * prediction + math.sqrt(1 - alpha) * latents
        self.forward += 1
        self._last_prediction = {"prediction_l2": float(prediction.norm()), "epsilon_l2": float(epsilon.norm()),
                                 "prediction_type": self.prediction_type,
                                 "epsilon_conversion": "identity" if self.prediction_type == "epsilon" else "sqrt(alpha)*v + sqrt(1-alpha)*x_t"}
        return epsilon

    def _recorded_single_step(self, latents, classes, timestep, target_timestep):
        before = self.forward
        output = self._original_single_step(latents, classes, timestep, target_timestep)
        assert self.forward - before == 1 and bool(torch.isfinite(output).all())
        self.stages.append({"stage": len(self.stages), "timestep": int(timestep), "target_timestep": int(target_timestep),
                            "alpha_input": self.get_alpha_cumprod(timestep), "alpha_target": self.get_alpha_cumprod(target_timestep),
                            "input_l2": float(latents.norm()), "output_l2": float(output.norm()),
                            "input_max_abs": float(latents.abs().max()), "output_max_abs": float(output.abs().max()),
                            **self._last_prediction})
        return output

    def assert_weights_unchanged(self):
        named = dict(self.unet.named_parameters())
        assert set(named) == set(self.initial_versions)
        assert all(p._version == self.initial_versions[n] and p.grad is None and not p.requires_grad for n, p in named.items())
        return True

    def score(self, image_id):
        started = time.perf_counter()
        before_f, before_b = self.forward, self.backward
        self.stages = []
        latent = self.cache["latents"][image_id].detach().clone().to(self.device, torch.float32)
        with torch.no_grad():
            features = self.extractor.process_batch((latent, self.hidden))
            assert features.shape == (1, 2, 4, 32, 32) and features.dtype == torch.float32
            discrepancy = self.statistic.compute_score(features)
        assert discrepancy.shape == (1,) and bool(torch.isfinite(discrepancy).all())
        assert [(s["timestep"], s["target_timestep"]) for s in self.stages] == EXPECTED_STEPS
        assert self.forward-before_f == 12 and self.backward-before_b == 0
        self.assert_weights_unchanged()
        return {"image_id": image_id, "z_det": features[:, 0].detach().cpu().clone(),
                "z_recon": features[:, 1].detach().cpu().clone(), "l2": float(discrepancy[0]),
                "forward": self.forward-before_f, "backward": self.backward-before_b,
                "stages": list(self.stages), "seconds": time.perf_counter()-started,
                "weights_unchanged": True, "source_statistic": "unmodified SecMIStat.compute_score"}


def cpu_conformance():
    """Synthetic analytical oracle, independent of the acquired update methods."""
    class OracleUNet(torch.nn.Module):
        def __init__(self, kind, alphas):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.tensor(0.0))
            self.kind, self.alphas = kind, alphas

        def forward(self, x, t, encoder_hidden_states=None):
            # Prescribed epsilon is state dependent, so a nontrivial roundtrip remains.
            alpha = float(self.alphas[int(t[0])])
            epsilon = .13*x + .017*(int(t[0])+1)/100
            value = epsilon if self.kind == "epsilon" else (epsilon-math.sqrt(1-alpha)*x)/math.sqrt(alpha)
            return SimpleNamespace(sample=value + 0*self.anchor)

    alphas = torch.cumprod(1-torch.linspace(.0001,.02,1000,dtype=torch.float32),dim=0)
    latent = torch.linspace(-.5,.5,4096,dtype=torch.float32).reshape(1,4,32,32)
    cache = {"latents":{"oracle":latent},"audit_prompt":"fixed","hidden":{"fixed":torch.zeros(1,77,1024)}}
    results = {}
    expected_states = {}
    for kind in ("epsilon","v_prediction"):
        unet = OracleUNet(kind,alphas)
        adapter = SecMIAdapter(unet,SimpleNamespace(config=SimpleNamespace(prediction_type=kind),alphas_cumprod=alphas),cache)
        result = adapter.score("oracle")
        state = latent.clone()
        for index,(t,target) in enumerate(EXPECTED_STEPS):
            alpha, next_alpha = float(alphas[t]),float(alphas[target])
            epsilon = .13*state + .017*(t+1)/100
            # Algebraic affine form is independent of the upstream x0 expression.
            ratio = math.sqrt(next_alpha/alpha)
            state = ratio*state + (math.sqrt(1-next_alpha)-ratio*math.sqrt(1-alpha))*epsilon
            if index == 9:
                expected_det = state.clone()
        expected_l2 = float(torch.linalg.vector_norm((expected_det-state).double()))
        det_error = float((result["z_det"]-expected_det).abs().max())
        recon_error = float((result["z_recon"]-state).abs().max())
        # Fixed float32 arithmetic envelope, not any MIA acceptance threshold.
        envelope = 256*torch.finfo(torch.float32).eps*max(float(state.abs().max()),1.)
        assert det_error <= envelope and recon_error <= envelope
        direct_l2 = float(torch.linalg.vector_norm((result["z_det"]-result["z_recon"]).double()))
        assert abs(result["l2"]-direct_l2) <= 32*torch.finfo(torch.float32).eps*max(direct_l2,1e-8)
        assert result["forward"] == adapter.forward == 12 and adapter.backward == 0
        assert adapter.assert_weights_unchanged()
        with torch.no_grad():
            sample_t = 100
            epsilon_actual = adapter.predict_noise_from_latent(latent,adapter.hidden,sample_t)
            epsilon_expected = .13*latent + .017*(sample_t+1)/100
        conversion_error = float((epsilon_actual-epsilon_expected).abs().max())
        assert conversion_error <= envelope
        results[kind] = dict(forward_score=12,extra_conversion_forward=1,backward=0,
            z_det_oracle_max_abs=det_error,z_recon_oracle_max_abs=recon_error,
            runtime_l2=result["l2"],independent_double_l2=direct_l2,affine_oracle_l2=expected_l2,
            epsilon_conversion_max_abs=conversion_error,weights_unchanged=True,
            stage_pairs=[list(v) for v in EXPECTED_STEPS])
        expected_states[kind]=result
    return {"status":"PASS_SYNTHETIC_CPU_CONFORMANCE","GPU_executions":0,"source_bindings":source_bindings(),
            "cases":results,"epsilon_vs_v_reconstruction_max_abs":float((expected_states["epsilon"]["z_recon"]-expected_states["v_prediction"]["z_recon"]).abs().max()),
            "scope":"upstream AST reuse, synthetic affine recurrence, epsilon conversion, L2/counters/freeze; not attack efficacy"}
