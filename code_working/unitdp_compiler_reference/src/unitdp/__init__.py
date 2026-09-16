"""Unit-aligned DP training utilities for windowed sequential data."""

from unitdp.accountant import RDPConfig, calibrate_noise, epsilon_for_noise
from unitdp.certificate import TrainingCertificate, write_certificate
from unitdp.compiler import CompiledRoute, compile_contract
from unitdp.contract import UnitContract, load_contract
from unitdp.mapping import WindowMapping
from unitdp.owa_dpsgd import OwaDpsgdConfig, OwaDpsgdResult, train_owa_dpsgd
from unitdp.policies import ContributionPolicy
from unitdp.rng import gaussian_noise_sanity, make_random_source
from unitdp.window_dpsgd import WindowDpsgdConfig, WindowDpsgdResult, train_window_dpsgd

__all__ = [
    "CompiledRoute",
    "ContributionPolicy",
    "OwaDpsgdConfig",
    "OwaDpsgdResult",
    "RDPConfig",
    "TrainingCertificate",
    "UnitContract",
    "WindowMapping",
    "WindowDpsgdConfig",
    "WindowDpsgdResult",
    "calibrate_noise",
    "compile_contract",
    "epsilon_for_noise",
    "gaussian_noise_sanity",
    "load_contract",
    "make_random_source",
    "train_owa_dpsgd",
    "train_window_dpsgd",
    "write_certificate",
]
