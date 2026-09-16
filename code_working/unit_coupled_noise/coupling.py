"""Deterministic joint-randomness construction for unit-coupled diffusion noise."""

from __future__ import annotations

import hashlib
from typing import Any


METHODS = (
    "iid_independent_t",
    "antithetic_independent_t",
    "iid_pair_shared_t",
    "antithetic_pair_shared_t",
)


def stable_hash(*parts: object) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest().upper()


def hash_uint63(*parts: object) -> int:
    return int(stable_hash(*parts)[:16], 16) % (2**63)


def paired_plan(
    *,
    salt: str,
    bank: str,
    patient_id: str,
    method: str,
    record_count: int,
    num_train_timesteps: int,
) -> list[dict[str, Any]]:
    """Return record-wise timestep/noise instructions without drawing tensors.

    Records are paired in their already-frozen order.  Each marginal timestep is
    uniform and each marginal noise seed denotes a standard Gaussian draw.  An
    antithetic second record reuses the first record's draw with sign -1.
    """

    if method not in METHODS:
        raise ValueError(f"unknown method: {method}")
    if record_count < 1:
        raise ValueError("record_count must be positive")
    if num_train_timesteps < 1:
        raise ValueError("num_train_timesteps must be positive")

    antithetic = method.startswith("antithetic_")
    shared_t = method.endswith("pair_shared_t")
    plan: list[dict[str, Any]] = []
    for index in range(record_count):
        pair_index = index // 2
        position = index % 2
        if shared_t:
            timestep_key = (salt, "timestep_pair", bank, patient_id, pair_index)
        else:
            timestep_key = (salt, "timestep_record", bank, patient_id, index)
        timestep = hash_uint63(*timestep_key) % num_train_timesteps

        if antithetic and position == 1:
            noise_index = index - 1
            noise_sign = -1
        else:
            noise_index = index
            noise_sign = 1
        noise_seed = hash_uint63(salt, "noise_record", bank, patient_id, noise_index)
        plan.append(
            {
                "record_index": index,
                "pair_index": pair_index,
                "timestep": int(timestep),
                "noise_seed": int(noise_seed),
                "noise_sign": noise_sign,
            }
        )
    return plan

