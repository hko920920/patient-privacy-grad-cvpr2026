import json
from pathlib import Path

import pytest

pytest.importorskip("poseidon_py.poseidon_hash")
pytest.importorskip("reedsolo")

from ppmark_v03.config import GlobalConfig
from ppmark_v03.halo2_interface import FIXED_SCALE, Halo2Package, PublicInputs, Witness, build_sample_merkle
from ppmark_v03.halo2_runner import prepare_prover_package
from ppmark_v03.payload import build_payload_pallas
from ppmark_v03.rs import ReedSolomonCodec
from ppmark_v03.sampling import deterministic_sample
from ppmark_v03.embedding import inject_watermark
from ppmark_v03.tables import InverseCDFTable


def test_prepare_prover_package_serializes_merkle_paths(tmp_path: Path):
    cfg = GlobalConfig.load(Path("config_small.json"))
    codec = ReedSolomonCodec(n=cfg.watermark.rs_n, k=cfg.watermark.rs_k)
    payload = build_payload_pallas(
        prompt="runner-test",
        seed=b"\x03" * 32,
        secret=b"\x04" * 32,
        codec=codec,
    )
    sample_set = deterministic_sample(
        total_pixels=cfg.image.total_pixels,
        width=cfg.image.width,
        height=cfg.image.height,
        key_material=payload.codeword,
        count=cfg.image.sample_count(),
    )
    inverse_cdf = InverseCDFTable.from_file(cfg.tables.inverse_cdf_path)
    embedding = inject_watermark(
        binding=payload.binding,
        bit_sequence=payload.bits,
        config=cfg,
        inverse_cdf=inverse_cdf,
        sample_set=sample_set,
    )

    bits_for_samples = [payload.bits[idx % len(payload.bits)] for idx in sample_set.indices]
    alpha_fixed = int(round(cfg.image.alpha * FIXED_SCALE))
    gaussian_fixed = [int(round(obs.z_expected * FIXED_SCALE)) for obs in embedding.sample_trace.entries]
    combined_fixed = [g + alpha_fixed * (2 * b - 1) for g, b in zip(gaussian_fixed, bits_for_samples)]
    sample_root, merkle_paths = build_sample_merkle(
        embedding.sample_trace.entries,
        gaussian_fixed=gaussian_fixed,
        combined_fixed=combined_fixed,
        bits=bits_for_samples,
    )

    public_inputs = PublicInputs(
        prompt_hash=payload.prompt_hash,
        binding=payload.binding,
        sample_merkle_root=sample_root,
        alpha=alpha_fixed,
    )
    witness = Witness(
        secret_key=b"\x04" * 32,
        seed=b"\x03" * 32,
        codeword=payload.codeword,
        sample_observations=list(embedding.sample_trace.entries),
        sample_merkle_paths=merkle_paths,
        gaussian_fixed=gaussian_fixed,
        combined_fixed=combined_fixed,
        bits=bits_for_samples,
        alpha_fixed=alpha_fixed,
    )
    package = Halo2Package(public_inputs=public_inputs, witness=witness)
    paths = prepare_prover_package(package, sample_set, tmp_path)

    with paths.public.open("r", encoding="utf-8") as handle:
        public_json = json.load(handle)
    assert public_json["sample_count"] == len(sample_set.indices)
    assert public_json["sample_merkle_root"] == sample_root.hex()

    with paths.witness.open("r", encoding="utf-8") as handle:
        witness_json = json.load(handle)
    assert len(witness_json["sample_merkle_paths"]) == len(sample_set.indices)
    assert all(len(p) == len(merkle_paths[i]) for i, p in enumerate(witness_json["sample_merkle_paths"]))
    # Hex strings should decode to 32-byte siblings and include 0x prefix.
    assert all(
        all(s.startswith("0x") and len(s) == 66 for s in path) for path in witness_json["sample_merkle_paths"]
    )
