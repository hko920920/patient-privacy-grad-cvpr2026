import pytest
from pathlib import Path

pytest.importorskip("poseidon_py.poseidon_hash")
pytest.importorskip("reedsolo")

from ppmark_v03.config import GlobalConfig
from ppmark_v03.embedding import inject_watermark
from ppmark_v03.halo2_interface import FIXED_SCALE, build_sample_merkle
from ppmark_v03.payload import build_payload_pallas
from ppmark_v03.rs import ReedSolomonCodec
from ppmark_v03.sampling import deterministic_sample
from ppmark_v03.tables import InverseCDFTable


def test_sample_merkle_root_matches_trace():
    cfg = GlobalConfig.load(Path("config_small.json"))
    codec = ReedSolomonCodec(n=cfg.watermark.rs_n, k=cfg.watermark.rs_k)
    payload = build_payload_pallas(
        prompt="merkle-test",
        seed=b"\x01" * 32,
        secret=b"\x02" * 32,
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

    assert len(embedding.sample_trace.entries) == len(sample_set.indices)

    bits_for_samples = [payload.bits[idx % len(payload.bits)] for idx in sample_set.indices]
    alpha_fixed = int(round(cfg.image.alpha * FIXED_SCALE))
    gaussian_fixed = [int(round(obs.z_expected * FIXED_SCALE)) for obs in embedding.sample_trace.entries]
    combined_fixed = [g + alpha_fixed * (2 * b - 1) for g, b in zip(gaussian_fixed, bits_for_samples)]

    root, paths = build_sample_merkle(
        embedding.sample_trace.entries,
        gaussian_fixed=gaussian_fixed,
        combined_fixed=combined_fixed,
        bits=bits_for_samples,
    )
    # Determinism and structural sanity checks.
    assert len(root) == 32
    assert len(paths) == len(sample_set.indices)
    root_again, _ = build_sample_merkle(
        embedding.sample_trace.entries,
        gaussian_fixed=gaussian_fixed,
        combined_fixed=combined_fixed,
        bits=bits_for_samples,
    )
    assert root_again == root
