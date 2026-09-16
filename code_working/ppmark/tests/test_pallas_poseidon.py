from ppmark_v03.pallas_poseidon import poseidon_permute_pallas


def limbs_to_int(a: int, b: int, c: int, d: int) -> int:
    return a + (b << 64) + (c << 128) + (d << 192)


def test_poseidon_permutation_matches_reference():
    # Expected values from halo2_poseidon P128Pow5T3 reference vector (pasta-hadeshash).
    expected = [
        limbs_to_int(
            0x0AEB1_BC02_4AEC_A456,
            0x0F7E6_9A71_D0B6_42A0,
            0x094EF_B364_F966_240F,
            0x02A52_6ACD_0B64_B453,
        ),
        limbs_to_int(
            0x0012A_3E96_28E5_B82A,
            0x0DCD4_2E7F_BED9_DAFE,
            0x076FF_7DAE_343D_5512,
            0x013C5_D156_8B4A_A430,
        ),
        limbs_to_int(
            0x03590_29A1_D34E_9DDD,
            0x0F7CF_DFE1_BDA4_2C7B,
            0x0256F_CD59_7984_561A,
            0x00A49_C868_C697_6544,
        ),
    ]
    assert poseidon_permute_pallas([0, 1, 2]) == expected
