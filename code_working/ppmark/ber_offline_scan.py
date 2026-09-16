import os
import json
import numpy as np

DUMP_DIR = "outputs/sp1_sweetspot_ddim10_clean/k1000/bit_dump"

PRED_BITS_NPY = os.path.join(DUMP_DIR, "pred_bits.npy")
GT_BITS_NPY   = os.path.join(DUMP_DIR, "gt_bits.npy")
PRED_BYTES_BIN= os.path.join(DUMP_DIR, "pred_bytes.bin")
GT_BYTES_BIN  = os.path.join(DUMP_DIR, "gt_bytes.bin")
META_JSON     = os.path.join(DUMP_DIR, "dump_meta.json")

RS_N_BYTES = 64  # RS(64,32)
RS_K_BYTES = 32
RS_T = (RS_N_BYTES - RS_K_BYTES) // 2  # max correctable symbol errors if erasures not used

def load_bits(path: str) -> np.ndarray:
    b = np.load(path)
    b = b.astype(np.uint8).reshape(-1)
    assert set(np.unique(b)).issubset({0, 1}), f"Non-binary bits in {path}: {np.unique(b)[:10]}"
    return b

def load_bytes(path: str) -> bytes:
    data = open(path, "rb").read()
    return data

def ber(a: np.ndarray, b: np.ndarray) -> float:
    assert a.shape == b.shape
    return float(np.mean(a != b))

def circular_shift_bits(bits: np.ndarray, shift: int) -> np.ndarray:
    shift = shift % len(bits)
    if shift == 0:
        return bits.copy()
    return np.concatenate([bits[-shift:], bits[:-shift]])

def bit_reverse_each_byte_from_bits(bits_msb: np.ndarray) -> np.ndarray:
    """
    bits_msb: length multiple of 8, msb-first within each byte.
    returns bits also msb-first but after reversing bit order within each byte.
    Example: [b7 b6 ... b0] -> [b0 b1 ... b7]
    """
    assert len(bits_msb) % 8 == 0
    out = bits_msb.copy().reshape(-1, 8)[:, ::-1].reshape(-1)
    return out

def bytes_to_bits_msb(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    # msb-first
    bits = np.unpackbits(arr, bitorder="big")
    return bits.astype(np.uint8)

def bits_msb_to_bytes(bits: np.ndarray) -> bytes:
    assert len(bits) % 8 == 0
    packed = np.packbits(bits.astype(np.uint8), bitorder="big")
    return packed.tobytes()

def symbol_errors(pred_bytes: bytes, gt_bytes: bytes) -> int:
    assert len(pred_bytes) == len(gt_bytes) == RS_N_BYTES
    return sum(pb != gb for pb, gb in zip(pred_bytes, gt_bytes))

def scan_best_shift(pred_bits: np.ndarray, gt_bits: np.ndarray, shift_range=None):
    n = len(pred_bits)
    if shift_range is None:
        shift_range = range(n)  # full scan
    best = None
    for s in shift_range:
        shifted = circular_shift_bits(pred_bits, s)
        e = ber(shifted, gt_bits)
        if best is None or e < best[0]:
            best = (e, s)
    return best  # (min_ber, best_shift)

def eval_case(name: str, pred_bits: np.ndarray, gt_bits: np.ndarray):
    # BER no-shift
    e0 = ber(pred_bits, gt_bits)

    # quick shift scan around common small offsets first, then full if needed
    quick = list(range(0, 129))  # 0..128
    best_quick = scan_best_shift(pred_bits, gt_bits, quick)
    best_full = scan_best_shift(pred_bits, gt_bits, None) if best_quick[0] > 0.05 else best_quick

    # derive bytes under no-shift for symbol error check
    pred_bytes = bits_msb_to_bytes(pred_bits[:RS_N_BYTES*8])
    gt_bytes   = bits_msb_to_bytes(gt_bits[:RS_N_BYTES*8])
    sym_err = symbol_errors(pred_bytes, gt_bytes)

    # bytes under best shift too
    shifted_bits = circular_shift_bits(pred_bits, best_full[1])
    shifted_bytes = bits_msb_to_bytes(shifted_bits[:RS_N_BYTES*8])
    sym_err_shift = symbol_errors(shifted_bytes, gt_bytes)

    return {
        "case": name,
        "ber": e0,
        "best_shift_ber": best_full[0],
        "best_shift": int(best_full[1]),
        "sym_err_bytes_no_shift": int(sym_err),
        "sym_err_bytes_best_shift": int(sym_err_shift),
        "rs_t": RS_T,
        "rs_decode_plausible_no_shift": sym_err <= RS_T,
        "rs_decode_plausible_best_shift": sym_err_shift <= RS_T,
    }

def main():
    meta = {}
    if os.path.exists(META_JSON):
        meta = json.load(open(META_JSON, "r"))
    print("META:", meta)

    pred = load_bits(PRED_BITS_NPY)
    gt   = load_bits(GT_BITS_NPY)

    print("len(bits):", len(pred), " (expect", RS_N_BYTES*8, "for RS(64,32) codeword bits)")
    pred = pred[:RS_N_BYTES*8]
    gt   = gt[:RS_N_BYTES*8]

    # sanity: compare bin bytes to bits reconstruction
    pred_bytes_bin = load_bytes(PRED_BYTES_BIN)
    gt_bytes_bin   = load_bytes(GT_BYTES_BIN)
    assert len(pred_bytes_bin) >= RS_N_BYTES and len(gt_bytes_bin) >= RS_N_BYTES

    pred_bits_from_bin = bytes_to_bits_msb(pred_bytes_bin[:RS_N_BYTES])
    gt_bits_from_bin   = bytes_to_bits_msb(gt_bytes_bin[:RS_N_BYTES])
    # These should match the corresponding arrays if meta is consistent
    print("sanity pred_bits vs pred_bytes.bin bits BER:", ber(pred, pred_bits_from_bin))
    print("sanity gt_bits   vs gt_bytes.bin bits BER:", ber(gt, gt_bits_from_bin))

    cases = {}

    # Base
    cases["base"] = (pred, gt)

    # Flip pred bits
    cases["flip"] = (1 - pred, gt)

    # Bit-reverse each byte
    cases["bitrev"] = (bit_reverse_each_byte_from_bits(pred), gt)
    cases["flip+bitrev"] = (1 - bit_reverse_each_byte_from_bits(pred), gt)

    # Byte order reverse
    pred_bytes = bits_msb_to_bytes(pred)
    pred_bytes_rev = pred_bytes[::-1]
    pred_bits_byterev = bytes_to_bits_msb(pred_bytes_rev)
    cases["byterev"] = (pred_bits_byterev, gt)
    cases["flip+byterev"] = (1 - pred_bits_byterev, gt)

    # Byte order reverse + bit reverse each byte
    pred_bytes_rev_bits = bit_reverse_each_byte_from_bits(pred_bits_byterev)
    cases["byterev+bitrev"] = (pred_bytes_rev_bits, gt)
    cases["flip+byterev+bitrev"] = (1 - pred_bytes_rev_bits, gt)

    # Evaluate
    results = []
    for name, (p, g) in cases.items():
        results.append(eval_case(name, p, g))

    # Sort by best_shift_ber then by ber
    results.sort(key=lambda r: (r["best_shift_ber"], r["ber"]))
    print("\n=== Top candidates (sorted by best_shift_ber) ===")
    for r in results[:10]:
        print(r)

    # Save full results
    out_json = os.path.join(DUMP_DIR, "ber_scan_results.json")
    json.dump(results, open(out_json, "w"), indent=2)
    print("\nSaved:", out_json)

if __name__ == "__main__":
    main()
