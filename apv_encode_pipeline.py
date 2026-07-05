#!/usr/bin/env python3
"""
apv_encode_pipeline.py
----------------------
Complete APV encoding pipeline producing 4 output files:

  fdct_output.txt      — FDCT coefficients          (already generated)
  quant_output.txt     — quantised coefficients
  zigzag_output.txt    — zigzag-scanned coefficients
  encoded_output.txt   — run-level encoded stream

encoded_output.txt format (one symbol per line, 3 columns):
  TYPE  VALUE  SIGN
  TYPE: 0=DC_DIFF  1=RUN  2=LEVEL  3=EOB
  VALUE: abs value of the symbol
  SIGN:  0=positive  1=negative  (always 0 for RUN and EOB)

Blank line between blocks in all files.

Usage:
  python3 apv_encode_pipeline.py
  python3 apv_encode_pipeline.py --qp 32 --seed 42
"""

import argparse
import os
import random

# -- APV transform matrix T (RFC 9924 §6.3.2.3) ------------------------------
T = [
    [ 64,  64,  64,  64,  64,  64,  64,  64],
    [ 89,  75,  50,  18, -18, -50, -75, -89],
    [ 84,  35, -35, -84, -84, -35,  35,  84],
    [ 75, -18, -89, -50,  50,  89,  18, -75],
    [ 64, -64, -64,  64,  64, -64, -64,  64],
    [ 50, -89,  18,  75, -75, -18,  89, -50],
    [ 35, -84,  84, -35, -35,  84, -84,  35],
    [ 18, -50,  75, -89,  89, -75,  50, -18],
]

# -- APV level scale table (RFC 9924 §6.3.3) ---------------------------------
LEVEL_SCALE = [40, 45, 51, 57, 64, 72]
QM_FLAT     = [[16]*8 for _ in range(8)]


# -- FDCT ---------------------------------------------------------------------
def fdct_shift1(bit_depth): return bit_depth - 6   # row pass: 4 for 10-bit
def fdct_shift2():          return 9               # column pass: constant

def fdct_1d(x, shift):
    """8-point FDCT butterfly pass with correct per-pass shift (RFC 9924)."""
    e = [x[k] + x[7-k] for k in range(4)]
    o = [x[k] - x[7-k] for k in range(4)]
    ee0, eo0 = e[0]+e[3], e[0]-e[3]
    ee1, eo1 = e[1]+e[2], e[1]-e[2]
    add = 1 << (shift-1)
    out = [0]*8
    out[0] = (T[0][0]*ee0 + T[0][1]*ee1 + add) >> shift
    out[4] = (T[4][0]*ee0 + T[4][1]*ee1 + add) >> shift
    out[2] = (T[2][0]*eo0 + T[2][1]*eo1 + add) >> shift
    out[6] = (T[6][0]*eo0 + T[6][1]*eo1 + add) >> shift
    for row in (1, 3, 5, 7):
        out[row] = (sum(T[row][m]*o[m] for m in range(4)) + add) >> shift
    return out

def fdct_2d(block, bit_depth=10):
    """Full 8x8 2D FDCT. Returns result in hardware-natural order (no final
    transpose): result[c] is the column-pass output for column c."""
    s1, s2 = fdct_shift1(bit_depth), fdct_shift2()
    row_out = [fdct_1d(block[r], s1) for r in range(8)]
    cols    = [[row_out[r][c] for r in range(8)] for c in range(8)]
    return  [fdct_1d(cols[c], s2) for c in range(8)]


# -- Quantisation (RFC 9924 §6.3.3) ------------------------------------------
def quantise(block, qp, bit_depth=10, qm=None):
    if qm is None: qm = QM_FLAT
    scale   = LEVEL_SCALE[qp % 6]
    shift   = qp // 6
    bdShift = bit_depth - 2
    result  = []
    for r in range(8):
        row = []
        for c in range(8):
            coeff = block[r][c]
            step  = (qm[r][c] * scale * (1 << shift)) >> bdShift
            if step == 0:
                row.append(0)
            elif coeff >= 0:
                row.append( (coeff + step//2) // step)
            else:
                row.append(-( (-coeff + step//2) // step))
        result.append(row)
    return result


# -- Zigzag scan ---------------------------------------------------------------
def _gen_zigzag():
    scan = []
    for d in range(15):
        if d % 2 == 0:
            r = min(d, 7); c = d - r
            while r >= 0 and c <= 7:
                scan.append(r*8+c); r -= 1; c += 1
        else:
            c = min(d, 7); r = d - c
            while c >= 0 and r <= 7:
                scan.append(r*8+c); r += 1; c -= 1
    return scan

SCAN = _gen_zigzag()

def zigzag_scan(block):
    flat = [block[r][c] for r in range(8) for c in range(8)]
    return [flat[i] for i in SCAN]


# -- APV h(v) VLC encoder (RFC 9924 §6.4.3) -----------------------------------
def encode_hv(value, kParam):
    """
    Encode a non-negative integer using the APV h(v) variable length code.
    Returns the codeword as a binary string (magnitude only, no sign bit).
    """
    k         = kParam
    short_max  = (1 << k) - 1          # values 0 .. 2^k-1
    medium_max = (1 << (k+1)) - 1      # values 2^k .. 2^(k+1)-1

    if value <= short_max:
        # SHORT: prefix '1' + k-bit suffix
        bits = '1'
        if k > 0:
            bits += format(value, f'0{k}b')

    elif value <= medium_max:
        # MEDIUM: prefix '00' + k-bit suffix
        bits = '00'
        if k > 0:
            bits += format(value - (1 << k), f'0{k}b')

    else:
        # EXTENDED: prefix '01' + extension zeros + stop '1' + k-bit suffix
        sym  = 2 << k
        bits = '01'
        kk   = k
        while sym + (1 << kk) <= value:
            sym  += (1 << kk)
            kk   += 1
            bits += '0'
        bits += '1'                     # stop bit
        remainder = value - sym
        if kk > 0:
            bits += format(remainder, f'0{kk}b')

    return bits


def clip_kp(value, rsh, hi):
    return max(0, min(hi, value >> rsh))
# Symbol types
DC    = 0
RUN   = 1
LEVEL = 2
EOB   = 3

def encode_block(zz_coeffs, prev_dc, kp_dc, kp_run, kp_ac):
    """
    Encode one zigzag-ordered 64-element coefficient list.
    Returns list of (type, value, sign, kparam, codeword) tuples
    and updated (prev_dc, kp_dc, kp_run, kp_ac).

    TYPE  VALUE                       SIGN  KPARAM  CODEWORD
    DC    abs(current_dc - prev_dc)   0/1   kp_dc   magnitude + sign bits
    RUN   zero count before level     0     kp_run  magnitude bits
    LVL   abs(coeff) - 1              0/1   kp_ac  magnitude + sign bits
    EOB   0                           0     kp_run  magnitude bits
    """
    symbols = []

    # DC coefficient
    current_dc = zz_coeffs[0]
    dc_diff    = current_dc - prev_dc
    abs_dc_coeff_diff     = abs(dc_diff)
    sign_dc_coeff_diff    = 1 if dc_diff < 0 else 0
    cw_dc_diff      = encode_hv(abs_dc_coeff_diff, kp_dc)
    if abs_dc_coeff_diff != 0:
        cw_dc_diff += str(sign_dc_coeff_diff)           # sign bit appended after magnitude
    symbols.append((DC, abs_dc_coeff_diff, sign_dc_coeff_diff, kp_dc, cw_dc_diff))
    kp_dc = clip_kp(abs_dc_coeff_diff, 1, 5)      # update kParam_DC

    # AC coefficients
    run          = 0
    last_nonzero = -1
    for pos in range(1, 64):
        if zz_coeffs[pos] != 0:
            last_nonzero = pos

    for pos in range(1, 64):
        coeff = zz_coeffs[pos]
        if coeff == 0:
            run += 1
        else:
            # RUN symbol
            cw_run = encode_hv(run, kp_run)
            symbols.append((RUN, run, 0, kp_run, cw_run))
            kp_run = clip_kp(run, 2, 2)

            # LEVEL symbol  (abs_ac_coeff_minus1 per RFC 9924)
            abs_coeff      = abs(coeff)           # actual magnitude
            abs_ac_coeff_min1 = abs_coeff - 1    # bitstream value (min 0)
            sign_ac           = 1 if coeff < 0 else 0
            cw_ac             = encode_hv(abs_ac_coeff_min1, kp_ac)
            cw_ac            += str(sign_ac)         # sign bit always present for non-zero AC
            symbols.append((LEVEL, abs_ac_coeff_min1, sign_ac, kp_ac, cw_ac))
            kp_ac = clip_kp(abs_ac_coeff_min1, 2, 4)
            run    = 0

    # EOB — trailing zeros from last nonzero to end of block
    if last_nonzero < 63:
        # 'run' already holds the exact trailing-zero count accumulated since
        # the last nonzero coefficient; no further arithmetic needed.
        eob_run = run if last_nonzero >= 0 else 63
        cw_eob  = encode_hv(eob_run, kp_run)
        symbols.append((EOB, eob_run, 0, kp_run, cw_eob))

    return symbols, current_dc, kp_dc, kp_run, kp_ac


# -- Compression rate ----------------------------------------------------------
def block_compression_rate(symbols, bit_depth=10):
    """
    Compute compression statistics for one encoded block.

    Returns a dict with:
      raw_bits     — bits needed to store the original 8x8 block uncompressed
      coded_bits   — actual bits used by the h(v) codewords (incl. sign bits)
      ratio        — raw_bits / coded_bits  (>1 means compression achieved)
      bpp          — coded bits per pixel (coded_bits / 64)
      zeros        — count of LEVEL symbols that are zero (always 0, kept for clarity)
      num_symbols  — total symbols (DC + RUN + LEVEL + EOB)
    """
    raw_bits   = 64 * bit_depth                  # uncompressed size of 8x8 block
    coded_bits = sum(len(cw) for _, _, _, _, cw in symbols)

    ratio = raw_bits / coded_bits if coded_bits > 0 else float('inf')
    bpp   = coded_bits / 64.0

    return {
        "raw_bits":    raw_bits,
        "coded_bits":  coded_bits,
        "ratio":       ratio,
        "bpp":         bpp,
        "num_symbols": len(symbols),
    }
def special_cases():
    MAX = 511; MIN = -512
    cases = []
    cases.append([MAX]*8)
    cases.append([MIN]*8)
    for i in range(8):
        v=[0]*8; v[i]=128; cases.append(v)
    for n in range(8):
        d=[MAX if T[k][n]>0 else MIN if T[k][n]<0 else 0 for k in range(8)]
        cases.append(d)
    cases.append([MAX if i%2==0 else MIN for i in range(8)])
    cases.append([MIN if i%2==0 else MAX for i in range(8)])
    cases.append([0]*8)
    cases.append([MIN + i*(MAX-MIN)//7 for i in range(8)])
    cases.append([MAX - i*(MAX-MIN)//7 for i in range(8)])
    return cases


# -- Write helpers -------------------------------------------------------------
def write_8x8(f, block, fmt):
    """Write 8x8 block as 8 lines of 8 values, then blank line."""
    for row in block:
        f.write(" ".join(fmt(v) for v in row) + "\n")
    f.write("\n")

def write_1d_as_8rows(f, flat, fmt):
    """Write 64-element list as 8 lines of 8 values, then blank line."""
    for row in range(8):
        f.write(" ".join(fmt(flat[row*8+c]) for c in range(8)) + "\n")
    f.write("\n")


# -- Main ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows",      type=int,  default=100)
    parser.add_argument("--smooth",    action="store_true",
                        help="Generate smooth (image-like) blocks instead of pure random")
    parser.add_argument("--smooth_var",type=int,  default=8,
                        help="Pixel variation around DC for smooth blocks (default 8)")
    parser.add_argument("--seed",      type=int,  default=None)
    parser.add_argument("--qp",        type=int,  default=22)
    parser.add_argument("--bit_depth", type=int,  default=10)
    parser.add_argument("--in_file",   default="fdct_input.txt")
    parser.add_argument("--fdct_file", default="fdct_output.txt")
    parser.add_argument("--qout_file", default="quant_output.txt")
    parser.add_argument("--zzout_file",default="zigzag_output.txt")
    parser.add_argument("--enc_file",  default="encoded_output.txt")
    parser.add_argument("--bits_file", default="bitstream_input.txt")
    parser.add_argument("--stats_file",default="compression_stats.txt")
    parser.add_argument("--pad",       action="store_true", default=True,
                        help="Pad last 32-bit word with zeros (default True)")
    args = parser.parse_args()

    rng         = random.Random(args.seed)
    actual_seed = args.seed if args.seed is not None else rng.randint(0, 2**32)
    rng         = random.Random(actual_seed)

    MIN10 = -512; MAX10 = 511

    # Generate rows
    if args.smooth:
        # Image-like data: each 8x8 block has a slowly varying base level
        # with small random variation — concentrates energy in low frequencies
        random_rows = []
        block_base  = rng.randint(MIN10//2, MAX10//2)
        for i in range(args.rows):
            if i % 8 == 0:
                # New block: drift the base level slightly
                block_base += rng.randint(-20, 20)
                block_base  = max(MIN10, min(MAX10, block_base))
            row = [max(MIN10, min(MAX10,
                       block_base + rng.randint(-args.smooth_var, args.smooth_var)))
                   for _ in range(8)]
            random_rows.append(row)
    else:
        random_rows = [[rng.randint(MIN10, MAX10) for _ in range(8)]
                       for _ in range(args.rows)]
    all_rows = random_rows + special_cases()
    total    = len(all_rows)

    # Group into 8x8 blocks (discard remainder)
    num_blocks = total // 8
    blocks     = [all_rows[b*8:(b+1)*8] for b in range(num_blocks)]

    # -- Open all output files -------------------------------------------------
    f_in   = open(args.in_file,    "w")
    f_fdct = open(args.fdct_file,  "w")
    f_qout = open(args.qout_file,  "w")
    f_zz   = open(args.zzout_file, "w")
    f_enc  = open(args.enc_file,   "w")

    bitstream   = ""   # accumulate all codeword bits here
    block_stats = []   # per-block compression stats

    prev_dc = 0   # DC predictor resets at tile start
    kp_dc   = 0   # initial kParam_DC: clip(0, 5, PrevDcDiff>>1) with PrevDcDiff=0 → 0
    kp_run  = 0   # initial kParam_run
    kp_ac  = 0   # initial kParam_lvl

    for b, block in enumerate(blocks):

        # -- Write input (4 samples per line) ---------------------------------
        for row in block:
            f_in.write(" ".join(f"{v:6d}" for v in row[:4]) + "\n")
            f_in.write(" ".join(f"{v:6d}" for v in row[4:]) + "\n")
        f_in.write("\n")

        # -- FDCT --------------------------------------------------------------
        dct_block = fdct_2d(block, args.bit_depth)
        write_8x8(f_fdct, dct_block, lambda v: f"{v:7d}")

        # -- Quantise ----------------------------------------------------------
        q_block = quantise(dct_block, args.qp, args.bit_depth)
        write_8x8(f_qout, q_block, lambda v: f"{v:6d}")

        # -- Zigzag scan -------------------------------------------------------
        zz = zigzag_scan(q_block)
        write_1d_as_8rows(f_zz, zz, lambda v: f"{v:6d}")

        # -- Run-level encode with codewords -----------------------------------
        symbols, prev_dc, kp_dc, kp_run, kp_ac = encode_block(zz, prev_dc, kp_dc, kp_run, kp_ac)

        for sym_type, value, sign, kparam, codeword in symbols:
            # For LEVEL symbols, value = abs(coeff)-1, so restore actual coeff for display
            if sym_type == LEVEL:
                actual = (value + 1) * (-1 if sign == 1 else 1)
            else:
                actual = -value if sign == 1 else value
            f_enc.write(f"{sym_type:2d}  {value:6d}  {sign:1d}  {actual:7d}  "
                        f"{kparam:1d}  {len(codeword):2d}  {codeword}\n")
            bitstream += codeword       # accumulate bits
        f_enc.write("\n")

        # -- Compression stats for this block ---------------------------------
        stats = block_compression_rate(symbols, args.bit_depth)
        block_stats.append(stats)

        # kParam_run and kParam_lvl reset at each block start
        kp_run = 0
        kp_ac = 0

    # Close all files
    for f in [f_in, f_fdct, f_qout, f_zz, f_enc]:
        f.close()

    # -- Write bitstream file — 32 bits per line -------------------------------
    total_bits = len(bitstream)
    remainder  = total_bits % 32

    if remainder != 0:
        if args.pad:
            bitstream += '0' * (32 - remainder)   # pad last word with zeros
        else:
            bitstream  = bitstream[:total_bits - remainder]  # drop incomplete word

    with open(args.bits_file, "w") as f:
        for i in range(0, len(bitstream), 32):
            f.write(bitstream[i:i+32] + "\n")

    # -- Write compression stats file ------------------------------------------
    with open(args.stats_file, "w") as f:
        f.write(f"{'Block':>6} {'RawBits':>8} {'CodedBits':>10} "
                f"{'Ratio':>8} {'BPP':>6} {'Symbols':>8}\n")
        for b, s in enumerate(block_stats):
            f.write(f"{b:6d} {s['raw_bits']:8d} {s['coded_bits']:10d} "
                    f"{s['ratio']:8.3f} {s['bpp']:6.3f} {s['num_symbols']:8d}\n")

        total_raw   = sum(s['raw_bits']   for s in block_stats)
        total_coded = sum(s['coded_bits'] for s in block_stats)
        f.write(f"\n{'TOTAL':>6} {total_raw:8d} {total_coded:10d} "
                f"{total_raw/total_coded:8.3f} "
                f"{total_coded/(64*len(block_stats)):6.3f} "
                f"{sum(s['num_symbols'] for s in block_stats):8d}\n")

    # -- Print summary ---------------------------------------------------------
    print(f"Seed:      {actual_seed}")
    print(f"Blocks:    {num_blocks}  ({args.rows} random rows → {num_blocks} 8x8 blocks)")
    print(f"qP:        {args.qp}  bit_depth={args.bit_depth}")
    print()
    type_names = {DC:"DC", RUN:"RUN", LEVEL:"LVL", EOB:"EOB"}
    print(f"encoded_output.txt format:")
    print(f"  TYPE  VALUE  SIGN  SIGNED_VALUE  KPARAM  CW_LEN  CODEWORD")
    print(f"  0=DC_DIFF  1=RUN  2=LEVEL  3=EOB")
    print(f"  SIGN:    0=positive  1=negative  (always 0 for RUN and EOB)")
    print(f"  KPARAM:  kParam used to encode this symbol")
    print(f"  CODEWORD: h(v) binary codeword (magnitude + sign bit if applicable)")
    print()
    for fn in [args.in_file, args.fdct_file, args.qout_file,
               args.zzout_file, args.enc_file, args.bits_file, args.stats_file]:
        print(f"  {fn:<25} {os.path.getsize(fn):>8,} bytes")

    words     = len(bitstream) // 32
    last_pad  = (32 - total_bits % 32) % 32
    print()
    print(f"  Total bits:   {total_bits}")
    print(f"  32-bit words: {words}")
    print(f"  Padding:      {last_pad} zero bits added to last word"
          if last_pad else "  Last word:   exact 32-bit boundary, no padding")
    print()
    print(f"Compression summary:")
    total_raw   = sum(s['raw_bits']   for s in block_stats)
    total_coded = sum(s['coded_bits'] for s in block_stats)
    print(f"  Total raw bits:    {total_raw}  ({total_raw//8} bytes)")
    print(f"  Total coded bits:  {total_coded}  ({total_coded//8} bytes)")
    print(f"  Overall ratio:     {total_raw/total_coded:.3f}x")
    print(f"  Avg bits/pixel:    {total_coded/(64*len(block_stats)):.3f}  "
          f"(raw = {args.bit_depth}.000)")
    print()
    print(f"  Per-block ratio range: "
          f"{min(s['ratio'] for s in block_stats):.3f}x "
          f"to {max(s['ratio'] for s in block_stats):.3f}x")


if __name__ == "__main__":
    main()



#!/usr/bin/env python3
"""
apv_encode_pipeline.py
----------------------
Complete APV encoding pipeline producing 4 output files:

  fdct_output.txt      — FDCT coefficients          (already generated)
  quant_output.txt     — quantised coefficients
  zigzag_output.txt    — zigzag-scanned coefficients
  encoded_output.txt   — run-level encoded stream

encoded_output.txt format (one symbol per line, 3 columns):
  TYPE  VALUE  SIGN
  TYPE: 0=DC_DIFF  1=RUN  2=LEVEL  3=EOB
  VALUE: abs value of the symbol
  SIGN:  0=positive  1=negative  (always 0 for RUN and EOB)

Blank line between blocks in all files.

Usage:
  python3 apv_encode_pipeline.py
  python3 apv_encode_pipeline.py --qp 32 --seed 42
"""

import argparse
import os
import random
import math

# ── APV transform matrix T (RFC 9924 §6.3.2.3) ──────────────────────────────
T = [
    [ 64,  64,  64,  64,  64,  64,  64,  64],
    [ 89,  75,  50,  18, -18, -50, -75, -89],
    [ 84,  35, -35, -84, -84, -35,  35,  84],
    [ 75, -18, -89, -50,  50,  89,  18, -75],
    [ 64, -64, -64,  64,  64, -64, -64,  64],
    [ 50, -89,  18,  75, -75, -18,  89, -50],
    [ 35, -84,  84, -35, -35,  84, -84,  35],
    [ 18, -50,  75, -89,  89, -75,  50, -18],
]

# ── APV level scale table (RFC 9924 §6.3.3) ─────────────────────────────────
LEVEL_SCALE = [40, 45, 51, 57, 64, 72]
QM_FLAT     = [[16]*8 for _ in range(8)]


# ── FDCT ─────────────────────────────────────────────────────────────────────
def fdct_shift1(bit_depth): return bit_depth - 6   # row pass: 4 for 10-bit
def fdct_shift2():          return 9               # column pass: constant

def fdct_1d(x, shift):
    """8-point FDCT butterfly pass with correct per-pass shift (RFC 9924)."""
    e = [x[k] + x[7-k] for k in range(4)]
    o = [x[k] - x[7-k] for k in range(4)]
    ee0, eo0 = e[0]+e[3], e[0]-e[3]
    ee1, eo1 = e[1]+e[2], e[1]-e[2]
    add = 1 << (shift-1)
    out = [0]*8
    out[0] = (T[0][0]*ee0 + T[0][1]*ee1 + add) >> shift
    out[4] = (T[4][0]*ee0 + T[4][1]*ee1 + add) >> shift
    out[2] = (T[2][0]*eo0 + T[2][1]*eo1 + add) >> shift
    out[6] = (T[6][0]*eo0 + T[6][1]*eo1 + add) >> shift
    for row in (1, 3, 5, 7):
        out[row] = (sum(T[row][m]*o[m] for m in range(4)) + add) >> shift
    return out

def fdct_2d(block, bit_depth=10):
    """Full 8x8 2D FDCT. Returns result in hardware-natural order (no final
    transpose): result[c] is the column-pass output for column c."""
    s1, s2 = fdct_shift1(bit_depth), fdct_shift2()
    row_out = [fdct_1d(block[r], s1) for r in range(8)]
    cols    = [[row_out[r][c] for r in range(8)] for c in range(8)]
    return  [fdct_1d(cols[c], s2) for c in range(8)]


# ── Quantisation (RFC 9924 §6.3.3) ──────────────────────────────────────────
def quantise(block, qp, bit_depth=10, qm=None):
    if qm is None: qm = QM_FLAT
    scale   = LEVEL_SCALE[qp % 6]
    shift   = qp // 6
    bdShift = bit_depth - 2
    result  = []
    for r in range(8):
        row = []
        for c in range(8):
            coeff = block[r][c]
            step  = (qm[r][c] * scale * (1 << shift)) >> bdShift
            if step == 0:
                row.append(0)
            elif coeff >= 0:
                row.append( (coeff + step//2) // step)
            else:
                row.append(-( (-coeff + step//2) // step))
        result.append(row)
    return result


# ── Zigzag scan ───────────────────────────────────────────────────────────────
def _gen_zigzag():
    scan = []
    for d in range(15):
        if d % 2 == 0:
            r = min(d, 7); c = d - r
            while r >= 0 and c <= 7:
                scan.append(r*8+c); r -= 1; c += 1
        else:
            c = min(d, 7); r = d - c
            while c >= 0 and r <= 7:
                scan.append(r*8+c); r += 1; c -= 1
    return scan

SCAN = _gen_zigzag()

def zigzag_scan(block):
    flat = [block[r][c] for r in range(8) for c in range(8)]
    return [flat[i] for i in SCAN]


# ── APV h(v) VLC encoder (RFC 9924 §6.4.3) ───────────────────────────────────
def encode_hv(value, kParam):
    """
    Encode a non-negative integer using the APV h(v) variable length code.
    Returns the codeword as a binary string (magnitude only, no sign bit).
    """
    k         = kParam
    short_max  = (1 << k) - 1          # values 0 .. 2^k-1
    medium_max = (1 << (k+1)) - 1      # values 2^k .. 2^(k+1)-1

    if value <= short_max:
        # SHORT: prefix '1' + k-bit suffix
        bits = '1'
        if k > 0:
            bits += format(value, f'0{k}b')

    elif value <= medium_max:
        # MEDIUM: prefix '00' + k-bit suffix
        bits = '00'
        if k > 0:
            bits += format(value - (1 << k), f'0{k}b')

    else:
        # EXTENDED: prefix '01' + extension zeros + stop '1' + k-bit suffix
        sym  = 2 << k
        bits = '01'
        kk   = k
        while sym + (1 << kk) <= value:
            sym  += (1 << kk)
            kk   += 1
            bits += '0'
        bits += '1'                     # stop bit
        remainder = value - sym
        if kk > 0:
            bits += format(remainder, f'0{kk}b')

    return bits


def clip_kp(value, rsh, hi):
    return max(0, min(hi, value >> rsh))
# Symbol types
DC    = 0
RUN   = 1
LEVEL = 2
EOB   = 3

def encode_block(zz_coeffs, prev_dc, kp_dc, kp_run, kp_lvl):
    """
    Encode one zigzag-ordered 64-element coefficient list.
    Returns list of (type, value, sign, kparam, codeword) tuples
    and updated (prev_dc, kp_dc, kp_run, kp_lvl).

    TYPE  VALUE                       SIGN  KPARAM  CODEWORD
    DC    abs(current_dc - prev_dc)   0/1   kp_dc   magnitude + sign bits
    RUN   zero count before level     0     kp_run  magnitude bits
    LVL   abs(coeff) - 1              0/1   kp_lvl  magnitude + sign bits
    EOB   0                           0     kp_run  magnitude bits
    """
    symbols = []

    # DC coefficient
    current_dc = zz_coeffs[0]
    dc_diff    = current_dc - prev_dc
    abs_dc     = abs(dc_diff)
    sign_dc    = 1 if dc_diff < 0 else 0
    cw_dc      = encode_hv(abs_dc, kp_dc)
    if abs_dc != 0:
        cw_dc += str(sign_dc)           # sign bit appended after magnitude
    symbols.append((DC, abs_dc, sign_dc, kp_dc, cw_dc))
    kp_dc = clip_kp(abs_dc, 1, 5)      # update kParam_DC

    # AC coefficients
    run          = 0
    last_nonzero = -1
    for pos in range(1, 64):
        if zz_coeffs[pos] != 0:
            last_nonzero = pos

    for pos in range(1, 64):
        coeff = zz_coeffs[pos]
        if coeff == 0:
            run += 1
        else:
            # RUN symbol
            cw_run = encode_hv(run, kp_run)
            symbols.append((RUN, run, 0, kp_run, cw_run))
            kp_run = clip_kp(run, 2, 2)

            # LEVEL symbol  (abs_ac_coeff_minus1)
            abs_lvl = abs(coeff) - 1
            sign_lv = 1 if coeff < 0 else 0
            cw_lvl  = encode_hv(abs_lvl, kp_lvl)
            if abs_lvl + 1 != 0:        # always true since coeff != 0
                cw_lvl += str(sign_lv)  # sign bit
            symbols.append((LEVEL, abs_lvl, sign_lv, kp_lvl, cw_lvl))
            kp_lvl = clip_kp(abs_lvl, 2, 4)
            run    = 0

    # EOB — trailing zeros from last nonzero to end of block
    if last_nonzero < 63:
        # 'run' already holds the exact trailing-zero count accumulated since
        # the last nonzero coefficient; no further arithmetic needed.
        eob_run = run if last_nonzero >= 0 else 63
        cw_eob  = encode_hv(eob_run, kp_run)
        symbols.append((EOB, eob_run, 0, kp_run, cw_eob))

    return symbols, current_dc, kp_dc, kp_run, kp_lvl


# ── Compression rate ──────────────────────────────────────────────────────────
def block_compression_rate(symbols, bit_depth=10):
    """
    Compute compression statistics for one encoded block.

    Returns a dict with:
      raw_bits     — bits needed to store the original 8x8 block uncompressed
      coded_bits   — actual bits used by the h(v) codewords (incl. sign bits)
      ratio        — raw_bits / coded_bits  (>1 means compression achieved)
      bpp          — coded bits per pixel (coded_bits / 64)
      zeros        — count of LEVEL symbols that are zero (always 0, kept for clarity)
      num_symbols  — total symbols (DC + RUN + LEVEL + EOB)
    """
    raw_bits   = 64 * bit_depth                  # uncompressed size of 8x8 block
    coded_bits = sum(len(cw) for _, _, _, _, cw in symbols)

    ratio = raw_bits / coded_bits if coded_bits > 0 else float('inf')
    bpp   = coded_bits / 64.0

    return {
        "raw_bits":    raw_bits,
        "coded_bits":  coded_bits,
        "ratio":       ratio,
        "bpp":         bpp,
        "num_symbols": len(symbols),
    }
def special_cases():
    MAX = 511; MIN = -512
    cases = []
    cases.append([MAX]*8)
    cases.append([MIN]*8)
    for i in range(8):
        v=[0]*8; v[i]=128; cases.append(v)
    for n in range(8):
        d=[MAX if T[k][n]>0 else MIN if T[k][n]<0 else 0 for k in range(8)]
        cases.append(d)
    cases.append([MAX if i%2==0 else MIN for i in range(8)])
    cases.append([MIN if i%2==0 else MAX for i in range(8)])
    cases.append([0]*8)
    cases.append([MIN + i*(MAX-MIN)//7 for i in range(8)])
    cases.append([MAX - i*(MAX-MIN)//7 for i in range(8)])
    return cases


# ── Write helpers ─────────────────────────────────────────────────────────────
def write_8x8(f, block, fmt):
    """Write 8x8 block as 8 lines of 8 values, then blank line."""
    for row in block:
        f.write(" ".join(fmt(v) for v in row) + "\n")
    f.write("\n")

def write_1d_as_8rows(f, flat, fmt):
    """Write 64-element list as 8 lines of 8 values, then blank line."""
    for row in range(8):
        f.write(" ".join(fmt(flat[row*8+c]) for c in range(8)) + "\n")
    f.write("\n")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows",      type=int,  default=100)
    parser.add_argument("--smooth",    action="store_true",
                        help="Generate smooth (image-like) blocks instead of pure random")
    parser.add_argument("--smooth_var",type=int,  default=8,
                        help="Pixel variation around DC for smooth blocks (default 8)")
    parser.add_argument("--seed",      type=int,  default=None)
    parser.add_argument("--qp",        type=int,  default=22)
    parser.add_argument("--bit_depth", type=int,  default=10)
    parser.add_argument("--in_file",   default="fdct_input.txt")
    parser.add_argument("--fdct_file", default="fdct_output.txt")
    parser.add_argument("--qout_file", default="quant_output.txt")
    parser.add_argument("--zzout_file",default="zigzag_output.txt")
    parser.add_argument("--enc_file",  default="encoded_output.txt")
    parser.add_argument("--bits_file", default="bitstream_input.txt")
    parser.add_argument("--stats_file",default="compression_stats.txt")
    parser.add_argument("--pad",       action="store_true", default=True,
                        help="Pad last 32-bit word with zeros (default True)")
    args = parser.parse_args()

    rng         = random.Random(args.seed)
    actual_seed = args.seed if args.seed is not None else rng.randint(0, 2**32)
    rng         = random.Random(actual_seed)

    MIN10 = -512; MAX10 = 511

    # Generate rows
    if args.smooth:
        # Image-like data: each 8x8 block has a slowly varying base level
        # with small random variation — concentrates energy in low frequencies
        random_rows = []
        block_base  = rng.randint(MIN10//2, MAX10//2)
        for i in range(args.rows):
            if i % 8 == 0:
                # New block: drift the base level slightly
                block_base += rng.randint(-20, 20)
                block_base  = max(MIN10, min(MAX10, block_base))
            row = [max(MIN10, min(MAX10,
                       block_base + rng.randint(-args.smooth_var, args.smooth_var)))
                   for _ in range(8)]
            random_rows.append(row)
    else:
        random_rows = [[rng.randint(MIN10, MAX10) for _ in range(8)]
                       for _ in range(args.rows)]
    all_rows = random_rows + special_cases()
    total    = len(all_rows)

    # Group into 8x8 blocks (discard remainder)
    num_blocks = total // 8
    blocks     = [all_rows[b*8:(b+1)*8] for b in range(num_blocks)]

    # ── Open all output files ─────────────────────────────────────────────────
    f_in   = open(args.in_file,    "w")
    f_fdct = open(args.fdct_file,  "w")
    f_qout = open(args.qout_file,  "w")
    f_zz   = open(args.zzout_file, "w")
    f_enc  = open(args.enc_file,   "w")

    bitstream   = ""   # accumulate all codeword bits here
    block_stats = []   # per-block compression stats

    prev_dc = 0   # DC predictor resets at tile start
    kp_dc   = 0   # initial kParam_DC: clip(0, 5, PrevDcDiff>>1) with PrevDcDiff=0 → 0
    kp_run  = 0   # initial kParam_run
    kp_lvl  = 0   # initial kParam_lvl

    for b, block in enumerate(blocks):

        # ── Write input (4 samples per line) ─────────────────────────────────
        for row in block:
            f_in.write(" ".join(f"{v:6d}" for v in row[:4]) + "\n")
            f_in.write(" ".join(f"{v:6d}" for v in row[4:]) + "\n")
        f_in.write("\n")

        # ── FDCT ──────────────────────────────────────────────────────────────
        dct_block = fdct_2d(block, args.bit_depth)
        write_8x8(f_fdct, dct_block, lambda v: f"{v:7d}")

        # ── Quantise ──────────────────────────────────────────────────────────
        q_block = quantise(dct_block, args.qp, args.bit_depth)
        write_8x8(f_qout, q_block, lambda v: f"{v:6d}")

        # ── Zigzag scan ───────────────────────────────────────────────────────
        zz = zigzag_scan(q_block)
        write_1d_as_8rows(f_zz, zz, lambda v: f"{v:6d}")

        # ── Run-level encode with codewords ───────────────────────────────────
        symbols, prev_dc, kp_dc, kp_run, kp_lvl = encode_block(
            zz, prev_dc, kp_dc, kp_run, kp_lvl)

        for sym_type, value, sign, kparam, codeword in symbols:
            # For LEVEL symbols, value = abs(coeff)-1, so restore actual coeff for display
            if sym_type == LEVEL:
                actual = (value + 1) * (-1 if sign == 1 else 1)
            else:
                actual = -value if sign == 1 else value
            f_enc.write(f"{sym_type:2d}  {value:6d}  {sign:1d}  {actual:7d}  "
                        f"{kparam:1d}  {len(codeword):2d}  {codeword}\n")
            bitstream += codeword       # accumulate bits
        f_enc.write("\n")

        # ── Compression stats for this block ─────────────────────────────────
        stats = block_compression_rate(symbols, args.bit_depth)
        block_stats.append(stats)

        # kParam_run and kParam_lvl reset at each block start
        kp_run = 0
        kp_lvl = 0

    # Close all files
    for f in [f_in, f_fdct, f_qout, f_zz, f_enc]:
        f.close()

    # ── Write bitstream file — 32 bits per line ───────────────────────────────
    total_bits = len(bitstream)
    remainder  = total_bits % 32

    if remainder != 0:
        if args.pad:
            bitstream += '0' * (32 - remainder)   # pad last word with zeros
        else:
            bitstream  = bitstream[:total_bits - remainder]  # drop incomplete word

    with open(args.bits_file, "w") as f:
        for i in range(0, len(bitstream), 32):
            f.write(bitstream[i:i+32] + "\n")

    # ── Write compression stats file ──────────────────────────────────────────
    with open(args.stats_file, "w") as f:
        f.write(f"{'Block':>6} {'RawBits':>8} {'CodedBits':>10} "
                f"{'Ratio':>8} {'BPP':>6} {'Symbols':>8}\n")
        for b, s in enumerate(block_stats):
            f.write(f"{b:6d} {s['raw_bits']:8d} {s['coded_bits']:10d} "
                    f"{s['ratio']:8.3f} {s['bpp']:6.3f} {s['num_symbols']:8d}\n")

        total_raw   = sum(s['raw_bits']   for s in block_stats)
        total_coded = sum(s['coded_bits'] for s in block_stats)
        f.write(f"\n{'TOTAL':>6} {total_raw:8d} {total_coded:10d} "
                f"{total_raw/total_coded:8.3f} "
                f"{total_coded/(64*len(block_stats)):6.3f} "
                f"{sum(s['num_symbols'] for s in block_stats):8d}\n")

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"Seed:      {actual_seed}")
    print(f"Blocks:    {num_blocks}  ({args.rows} random rows → {num_blocks} 8x8 blocks)")
    print(f"qP:        {args.qp}  bit_depth={args.bit_depth}")
    print()
    type_names = {DC:"DC", RUN:"RUN", LEVEL:"LVL", EOB:"EOB"}
    print(f"encoded_output.txt format:")
    print(f"  TYPE  VALUE  SIGN  SIGNED_VALUE  KPARAM  CW_LEN  CODEWORD")
    print(f"  0=DC_DIFF  1=RUN  2=LEVEL  3=EOB")
    print(f"  SIGN:    0=positive  1=negative  (always 0 for RUN and EOB)")
    print(f"  KPARAM:  kParam used to encode this symbol")
    print(f"  CODEWORD: h(v) binary codeword (magnitude + sign bit if applicable)")
    print()
    for fn in [args.in_file, args.fdct_file, args.qout_file,
               args.zzout_file, args.enc_file, args.bits_file, args.stats_file]:
        print(f"  {fn:<25} {os.path.getsize(fn):>8,} bytes")

    words     = len(bitstream) // 32
    last_pad  = (32 - total_bits % 32) % 32
    print()
    print(f"  Total bits:   {total_bits}")
    print(f"  32-bit words: {words}")
    print(f"  Padding:      {last_pad} zero bits added to last word"
          if last_pad else "  Last word:   exact 32-bit boundary, no padding")
    print()
    print(f"Compression summary:")
    total_raw   = sum(s['raw_bits']   for s in block_stats)
    total_coded = sum(s['coded_bits'] for s in block_stats)
    print(f"  Total raw bits:    {total_raw}  ({total_raw//8} bytes)")
    print(f"  Total coded bits:  {total_coded}  ({total_coded//8} bytes)")
    print(f"  Overall ratio:     {total_raw/total_coded:.3f}x")
    print(f"  Avg bits/pixel:    {total_coded/(64*len(block_stats)):.3f}  "
          f"(raw = {args.bit_depth}.000)")
    print()
    print(f"  Per-block ratio range: "
          f"{min(s['ratio'] for s in block_stats):.3f}x "
          f"to {max(s['ratio'] for s in block_stats):.3f}x")


if __name__ == "__main__":
    main()


# =============================================================================
# Generic APV frame geometry (per RFC 9924 / draft-lim-apv)
#   Section 4.2   : chroma format (SubWidthC / SubHeightC), Table 2
#   Section 5.3.15: TrSize = 8 (transform block size)
#   Section 9.3   : profile bit-depth constraints
#   Section 9.4   : level luma sample rate limits, Table 4
#   Section 9.4.2 : typical resolution/frame-rate configurations, Table 5
# =============================================================================

# ---- Section 4.2, Table 2: chroma_format_idc -> (SubWidthC, SubHeightC) ----
CHROMA_FORMAT_SUBSAMPLING = {
    0: (1, 1),   # 4:4:4 (monochrome uses idc=0 too, chroma planes absent)
    2: (2, 1),   # 4:2:2
    3: (2, 2),   # 4:2:0  (not defined as a profile in this draft, kept for completeness)
}

CHROMA_FORMAT_NAMES = {
    0: "4:4:4",
    2: "4:2:2",
    3: "4:2:0",
}

TR_SIZE = 8   # Section 5.3.15 transform block size, fixed for all profiles

# ---- Section 9.4, Table 4: level_idc -> MaxLumaSr (samples/second) --------
# level number -> level_idc is 30 * level number (Section 9.4)
LEVEL_MAX_LUMA_SAMPLE_RATE = {
    1.0:      3_041_280,
    1.1:      6_082_560,
    2.0:     15_667_200,
    2.1:     31_334_400,
    3.0:     66_846_720,
    3.1:    133_693_440,
    4.0:    265_420_800,
    4.1:    530_841_600,
    5.0:  1_061_683_200,
    5.1:  2_123_366_400,
    6.0:  4_777_574_400,
    6.1:  8_493_465_600,
    7.0: 16_986_931_200,
    7.1: 33_973_862_400,
}

# ---- Section 9.4.2, Table 4: level+band -> Max coded data rate (bit/s) ----
# This is a SEPARATE, independent constraint from MaxLumaSr above: a
# conformant bitstream must satisfy BOTH its level's luma sample rate
# limit AND its (level, band) pair's coded data rate limit. band_idc
# MUST be in range 0..3 (Section 5.3.6). Values below are Mbit/s from
# Table 4, converted to bit/s.
LEVEL_BAND_MAX_CODED_DATA_RATE_BPS = {
    1.0: {0:  8_000_000, 1: 11_000_000, 2: 15_000_000, 3: 23_000_000},
    1.1: {0: 16_000_000, 1: 21_000_000, 2: 30_000_000, 3: 45_000_000},
    2.0: {0: 39_000_000, 1: 54_000_000, 2: 76_000_000, 3: 114_000_000},
    2.1: {0: 78_000_000, 1: 108_000_000, 2: 152_000_000, 3: 227_000_000},
    3.0: {0: 114_000_000, 1: 159_000_000, 2: 222_000_000, 3: 333_000_000},
    3.1: {0: 227_000_000, 1: 317_000_000, 2: 444_000_000, 3: 666_000_000},
    4.0: {0: 455_000_000, 1: 637_000_000, 2: 892_000_000, 3: 1_338_000_000},
    4.1: {0: 910_000_000, 1: 1_274_000_000, 2: 1_784_000_000, 3: 2_675_000_000},
    5.0: {0: 1_820_000_000, 1: 2_548_000_000, 2: 3_567_000_000, 3: 5_350_000_000},
    5.1: {0: 3_639_000_000, 1: 5_095_000_000, 2: 7_133_000_000, 3: 10_699_000_000},
    6.0: {0: 7_278_000_000, 1: 10_189_000_000, 2: 14_265_000_000, 3: 21_397_000_000},
    6.1: {0: 14_556_000_000, 1: 20_378_000_000, 2: 28_529_000_000, 3: 42_793_000_000},
    7.0: {0: 29_111_000_000, 1: 40_756_000_000, 2: 57_058_000_000, 3: 85_586_000_000},
    7.1: {0: 58_222_000_000, 1: 81_511_000_000, 2: 114_115_000_000, 3: 171_172_000_000},
}

# ---- Section 9.3: profile_idc -> constraints (chroma_format_idc range,
#      bit_depth_minus8 range) ----
PROFILE_CONSTRAINTS = {
    33: {'name': '422-10',  'chroma_format_idc': (2, 2), 'bit_depth_minus8': (2, 2)},
    44: {'name': '422-12',  'chroma_format_idc': (2, 2), 'bit_depth_minus8': (2, 4)},
    55: {'name': '444-10',  'chroma_format_idc': (2, 3), 'bit_depth_minus8': (2, 2)},
    66: {'name': '444-12',  'chroma_format_idc': (2, 3), 'bit_depth_minus8': (2, 4)},
    77: {'name': '4444-10', 'chroma_format_idc': (2, 4), 'bit_depth_minus8': (2, 2)},
    88: {'name': '4444-12', 'chroma_format_idc': (2, 4), 'bit_depth_minus8': (2, 4)},
    99: {'name': '400-10',  'chroma_format_idc': (0, 0), 'bit_depth_minus8': (2, 2)},
}


def check_profile_conformance(profile_idc, chroma_format_idc, bit_depth):
    """
    Verify (chroma_format_idc, bit_depth) satisfy the constraints of a
    given profile_idc per Section 9.3. Raises AssertionError with a
    descriptive message on violation. Returns the profile info dict on
    success.
    """
    assert profile_idc in PROFILE_CONSTRAINTS, \
        f"Unknown profile_idc={profile_idc}"

    info = PROFILE_CONSTRAINTS[profile_idc]
    cf_lo, cf_hi = info['chroma_format_idc']
    bd_lo, bd_hi = info['bit_depth_minus8']
    bit_depth_minus8 = bit_depth - 8

    assert cf_lo <= chroma_format_idc <= cf_hi, \
        (f"profile {info['name']} requires chroma_format_idc in "
         f"[{cf_lo},{cf_hi}], got {chroma_format_idc}")
    assert bd_lo <= bit_depth_minus8 <= bd_hi, \
        (f"profile {info['name']} requires bit_depth_minus8 in "
         f"[{bd_lo},{bd_hi}] (BitDepth {bd_lo+8}-{bd_hi+8}), got "
         f"bit_depth_minus8={bit_depth_minus8} (BitDepth={bit_depth})")

    return info


def check_coded_data_rate_conformance(bits_per_second, level, band=3):
    """
    Verify a bitstream's coded data rate (bits/second) does not exceed
    the Max coded data rate for the given (level, band) pair, per
    Table 4. band defaults to 3 (the most permissive band) since a
    specific band is a deployment choice, not implied by resolution
    alone -- callers targeting a specific band should pass it explicitly.

    Raises AssertionError on violation. Returns True on success.
    """
    assert level in LEVEL_BAND_MAX_CODED_DATA_RATE_BPS, f"Unknown level {level}"
    assert band in (0, 1, 2, 3), f"band_idc must be 0-3, got {band}"

    limit = LEVEL_BAND_MAX_CODED_DATA_RATE_BPS[level][band]
    assert bits_per_second <= limit, \
        (f"coded data rate {bits_per_second:,.0f} bit/s exceeds level "
         f"{level} band {band} limit {limit:,} bit/s")
    return True


def find_min_conformant_level_and_band(luma_sample_rate, bits_per_second):
    """
    Given a luma sample rate AND a coded data rate, find the lowest
    (level, band) pair that satisfies BOTH constraints simultaneously --
    the sample-rate constraint (Table 4 left column) and the bit-rate
    constraint (Table 4 right columns, per band). This is the two-part
    conformance check Section 9.4.1 actually requires; checking sample
    rate alone (as done previously) is insufficient for real conformance.

    Returns (level, band) or (None, None) if no defined combination
    covers both requirements.
    """
    for level in sorted(LEVEL_MAX_LUMA_SAMPLE_RATE):
        if LEVEL_MAX_LUMA_SAMPLE_RATE[level] < luma_sample_rate:
            continue
        for band in (0, 1, 2, 3):
            if LEVEL_BAND_MAX_CODED_DATA_RATE_BPS[level][band] >= bits_per_second:
                return level, band
    return None, None


# ---- Section 9.4.2, Table 5: named reference resolutions (informative) ----
# name -> (luma_width, luma_height, fps)
NAMED_RESOLUTIONS = {
    "HD_1080P":       (1920, 1080, 60),
    "HD_1080P_DCI":   (2048, 1080, 60),
    "UHD_4K":         (3840, 2160, 60),
    "UHD_4K_120":     (3840, 2160, 120),
    "DCI_4K":         (4096, 2160, 60),
    "UHD_8K":         (7680, 4320, 60),
    "UHD_8K_120":     (7680, 4320, 120),
    "DCI_8K":         (8192, 4320, 60),
}


def get_frame_geometry(luma_width, luma_height, chroma_format_idc=2,
                        bit_depth=10, fps=60):
    """
    Generic frame geometry calculator for ANY resolution / chroma format /
    bit depth / frame rate combination, per the formulas in Section 4.2
    and Section 5.3.6 of the APV specification.

    Parameters:
      luma_width, luma_height : frame dimensions in luma samples
                                 (MUST be multiples of TR_SIZE=8; MUST be
                                 multiples of 2 when chroma_format_idc==2,
                                 per Section 5.3.6 frame_width constraint)
      chroma_format_idc       : 0 (4:4:4), 2 (4:2:2), or 3 (4:2:0)
                                 -- see CHROMA_FORMAT_SUBSAMPLING
      bit_depth                : BitDepth (e.g. 10 for the 422-10 profile)
      fps                      : frame rate, used only to derive the luma
                                  sample rate for level conformance checks

    Returns a dict with luma/chroma plane sizes in samples and in 8x8
    transform blocks, plus the derived luma sample rate. Raises
    AssertionError if the dimensions violate the spec's divisibility
    constraints.
    """
    assert chroma_format_idc in CHROMA_FORMAT_SUBSAMPLING, \
        f"Unsupported chroma_format_idc={chroma_format_idc}"

    assert luma_width % TR_SIZE == 0, \
        f"luma_width={luma_width} must be a multiple of TR_SIZE={TR_SIZE}"
    assert luma_height % TR_SIZE == 0, \
        f"luma_height={luma_height} must be a multiple of TR_SIZE={TR_SIZE}"

    if chroma_format_idc == 2:
        # Section 5.3.6: frame_width MUST be a multiple of 2 when
        # chroma_format_idc == 2
        assert luma_width % 2 == 0, \
            "frame_width must be a multiple of 2 for chroma_format_idc=2"

    sub_w_c, sub_h_c = CHROMA_FORMAT_SUBSAMPLING[chroma_format_idc]

    chroma_width  = luma_width  // sub_w_c
    chroma_height = luma_height // sub_h_c

    assert chroma_width % TR_SIZE == 0, \
        (f"derived chroma_width={chroma_width} is not a multiple of "
         f"TR_SIZE={TR_SIZE} for chroma_format_idc={chroma_format_idc}")
    assert chroma_height % TR_SIZE == 0, \
        (f"derived chroma_height={chroma_height} is not a multiple of "
         f"TR_SIZE={TR_SIZE} for chroma_format_idc={chroma_format_idc}")

    luma_samples   = luma_width * luma_height
    chroma_samples = chroma_width * chroma_height   # per Cb or Cr plane

    luma_blocks   = (luma_width  // TR_SIZE) * (luma_height  // TR_SIZE)
    chroma_blocks = (chroma_width // TR_SIZE) * (chroma_height // TR_SIZE)

    has_chroma = chroma_format_idc != 0 or True   # idc=0 can be monochrome
    # (monochrome vs 4:4:4-with-idc-0 is a separate flag in the real spec;
    #  this helper always reports chroma plane sizes for idc in the table)

    luma_sample_rate = luma_samples * fps

    return {
        'chroma_format_idc': chroma_format_idc,
        'chroma_format_name': CHROMA_FORMAT_NAMES[chroma_format_idc],
        'bit_depth': bit_depth,
        'fps': fps,
        'luma_width': luma_width, 'luma_height': luma_height,
        'chroma_width': chroma_width, 'chroma_height': chroma_height,
        'luma_samples_per_frame': luma_samples,
        'chroma_samples_per_frame': chroma_samples,   # per Cb or Cr plane
        'luma_blocks_per_frame': luma_blocks,
        'chroma_blocks_per_frame': chroma_blocks,     # per Cb or Cr plane
        'total_blocks_per_frame': luma_blocks + 2 * chroma_blocks,
        'luma_sample_rate': luma_sample_rate,
    }


def get_named_frame_geometry(name, chroma_format_idc=2, bit_depth=10):
    """
    Convenience wrapper: look up a named reference resolution from
    NAMED_RESOLUTIONS (e.g. 'UHD_4K', 'UHD_8K', 'DCI_4K') and compute its
    full geometry via get_frame_geometry(). fps comes from the table
    entry itself (Section 9.4.2, Table 5).
    """
    assert name in NAMED_RESOLUTIONS, \
        (f"Unknown resolution name '{name}'. "
         f"Available: {sorted(NAMED_RESOLUTIONS)}")

    luma_w, luma_h, fps = NAMED_RESOLUTIONS[name]
    return get_frame_geometry(luma_w, luma_h, chroma_format_idc,
                               bit_depth, fps)


def find_min_conformant_level(luma_sample_rate):
    """
    Given a luma sample rate (samples/second), return the lowest level
    number (per Table 4) whose MaxLumaSr is greater than or equal to it.
    Returns None if no defined level covers the rate.
    """
    candidates = sorted(
        (lvl for lvl, max_sr in LEVEL_MAX_LUMA_SAMPLE_RATE.items()
         if max_sr >= luma_sample_rate),
    )
    return candidates[0] if candidates else None


def check_geometry_conformance(geometry, required_bit_depth=None,
                                max_level=None):
    """
    Verify a geometry dict (as returned by get_frame_geometry /
    get_named_frame_geometry) conforms to:
      - required_bit_depth, if given (e.g. 10 for the 422-10 profile)
      - max_level, if given: the geometry's luma sample rate MUST NOT
        exceed LEVEL_MAX_LUMA_SAMPLE_RATE[max_level]

    Always computes and attaches 'min_conformant_level' to the returned
    dict, showing the lowest level that covers this geometry's rate,
    regardless of whether max_level was specified.

    Raises AssertionError on any violation. Returns the (possibly
    annotated) geometry dict on success.
    """
    geometry = dict(geometry)   # don't mutate caller's dict

    if required_bit_depth is not None:
        assert geometry['bit_depth'] == required_bit_depth, \
            (f"profile requires BitDepth={required_bit_depth}, "
             f"got {geometry['bit_depth']}")

    min_level = find_min_conformant_level(geometry['luma_sample_rate'])
    geometry['min_conformant_level'] = min_level

    if max_level is not None:
        assert max_level in LEVEL_MAX_LUMA_SAMPLE_RATE, \
            f"Unknown level {max_level}"
        limit = LEVEL_MAX_LUMA_SAMPLE_RATE[max_level]
        assert geometry['luma_sample_rate'] <= limit, \
            (f"luma sample rate {geometry['luma_sample_rate']:,}/s exceeds "
             f"Level {max_level} limit {limit:,}/s")

    return geometry


def generate_samples_for_geometry(geometry, seed=42, plane='luma',
                                   max_blocks=None):
    """
    Generate 8x8 pixel blocks whose count matches the given geometry's
    plane block count (luma_blocks_per_frame or chroma_blocks_per_frame),
    using BitDepth from the geometry dict.

    Parameters:
      geometry   : a dict from get_frame_geometry() / get_named_frame_geometry()
      plane      : 'luma' or 'chroma' -- selects which plane's block
                   count to generate
      max_blocks : optional cap for practical test runtimes; if set,
                   generates min(max_blocks, full_block_count) blocks.
                   The full conformant count is always reported alongside.

    Returns (rows, info) where rows is a flat list of 8-sample rows ready
    for fdct_2d() input, and info is the geometry dict annotated with
    'plane', 'full_block_count', and 'generated_block_count'.
    """
    assert plane in ('luma', 'chroma'), f"plane must be 'luma' or 'chroma', got {plane}"

    full_block_count = (geometry['luma_blocks_per_frame'] if plane == 'luma'
                         else geometry['chroma_blocks_per_frame'])

    num_blocks = full_block_count if max_blocks is None \
                 else min(max_blocks, full_block_count)

    rng = random.Random(seed)
    bit_depth = geometry['bit_depth']
    MIN_V, MAX_V = -(1 << (bit_depth - 1)), (1 << (bit_depth - 1)) - 1

    rows = [[rng.randint(MIN_V, MAX_V) for _ in range(8)]
             for _ in range(num_blocks * 8)]

    info = dict(geometry)
    info['plane'] = plane
    info['full_block_count'] = full_block_count
    info['generated_block_count'] = num_blocks

    return rows, info



# =============================================================================
# Frame-level worst-case blocks for the entropy decoder
#
# Each function returns a 64-element ZIGZAG-ORDERED COEFFICIENT ARRAY --
# exactly the same shape encode_block() expects and exactly the same
# shape a real quantised DCT block produces. These are NOT hand-built
# symbol lists: they are coefficient arrays that, when run through the
# real encode_block(), naturally produce worst-case symbol patterns as
# a consequence of realistic block content, in the same way special_cases()
# produces worst-case PIXEL blocks.
#
# build_worst_case_frame() assembles many such blocks -- worst-case ones
# interleaved with ordinary random blocks -- into ONE continuous frame,
# so kParam state (kp_dc persists across the whole frame; kp_run/kp_lvl
# reset every block, per encode_block()'s calling convention in main())
# carries over exactly as it would in a real bitstream. The challenge is
# INSIDE the frame, in context, not as isolated fabricated symbols.
# =============================================================================

PIXEL_MIN, PIXEL_MAX = -512, 511   # 10-bit signed PIXEL range (BitDepth=10)
# NOTE: these are PIXEL bounds. The legal COEFFICIENT range per spec
# Section 5.3.15 is [-32768, 32767], independent of BitDepth -- but a
# decoder that only ever needs to support a SPECIFIC bit depth (e.g.
# your 10-bit-only design) should NOT be tested against that flat
# structural ceiling, since it describes what the format permits in
# general, not what your bit-depth-constrained system can ever
# actually receive. Every worst-case generator below derives its
# bounds from get_achievable_coeff_bounds(bit_depth), which computes
# the REAL achievable coefficient range for a given bit depth via
# actual fdct_2d() + quantise() -- never a hardcoded out-of-range
# constant.

SPEC_COEFF_MIN, SPEC_COEFF_MAX = -32768, 32767   # flat structural ceiling,
# Section 5.3.15 -- use ONLY if your decoder must be fully general
# across all bit depths / arbitrary encoders. Otherwise, use
# get_achievable_coeff_bounds(bit_depth) below.


def get_achievable_coeff_bounds(bit_depth=10, qp=0):
    """
    Compute the REAL maximum-magnitude positive and negative quantised
    coefficient values achievable from a genuinely bit_depth-constrained
    PIXEL input, via actual fdct_2d() + quantise() -- NOT the spec's
    flat, bit-depth-independent structural ceiling.

    This is the correct bound to use when your decoder's design
    constraint is "input is always N-bit pixels": it tells you what
    your encoder can ACTUALLY produce, so your worst-case test vectors
    stay within what your system needs to support, instead of
    accidentally testing against out-of-range values your design was
    never meant to handle.

    QP defaults to 0 (the finest, least-lossy legal QP), which
    preserves the most coefficient energy and is therefore the worst
    case (largest achievable magnitude) for a given bit depth.

    Returns a dict: {'dc_max', 'dc_min', 'max_dc_diff', 'ac_max_abs',
                      'bit_depth', 'qp'}.
    """
    MIN_V = -(1 << (bit_depth - 1))
    MAX_V =  (1 << (bit_depth - 1)) - 1

    flat_max = [[MAX_V] * 8 for _ in range(8)]
    flat_min = [[MIN_V] * 8 for _ in range(8)]

    dct_max = fdct_2d(flat_max, bit_depth=bit_depth)
    dct_min = fdct_2d(flat_min, bit_depth=bit_depth)

    dc_max = quantise(dct_max, qp=qp, bit_depth=bit_depth)[0][0]
    dc_min = quantise(dct_min, qp=qp, bit_depth=bit_depth)[0][0]

    # Search the true worst-case AC coefficient magnitude by brute-force
    # testing every outer-product +MAX/-MAX sign pattern through the
    # REAL fdct_2d() -- simpler AND more accurate than deriving a
    # closed-form DCT basis function: every 8x8 checkerboard-family
    # pattern (parameterised by an 8-bit row/column sign vector) is fed
    # directly through the actual transform, so results come straight
    # from fdct_2d()'s own arithmetic with no separate formula to keep
    # in sync with it. This exhaustively covers all 2^8=256 outer-
    # product sign combinations (128 up to a global sign flip).
    import itertools as _itertools

    ac_max_abs = 0
    for bits in _itertools.product((0, 1), repeat=8):
        pat = [[MAX_V if (bits[r] + bits[c]) % 2 == 0 else MIN_V
                for c in range(8)] for r in range(8)]
        dct_p = fdct_2d(pat, bit_depth=bit_depth)
        q_p   = quantise(dct_p, qp=qp, bit_depth=bit_depth)
        ac_vals = [q_p[r][c] for r in range(8) for c in range(8)
                   if not (r == 0 and c == 0)]
        ac_max_abs = max(ac_max_abs, max(abs(val) for val in ac_vals))

    return {
        'bit_depth': bit_depth,
        'qp': qp,
        'dc_max': dc_max,
        'dc_min': dc_min,
        'max_dc_diff': dc_max - dc_min,
        'ac_max_abs': ac_max_abs,
    }


def block_saturated_dc(dc_value):
    """
    A block whose DC COEFFICIENT (post-DCT/quantisation domain, NOT a
    pixel value) sits at a specified value, with all AC coefficients
    zero (immediate EOB). dc_value is injected directly at the
    coefficient level. Callers MUST pass a value within the achievable
    range for their target bit depth (see get_achievable_coeff_bounds())
    -- this function itself does not clamp, since what counts as
    "in range" depends entirely on the caller's bit-depth constraint.
    """
    zz = [0] * 64
    zz[0] = dc_value
    return zz


def find_10bit_input_extreme_dc_blocks(qp=0):
    """
    Convenience wrapper: get_achievable_coeff_bounds(bit_depth=10, qp)
    plus the actual pixel blocks used, for backward compatibility with
    earlier callers of this specific function name.
    """
    MIN10, MAX10 = -512, 511
    flat_max = [[MAX10] * 8 for _ in range(8)]
    flat_min = [[MIN10] * 8 for _ in range(8)]
    bounds = get_achievable_coeff_bounds(bit_depth=10, qp=qp)
    return {
        'block_max_pixels': flat_max,
        'block_min_pixels': flat_min,
        'dc_max': bounds['dc_max'],
        'dc_min': bounds['dc_min'],
        'max_achievable_dc_diff': bounds['max_dc_diff'],
        'qp': qp,
    }


def build_10bit_input_max_dc_diff_frame(qp=0):
    """
    Build a REAL, minimal bitstream from two consecutive REAL 10-bit
    pixel blocks (all-max-pixel block, then all-min-pixel block, placed
    IMMEDIATELY ADJACENT with no filler in between), run through the
    actual fdct_2d() -> quantise() -> zigzag_scan() -> encode_block()
    pipeline, that achieves the LARGEST DC diff a genuinely
    10-bit-constrained encoder can ever produce.

    Returns (bitstream, lengths, symbols, info).
    """
    extremes = find_10bit_input_extreme_dc_blocks(qp=qp)

    prev_dc = 0
    kp_dc = 0
    all_symbols = []

    for pixel_block in (extremes['block_max_pixels'], extremes['block_min_pixels']):
        zz = zigzag_scan(quantise(fdct_2d(pixel_block, bit_depth=10),
                                   qp=qp, bit_depth=10))
        kp_run = 0
        kp_lvl = 0
        symbols, prev_dc, kp_dc, kp_run, kp_lvl = encode_block(
            zz, prev_dc, kp_dc, kp_run, kp_lvl)
        all_symbols.extend(symbols)

    bitstream = ''.join(sym[4] for sym in all_symbols)
    lengths   = [len(sym[4]) for sym in all_symbols]

    dc_symbols = [s for s in all_symbols if s[0] == DC]
    achieved_diffs = [s[1] for s in dc_symbols]

    info = {
        'qp': qp,
        'dc_max': extremes['dc_max'],
        'dc_min': extremes['dc_min'],
        'expected_max_diff': extremes['max_achievable_dc_diff'],
        'achieved_diffs': achieved_diffs,
        'dc_symbols': dc_symbols,
    }

    return bitstream, lengths, all_symbols, info


def block_max_run_before_level(coeff_value, run_length):
    """
    A block whose zigzag AC coefficients are all zero for run_length
    positions, followed by one nonzero coefficient. run_length may be
    0..63: at run_length=63, ALL 63 AC positions are zero (no LEVEL
    symbol is produced at all -- this becomes a pure EOB(63) block,
    which is the true boundary case at the o_run port's 6-bit ceiling
    with NO trailing coefficient, distinct from run_length=62 which
    still has a real coefficient at the very last position).
    coeff_value should be within the achievable range for the
    caller's target bit depth (see get_achievable_coeff_bounds());
    it is ignored when run_length=63.
    """
    zz = [0] * 64
    zz[0] = 0
    run_length = min(run_length, 63)
    if run_length < 63:
        zz[1 + run_length] = coeff_value
    # else: all 63 AC positions stay zero -> encode_block() emits EOB(63)
    return zz


def block_max_run_and_max_level(bit_depth=10, qp=0, run_length=62):
    """
    GAP 5: combine worst-case run length AND worst-case AC magnitude in
    the SAME symbol -- block_max_run_before_level() and block_dense_ac()
    each test these two pressures independently, but never together.
    This places the single nonzero AC coefficient at run_length
    positions in (default 62, the longest run that still has a
    trailing coefficient), with its magnitude set to the bit-depth's
    real achievable AC ceiling (via get_achievable_coeff_bounds()) --
    the longest possible RUN codeword immediately followed by the
    longest possible LEVEL codeword, back to back in one symbol pair.
    """
    bounds = get_achievable_coeff_bounds(bit_depth=bit_depth, qp=qp)
    return block_max_run_before_level(
        coeff_value=bounds['ac_max_abs'], run_length=run_length)


def build_zero_dc_diff_stress_frame(bit_depth=10, qp=0):
    """
    Build a REAL bitstream stressing the zero-DC-diff, no-sign-bit case:
    for each target diff in target_diffs, a "priming" block moves
    prev_dc by that diff (establishing a kp_dc value per
    clip_kp(abs_dc, rsh=1, hi=5) = min(5, abs_dc >> 1)), immediately
    followed by a block with the IDENTICAL DC coefficient (diff=0).

    Per encode_block()'s real behaviour, a diff=0 DC symbol omits the
    sign bit entirely (cw_dc += str(sign_dc) only when abs_dc != 0), so
    its codeword is PURE MAGNITUDE -- one bit shorter than a nonzero-
    diff codeword at the same kParam. A decoder that always expects a
    trailing sign bit after the DC magnitude field would silently
    misalign the shifter by exactly 1 bit on every zero-diff DC symbol.

    NOTE: when target_diff=0, the "priming" step is ALSO a zero-diff
    DC symbol in its own right (its diff from the previous block is
    exactly 0), so the returned info reports EVERY zero-diff DC symbol
    that actually occurs in the bitstream -- both from priming steps
    that happen to land on diff=0 AND from the deliberate same-DC
    follow-up step -- rather than assuming a fixed count.

    Returns (bitstream, lengths, symbols, info) where info contains
    the achieved kParam values and confirms EVERY zero-diff codeword's
    length matches the pure-magnitude expectation.
    """
    all_symbols = []
    prev_dc = 0
    kp_dc = 0

    # clip_kp(abs_dc, rsh=1, hi=5) = min(5, abs_dc >> 1):
    #   diff=0 -> kp=0   diff=3 -> kp=1   diff=5 -> kp=2
    #   diff=7 -> kp=3   diff=9 -> kp=4   diff=20 -> kp=5 (saturated)
    target_diffs = [0, 3, 5, 7, 9, 20]
    bounds = get_achievable_coeff_bounds(bit_depth=bit_depth, qp=qp)
    dc_min, dc_max = bounds['dc_min'], bounds['dc_max']

    for target_diff in target_diffs:
        new_dc = max(dc_min, min(dc_max, prev_dc + target_diff))

        # Priming block: establishes prev_dc = new_dc. NOTE: when
        # target_diff=0 this priming step is ITSELF a zero-diff DC
        # symbol -- that is a real, legitimate occurrence, not an
        # error, and is correctly counted below along with every
        # other zero-diff DC symbol in the stream.
        zz_prime = block_saturated_dc(new_dc)
        symbols, prev_dc, kp_dc, _, _ = encode_block(zz_prime, prev_dc, kp_dc, 0, 0)
        all_symbols.extend(symbols)

        # Zero-diff block: SAME dc value again -> diff=0, at the kp_dc
        # just established by the priming step above
        zz_same = block_saturated_dc(prev_dc)
        symbols2, prev_dc, kp_dc, _, _ = encode_block(zz_same, prev_dc, kp_dc, 0, 0)
        all_symbols.extend(symbols2)

    # Scan ALL symbols (priming AND follow-up alike) for every zero-diff
    # DC occurrence, rather than assuming only the follow-up steps
    # produce one -- this correctly captures the target_diff=0 case's
    # priming step too.
    zero_diff_kparams = []
    zero_diff_codeword_lens = []
    for sym_type, value, sign, kparam, codeword in all_symbols:
        if sym_type == DC and value == 0:
            zero_diff_kparams.append(kparam)
            zero_diff_codeword_lens.append(len(codeword))

    bitstream = ''.join(sym[4] for sym in all_symbols)
    lengths   = [len(sym[4]) for sym in all_symbols]

    info = {
        'bit_depth': bit_depth,
        'qp': qp,
        'zero_diff_kparams': zero_diff_kparams,
        'zero_diff_codeword_lens': zero_diff_codeword_lens,
        'expected_codeword_lens': [len(encode_hv(0, kp)) for kp in zero_diff_kparams],
    }

    return bitstream, lengths, all_symbols, info


def block_multi_extended_tier_stress(bit_depth=10, qp=0, num_extended=8):
    """
    GAP 2: multiple consecutive EXTENDED-tier codewords back-to-back
    within a single block, stressing the shifter's ability to refill
    mid-codeword across several long codewords in a row (as opposed to
    block_single_extended_tier_ac(), which places only ONE such
    codeword and surrounds it with zeros). Every num_extended-th AC
    position (evenly spaced across the 63 AC positions) is set to the
    bit-depth's achievable AC ceiling, guaranteeing an EXTENDED-tier
    h(v) codeword at each of those positions -- with ordinary small
    values in between so the codeword boundaries are distinguishable
    from each other during decode.
    """
    bounds = get_achievable_coeff_bounds(bit_depth=bit_depth, qp=qp)
    ac_ceiling = bounds['ac_max_abs']

    zz = [0] * 64
    zz[0] = 10   # unremarkable DC

    stride = max(1, 63 // num_extended)
    for pos in range(1, 64):
        if (pos - 1) % stride == 0:
            zz[pos] = ac_ceiling if (pos // stride) % 2 == 0 else -ac_ceiling
        else:
            zz[pos] = 3   # small ordinary value between EXTENDED-tier spikes

    return zz


def block_dense_ac(pattern='alternating', magnitude=40, bit_depth=10, qp=0):
    """
    A block with NO zero runs at all -- every one of the 63 AC positions
    is nonzero, forcing 63 consecutive (RUN=0, LEVEL) pairs in a single
    block. magnitude is used directly for 'alternating'; for 'ascending'
    it is clamped to the REAL achievable AC bound for bit_depth (via
    get_achievable_coeff_bounds()), not a hardcoded spec-ceiling value.

    pattern:
      'alternating' -- coefficient sign flips every position
      'ascending'   -- magnitude increases each position, clamped to
                        this bit depth's achievable AC ceiling
    """
    zz = [0] * 64
    zz[0] = 50   # arbitrary unremarkable DC

    if pattern == 'ascending':
        ac_ceiling = get_achievable_coeff_bounds(bit_depth=bit_depth, qp=qp)['ac_max_abs']
    else:
        ac_ceiling = None

    for pos in range(1, 64):
        if pattern == 'alternating':
            sign = 1 if pos % 2 == 0 else -1
            zz[pos] = sign * magnitude
        elif pattern == 'ascending':
            val = min(magnitude + pos * 6, ac_ceiling)
            zz[pos] = val
        else:
            raise ValueError(f"unknown pattern '{pattern}'")

    return zz


def block_all_zero():
    """
    A block with DC=0 and all 63 AC coefficients zero -- the minimum
    possible block, producing DC(0) + EOB(63) only.
    """
    return [0] * 64


def block_checkerboard_extreme(bit_depth=10, qp=0):
    """
    A block alternating between the two ACHIEVABLE coefficient extremes
    for the given bit_depth (via get_achievable_coeff_bounds()) at every
    AC position with no zero run at all, combined with a DC coefficient
    also at that bit depth's achievable extreme. This is the
    highest-energy, highest-symbol-density, most sign-alternating block
    achievable WITHIN your design's actual bit-depth constraint --
    never the spec's out-of-range structural ceiling.
    """
    bounds = get_achievable_coeff_bounds(bit_depth=bit_depth, qp=qp)
    dc_max, ac_max = bounds['dc_max'], bounds['ac_max_abs']

    zz = [0] * 64
    zz[0] = dc_max
    for pos in range(1, 64):
        zz[pos] = ac_max if pos % 2 == 1 else -ac_max
    return zz


def block_single_extended_tier_ac(position=32, value=400):
    """
    A block that is entirely zero except for ONE AC coefficient, placed
    mid-scan, with a magnitude large enough to guarantee an EXTENDED-tier
    h(v) codeword regardless of the running kp_lvl (kp_lvl resets to 0
    at the start of every block). Caller must choose `value` within the
    achievable range for their target bit depth.
    """
    zz = [0] * 64
    zz[0] = -30
    position = max(1, min(63, position))
    zz[position] = value
    return zz


def random_block(rng, bit_depth=10, qp=22):
    """An ordinary random 8x8 block's worth of DCT'd/quantised zigzag
    coefficients, used as filler between worst-case blocks so the
    worst-case blocks sit inside a realistic frame context rather than
    appearing in isolation."""
    MIN_V = -(1 << (bit_depth - 1))
    MAX_V =  (1 << (bit_depth - 1)) - 1
    pixel_rows = [[rng.randint(MIN_V, MAX_V) for _ in range(8)] for _ in range(8)]
    return zigzag_scan(quantise(fdct_2d(pixel_rows, bit_depth), qp, bit_depth))


def build_worst_case_frame(seed=0, filler_blocks_between=3, bit_depth=10, qp=0):
    """
    Assemble ONE continuous frame: an ordered list of 64-element zigzag
    coefficient blocks, mixing worst-case blocks with ordinary random
    filler blocks, then run the WHOLE SEQUENCE through encode_block()
    exactly once, carrying kp_dc across the entire frame and resetting
    kp_run/kp_lvl at each block boundary.

    ALL worst-case bounds are derived from get_achievable_coeff_bounds()
    for the given bit_depth -- never a hardcoded spec-ceiling constant --
    so the generated stimulus never exceeds what a bit_depth-constrained
    design is actually required to support.

    The two saturated-DC blocks (max then min) are placed IMMEDIATELY
    ADJACENT (no filler between them) so the true maximum achievable DC
    diff for this bit depth is deterministically reached, rather than
    depending on whatever a random filler block's DC happens to leave
    prev_dc at.

    Returns (bitstream, lengths, symbols, block_boundaries).
    """
    rng = random.Random(seed)
    bounds = get_achievable_coeff_bounds(bit_depth=bit_depth, qp=qp)

    blocks = []
    blocks.append(block_all_zero())
    blocks.append(random_block(rng, bit_depth=bit_depth))
    blocks.append(block_saturated_dc(bounds['dc_max']))
    blocks.append(block_saturated_dc(bounds['dc_min']))   # adjacent, no filler
    blocks.append(random_block(rng, bit_depth=bit_depth))
    blocks.append(block_max_run_before_level(
        coeff_value=bounds['ac_max_abs'], run_length=62))
    blocks.append(block_max_run_before_level(
        coeff_value=bounds['ac_max_abs'], run_length=63))   # GAP 1: pure EOB(63)
    blocks.append(random_block(rng, bit_depth=bit_depth))
    blocks.append(block_max_run_and_max_level(
        bit_depth=bit_depth, qp=qp, run_length=62))          # GAP 5
    blocks.append(random_block(rng, bit_depth=bit_depth))
    blocks.append(block_multi_extended_tier_stress(
        bit_depth=bit_depth, qp=qp, num_extended=8))          # GAP 2
    blocks.append(random_block(rng, bit_depth=bit_depth))
    blocks.append(block_dense_ac(pattern='alternating',
                                  magnitude=min(45, bounds['ac_max_abs']),
                                  bit_depth=bit_depth, qp=qp))
    blocks.append(random_block(rng, bit_depth=bit_depth))
    blocks.append(block_dense_ac(pattern='ascending', magnitude=5,
                                  bit_depth=bit_depth, qp=qp))
    blocks.append(random_block(rng, bit_depth=bit_depth))
    blocks.append(block_single_extended_tier_ac(
        position=1,  value=min(480, bounds['ac_max_abs'])))
    blocks.append(block_single_extended_tier_ac(
        position=63, value=min(480, bounds['ac_max_abs'])))
    blocks.append(random_block(rng, bit_depth=bit_depth))
    blocks.append(block_checkerboard_extreme(bit_depth=bit_depth, qp=qp))
    blocks.append(random_block(rng, bit_depth=bit_depth))
    blocks.append(block_all_zero())
    blocks.append(block_checkerboard_extreme(bit_depth=bit_depth, qp=qp))
    blocks.append(random_block(rng, bit_depth=bit_depth))

    padded_blocks = []
    for blk in blocks:
        padded_blocks.append(blk)
        for _ in range(filler_blocks_between):
            padded_blocks.append(random_block(rng, bit_depth=bit_depth))

    all_symbols     = []
    block_boundaries = []
    prev_dc = 0
    kp_dc   = 0

    for blk in padded_blocks:
        block_boundaries.append(len(all_symbols))
        kp_run = 0
        kp_lvl = 0

        syms, prev_dc, kp_dc, kp_run, kp_lvl = encode_block(
            blk, prev_dc, kp_dc, kp_run, kp_lvl)
        all_symbols.extend(syms)

    bitstream = ''.join(sym[4] for sym in all_symbols)
    lengths   = [len(sym[4]) for sym in all_symbols]

    return bitstream, lengths, all_symbols, block_boundaries


def build_worst_case_frame_at_geometry(geometry, plane='luma', seed=0,
                                        worst_case_period=200):
    """
    Build a worst-case bitstream sized to the REAL block count of a given
    frame geometry (from get_frame_geometry() / get_named_frame_geometry()),
    so the worst-case content is exercised at the SAME SCALE the decoder
    must sustain to meet the geometry's rated sample rate -- not just a
    fixed handful of blocks disconnected from any real frame size.

    A worst-case block is inserted every `worst_case_period` blocks,
    cycling through the full worst-case block catalogue (saturated DC,
    dense AC, checkerboard, extended-tier single AC, max run, all-zero),
    with ordinary random blocks filling every position in between. This
    means:
      - the TOTAL block count matches exactly one real frame at the
        target resolution (e.g. 129,600 blocks for UHD 4K luma), so the
        decoder is being asked to sustain its rated throughput across
        the worst-case stream, not just survive a short burst of it
      - worst-case blocks are spread evenly across the frame rather than
        clustered at the start, so timing/throughput analysis covers the
        whole frame duration, not just the first few microseconds

    Parameters:
      geometry           : dict from get_frame_geometry() /
                            get_named_frame_geometry()
      plane               : 'luma' or 'chroma' -- selects which plane's
                             block count defines the frame size
      worst_case_period   : insert one worst-case block every this many
                             blocks (lower = more worst-case pressure,
                             higher = more realistic average-case mix)

    Returns a dict with:
      'bitstream'        : full concatenated bit string
      'lengths'           : per-symbol codeword length list
      'symbols'           : full (type, value, sign, kparam, codeword) list
      'block_boundaries'  : symbol index where each block begins
      'total_blocks'      : block count actually generated (== the
                             geometry's full-frame block count for this
                             plane)
      'worst_case_block_indices' : which block indices (0-based) are
                             worst-case rather than random filler
      'required_frame_time_s'    : 1/fps -- the time budget the decoder
                             must finish this whole bitstream within, to
                             sustain the geometry's rated frame rate
      'bits_per_second_required' : len(bitstream) / required_frame_time_s
                             -- the sustained bit-consumption rate the
                             shifter/decoder pipeline must support to
                             keep up with this geometry in real time
    """
    full_block_count = (geometry['luma_blocks_per_frame'] if plane == 'luma'
                         else geometry['chroma_blocks_per_frame'])

    rng = random.Random(seed)
    bit_depth = geometry['bit_depth']
    bounds = get_achievable_coeff_bounds(bit_depth=bit_depth, qp=0)

    worst_case_catalogue = [
        lambda: block_saturated_dc(bounds['dc_max']),
        lambda: block_saturated_dc(bounds['dc_min']),
        lambda: block_checkerboard_extreme(bit_depth=bit_depth),
        lambda: block_dense_ac(pattern='alternating',
                                magnitude=min(45, bounds['ac_max_abs']),
                                bit_depth=bit_depth),
        lambda: block_dense_ac(pattern='ascending', magnitude=5,
                                bit_depth=bit_depth),
        lambda: block_max_run_before_level(
            coeff_value=min(250, bounds['ac_max_abs']), run_length=62),
        lambda: block_single_extended_tier_ac(
            position=1,  value=min(480, bounds['ac_max_abs'])),
        lambda: block_single_extended_tier_ac(
            position=63, value=min(480, bounds['ac_max_abs'])),
        lambda: block_all_zero(),
    ]

    all_symbols            = []
    block_boundaries        = []
    worst_case_block_indices = []
    prev_dc = 0
    kp_dc   = 0

    for block_idx in range(full_block_count):
        block_boundaries.append(len(all_symbols))

        if block_idx % worst_case_period == 0:
            gen_fn = worst_case_catalogue[
                (block_idx // worst_case_period) % len(worst_case_catalogue)]
            blk = gen_fn()
            worst_case_block_indices.append(block_idx)
        else:
            blk = random_block(rng, bit_depth=bit_depth)

        kp_run = 0
        kp_lvl = 0
        syms, prev_dc, kp_dc, kp_run, kp_lvl = encode_block(
            blk, prev_dc, kp_dc, kp_run, kp_lvl)
        all_symbols.extend(syms)

    bitstream = ''.join(sym[4] for sym in all_symbols)
    lengths   = [len(sym[4]) for sym in all_symbols]

    fps = geometry['fps']
    required_frame_time_s = 1.0 / fps
    bits_per_second_required = len(bitstream) / required_frame_time_s

    return {
        'bitstream': bitstream,
        'lengths': lengths,
        'symbols': all_symbols,
        'block_boundaries': block_boundaries,
        'total_blocks': full_block_count,
        'worst_case_block_indices': worst_case_block_indices,
        'required_frame_time_s': required_frame_time_s,
        'bits_per_second_required': bits_per_second_required,
    }


def compute_engines_needed(geometry, cycles_per_block, clock_freq,
                            include_chroma=True):
    """
    Compute the number of parallel decoder engines required to sustain
    a given frame geometry in real time, GIVEN a decoder architecture
    that takes `cycles_per_block` clock cycles to fully decode one
    8x8 (64-sample) block.

    IMPORTANT: by default this includes BOTH luma AND chroma block
    counts (Y + Cb + Cr), since all three planes must actually be
    decoded to reconstruct a frame -- the spec's MaxLumaSr limit only
    constrains LUMA for conformance-checking purposes, but a real
    decoder implementation still has to do the chroma work too. Set
    include_chroma=False only if you specifically want the (spec-only,
    non-physical) luma-alone workload.

    For 4:2:2, chroma work equals luma work exactly (two half-width
    planes = one full-width-equivalent plane), so omitting chroma
    silently halves the true required engine count.

    Returns a dict with the full breakdown so intermediate values are
    visible: block counts, total cycles per frame, total cycles/second
    needed, and the final engine count (ceil'd to a whole number).
    """
    if include_chroma:
        blocks_per_frame = geometry['total_blocks_per_frame']   # Y + Cb + Cr
    else:
        blocks_per_frame = geometry['luma_blocks_per_frame']    # luma only

    fps = geometry['fps']
    total_cycles_per_frame  = blocks_per_frame * cycles_per_block
    total_cycles_per_second = total_cycles_per_frame * fps
    engines_needed = math.ceil(total_cycles_per_second / clock_freq)

    return {
        'blocks_per_frame': blocks_per_frame,
        'include_chroma': include_chroma,
        'cycles_per_block': cycles_per_block,
        'total_cycles_per_frame': total_cycles_per_frame,
        'total_cycles_per_second': total_cycles_per_second,
        'clock_freq': clock_freq,
        'engines_needed': engines_needed,
    }

