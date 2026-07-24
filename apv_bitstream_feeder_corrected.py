#!/usr/bin/env python3
"""
apv_bitstream_feeder.py
-----------------------
Parse a raw APV (.apv) file (RFC 9924 / Appendix A raw bitstream format),
extract a specific tile's entropy-coded component data, and emit it as
32-bit big-endian words for FPGA simulation.

Raw .apv file layout (RFC 9924 Appendix A):
  [au_size        : u32]   size in bytes of the access_unit() that follows
  access_unit():
    [signature     : f32]  0x61507631 ('aPv1')
    do {
      [pbu_size    : u32]  size in bytes of the pbu() that follows
      pbu():
        pbu_header():
          [pbu_type          : u8 ]
          [group_id          : u16]
          [reserved_zero_8bits: u8 ]
        frame():                      <- when pbu_type in {1,2,25,26,27}
          frame_header():
            frame_info():
              [profile_idc           : u8 ]
              [level_idc             : u8 ]
              [band_idc              : u3 ]
              [reserved_zero_5bits   : u5 ]
              [frame_width           : u24]   <- luma pixels
              [frame_height          : u24]   <- luma pixels
              [chroma_format_idc     : u4 ]
              [bit_depth_minus8      : u4 ]
              [capture_time_distance : u8 ]
              [reserved_zero_8bits   : u8 ]
            [reserved_zero_8bits     : u8 ]
            [color_description_present_flag: u1]
            if color_description_present_flag:
              [color_primaries        : u8]
              [transfer_characteristics: u8]
              [matrix_coefficients    : u8]
              [full_range_flag        : u1]
            [use_q_matrix            : u1]
            if use_q_matrix:
              quantization_matrix()   <- NumComps * 64 bytes
            tile_info():
              [tile_width_in_mbs     : u20]
              [tile_height_in_mbs    : u20]
              ... derives NumTiles ...
              [tile_size_present_in_fh_flag: u1]
              if tile_size_present_in_fh_flag:
                [tile_size_in_fh[i]  : u32] * NumTiles
            [reserved_zero_8bits     : u8 ]
            byte_alignment()
          for i in 0..NumTiles-1:
            [tile_size[i]            : u32]   <- bytes of tile(i)
            tile(i):
              tile_header(i):
                [tile_header_size    : u16]   <- size of tile_header in bytes
                [tile_index          : u16]
                for c in 0..NumComps-1:
                  [tile_data_size[c] : u32]
                for c in 0..NumComps-1:
                  [tile_qp[c]        : u8 ]
                [reserved_zero_8bits : u8 ]
                byte_alignment()
              for c in 0..NumComps-1:
                tile_data(i,c)        <- tile_data_size[c] bytes of VLC data

NumComps is derived from chroma_format_idc (RFC 9924 Table 2):
  0 -> 1  (4:0:0)
  2 -> 3  (4:2:2)
  3 -> 3  (4:4:4)
  4 -> 4  (4:4:4:4)

Usage:
  python apv_bitstream_feeder.py sample.apv
  python apv_bitstream_feeder.py sample.apv --au 0 --frame-pbu 0 --tile 0 --comp 0
  python apv_bitstream_feeder.py sample.apv --cocotb > tile0_vectors.py
"""

import struct
import sys
import math
import argparse

# ---------------------------------------------------------------------------
# chroma_format_idc -> NumComps  (RFC 9924 Table 2)
# ---------------------------------------------------------------------------
CHROMA_FMT_TO_NUM_COMPS = {0: 1, 2: 3, 3: 3, 4: 4}


# ---------------------------------------------------------------------------
# Bit reader
# ---------------------------------------------------------------------------

class BitReader:
    """Read individual bits and multi-bit fields from a bytes buffer."""

    def __init__(self, data: bytes, start_byte: int = 0):
        self._data     = data
        self._byte_pos = start_byte
        self._bit_pos  = 0   # 0 = MSB of current byte

    # ---- position helpers ------------------------------------------------

    @property
    def byte_pos(self) -> int:
        """Current byte position (next whole byte boundary >= consumed bits)."""
        return self._byte_pos + (1 if self._bit_pos > 0 else 0)

    @property
    def bit_offset(self) -> int:
        """Total bits consumed so far."""
        return self._byte_pos * 8 + self._bit_pos

    def is_byte_aligned(self) -> bool:
        return self._bit_pos == 0

    # ---- core read -------------------------------------------------------

    def read_bits(self, n: int) -> int:
        """Read n bits MSB-first and return as unsigned integer."""
        if n == 0:
            return 0
        avail = (len(self._data) - self._byte_pos) * 8 - self._bit_pos
        if avail < n:
            raise EOFError(
                f"Need {n} bits but only {avail} available "
                f"(byte={self._byte_pos}, bit={self._bit_pos})"
            )
        result    = 0
        remaining = n
        while remaining > 0:
            bits_in_byte = 8 - self._bit_pos
            take         = min(bits_in_byte, remaining)
            shift        = bits_in_byte - take
            mask         = (1 << take) - 1
            bits         = (self._data[self._byte_pos] >> shift) & mask
            result       = (result << take) | bits
            self._bit_pos += take
            remaining    -= take
            if self._bit_pos == 8:
                self._byte_pos += 1
                self._bit_pos   = 0
        return result

    # ---- convenience wrappers --------------------------------------------

    def u(self, n: int) -> int:
        return self.read_bits(n)

    def u8(self)  -> int: return self.read_bits(8)
    def u16(self) -> int: return self.read_bits(16)
    def u32(self) -> int: return self.read_bits(32)
    def u24(self) -> int: return self.read_bits(24)

    # ---- alignment -------------------------------------------------------

    def byte_align(self):
        """Consume padding bits until the reader is byte-aligned."""
        if self._bit_pos != 0:
            self._byte_pos += 1
            self._bit_pos   = 0

    def byte_align_zero(self):
        """Consume byte-alignment bits and require every padding bit to be zero."""
        if self._bit_pos:
            count = 8 - self._bit_pos
            if self.read_bits(count) != 0:
                raise ValueError("Non-zero APV byte-alignment padding")

    def skip_bytes(self, n: int):
        """Skip n bytes (must be called when byte-aligned)."""
        self.byte_align()
        if n < 0 or self._byte_pos + n > len(self._data):
            raise EOFError(f"Cannot skip {n} bytes at offset {self._byte_pos}")
        self._byte_pos += n

    def read_bytes(self, n: int) -> bytes:
        """Read n raw bytes (must be called when byte-aligned)."""
        self.byte_align()
        if n < 0 or self._byte_pos + n > len(self._data):
            raise EOFError(f"Cannot read {n} bytes at offset {self._byte_pos}")
        start           = self._byte_pos
        self._byte_pos += n
        return self._data[start : self._byte_pos]

    def peek_bytes(self, n: int) -> bytes:
        """Peek at the next n bytes without advancing."""
        self.byte_align()
        return self._data[self._byte_pos : self._byte_pos + n]


# ---------------------------------------------------------------------------
# frame_info()  -- RFC 9924 Section 5.3.6
# ---------------------------------------------------------------------------

def parse_frame_info(br: BitReader) -> dict:
    fi = {}
    fi['profile_idc']           = br.u8()
    fi['level_idc']             = br.u8()
    fi['band_idc']              = br.u(3)
    fi['reserved_zero_5bits']   = br.u(5)
    fi['frame_width']           = br.u24()   # luma pixels
    fi['frame_height']          = br.u24()   # luma pixels
    fi['chroma_format_idc']     = br.u(4)
    fi['bit_depth_minus8']      = br.u(4)
    fi['capture_time_distance'] = br.u8()
    fi['reserved_zero_8bits']   = br.u8()

    fi['BitDepth']  = fi['bit_depth_minus8'] + 8
    if fi['chroma_format_idc'] not in CHROMA_FMT_TO_NUM_COMPS:
        raise ValueError(
            f"Reserved chroma_format_idc={fi['chroma_format_idc']}"
        )
    fi['NumComps'] = CHROMA_FMT_TO_NUM_COMPS[fi['chroma_format_idc']]

    if fi['reserved_zero_5bits'] != 0 or fi['reserved_zero_8bits'] != 0:
        raise ValueError("Non-zero reserved field in frame_info()")
    if fi['frame_width'] == 0 or fi['frame_height'] == 0:
        raise ValueError("APV frame dimensions must be non-zero")

    # Derived frame geometry (RFC 9924 Section 5.3.6)
    MbWidth  = 16
    MbHeight = 16
    fi['FrameWidthInMbsY']  = math.ceil(fi['frame_width']  / MbWidth)
    fi['FrameHeightInMbsY'] = math.ceil(fi['frame_height'] / MbHeight)

    return fi


# ---------------------------------------------------------------------------
# quantization_matrix()  -- RFC 9924 Section 5.3.7
# ---------------------------------------------------------------------------

def skip_quantization_matrix(br: BitReader, num_comps: int):
    """Each component has an 8x8 matrix of u8 entries = 64 bytes = 512 bits.
    These are read as contiguous u8 bitfield entries with NO byte-alignment
    padding before them (they follow directly after the use_q_matrix flag bit
    inside the same bitfield).  Must use read_bits, not skip_bytes."""
    br.read_bits(num_comps * 64 * 8)


# ---------------------------------------------------------------------------
# tile_info()  -- RFC 9924 Section 5.3.8
# ---------------------------------------------------------------------------

def parse_tile_info(br: BitReader, fi: dict) -> dict:
    ti = {}
    ti['tile_width_in_mbs']  = br.u(20)
    ti['tile_height_in_mbs'] = br.u(20)

    # Derive tile grid (same loop logic as spec pseudocode)
    FW = fi['FrameWidthInMbsY']
    FH = fi['FrameHeightInMbsY']
    tw = ti['tile_width_in_mbs']
    th = ti['tile_height_in_mbs']
    if tw == 0 or th == 0:
        raise ValueError("APV tile dimensions must be non-zero")

    tile_cols = math.ceil(FW / tw)
    tile_rows = math.ceil(FH / th)

    ti['TileCols'] = tile_cols
    ti['TileRows'] = tile_rows
    ti['NumTiles'] = tile_cols * tile_rows

    # Optional per-frame tile_size array
    ti['tile_size_present_in_fh_flag'] = br.u(1)
    ti['tile_size_in_fh'] = []
    if ti['tile_size_present_in_fh_flag']:
        for _ in range(ti['NumTiles']):
            ti['tile_size_in_fh'].append(br.u32())

    return ti


# ---------------------------------------------------------------------------
# frame_header()  -- RFC 9924 Section 5.3.5
# ---------------------------------------------------------------------------

def parse_frame_header(br: BitReader) -> tuple[dict, dict]:
    """
    Returns (frame_info dict, tile_info dict).
    Leaves br positioned at the first tile_size field.
    """
    fi = parse_frame_info(br)

    # Fields that follow frame_info() inside frame_header()
    if br.u8() != 0:
        raise ValueError("Non-zero reserved field in frame_header()")
    color_flag = br.u(1)
    fi['color_description_present_flag'] = color_flag
    fi['color_primaries']         = 0
    fi['transfer_characteristics']= 0
    fi['matrix_coefficients']     = 0
    fi['full_range_flag']         = 0
    if color_flag:
        fi['color_primaries']          = br.u8()
        fi['transfer_characteristics'] = br.u8()
        fi['matrix_coefficients']      = br.u8()
        fi['full_range_flag']          = br.u(1)

    use_q = br.u(1)
    fi['use_q_matrix'] = use_q
    if use_q:
        skip_quantization_matrix(br, fi['NumComps'])

    ti = parse_tile_info(br, fi)

    if br.u8() != 0:
        raise ValueError("Non-zero trailing reserved field in frame_header()")
    br.byte_align_zero()

    return fi, ti


# ---------------------------------------------------------------------------
# tile_header()  -- RFC 9924 Section 5.3.13
# ---------------------------------------------------------------------------

def parse_tile_header(br: BitReader, num_comps: int, expected_index: int) -> dict:
    """
    tile_header():
      tile_header_size  u16
      tile_index        u16
      tile_data_size[i] u32  x NumComps
      tile_qp[i]        u8   x NumComps
      reserved_zero_8bits u8
      byte_alignment()
    """
    th = {}
    th['tile_header_size'] = br.u16()   # size of tile_header in bytes
    th['tile_index']       = br.u16()
    th['tile_data_size']   = [br.u32() for _ in range(num_comps)]
    th['tile_qp']          = [br.u8()  for _ in range(num_comps)]
    if th['tile_index'] != expected_index:
        raise ValueError(
            f"tile_index={th['tile_index']} does not match expected "
            f"index {expected_index}"
        )
    expected_header_size = 5 + 5 * num_comps
    if th['tile_header_size'] != expected_header_size:
        raise ValueError(
            f"tile_header_size={th['tile_header_size']}, expected "
            f"{expected_header_size} for {num_comps} components"
        )
    if any(size == 0 for size in th['tile_data_size']):
        raise ValueError("tile_data_size value 0 is reserved")
    if br.u8() != 0:
        raise ValueError("Non-zero reserved field in tile_header()")
    br.byte_align_zero()
    return th


# ---------------------------------------------------------------------------
# Main parse entry point
# ---------------------------------------------------------------------------

def find_frame_pbu(raw: bytes, target_au: int, target_frame_pbu: int):
    """Return framing information for the selected frame PBU."""
    if target_au < 0 or target_frame_pbu < 0:
        raise ValueError("AU and frame-PBU indices must be non-negative")

    file_pos = 0
    au_index = 0
    while file_pos < len(raw):
        if file_pos + 4 > len(raw):
            raise ValueError("Truncated au_size field")
        au_size = struct.unpack_from('>I', raw, file_pos)[0]
        if au_size == 0 or au_size == 0xFFFFFFFF:
            raise ValueError(f"Invalid au_size={au_size} at offset {file_pos}")
        au_start = file_pos + 4
        au_end = au_start + au_size
        if au_end > len(raw):
            raise ValueError(
                f"AU {au_index} extends past EOF: end={au_end}, size={len(raw)}"
            )

        if au_index == target_au:
            if au_size < 4:
                raise ValueError(f"AU {au_index} is too short")
            signature = struct.unpack_from('>I', raw, au_start)[0]
            if signature != 0x61507631:
                raise ValueError(
                    f"Bad APV signature in AU {au_index}: 0x{signature:08X}"
                )

            pbu_field_pos = au_start + 4
            frame_index = 0
            while pbu_field_pos < au_end:
                if pbu_field_pos + 4 > au_end:
                    raise ValueError(f"Truncated pbu_size in AU {au_index}")
                pbu_size = struct.unpack_from('>I', raw, pbu_field_pos)[0]
                if pbu_size == 0 or pbu_size == 0xFFFFFFFF:
                    raise ValueError(
                        f"Invalid pbu_size={pbu_size} at offset {pbu_field_pos}"
                    )
                pbu_start = pbu_field_pos + 4
                pbu_end = pbu_start + pbu_size
                if pbu_end > au_end:
                    raise ValueError(
                        f"PBU at {pbu_field_pos} extends beyond AU {au_index}"
                    )
                if pbu_size < 4:
                    raise ValueError(f"PBU at {pbu_field_pos} is too short")

                pbu_type = raw[pbu_start]
                if pbu_type in (1, 2, 25, 26, 27):
                    if frame_index == target_frame_pbu:
                        group_id = struct.unpack_from('>H', raw, pbu_start + 1)[0]
                        reserved = raw[pbu_start + 3]
                        if reserved != 0:
                            raise ValueError(
                                "Non-zero reserved field in pbu_header()"
                            )
                        return {
                            'au_index': au_index,
                            'au_size': au_size,
                            'au_start': au_start,
                            'au_end': au_end,
                            'pbu_index': frame_index,
                            'pbu_size': pbu_size,
                            'pbu_size_field': pbu_field_pos,
                            'pbu_start': pbu_start,
                            'pbu_end': pbu_end,
                            'pbu_type': pbu_type,
                            'group_id': group_id,
                            'frame_header': pbu_start + 4,
                        }
                    frame_index += 1
                pbu_field_pos = pbu_end

            raise ValueError(
                f"AU {target_au} has only {frame_index} frame PBU(s); "
                f"requested frame PBU {target_frame_pbu}"
            )

        file_pos = au_end
        au_index += 1

    raise ValueError(f"Requested AU {target_au}, but file has {au_index} AU(s)")


def parse_apv_file(
    path: str,
    target_tile: int = 0,
    target_au: int = 0,
    target_frame_pbu: int = 0,
) -> dict:
    """
    Parse a raw .apv file (RFC 9924 Appendix A format) and return info
    about the requested tile.

    Returns dict:
      frame_info       : dict  (from frame_header)
      tile_info        : dict  (from tile_info)
      tile_header      : dict  (from tile_header of target tile)
      tile_comp_data   : list[bytes]  per-component entropy-coded bytes
      tile_size_field  : int   value of tile_size[i] field in frame()
      offsets          : dict  byte offsets of key structures in the file
    """
    with open(path, 'rb') as f:
        raw = f.read()

    if len(raw) < 12:
        raise ValueError("File too short to be a valid raw APV bitstream")
    if target_tile < 0:
        raise ValueError("Tile index must be non-negative")

    framing = find_frame_pbu(raw, target_au, target_frame_pbu)
    br = BitReader(raw, framing['frame_header'])
    offsets = {
        'au_size': framing['au_start'] - 4,
        'access_unit': framing['au_start'],
        'signature': framing['au_start'],
        'pbu_size': framing['pbu_size_field'],
        'pbu_header': framing['pbu_start'],
        'frame_header': framing['frame_header'],
    }

    # ------------------------------------------------------------------
    # frame_header()
    # ------------------------------------------------------------------
    fi, ti = parse_frame_header(br)

    num_comps = fi['NumComps']
    num_tiles = ti['NumTiles']

    if target_tile >= num_tiles:
        raise ValueError(
            f"Requested tile {target_tile} but frame has only "
            f"{num_tiles} tiles ({ti['TileCols']}x{ti['TileRows']})"
        )

    offsets['first_tile'] = br._byte_pos

    # ------------------------------------------------------------------
    # Walk tile_size[i] + tile(i) to reach the target tile
    # ------------------------------------------------------------------
    for tile_idx in range(num_tiles):
        tile_size_offset = br._byte_pos
        tile_size_val    = br.u32()

        tile_data_start  = br._byte_pos   # start of tile() payload
        tile_data_end    = tile_data_start + tile_size_val

        if tile_size_val == 0:
            raise ValueError(f"Tile {tile_idx}: tile_size value 0 is reserved")
        if tile_data_end > framing['pbu_end']:
            raise ValueError(
                f"Tile {tile_idx}: tile_size={tile_size_val} extends "
                f"past the selected PBU"
            )

        if tile_idx == target_tile:
            offsets['tile_size_field'] = tile_size_offset
            offsets['tile_data_start'] = tile_data_start

            # Parse tile_header
            th = parse_tile_header(br, num_comps, tile_idx)
            tile_header_end = br._byte_pos

            # Extract per-component entropy-coded data
            comp_data = []
            for c in range(num_comps):
                sz = th['tile_data_size'][c]
                comp_data.append(br.read_bytes(sz))
            if br._byte_pos > tile_data_end:
                raise ValueError(
                    f"Tile {tile_idx} component data exceeds tile_size"
                )
            if ti['tile_size_present_in_fh_flag']:
                header_size = ti['tile_size_in_fh'][tile_idx]
                if header_size != tile_size_val:
                    raise ValueError(
                        f"tile_size_in_fh[{tile_idx}]={header_size} does not "
                        f"match tile_size[{tile_idx}]={tile_size_val}"
                    )

            return {
                'frame_info':      fi,
                'tile_info':       ti,
                'tile_header':     th,
                'tile_comp_data':  comp_data,
                'tile_header_data': raw[tile_data_start:tile_header_end],
                'tile_dummy_data': raw[br._byte_pos:tile_data_end],
                'tile_payload': raw[tile_data_start:tile_data_end],
                'tile_size_field': tile_size_val,
                'offsets':         offsets,
                'pbu_type':        framing['pbu_type'],
                'group_id':        framing['group_id'],
                'pbu_size':        framing['pbu_size'],
                'au_size':         framing['au_size'],
                'au_index':        target_au,
                'frame_pbu_index': target_frame_pbu,
            }

        # Skip to next tile
        br._byte_pos = tile_data_end
        br._bit_pos  = 0

    raise RuntimeError(f"Tile {target_tile} not found after walking {num_tiles} tiles")


# ---------------------------------------------------------------------------
# 32-bit word packer
# ---------------------------------------------------------------------------

def bytes_to_words32(data: bytes, pad: bool = True) -> list[int]:
    """Pack bytes into 32-bit big-endian words, zero-padding the last word."""
    if pad:
        rem = len(data) % 4
        if rem:
            data = data + b'\x00' * (4 - rem)
    return [struct.unpack_from('>I', data, i)[0] for i in range(0, len(data), 4)]


# ---------------------------------------------------------------------------
# Optional entropy/EOB validator
# ---------------------------------------------------------------------------

def clip(lo: int, hi: int, value: int) -> int:
    return max(lo, min(hi, value))


def read_hv(br: BitReader, k_param: int) -> int:
    """Parse the APV h(v) symbolValue process from RFC 9924 Section 7.1.4."""
    symbol_value = 0
    k = k_param

    if br.read_bits(1) == 1:
        parse_exp_golomb = False
    elif br.read_bits(1) == 0:
        symbol_value += 1 << k
        parse_exp_golomb = False
    else:
        symbol_value += 2 << k
        parse_exp_golomb = True

    if parse_exp_golomb:
        while br.read_bits(1) == 0:
            symbol_value += 1 << k
            k += 1

    if k > 0:
        symbol_value += br.read_bits(k)
    return symbol_value


def tile_geometry(fi: dict, ti: dict, tile_index: int) -> tuple[int, int]:
    """Return (MB columns, MB rows) for a possibly truncated boundary tile."""
    tile_col = tile_index % ti['TileCols']
    tile_row = tile_index // ti['TileCols']
    mb_x0 = tile_col * ti['tile_width_in_mbs']
    mb_y0 = tile_row * ti['tile_height_in_mbs']
    mb_cols = min(
        ti['tile_width_in_mbs'],
        fi['FrameWidthInMbsY'] - mb_x0,
    )
    mb_rows = min(
        ti['tile_height_in_mbs'],
        fi['FrameHeightInMbsY'] - mb_y0,
    )
    return mb_cols, mb_rows


def blocks_per_mb(fi: dict, component: int) -> int:
    if component == 0:
        return 4
    if fi['chroma_format_idc'] == 2:  # 4:2:2: 8x16 chroma MB
        return 2
    return 4  # 4:4:4 and alpha: 16x16 component MB


def validate_component_entropy(
    data: bytes,
    num_blocks: int,
    component: int,
) -> dict:
    """
    Parse one tile_data() component and validate every 8x8 block boundary.

    A trailing zero run is accepted only when it reaches scan position 64
    exactly. A run that exceeds the remaining positions raises ValueError.
    """
    br = BitReader(data)
    prev_dc_diff = 20
    prev_first_ac_level = 0
    terminal_runs = 0
    blocks_ending_nonzero = 0
    nonzero_ac_coeffs = 0

    for block in range(num_blocks):
        dc_k = clip(0, 5, prev_dc_diff >> 1)
        abs_dc_diff = read_hv(br, dc_k)
        if abs_dc_diff:
            br.read_bits(1)  # sign_dc_coeff_diff
        prev_dc_diff = abs_dc_diff

        scan_pos = 1
        prev_run = 0
        prev_level = prev_first_ac_level
        first_ac = True
        ended_by_terminal_run = False

        while scan_pos < 64:
            run_k = clip(0, 2, prev_run >> 2)
            run = read_hv(br, run_k)
            remaining = 64 - scan_pos
            if run > remaining:
                raise ValueError(
                    f"component {component}, block {block}: zero run {run} "
                    f"exceeds {remaining} remaining positions "
                    f"(scan_pos={scan_pos})"
                )

            scan_pos += run
            if scan_pos == 64:
                terminal_runs += 1
                ended_by_terminal_run = True
                break

            prev_run = run
            level_k = clip(0, 4, prev_level >> 2)
            abs_ac_coeff_minus1 = read_hv(br, level_k)
            level = abs_ac_coeff_minus1 + 1
            if level > 32767:
                raise ValueError(
                    f"component {component}, block {block}: AC level "
                    f"{level} exceeds 32767"
                )
            br.read_bits(1)  # sign_ac_coeff
            scan_pos += 1
            nonzero_ac_coeffs += 1
            prev_level = level
            if first_ac:
                first_ac = False
                prev_first_ac_level = level

        if scan_pos != 64:
            raise ValueError(
                f"component {component}, block {block}: ended at "
                f"scan_pos={scan_pos}, expected 64"
            )
        # When coefficient 63 is non-zero, the block reaches 64 without a
        # separate trailing zero-run codeword.
        if not ended_by_terminal_run:
            blocks_ending_nonzero += 1

    padding_bits = len(data) * 8 - br.bit_offset
    if padding_bits < 0 or padding_bits > 7:
        raise ValueError(
            f"component {component}: {padding_bits} unconsumed bits after "
            f"{num_blocks} blocks"
        )
    if padding_bits and br.read_bits(padding_bits) != 0:
        raise ValueError(
            f"component {component}: non-zero tile_data alignment bits"
        )

    return {
        'blocks': num_blocks,
        'terminal_zero_runs': terminal_runs,
        'blocks_ending_nonzero': blocks_ending_nonzero,
        'nonzero_ac_coeffs': nonzero_ac_coeffs,
        'padding_bits': padding_bits,
    }


def validate_tile_entropy(result: dict) -> list[dict]:
    fi = result['frame_info']
    ti = result['tile_info']
    mb_cols, mb_rows = tile_geometry(fi, ti, result['tile_header']['tile_index'])
    num_mbs = mb_cols * mb_rows
    reports = []
    for component, data in enumerate(result['tile_comp_data']):
        num_blocks = num_mbs * blocks_per_mb(fi, component)
        reports.append(
            validate_component_entropy(data, num_blocks, component)
        )
    return reports


# ---------------------------------------------------------------------------
# Cocotb vector string
# ---------------------------------------------------------------------------

def cocotb_vector_string(words: list[int], label: str = "TILE_WORDS") -> str:
    lines = [f"{label} = ["]
    for i, w in enumerate(words):
        sep = "," if i < len(words) - 1 else ""
        lines.append(f"    0x{w:08X}{sep}  # [{i}]")
    lines.append("]")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pretty print
# ---------------------------------------------------------------------------

def print_frame_info(fi: dict):
    print("=== frame_info ===")
    print(f"  profile_idc        : {fi['profile_idc']}")
    print(f"  level_idc          : {fi['level_idc']}")
    print(f"  band_idc           : {fi['band_idc']}")
    print(f"  frame_width        : {fi['frame_width']} px")
    print(f"  frame_height       : {fi['frame_height']} px")
    print(f"  chroma_format_idc  : {fi['chroma_format_idc']}")
    print(f"  bit_depth          : {fi['BitDepth']}")
    print(f"  NumComps           : {fi['NumComps']}")
    print(f"  use_q_matrix       : {fi['use_q_matrix']}")


def print_tile_info(ti: dict):
    print("=== tile_info ===")
    print(f"  tile_width_in_mbs  : {ti['tile_width_in_mbs']}")
    print(f"  tile_height_in_mbs : {ti['tile_height_in_mbs']}")
    print(f"  TileCols x TileRows: {ti['TileCols']} x {ti['TileRows']}"
          f" = {ti['NumTiles']} tiles")


def print_tile_header(th: dict):
    print("=== tile_header ===")
    print(f"  tile_header_size   : {th['tile_header_size']} bytes")
    print(f"  tile_index         : {th['tile_index']}")
    for i, (sz, qp) in enumerate(
        zip(th['tile_data_size'], th['tile_qp'])
    ):
        print(f"  comp[{i}]  data_size={sz} bytes  qp={qp}")


def print_words(words: list[int], label: str, max_words: int = 256):
    n = min(len(words), max_words)
    print(f"\n=== {label} ({len(words)} x 32-bit words) ===")
    for i in range(n):
        print(f"  [{i:4d}] 0x{words[i]:08X}  {words[i]:032b}")
    if len(words) > max_words:
        print(f"  ... ({len(words) - max_words} more words not shown)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Parse raw APV file and emit first tile as 32-bit words"
    )
    ap.add_argument("file",
        help="Path to raw .apv file (RFC 9924 Appendix A format)")
    ap.add_argument("--au", type=int, default=0,
        help="0-based access-unit index (default: 0)")
    ap.add_argument("--frame-pbu", type=int, default=0,
        help="0-based frame-PBU index within the selected AU (default: 0)")
    ap.add_argument("--tile",  type=int, default=0,
        help="0-based tile index to extract (default: 0)")
    ap.add_argument("--comp",  type=int, default=None,
        help="Extract only this component index (0=Y,1=Cb,2=Cr). "
             "Default: concatenate all components")
    ap.add_argument("--words", type=int, default=None,
        help="Limit output to first N 32-bit words")
    ap.add_argument("--cocotb", action="store_true",
        help="Print a Python list literal for direct import in cocotb")
    ap.add_argument("--no-header-strip", action="store_true",
        help="Include tile_header bytes in the word stream "
             "(default: strip tile_header, emit only entropy-coded data)")
    ap.add_argument("--validate-entropy", action="store_true",
        help="Decode the selected tile's h(v) syntax and validate that every "
             "8x8 block ends exactly at scan position 64")
    args = ap.parse_args()

    result = parse_apv_file(
        args.file,
        target_tile=args.tile,
        target_au=args.au,
        target_frame_pbu=args.frame_pbu,
    )
    fi = result['frame_info']
    ti = result['tile_info']
    th = result['tile_header']
    off = result['offsets']

    print_frame_info(fi)
    print()
    print_tile_info(ti)
    print()
    print_tile_header(th)

    print(f"\n  pbu_type           : {result['pbu_type']}")
    print(f"  group_id           : {result['group_id']}")
    print(f"  AU / frame PBU     : {result['au_index']} / "
          f"{result['frame_pbu_index']}")
    print(f"  tile_size[{args.tile}]       : {result['tile_size_field']} bytes")
    print(f"  tile @ file offset : 0x{off['tile_size_field']:08X}"
          f"  (tile_data @ 0x{off['tile_data_start']:08X})")

    if args.validate_entropy:
        print("\n=== entropy/EOB validation ===")
        for component, report in enumerate(validate_tile_entropy(result)):
            print(
                f"  comp[{component}]: PASS, blocks={report['blocks']}, "
                f"terminal_zero_runs={report['terminal_zero_runs']}, "
                f"last-nonzero endings={report['blocks_ending_nonzero']}, "
                f"padding_bits={report['padding_bits']}"
            )

    # Choose bytes to serialise
    if args.no_header_strip:
        raw_bytes = result['tile_payload']
        label = (
            f"tile {args.tile} full payload "
            f"(real header + all comps + dummy bytes)"
        )
    elif args.comp is not None:
        if not 0 <= args.comp < fi['NumComps']:
            ap.error(
                f"--comp must be between 0 and {fi['NumComps'] - 1} "
                f"for chroma_format_idc={fi['chroma_format_idc']}"
            )
        raw_bytes = result['tile_comp_data'][args.comp]
        label = f"tile {args.tile} comp[{args.comp}] entropy data"
    else:
        raw_bytes = b''.join(result['tile_comp_data'])
        label = f"tile {args.tile} entropy data (all comps, header stripped)"

    words = bytes_to_words32(raw_bytes, pad=True)
    if args.words is not None:
        if args.words < 0:
            ap.error("--words must be non-negative")
        words = words[:args.words]

    if args.cocotb:
        varname = f"TILE{args.tile}_WORDS"
        if args.comp is not None:
            varname = f"TILE{args.tile}_COMP{args.comp}_WORDS"
        print()
        print(cocotb_vector_string(words, label=varname))
    else:
        print_words(words, label=label)

    print(f"\nTotal bytes : {len(raw_bytes)}")
    print(f"Total words : {len(words)} (zero-padded to 32-bit boundary)")


if __name__ == "__main__":
    main()
