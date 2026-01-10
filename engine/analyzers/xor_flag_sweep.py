"""XOR + bitshift sweep for flag-like markers across LSB/PVD/Chroma streams."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from ..color_spaces import rgb_to_hsl, rgb_to_lab
from ..option_decoders import _bits_to_bytes, _decode_with_length_prefix, _pvd_extract_bits, _pvd_ranges
from .utils import update_data

MAX_BYTES = 65536
MAX_HITS = 12
MAX_PREVIEW = 200
PATTERNS = [b"ctf{", b"flag{", b"steg{"]


def _ascii_preview(blob: bytes, limit: int = MAX_PREVIEW) -> str:
    out: List[str] = []
    for b in blob[:limit]:
        if 32 <= b < 127:
            out.append(chr(b))
        elif b == 9:
            out.append("\\t")
        elif b == 10:
            out.append("\\n")
        elif b == 13:
            out.append("\\r")
        else:
            out.append(f"\\x{b:02x}")
    return "".join(out)


def _find_pattern(data: bytes) -> Optional[Tuple[int, bytes, str]]:
    lower = data.lower()
    best_idx = None
    best_pat: Optional[bytes] = None
    for pat in PATTERNS:
        idx = lower.find(pat)
        if idx != -1 and (best_idx is None or idx < best_idx):
            best_idx = idx
            best_pat = pat
    if best_idx is None or best_pat is None:
        return None
    end = data.find(b"}", best_idx)
    if end != -1 and end - best_idx <= MAX_PREVIEW:
        snippet = data[best_idx : end + 1]
    else:
        snippet = data[best_idx : best_idx + MAX_PREVIEW]
    return best_idx, snippet, best_pat.decode("ascii")


def _scan_bytes(data: bytes) -> Optional[Dict[str, Any]]:
    if not data:
        return None
    direct = _find_pattern(data)
    if direct:
        idx, snippet, pat = direct
        return {
            "pattern": pat,
            "xor_key": None,
            "xor_key_hex": None,
            "offset": idx,
            "preview": _ascii_preview(snippet),
            "score": 0.95,
        }
    for key in range(256):
        xored = bytes(b ^ key for b in data)
        match = _find_pattern(xored)
        if match:
            idx, snippet, pat = match
            score = 0.85
            return {
                "pattern": pat,
                "xor_key": key,
                "xor_key_hex": f"0x{key:02x}",
                "offset": idx,
                "preview": _ascii_preview(snippet),
                "score": score,
            }
    return None


def _extract_lsb_bits(
    arr: np.ndarray, channels: List[int], *, bits_per_channel: int, max_bits: int
) -> np.ndarray:
    bit_mask = (1 << bits_per_channel) - 1
    flat = arr.reshape(-1, arr.shape[2])[:, channels]
    units_needed = int(math.ceil(max_bits / bits_per_channel))
    units = (flat & bit_mask).reshape(-1)[:units_needed]
    if bits_per_channel == 1:
        bits = units.astype(np.uint8)
    else:
        bit_list: List[int] = []
        for value in units.tolist():
            for shift in range(bits_per_channel - 1, -1, -1):
                bit_list.append((value >> shift) & 1)
        bits = np.array(bit_list, dtype=np.uint8)
    return bits[:max_bits]


def _scan_bitstream(
    bits: np.ndarray,
    *,
    method: str,
    config: Dict[str, Any],
    max_bytes: int,
) -> List[Dict[str, Any]]:
    hits: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []
    max_bits = max_bytes * 8 + 7
    if bits.size < 16:
        return hits
    if bits.size > max_bits:
        bits = bits[:max_bits]
    for shift in range(8):
        if bits.size <= shift + 8:
            continue
        shifted = bits[shift:]
        shifted_list = shifted.tolist()
        payloads: List[Tuple[str, bytes]] = []
        if len(shifted_list) >= 16:
            prefix = _decode_with_length_prefix(shifted_list)
            if prefix:
                payloads.append(("prefix", prefix))
        raw = _bits_to_bytes(shifted_list)
        if raw:
            payloads.append(("raw", raw))
        for source, payload in payloads:
            payload = payload[:max_bytes]
            hit = _scan_bytes(payload)
            if not hit:
                continue
            hit.update(
                {
                    "method": method,
                    "shift": shift,
                    "source": source,
                    "config": config,
                }
            )
            hit["score"] = max(0.1, hit["score"] - shift * 0.01)
            hits.append(hit)
            break
    return hits


def _dedupe_hits(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    unique: List[Dict[str, Any]] = []
    for hit in hits:
        sig = (hit.get("pattern"), hit.get("xor_key"), hit.get("preview"))
        if sig in seen:
            continue
        seen.add(sig)
        unique.append(hit)
    return unique


def analyze_xor_flag_sweep(input_img: Path, output_dir: Path) -> None:
    if not input_img.exists():
        update_data(
            output_dir,
            {"xor_flag_sweep": {"status": "error", "error": f"Input image not found: {input_img}"}},
        )
        return

    try:
        img = Image.open(input_img)
    except Exception as exc:
        update_data(
            output_dir,
            {
                "xor_flag_sweep": {
                    "status": "error",
                    "error": f"Failed to open image: {exc}",
                }
            },
        )
        return

    hits: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []
    scanned = {"lsb": 0, "pvd": 0, "chroma": 0}
    max_bits = MAX_BYTES * 8

    try:
        rgba = img.convert("RGBA")
        arr = np.array(rgba)
        channel_orders = [
            ("RGB", [0, 1, 2]),
            ("BGR", [2, 1, 0]),
            ("RGBA", [0, 1, 2, 3]),
            ("ARGB", [3, 0, 1, 2]),
        ]
        for bits_per_channel in (1, 2, 4):
            for name, channels in channel_orders:
                bits = _extract_lsb_bits(
                    arr, channels, bits_per_channel=bits_per_channel, max_bits=max_bits + 7
                )
                scanned["lsb"] += 1
                hits.extend(
                    _scan_bitstream(
                        bits,
                        method="lsb",
                        config={"channels": name, "bits_per_channel": bits_per_channel},
                        max_bytes=MAX_BYTES,
                    )
                )
    except Exception as exc:
        errors.append({"method": "lsb", "error": f"LSB sweep failed: {exc}"})

    try:
        gray = img.convert("L")
        values = np.array(gray)
        for direction in ("horizontal", "vertical", "both"):
            for range_kind in ("wu-tsai", "wide", "narrow"):
                ranges = _pvd_ranges(range_kind)
                if direction == "horizontal":
                    bits_list = _pvd_extract_bits(values, max_bits + 7, ranges)
                elif direction == "vertical":
                    bits_list = _pvd_extract_bits(values.T, max_bits + 7, ranges)
                else:
                    bits_h = _pvd_extract_bits(values, max_bits + 7, ranges)
                    bits_v = _pvd_extract_bits(values.T, max_bits + 7, ranges)
                    bits_list = (bits_h + bits_v)[: max_bits + 7]
                bits = np.array(bits_list, dtype=np.uint8)
                scanned["pvd"] += 1
                hits.extend(
                    _scan_bitstream(
                        bits,
                        method="pvd",
                        config={"direction": direction, "range": range_kind},
                        max_bytes=MAX_BYTES,
                    )
                )
    except Exception as exc:
        errors.append({"method": "pvd", "error": f"PVD sweep failed: {exc}"})

    try:
        rgb = np.array(img.convert("RGB"))
        height, width = rgb.shape[:2]
        yy, xx = np.mgrid[0:height, 0:width]
        masks = {
            "sequential": np.ones((height, width), dtype=bool),
            "checkerboard": (xx + yy) % 2 == 0,
        }
        gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
        gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
        gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
        edges = gx + gy
        threshold = np.percentile(edges, 75)
        masks["edges"] = edges >= threshold

        ycbcr = np.array(Image.fromarray(rgb).convert("YCbCr"))
        cb = ycbcr[:, :, 1].astype(np.uint8)
        cr = ycbcr[:, :, 2].astype(np.uint8)

        h, s, _l = rgb_to_hsl(rgb)
        h_u8 = (h * 255.0).clip(0, 255).astype(np.uint8)
        s_u8 = (s * 255.0).clip(0, 255).astype(np.uint8)

        _lab_l, lab_a, lab_b = rgb_to_lab(rgb)
        a_u8 = (lab_a + 128.0).clip(0, 255).astype(np.uint8)
        b_u8 = (lab_b + 128.0).clip(0, 255).astype(np.uint8)

        spaces = {
            "ycbcr": (cb, cr),
            "hsl": (s_u8, h_u8),
            "lab": (a_u8, b_u8),
        }

        for space_name, (ch1, ch2) in spaces.items():
            for pattern_name, mask in masks.items():
                for bit_pos in (0, 1, 2, 3):
                    bit_mask = 1 << bit_pos
                    ch1_bits = ((ch1 & bit_mask) >> bit_pos).astype(np.uint8)
                    ch2_bits = ((ch2 & bit_mask) >> bit_pos).astype(np.uint8)
                    for channel_mode in ("both", "cb", "cr"):
                        if channel_mode == "cb":
                            seq = ch1_bits[mask].tolist()
                        elif channel_mode == "cr":
                            seq = ch2_bits[mask].tolist()
                        else:
                            seq = []
                            flat1 = ch1_bits[mask].tolist()
                            flat2 = ch2_bits[mask].tolist()
                            limit = min(len(flat1), len(flat2))
                            for idx in range(limit):
                                seq.append(int(flat1[idx]))
                                if len(seq) >= max_bits + 7:
                                    break
                                seq.append(int(flat2[idx]))
                                if len(seq) >= max_bits + 7:
                                    break
                        if len(seq) > max_bits + 7:
                            seq = seq[: max_bits + 7]
                        bits = np.array(seq, dtype=np.uint8)
                        scanned["chroma"] += 1
                        hits.extend(
                            _scan_bitstream(
                                bits,
                                method="chroma",
                                config={
                                    "space": space_name,
                                    "pattern": pattern_name,
                                    "channel": channel_mode,
                                    "bit_pos": bit_pos,
                                },
                                max_bytes=MAX_BYTES,
                            )
                        )
    except Exception as exc:
        errors.append({"method": "chroma", "error": f"Chroma sweep failed: {exc}"})

    hits = _dedupe_hits(hits)
    hits.sort(key=lambda h: (-h.get("score", 0.0), h.get("method", ""), h.get("offset", 0)))
    hits = hits[:MAX_HITS]

    if hits:
        confidence = max(hit.get("score", 0.0) for hit in hits)
        summary = f"Found {len(hits)} potential flag patterns in XOR sweep."
        status = "ok"
    elif errors:
        confidence = 0.0
        summary = "XOR sweep failed to scan some sources."
        status = "error"
    else:
        confidence = 0.0
        summary = "No flag patterns detected in XOR sweep."
        status = "no_signal"

    update_data(
        output_dir,
        {
            "xor_flag_sweep": {
                "status": status,
                "summary": summary,
                "confidence": round(float(confidence), 3),
                "details": {
                    "hits": hits,
                    "scanned_streams": scanned,
                    "errors": errors,
                    "max_bytes": MAX_BYTES,
                },
                "artifacts": [],
                "timing_ms": 0,
            }
        },
    )
