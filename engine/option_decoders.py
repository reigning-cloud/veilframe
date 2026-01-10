"""Decode option implementations for stego analysis."""

from __future__ import annotations

import math
import random
import time
import zlib
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

MAGIC_SIGNATURES = {
    b"\x89PNG\r\n\x1a\n": "png",
    b"\xff\xd8\xff": "jpeg",
    b"PK\x03\x04": "zip",
    b"%PDF": "pdf",
    b"\x1f\x8b": "gzip",
    b"7z\xbc\xaf\x27\x1c": "7z",
}

DCT_COORDS = [(2, 3), (3, 2), (3, 4), (4, 3), (4, 4)]
DEFAULT_MAX_BYTES = 65536

_DCT_CACHE: Dict[Tuple[str, int, int], np.ndarray] = {}


def _now_ms() -> float:
    return time.perf_counter() * 1000


def _result(
    option_id: str,
    label: str,
    status: str,
    confidence: float,
    summary: str,
    details: Optional[Dict[str, Any]] = None,
    artifacts: Optional[List[Dict[str, Any]]] = None,
    started_ms: Optional[float] = None,
) -> Dict[str, Any]:
    timing_ms = int(max(0.0, _now_ms() - (started_ms or _now_ms())))
    return {
        "option_id": option_id,
        "label": label,
        "status": status,
        "confidence": round(float(confidence), 3),
        "summary": summary,
        "details": details or {},
        "artifacts": artifacts or [],
        "timing_ms": timing_ms,
    }


def _sniff_mime(path: Path) -> str:
    try:
        data = path.read_bytes()
    except Exception:
        data = b""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
        return "image/gif"
    try:
        img = Image.open(path)
        if img.format:
            return f"image/{img.format.lower()}"
    except Exception:
        pass
    return ""


def _bits_to_bytes(bits: List[int]) -> bytes:
    if not bits:
        return b""
    usable = (len(bits) // 8) * 8
    if usable <= 0:
        return b""
    bits = bits[:usable]
    out = bytearray(usable // 8)
    for idx in range(0, usable, 8):
        byte = 0
        for bit in bits[idx : idx + 8]:
            byte = (byte << 1) | int(bit)
        out[idx // 8] = byte
    return bytes(out)


def _bytes_to_bits(blob: bytes) -> List[int]:
    bits: List[int] = []
    for byte in blob:
        for shift in range(7, -1, -1):
            bits.append((byte >> shift) & 1)
    return bits


def _decode_with_length_prefix(bits: List[int]) -> bytes:
    if len(bits) < 16:
        return b""
    length = int("".join(str(b) for b in bits[:16]), 2)
    if length <= 0:
        return b""
    needed = 16 + length * 8
    if needed > len(bits):
        return b""
    payload_bits = bits[16:needed]
    return _bits_to_bytes(payload_bits)


def _detect_magic(blob: bytes) -> Optional[str]:
    if not blob:
        return None
    for sig, name in MAGIC_SIGNATURES.items():
        if blob.startswith(sig):
            return name
    return None


def _printable_ratio(text: str) -> float:
    if not text:
        return 0.0
    printable = sum(1 for ch in text if ch.isprintable() or ch in {"\n", "\t", "\r"})
    return printable / max(1, len(text))


def _decode_text(blob: bytes) -> Tuple[str, float]:
    if not blob:
        return "", 0.0
    try:
        text = blob.decode("utf-8", errors="ignore")
    except Exception:
        return "", 0.0
    ratio = _printable_ratio(text)
    return text.strip(), ratio


def _try_zlib(blob: bytes) -> Optional[bytes]:
    if not blob or len(blob) < 2:
        return None
    if blob[0] != 0x78:
        return None
    try:
        return zlib.decompress(blob)
    except Exception:
        return None


def _extract_lsb_bytes(
    img: Image.Image,
    channels: List[int],
    *,
    bits_per_channel: int = 1,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> bytes:
    arr = np.array(img)
    bit_mask = (1 << bits_per_channel) - 1
    flat = arr.reshape(-1, arr.shape[2])[:, channels]
    units_needed = int(math.ceil((max_bytes * 8) / bits_per_channel))
    units = (flat & bit_mask).reshape(-1)[:units_needed]
    bits: List[int] = []
    if bits_per_channel == 1:
        bits = units.astype(np.uint8).tolist()
    else:
        for value in units.tolist():
            for shift in range(bits_per_channel - 1, -1, -1):
                bits.append((value >> shift) & 1)
    return _bits_to_bytes(bits)[:max_bytes]


def analyze_lsb(
    input_img: Path,
    *,
    option_id: str,
    label: str,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> Dict[str, Any]:
    started = _now_ms()
    mime = _sniff_mime(input_img)
    if mime not in {"image/png", "image/jpeg", "image/gif"}:
        return _result(
            option_id,
            label,
            "no_signal",
            0.0,
            f"Unsupported format ({mime or 'unknown'}) for LSB extraction.",
            started_ms=started,
        )

    try:
        img = Image.open(input_img).convert("RGBA")
    except Exception as exc:
        return _result(
            option_id,
            label,
            "error",
            0.0,
            f"Failed to open image: {exc}",
            started_ms=started,
        )

    channel_orders = [
        ("RGB", [0, 1, 2]),
        ("BGR", [2, 1, 0]),
        ("RGBA", [0, 1, 2, 3]),
        ("ARGB", [3, 0, 1, 2]),
    ]
    candidates: List[Dict[str, Any]] = []

    for bits_per_channel in (1, 2, 4):
        for name, channels in channel_orders:
            blob = _extract_lsb_bytes(
                img, channels, bits_per_channel=bits_per_channel, max_bytes=max_bytes
            )
            if not blob:
                continue
            bits = _bytes_to_bits(blob)
            prefixed = _decode_with_length_prefix(bits)
            pref_text, pref_ratio = _decode_text(prefixed)
            magic = _detect_magic(blob)
            text, ratio = _decode_text(blob)
            zlib_payload = _try_zlib(blob)
            zlib_text, z_ratio = _decode_text(zlib_payload or b"")
            candidates.append(
                {
                    "order": name,
                    "bits_per_channel": bits_per_channel,
                    "magic": magic,
                    "pref_text": pref_text[:200] if pref_text else "",
                    "pref_ratio": round(pref_ratio, 3),
                    "text_preview": text[:200] if text else "",
                    "text_ratio": round(ratio, 3),
                    "zlib_preview": zlib_text[:200] if zlib_text else "",
                    "zlib_ratio": round(z_ratio, 3),
                    "bytes": len(blob),
                }
            )

    if not candidates:
        return _result(
            option_id,
            label,
            "no_signal",
            0.1,
            "No LSB payload candidates detected.",
            details={"candidates": []},
            started_ms=started,
        )

    best = max(
        candidates,
        key=lambda c: (
            0.9 if c["magic"] else 0.0,
            c["pref_ratio"],
            c["text_ratio"],
            c["zlib_ratio"],
        ),
    )
    confidence = 0.2
    if best["magic"]:
        confidence = 0.85
    elif best["pref_ratio"] >= 0.6:
        confidence = 0.7
    elif best["text_ratio"] >= 0.6:
        confidence = 0.6
    elif best["zlib_ratio"] >= 0.6:
        confidence = 0.55

    summary = "LSB bitstream extracted."
    if best["magic"]:
        summary = f"Found {best['magic']} signature in LSB stream."
    elif best["pref_text"]:
        summary = "Recovered length-prefixed text from LSB stream."
    elif best["text_preview"]:
        summary = "Recovered readable text from LSB stream."

    preview = best.get("pref_text") or best.get("text_preview", "")
    return _result(
        option_id,
        label,
        "ok" if confidence >= 0.2 else "no_signal",
        confidence,
        summary,
        details={"best": best, "candidates": candidates, "preview": preview},
        artifacts=[
            {"type": "lsb_preview", "data": preview},
            {"type": "lsb_zlib_preview", "data": best.get("zlib_preview", "")},
        ],
        started_ms=started,
    )


def _pvd_ranges(kind: str = "wu-tsai") -> List[Tuple[int, int, int]]:
    if kind == "wide":
        return [
            (0, 7, 4),
            (8, 15, 4),
            (16, 31, 5),
            (32, 63, 6),
            (64, 127, 7),
            (128, 255, 7),
        ]
    if kind == "narrow":
        return [
            (0, 7, 2),
            (8, 15, 2),
            (16, 31, 3),
            (32, 63, 4),
            (64, 127, 5),
            (128, 255, 6),
        ]
    return [
        (0, 7, 3),
        (8, 15, 3),
        (16, 31, 4),
        (32, 63, 5),
        (64, 127, 6),
        (128, 255, 7),
    ]


def _pvd_extract_bits(
    values: np.ndarray, max_bits: int, ranges: List[Tuple[int, int, int]]
) -> List[int]:
    bits: List[int] = []
    flat = values.reshape(-1)
    for idx in range(0, len(flat) - 1, 2):
        if len(bits) >= max_bits:
            break
        p1 = int(flat[idx])
        p2 = int(flat[idx + 1])
        diff = abs(p1 - p2)
        for low, high, width in ranges:
            if low <= diff <= high:
                value = diff - low
                for shift in range(width - 1, -1, -1):
                    bits.append((value >> shift) & 1)
                break
    return bits[:max_bits]


def analyze_pvd(
    input_img: Path,
    *,
    option_id: str,
    label: str,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> Dict[str, Any]:
    started = _now_ms()
    mime = _sniff_mime(input_img)
    if mime not in {"image/png", "image/jpeg", "image/gif"}:
        return _result(
            option_id,
            label,
            "no_signal",
            0.0,
            f"Unsupported format ({mime or 'unknown'}) for PVD extraction.",
            started_ms=started,
        )

    try:
        img = Image.open(input_img).convert("L")
    except Exception as exc:
        return _result(
            option_id,
            label,
            "error",
            0.0,
            f"Failed to open image: {exc}",
            started_ms=started,
        )

    values = np.array(img)
    max_bits = max_bytes * 8
    candidates: List[Dict[str, Any]] = []
    for direction in ("horizontal", "vertical", "both"):
        for range_kind in ("wu-tsai", "wide", "narrow"):
            ranges = _pvd_ranges(range_kind)
            if direction == "horizontal":
                bits = _pvd_extract_bits(values, max_bits, ranges)
            elif direction == "vertical":
                bits = _pvd_extract_bits(values.T, max_bits, ranges)
            else:
                bits_h = _pvd_extract_bits(values, max_bits, ranges)
                bits_v = _pvd_extract_bits(values.T, max_bits, ranges)
                bits = (bits_h + bits_v)[:max_bits]

            payload = _decode_with_length_prefix(bits)
            if not payload:
                payload = _bits_to_bytes(bits)
            magic = _detect_magic(payload)
            text, ratio = _decode_text(payload)
            candidates.append(
                {
                    "direction": direction,
                    "range": range_kind,
                    "payload": payload,
                    "magic": magic,
                    "text": text,
                    "ratio": ratio,
                }
            )

    best = max(candidates, key=lambda c: (0.9 if c["magic"] else 0.0, c["ratio"]))
    confidence = 0.2
    summary = "Extracted PVD bitstream."
    if best["magic"]:
        confidence = 0.8
        summary = f"Recovered {best['magic']} payload via PVD."
    elif best["ratio"] >= 0.6:
        confidence = 0.55
        summary = "Recovered readable text via PVD."

    status = "ok" if confidence >= 0.25 else "no_signal"
    return _result(
        option_id,
        label,
        status,
        confidence,
        summary,
        details={
            "payload_bytes": len(best["payload"]),
            "text_ratio": round(best["ratio"], 3),
            "magic": best["magic"],
            "direction": best["direction"],
            "range": best["range"],
            "preview": best["text"][:200],
        },
        artifacts=[{"type": "pvd_preview", "data": best["text"][:200]}],
        started_ms=started,
    )


def _dct_matrix(n: int = 8) -> np.ndarray:
    mat = np.zeros((n, n), dtype=np.float32)
    factor = math.pi / (2 * n)
    for k in range(n):
        alpha = math.sqrt(1 / n) if k == 0 else math.sqrt(2 / n)
        for i in range(n):
            mat[k, i] = alpha * math.cos((2 * i + 1) * k * factor)
    return mat


_DCT_MATS: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}


def _dct_mats(n: int) -> Tuple[np.ndarray, np.ndarray]:
    mats = _DCT_MATS.get(n)
    if mats is not None:
        return mats
    mat = _dct_matrix(n)
    mats = (mat, mat.T)
    _DCT_MATS[n] = mats
    return mats


def _dct_coords(block_size: int, robustness: str) -> List[Tuple[int, int]]:
    if robustness == "high":
        base = [(1, 2), (2, 1), (2, 2), (1, 3), (3, 1)]
    elif robustness == "low":
        base = [(5, 6), (6, 5), (6, 6), (5, 5), (4, 6)]
    else:
        base = DCT_COORDS
    scale = max(1, block_size // 8)
    coords: List[Tuple[int, int]] = []
    seen = set()
    for u, v in base:
        cu = min(block_size - 1, u * scale)
        cv = min(block_size - 1, v * scale)
        if (cu, cv) in seen:
            continue
        seen.add((cu, cv))
        coords.append((cu, cv))
    return coords


def _get_dct_coeffs(input_img: Path, block_size: int = 8) -> np.ndarray:
    stat = input_img.stat()
    key = (str(input_img.resolve()), stat.st_mtime_ns, stat.st_size, block_size)
    cached = _DCT_CACHE.get(key)
    if cached is not None:
        return cached

    img = Image.open(input_img).convert("YCbCr")
    arr = np.array(img)[:, :, 0].astype(np.float32) - 128.0
    height, width = arr.shape
    height -= height % block_size
    width -= width % block_size
    if height <= 0 or width <= 0:
        coeffs = np.zeros((0, block_size, block_size), dtype=np.float32)
        _DCT_CACHE[key] = coeffs
        return coeffs

    blocks = []
    dct_mat, idct_mat = _dct_mats(block_size)
    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            block = arr[y : y + block_size, x : x + block_size]
            dct = dct_mat @ block @ idct_mat
            blocks.append(dct)
    coeffs = (
        np.stack(blocks, axis=0)
        if blocks
        else np.zeros((0, block_size, block_size), dtype=np.float32)
    )
    _DCT_CACHE[key] = coeffs
    return coeffs


def analyze_dct(
    input_img: Path,
    *,
    option_id: str,
    label: str,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> Dict[str, Any]:
    started = _now_ms()
    mime = _sniff_mime(input_img)
    if mime != "image/jpeg":
        return _result(
            option_id,
            label,
            "no_signal",
            0.0,
            "DCT analysis is JPEG-only.",
            started_ms=started,
        )

    candidates: List[Dict[str, Any]] = []
    for block_size in (8, 16):
        try:
            coeffs = _get_dct_coeffs(input_img, block_size=block_size)
        except Exception as exc:
            return _result(
                option_id,
                label,
                "error",
                0.0,
                f"Failed to compute DCT coefficients: {exc}",
                started_ms=started,
            )

        if coeffs.size == 0:
            continue

        for robustness in ("low", "medium", "high"):
            coords = _dct_coords(block_size, robustness)
            bits: List[int] = []
            for block in coeffs:
                for u, v in coords:
                    bits.append(1 if block[u, v] >= 0 else 0)
                    if len(bits) >= max_bytes * 8:
                        break
                if len(bits) >= max_bytes * 8:
                    break

            payload = _decode_with_length_prefix(bits)
            if not payload:
                payload = _bits_to_bytes(bits)
            magic = _detect_magic(payload)
            text, ratio = _decode_text(payload)
            candidates.append(
                {
                    "block_size": block_size,
                    "robustness": robustness,
                    "coords": coords,
                    "payload": payload,
                    "magic": magic,
                    "text": text,
                    "ratio": ratio,
                    "coeff_abs_mean": float(np.mean(np.abs(coeffs))),
                    "blocks": int(coeffs.shape[0]),
                }
            )

    if not candidates:
        return _result(
            option_id,
            label,
            "no_signal",
            0.1,
            "No DCT blocks available for analysis.",
            started_ms=started,
        )

    best = max(candidates, key=lambda c: (0.9 if c["magic"] else 0.0, c["ratio"]))
    confidence = 0.2
    summary = "Computed DCT coefficient bitstream."
    if best["magic"]:
        confidence = 0.8
        summary = f"Recovered {best['magic']} payload from DCT coefficients."
    elif best["ratio"] >= 0.6:
        confidence = 0.55
        summary = "Recovered readable text from DCT coefficients."

    status = "ok" if confidence >= 0.25 else "no_signal"
    stats = {
        "blocks": int(best["blocks"]),
        "coords": best["coords"],
        "coeff_abs_mean": float(best["coeff_abs_mean"]),
        "block_size": best["block_size"],
        "robustness": best["robustness"],
    }
    return _result(
        option_id,
        label,
        status,
        confidence,
        summary,
        details={
            "stats": stats,
            "preview": best["text"][:200],
            "magic": best["magic"],
            "text_ratio": round(best["ratio"], 3),
        },
        artifacts=[{"type": "dct_preview", "data": best["text"][:200]}],
        started_ms=started,
    )


def _f5_extract_bits(
    coeffs: np.ndarray, k: int, max_bits: int, password: Optional[str] = None
) -> List[int]:
    n = (1 << k) - 1
    bits: List[int] = []
    ac_values = []
    for block in coeffs:
        for u in range(8):
            for v in range(8):
                if u == 0 and v == 0:
                    continue
                val = int(round(block[u, v]))
                if val == 0:
                    continue
                ac_values.append(val)
    if password:
        rng = random.Random(password)
        rng.shuffle(ac_values)
    total_groups = len(ac_values) // n
    idx = 0
    for _ in range(total_groups):
        if len(bits) >= max_bits:
            break
        group = ac_values[idx : idx + n]
        idx += n
        syndrome = 0
        for i, val in enumerate(group, start=1):
            if val >= 0:
                syndrome ^= i
        for shift in range(k - 1, -1, -1):
            bits.append((syndrome >> shift) & 1)
    return bits[:max_bits]


def analyze_f5(
    input_img: Path,
    *,
    option_id: str,
    label: str,
    max_bytes: int = DEFAULT_MAX_BYTES,
    password: Optional[str] = None,
) -> Dict[str, Any]:
    started = _now_ms()
    mime = _sniff_mime(input_img)
    if mime != "image/jpeg":
        return _result(
            option_id,
            label,
            "no_signal",
            0.0,
            "F5 analysis is JPEG-only.",
            started_ms=started,
        )

    try:
        coeffs = _get_dct_coeffs(input_img)
    except Exception as exc:
        return _result(
            option_id,
            label,
            "error",
            0.0,
            f"Failed to compute DCT coefficients: {exc}",
            started_ms=started,
        )

    if coeffs.size == 0:
        return _result(
            option_id,
            label,
            "no_signal",
            0.1,
            "No DCT blocks available for analysis.",
            started_ms=started,
        )

    bits = _f5_extract_bits(coeffs, k=2, max_bits=max_bytes * 8, password=password)
    payload = _decode_with_length_prefix(bits)
    if not payload:
        payload = _bits_to_bytes(bits)

    magic = _detect_magic(payload)
    text, ratio = _decode_text(payload)
    confidence = 0.2
    summary = "Computed F5 matrix decode stream."
    if magic:
        confidence = 0.75
        summary = f"Recovered {magic} payload via F5 matrix decode."
    elif ratio >= 0.6:
        confidence = 0.5
        summary = "Recovered readable text via F5 matrix decode."

    status = "ok" if confidence >= 0.25 else "no_signal"
    return _result(
        option_id,
        label,
        status,
        confidence,
        summary,
        details={"text_ratio": round(ratio, 3), "preview": text[:200], "magic": magic},
        artifacts=[{"type": "f5_preview", "data": text[:200]}],
        started_ms=started,
    )


def _spread_spectrum_bits(
    values: np.ndarray,
    password: str,
    total_bits: int,
    chip_length: int = 32,
) -> Tuple[List[int], List[float]]:
    rng = random.Random(password)
    flat = values.reshape(-1).astype(np.float32)
    flat = flat - float(np.mean(flat))
    bits: List[int] = []
    scores: List[float] = []
    max_index = len(flat)
    max_bits = max_index // chip_length
    total_bits = min(total_bits, max_bits)
    for _ in range(total_bits):
        if max_index < chip_length:
            break
        idxs = rng.sample(range(max_index), chip_length)
        seq = [1.0 if rng.random() > 0.5 else -1.0 for _ in range(chip_length)]
        corr = sum(flat[idx] * seq[i] for i, idx in enumerate(idxs)) / chip_length
        bits.append(1 if corr > 0 else 0)
        scores.append(corr)
    return bits, scores


def analyze_spread_spectrum(
    input_img: Path,
    *,
    option_id: str,
    label: str,
    password: Optional[str] = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> Dict[str, Any]:
    started = _now_ms()
    mime = _sniff_mime(input_img)
    if mime not in {"image/png", "image/jpeg", "image/gif"}:
        return _result(
            option_id,
            label,
            "no_signal",
            0.0,
            f"Unsupported format ({mime or 'unknown'}) for spread spectrum.",
            started_ms=started,
        )

    if not password:
        return _result(
            option_id,
            label,
            "no_signal",
            0.0,
            "Spread spectrum requires a password.",
            started_ms=started,
        )

    try:
        img = Image.open(input_img).convert("L")
    except Exception as exc:
        return _result(
            option_id,
            label,
            "error",
            0.0,
            f"Failed to open image: {exc}",
            started_ms=started,
        )

    values = np.array(img)
    max_bits = max_bytes * 8
    candidates: List[Dict[str, Any]] = []
    for chip_length in (8, 16, 32, 64):
        bits, scores = _spread_spectrum_bits(values, password, max_bits, chip_length=chip_length)
        payload = _decode_with_length_prefix(bits)
        if not payload:
            payload = _bits_to_bytes(bits)
        text, ratio = _decode_text(payload)

        avg_score = float(np.mean(np.abs(scores))) if scores else 0.0
        confidence = min(0.9, max(0.2, avg_score / 10.0))
        candidates.append(
            {
                "chip_length": chip_length,
                "payload": payload,
                "text": text,
                "ratio": ratio,
                "avg_score": avg_score,
                "confidence": confidence,
            }
        )

    best = max(candidates, key=lambda c: (c["ratio"], c["avg_score"]))
    confidence = best["confidence"]
    summary = "Computed spread spectrum correlation stream."
    if best["ratio"] >= 0.6:
        confidence = max(confidence, 0.6)
        summary = "Recovered readable text via spread spectrum."

    status = "ok" if confidence >= 0.25 else "no_signal"
    return _result(
        option_id,
        label,
        status,
        confidence,
        summary,
        details={
            "chip_length": best["chip_length"],
            "avg_correlation": round(best["avg_score"], 4),
            "preview": best["text"][:200],
            "text_ratio": round(best["ratio"], 3),
        },
        artifacts=[{"type": "spread_preview", "data": best["text"][:200]}],
        started_ms=started,
    )


def analyze_palette(
    input_img: Path,
    *,
    option_id: str,
    label: str,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> Dict[str, Any]:
    started = _now_ms()
    mime = _sniff_mime(input_img)
    if mime not in {"image/png", "image/gif"}:
        return _result(
            option_id,
            label,
            "no_signal",
            0.0,
            "Palette analysis expects indexed PNG/GIF.",
            started_ms=started,
        )

    try:
        img = Image.open(input_img)
    except Exception as exc:
        return _result(
            option_id,
            label,
            "error",
            0.0,
            f"Failed to open image: {exc}",
            started_ms=started,
        )

    if img.mode != "P":
        return _result(
            option_id,
            label,
            "no_signal",
            0.1,
            f"Image mode {img.mode} is not indexed palette.",
            started_ms=started,
        )

    indices = np.array(img)
    candidates: List[Dict[str, Any]] = []

    bits = (indices.reshape(-1) & 1).tolist()
    payload = _decode_with_length_prefix(bits)
    if not payload:
        payload = _bits_to_bytes(bits[: max_bytes * 8])
    text, ratio = _decode_text(payload)
    magic = _detect_magic(payload)
    candidates.append(
        {
            "mode": "index",
            "payload": payload,
            "text": text,
            "ratio": ratio,
            "magic": magic,
        }
    )

    palette = img.getpalette() or []
    if palette:
        entries = [palette[i : i + 3] for i in range(0, len(palette), 3)]
        order_bits: List[int] = []
        for idx in range(0, len(entries) - 1, 2):
            e1 = entries[idx]
            e2 = entries[idx + 1]
            lum1 = 0.2126 * e1[0] + 0.7152 * e1[1] + 0.0722 * e1[2]
            lum2 = 0.2126 * e2[0] + 0.7152 * e2[1] + 0.0722 * e2[2]
            order_bits.append(0 if lum1 <= lum2 else 1)
        order_payload = _decode_with_length_prefix(order_bits)
        if not order_payload:
            order_payload = _bits_to_bytes(order_bits)
        order_text, order_ratio = _decode_text(order_payload)
        order_magic = _detect_magic(order_payload)
        candidates.append(
            {
                "mode": "order",
                "payload": order_payload,
                "text": order_text,
                "ratio": order_ratio,
                "magic": order_magic,
            }
        )

    best = max(candidates, key=lambda c: (0.9 if c["magic"] else 0.0, c["ratio"]))
    confidence = 0.2
    summary = "Extracted palette bitstream."
    if best["magic"]:
        confidence = 0.7
        summary = f"Recovered {best['magic']} payload from palette data."
    elif best["ratio"] >= 0.6:
        confidence = 0.55
        summary = "Recovered readable text from palette data."

    status = "ok" if confidence >= 0.25 else "no_signal"
    return _result(
        option_id,
        label,
        status,
        confidence,
        summary,
        details={
            "palette_entries": len(palette) // 3 if palette else 0,
            "unique_indices": int(len(np.unique(indices))),
            "mode": best["mode"],
            "preview": best["text"][:200],
            "text_ratio": round(best["ratio"], 3),
            "magic": best["magic"],
        },
        artifacts=[{"type": "palette_preview", "data": best["text"][:200]}],
        started_ms=started,
    )


def analyze_chroma(
    input_img: Path,
    *,
    option_id: str,
    label: str,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> Dict[str, Any]:
    started = _now_ms()
    mime = _sniff_mime(input_img)
    if mime not in {"image/png", "image/jpeg", "image/gif"}:
        return _result(
            option_id,
            label,
            "no_signal",
            0.0,
            f"Unsupported format ({mime or 'unknown'}) for chroma analysis.",
            started_ms=started,
        )

    try:
        img = Image.open(input_img).convert("RGB")
    except Exception as exc:
        return _result(
            option_id,
            label,
            "error",
            0.0,
            f"Failed to open image: {exc}",
            started_ms=started,
        )

    rgb = np.array(img)
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

    ycbcr = np.array(img.convert("YCbCr"))
    cb = ycbcr[:, :, 1].astype(np.uint8)
    cr = ycbcr[:, :, 2].astype(np.uint8)

    from .color_spaces import rgb_to_hsl, rgb_to_lab

    h, s, l = rgb_to_hsl(rgb)
    h_u8 = (h * 255.0).clip(0, 255).astype(np.uint8)
    s_u8 = (s * 255.0).clip(0, 255).astype(np.uint8)

    lab_l, lab_a, lab_b = rgb_to_lab(rgb)
    a_u8 = (lab_a + 128.0).clip(0, 255).astype(np.uint8)
    b_u8 = (lab_b + 128.0).clip(0, 255).astype(np.uint8)

    spaces = {
        "ycbcr": (cb, cr),
        "hsl": (s_u8, h_u8),
        "lab": (a_u8, b_u8),
    }

    candidates: List[Dict[str, Any]] = []
    max_bits = max_bytes * 8

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
                            if len(seq) >= max_bits:
                                break
                            seq.append(int(flat2[idx]))
                            if len(seq) >= max_bits:
                                break

                    if len(seq) > max_bits:
                        seq = seq[:max_bits]

                    payload = _decode_with_length_prefix(seq)
                    if not payload:
                        payload = _bits_to_bytes(seq)
                    magic = _detect_magic(payload)
                    text, ratio = _decode_text(payload)
                    candidates.append(
                        {
                            "space": space_name,
                            "pattern": pattern_name,
                            "channel": channel_mode,
                            "bit_pos": bit_pos,
                            "payload": payload,
                            "magic": magic,
                            "text": text,
                            "ratio": ratio,
                        }
                    )

    if not candidates:
        return _result(
            option_id,
            label,
            "no_signal",
            0.1,
            "No chroma payload candidates detected.",
            started_ms=started,
        )

    best = max(candidates, key=lambda c: (0.9 if c["magic"] else 0.0, c["ratio"]))
    confidence = 0.2
    summary = "Extracted chroma-channel bitstream."
    if best["magic"]:
        confidence = 0.75
        summary = f"Recovered {best['magic']} payload from chroma channels."
    elif best["ratio"] >= 0.6:
        confidence = 0.55
        summary = "Recovered readable text from chroma channels."

    status = "ok" if confidence >= 0.25 else "no_signal"
    return _result(
        option_id,
        label,
        status,
        confidence,
        summary,
        details={
            "color_space": best["space"],
            "pattern": best["pattern"],
            "channel": best["channel"],
            "bit_pos": best["bit_pos"],
            "preview": best["text"][:200],
            "text_ratio": round(best["ratio"], 3),
            "magic": best["magic"],
        },
        artifacts=[{"type": "chroma_preview", "data": best["text"][:200]}],
        started_ms=started,
    )


def _entropy(data: bytes) -> float:
    if not data:
        return 0.0
    counts = Counter(data)
    total = len(data)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)
    return entropy


def analyze_png_chunks(
    input_img: Path,
    *,
    option_id: str,
    label: str,
) -> Dict[str, Any]:
    started = _now_ms()
    mime = _sniff_mime(input_img)
    if mime != "image/png":
        return _result(
            option_id,
            label,
            "no_signal",
            0.0,
            "PNG chunk analysis is PNG-only.",
            started_ms=started,
        )

    try:
        data = input_img.read_bytes()
    except Exception as exc:
        return _result(
            option_id,
            label,
            "error",
            0.0,
            f"Failed to read PNG file: {exc}",
            started_ms=started,
        )

    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        return _result(
            option_id,
            label,
            "error",
            0.0,
            "Invalid PNG signature.",
            started_ms=started,
        )

    offset = 8
    chunks: List[Dict[str, Any]] = []
    extracted_text: List[Dict[str, str]] = []

    while offset + 8 <= len(data):
        length = int.from_bytes(data[offset : offset + 4], "big")
        chunk_type = data[offset + 4 : offset + 8].decode("latin-1")
        start = offset + 8
        end = start + length
        chunk_data = data[start:end]
        crc = data[end : end + 4]
        offset = end + 4

        entry = {
            "type": chunk_type,
            "length": length,
            "entropy": round(_entropy(chunk_data), 3),
        }

        if chunk_type == "tEXt":
            try:
                key, text = chunk_data.split(b"\x00", 1)
                extracted_text.append(
                    {
                        "keyword": key.decode("latin-1"),
                        "text": text.decode("latin-1", errors="replace"),
                    }
                )
            except Exception:
                pass
        elif chunk_type == "zTXt":
            try:
                key, rest = chunk_data.split(b"\x00", 1)
                if len(rest) >= 1:
                    compressed = rest[1:]
                    decompressed = zlib.decompress(compressed)
                    extracted_text.append(
                        {
                            "keyword": key.decode("latin-1"),
                            "text": decompressed.decode("latin-1", errors="replace"),
                        }
                    )
            except Exception:
                pass
        elif chunk_type == "iTXt":
            try:
                parts = chunk_data.split(b"\x00", 5)
                if len(parts) >= 6:
                    keyword = parts[0].decode("latin-1")
                    comp_flag = parts[1]
                    comp_method = parts[2]
                    text = parts[5]
                    if comp_flag == b"\x01" and comp_method == b"\x00":
                        text = zlib.decompress(text)
                    extracted_text.append(
                        {
                            "keyword": keyword,
                            "text": text.decode("utf-8", errors="replace"),
                        }
                    )
            except Exception:
                pass

        chunks.append(entry)
        if chunk_type == "IEND":
            break

    status = "ok" if chunks else "no_signal"
    summary = "Parsed PNG chunks."
    if extracted_text:
        summary = f"Extracted {len(extracted_text)} text chunks from PNG."

    return _result(
        option_id,
        label,
        status,
        0.6 if extracted_text else 0.2,
        summary,
        details={"chunks": chunks, "text": extracted_text},
        artifacts=[{"type": "png_text", "data": extracted_text}],
        started_ms=started,
    )


def analyze_auto_detect(
    input_img: Path,
    *,
    option_id: str,
    label: str,
    registry: Dict[str, Any],
    password: Optional[str] = None,
) -> Dict[str, Any]:
    started = _now_ms()
    mime = _sniff_mime(input_img)
    candidates: List[Dict[str, Any]] = []
    order: List[str] = []

    if mime == "image/png":
        order = ["png_chunks", "palette", "lsb", "chroma", "pvd", "spread_spectrum"]
    elif mime == "image/jpeg":
        order = ["dct", "f5", "chroma", "lsb", "pvd", "spread_spectrum"]
    else:
        order = ["lsb", "pvd", "spread_spectrum"]

    for opt_id in order:
        option = registry.get(opt_id)
        if not option:
            continue
        params = {"password": password}
        result = option["analyzer"](input_img, **option["params"](option, params))
        if result["status"] == "ok":
            candidates.append(
                {
                    "option_id": opt_id,
                    "label": result["label"],
                    "confidence": result["confidence"],
                    "summary": result["summary"],
                }
            )

    if not candidates:
        return _result(
            option_id,
            label,
            "no_signal",
            0.2,
            "No strong candidates detected in auto mode.",
            details={"candidates": []},
            started_ms=started,
        )

    candidates.sort(key=lambda c: c["confidence"], reverse=True)
    summary = f"Top candidate: {candidates[0]['label']} ({candidates[0]['confidence']})."
    return _result(
        option_id,
        label,
        "ok",
        candidates[0]["confidence"],
        summary,
        details={"candidates": candidates},
        started_ms=started,
    )


def build_auto_detect_result(
    option_id: str,
    label: str,
    option_results: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    started = _now_ms()
    candidates: List[Dict[str, Any]] = []
    for opt_id, result in option_results.items():
        if not result or result.get("status") != "ok":
            continue
        confidence = float(result.get("confidence", 0.0))
        candidates.append(
            {
                "option_id": opt_id,
                "label": result.get("label", opt_id),
                "confidence": round(confidence, 3),
                "summary": result.get("summary", ""),
            }
        )

    if not candidates:
        return _result(
            option_id,
            label,
            "no_signal",
            0.2,
            "No strong candidates detected in auto mode.",
            details={"candidates": []},
            started_ms=started,
        )

    candidates.sort(key=lambda c: c["confidence"], reverse=True)
    summary = f"Top candidate: {candidates[0]['label']} ({candidates[0]['confidence']})."
    return _result(
        option_id,
        label,
        "ok",
        candidates[0]["confidence"],
        summary,
        details={"candidates": candidates},
        started_ms=started,
    )


def param_adapter(option: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    payload = {"option_id": option["id"], "label": option["label"]}
    if (option.get("requires_password") or option["id"] in {"f5"}) and params.get("password"):
        payload["password"] = params["password"]
    return payload
