"""Zero-width Unicode steganography decoder."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

from .utils import update_data

ZWSP = "\u200b"  # zero-width space
ZWNJ = "\u200c"  # zero-width non-joiner
ZWJ = "\u200d"  # zero-width joiner (delimiter)

MAX_EXTRACT_BYTES = 131072
MAX_RESULTS = 6

CHANNEL_CONFIGS: List[Dict[str, object]] = [
    {"name": "RGB-1", "channels": [0, 1, 2], "bits": 1},
    {"name": "RGBA-1", "channels": [0, 1, 2, 3], "bits": 1},
    {"name": "RGB-2", "channels": [0, 1, 2], "bits": 2},
    {"name": "RGBA-2", "channels": [0, 1, 2, 3], "bits": 2},
    {"name": "R-1", "channels": [0], "bits": 1},
    {"name": "G-1", "channels": [1], "bits": 1},
    {"name": "B-1", "channels": [2], "bits": 1},
    {"name": "R-2", "channels": [0], "bits": 2},
    {"name": "G-2", "channels": [1], "bits": 2},
    {"name": "B-2", "channels": [2], "bits": 2},
    {"name": "RG-1", "channels": [0, 1], "bits": 1},
    {"name": "RB-1", "channels": [0, 2], "bits": 1},
    {"name": "GB-1", "channels": [1, 2], "bits": 1},
    {"name": "RG-2", "channels": [0, 1], "bits": 2},
    {"name": "RB-2", "channels": [0, 2], "bits": 2},
    {"name": "GB-2", "channels": [1, 2], "bits": 2},
]


def _units_to_bytes(units: np.ndarray, bits_per_unit: int) -> bytes:
    if bits_per_unit <= 0:
        return b""
    bit_array: List[int] = []
    if bits_per_unit == 1:
        bit_array = units.astype(np.uint8).tolist()
    else:
        for value in units.tolist():
            for shift in range(bits_per_unit - 1, -1, -1):
                bit_array.append((value >> shift) & 1)

    if not bit_array:
        return b""
    byte_len = math.ceil(len(bit_array) / 8)
    out = bytearray(byte_len)
    for i, bit in enumerate(bit_array):
        byte_idx = i // 8
        out[byte_idx] = (out[byte_idx] << 1) | bit
    remaining = len(bit_array) % 8
    if remaining:
        out[-1] <<= 8 - remaining
    return bytes(out)


def _extract_raw_bytes(
    arr: np.ndarray, channels: List[int], bits_per_channel: int, max_bytes: int
) -> bytes:
    bit_mask = (1 << bits_per_channel) - 1
    flat = arr.reshape(-1, arr.shape[2])[:, channels]
    units_needed = int(math.ceil((max_bytes * 8) / bits_per_channel))
    units = (flat & bit_mask).reshape(-1)[:units_needed]
    return _units_to_bytes(units, bits_per_channel)[:max_bytes]


def _decode_zero_width_payloads(text: str) -> List[bytes]:
    payloads: List[bytes] = []
    idx = 0
    while idx < len(text):
        start = text.find(ZWJ, idx)
        if start == -1:
            break
        end = text.find(ZWJ, start + 1)
        if end == -1:
            break
        content = text[start + 1 : end]
        bits = []
        for ch in content:
            if ch == ZWSP:
                bits.append("0")
            elif ch == ZWNJ:
                bits.append("1")
        if len(bits) >= 8:
            bit_str = "".join(bits)
            usable = len(bit_str) // 8 * 8
            data = bytes(
                int(bit_str[i : i + 8], 2) for i in range(0, usable, 8)
            )
            if data:
                payloads.append(data)
        idx = end + 1
    return payloads


def _scan_text_for_payloads(text: str) -> List[Dict[str, object]]:
    results = []
    payloads = _decode_zero_width_payloads(text)
    for payload in payloads:
        try:
            decoded = payload.decode("utf-8", errors="replace")
        except Exception:
            decoded = ""
        if not _is_plausible_text(decoded):
            continue
        results.append({"payload": decoded.strip(), "length": len(payload)})
        if len(results) >= MAX_RESULTS:
            break
    return results


def _is_plausible_text(text: str) -> bool:
    if not text:
        return False
    printable = 0
    for ch in text:
        if ch.isprintable() or ch in {"\n", "\t"}:
            printable += 1
    ratio = printable / max(1, len(text))
    if ratio < 0.6:
        return False
    if re.search(r"(flag|ctf|steg|secret)", text, re.IGNORECASE):
        return True
    return len(text.strip()) >= 4


def analyze_zero_width(input_img: Path, output_dir: Path) -> None:
    """Decode zero-width Unicode payloads embedded in extracted text."""
    try:
        img = Image.open(input_img).convert("RGBA")
    except Exception as exc:
        update_data(output_dir, {"zero_width": {"status": "error", "error": str(exc)}})
        return

    arr = np.array(img)
    results = []
    seen_payloads = set()

    for cfg in CHANNEL_CONFIGS:
        channels = cfg["channels"]
        bits = cfg["bits"]
        raw = _extract_raw_bytes(arr, channels, bits, MAX_EXTRACT_BYTES)
        if not raw:
            continue
        text = raw.decode("utf-8", errors="ignore")
        if ZWJ not in text:
            continue
        payloads = _scan_text_for_payloads(text)
        for payload in payloads:
            payload_text = payload["payload"]
            if payload_text in seen_payloads:
                continue
            seen_payloads.add(payload_text)
            results.append(
                {
                    "config": cfg["name"],
                    "channels": "".join("RGBA"[idx] for idx in channels),
                    "bits": bits,
                    "payload": payload_text,
                    "length": payload["length"],
                }
            )
            if len(results) >= MAX_RESULTS:
                break
        if len(results) >= MAX_RESULTS:
            break

    if len(results) < MAX_RESULTS:
        try:
            raw_bytes = input_img.read_bytes()
        except Exception:
            raw_bytes = b""
        if raw_bytes:
            raw_text = raw_bytes.decode("utf-8", errors="ignore")
            if ZWJ in raw_text:
                payloads = _scan_text_for_payloads(raw_text)
                for payload in payloads:
                    payload_text = payload["payload"]
                    if payload_text in seen_payloads:
                        continue
                    seen_payloads.add(payload_text)
                    results.append(
                        {
                            "config": "raw-file",
                            "payload": payload_text,
                            "length": payload["length"],
                        }
                    )
                    if len(results) >= MAX_RESULTS:
                        break

    if results:
        update_data(output_dir, {"zero_width": {"status": "ok", "output": results}})
    else:
        update_data(
            output_dir,
            {"zero_width": {"status": "empty", "reason": "No zero-width payloads detected"}},
        )
