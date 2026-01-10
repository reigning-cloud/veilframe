"""Per-channel LSB decoder for advanced (multi-channel) encoder payloads."""

from __future__ import annotations

import base64
import zlib
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

from .utils import update_data

MAX_BYTES = 2 * 1024 * 1024
MAX_PREVIEW = 240
MAX_BASE64_BYTES = 256 * 1024


def _bits_to_bytes(bits: np.ndarray, max_bytes: int = MAX_BYTES) -> bytes:
    usable = (bits.size // 8) * 8
    if usable <= 0:
        return b""
    bits = bits[:usable]
    data = np.packbits(bits, bitorder="big").tobytes()
    return data[:max_bytes]


def _printable_ratio(text: str) -> float:
    if not text:
        return 0.0
    printable = sum(1 for ch in text if ch.isprintable() or ch in {"\n", "\t", "\r"})
    return printable / max(1, len(text))


def _decode_text(data: bytes) -> Tuple[str, float]:
    if not data:
        return "", 0.0
    if b"\x00" in data:
        data = data.split(b"\x00", 1)[0]
    if not data:
        return "", 0.0
    try:
        text = data.decode("utf-8", errors="ignore").strip()
    except Exception:
        text = ""
    return text, _printable_ratio(text)


def _decode_zlib_with_length(data: bytes) -> Tuple[bytes, str]:
    if len(data) < 4:
        return b"", "missing length prefix"
    length = int.from_bytes(data[:4], "big")
    if length <= 0:
        return b"", "invalid length prefix"
    if length > len(data) - 4:
        return b"", "length exceeds available data"
    payload = data[4 : 4 + length]
    try:
        return zlib.decompress(payload), ""
    except Exception as exc:
        return b"", f"zlib decompress failed: {exc}"


def analyze_advanced_lsb(input_img: Path, output_dir: Path) -> None:
    """Recover per-channel text and zlib file payloads from advanced LSB encoding."""
    if not input_img.exists():
        update_data(
            output_dir,
            {"advanced_lsb": {"status": "error", "error": f"Input image not found: {input_img}"}},
        )
        return

    if not output_dir.exists():
        update_data(
            output_dir,
            {
                "advanced_lsb": {
                    "status": "error",
                    "error": f"Output directory not found: {output_dir}",
                }
            },
        )
        return

    try:
        img = Image.open(input_img).convert("RGBA")
    except Exception as exc:
        update_data(
            output_dir,
            {
                "advanced_lsb": {
                    "status": "error",
                    "error": f"Failed to open image: {exc}",
                }
            },
        )
        return

    arr = np.array(img)
    channel_map = {"R": 0, "G": 1, "B": 2, "A": 3}
    text_hits: Dict[str, Dict[str, object]] = {}
    file_hits: List[Dict[str, object]] = []
    errors: List[str] = []

    for name, idx in channel_map.items():
        try:
            bits = (arr[..., idx] & 1).reshape(-1)
            data = _bits_to_bytes(bits)
        except Exception as exc:
            errors.append(f"{name}: {exc}")
            continue

        text, ratio = _decode_text(data)
        if text and ratio >= 0.35:
            text_hits[name] = {
                "text_preview": text[:MAX_PREVIEW],
                "text_ratio": round(ratio, 3),
            }

        payload, error = _decode_zlib_with_length(data)
        if payload:
            entry: Dict[str, object] = {
                "channel": name,
                "payload_bytes": len(payload),
                "preview": payload[:MAX_PREVIEW].decode("utf-8", errors="replace"),
                "base64_preview": base64.b64encode(payload[:120]).decode(),
            }
            if len(payload) <= MAX_BASE64_BYTES:
                entry["base64"] = base64.b64encode(payload).decode()
            else:
                entry["base64"] = ""
                entry["note"] = "payload too large for full base64; use preview and re-run with a smaller file."
            file_hits.append(entry)
        elif error not in {"missing length prefix", "invalid length prefix"}:
            errors.append(f"{name}: {error}")

    status = "ok" if text_hits or file_hits else "empty"
    confidence = 0.2
    if file_hits:
        confidence = 0.85
    elif text_hits:
        confidence = 0.6

    summary = "Recovered advanced LSB payloads."
    if status == "empty":
        summary = "No advanced LSB payloads detected."

    update_data(
        output_dir,
        {
            "advanced_lsb": {
                "status": status,
                "summary": summary,
                "confidence": confidence,
                "details": {
                    "text_channels": text_hits,
                    "file_payloads": file_hits,
                    "errors": errors,
                },
                "artifacts": [],
                "timing_ms": 0,
            }
        },
    )
