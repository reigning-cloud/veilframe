"""Decoder for STE.GG (STEG v3) payloads stored in bit planes."""

from __future__ import annotations

import re
import shutil
import subprocess
import zlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from .utils import MAX_PENDING_TIME, update_data

MAGIC = b"STEG"
HEADER_SIZE = 32
FORMAT_VERSION = 3

CHANNEL_PRESETS: Dict[str, List[int]] = {
    "R": [0],
    "G": [1],
    "B": [2],
    "A": [3],
    "RG": [0, 1],
    "RB": [0, 2],
    "GB": [1, 2],
    "RGB": [0, 1, 2],
    "RGBA": [0, 1, 2, 3],
}

HEADER_CONFIGS: List[Tuple[str, List[int], int]] = [
    ("RGB-1", [0, 1, 2], 1),
    ("RGBA-2", [0, 1, 2, 3], 2),
    ("RGBA-1", [0, 1, 2, 3], 1),
    ("RGB-2", [0, 1, 2], 2),
]

SCAN_CONFIGS: List[Tuple[str, List[int], int]] = [
    ("RGB", [0, 1, 2], 1),
    ("RGB-2bit", [0, 1, 2], 2),
    ("RGBA", [0, 1, 2, 3], 1),
    ("RGBA-2bit", [0, 1, 2, 3], 2),
    ("R", [0], 1),
    ("G", [1], 1),
    ("B", [2], 1),
    ("R-2bit", [0], 2),
    ("G-2bit", [1], 2),
    ("B-2bit", [2], 2),
    ("RG", [0, 1], 1),
    ("RB", [0, 2], 1),
    ("GB", [1, 2], 1),
    ("RG-2bit", [0, 1], 2),
    ("RB-2bit", [0, 2], 2),
    ("GB-2bit", [1, 2], 2),
]

VALID_EXTENSIONS = [
    ".txt",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".webp",
    ".svg",
    ".pdf",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".ppt",
    ".pptx",
    ".zip",
    ".rar",
    ".7z",
    ".tar",
    ".gz",
    ".mp3",
    ".wav",
    ".ogg",
    ".mp4",
    ".avi",
    ".mkv",
    ".mov",
    ".html",
    ".css",
    ".js",
    ".json",
    ".xml",
    ".csv",
    ".py",
    ".java",
    ".c",
    ".cpp",
    ".h",
    ".rs",
    ".go",
    ".rb",
    ".md",
    ".rtf",
    ".log",
    ".dat",
    ".bin",
    ".exe",
    ".dll",
]


def _units_to_bytes(units: np.ndarray, bits_per_unit: int) -> bytes:
    if bits_per_unit <= 0:
        return b""
    if bits_per_unit == 1:
        bit_array = units.astype(np.uint8).tolist()
    else:
        bit_array: List[int] = []
        for value in units.tolist():
            for shift in range(bits_per_unit - 1, -1, -1):
                bit_array.append((value >> shift) & 1)

    if not bit_array:
        return b""
    byte_len = (len(bit_array) + 7) // 8
    out = bytearray(byte_len)
    for i, bit in enumerate(bit_array):
        byte_idx = i // 8
        out[byte_idx] = (out[byte_idx] << 1) | bit
    # Pad the last byte if needed
    remaining = len(bit_array) % 8
    if remaining:
        out[-1] <<= 8 - remaining
    return bytes(out)


def _extract_raw_bytes(
    arr: np.ndarray, channels: List[int], bits_per_channel: int, max_bytes: int
) -> bytes:
    if bits_per_channel <= 0:
        return b""
    bit_mask = (1 << bits_per_channel) - 1
    flat = arr.reshape(-1, arr.shape[2])[:, channels]
    units_needed = int(np.ceil((max_bytes * 8) / bits_per_channel))
    units = (flat & bit_mask).reshape(-1)[:units_needed]
    return _units_to_bytes(units, bits_per_channel)[:max_bytes]


def _parse_header(data: bytes) -> Optional[Dict[str, object]]:
    if len(data) < HEADER_SIZE:
        return None
    if data[0:4] != MAGIC:
        return None
    version = data[4]
    channel_mask = data[5]
    bits_per_channel = data[6]
    bit_offset = data[7]
    flags = data[8]

    channels = [idx for idx in range(4) if channel_mask & (1 << idx)]
    payload_length = int.from_bytes(data[16:20], "big")
    original_length = int.from_bytes(data[20:24], "big")
    crc = int.from_bytes(data[24:28], "big")

    return {
        "version": version,
        "channels": channels,
        "bits_per_channel": bits_per_channel,
        "bit_offset": bit_offset,
        "compressed": bool(flags & 1),
        "interleaved": bool(flags & 2),
        "payload_length": payload_length,
        "original_length": original_length,
        "crc": crc,
    }


def _find_header(arr: np.ndarray) -> Tuple[Optional[Dict[str, object]], Optional[str]]:
    configs = list(HEADER_CONFIGS)
    for name, channels in CHANNEL_PRESETS.items():
        for bits in (1, 2):
            cfg_name = f"{name}-{bits}"
            if cfg_name not in [c[0] for c in configs]:
                configs.append((cfg_name, channels, bits))

    for cfg_name, channels, bits in configs:
        header_bytes = _extract_raw_bytes(arr, channels, bits, HEADER_SIZE)
        header = _parse_header(header_bytes)
        if header:
            return header, cfg_name
    return None, None


def _maybe_decompress(data: bytes) -> Tuple[bytes, bool]:
    for wbits in (zlib.MAX_WBITS, -zlib.MAX_WBITS):
        try:
            return zlib.decompress(data, wbits), True
        except zlib.error:
            continue
    return data, False


def _is_image_data(data: bytes) -> bool:
    return data.startswith(b"\x89PNG\r\n\x1a\n") or data.startswith(b"\xff\xd8")


def _detect_image_ext(data: bytes) -> Optional[str]:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if data.startswith(b"\xff\xd8"):
        return ".jpg"
    return None


def _extract_file_from_data(data: bytes) -> Tuple[Optional[str], bytes]:
    if _is_image_data(data):
        return None, data

    if len(data) > 2 and 0 < data[0] <= 100:
        name_len = data[0]
        if name_len + 1 < len(data):
            try:
                filename = data[1 : 1 + name_len].decode("utf-8", errors="strict")
            except Exception:
                filename = None
            if filename:
                if re.match(r"^[a-zA-Z0-9_\\-. ]+$", filename) and "/" not in filename and "\\" not in filename:
                    lower = filename.lower()
                    if any(lower.endswith(ext) for ext in VALID_EXTENSIONS):
                        return filename, data[1 + name_len :]
    return None, data


def _safe_filename(name: str) -> str:
    return Path(name).name.replace("\n", "_").replace("\r", "_")


def _detect_coherent_text(data: bytes) -> Dict[str, object]:
    if not data:
        return {"is_text": False, "confidence": 0, "preview": "", "ratio": 0.0}
    try:
        text = data.decode("utf-8", errors="ignore")
    except Exception:
        return {"is_text": False, "confidence": 0, "preview": "", "ratio": 0.0}

    sample = text[:500]
    printable = 0
    for ch in sample:
        code = ord(ch)
        if 32 <= code < 127 or ch in {"\n", "\t"}:
            printable += 1
    ratio = printable / max(1, len(sample))
    common_words = ["the", "and", "is", "in", "to", "of", "a", "for", "flag", "ctf", "secret"]
    text_lower = text.lower()
    word_matches = sum(1 for w in common_words if w in text_lower)
    confidence = min(100, max(0, ratio * 50 + word_matches * 5))

    preview = re.sub(r"[^\x20-\x7E\n\t]", "·", sample[:120]).strip()
    return {
        "is_text": confidence > 35 and ratio > 0.7,
        "confidence": confidence,
        "preview": preview,
        "ratio": ratio,
    }


def _smart_scan(arr: np.ndarray) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    for name, channels, bits in SCAN_CONFIGS:
        try:
            raw = _extract_raw_bytes(arr, channels, bits, 2048)
            has_magic = raw[:4] == MAGIC
            analysis = _detect_coherent_text(raw)
            if has_magic:
                status = "STEG_HEADER"
            elif analysis["is_text"] and analysis["confidence"] >= 50:
                status = "TEXT_FOUND"
            elif analysis["is_text"]:
                status = "POSSIBLE_TEXT"
            elif analysis["ratio"] > 0.5:
                status = "MIXED_DATA"
            else:
                status = "BINARY/NOISE"

            if has_magic or analysis["confidence"] >= 30:
                results.append(
                    {
                        "config": name,
                        "channels": "".join("RGBA"[idx] for idx in channels),
                        "bits": bits,
                        "status": status,
                        "confidence": analysis["confidence"],
                        "preview": analysis["preview"],
                    }
                )
        except Exception:
            continue

    results.sort(key=lambda item: (item["status"] != "STEG_HEADER", -item["confidence"]))
    return results[:5]


def _write_payload(
    output_dir: Path, filename: Optional[str], payload: bytes
) -> Tuple[Optional[str], Optional[str]]:
    stegg_dir = output_dir / "stegg"
    stegg_dir.mkdir(parents=True, exist_ok=True)

    ext = _detect_image_ext(payload)
    if filename:
        safe_name = _safe_filename(filename)
    elif ext:
        safe_name = f"stegg_payload{ext}"
    else:
        safe_name = "stegg_payload.bin"

    out_path = stegg_dir / safe_name
    try:
        out_path.write_bytes(payload)
    except Exception:
        return None, None

    archive = output_dir / "stegg_payload.7z"
    if shutil.which("7z"):
        try:
            subprocess.run(
                ["7z", "a", "-t7z", "-y", str(archive), out_path.name],
                cwd=str(stegg_dir),
                capture_output=True,
                text=True,
                timeout=MAX_PENDING_TIME,
                check=False,
            )
        except Exception:
            pass

    return str(out_path.relative_to(output_dir)), str(archive.relative_to(output_dir))


def analyze_stegg(input_img: Path, output_dir: Path) -> None:
    """Decode STE.GG (STEG v3) payloads from an image."""
    try:
        img = Image.open(input_img).convert("RGBA")
    except Exception as exc:
        update_data(output_dir, {"stegg": {"status": "error", "error": str(exc)}})
        return

    arr = np.array(img)
    header, detected_cfg = _find_header(arr)
    if not header:
        scan_results = _smart_scan(arr)
        if scan_results:
            update_data(
                output_dir,
                {
                    "stegg": {
                        "status": "ok",
                        "output": {
                            "status": "no_header",
                            "scan_results": scan_results,
                            "note": "No STEG header detected; scan shows possible text candidates.",
                        },
                    }
                },
            )
            return
        update_data(
            output_dir,
            {"stegg": {"status": "empty", "reason": "No STEG header detected"}},
        )
        return

    bits_per_channel = int(header["bits_per_channel"] or 1)
    if bits_per_channel <= 0 or bits_per_channel > 4:
        update_data(
            output_dir,
            {"stegg": {"status": "error", "error": f"Unsupported bits_per_channel {bits_per_channel}"}},
        )
        return
    channels = header["channels"] or [0, 1, 2]
    payload_length = int(header["payload_length"] or 0)

    total_pixels = arr.shape[0] * arr.shape[1]
    capacity = (total_pixels * len(channels) * max(bits_per_channel, 1)) // 8
    if payload_length <= 0 or payload_length > capacity:
        update_data(
            output_dir,
            {
                "stegg": {
                    "status": "error",
                    "error": f"Invalid payload length {payload_length} (capacity {capacity})",
                }
            },
        )
        return

    total_len = HEADER_SIZE + max(payload_length, 0)
    raw = _extract_raw_bytes(arr, channels, bits_per_channel, total_len)
    payload = raw[HEADER_SIZE : HEADER_SIZE + payload_length]

    decompressed = payload
    decompressed_ok = True
    if header["compressed"]:
        decompressed, decompressed_ok = _maybe_decompress(payload)

    if header["original_length"]:
        decompressed = decompressed[: int(header["original_length"])]

    crc_expected = int(header["crc"])
    crc_actual = zlib.crc32(decompressed) & 0xFFFFFFFF
    crc_ok = crc_expected == crc_actual

    filename, file_data = _extract_file_from_data(decompressed)
    file_path, archive_path = _write_payload(output_dir, filename, file_data)

    channel_str = "".join("RGBA"[idx] for idx in channels)
    preview = ""
    try:
        preview_text = file_data[:500].decode("utf-8", errors="replace")
        preview = preview_text.replace("\x00", "").strip()
    except Exception:
        preview = ""

    output = {
        "version": header["version"],
        "format": f"STEG v{header['version']}",
        "channels": channel_str,
        "bits_per_channel": bits_per_channel,
        "compressed": header["compressed"],
        "payload_length": payload_length,
        "original_length": int(header["original_length"]),
        "crc_ok": crc_ok,
        "crc_expected": f"{crc_expected:08x}",
        "crc_actual": f"{crc_actual:08x}",
        "filename": filename,
        "file": file_path,
        "archive": archive_path,
        "detected_config": detected_cfg,
        "decompress_ok": decompressed_ok,
        "preview": preview[:200] if preview else "",
    }

    update_data(output_dir, {"stegg": {"status": "ok", "output": output}})
