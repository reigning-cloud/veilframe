"""Bit-plane carver for hidden payload discovery."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

from .utils import MAX_PENDING_TIME, update_data

MAX_BYTES = 512_000
MAX_CANDIDATES = 8
MAX_SECONDARY = 4
PRIORITY_LABELS = {"jpeg", "png", "zip", "7z", "pdf", "gzip", "bmp", "openpgp"}
MAGICS = [
    ("jpeg", b"\xff\xd8\xff"),
    ("png", b"\x89PNG\r\n\x1a\n"),
    ("zip", b"PK\x03\x04"),
    ("gzip", b"\x1f\x8b\x08"),
    ("zlib", b"\x78\x9c"),
    ("zlib", b"\x78\xda"),
    ("zlib", b"\x78\x01"),
    ("pdf", b"%PDF"),
    ("bmp", b"BM"),
    ("7z", b"7z\xbc\xaf\x27\x1c"),
]


def _prime_mask(n: int) -> np.ndarray:
    if n <= 2:
        return np.zeros(n, dtype=bool)
    sieve = np.ones(n, dtype=bool)
    sieve[:2] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            sieve[i * i : n : i] = False
    return sieve


def _pack_bits(bits: np.ndarray, bitorder: str) -> bytes:
    max_bits = MAX_BYTES * 8
    if bits.size > max_bits:
        bits = bits[:max_bits]
    packed = np.packbits(bits, bitorder="little" if bitorder == "lsb" else "big")
    return packed.tobytes()


def _classify_bytes(tmp_path: Path, data: bytes) -> str:
    if not shutil.which("file"):
        return ""
    try:
        tmp_path.write_bytes(data)
    except Exception:
        return ""
    try:
        proc = subprocess.run(
            ["file", "-b", str(tmp_path)],
            capture_output=True,
            text=True,
            timeout=MAX_PENDING_TIME,
            check=False,
        )
    except Exception:
        return ""
    if proc.returncode != 0:
        return ""
    return proc.stdout.strip()


def _guess_extension(desc: str) -> str:
    lower = desc.lower()
    if "jpeg image" in lower:
        return ".jpg"
    if "png image" in lower:
        return ".png"
    if "zip archive" in lower:
        return ".zip"
    if "gzip compressed" in lower:
        return ".gz"
    if "bzip2 compressed" in lower:
        return ".bz2"
    if "7-zip archive" in lower:
        return ".7z"
    if "pdf" in lower:
        return ".pdf"
    if "openpgp" in lower:
        return ".pgp"
    if "sqlite" in lower:
        return ".sqlite"
    if "ms-dos executable" in lower:
        return ".exe"
    return ".bin"


def _guess_magic_ext(label: str) -> str:
    if label == "jpeg":
        return ".jpg"
    if label == "png":
        return ".png"
    if label == "zip":
        return ".zip"
    if label == "gzip":
        return ".gz"
    if label == "zlib":
        return ".zlib"
    if label == "pdf":
        return ".pdf"
    if label == "bmp":
        return ".bmp"
    if label == "7z":
        return ".7z"
    return ".bin"


def _sanitize_label(desc: str) -> str:
    label = desc.strip().split(",", 1)[0].strip().replace(" ", "_")
    return "".join(ch for ch in label if ch.isalnum() or ch in {"_", "-"}).lower()


def _find_magic_hits(stream: bytes) -> List[Tuple[str, int]]:
    hits = []
    for label, magic in MAGICS:
        offset = stream.find(magic)
        if offset != -1:
            hits.append((label, offset))
    return sorted(hits, key=lambda item: item[1])


def _carve_png(stream: bytes, offset: int) -> bytes | None:
    if offset + 8 >= len(stream):
        return None
    pos = offset + 8
    while pos + 12 <= len(stream):
        try:
            length = int.from_bytes(stream[pos : pos + 4], "big")
        except Exception:
            return None
        chunk_type = stream[pos + 4 : pos + 8]
        pos += 8 + length + 4
        if pos > len(stream):
            return None
        if chunk_type == b"IEND":
            return stream[offset:pos]
    return None


def _carve_bmp(stream: bytes, offset: int) -> bytes | None:
    if offset + 54 > len(stream):
        return None
    size = int.from_bytes(stream[offset + 2 : offset + 6], "little")
    if size <= 0 or offset + size > len(stream):
        return None
    return stream[offset : offset + size]


def _carve_jpeg(stream: bytes, offset: int) -> bytes | None:
    end = stream.find(b"\xff\xd9", offset)
    if end == -1:
        return None
    return stream[offset : end + 2]


def _extract_stream(
    arr: np.ndarray,
    order: List[int],
    bit: int,
    bitorder: str,
    traversal: str,
) -> bytes:
    if traversal == "yx":
        data_arr = arr.transpose(1, 0, 2)
        data_arr = data_arr[..., order]
        bits = ((data_arr >> bit) & 1).astype(np.uint8)
        return _pack_bits(bits.reshape(-1), bitorder)
    if traversal == "prime":
        flat = arr.reshape(-1, arr.shape[2])
        prime = _prime_mask(flat.shape[0])
        data_arr = flat[prime][:, order]
        bits = ((data_arr >> bit) & 1).astype(np.uint8)
        return _pack_bits(bits.reshape(-1), bitorder)
    data_arr = arr[..., order]
    bits = ((data_arr >> bit) & 1).astype(np.uint8)
    return _pack_bits(bits.reshape(-1), bitorder)


def analyze_plane_carver(input_img: Path, output_dir: Path) -> None:
    """Scan bit-planes for file-like payloads using simple signatures + file(1)."""
    try:
        img = Image.open(input_img)
        if img.mode not in {"RGB", "RGBA", "L"}:
            img = img.convert("RGBA")
        elif img.mode == "L":
            img = img.convert("L")
    except Exception as exc:
        update_data(output_dir, {"plane_carver": {"status": "error", "error": str(exc)}})
        return

    arr = np.array(img)
    if arr.ndim == 2:
        arr = arr[..., None]
    channels = arr.shape[2]

    if channels == 1:
        orders: List[Tuple[str, List[int]]] = [("l", [0])]
    elif channels == 3:
        orders = [
            ("rgb", [0, 1, 2]),
            ("bgr", [2, 1, 0]),
            ("r", [0]),
            ("g", [1]),
            ("b", [2]),
        ]
    else:
        orders = [
            ("rgba", [0, 1, 2, 3]),
            ("abgr", [3, 2, 1, 0]),
            ("rgb", [0, 1, 2]),
            ("bgr", [2, 1, 0]),
            ("r", [0]),
            ("g", [1]),
            ("b", [2]),
            ("a", [3]),
        ]

    traversal_modes = ["xy", "yx", "prime"]
    bitorders = ["lsb", "msb"]
    bits = [0, 1, 2]

    carver_dir = output_dir / "plane_carver"
    carver_dir.mkdir(parents=True, exist_ok=True)
    tmp_probe = carver_dir / "probe.bin"

    results = []
    secondary = []
    seen = set()

    for order_name, order in orders:
        for bit in bits:
            for bitorder in bitorders:
                for traversal in traversal_modes:
                    stream = _extract_stream(arr, order, bit, bitorder, traversal)
                    if not stream:
                        continue
                    hits = _find_magic_hits(stream)
                    if hits:
                        for label, offset in hits:
                            carve = None
                            if label == "png":
                                carve = _carve_png(stream, offset)
                            elif label == "jpeg":
                                carve = _carve_jpeg(stream, offset)
                            elif label == "bmp":
                                carve = _carve_bmp(stream, offset)
                            else:
                                carve = stream[offset:]
                            if not carve:
                                continue
                            key = (order_name, bit, bitorder, traversal, label, offset)
                            if key in seen:
                                continue
                            seen.add(key)
                            ext = _guess_magic_ext(label)
                            filename = (
                                f"{order_name}_b{bit}_{bitorder}_{traversal}_{label}_{offset}{ext}"
                            )
                            out_path = carver_dir / filename
                            try:
                                out_path.write_bytes(carve)
                            except Exception:
                                continue
                            note = f"truncated to {MAX_BYTES} bytes (sampled window)"
                            record = {
                                "order": order_name,
                                "bit": bit,
                                "bitorder": bitorder,
                                "traversal": traversal,
                                "type": label,
                                "offset": offset,
                                "file": str(out_path.relative_to(output_dir)),
                                "note": note,
                            }
                            if label in PRIORITY_LABELS:
                                results.append(record)
                            elif len(secondary) < MAX_SECONDARY:
                                secondary.append(record)
                            if len(results) >= MAX_CANDIDATES:
                                break
                    if len(results) >= MAX_CANDIDATES:
                        break
                    desc = _classify_bytes(tmp_probe, stream)
                    if not desc or desc.lower().endswith("data"):
                        continue
                    if "text" in desc.lower() and "openpgp" not in desc.lower():
                        continue
                    key = (order_name, bit, bitorder, traversal, desc)
                    if key in seen:
                        continue
                    seen.add(key)

                    label = _sanitize_label(desc) or "payload"
                    ext = _guess_extension(desc)
                    filename = f"{order_name}_b{bit}_{bitorder}_{traversal}_{label}{ext}"
                    out_path = carver_dir / filename
                    try:
                        out_path.write_bytes(stream)
                    except Exception:
                        continue

                    record = {
                        "order": order_name,
                        "bit": bit,
                        "bitorder": bitorder,
                        "traversal": traversal,
                        "type": desc,
                        "file": str(out_path.relative_to(output_dir)),
                        "note": f"truncated to {MAX_BYTES} bytes",
                    }
                    if "openpgp" in desc.lower():
                        results.append(record)
                    elif len(secondary) < MAX_SECONDARY:
                        secondary.append(record)
                    if len(results) >= MAX_CANDIDATES:
                        break
                if len(results) >= MAX_CANDIDATES:
                    break
            if len(results) >= MAX_CANDIDATES:
                break
        if len(results) >= MAX_CANDIDATES:
            break

    if results or secondary:
        if not results and secondary:
            results.extend(secondary)
        elif secondary:
            results.extend(secondary[: max(0, MAX_CANDIDATES - len(results))])
        archive = output_dir / "plane_carver.7z"
        try:
            subprocess.run(
                ["7z", "a", "-t7z", "-y", str(archive), "."],
                cwd=str(carver_dir),
                capture_output=True,
                text=True,
                timeout=MAX_PENDING_TIME,
                check=False,
            )
        except Exception:
            pass
        update_data(output_dir, {"plane_carver": {"status": "ok", "output": results}})
    else:
        update_data(
            output_dir,
            {
                "plane_carver": {
                    "status": "empty",
                    "output": "No file-like payloads detected in sampled planes.",
                }
            },
        )
