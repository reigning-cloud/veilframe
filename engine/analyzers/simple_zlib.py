"""Attempt to recover zlib-embedded payloads without external tools."""

import base64
import zlib
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

from .utils import update_data

MAX_BYTES = 2 * 1024 * 1024  # safety cap for in-memory extraction


def _read_bytes(img: Image.Image, plane: str, max_bytes: int = MAX_BYTES) -> bytes:
    """Read LSBs in the given plane order into bytes (row-major)."""
    if not isinstance(img, Image.Image):
        raise TypeError(f"Expected PIL Image, got {type(img).__name__}")

    if not plane:
        raise ValueError("Plane string cannot be empty")

    if max_bytes <= 0:
        raise ValueError(f"max_bytes must be positive, got {max_bytes}")

    channels = [ch for ch in plane if ch in "RGBA"]
    channel_map = {"R": 0, "G": 1, "B": 2, "A": 3}

    if not channels:
        return b""

    try:
        arr = np.array(img)
        indices = [channel_map[ch] for ch in channels]
        bits = (arr[..., indices] & 1).reshape(-1)
    except Exception as e:
        raise ValueError(f"Failed to extract bits from image: {str(e)}")

    try:
        max_bits = max_bytes * 8
        if bits.size > max_bits:
            bits = bits[:max_bits]
        usable = (bits.size // 8) * 8
        if usable == 0:
            return b""
        bits = bits[:usable]
        data = np.packbits(bits, bitorder="big").tobytes()
        if len(data) > max_bytes:
            return data[:max_bytes]
        return data
    except Exception as e:
        raise ValueError(f"Failed to convert bits to bytes: {str(e)}")


def _try_decompress(blob: bytes) -> Tuple[bytes, bool]:
    """
    Try to decompress, allowing trailing data. Returns (payload, success).
    """
    if not isinstance(blob, bytes):
        return b"", False

    if not blob:
        return b"", False

    obj = zlib.decompressobj()
    try:
        payload = obj.decompress(blob)
        # Accept if stream ended; unused_data is fine.
        return (payload, obj.eof)
    except zlib.error as e:
        # Zlib-specific errors are expected when data is not compressed
        return b"", False
    except Exception as e:
        # Log unexpected errors but still return False
        print(f"Warning: Unexpected error during zlib decompression: {str(e)}")
        return b"", False


def analyze_simple_zlib(input_img: Path, output_dir: Path) -> None:
    """
    Try to recover zlib-compressed payloads from common plane orderings.
    """
    if not input_img.exists():
        update_data(
            output_dir,
            {"simple_zlib": {"status": "error", "error": f"Input image not found: {input_img}"}}
        )
        return

    if not output_dir.exists():
        update_data(
            output_dir,
            {"simple_zlib": {"status": "error", "error": f"Output directory not found: {output_dir}"}}
        )
        return

    try:
        img = Image.open(input_img).convert("RGBA")
    except FileNotFoundError:
        update_data(
            output_dir,
            {"simple_zlib": {"status": "error", "error": f"Image file not found: {input_img}"}}
        )
        return
    except Exception as exc:
        update_data(
            output_dir,
            {"simple_zlib": {"status": "error", "error": f"Failed to open image: {str(exc)}"}}
        )
        return

    try:
        planes = ["RGB", "R", "G", "B", "A", "RGBA"]
        matches: List[Dict[str, str]] = []

        for plane in planes:
            try:
                data_full = _read_bytes(img, plane)
                candidates: List[Tuple[str, bytes]] = [("full", data_full)]

                # Try trimming at the first explicit NULL (legacy terminator), if present.
                if b"\x00" in data_full:
                    trim = data_full.split(b"\x00", 1)[0]
                    candidates.append(("trim_null", trim))

                for label, blob in candidates:
                    payload, ok = _try_decompress(blob)
                    if not ok:
                        continue

                    preview = ""
                    try:
                        preview = payload[:200].decode("utf-8")
                    except UnicodeDecodeError:
                        # Binary data - show base64 preview
                        preview = base64.b64encode(payload[:60]).decode()
                    except Exception as e:
                        preview = f"<preview error: {str(e)}>"

                    try:
                        matches.append(
                            {
                                "plane": plane,
                                "strategy": label,
                                "size_bytes": str(len(payload)),
                                "preview": preview,
                                "base64": base64.b64encode(payload).decode(),
                            }
                        )
                    except Exception as e:
                        print(f"Warning: Failed to encode payload for plane {plane}: {str(e)}")
                        continue

                    break  # prefer first success per plane
            except Exception as e:
                # Continue with other planes even if one fails
                print(f"Warning: Failed to process plane {plane}: {str(e)}")
                continue

        status = "ok" if matches else "empty"
        update_data(
            output_dir,
            {"simple_zlib": {"status": status, "matches": matches}},
        )
    except Exception as exc:
        update_data(
            output_dir,
            {"simple_zlib": {"status": "error", "error": f"Zlib analysis failed: {str(exc)}"}}
        )
