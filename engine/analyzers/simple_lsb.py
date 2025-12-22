"""In-app LSB text extractor (no external binaries)."""

from pathlib import Path
from typing import Dict, List

from PIL import Image

from .utils import update_data


def _decode_plane(img: Image.Image, plane: str) -> str:
    """Decode text until a NULL byte from the given color plane ordering."""
    if not isinstance(img, Image.Image):
        raise TypeError(f"Expected PIL Image, got {type(img).__name__}")

    if not plane:
        raise ValueError("Plane string cannot be empty")

    bits: List[int] = []
    channels = {
        "R": 0,
        "G": 1,
        "B": 2,
        "A": 3,
    }

    try:
        for y in range(img.height):
            for x in range(img.width):
                pixel = img.getpixel((x, y))
                for ch in plane:
                    idx = channels.get(ch)
                    if idx is None or idx >= len(pixel):
                        continue
                    bits.append(pixel[idx] & 1)
    except Exception as e:
        raise ValueError(f"Failed to extract LSB bits from image: {str(e)}")

    try:
        chars: List[str] = []
        for i in range(0, len(bits), 8):
            byte_bits = bits[i : i + 8]
            if len(byte_bits) < 8:
                break
            val = int("".join(str(b) for b in byte_bits), 2)
            if val == 0:  # NULL terminator
                break
            # Skip non-printable characters to avoid errors
            if 32 <= val < 127 or val in (9, 10, 13):  # Printable ASCII + tab, newline, return
                chars.append(chr(val))
        return "".join(chars)
    except Exception as e:
        raise ValueError(f"Failed to decode bits to text: {str(e)}")


def analyze_simple_lsb(input_img: Path, output_dir: Path) -> None:
    """
    Try to recover plain-text payloads from common LSB planes without external tools.
    """
    if not input_img.exists():
        update_data(
            output_dir,
            {"simple_lsb": {"status": "error", "error": f"Input image not found: {input_img}"}}
        )
        return

    if not output_dir.exists():
        update_data(
            output_dir,
            {"simple_lsb": {"status": "error", "error": f"Output directory not found: {output_dir}"}}
        )
        return

    try:
        img = Image.open(input_img).convert("RGBA")
    except FileNotFoundError:
        update_data(
            output_dir,
            {"simple_lsb": {"status": "error", "error": f"Image file not found: {input_img}"}}
        )
        return
    except Exception as exc:
        update_data(
            output_dir,
            {"simple_lsb": {"status": "error", "error": f"Failed to open image: {str(exc)}"}}
        )
        return

    try:
        planes = ["RGB", "R", "G", "B", "A", "RGBA"]
        decoded: Dict[str, str] = {}
        for plane in planes:
            try:
                text = _decode_plane(img, plane)
                if text:
                    decoded[plane] = text
            except Exception as e:
                # Continue with other planes even if one fails
                print(f"Warning: Failed to decode plane {plane}: {str(e)}")
                continue

        status = "ok" if decoded else "empty"
        update_data(
            output_dir,
            {
                "simple_lsb": {
                    "status": status,
                    "decoded_text": decoded,
                }
            },
        )
    except Exception as exc:
        update_data(
            output_dir,
            {"simple_lsb": {"status": "error", "error": f"LSB analysis failed: {str(exc)}"}}
        )

