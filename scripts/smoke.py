#!/usr/bin/env python3
"""Quick smoke test: import app, verify tools, and multi-channel encode/decode sanity."""

import os
import sys
from pathlib import Path
import zlib

ROOT = Path(__file__).resolve().parent.parent
try:
    import engine  # type: ignore  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, str(ROOT))
    import engine  # type: ignore  # noqa: F401

from engine.encoder import encode_multi_channel
from engine.tooling import get_tool_status

CORE_TOOLS = ["binwalk", "foremost", "steghide", "zsteg", "exiftool", "strings", "7z", "file"]
OPTIONAL = ["outguess"]


def decode_channel_text(img_path: Path, channel: str) -> str:
    from PIL import Image

    img = Image.open(img_path).convert("RGBA")
    bits = []
    channel_idx = {"R": 0, "G": 1, "B": 2, "A": 3}[channel]
    for y in range(img.height):
        for x in range(img.width):
            val = img.getpixel((x, y))[channel_idx]
            bits.append(val & 1)
    chars = []
    for i in range(0, len(bits), 8):
        b = bits[i : i + 8]
        if len(b) < 8:
            break
        byte = int("".join(str(bit) for bit in b), 2)
        if byte == 0:
            break
        chars.append(chr(byte))
    return "".join(chars)


def decode_channel_zlib(img_path: Path, channel: str) -> bytes:
    from PIL import Image

    img = Image.open(img_path).convert("RGBA")
    bits = []
    channel_idx = {"R": 0, "G": 1, "B": 2, "A": 3}[channel]
    for y in range(img.height):
        for x in range(img.width):
            val = img.getpixel((x, y))[channel_idx]
            bits.append(val & 1)
    data = bytearray()
    expected_total = None
    for i in range(0, len(bits), 8):
        b = bits[i : i + 8]
        if len(b) < 8:
            break
        byte = int("".join(str(bit) for bit in b), 2)
        data.append(byte)
        if expected_total is None and len(data) >= 4:
            length = int.from_bytes(data[:4], "big")
            expected_total = length + 4
        if expected_total is not None and len(data) >= expected_total:
            break

    if expected_total is None or len(data) < 4:
        raise ValueError("Incomplete length header in channel payload")
    payload = bytes(data[4:expected_total])
    if len(payload) < expected_total - 4:
        raise ValueError("Incomplete payload in channel data")
    return zlib.decompress(payload)


def main() -> int:
    allow_missing = os.getenv("ALLOW_MISSING_TOOLS", "").lower() in {"1", "true", "yes"}
    status = get_tool_status()
    print("Tool availability:")
    missing = []
    for name, info in status.items():
        mark = "✅" if info.get("available") else "❌"
        path = info.get("path") or ""
        print(f"  {mark} {name} {path}")
        if name in CORE_TOOLS and not info.get("available"):
            missing.append(name)

    if missing:
        msg = f"Missing required tools: {', '.join(missing)}"
        if allow_missing:
            print(f"{msg} (continuing because ALLOW_MISSING_TOOLS=1)")
        else:
            print(msg)
            return 1

    missing_optional = [n for n in OPTIONAL if not status.get(n, {}).get("available")]
    if missing_optional:
        print(f"Optional tools missing: {', '.join(missing_optional)} (deep analysis may be skipped)")

    # Multi-channel encode/decode sanity
    from PIL import Image

    img = Image.new("RGBA", (64, 64), color=(200, 200, 200, 255))
    img_path = ROOT / "tmp_smoke_cover.png"
    img.save(img_path)

    channels = {
        "R": {"enabled": True, "type": "text", "text": "hello R"},
        "G": {"enabled": True, "type": "text", "text": "hello G"},
        "B": {"enabled": True, "type": "file", "file_data": b"meow"},
        "A": {"enabled": False},
    }
    name, data = encode_multi_channel(img_path.read_bytes(), channels, filename="cover.png")
    out = ROOT / "tmp_smoke_encoded.png"
    out.write_bytes(data)

    assert decode_channel_text(out, "R") == "hello R"
    assert decode_channel_text(out, "G") == "hello G"
    assert decode_channel_zlib(out, "B") == b"meow"
    print("Multi-channel encode/decode sanity passed.")

    print("Smoke test passed: core tools available and modules importable.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
