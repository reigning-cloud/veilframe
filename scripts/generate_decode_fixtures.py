from __future__ import annotations

import math
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image, PngImagePlugin

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "tests" / "fixtures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DCT_COORDS = [(2, 3), (3, 2), (3, 4), (4, 3), (4, 4)]


def bits_from_bytes(data: bytes) -> List[int]:
    bits = [int(b) for b in f"{len(data):016b}"]
    for byte in data:
        bits.extend(int(b) for b in f"{byte:08b}")
    return bits


def dct_matrix(n: int = 8) -> np.ndarray:
    mat = np.zeros((n, n), dtype=np.float32)
    factor = math.pi / (2 * n)
    for k in range(n):
        alpha = math.sqrt(1 / n) if k == 0 else math.sqrt(2 / n)
        for i in range(n):
            mat[k, i] = alpha * math.cos((2 * i + 1) * k * factor)
    return mat


def encode_lsb(path: Path, payload: bytes) -> None:
    rng = np.random.default_rng(42)
    arr = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
    bits = bits_from_bytes(payload)
    flat = arr.reshape(-1, 3)
    for idx, bit in enumerate(bits):
        channel = idx % 3
        flat[idx // 3, channel] = (flat[idx // 3, channel] & 0xFE) | bit
    img = Image.fromarray(arr, "RGB")
    img.save(path)


def encode_pvd(path: Path, payload: bytes) -> None:
    base = np.full((64, 64), 128, dtype=np.uint8)
    bits = bits_from_bytes(payload)
    ranges: List[Tuple[int, int, int]] = [
        (0, 7, 3),
        (8, 15, 3),
        (16, 31, 4),
        (32, 63, 5),
        (64, 127, 6),
    ]
    flat = base.reshape(-1)
    bit_idx = 0
    for i in range(0, len(flat) - 1, 2):
        if bit_idx >= len(bits):
            break
        for low, high, width in ranges:
            remaining = bits[bit_idx : bit_idx + width]
            if not remaining:
                value = 0
            else:
                value = 0
                for b in remaining:
                    value = (value << 1) | b
                if len(remaining) < width:
                    value <<= (width - len(remaining))
                bit_idx += len(remaining)
            diff = low + value
            if flat[i] >= flat[i + 1]:
                flat[i + 1] = np.clip(flat[i] - diff, 0, 255)
            else:
                flat[i + 1] = np.clip(flat[i] + diff, 0, 255)
            break
    img = Image.fromarray(base, "L")
    img.save(path)


def encode_palette(path: Path, payload: bytes) -> None:
    rng = np.random.default_rng(7)
    indices = rng.integers(0, 255, size=(64, 64), dtype=np.uint8)
    bits = bits_from_bytes(payload)
    flat = indices.reshape(-1)
    for idx, bit in enumerate(bits):
        flat[idx] = (flat[idx] & 0xFE) | bit
    img = Image.fromarray(indices, "P")
    palette = []
    for i in range(256):
        palette.extend([i, i, i])
    img.putpalette(palette)
    img.save(path)


def encode_chroma(path: Path, payload: bytes) -> None:
    rng = np.random.default_rng(3)
    arr = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
    img = Image.fromarray(arr, "RGB").convert("YCbCr")
    ycbcr = np.array(img)
    bits = bits_from_bytes(payload)
    bit_pos = 3
    mask = 1 << bit_pos
    repeat = 7
    expanded: List[int] = []
    for bit in bits:
        expanded.extend([bit] * repeat)
    for idx, bit in enumerate(expanded):
        channel = 1 if idx % 2 == 0 else 2
        flat = ycbcr[:, :, channel].reshape(-1)
        flat[idx // 2] = (flat[idx // 2] & ~mask) | (bit << bit_pos)
    out = Image.fromarray(ycbcr, "YCbCr").convert("RGB")
    out.save(path)


def encode_png_chunks(path: Path) -> None:
    base = Image.new("RGB", (32, 32), color=(15, 20, 30))
    info = PngImagePlugin.PngInfo()
    info.add_text("comment", "PNG_CHUNK_OK")
    info.add_itxt("note", "PNG_ITXT_OK")
    base.save(path, pnginfo=info)


def encode_spread_spectrum(path: Path, payload: bytes, password: str) -> None:
    rng = np.random.default_rng(11)
    arr = rng.integers(0, 255, size=(64, 64), dtype=np.uint8)
    bits = bits_from_bytes(payload)
    flat = arr.reshape(-1).astype(np.int32)
    prng = random.Random(password)
    chip_length = 32
    alpha = 40
    for bit in bits:
        idxs = prng.sample(range(len(flat)), chip_length)
        seq = [1 if prng.random() > 0.5 else -1 for _ in range(chip_length)]
        delta = alpha if bit == 1 else -alpha
        for j, idx in enumerate(idxs):
            flat[idx] = int(np.clip(flat[idx] + delta * seq[j], 0, 255))
    out = flat.reshape(arr.shape).astype(np.uint8)
    Image.fromarray(out, "L").save(path)


def encode_dct(path: Path, payload: bytes) -> None:
    rng = np.random.default_rng(19)
    arr = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
    img = Image.fromarray(arr, "RGB").convert("YCbCr")
    ycbcr = np.array(img)
    y = ycbcr[:, :, 0].astype(np.float32) - 128.0
    dct_mat = dct_matrix(8)
    idct_mat = dct_mat.T
    bits = bits_from_bytes(payload)
    bit_idx = 0
    for y0 in range(0, 64, 8):
        for x0 in range(0, 64, 8):
            block = y[y0 : y0 + 8, x0 : x0 + 8]
            coeff = dct_mat @ block @ idct_mat
            for u, v in DCT_COORDS:
                if bit_idx >= len(bits):
                    break
                coeff[u, v] = 30 if bits[bit_idx] else -30
                bit_idx += 1
            block = idct_mat @ coeff @ dct_mat
            y[y0 : y0 + 8, x0 : x0 + 8] = block
            if bit_idx >= len(bits):
                break
        if bit_idx >= len(bits):
            break
    y = np.clip(y + 128.0, 0, 255).astype(np.uint8)
    ycbcr[:, :, 0] = y
    out = Image.fromarray(ycbcr, "YCbCr").convert("RGB")
    out.save(path, format="JPEG", quality=100, subsampling=0, optimize=False)


def encode_f5(path: Path, payload: bytes) -> None:
    rng = np.random.default_rng(23)
    arr = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
    img = Image.fromarray(arr, "RGB").convert("YCbCr")
    ycbcr = np.array(img)
    y = ycbcr[:, :, 0].astype(np.float32) - 128.0
    dct_mat = dct_matrix(8)
    idct_mat = dct_mat.T
    bits = bits_from_bytes(payload)
    k = 2
    n = (1 << k) - 1

    coeffs = []
    coords = []
    for y0 in range(0, 64, 8):
        for x0 in range(0, 64, 8):
            block = y[y0 : y0 + 8, x0 : x0 + 8]
            coeff = dct_mat @ block @ idct_mat
            coeffs.append(coeff)
            coords.append((y0, x0))

    flat = []
    locations = []
    for b_idx, coeff in enumerate(coeffs):
        for u in range(8):
            for v in range(8):
                if u == 0 and v == 0:
                    continue
                val = coeff[u, v]
                if abs(val) < 5:
                    coeff[u, v] = 5 if val >= 0 else -5
                flat.append((b_idx, u, v))

    bit_idx = 0
    group_idx = 0
    while bit_idx + k <= len(bits) and group_idx + n <= len(flat):
        group = flat[group_idx : group_idx + n]
        group_idx += n
        target = 0
        for b in bits[bit_idx : bit_idx + k]:
            target = (target << 1) | b
        bit_idx += k

        syndrome = 0
        for i, (b_idx, u, v) in enumerate(group, start=1):
            val = coeffs[b_idx][u, v]
            if val >= 0:
                syndrome ^= i
        if syndrome != target:
            flip_idx = syndrome ^ target
            if 1 <= flip_idx <= n:
                b_idx, u, v = group[flip_idx - 1]
                coeffs[b_idx][u, v] = -coeffs[b_idx][u, v]

    for coeff, (y0, x0) in zip(coeffs, coords):
        block = idct_mat @ coeff @ dct_mat
        y[y0 : y0 + 8, x0 : x0 + 8] = block

    y = np.clip(y + 128.0, 0, 255).astype(np.uint8)
    ycbcr[:, :, 0] = y
    out = Image.fromarray(ycbcr, "YCbCr").convert("RGB")
    out.save(path, format="JPEG", quality=100, subsampling=0, optimize=False)


if __name__ == "__main__":
    encode_lsb(OUT_DIR / "lsb.png", b"LSB_OK")
    encode_pvd(OUT_DIR / "pvd.png", b"PVD_OK")
    encode_palette(OUT_DIR / "palette.png", b"PAL_OK")
    encode_chroma(OUT_DIR / "chroma.png", b"CHROMA_OK")
    encode_png_chunks(OUT_DIR / "png_chunks.png")
    encode_spread_spectrum(OUT_DIR / "spread.png", b"SPREAD_OK", "veilframe")
    encode_dct(OUT_DIR / "dct.jpg", b"DCT_OK")
    encode_f5(OUT_DIR / "f5.jpg", b"F5_OK")
    print(f"Wrote fixtures to {OUT_DIR}")
