"""Color space conversion helpers (RGB <-> HSL/LAB)."""
from __future__ import annotations

import numpy as np


def rgb_to_hsl(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb = rgb.astype(np.float32) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = np.max(rgb, axis=2)
    minc = np.min(rgb, axis=2)
    l = (maxc + minc) / 2.0
    delta = maxc - minc

    s = np.zeros_like(l)
    nonzero = delta > 1e-6
    s[nonzero] = delta[nonzero] / (1.0 - np.abs(2.0 * l[nonzero] - 1.0))

    h = np.zeros_like(l)
    mask_r = nonzero & (maxc == r)
    mask_g = nonzero & (maxc == g)
    mask_b = nonzero & (maxc == b)
    h[mask_r] = ((g - b)[mask_r] / delta[mask_r]) % 6.0
    h[mask_g] = ((b - r)[mask_g] / delta[mask_g]) + 2.0
    h[mask_b] = ((r - g)[mask_b] / delta[mask_b]) + 4.0
    h = (h / 6.0) % 1.0

    return h, s, l


def hsl_to_rgb(h: np.ndarray, s: np.ndarray, l: np.ndarray) -> np.ndarray:
    c = (1.0 - np.abs(2.0 * l - 1.0)) * s
    h6 = (h * 6.0) % 6.0
    x = c * (1.0 - np.abs((h6 % 2.0) - 1.0))
    m = l - c / 2.0

    zeros = np.zeros_like(h)
    r1 = np.select(
        [h6 < 1, h6 < 2, h6 < 3, h6 < 4, h6 < 5, h6 >= 5],
        [c, x, zeros, zeros, x, c],
        default=zeros,
    )
    g1 = np.select(
        [h6 < 1, h6 < 2, h6 < 3, h6 < 4, h6 < 5, h6 >= 5],
        [x, c, c, x, zeros, zeros],
        default=zeros,
    )
    b1 = np.select(
        [h6 < 1, h6 < 2, h6 < 3, h6 < 4, h6 < 5, h6 >= 5],
        [zeros, zeros, x, c, c, x],
        default=zeros,
    )

    r = (r1 + m).clip(0, 1)
    g = (g1 + m).clip(0, 1)
    b = (b1 + m).clip(0, 1)
    rgb = np.stack([r, g, b], axis=2) * 255.0
    return rgb.astype(np.uint8)


def rgb_to_lab(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb = rgb.astype(np.float32) / 255.0
    mask = rgb > 0.04045
    rgb = np.where(mask, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)

    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    x = r * 0.4124 + g * 0.3576 + b * 0.1805
    y = r * 0.2126 + g * 0.7152 + b * 0.0722
    z = r * 0.0193 + g * 0.1192 + b * 0.9505

    x /= 0.95047
    z /= 1.08883

    def pivot(t: np.ndarray) -> np.ndarray:
        return np.where(t > 0.008856, np.cbrt(t), (7.787 * t) + (16.0 / 116.0))

    fx = pivot(x)
    fy = pivot(y)
    fz = pivot(z)

    l = (116.0 * fy) - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)

    return l, a, b


def lab_to_rgb(l: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    fy = (l + 16.0) / 116.0
    fx = fy + (a / 500.0)
    fz = fy - (b / 200.0)

    def inv_pivot(t: np.ndarray) -> np.ndarray:
        return np.where(t**3 > 0.008856, t**3, (t - 16.0 / 116.0) / 7.787)

    x = inv_pivot(fx) * 0.95047
    y = inv_pivot(fy)
    z = inv_pivot(fz) * 1.08883

    r = x * 3.2406 + y * -1.5372 + z * -0.4986
    g = x * -0.9689 + y * 1.8758 + z * 0.0415
    b = x * 0.0557 + y * -0.2040 + z * 1.0570

    rgb = np.stack([r, g, b], axis=2)
    rgb = np.clip(rgb, 0, 1)

    mask = rgb > 0.0031308
    rgb = np.where(mask, 1.055 * (rgb ** (1 / 2.4)) - 0.055, 12.92 * rgb)
    rgb = (rgb * 255.0).clip(0, 255)
    return rgb.astype(np.uint8)
