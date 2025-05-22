import functools
from typing import Tuple

import numpy as np
from PIL import Image

############################################################
#  Blue‑noise dithering – linear‑light, palette‑cached      #
#  Supports arbitrary gray levels and optional TPDF noise.  #
############################################################

##############################################################################
# 1.  sRGB ↔ linear helpers
##############################################################################

_rgb_to_linear_lut: np.ndarray | None = None


def srgb_to_linear_lut() -> np.ndarray:
    """Return / memoize a 256‑entry LUT mapping 8‑bit sRGB → linear [0,1]."""
    global _rgb_to_linear_lut
    if _rgb_to_linear_lut is None:
        srgb = np.arange(256, dtype=np.float64) / 255.0
        linear = np.where(
            srgb <= 0.04045,
            srgb / 12.92,
            ((srgb + 0.055) / 1.055) ** 2.4,
        )
        _rgb_to_linear_lut = linear.astype(np.float32)
    return _rgb_to_linear_lut


def rgb_to_grayscale_linear(rgb_lin: np.ndarray) -> np.ndarray:
    """ITU‑R BT.709 luminance (linear domain). rgb_lin shape = (H,W,3)."""
    return (
        0.2126 * rgb_lin[..., 0] +
        0.7152 * rgb_lin[..., 1] +
        0.0722 * rgb_lin[..., 2]
    )

##############################################################################
# 2.  Blue‑noise loader (verifies zero‑mean)
##############################################################################

_noise_tile: np.ndarray | None = None


def get_noise_tile(path: str = "LDR_LLL1_0.png") -> np.ndarray:
    """Load a blue‑noise image, cache it, ensure mean ≈ 0.5."""
    global _noise_tile
    if _noise_tile is None:
        tile = Image.open(path).convert("L")
        tile = np.asarray(tile, dtype=np.float32) / 255.0
        if abs(tile.mean() - 0.5) > 1e-3:
            raise ValueError(
                f"Blue‑noise mean {tile.mean():.4f}; expected ≈0.5 for unbiased dithering."
            )
        _noise_tile = tile
    return _noise_tile

##############################################################################
# 3.  Palette memoisation
##############################################################################

_palette_cache: dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


def get_gray_palette(num_levels: int, lut: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (palette_srgb, palette_lin, half_steps) – memoised by num_levels."""
    if num_levels in _palette_cache:
        return _palette_cache[num_levels]

    palette_srgb = np.linspace(0, 255, num_levels, dtype=np.uint8)
    palette_lin = lut[palette_srgb]
    half_steps = (palette_lin[1:] - palette_lin[:-1]) * 0.5
    _palette_cache[num_levels] = (palette_srgb, palette_lin, half_steps)
    return _palette_cache[num_levels]

##############################################################################
# 4.  Main dithering routine
##############################################################################

def dither_to_gray_levels(
    img_srgb: Image.Image,
    num_levels: int = 16,
    use_tpdf_noise: bool = False,
) -> Image.Image:
    """Return a paletted PIL image dithered to `num_levels` gray levels.

    * Linear‑light fidelity via sRGB → linear LUT.
    * Noise amplitude = local half‑step (perceptual weighting).
    * Pure 0/255 pixels bypass dithering (keeps solid blacks/whites).
    * Blue‑noise is optional TPDF (sum of two shifted tiles).
    """
    lut = srgb_to_linear_lut()
    palette_srgb, palette_lin, half_steps = get_gray_palette(num_levels, lut)

    # ---------- load source image ----------
    src = np.asarray(img_srgb, dtype=np.uint8)
    H, W, _ = src.shape

    # Masks for solid extremes (no dithering needed)
    is_black = np.all(src == 0, axis=2)
    is_white = np.all(src == 255, axis=2)
    needs_dither = ~(is_black | is_white)

    # Convert only the pixels that need processing
    lin_rgb = lut[src]
    gray_lin = rgb_to_grayscale_linear(lin_rgb)

    # ---------- noise tile ----------
    tile = get_noise_tile()
    y = np.arange(H)[:, None] % tile.shape[0]
    x = np.arange(W)[None, :] % tile.shape[1]
    if use_tpdf_noise:
        noise = 0.5 * (
            tile[y, x] +
            tile[(y + 17) % tile.shape[0], (x + 31) % tile.shape[1]]
        )
    else:
        noise = tile[y, x]

    # ---------- interval lookup ----------
    idx = np.searchsorted(palette_lin, gray_lin, side="right") - 1
    idx = np.clip(idx, 0, num_levels - 2)
    half = half_steps[idx]

    # ---------- add noise (brightness‑preserving) ----------
    dithered = gray_lin.copy()
    dithered[needs_dither] += (noise[needs_dither] - 0.5) * 2.0 * half[needs_dither]
    np.clip(dithered, 0.0, 1.0, out=dithered)

    # ---------- quantise ----------
    out_idx = np.empty((H, W), dtype=np.uint8)
    out_idx[is_black] = 0
    out_idx[is_white] = num_levels - 1

    work_mask = needs_dither
    if work_mask.any():
        diff = np.abs(palette_lin[None, None, :] - dithered[..., None])
        out_idx[work_mask] = np.argmin(diff, axis=2)[work_mask].astype(np.uint8)

    # ---------- pack into paletted image ----------
    pal_rgb = np.repeat(palette_srgb[:, None], 3, axis=1).flatten().tolist()
    out = Image.fromarray(out_idx, mode="P")
    out.putpalette(pal_rgb)
    return out

##############################################################################
# 5.  Convenience: LRU‑cached wrapper (keeps most recent palette)            #
##############################################################################

@functools.lru_cache(maxsize=4)
def dithering_wrapper(path: str, levels: int = 16) -> Image.Image:
    """Example helper that caches most‑recent palettes & results."""
    return dither_to_gray_levels(Image.open(path).convert("RGB"), num_levels=levels)
