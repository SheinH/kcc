import numpy as np
from PIL import Image

# --- Gamma Conversion Lookup Table ---
_rgb_to_linear_lut_cache = None

def srgb_to_linear_lookup_table():
    """
    Returns a lookup table mapping sRGB (0-255) to linear light (0.0-1.0).
    """
    global _rgb_to_linear_lut_cache
    if _rgb_to_linear_lut_cache is not None:
        return _rgb_to_linear_lut_cache

    srgb = np.arange(256, dtype=np.float64) / 255.0
    linear = np.where(
        srgb <= 0.04045,
        srgb / 12.92,
        ((srgb + 0.055) / 1.055) ** 2.4
    )
    _rgb_to_linear_lut_cache = linear.astype(np.float32)
    return _rgb_to_linear_lut_cache

# --- Linear Grayscale Conversion ---

def rgb_to_grayscale_linear(rgb_lin):
    """
    Converts an (H,W,3) linear RGB array to an (H,W) linear grayscale array.
    """
    return (
        0.2126 * rgb_lin[..., 0] +
        0.7152 * rgb_lin[..., 1] +
        0.0722 * rgb_lin[..., 2]
    )

_noise_image = None

def get_noise_image(path="LDR_LLL1_0.png"):
    """
    Loads and caches a blue‑noise texture as an (H,W) float32 array in [0,1].
    Verifies that its mean is ~0.5 (needed for unbiased dithering).
    """
    global _noise_image
    if _noise_image is None:
        bn_pil = Image.open(path).convert("L")
        bn = np.asarray(bn_pil, dtype=np.float32) / 255.0
        mean = float(bn.mean())
        if abs(mean - 0.5) > 1e-3:
            raise ValueError(
                f"Blue‑noise texture mean={mean:.4f}, expected ≈0.5"
            )
        _noise_image = bn
    return _noise_image

# --- Dithering with luminance‑adaptive blue‑noise ---

def dither_to_gray_levels(
        img_input_srgb_pil: Image.Image, num_levels: int = 16):
    # Convert PIL image to numpy array
    img_srgb = np.asarray(img_input_srgb_pil, dtype=np.uint8)

    # Create mask for pure black and pure white pixels
    pure_black_mask = np.all(img_srgb == 0, axis=2)
    pure_white_mask = np.all(img_srgb == 255, axis=2)
    no_dither_mask = pure_black_mask | pure_white_mask

    # Convert to linear space using lookup table
    lut = srgb_to_linear_lookup_table()
    img_linear = lut[img_srgb]

    # Convert to grayscale in linear space
    gray_linear = rgb_to_grayscale_linear(img_linear)

    # Create palette evenly spaced in sRGB space
    palette_srgb = np.linspace(0, 1, num_levels, dtype=np.float32)

    # Convert sRGB palette to linear space
    palette_linear = np.where(
        palette_srgb <= 0.04045,
        palette_srgb / 12.92,
        ((palette_srgb + 0.055) / 1.055) ** 2.4
    )

    # Convert each pixel's linear luminance back to sRGB to find which interval it falls in
    gray_srgb = np.where(
        gray_linear <= 0.0031308,
        gray_linear * 12.92,
        1.055 * (gray_linear ** (1.0 / 2.4)) - 0.055
    )

    # Find which sRGB quantization interval each pixel falls into
    scaled_srgb = gray_srgb * (num_levels - 1)
    lower_level_idx = np.floor(scaled_srgb).astype(np.int32)
    lower_level_idx = np.clip(lower_level_idx, 0, num_levels - 2)
    upper_level_idx = lower_level_idx + 1

    # Get the linear luminance values for the levels below and above each pixel
    lower_luminance = palette_linear[lower_level_idx]
    upper_luminance = palette_linear[upper_level_idx]

    # Calculate the luminance difference between adjacent levels for each pixel
    level_difference = upper_luminance - lower_luminance

    # Get blue noise pattern
    noise = get_noise_image()
    h, w = gray_linear.shape

    # Tile the noise to match image dimensions
    noise_h, noise_w = noise.shape
    noise_tiled = np.tile(noise, (h // noise_h + 1, w // noise_w + 1))[:h, :w]

    # Scale noise from [0,1] to [-0.5, 0.5] for unbiased dithering
    noise_centered = noise_tiled - 0.5

    # Scale the noise by the level difference for each pixel
    scaled_noise = noise_centered * level_difference

    # Apply dithering only where not masked
    dithered = gray_linear.copy()
    dithered[~no_dither_mask] += scaled_noise[~no_dither_mask]

    # Clamp to [0,1]
    dithered = np.clip(dithered, 0.0, 1.0)

    # Convert dithered linear back to sRGB for quantization
    dithered_srgb = np.where(
        dithered <= 0.0031308,
        dithered * 12.92,
        1.055 * (dithered ** (1.0 / 2.4)) - 0.055
    )

    # Quantize in sRGB space
    level_indices = np.round(dithered_srgb * (num_levels - 1))

    # Convert palette back to sRGB for display
    palette_srgb_display = np.round(palette_srgb * 255).astype(np.uint8)

    # Create output image data
    data = level_indices.astype(np.uint8)

    # Create palette for PIL (RGB format)
    palette_rgb = np.repeat(
        palette_srgb_display[:, None], 3, axis=1
    ).flatten().tolist()
    out_pil = Image.fromarray(data, mode="P")
    out_pil.putpalette(palette_rgb)
    return out_pil
