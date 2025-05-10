import numpy as np
from PIL import Image

# --- Gamma Conversion Lookup Table ---
# This global variable will cache the lookup table once computed.
# It's populated by srgb_to_linear_lookup_table().
rgb_to_linear_lut_cache = None


def srgb_to_linear_lookup_table():
    """
    Computes and returns a lookup table mapping sRGB color values (0-255)
    to linear floating point light values (0.0-1.0).
    Caches the table globally for efficiency on subsequent calls.

    The sRGB to linear conversion formula is:
    - L = S / 12.92, if S <= 0.04045
    - L = ((S + 0.055) / 1.055) ** 2.4, if S > 0.04045
    where S is the normalized sRGB value (sRGB_value / 255.0).

    Returns:
        numpy.ndarray: A 256-element array where the index represents
                       the sRGB value (0-255) and the value at that
                       index is the corresponding linear light value.
    """
    global rgb_to_linear_lut_cache
    if rgb_to_linear_lut_cache is not None:
        return rgb_to_linear_lut_cache

    srgb_int_values = np.arange(256, dtype=np.float32)
    s_values = srgb_int_values / 255.0
    linear_values = np.zeros_like(s_values, dtype=np.float32)

    condition1 = s_values <= 0.04045
    condition2 = ~condition1

    linear_values[condition1] = s_values[condition1] / 12.92
    linear_values[condition2] = ((s_values[condition2] + 0.055) / 1.055) ** 2.4

    rgb_to_linear_lut_cache = linear_values
    return linear_values


# --- RGB to Grayscale Conversion ---
def rgb_to_grayscale_linear(rgb_linear_array_0_1):
    """
    Converts an RGB linear NumPy array (0-1) to linear grayscale (luminance).
    Uses standard Rec. 709 luminance coefficients.
    """
    # Coefficients for Rec. 709 luminance
    # Y = 0.2126 R + 0.7152 G + 0.0722 B
    r = rgb_linear_array_0_1[..., 0]
    g = rgb_linear_array_0_1[..., 1]
    b = rgb_linear_array_0_1[..., 2]
    grayscale_linear = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return grayscale_linear


# --- Dithering Function ---
def dither_to_16_gray_levels(img_input_srgb_pil,num_levels=16):
    """
    Dithers an RGB input image to a specified number of gray levels
    using a blue noise texture, operating in linear color space
    and using a lookup table for sRGB to linear conversion. (Vectorized)

    Args:
        input_image_path (str): Path to the input RGB image file.
        blue_noise_image_path (str): Path to the blue noise texture file.
                                     Expected to be a grayscale image.
        num_levels (int): The number of output gray levels (default is 16).

    Returns:
        PIL.Image.Image: The dithered grayscale image (sRGB).
                         Returns None if an error occurs.
    """
    try:
        img_input_srgb_pil.convert('RGB')
        # Initialize the sRGB to Linear LUT
        srgb_to_linear_lut = srgb_to_linear_lookup_table()

        # 1. Load Images
        img_input_srgb_0_255 = np.array(img_input_srgb_pil, dtype=np.uint8)

        bn_img_pil = Image.open("LDR_LLL1_0.png").convert("L")
        bn_array_0_1 = np.array(bn_img_pil, dtype=np.float32) / 255.0

        img_height, img_width, _ = img_input_srgb_0_255.shape
        bn_height, bn_width = bn_array_0_1.shape

        # 2. Create Palettes
        palette_srgb_gray_0_255 = np.round(np.linspace(0, 255, num_levels)).astype(np.uint8)
        palette_linear_gray_0_1 = srgb_to_linear_lut[palette_srgb_gray_0_255].astype(np.float32)

        # 3. Convert Input to Linear Grayscale using LUT
        img_input_linear_rgb_0_1 = srgb_to_linear_lut[img_input_srgb_0_255]
        img_linear_gray_0_1 = rgb_to_grayscale_linear(img_input_linear_rgb_0_1)

        if img_linear_gray_0_1.ndim == 3 and img_linear_gray_0_1.shape[-1] == 1:
            img_linear_gray_0_1 = img_linear_gray_0_1.squeeze(axis=-1)
        elif img_linear_gray_0_1.ndim != 2:
            raise ValueError(f"Grayscale conversion resulted in unexpected dimensions: {img_linear_gray_0_1.shape}")

        # 4. Dither Strength in Linear Space
        dither_strength_linear = 1.0 / float(num_levels)

        # 5. Vectorized Dithering
        # Create tiled blue noise map matching image dimensions
        # y_indices will be (img_height, 1), x_indices will be (1, img_width)
        # Broadcasting will make bn_array_0_1[y_indices, x_indices] -> (img_height, img_width)
        y_indices = np.arange(img_height)[:, np.newaxis] % bn_height
        x_indices = np.arange(img_width)[np.newaxis, :] % bn_width
        bn_map_tiled_0_1 = bn_array_0_1[y_indices, x_indices]

        # Apply dither formula to the entire image array
        dithered_val_linear_array = img_linear_gray_0_1 + \
                                    (bn_map_tiled_0_1 - 0.5) * dither_strength_linear

        # Clip the dithered values
        dithered_val_linear_array = np.clip(dithered_val_linear_array, 0.0, 1.0)

        # Find closest linear palette color index for all pixels
        # Expand dimensions for broadcasting:
        # dithered_val_linear_array becomes (H, W, 1)
        # palette_linear_gray_0_1 becomes (1, 1, num_levels)
        # distances_all becomes (H, W, num_levels)
        dithered_expanded = dithered_val_linear_array[..., np.newaxis]
        palette_expanded = palette_linear_gray_0_1[np.newaxis, np.newaxis, :]

        distances_all = np.abs(palette_expanded - dithered_expanded)
        best_palette_indices_array = np.argmin(distances_all, axis=2).astype(np.uint8)  # Shape (H, W)

        raw_palette_list = []
        for gray_value in palette_srgb_gray_0_255:
            raw_palette_list.extend([gray_value, gray_value, gray_value])

        # PIL's putpalette will pad with zeros if the list is shorter than 768 entries.

        output_image_pil = Image.fromarray(best_palette_indices_array, mode="P")
        output_image_pil.putpalette(raw_palette_list)

        return output_image_pil

    except FileNotFoundError:
        print(f"Error: One of the image files was not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None