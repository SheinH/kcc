# -*- coding: utf-8 -*-
#
# Copyright (C) 2010  Alex Yatskov
# Copyright (C) 2011  Stanislav (proDOOMman) Kosolapov <prodoomman@gmail.com>
# Copyright (c) 2016  Alberto Planas <aplanas@gmail.com>
# Copyright (c) 2012-2014 Ciro Mattia Gonano <ciromattia@gmail.com>
# Copyright (c) 2013-2019 Pawel Jastrzebski <pawelj@iosphe.re>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
import io
import os
from pathlib import Path
import mozjpeg_lossless_optimization
from PIL import Image, ImageOps, ImageStat, ImageChops, ImageFilter

from .dither import dither_to_16_gray_levels
from .page_number_crop_alg import get_bbox_crop_margin_page_number, get_bbox_crop_margin
import png
from .inter_panel_crop_alg import crop_empty_inter_panel
import numpy as np
import ctypes
import time

AUTO_CROP_THRESHOLD = 0.015


class ProfileData:
    def __init__(self):
        pass

    Palette4 = [
        0x00, 0x00, 0x00,
        0x55, 0x55, 0x55,
        0xaa, 0xaa, 0xaa,
        0xff, 0xff, 0xff
    ]

    Palette15 = [
        0x00, 0x00, 0x00,
        0x11, 0x11, 0x11,
        0x22, 0x22, 0x22,
        0x33, 0x33, 0x33,
        0x44, 0x44, 0x44,
        0x55, 0x55, 0x55,
        0x66, 0x66, 0x66,
        0x77, 0x77, 0x77,
        0x88, 0x88, 0x88,
        0x99, 0x99, 0x99,
        0xaa, 0xaa, 0xaa,
        0xbb, 0xbb, 0xbb,
        0xcc, 0xcc, 0xcc,
        0xdd, 0xdd, 0xdd,
        0xff, 0xff, 0xff,
    ]

    Palette16 = [
        0x00, 0x00, 0x00,
        0x11, 0x11, 0x11,
        0x22, 0x22, 0x22,
        0x33, 0x33, 0x33,
        0x44, 0x44, 0x44,
        0x55, 0x55, 0x55,
        0x66, 0x66, 0x66,
        0x77, 0x77, 0x77,
        0x88, 0x88, 0x88,
        0x99, 0x99, 0x99,
        0xaa, 0xaa, 0xaa,
        0xbb, 0xbb, 0xbb,
        0xcc, 0xcc, 0xcc,
        0xdd, 0xdd, 0xdd,
        0xee, 0xee, 0xee,
        0xff, 0xff, 0xff,
    ]

    PalleteNull = [
    ]

    ProfilesKindleEBOK = {
        'K1': ("Kindle 1", (600, 670), Palette4, 1.8),
        'K2': ("Kindle 2", (600, 670), Palette15, 1.8),
        'KDX': ("Kindle DX/DXG", (824, 1000), Palette16, 1.8),
        'K34': ("Kindle Keyboard/Touch", (600, 800), Palette16, 1.8),
        'K578': ("Kindle", (600, 800), Palette16, 1.8),
        'KPW': ("Kindle Paperwhite 1/2", (758, 1024), Palette16, 1.8),
        'KV': ("Kindle Paperwhite 3/4/Voyage/Oasis", (1072, 1448), Palette16, 1.8),
    }

    ProfilesKindlePDOC = {
        'KO': ("Kindle Oasis 2/3/Paperwhite 12/Colorsoft 12", (1264, 1680), Palette16, 1.8),
        'K11': ("Kindle 11", (1072, 1448), Palette16, 1.8),
        'KPW5': ("Kindle Paperwhite 5/Signature Edition", (1236, 1648), Palette16, 1.8),
        'KS': ("Kindle Scribe", (1860, 2480), Palette16, 1.8),
    }

    ProfilesKindle = {
        **ProfilesKindleEBOK,
        **ProfilesKindlePDOC
    }

    ProfilesKobo = {
        'KoMT': ("Kobo Mini/Touch", (600, 800), Palette16, 1.8),
        'KoG': ("Kobo Glo", (768, 1024), Palette16, 1.8),
        'KoGHD': ("Kobo Glo HD", (1072, 1448), Palette16, 1.8),
        'KoA': ("Kobo Aura", (758, 1024), Palette16, 1.8),
        'KoAHD': ("Kobo Aura HD", (1080, 1440), Palette16, 1.8),
        'KoAH2O': ("Kobo Aura H2O", (1080, 1430), Palette16, 1.8),
        'KoAO': ("Kobo Aura ONE", (1404, 1872), Palette16, 1.8),
        'KoN': ("Kobo Nia", (758, 1024), Palette16, 1.8),
        'KoC': ("Kobo Clara HD/Kobo Clara 2E", (1072, 1448), Palette16, 1.8),
        'KoCC': ("Kobo Clara Colour", (1072, 1448), Palette16, 1.8),
        'KoL': ("Kobo Libra H2O/Kobo Libra 2", (1264, 1680), Palette16, 1.8),
        'KoLC': ("Kobo Libra Colour", (1264, 1680), Palette16, 1.8),
        'KoF': ("Kobo Forma", (1440, 1920), Palette16, 1.8),
        'KoS': ("Kobo Sage", (1440, 1920), Palette16, 1.8),
        'KoE': ("Kobo Elipsa", (1404, 1872), Palette16, 1.8),
    }

    ProfilesRemarkable = {
        'Rmk1': ("reMarkable 1", (1404, 1872), Palette16, 1.8),
        'Rmk2': ("reMarkable 2", (1404, 1872), Palette16, 1.8),
        'RmkPP': ("reMarkable Paper Pro", (1620, 2160), Palette16, 1.8),
    }

    Profiles = {
        **ProfilesKindle,
        **ProfilesKobo,
        **ProfilesRemarkable,
        'OTHER': ("Other", (0, 0), Palette16, 1.8),
    }

_libname = "../libatkinson.dylib"  # or "libatkinson.dylib" / "atkinson.dll"
_libpath = os.path.join(os.path.dirname(__file__), _libname)
lib = ctypes.CDLL(_libpath)

# 2) Tell ctypes about the function signature
lib.atkinson_dither.argtypes = [
    ctypes.POINTER(ctypes.c_uint8),  # in_rgb
    ctypes.POINTER(ctypes.c_uint8),  # out_idx
    ctypes.POINTER(ctypes.c_uint8),  # palette
    ctypes.c_int,  # width
    ctypes.c_int,  # height
    ctypes.c_int   # n_colors
]
lib.atkinson_dither.restype = None

lib.jjn_dither.argtypes = [
    ctypes.POINTER(ctypes.c_uint8),  # in_rgb
    ctypes.POINTER(ctypes.c_uint8),  # out_idx
    ctypes.POINTER(ctypes.c_uint8),  # palette
    ctypes.c_int,  # width
    ctypes.c_int,  # height
    ctypes.c_int   # n_colors
]
lib.atkinson_dither.restype = None


rgb_to_linear = None
def srgb_to_linear_lookup_table():
    """
    Computes a lookup table mapping sRGB color values (0-255)
    to linear floating point light values.

    The sRGB to linear conversion formula is:
    - L = S / 12.92, if S <= 0.04045
    - L = ((S + 0.055) / 1.055) ** 2.4, if S > 0.04045
    where S is the normalized sRGB value (sRGB_value / 255.0).

    Returns:
        numpy.ndarray: A 256-element array where the index represents
                       index is the corresponding linear light value.
    """
    global rgb_to_linear
    if rgb_to_linear:
        return rgb_to_linear
    # Create an array of sRGB integer values from 0 to 255
    srgb_int_values = np.arange(256)

    # Normalize sRGB values to the range [0, 1]
    s_values = srgb_int_values / 255.0

    # Initialize an empty array for linear values
    linear_values = np.zeros_like(s_values, dtype=float)

    # Condition for the first part of the formula
    condition1 = s_values <= 0.04045

    # Apply the first part of the formula
    linear_values[condition1] = s_values[condition1] / 12.92

    # Condition for the second part of the formula (implicitly where condition1 is False)
    condition2 = ~condition1 # or s_values > 0.04045

    # Apply the second part of the formula
    linear_values[condition2] = ((s_values[condition2] + 0.055) / 1.055) ** 2.4
    rgb_to_linear = linear_values

    return linear_values



def atkinson_quantize(img: Image.Image, pal: Image.Image) -> Image.Image:
    """
    img : RGB Pillow Image
    pal : mode 'P' palette image, with <=16 colors defined
    """
    # ensure modes
    img = img.convert("RGB")
    pal = pal.convert("P")
    # get raw pixel data
    W, H = img.size
    in_arr = np.frombuffer(img.tobytes(), dtype=np.uint8)
    # load palette (first n_colors*3 bytes)
    raw_pal = pal.palette.palette  # 768 bytes max
    # detect how many entries are actually used:
    # you can set n_colors manually if you know it
    # here we assume <=16
    n_colors = 16
    pal_arr = np.frombuffer(raw_pal[:n_colors*3], dtype=np.uint8)

    # prepare output index buffer
    out_idx = np.zeros((H*W,), dtype=np.uint8)

    # call into C++
    lib.atkinson_dither(
        in_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        out_idx.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        pal_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        W,
        H,
        n_colors
    )

    # wrap back into a P‑mode Image
    quant = Image.fromarray(out_idx.reshape((H, W)), mode="P")
    quant.putpalette(raw_pal)  # attach full 768-byte palette
    return quant

def jjn_quantize(img: Image.Image, pal: Image.Image) -> Image.Image:
    """
    img : RGB Pillow Image
    pal : mode 'P' palette image, with <=16 colors defined
    """
    # ensure modes
    img = img.convert("RGB")
    pal = pal.convert("P")
    # get raw pixel data
    W, H = img.size
    in_arr = np.frombuffer(img.tobytes(), dtype=np.uint8)
    # load palette (first n_colors*3 bytes)
    raw_pal = pal.palette.palette  # 768 bytes max
    # detect how many entries are actually used:
    # you can set n_colors manually if you know it
    # here we assume <=16
    n_colors = 16
    pal_arr = np.frombuffer(raw_pal[:n_colors*3], dtype=np.uint8)

    # prepare output index buffer
    out_idx = np.zeros((H*W,), dtype=np.uint8)

    # call into C++
    lib.jjn_dither(
        in_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        out_idx.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        pal_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        W,
        H,
        n_colors
    )

    # wrap back into a P‑mode Image
    quant = Image.fromarray(out_idx.reshape((H, W)), mode="P")
    quant.putpalette(raw_pal)  # attach full 768-byte palette
    return quant

def _atkinson_dither_paletted_numpy(rgb_array: np.ndarray, palette_array: np.ndarray) -> np.ndarray:
    """
    Core Atkinson dithering logic operating on NumPy arrays.
    rgb_array:     Input image as an (H, W, 3) NumPy array of float32.
    palette_array: Palette as an (P, 3) NumPy array of float32.
    returns:       (H, W) NumPy array of uint8 palette indices.
    """
    H, W = rgb_array.shape[:2]
    # P = palette_array.shape[0] # Number of colors in palette, not directly used in this optimized version of loop

    # Error buffer, initialized to zeros.
    # It stores the error propagated from previous pixels.
    err = np.zeros_like(rgb_array, dtype=np.float32)

    # Output array for palette indices.
    idxs = np.zeros((H, W), dtype=np.uint8)

    # Iterate over each row of the image.
    for y in range(H):
        # Current row's pixel values plus accumulated error from previous calculations.
        row = rgb_array[y] + err[y]
        # Clip values to be within the valid 0-255 range.
        np.clip(row, 0, 255, out=row)

        # --- Find nearest palette color for each pixel in the row ---
        # `row` is (W, 3). `palette_array` is (P, 3).
        # We want to find, for each pixel in `row`, which of the P palette colors is closest.
        # `palette_array[None, :, :]` reshapes palette to (1, P, 3).
        # `row[:, None, :]` reshapes row to (W, 1, 3).
        # Broadcasting subtraction: `diffs` becomes (W, P, 3), storing R,G,B differences.
        diffs = palette_array[None, :, :] - row[:, None, :]
        # Square the differences and sum along the color axis (axis=2) to get squared Euclidean distance.
        # `d2` shape is (W, P), storing squared distance of each pixel to each palette color.
        d2 = (diffs * diffs).sum(axis=2)

        # For each pixel, find the index of the palette color with the minimum distance.
        # `nearest` shape is (W,), containing palette indices for the current row.
        nearest_indices = np.argmin(d2, axis=1)
        idxs[y] = nearest_indices.astype(np.uint8)  # Store the palette indices

        # Get the actual color values for the new row from the palette.
        # `new_row` shape is (W, 3).
        new_row_colors = palette_array[nearest_indices]

        # --- Calculate quantization error ---
        # This is the difference between the (original + diffused error) color and the chosen palette color.
        # The error is divided by 8 for Atkinson dithering.
        qerr = (row - new_row_colors) / 8.0  # Shape (W, 3)

        # --- Diffuse the error to neighboring pixels ---
        # Current row, pixels to the right
        if W > 1:  # x+1
            err[y, 1:] += qerr[:-1]
        if W > 2:  # x+2
            err[y, 2:] += qerr[:-2]

        # Next row (y+1)
        if y + 1 < H:
            if W > 1:  # x-1 (on next row)
                err[y + 1, :-1] += qerr[1:]
            err[y + 1, :] += qerr  # x (on next row)
            if W > 1:  # x+1 (on next row)
                err[y + 1, 1:] += qerr[:-1]

        # Row after next (y+2)
        if y + 2 < H:
            err[y + 2, :] += qerr  # x (on row y+2)

    return idxs


# This is the wrapper function that KCC will call
def atkinson_dither_paletted(im: Image.Image, palette: list) -> Image.Image:
    """
    Applies Atkinson dithering to an image using a specified palette.
    im:      Pillow Image object (any mode, will be converted to RGB).
    palette: Can be a list of (R,G,B) tuples OR a flat list of R,G,B values.
             Up to 256 colors.
    returns: A P-mode Pillow Image, dithered with Atkinson, using the provided palette.
    """
    # This print statement can be useful for debugging.
    # print("Atkinson dithering called!")

    # --- 1. Prepare NumPy arrays from inputs ---
    # Convert the input Pillow image to an RGB NumPy array of float32.
    # Numba works best with NumPy arrays.
    rgb_np = np.asarray(im.convert('RGB'), dtype=np.float32)

    # Convert the input palette to a NumPy array of float32.
    # The palette can be a list of tuples or a flat list.
    # This code block normalizes it to an (P, 3) array.
    pal_np = np.array(palette, dtype=np.float32)
    if pal_np.ndim == 1:  # If palette is a flat list [r,g,b,r,g,b,...]
        if pal_np.shape[0] % 3 == 0 and pal_np.shape[0] <= 256 * 3:
            pal_np = pal_np.reshape(-1, 3)  # Reshape to (P, 3)
        else:
            raise ValueError(
                "Flat palette must have a number of elements divisible by 3 and represent at most 256 colors.")
    elif pal_np.ndim == 2:  # If palette is already list of lists/tuples [[r,g,b],...]
        if pal_np.shape[1] != 3 or pal_np.shape[0] > 256:
            raise ValueError("2D palette must have 3 columns (R,G,B) and at most 256 colors.")
    else:
        raise ValueError("Palette format is not recognized. Must be convertible to (P,3) NumPy array.")

    # --- 2. Call the Numba-optimized dithering function ---
    # This function performs the core dithering logic.
    idxs_np = _atkinson_dither_paletted_numpy(rgb_np, pal_np)

    # --- 3. Pack the result into a P-mode Pillow Image ---
    # `idxs_np` contains the palette indices for each pixel.
    # `Image.fromarray` creates a Pillow image from the NumPy array.
    out_image = Image.fromarray(idxs_np, mode='P')

    # Prepare the palette for `out_image.putpalette()`.
    # It needs a flat list of integers: [r1,g1,b1,r2,g2,b2,..., 0,0,0,...] up to 256*3=768 values.
    P_colors = pal_np.shape[0]  # Number of actual colors in the palette

    # Take the first P_colors from pal_np, convert to uint8, flatten, and convert to list.
    # This ensures we are using the processed palette data correctly.
    flat_palette_values = pal_np[:P_colors].astype(np.uint8).flatten().tolist()

    # Create padding if the palette has fewer than 256 colors.
    padding = [0, 0, 0] * (256 - P_colors)

    # Combine the actual palette values with the padding.
    final_palette_for_image = flat_palette_values + padding

    # Apply the palette to the output image.
    out_image.putpalette(final_palette_for_image)

    return out_image


class ComicPageParser:
    def __init__(self, source, options):
        Image.MAX_IMAGE_PIXELS = int(2048 * 2048 * 2048 // 4 // 3)
        self.opt = options
        self.source = source
        self.size = self.opt.profileData[1]
        self.payload = []

        # Detect corruption in source image, let caller catch any exceptions triggered.
        srcImgPath = os.path.join(source[0], source[1])
        self.image = Image.open(srcImgPath)
        self.image.verify()
        self.image = Image.open(srcImgPath).convert('RGB')

        self.color = self.colorCheck()
        self.fill = self.fillCheck()
        # backwards compatibility for Pillow >9.1.0
        if not hasattr(Image, 'Resampling'):
            Image.Resampling = Image
        self.splitCheck()

    def getImageHistogram(self, image):
        histogram = image.histogram()
        if histogram[0] == 0:
            return -1
        elif histogram[255] == 0:
            return 1
        else:
            return 0

    def splitCheck(self):
        width, height = self.image.size
        dstwidth, dstheight = self.size
        if self.opt.maximizestrips:
            leftbox = (0, 0, int(width / 2), height)
            rightbox = (int(width / 2), 0, width, height)
            if self.opt.righttoleft:
                pageone = self.image.crop(rightbox)
                pagetwo = self.image.crop(leftbox)
            else:
                pageone = self.image.crop(leftbox)
                pagetwo = self.image.crop(rightbox)
            new_image = Image.new("RGB", (int(width / 2), int(height*2)))
            new_image.paste(pageone, (0, 0))
            new_image.paste(pagetwo, (0, height))
            self.payload.append(['N', self.source, new_image, self.color, self.fill])
        elif (width > height) != (dstwidth > dstheight) and width <= dstheight and height <= dstwidth \
                and not self.opt.webtoon and self.opt.splitter == 1:
            spread = self.image
            if not self.opt.norotate:
                spread = spread.rotate(90, Image.Resampling.BICUBIC, True)
            self.payload.append(['R', self.source, spread, self.color, self.fill])
        elif (width > height) != (dstwidth > dstheight) and not self.opt.webtoon:
            if self.opt.splitter != 1:
                if width > height:
                    leftbox = (0, 0, int(width / 2), height)
                    rightbox = (int(width / 2), 0, width, height)
                else:
                    leftbox = (0, 0, width, int(height / 2))
                    rightbox = (0, int(height / 2), width, height)
                if self.opt.righttoleft:
                    pageone = self.image.crop(rightbox)
                    pagetwo = self.image.crop(leftbox)
                else:
                    pageone = self.image.crop(leftbox)
                    pagetwo = self.image.crop(rightbox)
                self.payload.append(['S1', self.source, pageone, self.color, self.fill])
                self.payload.append(['S2', self.source, pagetwo, self.color, self.fill])
            if self.opt.splitter > 0:
                spread = self.image
                if not self.opt.norotate:
                    spread = spread.rotate(90, Image.Resampling.BICUBIC, True)
                self.payload.append(['R', self.source, spread,
                                    self.color, self.fill])
        else:
            self.payload.append(['N', self.source, self.image, self.color, self.fill])

    def colorCheck(self):
        if self.opt.webtoon:
            return True
        else:
            img = self.image.copy()
            bands = img.getbands()
            if bands == ('R', 'G', 'B') or bands == ('R', 'G', 'B', 'A'):
                thumb = img.resize((40, 40))
                SSE, bias = 0, [0, 0, 0]
                bias = ImageStat.Stat(thumb).mean[:3]
                bias = [b - sum(bias) / 3 for b in bias]
                for pixel in thumb.getdata():
                    mu = sum(pixel) / 3
                    SSE += sum((pixel[i] - mu - bias[i]) * (pixel[i] - mu - bias[i]) for i in [0, 1, 2])
                MSE = float(SSE) / (40 * 40)
                if MSE > 22:
                    return True
                else:
                    return False
            else:
                return False

    def fillCheck(self):
        if self.opt.bordersColor:
            return self.opt.bordersColor
        else:
            bw = self.image.convert('L').point(lambda x: 0 if x < 128 else 255, '1')
            imageBoxA = bw.getbbox()
            imageBoxB = ImageChops.invert(bw).getbbox()
            if imageBoxA is None or imageBoxB is None:
                surfaceB, surfaceW = 0, 0
                diff = 0
            else:
                surfaceB = (imageBoxA[2] - imageBoxA[0]) * (imageBoxA[3] - imageBoxA[1])
                surfaceW = (imageBoxB[2] - imageBoxB[0]) * (imageBoxB[3] - imageBoxB[1])
                diff = ((max(surfaceB, surfaceW) - min(surfaceB, surfaceW)) / min(surfaceB, surfaceW)) * 100
            if diff > 0.5:
                if surfaceW < surfaceB:
                    return 'white'
                elif surfaceW > surfaceB:
                    return 'black'
            else:
                fill = 0
                startY = 0
                while startY < bw.size[1]:
                    if startY + 5 > bw.size[1]:
                        startY = bw.size[1] - 5
                    fill += self.getImageHistogram(bw.crop((0, startY, bw.size[0], startY + 5)))
                    startY += 5
                startX = 0
                while startX < bw.size[0]:
                    if startX + 5 > bw.size[0]:
                        startX = bw.size[0] - 5
                    fill += self.getImageHistogram(bw.crop((startX, 0, startX + 5, bw.size[1])))
                    startX += 5
                if fill > 0:
                    return 'black'
                else:
                    return 'white'


class ComicPage:
    def __init__(self, options, mode, path, image, color, fill):
        self.opt = options
        _, self.size, self.palette, self.gamma = self.opt.profileData
        if self.opt.hq:
            self.size = (int(self.size[0] * 1.5), int(self.size[1] * 1.5))
        self.kindle_scribe_azw3 = (options.profile == 'KS') and (options.format in ('MOBI', 'EPUB'))
        self.image = image
        self.color = color
        self.fill = fill
        self.rotated = False
        self.orgPath = os.path.join(path[0], path[1])
        if 'N' in mode:
            self.targetPath = os.path.join(path[0], os.path.splitext(path[1])[0]) + '-kcc'
        elif 'R' in mode:
            self.targetPath = os.path.join(path[0], os.path.splitext(path[1])[0]) + '-kcc-a'
            if not options.norotate:
                self.rotated = True
        elif 'S1' in mode:
            self.targetPath = os.path.join(path[0], os.path.splitext(path[1])[0]) + '-kcc-b'
        elif 'S2' in mode:
            self.targetPath = os.path.join(path[0], os.path.splitext(path[1])[0]) + '-kcc-c'
        # backwards compatibility for Pillow >9.1.0
        if not hasattr(Image, 'Resampling'):
            Image.Resampling = Image

    def saveToDir(self):
        try:
            flags = []
            if not self.opt.forcecolor and not self.opt.forcepng:
                self.image = self.image.convert('L')
            if self.rotated:
                flags.append('Rotated')
            if self.fill != 'white':
                flags.append('BlackBackground')
            if self.opt.forcepng:
                img_for_pypng = self.image  # Start with the current state of self.image

                # self.palette is the palette object used by the quantizeImage() method for this page,
                # taken from self.opt.profileData.
                # ProfileData.Palette16 is our specific target palette for the 4-bit PNG output.

                if img_for_pypng.mode == 'P' and self.palette == ProfileData.Palette16:
                    self.image.info["transparency"] = None
                    self.targetPath += '.png'

                    self.image.save(self.targetPath, 'PNG', optimize=1)
                else:

                    width = img_for_pypng.width
                    height = img_for_pypng.height

                    pypng_target_palette_tuples = [tuple(ProfileData.Palette16[i:i + 3])
                                                   for i in range(0, len(ProfileData.Palette16), 3)]

                    raw_pixel_data = list(img_for_pypng.getdata())  # This is your "python array" of indices
                    pixels_by_row = [raw_pixel_data[i * width:(i + 1) * width]
                                     for i in range(height)]

                    # Now, write using pypng
                    with open(self.targetPath + ".png", 'wb') as f:
                        writer = png.Writer(
                            width,
                            height,
                            palette=pypng_target_palette_tuples,
                            bitdepth=4  # Explicitly 4-bit
                        )
                        writer.write(f, pixels_by_row)  # "save that shit" :)

            else:
                self.targetPath += '.jpg'
                if self.opt.mozjpeg:
                    with io.BytesIO() as output:
                        self.image.save(output, format="JPEG", optimize=1, quality=85)
                        input_jpeg_bytes = output.getvalue()
                        output_jpeg_bytes = mozjpeg_lossless_optimization.optimize(input_jpeg_bytes)
                        with open(self.targetPath, "wb") as output_jpeg_file:
                            output_jpeg_file.write(output_jpeg_bytes)
                else:
                    self.image.save(self.targetPath, 'JPEG', optimize=1, quality=85)
            if os.path.isfile(self.orgPath):
                os.remove(self.orgPath)
            return [Path(self.targetPath).name, flags]
        except IOError as err:
            raise RuntimeError('Cannot save image. ' + str(err))

    def autocontrastImage(self):
        gamma = self.opt.gamma
        if gamma < 0.1:
            gamma = self.gamma
            if self.gamma != 1.0 and self.color:
                gamma = 1.0
        if gamma == 1.0:
            self.image = ImageOps.autocontrast(self.image)
        else:
            self.image = ImageOps.autocontrast(Image.eval(self.image, lambda a: int(255 * (a / 255.) ** gamma)))


    def quantizeImage(self):
        colors = len(self.palette) // 3
        if colors < 256:
            self.palette += self.palette[:3] * (256 - colors)
        palImg = Image.new('P', (1, 1))
        palImg.putpalette(self.palette)
        # self.image = self.image.convert('L')
        # self.image = self.image.convert('RGB')
        # Quantize is deprecated but new function call it internally anyway...
        # self.image = self.image.quantize(palette=palImg)
        # self.image = atkinson_dither_paletted(self.image,self.palette)
        start_time = time.perf_counter()
        print(self.palette)
        # if len(self.palette) == 16:
        self.image = dither_to_16_gray_levels(self.image)
        # else:
        #     self.image = jjn_quantize(self.image, palImg)
        end_time = time.perf_counter()

        print(f"Execution time: {end_time - start_time:.6f} seconds")

    def optimizeForDisplay(self, reducerainbow):
        # Reduce rainbow artifacts for grayscale images by breaking up dither patterns that cause Moire interference with color filter array
        if reducerainbow and not self.color:
            unsharpFilter = ImageFilter.UnsharpMask(radius=1, percent=100)
            self.image = self.image.filter(unsharpFilter)
            self.image = self.image.filter(ImageFilter.BoxBlur(1.0))
            self.image = self.image.filter(unsharpFilter)

    def resizeImage(self):
        # kindle scribe conversion to mobi is limited in resolution by kindlegen, same with send to kindle and epub
        if self.kindle_scribe_azw3:
            self.size = (1440, 1920)
        ratio_device = float(self.size[1]) / float(self.size[0])
        ratio_image = float(self.image.size[1]) / float(self.image.size[0])
        method = self.resize_method()
        if self.opt.stretch:
            self.image = self.image.resize(self.size, method)
        elif method == Image.Resampling.BICUBIC and not self.opt.upscale:
            if self.opt.format == 'CBZ' or self.opt.kfx:
                borderw = int((self.size[0] - self.image.size[0]) / 2)
                borderh = int((self.size[1] - self.image.size[1]) / 2)
                self.image = ImageOps.expand(self.image, border=(borderw, borderh), fill=self.fill)
                if self.image.size[0] != self.size[0] or self.image.size[1] != self.size[1]:
                    self.image = ImageOps.pad(self.image, self.size, method=method)
        else:  # if image bigger than device resolution or smaller with upscaling
            if self.rotated or self.under_crop_minimum:
                self.image = ImageOps.pad(self.image, self.size, method=method,color="white")
            elif abs(ratio_image - ratio_device) < AUTO_CROP_THRESHOLD:
                self.image = ImageOps.pad(self.image, self.size, method=method,color="white")
            elif self.opt.format == 'CBZ' or self.opt.kfx:
                self.image = ImageOps.pad(self.image, self.size, method=method,color="white")
            else:
                if self.kindle_scribe_azw3:
                    self.size = (1860, 1920)
                self.image = ImageOps.pad(self.image, self.size, method=method,color="white")

    def resize_method(self):
        return Image.Resampling.LANCZOS

    def maybeCrop(self, box, minimum):
        w, h = self.image.size
        left, upper, right, lower = box
        if self.opt.preservemargin:
            ratio = 1 - self.opt.preservemargin / 100
            box = left * ratio, upper * ratio, right + (w - right) * (1 - ratio), lower + (h - lower) * (1 - ratio)
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        image_area = self.image.size[0] * self.image.size[1]
        if (box_area / image_area) >= minimum:
            self.image = self.image.crop(box)
        else:
            self.under_crop_minimum = True

    def cropPageNumber(self, power, minimum):
        bbox = get_bbox_crop_margin_page_number(self.image, power, self.fill)
        
        if bbox:
            self.maybeCrop(bbox, minimum)

    def cropMargin(self, power, minimum):
        bbox = get_bbox_crop_margin(self.image, power, self.fill)
        
        if bbox:
            self.maybeCrop(bbox, minimum)

    def cropInterPanelEmptySections(self, direction):
        self.image = crop_empty_inter_panel(self.image, direction, background_color=self.fill)

class Cover:
    def __init__(self, source, target, opt, tomeid):
        self.options = opt
        self.source = source
        self.target = target
        if tomeid == 0:
            self.tomeid = 1
        else:
            self.tomeid = tomeid
        self.image = Image.open(source)
        # backwards compatibility for Pillow >9.1.0
        if not hasattr(Image, 'Resampling'):
            Image.Resampling = Image
        self.process()

    def process(self):
        self.image = self.image.convert('RGB')
        self.image = ImageOps.autocontrast(self.image)
        if not self.options.forcecolor:
            self.image = self.image.convert('L')
        w, h = self.image.size
        if w / h > 2:
            if self.options.righttoleft:
                self.image = self.image.crop((w/6, 0, w/2 - w * 0.02, h))
            else:
                self.image = self.image.crop((w/2 + w * 0.02, 0, 5/6 * w, h))
        elif w / h > 1.3:
            if self.options.righttoleft:
                self.image = self.image.crop((0, 0, w/2 - w * 0.03, h))
            else:
                self.image = self.image.crop((w/2 + w * 0.03, 0, w, h))
        self.image.thumbnail(self.options.profileData[1], Image.Resampling.LANCZOS)
        self.save()

    def save(self):
        try:
            self.image.save(self.target, "JPEG", optimize=1, quality=85)
        except IOError:
            raise RuntimeError('Failed to save cover.')

    def saveToKindle(self, kindle, asin):
        self.image = self.image.resize((300, 470), Image.Resampling.LANCZOS)
        try:
            self.image.save(os.path.join(kindle.path.split('documents')[0], 'system', 'thumbnails',
                                         'thumbnail_' + asin + '_EBOK_portrait.jpg'), 'JPEG', optimize=1, quality=85)
        except IOError:
            raise RuntimeError('Failed to upload cover.')
