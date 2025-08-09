import argparse
import json
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np
from tqdm import tqdm

# ---------- sRGB (0..255) -> CIE L*a*b (vectorized) ----------
def srgb_to_lab(arr_rgb):
    """
    arr_rgb: array shape (..., 3), dtype uint8 or 0..255 ints
    returns: array shape (..., 3), dtype float (L*, a*, b*)
    """
    rgb = arr_rgb.astype('float32') / 255.0

    # linearize sRGB
    mask = rgb <= 0.04045
    rgb_lin = np.empty_like(rgb, dtype='float32')
    rgb_lin[mask] = rgb[mask] / 12.92
    rgb_lin[~mask] = ((rgb[~mask] + 0.055) / 1.055) ** 2.4

    # linear RGB -> XYZ (D65)
    M = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ], dtype='float32')
    xyz = np.dot(rgb_lin, M.T) * 100.0  # scale to [0..100]

    # Reference white D65
    Xn, Yn, Zn = 95.047, 100.000, 108.883
    x = xyz[..., 0] / Xn
    y = xyz[..., 1] / Yn
    z = xyz[..., 2] / Zn

    eps = 0.008856  # 216/24389
    k = 903.3

    fx = np.where(x > eps, np.cbrt(x), (k * x + 16.0) / 116.0)
    fy = np.where(y > eps, np.cbrt(y), (k * y + 16.0) / 116.0)
    fz = np.where(z > eps, np.cbrt(z), (k * z + 16.0) / 116.0)

    L = (116.0 * fy) - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)

    lab = np.stack([L, a, b], axis=-1)
    return lab


# ---------- Vectorized CIEDE2000 (ΔE00) ----------
def delta_e_ciede2000(lab1, lab2):
    """
    lab1: (N,3)  L*,a*,b*
    lab2: (M,3)
    returns: (N,M) matrix of deltaE00 between every lab1[i] and lab2[j]
    """
    # Expand dims for broadcasting: (N,1,3) and (1,M,3) -> (N,M,3)
    L1 = lab1[:, None, 0]
    a1 = lab1[:, None, 1]
    b1 = lab1[:, None, 2]
    L2 = lab2[None, :, 0]
    a2 = lab2[None, :, 1]
    b2 = lab2[None, :, 2]

    # Step 1: C'
    C1 = np.sqrt(a1**2 + b1**2)
    C2 = np.sqrt(a2**2 + b2**2)
    C_bar = (C1 + C2) / 2.0

    # Step 2: G factor
    C_bar7 = C_bar ** 7
    G = 0.5 * (1 - np.sqrt(C_bar7 / (C_bar7 + (25.0**7) + 1e-30)))

    # Step 3: a' and C'
    a1p = (1 + G) * a1
    a2p = (1 + G) * a2
    C1p = np.sqrt(a1p**2 + b1**2)
    C2p = np.sqrt(a2p**2 + b2**2)

    # Step 4: h' (in degrees 0..360)
    def hp_fun(a_prime, b):
        h = np.degrees(np.arctan2(b, a_prime))
        h = np.where(h < 0, h + 360.0, h)
        return h

    h1p = hp_fun(a1p, b1)
    h2p = hp_fun(a2p, b2)

    # Step 5: delta L', delta C'
    dLp = L2 - L1
    dCp = C2p - C1p

    # Step 6: delta h' (in degrees)
    # If C1p*C2p == 0, delta_hp = 0
    zero_mask = (C1p * C2p) == 0
    dh = h2p - h1p
    dh_mod = dh.copy()

    # Bring dh into range -360..360 then adjust
    dh_mod = (dh + 360.0) % 360.0
    dh_mod = np.where(dh_mod > 180.0, dh_mod - 360.0, dh_mod)

    dh_final = np.where(zero_mask, 0.0, dh_mod)

    # Step 7: delta H'
    dHp = 2.0 * np.sqrt(C1p * C2p + 1e-30) * np.sin(np.radians(dh_final / 2.0))

    # Step 8: average L', C', h'
    L_bar_p = (L1 + L2) / 2.0
    C_bar_p = (C1p + C2p) / 2.0

    # h_bar': if zero -> h1p+h2p ; else see difference
    h_sum = h1p + h2p
    h_diff = np.abs(h1p - h2p)
    h_bar_p = np.where(zero_mask, h_sum, (h_sum / 2.0))

    # For large hue differences (>180), adjust
    cond = (C1p * C2p != 0) & (h_diff > 180.0)
    h_bar_p = np.where(cond & ((h1p + h2p) < 360.0), (h_sum + 360.0) / 2.0, h_bar_p)
    h_bar_p = np.where(cond & ((h1p + h2p) >= 360.0), (h_sum - 360.0) / 2.0, h_bar_p)

    # Step 9: T
    T = (1
         - 0.17 * np.cos(np.radians(h_bar_p - 30.0))
         + 0.24 * np.cos(np.radians(2.0 * h_bar_p))
         + 0.32 * np.cos(np.radians(3.0 * h_bar_p + 6.0))
         - 0.20 * np.cos(np.radians(4.0 * h_bar_p - 63.0))
         )

    # Step 10: delta theta
    delta_theta = 30.0 * np.exp(-(((h_bar_p - 275.0) / 25.0) ** 2.0))

    # Step 11: R_C
    C_bar_p7 = C_bar_p ** 7
    R_C = 2.0 * np.sqrt(C_bar_p7 / (C_bar_p7 + (25.0**7) + 1e-30))

    # Step 12: S_L, S_C, S_H
    S_L = 1.0 + ((0.015 * ((L_bar_p - 50.0) ** 2.0)) / np.sqrt(20.0 + ((L_bar_p - 50.0) ** 2.0)))
    S_C = 1.0 + 0.045 * C_bar_p
    S_H = 1.0 + 0.015 * C_bar_p * T

    # Step 13: R_T
    R_T = -np.sin(np.radians(2.0 * delta_theta)) * R_C

    # Step 14: final ΔE₀₀
    kL = kC = kH = 1.0
    termL = (dLp / (kL * S_L)) ** 2.0
    termC = (dCp / (kC * S_C)) ** 2.0
    termH = (dHp / (kH * S_H)) ** 2.0
    deltaE = np.sqrt(termL + termC + termH + R_T * (dCp / (kC * S_C)) * (dHp / (kH * S_H)))

    # deltaE shape: (N, M)
    return deltaE


def quantize_image(img_path, output_path, palette):
    # Load image and palette
    img = Image.open(img_path).convert('RGB')
    data = np.array(img)                # shape (H, W, 3), uint8
    pixels = data.reshape(-1, 3)        # shape (N, 3)

    pal = np.array(palette)  # (P, 3)

    # Convert to Lab
    lab_pixels = srgb_to_lab(pixels)    # (N, 3)
    lab_pal = srgb_to_lab(pal)          # (P, 3)

    # Compute pairwise ΔE00 distances (N, P)
    distances = delta_e_ciede2000(lab_pixels, lab_pal)

    # nearest palette index per pixel
    nearest = np.argmin(distances, axis=1)  # (N,)

    # Map back to palette RGB and reshape to image
    quantized = pal[nearest].reshape(data.shape).astype(np.uint8)
    Image.fromarray(quantized).save(output_path)


def split_and_upscale(img_path, tiles_dir, tile_w, tile_h, pixel_scale):
    img = Image.open(img_path)
    w, h = img.size
    tiles_dir.mkdir(parents=True, exist_ok=True)

    num_cols = (w + tile_w - 1) // tile_w
    num_rows = (h + tile_h - 1) // tile_h

    for row in range(num_rows):
        for col in range(num_cols):
            # box
            left = col*tile_w
            top = row * tile_h
            right = left + tile_w
            bottom = top + tile_h

            box = (left, top, min(right,w), min(bottom,h))
            tile = img.crop(box)

            tile_img = Image.new('RGB', (tile_w, tile_h), (0,0,0))
            tile_img.paste(tile, (0,0))

            up_w, up_h = tile_w * pixel_scale, tile_h * pixel_scale
            up_tile = tile_img.resize((up_w, up_h), Image.NEAREST)
            
            # Draw grid lines
            draw = ImageDraw.Draw(up_tile)
            for x in range(0, up_w + 1, pixel_scale):
                draw.line(((x, 0), (x, up_h)), fill=(0, 0, 0))
            for y in range(0, up_h + 1, pixel_scale):
                draw.line(((0, y), (up_w, y)), fill=(0, 0, 0))

            tile_name = f"tile_r{row}_c{col}.png"
            up_tile.save(tiles_dir / tile_name)
    
    return tiles_dir


def load_palette(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    if not all(isinstance(c, list) and len(c) == 3 for c in data):
        raise ValueError("Palette JSON must be a list of [R, G, B] lists.")
    return [tuple(c) for c in data]


def main():
    parser = argparse.ArgumentParser(
        description="Batch-quantize PNGs, generate tiles with upscaling and gridlines"
    )
    parser.add_argument(
        'palette', nargs='?', default='palette.json',
        help="JSON file with palette colors (default: palette.json)"
    )
    parser.add_argument(
        '--tile-w', type=int, default=32,
        help="Tile width in source pixels (default 32)."
    )
    parser.add_argument(
        '--tile-h', type=int, default=32,
        help="Tile height in source pixels (default 32)."
    )
    parser.add_argument(
        '--pixel-scale', type=int, default=10,
        help="How big the pixels will look, larger number is smaller gridlines (default 10)"
    )
    args = parser.parse_args()

    base_dir = Path(__file__).parent
    input_dir = base_dir / 'input'
    output_dir = base_dir / 'output'
    output_dir.mkdir(exist_ok=True)

    palette_path = Path(args.palette)
    if not palette_path.is_file():
        raise FileNotFoundError(f"Palette file not found: {palette_path}")
    palette = load_palette(palette_path)

    png_files = list(input_dir.glob('*.png'))
    for img_file in tqdm(png_files, desc='Processing images'):
        out_file = output_dir / img_file.name
        quantize_image(img_file, out_file, palette)
        # Tiles subfolder per image
        tiles_dir = output_dir / img_file.stem / 'tiles'
        split_and_upscale(out_file, tiles_dir, args.tile_w, args.tile_h, args.pixel_scale)

    print(f"Done. Check '{output_dir.resolve()}' for quantized images and tiles.")


if __name__ == '__main__':
    main()
