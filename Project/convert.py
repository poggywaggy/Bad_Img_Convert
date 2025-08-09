import argparse
import json
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np
from tqdm import tqdm


def quantize_image(img_path, output_path, palette):
    img = Image.open(img_path).convert('RGB')
    data = np.array(img)
    pixels = data.reshape(-1, 3)
    pal = np.array(palette)
    distances = ((pixels[:, None, :] - pal[None, :, :]) ** 2).sum(axis=2)
    nearest = np.argmin(distances, axis=1)
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
