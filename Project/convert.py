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


def split_and_upscale(img_path, tiles_dir, tile_size, upscale):
    img = Image.open(img_path)
    w, h = img.size
    tiles_dir.mkdir(parents=True, exist_ok=True)
    for row in range(0, h, tile_size):
        for col in range(0, w, tile_size):
            box = (col, row, col + tile_size, row + tile_size)
            tile = img.crop(box)
            up_w, up_h = tile_size * upscale, tile_size * upscale
            up_tile = tile.resize((up_w, up_h), Image.NEAREST)
            draw = ImageDraw.Draw(up_tile)
            # Draw grid lines
            for x in range(0, up_w + 1, upscale):
                draw.line(((x, 0), (x, up_h)), fill=(0, 0, 0))
            for y in range(0, up_h + 1, upscale):
                draw.line(((0, y), (up_w, y)), fill=(0, 0, 0))
            tile_name = f"tile_r{row//tile_size}_c{col//tile_size}.png"
            up_tile.save(tiles_dir / tile_name)


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
        '--tile-size', type=int, default=32,
        help="Size of square tiles to split into, recomend to be divisible (default: 32)"
    )
    parser.add_argument(
        '--upscale', type=int, default=10,
        help="Factor to upscale each tile (default: 10)"
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
        split_and_upscale(out_file, tiles_dir, args.tile_size, args.upscale)

    print(f"Done. Check '{output_dir.resolve()}' for quantized images and tiles.")


if __name__ == '__main__':
    main()
