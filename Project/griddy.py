import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFile
from tqdm import tqdm
import sys

def griddy_image(
        img_path: Path,
        pixel_size: int,
        out_path: Path,
        tile_w: int = 32,
        tile_h: int = 32,
        border_px: int = 1
    ):
    img = Image.open(img_path).convert("RGB")
    scale_w, scale_h = img.size
    pixel_scale = int(pixel_size)
    if pixel_scale < 1:
        raise ValueError("pixel_size must be >= 1")

    # warn when input size isn't an exact multiple of the scale
    if (scale_w % pixel_scale) != 0 or (scale_h % pixel_scale) != 0:
        print(f"Warning: image {img_path.name} size {scale_w}x{scale_h} is not an exact multiple "
              f"of pixel_size={pixel_scale}. Partial cells at edges will be handled.", file=sys.stderr)

    # source (original small) dimensions
    src_w = scale_w // pixel_scale
    src_h = scale_h // pixel_scale

    # how many tiles in source-pixel coordinates
    num_cols = (src_w + tile_w - 1) // tile_w
    num_rows = (src_h + tile_h - 1) // tile_h

    tiles_dir = out_path.parent / (img_path.stem + "_tiles")
    tiles_dir.mkdir(parents=True, exist_ok=True)

    for row in range(num_rows):
        for col in range(num_cols):
            # tile bounds in SOURCE pixels
            src_left = col * tile_w
            src_top = row * tile_h
            src_right = min(src_left + tile_w, src_w)
            src_bottom = min(src_top + tile_h, src_h)

            cells_x = src_right - src_left
            cells_y = src_bottom - src_top
            if cells_x <= 0 or cells_y <= 0:
                continue

            # convert to upscaled coordinates (pixels)
            up_left = src_left * pixel_scale
            up_top = src_top * pixel_scale
            up_right = src_right * pixel_scale
            up_bottom = src_bottom * pixel_scale

            box = (up_left, up_top, up_right, up_bottom)
            tile_img = img.crop(box)

            # draw internal black borders and fill inner color from tile_img itself
            out_img = Image.new("RGB", tile_img.size, (0, 0, 0))
            draw = ImageDraw.Draw(out_img)

            up_w, up_h = tile_img.size
            for sy in range(cells_y):
                for sx in range(cells_x):
                    cell_x0 = sx * pixel_scale
                    cell_y0 = sy * pixel_scale
                    cell_w = min(pixel_scale, up_w - cell_x0)
                    cell_h = min(pixel_scale, up_h - cell_y0)

                    # if the cell is too small to hold an inner fill after border, skip fill
                    if cell_w <= 2 * border_px or cell_h <= 2 * border_px:
                        continue

                    # pick a sample color from the original tile
                    color = tile_img.getpixel((cell_x0, cell_y0))

                    inner_left = cell_x0 + border_px
                    inner_top = cell_y0 + border_px
                    inner_right = cell_x0 + cell_w - border_px - 1
                    inner_bottom = cell_y0 + cell_h - border_px - 1

                    draw.rectangle([inner_left, inner_top, inner_right, inner_bottom], fill=color)

            tile_name = f"tile_r{row}_c{col}_s{cells_x}x{cells_y}.png"
            out_img.save(tiles_dir / tile_name)

    return tiles_dir

def main():
    parser = argparse.ArgumentParser(
        description="Batch griddify upscaled PNGs in toGriddy -> Griddified"
    )
    parser.add_argument(
        "pixel_size", type=int, 
        help="Size (in output px) of each upscaled source pixel (e.g. 10)"
    )
    args = parser.parse_args()

    input_dir = Path("toGriddy")
    output_dir = Path("Griddified")
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    png_files = list(input_dir.glob('*.png'))
    if not png_files:
        print(f"No PNG files found in '{input_dir.resolve()}'. Put .png files there and run again.")
        return

    for img_file in tqdm(png_files, desc='Processing images'):
        try:
            out_name = img_file.stem + "_griddified.png"
            out_path = output_dir / out_name
            griddy_image(img_file, args.pixel_size, out_path)
        except Exception as e:
            print(f"Error processing {img_file.name}: {e}", file=sys.stderr)

    print(f"Done. Processed {len(png_files)} files -> '{output_dir.resolve()}'")

if __name__ == "__main__":
    main()
    