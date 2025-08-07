# wPlace_convert

**wPlace_convert** is a simple tool to convert images so that they only use colors from a specific palette. It also splits the image into tiles, upscales them, and adds grid lines for easy viewing. (MAKE SURE THE TILES ARE DIVISIBLE) [More Info](#more-info)

Jump to: [Quick Start](#quick-start) | [Demo](#demo) | [Dependencies](#dependencies) | [Installing Python](#installing-python)



## Quick Start

1. **Install Python** ([see instructions](#installing-python))
2. **Install dependencies** ([see instructions](#dependencies))
3. **Place your PNG images** in the `Project/input` folder.
4. **Edit `palette.json`** in the `Project` folder if you want a custom palette.
5. **Open a terminal** in the `Project` folder.
6. **Run the program:**
   ```
   python convert.py
   ```
7. **Find your results** in the `Project/output` folder.



## Demo

Here’s what happens when you run the program:

- **Input:**  
  Place your image in `Project/input/input.png`

- **Output:**  
  After running, you’ll see:
  - `Project/output/input.png` (the quantized image)
  - `Project/output/input/tiles/tile_r0_c0.png`, `tile_r0_c1.png`, ... (upscaled tiles with grid lines)

**Example folder structure after running:**
```
Project/
  input/
    input.png
  output/
    input.png
    input/
      tiles/
        tile_r0_c0.png
        tile_r0_c1.png
        tile_r0_c2.png
        tile_r0_c3.png
        ...
```



## Dependencies

You need Python and a few Python packages.

### 1. Install Python packages

Open a terminal (Command Prompt or PowerShell on Windows) in the `Project` folder and run:

```
pip install pillow numpy tqdm
```

- `pillow` (for image processing)
- `numpy` (for fast math)
- `tqdm` (for progress bars)

If you see an error about `pip` not being found, see [Installing Python](#installing-python).



## Installing Python

If you don’t have Python:

1. **Download Python:**  
   Go to the [official Python website](https://www.python.org/downloads/) and click **Download Python**.

2. **Install Python:**  
   - Run the installer.
   - **IMPORTANT:** On the first screen, check the box that says **"Add Python to PATH"**.
   - Click **Install Now**.

3. **Check installation:**  
   Open Command Prompt and type:
   ```
   python --version
   ```
   You should see something like `Python 3.12.0`.

4. **Install pip (if needed):**  
   Most Python installations include `pip`. If not, see the [pip installation guide](https://pip.pypa.io/en/stable/installation/).



## More Info

- **WARNING:** This program intends that the dimentions of the image are squares and that the square can be divisible by the tile size (128x128 is divisible by 32x32) I havent tested rectangles or partial squares, if you care to do that tell me how it goes!
- **Change the palette:** Edit `palette.json` in the `Project` folder. Each color is a list like `[R, G, B]`.
- **Change tile size or upscale:**  
  Run with options, for example:
  ```
  python convert.py --tile-size=16 --upscale=8
  ```
- **Questions?**  
  Open an [issue on GitHub](https://github.com/poggywaggy/wPlace_convert/issues) or ask me on twitter or smth.


