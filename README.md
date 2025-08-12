# wPlace_convert

**wPlace_convert** is a simple tool to convert images so that they only use colors from a specific palette. It also splits the image into tiles, upscales them, and adds grid lines for easy viewing. (MAKE SURE THE TILES ARE DIVISIBLE) [More Info](#more-info)

Jump to: [Quick Start](#quick-start) | [Demo](#demo) | [Dependencies](#dependencies) | [Installing Python](#installing-python) | [How to Griddy](#griddy)



## Quick Start

1. **Install Python** ([see instructions](#installing-python))
2. **Install dependencies** ([see instructions](#dependencies))
3. **Download files** Both `convert.py` and `palette.json`
4. **Move Files** into a folder
5. **Place your PNG images** in the `Project/input` folder.
6. **Edit `palette.json`** in the `Project` folder if you want a custom palette. the file `special_pallette.json` contains the premium colors as well!
7. **Open a terminal** in the `Project` folder.
8. **Run the program:**
   ```
   python convert.py 
   ```
   or
   ```
   python convert_new.py
   ```

   If you want the premium pallete colors, do:
   ```
   python convert.py special_pallete.json
   ```
   or
   ```
   python convert_new.py special_pallete.json
   ```

9. **Find your results** in the `Project/output` folder.



## Demo

Here’s what happens when you run the program:

- **Input:**  
  Place your image in `Project/input/`

- **Output:**  
  After running, you’ll see:
  - `Project/output/input.png` 
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

- **Change the palette:** Edit `palette.json` in the `Project` folder. Each color is a list like `[R, G, B]`.
- **Change tile size or upscale:**  
  Run with options, for example:
  ```
  python convert.py --tile-w=16 --tile-h=32 --pixel-scale=5
  ```
  or
  ```
  python convert_new.py special_pallete.json --tile-w=32 --tile-h=18 --pixel-scale=5
  ```

- **Questions?**  
  Open an [issue on GitHub](https://github.com/poggywaggy/wPlace_convert/issues) or ask me on twitter or smth.

## Griddy
- If you are using the [wplace](https://www.wplace.org/) image converter already, you can use griddy.py!
- Just run this command snippet to create the needed folders
  ```
  python griddy.py
  ```
- After running, you should tell the program the pixel size to make a proper grid. Shown like the following:
  ```
  python griddy.py {valueGreaterThan2}
  ```

