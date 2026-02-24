# SIFT Image Alignment

A lightweight, robust, and standalone pipeline for image registration using the **SIFT (Scale-Invariant Feature Transform)** algorithm.

This project is built using **Pure OpenCV and NumPy**, meaning it does **not** require heavy deep learning frameworks like PyTorch or TensorFlow. It extracts keypoints, matches them using Nearest Neighbor distance ratio, estimates affine transformations (via RANSAC or LMEDS), and generates comprehensive visual evaluations (Heatmaps, Error lines, Displacement Vector Fields, and Checkerboards).

## 🌟 Features

* **Standalone & Lightweight:** No PyTorch dependency. Pure NumPy mathematical operations.
* **Auto-Thresholding:** Automatically adjusts the Nearest Neighbor ratio threshold if too few matches are found.
* **Multiple Execution Modes:** Run via Command Line Interface (CLI) or Jupyter Notebook.
* **Batch Processing:** Support reading image pairs directly from the terminal or via a CSV file.
* **Comprehensive Visualization:** Generates side-by-side matches, error line plots, and checkerboard blends to visually verify the alignment accuracy.

---

## 🛠️ Installation & Setup

We highly recommend using a Virtual Environment (`venv`) to keep your dependencies clean.

### Option 1: Automated Setup via Makefile (Linux/Ubuntu/macOS)

If you are on a Unix-based system, you can use the provided `Makefile` to automatically install system dependencies, create a virtual environment, and install the required Python packages.

1. Open your terminal in the project directory.
2. Run the setup command:
```bash
make setup

```


3. Activate the virtual environment:
```bash
source .venv/bin/activate

```



### Option 2: Manual Setup (Any OS)

If you prefer to install things manually or are using Windows:

1. Create a virtual environment:
```bash
# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\Scripts\activate

```


2. Install the required minimalist libraries:
```bash
pip install --upgrade pip
pip install opencv-python numpy matplotlib scikit-image tqdm jupyter ipywidgets

```



---

## 🚀 Usage 1: Command Line Interface (Python Script)

You can run the script `SIFT_align_image.py` using two main methods.

### Method 1: Passing Images directly via CLI

Use the `--img_list` argument followed by pairs of images (Source and Target).

```bash
python SIFT_align_image.py --img_list ./images/eye1_src.jpg ./images/eye1_tgt.jpg ./images/eye2_src.jpg ./images/eye2_tgt.jpg

```

### Method 2: Batch processing via CSV

Create a `.csv` or `.txt` file (e.g., `my_pairs.csv`) where each line contains the path to the source image and the target image, separated by a comma:

```text
test_images/src_1.jpg, test_images/tgt_1.jpg
test_images/src_2.jpg, test_images/tgt_2.jpg

```

Then run:

```bash
python SIFT_align_image.py --img_file my_pairs.csv

```

### ⚙️ Available CLI Options

You can customize the execution using the following arguments:

* `--nn_threshold` (float): Initial Nearest Neighbor ratio threshold. Default is `0.7`.
* `--method` (string): Affine estimation method. Choices are `RANSAC` (default) or `LMEDS`.
* `--save` (int):
* `0`: Saves only the evaluation plot.
* `1` (Default): Saves the evaluation plot AND the original 256x256 color images (`source`, `target`, `warped`).



**Full Command Example:**

```bash
python SIFT_align_image.py --img_file my_pairs.csv --nn_threshold 0.75 --method LMEDS --save 1

```

---

## 📓 Usage 2: Jupyter Notebook

If you prefer an interactive environment, use the `SIFT_align_image.ipynb` file.

1. Start Jupyter Notebook from your terminal (ensure your `.venv` is activated):
```bash
jupyter notebook

```


2. Open `SIFT_align_image.ipynb`.
3. Scroll down to **Cell 4: Configuration / Parameters**.
4. Modify the variables directly in the cell to fit your needs:
```python
# Option 1: Provide a direct list of image paths
img_list = ['test_images/src.jpg', 'test_images/tgt.jpg'] 

# Option 2: Provide a path to a CSV file (Leave img_list as [] to use this)
img_file = "my_pairs.csv" 

nn_threshold = 0.7
method = 'RANSAC' 
save_images = 1

```


5. Click **"Run All"** to execute the pipeline. The output logs and progress bar will appear within the notebook.

---

## 📁 Output Structure

Upon successful execution, the script will automatically create a timestamped folder inside the `output/` directory (e.g., `output/SIFT_Standalone_20260224-153000/`).

Inside this folder, you will find:

1. `pair_000_imgName_result.png`: A high-resolution matplotlib figure showing matches, error vectors, and checkerboard blends.
2. `pair_000_imgName_src.png`, `*_tgt.png`, `*_warp.png`: The standalone 256x256 color images (if `--save 1` was used).
3. `results_log.csv`: A summary log containing the file paths, number of valid SIFT matches found, and the execution status of each pair.

---

## 📄 File Overview

* `SIFT_align_image.py`: The main command-line executable.
* `SIFT_align_image.ipynb`: The interactive Jupyter Notebook version.
* `SIFT_align_image_functions.py`: The core engine containing pure NumPy math, metrics (MSE, TRE, SSIM), and Matplotlib plotting logic.
* `Makefile`: Automation script for environment setup.