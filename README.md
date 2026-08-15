# photo-AI-editor: Parametric & Layered Image Style Transfer

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end deep learning framework that learns custom photographic editing styles from paired images and native project files (GIMP/Photoshop). Instead of producing destructive, flat RGB images, **photo-AI-editor** predicts structured non-destructive editing parameters and localized spatial masks, exporting fully editable **layered PSD/XCF project files** compatible with Photoshop, GIMP, and Affinity Photo.

---

## 🎯 Project Goals & Overview

Most deep learning models for image editing (e.g., Image-to-Image GANs, Diffusion Models) act as "black box" pixel generators. While effective, they introduce visual artifacts, alter fine textures, and output flattened images that photographers cannot easily tweak or refine.

**photo-AI-editor** bridges machine learning and professional photo editing workflows by adopting a **Hybrid Supervised & Parametric Approach**:

* **Direct Parameter Supervision:** Learns directly from the layer settings (exposure, contrast, tone curves, color balance) and layer masks stored inside your native GIMP/PSD edit files.
* **Non-Destructive Output:** Predicts numeric slider parameters and spatial region masks rather than raw pixels.
* **Layered File Export:** Generates standard `.psd` (or `.xcf`) files with separate adjustment layers, custom opacities, blend modes, and embedded layer masks.
* **Full Resolution Independence:** Network inference operates on downsampled thumbnails ($256 \times 256$) for extreme execution speed, while predicted parametric curves scale losslessly to high-res images.
* **Human-in-the-Loop Workflow:** Provides an automated $90\%$ visual baseline while preserving complete creative control for manual fine-tuning.

---

## 🏗 System Architecture

The pipeline consists of four distinct modules: a **Metadata & Layer Parser**, a **Dual-Head Neural Network**, a **Differentiable PyTorch Renderer**, and a **Parametric PSD/XCF Exporter**.

```
                           ┌──► Global Parameter Head (MLP) ──► Parameter Vector θ ──┐
                           │                                                          ├─► Hybrid Loss (θ + Masks + Image)
Input Image (256x256x3) ───┤                                                          │
                           └──► Local Mask Head (U-Net)  ──► Masks (M) ──────────────┤
                                                                                      │
                                                                                      ├─► Differentiable Renderer (Preview)
                                                                                      │
                                                                                      └─► Layered PSD Exporter ──► Editable .PSD / .XCF
```

### 1. Layer & Metadata Parser (Preprocessing)
Extracts exact numerical layer parameters (saved to `.json`) and layer masks (saved as grayscale `.png` images) directly from your edited GIMP (`.xcf`) or Photoshop (`.psd`) source files.

### 2. Dual-Head Parameter Network
* **Shared Encoder Backbone:** A lightweight convolutional network (e.g., ResNet-18 / MobileNetV3) extracts global color, contrast, and spatial layout features.
* **Global Parameter Head:** A Multi-Layer Perceptron (MLP) outputs a normalized parameter vector $\vec{\theta} \in \mathbb{R}^N$ matching your GIMP layer settings.
* **Local Mask Head:** A lightweight transposed-convolution decoder generates single-channel grayscale spatial masks $M_k \in [0, 1]^{H \times W}$ matching your GIMP layer masks.

### 3. Differentiable PyTorch Renderer & Hybrid Loss (Training Phase)
Training combines direct parameter supervision with rendered image verification:
$$\mathcal{L}_{\text{total}} = \lambda_1 \|\vec{\theta}_{\text{pred}} - \vec{\theta}_{\text{target}}\|_2^2 + \lambda_2 \|M_{\text{pred}} - M_{\text{target}}\|_1 + \lambda_3 \|\hat{I} - Y\|_1$$

### 4. Layered File Exporter (Inference Phase)
During inference, the predicted parameters $\vec{\theta}$ and spatial masks $M$ are mapped into a newly assembled `.psd` file containing non-destructive adjustment layers.

---

## 📂 Project Directory Structure

```
photo-ai-editor/
├── data/
│   ├── raw/             # Original unedited images (.jpg, .png)
│   ├── edited/          # Source edit project files (.xcf, .psd) or target exports (.jpg, .png)
│   └── processed/       # Extracted layer parameter JSONs, target masks, and downsampled pairs
├── src/
│   ├── config.py        # Centralized paths, slider bounds, and hyperparameters
│   ├── parse_gimp.py    # Preprocessing script to extract parameters/masks from GIMP/PSD files
│   ├── dataset.py       # PyTorch Dataset loader for images, JSON targets, and masks
│   ├── models.py        # Dual-head network architecture
│   ├── renderer.py      # Differentiable parameter rendering ops
│   └── export_psd.py    # Parameter-to-PSD translation engine
├── notebooks/
│   └── test_env.ipynb   # Environment verification & visualization
├── checkpoints/         # Model weights & training state saves (.pth)
├── outputs/             # Generated test PSD/XCF files
├── .gitignore
├── README.md
└── requirements.txt
```

---

## ⚡ Installation & Setup

### Prerequisites
* **OS:** Linux (Ubuntu/Debian recommended) or macOS
* **Shell:** Bash or Fish Shell
* **Python:** 3.10 or 3.11
* **Hardware:** Nvidia GPU with CUDA 11.8/12.1 (recommended) or Apple Silicon (MPS)

### 1. Clone & Set Up Virtual Environment

```bash
# Clone the repository
git clone https://github.com/YOUR-USERNAME/photo-ai-editor.git
cd photo-ai-editor

# Create virtual environment
python3 -m venv .venv

# Activate environment (Fish Shell)
source .venv/bin/activate.fish

# Activate environment (Bash/Zsh)
# source .venv/bin/activate
```

### 2. Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA 12.1 support
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Install computer vision, RAW parsing, and PSD export utilities
pip install rawpy opencv-python pillow psd-tools numpy matplotlib tqdm jupyter ipykernel
```

### 3. Register Jupyter Kernel

```bash
python -m ipykernel install --user --name=photo-ai-env --display-name "Python (photo-ai-env)"
```

---

## 🚀 Quickstart Workflow

### Step 1: Prepare Dataset
Place unedited photos in `data/raw/` and corresponding GIMP/Photoshop files in `data/edited/` using matching basenames:
```
data/raw/photo_001.jpg
data/edited/photo_001.xcf
```

### Step 2: Parse Layers & Preprocess Data
Extract parameter targets (`.json`) and layer masks (`.png`), then cache downsampled image pairs:
```bash
python src/parse_gimp.py
python src/dataset.py --preprocess --size 256
```

### Step 3: Train Model
Train using direct parameter loss, mask loss, and perceptual image rendering loss:
```bash
python src/train.py --epochs 100 --batch-size 16 --lr 1e-4
```

### Step 4: Run Inference & Generate Layered PSD
Pass an unedited image through the trained model to generate an editable `.psd` project:
```bash
python src/export_psd.py --input data/raw/sample.jpg --output outputs/sample_edit.psd
```

Open `outputs/sample_edit.psd` in **GIMP**, **Photoshop**, or **Affinity Photo** to inspect and fine-tune individual layers.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.