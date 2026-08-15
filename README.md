# DeepEdit AI: Parametric & Layered Image Style Transfer

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end deep learning framework that learns custom photographic editing styles from paired RAW and edited photo datasets. Instead of producing destructive, flat RGB images, **DeepEdit AI** predicts structured non-destructive editing parameters and localized spatial masks, exporting fully editable **layered PSD/XCF project files** compatible with Photoshop, GIMP, and Affinity Photo.

---

## 🎯 Project Goals & Overview

Most deep learning models for image editing (e.g., Image-to-Image GANs, Diffusion Models) act as "black box" pixel generators. While effective, they introduce visual artifacts, alter fine textures, and output flattened images that photographers cannot easily tweak or refine.

**DeepEdit AI** bridges machine learning and professional photo editing workflows by adopting a **Hybrid Parametric Approach**:

* **Non-Destructive Output:** Predicts numeric slider parameters (exposure, white balance, RGB tone curves, HSL shifts) and spatial region masks rather than raw pixels.
* **Layered File Export:** Generates standard `.psd` (or `.xcf`) files with separate adjustment layers, custom opacities, blend modes, and embedded layer masks.
* **Full Resolution Independence:** Network inference operates on downsampled thumbnails ($256 \times 256$) for extreme execution speed, while predicted parametric curves scale losslessly to high-res $45\text{MP}+$ RAW files.
* **Human-in-the-Loop Workflow:** Provides an automated $90\%$ visual baseline while preserving complete creative control for manual fine-tuning.

---

## 🏗 System Architecture

The pipeline consists of three distinct modules: a **Dual-Head Neural Network**, a **Differentiable PyTorch Renderer**, and a **Parametric PSD/XCF Exporter**.

```
                           ┌──► Global Parameter Head (MLP) ──► Parameter Vector θ
                           │                                          │
Downsampled RAW Image ─────┤                                          ├─► Differentiable Renderer ─► Rendered Image ─► Loss
(256x256x3)                │                                          │   (Training Mode)
                           └──► Local Mask Head (U-Net)  ──► Masks (M) ┘
                                                                      │
                                                                      └─► Layered PSD Exporter ──► Editable .PSD / .XCF
                                                                          (Inference Mode)
```

### 1. Dual-Head Parameter Network
* **Shared Encoder Backbone:** A lightweight convolutional network (e.g., ResNet-18 / MobileNetV3) extracts global color, contrast, and spatial layout features.
* **Global Parameter Head:** A Multi-Layer Perceptron (MLP) with bounded activations outputs a normalized parameter vector $\vec{\theta} \in \mathbb{R}^N$ representing global slider positions (Exposure, Contrast, Temp, Tint, RGB Curves, HSL).
* **Local Mask Head:** A lightweight transposed-convolution decoder generates single-channel grayscale spatial masks $M_k \in [0, 1]^{H \times W}$ for region-specific adjustments (e.g., subject isolation, sky gradients, vignette).

### 2. Differentiable PyTorch Renderer (Training Phase)
During training, backpropagation requires evaluating how parameter updates impact output image quality. The Differentiable Renderer executes standard photo-editing mathematics natively inside PyTorch computational graphs:
$$\hat{I} = \text{Render}(X, \vec{\theta}, M_1, \dots, M_K)$$

### 3. Layered File Exporter (Inference Phase)
During inference, the neural renderer is bypassed. The model outputs raw numeric parameters and grayscale mask tensors, which a Python export script compiles into a native layered `.psd` file using `psd-tools`.

---

## 📂 Project Directory Structure

```
photo-ai-editor/
├── data/
│   ├── raw/             # Unedited original RAW/flat files (.CR2, .NEF, .ARW, .dng)
│   ├── edited/          # Final ground-truth edits (.jpg, .png)
│   └── processed/       # Downsampled & aligned pair cache (256x256)
├── src/
│   ├── dataset.py       # PyTorch Dataset loader & RawPy processing
│   ├── models.py        # ResNet dual-head network architecture
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
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install computer vision, RAW parsing, and PSD export utilities
pip install rawpy opencv-python pillow psd-tools numpy matplotlib tqdm jupyter ipykernel
```

### 3. Register Jupyter Kernel

```bash
python -m ipykernel install --user --name=photo-ai-env --display-name "Python (photo-ai-env)"
```

---

## 🚀 Quickstart Workflow

### Step 1: Prepare Paired Dataset
Place your unedited RAW photos in `data/raw/` and corresponding edited ground-truth images in `data/edited/` using identical filenames:
```
data/raw/photo_001.ARW
data/edited/photo_001.png
```

### Step 2: Run Dataset Alignment & Cache
Generate downsampled, aligned paired thumbnails for high-speed model training:
```bash
python src/dataset.py --preprocess --size 256
```

### Step 3: Train Model
Train the dual-head network using combined $L_1$ pixel loss and VGG perceptual loss:
```bash
python src/train.py --epochs 100 --batch-size 16 --lr 1e-4
```

### Step 4: Run Inference & Generate Layered PSD
Pass a new, unseen RAW image through the trained model to output a non-destructive PSD project:
```bash
python src/export_psd.py --input data/raw/sample.ARW --output outputs/sample_edit.psd
```

Open `outputs/sample_edit.psd` in **GIMP**, **Photoshop**, or **Affinity Photo** to inspect and fine-tune individual layers.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.