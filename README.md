# Photo AI Editor

An end-to-end deep learning pipeline built with PyTorch to automatically predict photo editing parameters from unedited images. The model extracts spatial visual features (luminance balance, color temperature, contrast) and translates them into a 31-dimensional bounded target vector that directly maps to GIMP / GEGL filter settings.

---

## Project Architecture

```text
photo-AI-editor/
├── config.py             # Global paths, filter definitions, and parameter bounds
├── src/
│   ├── dataset.py        # PyTorch Dataset for loading image/JSON pairs
│   ├── model.py          # ResNet-18 Backbone + 31-dim Regression Head
│   ├── train.py          # Training loop, MaskedParamLoss, & checkpoint management
│   ├── infer.py          # Inference script exporting GEGL JSON parameters
│   └── parse_xcf.py      # Script which parses GEGL parameters into a JSON file to be used in training
├── checkpoints/          # Model weights (.pth)
└── data/                 # Raw images and ground-truth/predicted JSONs
```

### Model Pipeline Overview
* **Backbone Feature Extractor:** Pre-trained ResNet-18 maps input images ($3 \times 256 \times 256$) to a 512-dimensional feature embedding.
* **Regression Head:** Dense sequential layers (Linear $\rightarrow$ BatchNorm1d $\rightarrow$ ReLU $\rightarrow$ Dropout) project features to 31 parameter targets.
* **Bounding Activation:** A final nn.Sigmoid() function constrains all outputs strictly to $[0.0, 1.0]$.
* **Parameter Denormalization:** src/infer.py converts normalized predictions back to real physical GEGL slider ranges (e.g., exposure stops, Kelvin shifts, unsharp mask scale).