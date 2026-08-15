from pathlib import Path

#ROOT
PROJECT_ROOT = Path(__file__).resolve().parent.parent

#DATA folders
DATA_DIR = PROJECT_ROOT/'data'
RAW_DATA_DIR = DATA_DIR/'raw'
EDITED_DATA_DIR = DATA_DIR/'edited'
PROCESSED_DATA_DIR = DATA_DIR/'processed'

#Outputs and Model Saves
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

for folder in [RAW_DATA_DIR, EDITED_DATA_DIR, PROCESSED_DATA_DIR, CHECKPOINT_DIR, OUTPUT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# Strict layer stack enforcement (lowest to highest)
LAYER_STACK = [
    "Exposure",
    "Shadows-Highlights",
    "Levels",
    "Colour Temperature",
    "Curves",
    "Sharpen (Unsharpen Mask)",
    "Noise Reduction",
    "Vignette"
]

SLIDER_BOUNDS = {
    "exposure": (-2.0, 2.0),
    "contrast": (0.5, 1.5),
    "temperature": (-100, 100)
}

