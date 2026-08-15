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

# Map: (Filter Name, GEGL Property Name) -> (Min Bound, Max Bound)
PARAM_BOUNDS = {
    # 1. Exposure
    ("Exposure", "exposure"): (-3.0, 3.0),  # EV Shift
    ("Exposure", "black-level"): (-0.1, 0.1),
    # 2. Shadows-Highlights
    ("Shadows-Highlights", "shadows"): (-100.0, 100.0),
    ("Shadows-Highlights", "highlights"): (-100.0, 100.0),
    ("Shadows-Highlights", "whitepoint"): (-10.0, 10.0),
    # 3. Levels
    ("Levels", "in-low"): (0.0, 1.0),
    ("Levels", "in-high"): (0.0, 1.0),
    ("Levels", "gamma"): (0.1, 5.0),
    ("Levels", "out-low"): (0.0, 1.0),
    ("Levels", "out-high"): (0.0, 1.0),
    # 4. Colour Temperature
    ("Colour Temperature", "original-temperature"): (2000.0, 12000.0),  # Kelvin
    ("Colour Temperature", "intended-temperature"): (2000.0, 12000.0),  # Kelvin
    # 5. Curves (Spline sample points or curve opacity)
    ("Curves", "p1_y"): (0.0, 1.0),
    ("Curves", "p2_y"): (0.0, 1.0),
    ("Curves", "p3_y"): (0.0, 1.0),
    # 6. Sharpen (Unsharpen Mask)
    ("Sharpen (Unsharpen Mask)", "std-dev"): (0.0, 10.0),  # Radius
    ("Sharpen (Unsharpen Mask)", "scale"): (0.0, 3.0),  # Amount
    ("Sharpen (Unsharpen Mask)", "threshold"): (0.0, 1.0),
    # 7. Noise Reduction
    ("Noise Reduction", "iterations"): (1.0, 10.0),
    # 8. Vignette
    ("Vignette", "radius"): (0.0, 2.0),
    ("Vignette", "softness"): (0.0, 1.0),
    ("Vignette", "gamma"): (0.0, 5.0),
}
