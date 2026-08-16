import json
import os
import sys
import gi

gi.require_version("Gimp", "3.0")
from gi.repository import Gimp, Gio

TARGET_FILTERS = [
    "Exposure",
    "Shadows-Highlights",
    "Levels",
    "Colour Temperature",
    #"Curves",#IGNORED FOR NOW
    "Sharpen (Unsharp Mask)",
    "Noise Reduction",
    "Vignette",
]


def get_filter_enabled(flt):
    """Safely queries active/enabled state on a Gimp.DrawableFilter."""
    for method in ["get_enabled", "is_enabled", "get_active", "get_visible"]:
        if hasattr(flt, method):
            try:
                return bool(getattr(flt, method)())
            except Exception:
                pass
    return True


def get_filter_opacity(flt):
    """Safely extracts opacity on a Gimp.DrawableFilter."""
    if hasattr(flt, "get_opacity"):
        try:
            val = flt.get_opacity()
            # Normalize to 0.0 - 1.0 range if needed
            return round(val if val <= 1.0 else val / 100.0, 4)
        except Exception:
            pass
    return 1.0


def extract_filter_properties(flt):
    """Extracts GEGL slider settings from a DrawableFilter config."""
    properties = {}
    config = None

    if hasattr(flt, "get_config"):
        config = flt.get_config()

    if config and hasattr(config, "list_properties"):
        for pspec in config.list_properties():
            prop_name = pspec.name
            try:
                prop_val = config.get_property(prop_name)
                if isinstance(prop_val, (int, float, bool)):
                    properties[prop_name] = (
                        round(prop_val, 4)
                        if isinstance(prop_val, float)
                        else prop_val
                    )
            except Exception:
                continue

    return properties

#NOT WORKING AS INTENDED FIX AT A LATER STAGE
def extract_curve_samples(flt, num_points=5):
    """Samples Y values from a Curves filter across uniform X intervals."""
    config = flt.get_config() if hasattr(flt, "get_config") else None
    if config and hasattr(config, "get_property"):
        try:
            curve = config.get_property("curve")
            if curve:
                samples = {}
                for i in range(num_points):
                    x = i / (num_points - 1)
                    y = None
                    # Attempt safe evaluation across GimpCurve methods
                    for method_name in ["eval_at_offset", "eval", "get_y_at_x"]:
                        if hasattr(curve, method_name):
                            try:
                                y = getattr(curve, method_name)(x)
                                break
                            except Exception:
                                pass
                    if y is None:
                        y = x
                    samples[f"p{i}_y"] = round(float(y), 4)
                return samples
        except Exception:
            pass

    # Default identity curve fallback: Y = X
    return {
        f"p{i}_y": round(i / (num_points - 1), 4) for i in range(num_points)
    }


#Run on a single file
#gimp -i --quit --batch-interpreter python-fu-eval -b "import sys; sys.path.append('src'); import parse_xcf; parse_xcf.parse_single_file('data/edited/edit_3753.xcf')"
def parse_single_file(xcf_path):
    """Extracts GEGL filter slider parameters and curve points into JSON."""
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
    processed_dir = os.path.join(project_root, "data", "processed")
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)

    gfile = Gio.File.new_for_path(xcf_path)
    image = Gimp.file_load(Gimp.RunMode.NONINTERACTIVE, gfile)

    layers = image.get_layers()
    base_layer = layers[0] if len(layers) > 0 else None

    attached_filters = {}
    if base_layer and hasattr(base_layer, "get_filters"):
        for flt in base_layer.get_filters():
            attached_filters[flt.get_name().strip()] = flt

    param_vector = []
    filter_metadata = {}

    for idx, filter_name in enumerate(TARGET_FILTERS):
        if filter_name in attached_filters:
            flt = attached_filters[filter_name]
            is_enabled = get_filter_enabled(flt)
            opacity = get_filter_opacity(flt)

            # 1. Standard scalar properties
            props = extract_filter_properties(flt)

            # 2. Directly extract 5-point curve samples if this is Curves
            if filter_name == "Curves":
                curve_samples = extract_curve_samples(flt, num_points=5)
                props.update(curve_samples)

            param_vector.append(opacity if is_enabled else 0.0)
            filter_metadata[filter_name] = {
                "index": idx,
                "active": is_enabled,
                "opacity": opacity,
                "properties": props,
            }
        else:
            param_vector.append(0.0)
            filter_metadata[filter_name] = {
                "index": idx,
                "active": False,
                "opacity": 0.0,
                "properties": {},
            }

    output_data = {
        "filename": os.path.basename(xcf_path),
        "parameter_vector": param_vector,
        "filters": filter_metadata,
    }

    json_name = os.path.basename(xcf_path).replace(".xcf", "_params.json")
    out_path = os.path.join(processed_dir, json_name)
    with open(out_path, "w") as f:
        json.dump(output_data, f, indent=4)

    image.delete()
    print("[SUCCESS] Extracted properties with 5-point curve to: " + out_path)


#Run through all files
#gimp -i --quit --batch-interpreter python-fu-eval -b "import sys; sys.path.append('src'); import parse_xcf; parse_xcf.parse_all_files()"
def parse_all_files():
    ###Loops through all .xcf files in data/edited/ and extracts GEGL parameters.
    project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
    )
    edited_dir = os.path.join(project_root, "data", "edited")

    if not os.path.exists(edited_dir):
        print(f"[ERROR] Directory not found: {edited_dir}")
        return

    xcf_files = [f for f in os.listdir(edited_dir) if f.endswith(".xcf")]
    print(f"[INFO] Found {len(xcf_files)} .xcf files to process...")

    for xcf_name in xcf_files:
        full_path = os.path.join(edited_dir, xcf_name)
        try:
            parse_single_file(full_path)
        except Exception as e:
            print(f"[ERROR] Failed to parse {xcf_name}: {e}")