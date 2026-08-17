import os
import json
import torch
from PIL import Image
from torchvision import transforms

import config
from models import PhotoParamRegressor


def load_model(checkpoint_path: str, device: torch.device) -> PhotoParamRegressor:
    """Loads trained model weights from checkpoint."""
    model = PhotoParamRegressor(num_target_params=31, pretrained=False).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def preprocess_image(image_path: str) -> torch.Tensor:
    """Loads and preprocesses an input image for model inference."""
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0)  # Add batch dimension [1, 3, 256, 256]
    return tensor


def vector_to_gegl_dict(predicted_vector: list[float]) -> dict:
    """
    Translates a 31-dimensional normalized predicted vector back into 
    structured GEGL filter properties and actual slider values.
    """
    output_dict = {"filters": {}}
    vec_idx = 0

    for filter_name in config.LAYER_STACK:
        # 1. Extract Opacity (Active status inferred if opacity > 0.05)
        opacity = float(predicted_vector[vec_idx])
        vec_idx += 1
        
        active = opacity > 0.05
        
        filter_props = {}
        # 2. Extract and Denormalize Properties
        for (f_name, prop_key), (min_v, max_v) in config.PARAM_BOUNDS.items():
            if f_name == filter_name:
                norm_val = float(predicted_vector[vec_idx])
                vec_idx += 1
                
                # Denormalize [0.0, 1.0] -> [min_v, max_v]
                denorm_val = min_v + norm_val * (max_v - min_v)
                filter_props[prop_key] = round(denorm_val, 4)

        output_dict["filters"][filter_name] = {
            "active": active,
            "opacity": round(opacity, 4),
            "properties": filter_props
        }

    return output_dict


def run_inference(image_path: str, output_json_path: str, checkpoint_path: str = None):
    """Runs end-to-end inference on an image and saves predicted GEGL parameters to JSON."""
    if checkpoint_path is None:
        checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    
    # 1. Load model and image
    model = load_model(checkpoint_path, device)
    image_tensor = preprocess_image(image_path).to(device)

    # 2. Predict parameters
    with torch.no_grad():
        predicted_tensor = model(image_tensor).squeeze(0).cpu().tolist()

    # 3. Structure and denormalize parameters
    gegl_config = vector_to_gegl_dict(predicted_tensor)

    # 4. Save to JSON
    os.makedirs(os.path.dirname(os.path.abspath(output_json_path)), exist_ok=True)
    with open(output_json_path, "w") as f:
        json.dump(gegl_config, f, indent=4)

    print(f"[SUCCESS] Predicted GEGL parameters saved to: {output_json_path}")
    return gegl_config


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        test_img = sys.argv[1]
    else:
        # Default test fallback
        test_img = os.path.join(config.RAW_DATA_DIR, "IMG_3753.JPG")

    output_path = os.path.join(config.OUTPUT_DIR, "predicted_params.json")
    
    if os.path.exists(test_img):
        run_inference(test_img, output_path)
    else:
        print(f"[USAGE] Provide an image path: python src/infer.py <path_to_image>")