#imports
import os
import json
import re

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import config

#Target Vector Construction: Parses each sample's JSON file to extract the 8-filter GEGL stack, flattening the relevant opacities and numerical slider properties into a single 1D target FloatTensor ($\vec{\theta}$).
def normalize_val(val: float, min_val: float, max_val: float) -> float:
    """Clamps and scales a parameter value to a normalized [0.0, 1.0] range."""
    clamped = max(min_val, min(max_val, float(val)))
    return (clamped - min_val) / (max_val - min_val)

def parse_json_target(json_path: str) -> torch.Tensor:
    """Parses a sample's JSON file and returns a 1D target FloatTensor (\vec{\theta})"""
    with open(json_path, "r") as f:
        data = json.load(f)
    
    filter_dict = data.get("filters",{})
    target_vector = []

    for filter_name in config.LAYER_STACK:
        flt_data = filter_dict.get(
            filter_name, {"active": False, "opacity": 0.0, "properties":{}}
        )

        #1.Active Opacity Target
        is_active = flt_data.get("active", False)
        opacity = flt_data.get("opacity", 0.0) if is_active else 0.0
        target_vector.append(float(opacity))

        #2.Normalize GEGL Slider Properties
        props = flt_data.get("properties", {})
        for (f_name, prop_key), (min_v, max_v) in config.PARAM_BOUNDS.items():
            if f_name == filter_name:
                default_val = (min_v + max_v) / 2.0
                raw_val = props.get(prop_key, default_val)
                norm_val = normalize_val(raw_val, min_v, max_v)
                target_vector.append(norm_val)
        
    return torch.tensor(target_vector, dtype=torch.float32)
            

#Input Image Pipeline: Loads raw image files, applies standard spatial transforms (resizing, cropping, PyTorch ToTensor conversion), and normalizes pixel ranges (e.g., standard ImageNet mean and standard deviation) to output clean [C, H, W] image tensors.

class PhotoTransformPipeline:
    def __init__(self, img_size=(config.IMG_SIZE)):
        self.transform = transforms.Compose(
            [
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    
    def load_and_process(self, img_path:str) -> torch.Tensor:
        # .convert("RGB") enforces 3 channels, discarding RGBA alphas or expanding 1-channel grayscales
        with Image.open(img_path) as img:
            img_rgb = img.convert('RGB')
            return self.transform(img_rgb)


#PyTorch Dataset Interface: Implements __len__ to return total valid sample counts and __getitem__ to return the (input_image_tensor, target_parameter_tensor) tuple required by PyTorch DataLoader batching.
class PhotoEditDataset(Dataset):

    def __init__(
        self,
        raw_dir: str = config.RAW_DATA_DIR,
        processed_dir: str = config.PROCESSED_DATA_DIR
    ):
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        self.pipeline = PhotoTransformPipeline()
        self.valid_list = self._index_files()

    def _index_files(self) -> list[tuple[str, str]]:
        """Indexes matching (raw_image, params_json) file pairs by matching the embedded sample ID number."""
        #1. Build a map of: ID Number string -> Full JSON Path
        json_map = {}
        with os.scandir(self.processed_dir) as entries:
            for entry in entries:
                if entry.is_file() and entry.name.endswith(".json"):
                    match = re.search(r"\d+", entry.name)
                    if match:
                        sample_id = match.group(0)  # Extracts e.g. "3753"
                        json_map[sample_id] = os.path.join(
                            self.processed_dir, entry.name
                        )

        # 2. Match raw images against the JSON map using the extracted ID number
        valid_pairs = []
        with os.scandir(self.raw_dir) as entries:
            for entry in entries:
                if entry.is_file() and entry.name.lower().endswith(
                    (".jpg", ".jpeg", ".png")
                ):
                    match = re.search(r"\d+", entry.name)
                    if match:
                        sample_id = match.group(0)  # Extracts e.g. "3753"
                        if sample_id in json_map:
                            img_path = os.path.join(self.raw_dir, entry.name)
                            json_path = json_map[sample_id]
                            valid_pairs.append((img_path, json_path))

        print(
        f"[INFO] Initialized PhotoEditDataset with {len(valid_pairs)} samples."
        )
        return valid_pairs

    def __len__(self) -> int:
        return len(self.valid_list)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path, json_path = self.valid_list[idx]

        x = self.pipeline.load_and_process(img_path)
        y = parse_json_target(json_path)

        return x, y

if __name__ == "__main__":
    dataset = PhotoEditDataset()
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    images, targets = next(iter(dataloader))
    print(f"Batch Image Tensor Shape:  {images.shape}")  # e.g., [4, 3, 256, 256]
    print(f"Batch Target Tensor Shape: {targets.shape}")  # e.g., [4, N_PARAMS]
