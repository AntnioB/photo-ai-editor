#imports
import os
import config

from PIL import Image
import torch
from torchvision import transforms

#File Indexing & Pairing: Scans your raw image directory, finds matching _params.json files in data/processed/, and builds a validated list of (image_path, json_path) tuples while filtering out any missing or corrupt files.
processed_files = set(os.listdir(config.PROCESSED_DATA_DIR))
valid_list = []

with os.scandir(config.RAW_DATA_DIR) as entries:
    for entry in entries:
        # 1. Check if it's a file and has an image extension
        if entry.is_file() and entry.name.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):

            # 2. Derive expected JSON filename
            base_name = os.path.splitext(entry.name)[0]
            expected_json = f"{base_name}_params.json"

            # 3. Verify matching JSON exists in set
            if expected_json in processed_files:
                img_path = os.path.join(config.RAW_DATA_DIR, entry.name)
                json_path = os.path.join(
                    config.PROCESSED_DATA_DIR, expected_json
                )
                valid_list.append((img_path, json_path))
            

#Input Image Pipeline: Loads raw image files, applies standard spatial transforms (resizing, cropping, PyTorch ToTensor conversion), and normalizes pixel ranges (e.g., standard ImageNet mean and standard deviation) to output clean [C, H, W] image tensors.

class PhotoTransformPipeline:
    def __init__(self, img_size=(config.IMAGE_SIZE)):
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

#Target Vector Construction: Parses each sample's JSON file to extract the 8-filter GEGL stack, flattening the relevant opacities and numerical slider properties into a single 1D target FloatTensor ($\vec{\theta}$).

#Parameter Normalization: Scales disparate slider ranges (e.g., Colour Temperature in Kelvin vs. Exposure EV steps) into unified bounds like $[0, 1]$ or $[-1, 1]$ so large numbers don't overwhelm model gradients during loss calculation.

#PyTorch Dataset Interface: Implements __len__ to return total valid sample counts and __getitem__ to return the (input_image_tensor, target_parameter_tensor) tuple required by PyTorch DataLoader batching.
