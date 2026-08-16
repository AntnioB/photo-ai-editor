import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

class PhotoParamRegressor(nn.Module):

    def __init__(self, num_target_params: int = 31, pretrained: bool = True):
        super().__init__()
        
        #Pre-trained ResNet-18 backbone
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = resnet18(weights=weights)

        in_features = self.backbone.fc.in_features

        self.backbone.fc = nn.Identity()

        #Dense Regression Head with Sigmoid Bounding
        self.head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_target_params),
            nn.Sigmoid() #Maps outputs strictlyu to [0.0, 1.0]
        )

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Passes image tensor [B, 3, H, W] -> outputs parameter tensor [B, num_target_params]"""
        features = self.backbone(x) # [B, 512]
        predictions = self.head(features) # [B, 31]
        return predictions


if __name__ == "__main__":
    #Test model shape compatibility
    dummy_image_batch = torch.randn(2, 3, 256, 256) # Batch of 2 images
    model = PhotoParamRegressor(num_target_params=31)

    with torch.no_grad():
        output = model(dummy_image_batch)

    print(f"[SUCCESS] Model initialized successfully.")
    print(f"Input Shape:  {dummy_image_batch.shape}")
    print(f"Output Shape: {output.shape}")  # Expected: [2, 31]
    print(
        f"Value Bounds: Min={output.min().item():.4f}, Max={output.max().item():.4f}"
    )
