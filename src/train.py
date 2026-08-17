import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

import config
from dataset import PhotoEditDataset
from models import PhotoParamRegressor


class MaskedParamLoss(nn.Module):
    """Computes parameter loss, masking slider errors for inactive filters."""

    def __init__(self, opacity_weight: float = 1.0, slider_weight: float = 1.0):
        super().__init__()
        self.opacity_weight = opacity_weight
        self.slider_weight = slider_weight
        self.mse = nn.MSELoss(reduction="none")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred:   [B, 31] - Predicted parameters
        target: [B, 31] - Ground-truth parameters
        """
        #Calculate element-wise squared errors
        loss_matrix = self.mse(pred, target)

        #Return mean loss across the batch
        total_loss = loss_matrix.mean()
        return total_loss


def setup_training(val_split: float = 0.2, batch_size: int = 16, lr: float = 1e-4):

    """Initializes hardware device, DataLoaders, model instance, and optimizer."""
    #Device Selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using compute device: {device}")

    #Dataset Loading and Train/Val Split
    full_dataset = PhotoEditDataset()
    total_samples = len(full_dataset)

    if total_samples == 0:
        raise ValueError(
            "Dataset is empty."
        )
    
    val_size = int(total_samples * val_split)
    train_size = total_samples - val_size

    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size]
    )
    print(f"[INFO] Data Split: {train_size} training samples, {val_size} validation samples")

    #DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    #Model and Optimizer Initialization
    model = PhotoParamRegressor(num_target_params=31, pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    return device, train_loader, val_loader, model, optimizer

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> float:
    """Executes a single training epoch with backpropagation"""
    model.train()
    running_loss = 0.0

    for images, targets in dataloader:
        images, targets = images.to(device), targets.to(device)

        #Clear previous gradients
        optimizer.zero_grad()
        #Forward pass
        predictions = model(images)
        #Compute loss
        loss = criterion(predictions, targets)
        #Backward pass and optimizer updatte
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
    
    return running_loss / len(dataloader.dataset)

def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> float:
    """Evaluates model performance on validation data without computing gradients"""
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for images, targets in dataloader:
            images, targets = images.to(device), targets.to(device)

            predictions = model(images)
            loss = criterion(predictions, targets)

            running_loss += loss.item() * images.size(0)
        
    return running_loss / len(dataloader.dataset)


def main(epochs: int = 50, batch_size: int = 16, lr: float = 1e-4):
    """Main training execution loop with checkpoint management"""
    #Setup components
    device, train_loader, val_loader, model, optimizer = setup_training(
        val_split=0.2, batch_size= batch_size, lr=lr
    )
    criterion = MaskedParamLoss()

    #Prepare Checkpoint Directory
    checkpoint_dir = config.CHECKPOINT_DIR
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_model_path = os.path.join(checkpoint_dir, "best_model.pth")

    best_val_loss = float("inf")

    print(f"\n[INFO] Starting training loop for {epochs} epochs...")

    #Training Loop
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f}"
        )

        #Save Checkpoint on Validation Improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                },
                best_model_path,
            )
            print(f"  --> Saved new best model checkpoint to {best_model_path}")

    print(f"\n[SUCCESS] Training completed. Lowest Validation Loss: {best_val_loss:.6f}")

if __name__ == "__main__":
    main()