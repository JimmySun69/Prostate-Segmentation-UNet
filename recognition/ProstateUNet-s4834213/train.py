# train.py
"""
Main training script for the Improved 3D U-Net, using a fully custom,
non-monai pipeline for loss and metrics.
"""
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

# Local imports from your project modules
from modules import ImprovedUNet3D_CAN
from dataset import get_dataloaders, infer_num_classes

# ===================================================================
# Custom Loss Function (formerly from MONAI)
# ===================================================================

class DiceCELoss(nn.Module):
    """
    A from-scratch implementation that combines Dice Loss and Cross Entropy Loss.
    """
    def __init__(self, ce_weight=1.0, dice_weight=1.0, smooth=1e-6):
        super().__init__()
        # Use PyTorch's built-in CrossEntropyLoss
        self.cross_entropy = nn.CrossEntropyLoss()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth

    def forward(self, logits, targets):
        # --- Cross Entropy Loss ---
        # CrossEntropyLoss expects target of shape (N, D, H, W), not (N, 1, D, H, W)
        ce_loss = self.cross_entropy(logits, targets.squeeze(1).long())

        # --- Dice Loss ---
        probs = F.softmax(logits, dim=1)
        num_classes = logits.shape[1]
        
        # One-hot encode the target tensor
        targets_onehot = F.one_hot(targets.squeeze(1).long(), num_classes).permute(0, 4, 1, 2, 3)
        
        # Sum over the spatial dimensions (D, H, W)
        dims = (2, 3, 4)
        intersection = torch.sum(probs * targets_onehot, dims)
        cardinality = torch.sum(probs + targets_onehot, dims)
        
        dice_score = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        # Average the Dice score across the batch and classes
        dice_loss = 1. - dice_score.mean()

        # Combine the two losses
        return self.ce_weight * ce_loss + self.dice_weight * dice_loss

# ===================================================================
# Custom Metric Calculation (formerly from MONAI)
# ===================================================================

def calculate_dice_per_class(logits, targets, num_classes, smooth=1e-6):
    """
    Calculates the Dice score for each class separately across a batch.
    Returns a numpy array of shape (num_classes,).
    """
    probs = F.softmax(logits, dim=1)
    preds = torch.argmax(probs, dim=1)
    
    preds_onehot = F.one_hot(preds, num_classes).permute(0, 4, 1, 2, 3)
    targets_onehot = F.one_hot(targets.squeeze(1).long(), num_classes).permute(0, 4, 1, 2, 3)
    
    # Sum over batch and spatial dimensions (N, D, H, W) to get per-class sums
    dims = (0, 2, 3, 4)
    intersection = torch.sum(preds_onehot * targets_onehot, dims)
    cardinality = torch.sum(preds_onehot + targets_onehot, dims)
    
    dice_score = (2. * intersection + smooth) / (cardinality + smooth)
    return dice_score.cpu().numpy()

# ===================================================================
# Main Training Loop
# ===================================================================

def train(args):
    """The main training loop using custom components."""
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    num_classes, class_map = infer_num_classes(args.labels_dir)
    train_loader, val_loader = get_dataloaders(
        args.images_dir, args.labels_dir,
        batch_size=args.batch_size,
        num_workers=args.workers
    )

    model = ImprovedUNet3D_CAN(in_ch=1, out_ch=num_classes, base=args.base).to(device)
    criterion = DiceCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.out_dir, exist_ok=True)
    train_losses = []
    best_val_dice = -1.0

    for ep in range(1, args.epochs + 1):
        print(f"\n--- Epoch {ep}/{args.epochs} ---")
        model.train()
        epoch_loss = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        for batch in progress_bar:
            imgs, lbls = batch["image"].to(device), batch["label"].to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, lbls)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        epoch_loss /= len(train_loader)
        train_losses.append(epoch_loss)
        print(f"Train Loss: {epoch_loss:.4f}")

        # Validation
        model.eval()
        all_val_dice_scores = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                imgs, lbls = batch["image"].to(device), batch["label"].to(device)
                val_logits = model(imgs)
                dice_scores = calculate_dice_per_class(val_logits, lbls, num_classes)
                all_val_dice_scores.append(dice_scores)
        
        # Average Dice scores across all validation cases to get a final score per class
        mean_dice_per_class = np.mean(all_val_dice_scores, axis=0)
        # Calculate the mean dice, excluding the background class (class 0)
        mean_val_dice_no_bg = np.mean(mean_dice_per_class[1:])
        print(f"Validation Mean Dice (no BG): {mean_val_dice_no_bg:.4f}")

        if mean_val_dice_no_bg > best_val_dice:
            best_val_dice = mean_val_dice_no_bg
            torch.save(model.state_dict(), os.path.join(args.out_dir, "best_model.pth"))
            print(f"Saved best model (Val Dice = {best_val_dice:.4f})")

    # Plotting the training loss
    plt.figure("Training Loss Curve")
    plt.plot(train_losses, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (DiceCE)")
    plt.legend()
    plt.title("Training Loss Progress")
    plt.savefig(os.path.join(args.out_dir, "training_curve.png"), dpi=300)
    plt.close()

    print(f"\nTraining complete. Best validation Mean Dice (no BG): {best_val_dice:.4f}")

    # Final evaluation using the best saved model
    print("\nEvaluating final Dice score per class on validation set...")
    model.load_state_dict(torch.load(os.path.join(args.out_dir, "best_model.pth")))
    model.eval() # Ensure model is in eval mode
    final_dice_scores_list = []
    with torch.no_grad():
        for batch in val_loader:
            imgs, lbls = batch["image"].to(device), batch["label"].to(device)
            final_logits = model(imgs)
            dice_scores = calculate_dice_per_class(final_logits, lbls, num_classes)
            final_dice_scores_list.append(dice_scores)

    final_dice_scores = np.mean(final_dice_scores_list, axis=0)
    
    print("\n--- Final Per-Class Dice Scores ---")
    with open(os.path.join(args.out_dir, "final_dice_score.txt"), "w") as f:
        for i, score in enumerate(final_dice_scores):
            class_name = class_map.get(i, f"Class {i}")
            print(f"{class_name}: {score:.4f}")
            f.write(f"Dice score for {class_name}: {score:.4f}\n")
        
        mean_score_no_bg = np.mean(final_dice_scores[1:])
        print(f"\nMean Dice (no BG): {mean_score_no_bg:.4f}")
        f.write(f"\nMean Dice (no BG): {mean_score_no_bg:.4f}\n")

    print(f"\nSaved final Dice scores to {args.out_dir}/final_dice_score.txt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an Improved 3D U-Net model without MONAI.")
    parser.add_argument('--images_dir', type=str, default="/home/groups/comp3710/HipMRI_Study_open/semantic_MRs", help='Directory for input images.')
    parser.add_argument('--labels_dir', type=str, default="/home/groups/comp3710/HipMRI_Study_open/semantic_labels_only", help='Directory for label maps.')
    parser.add_argument('--out_dir', type=str, default="runs/Custom_UNet3D", help='Directory for saving outputs.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training.')
    parser.add_argument('--base', type=int, default=32, help='Base number of channels for the U-Net.')
    parser.add_argument('--workers', type=int, default=4, help='Number of workers for data loading.')

    args = parser.parse_args()
    train(args)
