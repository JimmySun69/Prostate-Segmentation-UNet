"""
train.py
Main training script for the Improved 3D U-Net.
"""
import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

# Local imports from your project modules
from modules import ImprovedUNet3D_CAN, DiceSquaredFocalLoss
from dataset import get_dataloaders, infer_num_classes
from monai.utils import set_determinism
from monai.transforms import AsDiscrete
from monai.metrics import DiceMetric

def evaluate_dice(model, dataloader, device, num_classes, include_background=True):
    """Calculates the mean Dice score on a given dataset."""
    model.eval()
    post_pred = AsDiscrete(argmax=True, to_onehot=num_classes)
    post_label = AsDiscrete(to_onehot=num_classes)
    dice_metric = DiceMetric(include_background=include_background, reduction="mean")

    with torch.no_grad():
        for batch in dataloader:
            imgs, lbls = batch["image"].to(device), batch["label"].to(device)
            preds_logits = model(imgs)
            preds_onehot = post_pred(preds_logits)
            lbls_onehot = post_label(lbls)
            dice_metric(y_pred=preds_onehot, y=lbls_onehot)

    mean_dice = dice_metric.aggregate().item()
    dice_metric.reset()
    return mean_dice

def train(args):
    """The main training loop."""
    set_determinism(seed=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    num_classes = infer_num_classes(args.labels_dir)
    train_loader, val_loader = get_dataloaders(
        args.images_dir, args.labels_dir,
        batch_size=args.batch_size,
        num_workers=args.workers
    )

    model = ImprovedUNet3D_CAN(in_ch=1, out_ch=num_classes, base=args.base).to(device)
    criterion = DiceSquaredFocalLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.out_dir, exist_ok=True)
    train_losses, val_losses = [], []
    best_val_loss = float('inf')

    for ep in range(1, args.epochs + 1):
        print(f"\n--- Epoch {ep}/{args.epochs} ---")
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            imgs, lbls = batch["image"].to(device), batch["label"].to(device).long()
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, lbls)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(train_loader)
        train_losses.append(epoch_loss)
        print(f"Train Loss: {epoch_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                imgs, lbls = batch["image"].to(device), batch["label"].to(device).long()
                val_loss += criterion(model(imgs), lbls).item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        print(f"Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.out_dir, "best_model.pth"))
            print(f"Saved best model (Val Loss = {best_val_loss:.4f})")

    # Plot training curves
    plt.figure("Training Progress")
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training Progress (DSF Loss)")
    plt.savefig(os.path.join(args.out_dir, "training_curve.png"), dpi=300)
    plt.close()

    print(f"\nTraining complete. Best validation loss: {best_val_loss:.4f}")

    # Final Dice evaluation
    print("\nEvaluating final Dice score on validation set...")
    model.load_state_dict(torch.load(os.path.join(args.out_dir, "best_model.pth")))
    final_dice = evaluate_dice(model, val_loader, device, num_classes)
    print(f"Mean Dice on Validation Set: {final_dice:.4f}")

    # Save final Dice score
    with open(os.path.join(args.out_dir, "final_dice_score.txt"), "w") as f:
        f.write(f"Mean Validation Dice: {final_dice:.4f}\n")
    print(f"Saved final Dice score to {args.out_dir}/final_dice_score.txt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an Improved 3D U-Net model.")
    parser.add_argument('--images_dir', type=str, default="/home/groups/comp3710/HipMRI_Study_open/semantic_MRs", help='Directory for input images.')
    parser.add_argument('--labels_dir', type=str, default="/home/groups/comp3710/HipMRI_Study_open/semantic_labels_only", help='Directory for label maps.')
    parser.add_argument('--out_dir', type=str, default="runs/ImprovedUNet3D_CAN_DSF", help='Directory for saving outputs.')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs.')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate.')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training.')
    parser.add_argument('--base', type=int, default=32, help='Base number of channels for the U-Net.')
    parser.add_argument('--workers', type=int, default=4, help='Number of workers for data loading.')

    args = parser.parse_args()
    train(args)
