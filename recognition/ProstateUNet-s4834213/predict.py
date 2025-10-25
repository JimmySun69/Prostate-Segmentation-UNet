# predict.py
"""
Loads a trained model and runs prediction on validation cases, saving a
visualization for each one. This script uses a custom, non-monai sliding
window inferer for robust prediction on full-sized images.
"""
import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import scipy.ndimage

# Local imports from your project modules
from modules import ImprovedUNet3D_CAN
from dataset import (
    collect_case_pairs,
    infer_num_classes,
    ProstateDataset, # We can reuse the dataset class
    Compose,
    LoadNifti,
    EnsureChannelFirst,
    NormalizeIntensity,
    ToTensor,
)
from torch.utils.data import DataLoader

# ===================================================================
# Custom Sliding Window Inference (formerly from MONAI)
# ===================================================================

def sliding_window_inference(inputs, roi_size, sw_batch_size, model, overlap=0.5, mode="gaussian"):
    """
    A from-scratch implementation of sliding window inference for 3D images.

    Args:
        inputs (torch.Tensor): The input image tensor of shape (1, 1, D, H, W).
        roi_size (tuple): The size of the sliding window (patch size), e.g., (96, 96, 96).
        sw_batch_size (int): The batch size for processing windows.
        model (torch.nn.Module): The segmentation model.
        overlap (float): The overlap ratio between adjacent windows (0.0 to 1.0).
        mode (str): The blending mode, "constant" or "gaussian".
    """
    device = inputs.device
    image_size = inputs.shape[2:]
    num_classes = model.outc.out_channels

    # Allocate tensors for the output and the normalization count
    output_image = torch.zeros((1, num_classes) + tuple(image_size), device=device)
    count_map = torch.zeros((1, 1) + tuple(image_size), device=device)

    # Generate the importance map (blending weights)
    if mode == "gaussian":
        # Create a Gaussian importance map that gives more weight to the center of the patch
        # sigma=1/8 of patch size is a common choice
        importance_map = np.zeros(roi_size, dtype=np.float32)
        center_coords = [i // 2 for i in roi_size]
        sigmas = [i * 0.125 for i in roi_size]
        importance_map[tuple(center_coords)] = 1
        importance_map = scipy.ndimage.gaussian_filter(importance_map, sigmas, 0, mode='constant', cval=0)
        importance_map = torch.from_numpy(importance_map).to(device)
    else: # "constant" mode
        importance_map = torch.ones(roi_size, device=device)

    # Calculate the step size for sliding the window
    step_size = [int(s * (1 - overlap)) for s in roi_size]
    
    # Generate all patch coordinates
    coords = []
    for z in range(0, image_size[0] - roi_size[0] + 1, step_size[0]):
        for y in range(0, image_size[1] - roi_size[1] + 1, step_size[1]):
            for x in range(0, image_size[2] - roi_size[2] + 1, step_size[2]):
                coords.append((z, y, x))

    # Process windows in batches
    for i in range(0, len(coords), sw_batch_size):
        batch_coords = coords[i:i + sw_batch_size]
        image_patches = []
        for z, y, x in batch_coords:
            patch = inputs[
                :, :, z:z + roi_size[0], y:y + roi_size[1], x:x + roi_size[2]
            ]
            image_patches.append(patch)
        
        image_patches_tensor = torch.cat(image_patches, dim=0)
        
        # Run model on the batch of patches
        logits_patches = model(image_patches_tensor)

        # Stitch the results back into the output image
        for j, (z, y, x) in enumerate(batch_coords):
            roi = (
                slice(z, z + roi_size[0]),
                slice(y, y + roi_size[1]),
                slice(x, x + roi_size[2]),
            )
            output_image[0, :, roi[0], roi[1], roi[2]] += logits_patches[j] * importance_map
            count_map[0, 0, roi[0], roi[1], roi[2]] += importance_map
    
    # Normalize the output by the count map to average the overlapping predictions
    output_image /= count_map
    return output_image


# ===================================================================
# Main Prediction and Visualization Logic
# ===================================================================

def visualize_prediction_prostate(image, label, prediction, save_path, slice_idx=None):
    """(Unchanged) Saves a plot comparing image, label, and prediction."""
    image = image.cpu().numpy().squeeze()
    label = label.cpu().numpy().squeeze()
    prediction = prediction.cpu().numpy().squeeze()
    
    prostate_pred = (prediction == 5).astype(np.uint8) # Assuming prostate is class 5
    prostate_label = (label == 5).astype(np.uint8)

    if slice_idx is None:
        slice_scores = [np.sum(label[:, :, i] == 5) for i in range(label.shape[2])]
        slice_idx = np.argmax(slice_scores) if max(slice_scores) > 0 else image.shape[2] // 2

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    titles = ['Input Image', 'Ground Truth (All)', 'Prediction (All)', 'GT Prostate', 'Predicted Prostate']
    
    # Plotting logic
    axes[0].imshow(image[:, :, slice_idx], cmap='gray')
    axes[1].imshow(label[:, :, slice_idx], cmap='jet', vmin=0, vmax=5)
    axes[2].imshow(prediction[:, :, slice_idx], cmap='jet', vmin=0, vmax=5)
    axes[3].imshow(image[:, :, slice_idx], cmap='gray')
    axes[3].imshow(prostate_label[:, :, slice_idx], cmap='Greens', alpha=0.6)
    axes[4].imshow(image[:, :, slice_idx], cmap='gray')
    axes[4].imshow(prostate_pred[:, :, slice_idx], cmap='Reds', alpha=0.6)

    for ax, title in zip(axes, titles):
        ax.set_title(title)
        ax.axis('off')

    plt.suptitle(f'Comparison at Slice {slice_idx}')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved visualization to {save_path}")

def predict(args):
    """Main prediction function using custom components."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    pairs = collect_case_pairs(args.images_dir, args.labels_dir)
    data_dicts = [{"image": i, "label": l} for i, l in pairs]
    n_train = int(0.8 * len(data_dicts))
    val_files = data_dicts[n_train:]

    # Use the same validation transforms as in training
    val_transforms = Compose([
        LoadNifti(),
        EnsureChannelFirst(),
        NormalizeIntensity(),
        ToTensor(),
    ])

    val_ds = ProstateDataset(data_dicts=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=args.workers)
    
    num_classes, _ = infer_num_classes(args.labels_dir)

    model = ImprovedUNet3D_CAN(in_ch=1, out_ch=num_classes, base=args.base).to(device)
    model_path = os.path.join(args.out_dir, "best_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please run train.py first.")
        
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        for i, case_data in enumerate(tqdm(val_loader, desc="Predicting")):
            image, label = case_data["image"].to(device), case_data["label"].to(device)
            
            # Use the custom sliding window inferer
            pred_logits = sliding_window_inference(
                inputs=image,
                roi_size=(96, 96, 96), # Must match training patch size
                sw_batch_size=4,
                model=model,
                overlap=0.5,
                mode="gaussian"
            )
            prediction = torch.argmax(pred_logits, dim=1)

            original_filepath = val_files[i]["image"]
            original_filename = Path(original_filepath).stem
            save_path = os.path.join(args.out_dir, f'prediction_{original_filename}.png')
            
            visualize_prediction_prostate(image, label, prediction, save_path)
            
    print(f"\nCompleted generating predictions for {len(val_loader)} validation images.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run prediction and visualize results without MONAI.")
    parser.add_argument('--images_dir', type=str, default="/home/groups/comp3710/HipMRI_Study_open/semantic_MRs", help='Directory for input images.')
    parser.add_argument('--labels_dir', type=str, default="/home/groups/comp3710/HipMRI_Study_open/semantic_labels_only", help='Directory for label maps.')
    parser.add_argument('--out_dir', type=str, default="runs/Custom_UNet3D", help='Directory where the trained model is and outputs will be placed.')
    parser.add_argument('--base', type=int, default=32, help='Base number of channels for the U-Net (must match training).')
    parser.add_argument('--workers', type=int, default=4, help='Number of workers for data loading.')

    args = parser.parse_args()
    predict(args)
