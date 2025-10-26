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
import scipy.ndimage  # For generating the Gaussian importance map

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
    This allows inference on images larger than the training patch size.

    Args:
        inputs (torch.Tensor): The input image tensor of shape (1, 1, D, H, W).
        roi_size (tuple): The size of the sliding window (patch size), e.g., (96, 96, 96).
        sw_batch_size (int): The batch size for processing windows.
        model (torch.nn.Module): The segmentation model.
        overlap (float): The overlap ratio between adjacent windows (0.0 to 1.0).
        mode (str): The blending mode, "constant" or "gaussian".
    """
    device = inputs.device
    image_size = inputs.shape[2:]  # (D, H, W)
    num_classes = model.outc.out_channels # Get num_classes from model's final layer

    # Allocate tensors for the final output and the normalization map
    # output_image stores the sum of logits
    output_image = torch.zeros((1, num_classes) + tuple(image_size), device=device)
    # count_map stores the sum of weights (e.g., Gaussian weights)
    count_map = torch.zeros((1, 1) + tuple(image_size), device=device)

    # Generate the importance map (blending weights)
    if mode == "gaussian":
        # Create a Gaussian importance map that gives more weight to the center
        importance_map_np = np.zeros(roi_size, dtype=np.float32)
        center_coords = [i // 2 for i in roi_size]
        sigmas = [i * 0.125 for i in roi_size]  # Sigma is 1/8 of patch size
        importance_map_np[tuple(center_coords)] = 1
        # Apply Gaussian filter to the single-point delta function
        importance_map_np = scipy.ndimage.gaussian_filter(importance_map_np, sigmas, 0, mode='constant', cval=0)
        importance_map_np /= importance_map_np.max() # Normalize to [0, 1]
        importance_map = torch.from_numpy(importance_map_np).to(device)
    else: # "constant" mode (simple averaging)
        importance_map = torch.ones(roi_size, device=device)

    # Calculate the step size for sliding the window
    step_size = [int(s * (1 - overlap)) for s in roi_size]
    
    # Generate all patch coordinates (top-left corners)
    coords = []
    for z in range(0, image_size[0] - roi_size[0] + 1, step_size[0]):
        for y in range(0, image_size[1] - roi_size[1] + 1, step_size[1]):
            for x in range(0, image_size[2] - roi_size[2] + 1, step_size[2]):
                coords.append((z, y, x))

    # Process windows in batches
    for i in range(0, len(coords), sw_batch_size):
        batch_coords = coords[i:i + sw_batch_size]
        image_patches = []
        
        # Crop patches from the input image
        for z, y, x in batch_coords:
            patch = inputs[
                :, :, z:z + roi_size[0], y:y + roi_size[1], x:x + roi_size[2]
            ]
            image_patches.append(patch)
        
        image_patches_tensor = torch.cat(image_patches, dim=0) # [sw_batch_size, C, D, H, W]
        
        # Run model on the batch of patches
        logits_patches = model(image_patches_tensor) # [sw_batch_size, n_classes, D, H, W]

        # Stitch the results back into the output image
        for j, (z, y, x) in enumerate(batch_coords):
            # Define the Region of Interest (ROI) in the full output_image
            roi = (
                slice(z, z + roi_size[0]),
                slice(y, y + roi_size[1]),
                slice(x, x + roi_size[2]),
            )
            # Add the patch logits, weighted by the importance map
            output_image[0, :, roi[0], roi[1], roi[2]] += logits_patches[j] * importance_map
            # Add the weights to the count map
            count_map[0, 0, roi[0], roi[1], roi[2]] += importance_map
    
    # Normalize the output by the count map to average the overlapping predictions
    output_image /= count_map
    return output_image


# ===================================================================
# Main Prediction and Visualization Logic
# ===================================================================

def visualize_prediction_prostate(image, label, prediction, save_path, slice_idx=None):
    """(Unchanged) Saves a plot comparing image, label, and prediction."""
    # Move to CPU and convert to numpy, remove batch/channel dims
    image = image.cpu().numpy().squeeze()
    label = label.cpu().numpy().squeeze()
    prediction = prediction.cpu().numpy().squeeze()
    
    # Isolate the prostate class (hardcoded as 5) for clearer visualization
    prostate_pred = (prediction == 5).astype(np.uint8)
    prostate_label = (label == 5).astype(np.uint8)

    # If no slice is specified, find the slice with the most prostate label
    if slice_idx is None:
        slice_scores = [np.sum(label[:, :, i] == 5) for i in range(label.shape[2])]
        slice_idx = np.argmax(slice_scores) if max(slice_scores) > 0 else image.shape[2] // 2

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    titles = ['Input Image', 'Ground Truth (All)', 'Prediction (All)', 'GT Prostate', 'Predicted Prostate']
    
    # Plotting logic
    axes[0].imshow(image[:, :, slice_idx], cmap='gray')
    axes[1].imshow(label[:, :, slice_idx], cmap='jet', vmin=0, vmax=5)
    axes[2].imshow(prediction[:, :, slice_idx], cmap='jet', vmin=0, vmax=5)
    # Overlays
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
    
    # --- 80/10/10 Split ---
    n_total = len(data_dicts)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    
    # Select the final 10% as the test set for prediction
    val_files = data_dicts[n_train + n_val:] 
    # --- End Split ---

    # Use the same transforms as validation (load full image)
    val_transforms = Compose([
        LoadNifti(),
        EnsureChannelFirst(),
        NormalizeIntensity(),
        ToTensor(),
    ])

    val_ds = ProstateDataset(data_dicts=val_files, transform=val_transforms)
    # This DataLoader now loads the *test set*
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=args.workers)
    
    num_classes, _ = infer_num_classes(args.labels_dir)

    model = ImprovedUNet3D_CAN(in_ch=1, out_ch=num_classes, base=args.base).to(device)
    model_path = os.path.join(args.out_dir, "best_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please run train.py first.")
        
    # Load the best model saved during training
    model.load_state_dict(torch.load(model_path))
    model.eval() # Set model to evaluation mode (disables dropout, etc.)

    # Disable gradient calculations for inference
    with torch.no_grad():
        for i, case_data in enumerate(tqdm(val_loader, desc="Predicting on Test Set")):
            image, label = case_data["image"].to(device), case_data["label"].to(device)
            
            # Use the custom sliding window inferer
            pred_logits = sliding_window_inference(
                inputs=image,
                roi_size=(96, 96, 96), # Must match training patch size
                sw_batch_size=4,       # Process 4 patches at a time
                model=model,
                overlap=0.5,           # 50% overlap
                mode="gaussian"        # Use Gaussian blending
            )
            
            # Get final class predictions from logits
            prediction = torch.argmax(pred_logits, dim=1)

            # --- Save Visualization ---
            original_filepath = val_files[i]["image"]
            original_filename = Path(original_filepath).stem
            save_path = os.path.join(args.out_dir, f'prediction_{original_filename}.png')
            
            visualize_prediction_prostate(image, label, prediction, save_path)
            
    print(f"\nCompleted generating predictions for {len(val_loader)} test images.")


if __name__ == "__main__":
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run prediction and visualize results without MONAI.")
    parser.add_argument('--images_dir', type=str, default="/home/groups/comp3710/HipMRI_Study_open/semantic_MRs", help='Directory for input images.')
    parser.add_argument('--labels_dir', type=str, default="/home/groups/comp3710/HipMRI_Study_open/semantic_labels_only", help='Directory for label maps.')
    parser.add_argument('--out_dir', type=str, default="runs/Custom_UNet3D", help='Directory where the trained model is and outputs will be placed.')
    parser.add_argument('--base', type=int, default=32, help='Base number of channels for the U-Net (must match training).')
    parser.add_argument('--workers', type=int, default=4, help='Number of workers for data loading.')

    args = parser.parse_args()
    predict(args)
