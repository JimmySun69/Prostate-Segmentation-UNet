# predict.py

"""
Inference + dual visualization (5-panel and overlay/split style).

- dataset's class order:
  0 Background, 1 Body Outline, 2 Bone, 3 Bladder, 4 Rectum, 5 Prostate
- Draws body lightly, organs on top; fixed legend and palette.
- Visualize BOTH styles for the first 5 test cases.
"""

import os
import argparse
from pathlib import Path

import numpy as np
import scipy.ndimage
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader

# Project imports
from modules import ImprovedUNet3D_CAN
from dataset import (
    collect_case_pairs,
    infer_num_classes,
    ProstateDataset,
    Compose,
    LoadNifti,
    EnsureChannelFirst,
    NormalizeIntensity,
    ToTensor,
)

# ==============================
# Sliding-window inference
# ==============================
def sliding_window_inference(inputs, roi_size, sw_batch_size, model, overlap=0.5, mode="gaussian"):
    """
    Minimal sliding-window inference for 3D volumes.
    inputs: [1, 1, D, H, W]
    returns: logits [1, C, D, H, W]
    """
    device = inputs.device
    D, H, W = inputs.shape[2:]
    num_classes = model.outc.out_channels  # final channels

    out = torch.zeros((1, num_classes, D, H, W), device=device)
    norm = torch.zeros((1, 1, D, H, W), device=device)

    # importance map
    if mode == "gaussian":
        imp = np.zeros(roi_size, dtype=np.float32)
        center = [r // 2 for r in roi_size]
        sigmas = [r * 0.125 for r in roi_size]
        imp[tuple(center)] = 1.0
        imp = scipy.ndimage.gaussian_filter(imp, sigmas, 0, mode="constant", cval=0)
        imp = imp / max(imp.max(), 1e-8)
        imp = torch.from_numpy(imp).to(device)
    else:
        imp = torch.ones(roi_size, device=device)

    # strides
    strides = [max(1, int(r * (1 - overlap))) for r in roi_size]
    z_starts = list(range(0, max(D - roi_size[0], 0) + 1, strides[0])) or [0]
    y_starts = list(range(0, max(H - roi_size[1], 0) + 1, strides[1])) or [0]
    x_starts = list(range(0, max(W - roi_size[2], 0) + 1, strides[2])) or [0]
    if z_starts[-1] != D - roi_size[0]: z_starts.append(max(D - roi_size[0], 0))
    if y_starts[-1] != H - roi_size[1]: y_starts.append(max(H - roi_size[1], 0))
    if x_starts[-1] != W - roi_size[2]: x_starts.append(max(W - roi_size[2], 0))
    coords = [(z, y, x) for z in z_starts for y in y_starts for x in x_starts]

    for i in range(0, len(coords), sw_batch_size):
        batch_coords = coords[i:i+sw_batch_size]
        patches = []
        for z, y, x in batch_coords:
            patches.append(inputs[:, :, z:z+roi_size[0], y:y+roi_size[1], x:x+roi_size[2]])
        patches = torch.cat(patches, dim=0)  # [B,1,d,h,w]

        logits = model(patches)  # [B,C,d,h,w]

        for j, (z, y, x) in enumerate(batch_coords):
            zs, ys, xs = slice(z, z+roi_size[0]), slice(y, y+roi_size[1]), slice(x, x+roi_size[2])
            out[0, :, zs, ys, xs] += logits[j] * imp
            norm[0, 0, zs, ys, xs] += imp

    out /= torch.clamp(norm, min=1e-6)
    return out


# ==============================
# Visualization configuration
# ==============================
CLASS_NAMES  = ["Background", "Body Outline", "Bone", "Bladder", "Rectum", "Prostate"]
PROSTATE_ID  = 5
CLASS_COLORS = [
    (0.00, 0.00, 0.00),   # 0 Background - black
    (0.10, 0.60, 1.00),   # 1 Body Outline - cyan/blue
    (0.20, 1.00, 0.80),   # 2 Bone - teal
    (1.00, 0.45, 0.10),   # 3 Bladder - orange
    (0.10, 0.80, 0.20),   # 4 Rectum - green
    (0.70, 0.00, 0.20),   # 5 Prostate - maroon
]

def _window_image(img2d, p_low=1, p_high=99):
    m = img2d[img2d > 0]
    if m.size == 0:
        mn, mx = float(img2d.min()), float(img2d.max())
        return (img2d - mn) / max(mx - mn, 1e-6)
    lo, hi = np.percentile(m, [p_low, p_high])
    lo, hi = float(lo), float(hi if hi > lo else lo + 1e-6)
    img2d = np.clip(img2d, lo, hi)
    return (img2d - lo) / (hi - lo)

def _find_best_slice_for_prostate(label3d, prostate_id=PROSTATE_ID):
    counts = [int(np.sum(label3d[:, :, k] == prostate_id)) for k in range(label3d.shape[2])]
    return int(np.argmax(counts)) if max(counts) > 0 else int(label3d.shape[2] // 2)

def _add_legend(ax):
    import matplotlib.patches as mpatches
    handles = [mpatches.Patch(color=CLASS_COLORS[i], label=CLASS_NAMES[i]) for i in range(len(CLASS_NAMES))]
    ax.legend(handles=handles, loc="lower center", ncol=3, fontsize=7, frameon=False)

def _draw_overlay(ax, img2d, mask2d, alpha_img=0.92):
    ax.imshow(img2d, cmap="gray", alpha=alpha_img)
    # draw body lightly, then organs
    layered = [
        (1, CLASS_COLORS[1], 0.15),                  # Body Outline (light)
        (3, CLASS_COLORS[3], 0.55),                  # Bladder
        (4, CLASS_COLORS[4], 0.55),                  # Rectum
        (PROSTATE_ID, CLASS_COLORS[PROSTATE_ID], 0.65),  # Prostate
        (2, CLASS_COLORS[2], 0.40),                  # Bone (optional)
    ]
    H, W = mask2d.shape
    rgb = np.zeros((H, W, 3), dtype=np.float32)
    for cid, col, a in layered:
        m = (mask2d == cid)
        if m.any():
            rgb[:] = 0.0
            rgb[m] = col
            ax.imshow(rgb, alpha=a, interpolation="nearest")
    ax.set_axis_off()

def _draw_split(ax, gt2d, pred2d):
    h, w = gt2d.shape
    split = w // 2
    panel = np.zeros_like(gt2d)
    panel[:, :split] = gt2d[:, :split]
    panel[:, split:] = pred2d[:, split:]
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    for cid, col in enumerate(CLASS_COLORS):
        rgb[panel == cid] = col
    ax.imshow(rgb, interpolation="nearest")
    ax.axvline(split, color="white", linestyle="--", linewidth=2)
    ax.set_title("GT (left) | Prediction (right)")
    ax.set_axis_off()


# ==============================
# 5-panel visualization
# ==============================
def visualize_original_five_panel(image_t, label_t, pred_t, save_path, slice_idx=None):
    image = image_t.cpu().numpy().squeeze()   # [D,H,W]
    label = label_t.cpu().numpy().squeeze()
    pred  = pred_t.cpu().numpy().squeeze()

    if slice_idx is None:
        slice_idx = _find_best_slice_for_prostate(label, prostate_id=PROSTATE_ID)

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    titles = ['Input Image', 'Ground Truth (All)', 'Prediction (All)', 'GT Prostate', 'Predicted Prostate']
    for ax, t in zip(axes, titles): ax.set_title(t)

    axes[0].imshow(image[:, :, slice_idx], cmap='gray'); axes[0].set_axis_off()
    axes[1].imshow(label[:, :, slice_idx], cmap='jet', vmin=0, vmax=5); axes[1].set_axis_off()
    axes[2].imshow(pred[:, :, slice_idx],  cmap='jet', vmin=0, vmax=5); axes[2].set_axis_off()

    gt_mask = (label[:, :, slice_idx] == PROSTATE_ID).astype(np.uint8)
    pr_mask = (pred[:, :,  slice_idx] == PROSTATE_ID).astype(np.uint8)

    axes[3].imshow(image[:, :, slice_idx], cmap='gray'); axes[3].imshow(gt_mask, cmap='Greens', alpha=0.6); axes[3].set_axis_off()
    axes[4].imshow(image[:, :, slice_idx], cmap='gray'); axes[4].imshow(pr_mask, cmap='Reds',   alpha=0.6); axes[4].set_axis_off()

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches='tight', dpi=200)
    plt.close(fig)


# ==============================
# Overlay/split visualization
# ==============================
def visualize_reference_style(image_t, label_t, pred_t, save_path, slice_idx=None):
    image = image_t.cpu().numpy().squeeze()
    label = label_t.cpu().numpy().squeeze()
    pred  = pred_t.cpu().numpy().squeeze()

    if slice_idx is None:
        slice_idx = _find_best_slice_for_prostate(label, prostate_id=PROSTATE_ID)

    img2d = _window_image(image[:, :, slice_idx])
    gt2d  = label[:, :, slice_idx].astype(np.int32)
    pr2d  = pred[:, :,  slice_idx].astype(np.int32)

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    axes[0].imshow(img2d, cmap="gray"); axes[0].set_title("Original Slice"); axes[0].set_axis_off()
    _draw_overlay(axes[1], img2d, gt2d);  axes[1].set_title("Ground Truth Overlay"); _add_legend(axes[1])
    _draw_overlay(axes[2], img2d, pr2d);  axes[2].set_title("Prediction Overlay")
    _draw_split(axes[3], gt2d, pr2d)

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches='tight', dpi=200)
    plt.close(fig)


# ==============================
# Main
# ==============================
def predict(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device.type}")

    pairs = collect_case_pairs(args.images_dir, args.labels_dir)
    files = [{"image": i, "label": l} for i, l in pairs]

    n_total = len(files)
    n_train = int(0.8 * n_total)
    n_val   = int(0.1 * n_total)
    test_files = files[n_train + n_val:]  # last 10%

    # loader
    tfm = Compose([LoadNifti(), EnsureChannelFirst(), NormalizeIntensity(), ToTensor()])
    ds  = ProstateDataset(data_dicts=test_files, transform=tfm)
    loader = DataLoader(ds, batch_size=1, num_workers=args.workers)

    # model
    num_classes, _ = infer_num_classes(args.labels_dir)
    model = ImprovedUNet3D_CAN(in_ch=1, out_ch=num_classes, base=args.base).to(device)
    model_path = os.path.join(args.out_dir, "best_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing model weights at {model_path}. Train first.")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    os.makedirs(args.out_dir, exist_ok=True)

    # generate BOTH styles for the first 5 cases
    num_cases = 5
    produced = 0

    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc="Predicting")):
            image = batch["image"].to(device)
            label = batch["label"].to(device)

            logits = sliding_window_inference(
                image, roi_size=(96, 96, 96), sw_batch_size=4, model=model, overlap=0.5, mode="gaussian"
            )
            pred = torch.argmax(logits, dim=1)  # [1, D, H, W]

            stem = Path(test_files[i]["image"]).stem

            # original 5-panel
            p1 = os.path.join(args.out_dir, f"prediction_{stem}.png")
            visualize_original_five_panel(image, label, pred, p1)
            print(f"Saved original-style: {p1}")

            # overlay/split
            p2 = os.path.join(args.out_dir, f"prediction_{stem}_overlay.png")
            visualize_reference_style(image, label, pred, p2)
            print(f"Saved overlay-style:  {p2}")

            produced += 1
            if produced >= num_cases:
                break

    print(f"\nDone. Generated BOTH styles for {produced} case(s). Outputs in: {args.out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prediction + dual visualization (5 cases each).")
    parser.add_argument('--images_dir', type=str,
                        default="/home/groups/comp3710/HipMRI_Study_open/semantic_MRs",
                        help='Directory of input images.')
    parser.add_argument('--labels_dir', type=str,
                        default="/home/groups/comp3710/HipMRI_Study_open/semantic_labels_only",
                        help='Directory of label maps.')
    parser.add_argument('--out_dir', type=str, default="runs/Custom_UNet3D",
                        help='Directory for model and outputs.')
    parser.add_argument('--base', type=int, default=32,
                        help='Base channels for UNet (must match training).')
    parser.add_argument('--workers', type=int, default=4,
                        help='DataLoader workers.')
    args = parser.parse_args()
    predict(args)
