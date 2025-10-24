"""
dataset.py
Contains functions for loading and preprocessing the 3D prostate dataset.
"""
from pathlib import Path
import numpy as np
import nibabel as nib
from torch.utils.data import DataLoader
from monai.data import Dataset
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd,
    RandFlipd, RandRotate90d, RandScaleIntensityd, RandShiftIntensityd,
    ToTensord
)

def infer_num_classes(label_dir, max_files=3):
    """Auto-detects the number of classes from the label files."""
    vals = set()
    for i, f in enumerate(Path(label_dir).glob("*.nii*")):
        arr = nib.load(str(f)).get_fdata()
        vals.update(np.unique(arr))
        if i + 1 >= max_files: break
    n_cls = int(max(vals)) + 1
    print(f"Detected {n_cls} unique label classes: {sorted(list(vals))}")
    return n_cls

def collect_case_pairs(images_dir, labels_dir):
    """Matches image and label files based on their filename stems."""
    image_files = sorted(Path(images_dir).glob("*.nii*"))
    label_files = sorted(Path(labels_dir).glob("*.nii*"))
    pairs = []
    for img in image_files:
        stem = img.stem.replace("_LFOV", "").replace("_image", "").replace("_img", "")
        for lbl in label_files:
            lblstem = lbl.stem.replace("_SEMANTIC", "").replace("_label", "").replace("_mask", "")
            if lblstem == stem:
                pairs.append((img, lbl))
                break
    if not pairs:
        raise RuntimeError("No (image,label) pairs matched by filename stem.")
    print(f"Found {len(pairs)} image–label pairs.")
    return pairs

def get_dataloaders(images_dir, labels_dir, batch_size=1, num_workers=2):
    """Creates and returns training and validation dataloaders."""
    pairs = collect_case_pairs(images_dir, labels_dir)
    data_dicts = [{"image": str(i), "label": str(l)} for i, l in pairs]

    # 80/20 split for training and validation
    n_train = int(0.8 * len(data_dicts))
    train_files, val_files = data_dicts[:n_train], data_dicts[n_train:]

    # Define transforms for data augmentation and preprocessing
    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityd(keys=["image"]),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=[0]),
        RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
        RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
        ToTensord(keys=["image", "label"]),
    ])
    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityd(keys=["image"]),
        ToTensord(keys=["image", "label"]),
    ])

    train_ds = Dataset(data=train_files, transform=train_transforms)
    val_ds = Dataset(data=val_files, transform=val_transforms)

    # Create DataLoader instances
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=num_workers)
    
    return train_loader, val_loader


