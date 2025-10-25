# dataset.py
"""
Contains all components for loading and preprocessing the 3D prostate dataset
using standard PyTorch and custom, self-contained transforms.
MODIFIED to include a padding transform to fix tensor stacking errors.
"""
from pathlib import Path
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import rotate

# ===================================================================
# Custom Transform Classes
# ===================================================================

class Compose:
    # ... (code is identical to previous version)
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, data):
        for transform in self.transforms:
            data = transform(data)
        return data

class LoadNifti:
    # ... (code is identical to previous version)
    def __call__(self, data):
        data['image'] = nib.load(data['image_path']).get_fdata().astype(np.float32)
        data['label'] = nib.load(data['label_path']).get_fdata().astype(np.uint8)
        return data

class EnsureChannelFirst:
    # ... (code is identical to previous version)
    def __call__(self, data):
        data['image'] = np.expand_dims(data['image'], axis=0)
        data['label'] = np.expand_dims(data['label'], axis=0)
        return data

class NormalizeIntensity:
    # ... (code is identical to previous version)
    def __call__(self, data):
        image = data['image']
        mask = image > 1e-8
        if np.any(mask):
            mean = image[mask].mean()
            std = image[mask].std()
            image = (image - mean) / (std + 1e-8)
        data['image'] = image
        return data

class RandCropByPosNegLabel:
    # ... (code is identical to previous version)
    def __init__(self, spatial_size, num_samples):
        self.spatial_size = np.array(spatial_size)
        self.num_samples = num_samples
    def __call__(self, data):
        image, label = data['image'], data['label']
        dims = np.array(image.shape[1:])
        foreground_coords = np.argwhere(label[0] > 0)
        cropped_samples = []
        for _ in range(self.num_samples):
            if len(foreground_coords) > 0 and np.random.rand() < 0.5:
                center_coord = foreground_coords[np.random.randint(len(foreground_coords))]
            else:
                center_coord = np.array([np.random.randint(d) for d in dims])
            low = center_coord - self.spatial_size // 2
            high = low + self.spatial_size
            low = np.maximum(low, 0)
            high = np.minimum(high, dims)
            low = high - self.spatial_size
            low = np.maximum(low, 0)
            s = tuple(slice(int(l), int(h)) for l, h in zip(low, high))
            cropped_image = image[:, s[0], s[1], s[2]]
            cropped_label = label[:, s[0], s[1], s[2]]
            cropped_samples.append({'image': cropped_image, 'label': cropped_label})
        return cropped_samples

# --- NEW TRANSFORM CLASS TO FIX THE ERROR ---
class PadToSize:
    """Pads image and label to a target spatial size."""
    def __init__(self, spatial_size):
        self.spatial_size = spatial_size

    def __call__(self, data):
        image, label = data['image'], data['label']
        current_size = image.shape[1:]
        
        # Calculate the padding needed for each dimension
        pad_needed = [(0, max(0, self.spatial_size[i] - current_size[i])) for i in range(3)]
        
        # The first dimension (channel) needs no padding: (0, 0)
        image_pad_width = [(0, 0)] + pad_needed
        label_pad_width = [(0, 0)] + pad_needed
        
        # Apply padding. 'constant_values=0' for image background
        data['image'] = np.pad(image, pad_width=image_pad_width, mode='constant', constant_values=0)
        # Pad label with 0 (background class)
        data['label'] = np.pad(label, pad_width=label_pad_width, mode='constant', constant_values=0)
        
        return data

class RandFlip:
    # ... (code is identical to previous version)
    def __init__(self, prob=0.5, axis=0):
        self.prob = prob
        self.axis = axis + 1
    def __call__(self, data):
        if np.random.rand() < self.prob:
            data['image'] = np.flip(data['image'], axis=self.axis).copy()
            data['label'] = np.flip(data['label'], axis=self.axis).copy()
        return data

class ToTensor:
    # ... (code is identical to previous version)
    def __call__(self, data):
        data['image'] = torch.from_numpy(data['image'])
        data['label'] = torch.from_numpy(data['label'])
        return data

# ===================================================================
# Dataset and DataLoader Logic (with updated transforms pipeline)
# ===================================================================

def infer_num_classes(label_dir, max_files=3):
    # ... (code is identical to previous version)
    vals = set()
    for i, f in enumerate(Path(label_dir).glob("*.nii*")):
        arr = nib.load(str(f)).get_fdata()
        vals.update(np.unique(arr))
        if i + 1 >= max_files: break
    n_cls = int(max(vals)) + 1
    class_map = {0: "Background", 1: "Body Outline", 2: "Bone", 3: "Bladder", 4: "Rectum", 5: "Prostate"}
    print(f"Detected {n_cls} unique label classes: {sorted(list(vals))}")
    return n_cls, class_map

def collect_case_pairs(images_dir, labels_dir):
    # ... (code is identical to previous version)
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

class ProstateDataset(Dataset):
    # ... (code is identical to previous version)
    def __init__(self, data_dicts, transform=None, post_crop_transform=None):
        self.data_dicts = data_dicts
        self.transform = transform
        self.post_crop_transform = post_crop_transform
    def __len__(self):
        return len(self.data_dicts)
    def __getitem__(self, idx):
        data = {'image_path': str(self.data_dicts[idx]['image']), 'label_path': str(self.data_dicts[idx]['label'])}
        if self.transform: data = self.transform(data)
        if isinstance(data, list) and self.post_crop_transform: data = [self.post_crop_transform(item) for item in data]
        return data

def get_dataloaders(images_dir, labels_dir, batch_size=1, num_workers=2):
    pairs = collect_case_pairs(images_dir, labels_dir)
    data_dicts = [{"image": i, "label": l} for i, l in pairs]
    n_train = int(0.8 * len(data_dicts))
    train_files, val_files = data_dicts[:n_train], data_dicts[n_train:]

    train_pre_transforms = Compose([
        LoadNifti(),
        EnsureChannelFirst(),
        NormalizeIntensity(),
        RandCropByPosNegLabel(spatial_size=(96, 96, 96), num_samples=4),
    ])
    
    # --- MODIFICATION: Insert the new PadToSize transform here ---
    train_post_transforms = Compose([
        PadToSize(spatial_size=(96, 96, 96)),
        RandFlip(axis=0),
        ToTensor()
    ])
    
    val_transforms = Compose([
        LoadNifti(),
        EnsureChannelFirst(),
        NormalizeIntensity(),
        ToTensor(),
    ])
    
    train_ds = ProstateDataset(data_dicts=train_files, transform=train_pre_transforms, post_crop_transform=train_post_transforms)
    val_ds = ProstateDataset(data_dicts=val_files, transform=val_transforms)
    
    def train_collate_fn(batch):
        # ... (code is identical to previous version)
        images, labels = [], []
        for item_list in batch:
            for item in item_list:
                images.append(item['image'])
                labels.append(item['label'])
        return {'image': torch.stack(images), 'label': torch.stack(labels)}

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=train_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=num_workers)

    return train_loader, val_loader
