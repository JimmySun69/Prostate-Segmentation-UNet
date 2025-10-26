# dataset.py
"""
Contains all components for loading and preprocessing the 3D prostate dataset
using standard PyTorch and custom, self-contained transforms.
MODIFIED to include a padding transform to fix tensor stacking errors.
"""
from pathlib import Path
import numpy as np
import nibabel as nib  # For loading .nii files
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import rotate

# ===================================================================
# Custom Transform Classes
# ===================================================================

class Compose:
    """A simple transform composer, like torchvision.transforms.Compose."""
    def __init__(self, transforms):
        """
        Args:
            transforms (list): A list of transform objects to be applied sequentially.
        """
        self.transforms = transforms
        
    def __call__(self, data): # <-- This was the line with the typo
        """
        Applies each transform in the list to the data dictionary.

        Args:
            data (dict): A dictionary containing 'image', 'label', etc.
        Returns:
            dict: The transformed data dictionary.
        """
        for transform in self.transforms:
            data = transform(data)
        return data

class LoadNifti:
    """Loads NIfTI image and label files from paths into numpy arrays."""
    def __call__(self, data):
        """
        Args:
            data (dict): Must contain 'image_path' and 'label_path'.
        Returns:
            dict: The data dict with 'image' and 'label' (numpy arrays) added.
        """
        data['image'] = nib.load(data['image_path']).get_fdata().astype(np.float32)
        data['label'] = nib.load(data['label_path']).get_fdata().astype(np.uint8)
        return data

class EnsureChannelFirst:
    """Adds a channel dimension (C) to 3D images, making them (C, D, H, W)."""
    def __call__(self, data):
        """
        Args:
            data (dict): Must contain 'image' and 'label' with shape (D, H, W).
        Returns:
            dict: Data with 'image' and 'label' shapes (1, D, H, W).
        """
        data['image'] = np.expand_dims(data['image'], axis=0)  # (D, H, W) -> (1, D, H, W)
        data['label'] = np.expand_dims(data['label'], axis=0)  # (D, H, W) -> (1, D, H, W)
        return data

class NormalizeIntensity:
    """
    Normalizes the intensity of the 'image' by subtracting mean and dividing by std.
    Calculates mean/std only from non-zero pixels (foreground).
    """
    def __call__(self, data):
        image = data['image']
        
        # Create a mask for non-background pixels (intensity > 1e-8)
        mask = image > 1e-8
        
        if np.any(mask):
            # Calculate mean and std from the masked region
            mean = image[mask].mean()
            std = image[mask].std()
            
            # Apply normalization. Add 1e-8 to std to avoid division by zero.
            image = (image - mean) / (std + 1e-8)
            
        data['image'] = image
        return data

class RandCropByPosNegLabel:
    """
    Performs random cropping, ensuring a mix of foreground and background patches.
    Returns a *list* of cropped samples.
    """
    def __init__(self, spatial_size, num_samples):
        """
        Args:
            spatial_size (tuple): The target size of the crop (D, H, W).
            num_samples (int): The number of patches to crop from the image.
        """
        self.spatial_size = np.array(spatial_size)
        self.num_samples = num_samples
        
    def __call__(self, data):
        image, label = data['image'], data['label']
        dims = np.array(image.shape[1:])  # Get spatial dimensions (D, H, W)
        
        # Find coordinates of all foreground pixels (label > 0)
        foreground_coords = np.argwhere(label[0] > 0)
        
        cropped_samples = []
        for _ in range(self.num_samples):
            # 50% chance to center on foreground, 50% chance to center randomly
            if len(foreground_coords) > 0 and np.random.rand() < 0.5:
                # Pick a random foreground pixel as the center
                center_coord = foreground_coords[np.random.randint(len(foreground_coords))]
            else:
                # Pick a random coordinate anywhere in the image
                center_coord = np.array([np.random.randint(d) for d in dims])
            
            # Calculate crop boundaries
            low = center_coord - self.spatial_size // 2
            high = low + self.spatial_size
            
            # Ensure boundaries are within the image dimensions
            # This is a robust way to handle edge cases
            low = np.maximum(low, 0)
            high = np.minimum(high, dims)
            
            # If the patch is now too small (e.g., at the edge), shift 'low' back
            low = high - self.spatial_size
            low = np.maximum(low, 0) # Ensure 'low' doesn't go negative
            
            # Create slice objects for cropping
            s = tuple(slice(int(l), int(h)) for l, h in zip(low, high))
            
            # Apply crop. Note: s[0], s[1], s[2] correspond to D, H, W
            cropped_image = image[:, s[0], s[1], s[2]]
            cropped_label = label[:, s[0], s[1], s[2]]
            
            cropped_samples.append({'image': cropped_image, 'label': cropped_label})
            
        # Return a list of dictionaries, one for each cropped sample
        return cropped_samples

# --- NEW TRANSFORM CLASS TO FIX THE ERROR ---
class PadToSize:
    """Pads image and label to a target spatial size."""
    def __init__(self, spatial_size):
        """
        Args:
            spatial_size (tuple): The target size (D, H, W) to pad to.
        """
        self.spatial_size = spatial_size

    def __call__(self, data):
        image, label = data['image'], data['label']
        current_size = image.shape[1:]  # (D, H, W)
        
        # Calculate the padding needed for each dimension (D, H, W)
        # (pad_before, pad_after)
        pad_needed = [(0, max(0, self.spatial_size[i] - current_size[i])) for i in range(3)]
        
        # Add padding for the channel dimension (which is 0)
        # np.pad expects a pad width for *every* dimension
        # (C, D, H, W) -> [(C_pad), (D_pad), (H_pad), (W_pad)]
        image_pad_width = [(0, 0)] + pad_needed
        label_pad_width = [(0, 0)] + pad_needed
        
        # Apply padding. 'constant_values=0' for image background
        data['image'] = np.pad(image, pad_width=image_pad_width, mode='constant', constant_values=0)
        # Pad label with 0 (background class)
        data['label'] = np.pad(label, pad_width=label_pad_width, mode='constant', constant_values=0)
        
        return data

class RandFlip:
    """Randomly flips the image and label along a specified axis."""
    def __init__(self, prob=0.5, axis=0):
        """
        Args:
            prob (float): Probability of flipping.
            axis (int): Axis to flip (0=D, 1=H, 2=W).
        """
        self.prob = prob
        self.axis = axis + 1  # Add 1 to account for the channel dimension (C)
        
    def __call__(self, data):
        if np.random.rand() < self.prob:
            # .copy() is important to avoid negative stride issues with PyTorch tensors
            data['image'] = np.flip(data['image'], axis=self.axis).copy()
            data['label'] = np.flip(data['label'], axis=self.axis).copy()
        return data

class ToTensor:
    """Converts 'image' and 'label' numpy arrays to PyTorch tensors."""
    def __call__(self, data):
        data['image'] = torch.from_numpy(data['image'])
        data['label'] = torch.from_numpy(data['label'])
        return data

# ===================================================================
# Dataset and DataLoader Logic
# ===================================================================

def infer_num_classes(label_dir, max_files=3):
    """
    Infers the number of segmentation classes by checking unique values
    in a few label files.
    """
    vals = set()
    # Check up to max_files to save time
    for i, f in enumerate(Path(label_dir).glob("*.nii*")):
        arr = nib.load(str(f)).get_fdata()
        vals.update(np.unique(arr))  # Add all unique values from this file
        if i + 1 >= max_files: break
        
    n_cls = int(max(vals)) + 1  # e.g., if max is 5, classes are 0,1,2,3,4,5 (6 total)
    
    # Hardcoded class map for this specific prostate dataset
    class_map = {0: "Background", 1: "Body Outline", 2: "Bone", 3: "Bladder", 4: "Rectum", 5: "Prostate"}
    print(f"Detected {n_cls} unique label classes: {sorted(list(vals))}")
    return n_cls, class_map

def collect_case_pairs(images_dir, labels_dir):
    """
    Matches image files with label files based on their filenames.
    Assumes filenames like 'case_001_image.nii.gz' and 'case_001_label.nii.gz'.
    """
    image_files = sorted(Path(images_dir).glob("*.nii*"))
    label_files = sorted(Path(labels_dir).glob("*.nii*"))
    pairs = []
    
    for img in image_files:
        # Standardize the image filename stem to get a case ID
        stem = img.stem.replace("_LFOV", "").replace("_image", "").replace("_img", "")
        
        # Find the matching label file
        for lbl in label_files:
            # Standardize the label filename stem
            lblstem = lbl.stem.replace("_SEMANTIC", "").replace("_label", "").replace("_mask", "")
            if lblstem == stem:
                pairs.append((img, lbl))
                break  # Found match, move to next image
                
    if not pairs:
        raise RuntimeError("No (image,label) pairs matched by filename stem.")
        
    print(f"Found {len(pairs)} image–label pairs.")
    return pairs

class ProstateDataset(Dataset):
    """
    PyTorch Dataset class for loading 3D prostate scans.
    Handles the two-stage transform pipeline (pre-crop and post-crop).
    """
    def __init__(self, data_dicts, transform=None, post_crop_transform=None):
        """
        Args:
            data_dicts (list): List of dicts, e.g., [{'image': Path, 'label': Path}, ...]
            transform (callable, optional): Transforms to apply *before* cropping.
            post_crop_transform (callable, optional): Transforms to apply *after* cropping.
        """
        self.data_dicts = data_dicts
        self.transform = transform
        self.post_crop_transform = post_crop_transform
        
    def __len__(self):
        return len(self.data_dicts)
        
    def __getitem__(self, idx):
        # Start with file paths
        data = {'image_path': str(self.data_dicts[idx]['image']), 'label_path': str(self.data_dicts[idx]['label'])}
        
        # Apply pre-crop transforms (LoadNifti, Normalize, RandCrop)
        if self.transform:
            data = self.transform(data)
            
        # RandCropByPosNegLabel returns a LIST of samples.
        # We must apply the post-crop transforms to *each* sample in the list.
        if isinstance(data, list) and self.post_crop_transform:
            data = [self.post_crop_transform(item) for item in data]
            
        return data

def get_dataloaders(images_dir, labels_dir, batch_size=1, num_workers=2):
    """
    Sets up and returns the train and validation DataLoaders.
    """
    pairs = collect_case_pairs(images_dir, labels_dir)
    data_dicts = [{"image": i, "label": l} for i, l in pairs]
    
    # --- 80/10/10 Split ---
    n_total = len(data_dicts)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    
    # 80% for training
    train_files = data_dicts[:n_train]
    # 10% for validation
    val_files = data_dicts[n_train : n_train + n_val]
    # Final 10% (data_dicts[n_train + n_val:]) is the test set, used by predict.py
    # --- End Split ---

    # Transforms applied *before* cropping
    train_pre_transforms = Compose([
        LoadNifti(),
        EnsureChannelFirst(),
        NormalizeIntensity(),
        RandCropByPosNegLabel(spatial_size=(96, 96, 96), num_samples=4), # Returns a list
    ])
    
    # Transforms applied *after* cropping (to each patch)
    train_post_transforms = Compose([
        PadToSize(spatial_size=(96, 96, 96)), # Pad if crop was at an edge
        RandFlip(axis=0), # Random Z-axis flip
        ToTensor()
    ])
    
    # Validation transforms load the *entire* image (no cropping)
    val_transforms = Compose([
        LoadNifti(),
        EnsureChannelFirst(),
        NormalizeIntensity(),
        ToTensor(),
    ])
    
    train_ds = ProstateDataset(data_dicts=train_files, transform=train_pre_transforms, post_crop_transform=train_post_transforms)
    val_ds = ProstateDataset(data_dicts=val_files, transform=val_transforms)
    
    def train_collate_fn(batch):
        """
        Custom collate function to handle the list of patches from RandCrop.
        `batch` is a list of lists: [ [patch1, patch2], [patch3, patch4] ]
        This function flattens it and stacks all patches into a single batch.
        """
        images, labels = [], []
        # `item_list` is the list of 4 patches from one original image
        for item_list in batch: 
            # `item` is a single patch dictionary
            for item in item_list:  
                images.append(item['image'])
                labels.append(item['label'])
        
        # Stack all patches into a single batch tensor
        return {'image': torch.stack(images), 'label': torch.stack(labels)}

    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        collate_fn=train_collate_fn  # Use our custom collate function
    )
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=1,  # Validate one full image at a time
        num_workers=num_workers
    )

    return train_loader, val_loader
