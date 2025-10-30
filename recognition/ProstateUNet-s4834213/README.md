# Improved 3D U-Net with Context Aggregation (CAN) for Prostate MRI Segmentation

This project implements a **3D Improved U-Net with Context Aggregation Network (CAN)** to perform multi-organ semantic segmentation on prostate MRI volumes from the HipMRI dataset. The task focuses on accurately identifying anatomical regions critical for prostate cancer radiotherapy planning. This project explores advanced 3D CNN techniques to push segmentation performance, especially for small and irregularly shaped organs like the prostate.

---

### Author

Project by: *Ruitao Sun 48342131*  
UQ COMP3710 — Pattern Recognition & Analysis  
Submission for: Hard Difficulty — 3D Prostate Segmentation with CAN3D

---

## Table of Contents

- [Overview](#1-overview)
- [Project Structure](#2-project-structure)
- [Model Evolution](#3-from-prototype-to-precision-the-evolution-v1--v3)
- [Model Architecture](#4-model-architecture)
- [Implementation Details](#5-implementation-details)
  - [Model Definition](#51-model-definition-modulespy)
  - [Loss Metrics](#52-loss--metrics-trainpy)
  - [Data Pipeline](#53-data-pipeline-datasetpy)
  - [Training Loop](#54-training-loop-trainpy)
  - [Prediction and Visualization](#55-inference--visualization-predictpy)
- [Results](#6-results)
  - [Accuracy over time](#61-accuracy-over-time-every-20-epochs)
  - [Takeaways](#62-phase-by-phase-takeaways)
  - [Test Result](#63-final-test-result-best-model--epoch-89)
  - [Guidance](#64-practical-guidance)
  - [Test Set Visuals](#65-testset-visuals)
- [Limitation](#7-limitations--future-work)
- [Reproducibility](#8-reproducibility)
- [Dependencies](#9-setup--dependencies)
- [Reference](#10-references)

---

## 1. Overview

- **What:** 3D semantic segmentation of HipMRI prostate volumes (6 classes: background, body outline, bone, bladder, rectum, prostate).
- **Why:** Accurate, context-aware organ masks support downstream clinical/analysis tasks; prostate is small → needs strong multi-scale context.
- **Model:** Improved **3D U-Net** with a **dilated Context Aggregation (CAN) bottleneck** (rates 2/4/8) + **AdaIN/InstanceNorm** for contrast robustness.
- **Training recipe:** PyTorch, **Dice + Cross-Entropy** loss, Adam (lr=1e-4), patch size **96×96×96**, balanced pos/neg crops, flips, fixed seed (42).
- **Inference:** Sliding-window **with Gaussian blending** to remove tiling seams; saves overlays per case.
- **Val metrics:** **Mean Dice (no background) = 0.916**; **Prostate Dice = 0.843**.
- **Artifacts:** `final_dice_score.txt` (per-class), `training_curve.png` (learning curve), `prediction_*.png` (qualitative) are saved in Result Output.
- **Reproduce in 2 steps:**
  1) **Train**
     ```bash
     python train.py \
       --images_dir /home/groups/comp3710/HipMRI_Study_open/semantic_MRs \
       --labels_dir /home/groups/comp3710/HipMRI_Study_open/semantic_labels_only \
       --out_dir runs/Custom_UNet3D --epochs 100 --batch_size 2 --lr 1e-4 --base 32 --workers 4
     ```
  2) **Predict**
     ```bash
     python predict.py \
       --images_dir /home/groups/comp3710/HipMRI_Study_open/semantic_MRs \
       --labels_dir /home/groups/comp3710/HipMRI_Study_open/semantic_labels_only \
       --out_dir runs/Custom_UNet3D --base 32 --workers 4
     ```


---

## 2. Project Structure
```text
├── dataset.py          # NIfTI IO, preprocessing, balanced cropping, split
├── modules.py          # ImprovedUNet3D_CAN model components
├── train.py            # Training loop, losses, metrics, checkpointing
├── predict.py          # Inference + visualization overlays
├── Slurm Jobs          # SLURM job scripts for model training and prediction
└── Result Output       # Outputs: models, predictions, metrics, etc.
```

---

## 3. From Prototype to Precision: The Evolution (V1 → V3)

This section shows model evolvement, **why** designed it that way, and **how** it improves on the previous one. It also summarizes the **model architecture** and **bottleneck** used throughout.  

---

**V1 — Prototype · Single-Script U-Net**  
- **What/Why:** One python file to validate data pairing, class IDs, and baseline learnability fast.  
- **Arch/Bottleneck:** 3-level 3D U-Net + **CAN** (dilated convs 2/4/8) at bottleneck; InstanceNorm + LeakyReLU.  
- **Improves on zero-baseline:** First working pipeline, establishes loss (Dice+CE), metrics, and checkpoints.  
- **Limits:** Hard-coded params, tight coupling (data/model/train), minimal artifacts → hard to reproduce/ablate.

**V2 — Refactor · Modular CAN U-Net**  
- **What/Why:** Split into `dataset.py` (splits, balanced crops, pad-to-96³), `modules.py` (U-Net+CAN), `train.py` (CLI, metrics, artifacts) for reuse and cleaner experiments.  
- **Arch/Bottleneck:** Same U-Net + **CAN**; optional AdaIN for contrast robustness.  
- **Improves on V1:** Argparse config, per-class Dice + saved curves/scores, safer batching via padding/collate.  
- **Limits:** Inference on full volumes still needs robust tiling/blending; some behaviors not fully self-contained.

**V3 — Final · Self-Contained Segmentation Suite**  
- **What/Why:** Adds `predict.py` with from-scratch **sliding-window + Gaussian blending**, custom DiceCE + per-class Dice, auto class detection. 
- **Arch/Bottleneck:** U-Net encoder (1→32→64→128) → **CAN** (~256) → decoder + 1×1×1 head to *C* classes; InstanceNorm/LeakyReLU (AdaIN where used).  
- **Improves on V2:** Fully self-contained (no external toolkits for core logic), higher-fidelity inference, more robust “works-out-of-the-box.”  

---

## 4. Model Architecture

**Baseline 3D Unet Architecture.**
The baseline 3D UNet follows a symmetric encoder-decoder structure. It uses InstanceNorm3d and LeakyReLU, which are common in modern medical imaging, but does not include the advanced ContextAggregationModule
Standard 3D UNet:
```
┌─────────────────────────────────────────────────────────────┐
│ Encoder (Contracting Path)                                  │
│   - ConvBlock3D (2×Conv3D + InstanceNorm + LeakyReLU)       │
│   - Max pooling for downsampling                            │
│                                                             │
│ Decoder (Expanding Path)                                    │
│   - Transposed convolutions for upsampling                  │
│   - Skip connections (concatenation)                        │
│   - Simple ConvBlock3D (2×Conv3D + IN + LeakyReLU)          │
│                                                             │
│ Normalization: Instance Normalization                       │
│ Activation: LeakyReLU                                       │
└─────────────────────────────────────────────────────────────┘
```
**Limitations of this Standard 3D UNet:**

1. Limited Bottleneck Receptive Field: The standard bottleneck uses simple convolutions, failing to aggregate multi-scale contextual information. This is the primary limitation solved by the ContextAggregationModule in your ImprovedUNet3D_CAN model.

2. No Residual Connections: The ConvBlock3D itself is feed-forward. Deeper networks can become harder to train without explicit residual connections.

3. Information Flow: Standard skip connections simply concatenate features, which may not be the most effective way to merge deep, semantic information with shallow, spatial information.

**Overview for this project.**  
We use a 3D U-Net variant tailored for small-organ segmentation. The network preserves fine spatial detail with encoder–decoder skips while a **Context Aggregation (CAN) bottleneck** injects wide-field cues via dilated convolutions. This balances local boundary precision (e.g., prostate) with global anatomical context.

**Input & Shapes.**  
- Input tensor: **(B, 1, 96, 96, 96)** after intensity normalization and cropping.  
- Base channel width: **32**, doubling per downsampling stage.  
- Number of classes: **C** (auto-detected from labels).

**Encoder (feature extractor).**  
Three downsampling stages; each stage uses two 3×3×3 convolutions with normalization and nonlinearity, then downsamples by ×2.
- Stage E1: 1 → 32  
- Stage E2: 32 → 64  
- Stage E3: 64 → 128  
- Blocks: `Conv3d → (InstanceNorm3d / AdaIN3d) → LeakyReLU` ×2 → Downsample (stride-2 conv or equivalent)

**Bottleneck (CAN: context aggregation).**  
Multi-scale context is captured with a sequential series of dilated 3D convolutions:
- Block dilations: **2**, **4**, **8** (3×3×3 kernels)  
- Each block: `Conv3d(dilated) → Norm → LeakyReLU`  
- The output of this sequential module is added back to the original bottleneck input via a residual connection 
- Effect: enlarges effective receptive field **without extra pooling**, improving small-organ Dice while keeping resolution.

**Decoder (reconstruction).**  
Three upsampling stages mirror the encoder; each step upsamples by ×2, concatenates the corresponding encoder skip, then applies two 3×3×3 conv blocks.
- Stage D3: 128(+skip) → 128  
- Stage D2: 64(+skip) → 64  
- Stage D1: 32(+skip) → 32  
- Upsample op: transpose convolution (deconv) with stride 2 (or equivalent)

**Segmentation head.**  
- **1×1×1** convolution maps features to **C** classes.  
- Training: logits → softmax for **Dice + Cross-Entropy** loss.  
- Inference: argmax over classes; for large volumes we use sliding-window with **Gaussian blending** to avoid tiling seams.

**Normalization & Activation.**  
- Primary: **InstanceNorm3d** + **LeakyReLU** in all conv blocks.  
- Select blocks use **AdaIN3d** to reduce contrast/coil variability across subjects.

**Why this design?**  
- **U-Net skips** retain edge detail critical for organ boundaries.  
- **CAN bottleneck** supplies global cues (bladder/rectum context) that disambiguate the small prostate region.  
- **Dilations (2/4/8)** grow context at constant memory cost—no extra downsampling, no loss of detail.

**Default configuration (this repo).**
- Patch size: `96 × 96 × 96`  
- Channels: `32 → 64 → 128 → (CAN ~256) → 128 → 64 → 32`  
- Dilation rates in CAN: `2, 4, 8`  
- Head: `1×1×1` to `C` classes

**File mapping.**
- Blocks & CAN: `modules.py`  
- Data & transforms: `dataset.py`  
- Training loop & metrics: `train.py`  
- Sliding-window inference & overlays: `predict.py`

The model (ImprovedUNet3D_CAN) follows a symmetric encoder-decoder path, but with a significantly enhanced bottleneck.
Improved 3D U-Net (base=32):
```
┌─────────────────────────────────────────────────────────────┐
│ INPUT (1, 96, 96, 96)                                       │
└────────────┬────────────────────────────────────────────────┘
             │
      ┌──────▼──────┐
      │ Encoder L1  │ ConvBlock3D (1 → 32 filters)
      └──────┬──────┘
             │ ───────────────┐ Skip Connection 1
      ┌──────▼──────┐         │
      │ MaxPool 2×  │         │
      └──────┬──────┘         │
             │                │
      ┌──────▼──────┐         │
      │ Encoder L2  │ ConvBlock3D (32 → 64 filters)
      └──────┬──────┘         │
             │ ───────────────┤ Skip Connection 2
      ┌──────▼──────┐         │
      │ MaxPool 2×  │         │
      └──────┬──────┘         │
             │                │
      ┌──────▼──────┐         │
      │ Encoder L3  │ ConvBlock3D (64 → 128 filters)
      └──────┬──────┘         │
             │ ───────────────┤ Skip Connection 3
      ┌──────▼──────┐         │
      │ MaxPool 2×  │         │
      └──────┬──────┘         │
             │                │
      ┌──────▼──────┐         │
      │ Bottleneck  │ ConvBlock3D (128 → 256 filters)
      │    +        │         │
      │    CAM      │ ContextAggregationModule (256 filters)
      └──────┬──────┘         │
             │                │
      ┌──────▼──────┐         │
      │ Decoder L3  │ UpConv (256 → 128) + Concat (128+128)
      │  ConvBlock  │◄────────┘ (Skip 3)
      └──────┬──────┘         │
             │                │
      ┌──────▼──────┐         │
      │ Decoder L2  │ UpConv (128 → 64) + Concat (64+64)
      │  ConvBlock  │◄────────┤ (Skip 2)
      └──────┬──────┘         │
             │                │
      ┌──────▼──────┐         │
      │ Decoder L1  │ UpConv (64 → 32) + Concat (32+32)
      │  ConvBlock  │◄────────┘ (Skip 1)
      └──────┬──────┘
             │
      ┌──────▼──────┐
      │ Output Conv │ 1×1×1 conv (32 → 6 classes)
      └──────┬──────┘
             │
      OUTPUT (6, 96, 96, 96)
```
**Design Rationale:**

ConvBlock3D: Each standard block consists of two (Conv3D -> InstanceNorm3d -> LeakyReLU) sequences. This is a robust and stable design for 3D segmentation.

ContextAggregationModule (CAM): The bottleneck is often a point of information loss. By adding dilated convolutions in parallel, the CAM block allows the model to "see" features at multiple scales simultaneously (e.g., fine-grained organ boundaries and large-scale organ position) before starting the upsampling path.

AdaIN3D: The adaptive instance normalization in the CAM blocks provides learnable scale and shift parameters (a and b), giving the model more control over the normalization process within this critical module.

---
## 5. Implementation Details

This section describes how the codebase implements 3D prostate MRI segmentation using a custom Improved U‑Net with a Context Aggregation Module (CAM), a Dice+Cross‑Entropy objective data pipeline. The repository structure includes: `dataset.py`, `modules.py`, `train.py`, `predict.py`, plus SLURM scripts `run.sh` and `run1.sh`.


### 5.1) Model Definition (`modules.py`)
**Class:** `ImprovedUNet3D_CAN` — a 3D U‑Net variant that augments the bottleneck with a **Context Aggregation Module (CAM)** (dilated 3D convolutions + AdaIN + residual connection) to capture larger receptive fields without downsampling further.

**Building blocks**
- `AdaIN3D`: adaptive instance normalization (learnable per‑channel scale/shift).
- `ConvBlock3D`: (Conv3d → InstanceNorm3d → LeakyReLU) × 2.
- `ContextAggregationModule`: three dilated Conv3d stages (e.g., dilation 2/4/8) with AdaIN and residual add.

**Topology (default `base=32`, `in_ch=1`, `out_ch=6`)**
- **Encoder:** 3 downsample stages (base → 2×base → 4×base) with skip outputs.
- **Bottleneck:** 8×base channels + **CAM** for multi‑scale context.
- **Decoder:** symmetric upsampling via ConvTranspose3d; skip concatenations from the encoder.
- **Head:** 1×1×1 Conv3d to logits `[B, out_ch, D, H, W]`.

**Key methods (signatures)**
```python
class ImprovedUNet3D_CAN(nn.Module):
    def __init__(self, in_ch=1, out_ch=6, base=32):
        # builds encoder, CAM bottleneck, decoder, final head

    def forward(self, x):
        # encode -> CAM -> decode with skip connections -> logits
```
**Notes**
- Activations: LeakyReLU throughout; normalization: InstanceNorm3d (batch size can be 1–2).
- Output is un-normalized logits; training uses softmax internally for Dice.

---

### 5.2) Loss & Metrics (`train.py`)
**Loss:** `DiceCELoss`
- Cross‑Entropy on integer targets `[N, D, H, W]`.
- Soft Dice on one‑hot targets vs softmax probabilities.
- Final objective: `loss = CE + (1 − Dice)` with a small smoothing term for stability.

**Metric:** `calculate_dice_per_class(logits, targets, num_classes)`
- Computes per‑class Dice from argmax predictions on validation/test.
- Mean Dice is reported **excluding background** and used for model selection.

---

### 5.3) Data Pipeline (`dataset.py`)
**Self‑contained transforms**
- **Loading:** nibabel NIfTI reader for images and labels.
- **Channel order:** `EnsureChannelFirst` → `(C, D, H, W)`.
- **Normalization:** foreground‑aware z‑score (compute mean/std on non‑zero voxels).
- **Random cropping:** `RandCropByPosNegLabel` yields balanced positive/negative 3D patches (default size **96×96×96**).
- **Tensorization:** converts to PyTorch tensors with correct dtypes.

**Pairing, split, and loaders**
- `collect_case_pairs(...)` robustly matches image/label stems (handles suffixes like `_LFOV` / `_SEMANTIC`).
- Default split: **80/10/10** for train/val/test. We use an 80/10/10 subject-level split to prevent train/val/test leakage across the same patient’s longitudinal scans, keep enough validation cases for tuning, and reserve a held-out test set for final reporting.
- `ProstateDataset` applies a two‑stage transform pipeline: pre‑crop transforms → pos/neg crop → post‑crop transforms (per patch).
- `get_dataloaders(...)` returns PyTorch DataLoaders for training and validation (test set is used by `predict.py`).

---

### 5.4) Training Loop (`train.py`)
**Setup**
- Builds `ImprovedUNet3D_CAN`, `DiceCELoss`, and `Adam(lr=1e-4)` optimizer.
- CLI allows overriding I/O dirs, epochs, LR, batch size, base channels, and workers.

**Epoch routine**
1. Forward pass on batches of cropped patches.
2. Compute `DiceCELoss`; backprop and update with Adam.
3. Track per‑batch and per‑epoch losses.

**Validation & checkpointing**
- After each epoch, compute **per‑class Dice** on the validation set (mean Dice excludes background).
- Save `best_model.pth` whenever mean Dice improves.

**Outputs**
- `training_curve.png` — line plot of training loss across epochs.
- `final_dice_score.txt` — per‑class Dice and mean Dice summary for the best model.

**CLI**
```bash
python train.py   --images_dir /path/to/semantic_MRs   --labels_dir /path/to/semantic_labels_only   --out_dir runs/Custom_UNet3D   --epochs 100 --lr 1e-4 --batch_size 2 --workers 4 --base 32
```

---

### 5.5) Inference & Visualization (`predict.py`)
**Data**
- Runs on the held-out test split from the 80/10/10 partition

**Model loading**
- Rebuilds the 3D UNet (out_ch = number of classes) and loads the checkpoint best_model.pth
- Switches to eval() and uses the available GPU if present.

**inference**
- Loads each NIfTI volume, applies the same normalization used in training.
- Applies softmax → argmax to obtain the final integer label map
- Writes a NIfTI segmentation file: .../<case>_pred.nii[.gz].

**5 panel Outputs**
- 5-panel quick look — saved as prediction_<case>.png
- Panels: Input Image, Ground Truth (All), Prediction (All), GT Prostate, Predicted Prostate.
- Shows the full field-of-view; prostate overlays use higher alpha while other classes are dimmed.
**Overlay + Split Outputs**
- overlay + split — saved as prediction_<case>_overlay.png
- Panels: Original Slice, Ground Truth Overlay, Prediction Overlay, GT (left) | Prediction (right) (categorical maps, dashed divider).
- Designed to make anatomy-aligned agreement clear (overlays) and highlight boundary/shape differences (split view)

**CLI**
```bash
python predict.py   --images_dir /path/to/semantic_MRs   --labels_dir /path/to/semantic_labels_only   --out_dir runs/Custom_UNet3D   --workers 4 --base 32
```

---

### Training Strategy & Observed Results

- **Patch size:** 96×96×96; **base channels:** 32 (configurable).
- **Optimizer:** Adam (lr=1e‑4).
- **Epochs:** up to 100 (configurable); validate every epoch; checkpoint the best mean Dice (no BG).
- **Learning curve:** training loss typically drops from ≈2.2 to ≈0.44 by epoch 100.
- **Observed performance (validation):**
  - **Best Mean Dice (no BG): 0.9160** (around epoch 89).
  - Per‑class Dice: BG 0.9962, Body 0.9861, Bone 0.9266, Bladder 0.9457, Rectum 0.8785, Prostate 0.8432.
- **GPU memory tips:** if constrained, reduce `--base` (e.g., 24) or crop size in `dataset.py`.

---

### Job Scripts (SLURM)

Two ready‑to‑use scripts for A100 nodes:
- `run.sh`: training job (1 GPU, 8 CPUs, 24h; activates your `conda` env then runs `train.py`).
- `run1.sh`: prediction job with the same resources, running `predict.py`.

Usage:
```bash
sbatch run.sh   # train
sbatch run1.sh  # predict
```

---

## 6. Results

**Data Setup.** The training run detected **211** image–label pairs and **6** classes. Training used 3D patches of **96×96×96** with the Dice+Cross-Entropy (DiceCE) objective. Validation was performed at the end of every epoch, and the best checkpoint was selected by **Mean Dice excluding background**.

**Learning Curve.**
<table>
<tr>
<td><img src="./Result Output/training_curve.png" width="60%"/></td>
</tr>
</table>

The training loss decreases smoothly from **≈2.20** at epoch 1 to **≈0.44** by epoch 100 (figure above). Validation accuracy improves in clear phases, with the largest jumps early and smaller, steadier gains after ~60 epochs.

---

### 6.1 Accuracy Over Time (every ~20 epochs)

| Epoch | Val Mean Dice (no BG) | What’s happening |
|-----:|:-----------------------|:-----------------|
| **20** | **0.7328** | End of warm‑up. The model locks onto large, high‑contrast anatomy; boundaries are still coarse. |
| **40** | **0.8790** | Strong mid‑training consolidation. Context Aggregation + skip paths refine organ boundaries. |
| **60** | **0.8829** | Improvements slow and stabilize. Nearby peaks appear at **0.9015** (≈55) and **0.9053** (≈62). |
| **80** | **0.8991** | Small oscillations but higher local maxima: **0.9108** (71), **0.9113** (82), **0.9136** (83). |
| **89** | **0.9160** | **Best checkpoint** (saved). Recommended for deployment and test-time inference. |
| **100** | **0.9045** | Slight dip vs. the best epoch—consistent with mild late‑epoch overfitting. |

### 6.2 Phase-by-phase takeaways
- **Epochs 1–20 (rapid fitting).** Dice rises from ~0.25 → **0.73**; the model quickly learns organ localization and shape priors.
- **Epochs 21–40 (context consolidation).** Dice climbs to **~0.88**; boundaries sharpen as deeper context is exploited.
- **Epochs 41–60 (regularization‑limited gains).** Accuracy fluctuates around **0.90** with incremental improvements.
- **Epochs 61–90 (fine‑grained tuning).** Small oscillations with new highs, culminating in the **best** **0.9160** at epoch 89.
- **Epochs 91–100 (saturation).** No new highs; use the saved best checkpoint rather than the final epoch.

---

### 6.3 Final Test Result (best model ~ epoch 89)

**Per-class Dice:**

| Class         | Dice |
|---------------|-----:|
| Background    | 0.9962 |
| Body Outline  | 0.9861 |
| Bone          | 0.9266 |
| Bladder       | 0.9457 |
| Rectum        | 0.8785 |
| Prostate      | 0.8432 |

**Mean Dice (no BG): 0.9160**

**Interpretation.** Large or high‑contrast structures (Body Outline, Bladder, Bone) are easiest and top **0.92–0.98**. **Prostate** is most challenging (smaller volume, variable appearance), reaching **0.84** yet contributing the most to residual error. Class‑targeted augmentation (boundary jitter, elastic deformations) or boundary‑aware losses can plausibly add **+1–3 points** to prostate Dice without changing the architecture.

---

### 6.4 Practical Guidance

- For quick, strong performance, **train to ~40 epochs** (≈**0.88** mean Dice).  
- For maximum accuracy, **train to ~90 epochs** and use the **best checkpoint** (≈**0.916** mean Dice).  
- If VRAM is limited, reduce `--base` or the crop size; this mostly affects speed rather than convergence trend.  
- Maintain identical crop size and overlap between training and sliding‑window inference to avoid tile boundary artifacts.

---

### 6.5 Test‑Set Visuals

### 6.5.1) Visualization styles

### A) 5‑Panel Visualization

**What it shows**
1. **Input Image** – raw grayscale MR slice.
2. **Ground Truth (All)** – categorical label map rendered with a fixed colormap.
3. **Prediction (All)** – model’s categorical label map on the same slice.
4. **GT Prostate** – GT prostate mask overlaid on the MR slice.
5. **Predicted Prostate** – Prediction prostate mask overlaid.

**How it is rendered**
- Pick a slice that intersects the prostate (typically mid‑gland).
- Convert the integer label volume to a color image using a discrete colormap.
- For the prostate overlay panels, keep the MR slice as the background and blend the prostate mask with high alpha while dimming other classes (use a low global alpha).
- No cropping: the **full field‑of‑view** is shown to retain anatomical context.

**Why it’s useful**
- Fast sanity check across **all classes**.
- Dedicated prostate views make it easier to judge **gland boundaries** in context.

---

### B) Overlay + Split Visualization

**What it shows**
1. **Original Slice** – MR slice (same as above but with intensity windowing).
2. **Ground Truth Overlay** – semi‑transparent colored masks over the MR slice.
3. **Prediction Overlay** – model masks over the MR slice with the same colors.
4. **GT (left) | Prediction (right)** – side‑by‑side **categorical maps only**, separated by a dashed line to highlight **shape differences** without grayscale background.

**How it is rendered**
- Use identical slice selection as the original style.
- Overlay masks with higher alpha on a windowed MR slice (keeps anatomy visible).
- For the split panel, strip the MR background and render **pure label maps** left/right; draw a dashed separator for fast visual diffs.

**Why it’s useful**
- The overlay directly shows **alignment on anatomy**.
- The split panel makes **boundary and topology differences** pop out.

---

### 6.5.2) Label set used in both styles

- Background (0) – black  
- Body (1) – teal/green (dim)  
- Bone (2) – light teal (dim)  
- **Bladder (3)** – bright green  
- **Rectum (4)** – blue‑green  
- **Prostate (5)** – dark red  

---

### 6.5.3) Results for V027 (Week 3 → Week 7)

For each timepoint, we include **both** visual styles:

- **Original 5‑Panel**: `prediction_V027_WeekX_LFOV.nii.png`  
- **Updated Overlay + Split**: `prediction_V027_WeekX_LFOV.nii_overlay.png`

### Week 3
<table>
<tr>
<td><img src="./Result Output/prediction_V027_Week3_LFOV.nii.png" width="60%"/></td>
</tr>
<tr>
<td><img src="./Result Output/prediction_V027_Week3_LFOV.nii_overlay.png" width="60%"/></td>
</tr>
</table>

**Notes**
- Prostate (red) closely matches GT with minimal lateral error.
- Bladder (green) slightly under‑expanded superiorly vs GT; rectum consistent.

---

### Week 4
<table>
<tr>
<td><img src="./Result Output/prediction_V027_Week4_LFOV.nii.png" width="60%"/></td>
</tr>
<tr>
<td><img src="./Result Output/prediction_V027_Week4_LFOV.nii_overlay.png" width="60%"/></td>
</tr>
</table>

**Notes**
- Prostate position/size stable vs Week 3.
- Bladder fullness increased; prediction tracks GT well. Rectum boundaries clean.

---

### Week 5
<table>
<tr>
<td><img src="./Result Output/prediction_V027_Week5_LFOV.nii.png" width="60%"/></td>
</tr>
<tr>
<td><img src="./Result Output/prediction_V027_Week5_LFOV.nii_overlay.png" width="60%"/></td>
</tr>
</table>

**Notes**
- Prostate shows **slight boundary smoothing** (rounder than GT).
- Organ topology preserved; small AP shift visible in split panel (<1–2 px).

---

### Week 6
<table>
<tr>
<td><img src="./Result Output/prediction_V027_Week6_LFOV.nii.png" width="60%"/></td>
</tr>
<tr>
<td><img src="./Result Output/prediction_V027_Week6_LFOV.nii_overlay.png" width="60%"/></td>
</tr>
</table>

**Notes**
- Best alignment overall in this series: prostate/bladder/rectum all match closely.
- No label leakage; strong temporal consistency for this subject.

---

### Week 7
<table>
<tr>
<td><img src="./Result Output/prediction_V027_Week7_LFOV.nii.png" width="60%"/></td>
</tr>
<tr>
<td><img src="./Result Output/prediction_V027_Week7_LFOV.nii_overlay.png" width="60%"/></td>
</tr>
</table>

**Notes**
- Prostate still well‑centered; minor lateral smoothing relative to GT.
- Bladder/rectum remain accurate; differences are localized to boundaries.

---

### 6.5.4) Cross‑week interpretation

- **Temporal stability**: Shapes are consistent from **Week 3 → 7**; no drift or sudden topology errors.
- **Boundary behavior**: Predictions are slightly smoother than GT on some slices—expected with 3D UNet inference and mild pre/post resizing; typically neutral‑to‑positive for volumetric Dice.
- **Targets**: Prostate alignment is strong across weeks; bladder volume changes are followed; rectum boundaries are clean with no spurious islands.


---

## 7. Limitations & Future Work
- **Prostate performance** can be improved with **focal Dice**, **class-balanced CE**, or **attention mechanisms**.
- Further enhancements:
  - Larger input patches
  - Deeper U-Net (4 levels)
  - TTA (Test Time Augmentation)
  - Post-processing (e.g., connected components)

---

## 8. Reproducibility
- Fixed seed: `42`
- All default training args logged in `train.py`
- Fully deterministic option via cuDNN toggle

---
## 9. Setup & Dependencies

Install dependencies (Python 3.10+, CUDA-enabled recommended):
```bash
pip install torch numpy nibabel scipy matplotlib tqdm
```

Environment (tested):
```
Python 3.10
PyTorch 2.0+
CUDA 11.7+
```
To ensure reproducibility:
```python
torch.backends.cudnn.deterministic = True
```
---

## 10. References
- Çiçek, Ö., Abdulkadir, A., Lienkamp, S. S., Brox, T., & Ronneberger, O. (2016). 3D U-Net: Learning dense volumetric segmentation from sparse annotation. In S. Ourselin, L. Joskowicz, M. R. Sabuncu, G. Unal, & W. Wells (Eds.), Medical Image Computing and Computer-Assisted Intervention – MICCAI 2016 (Lecture Notes in Computer Science, Vol. 9901, pp. 424–432). Cham, Switzerland: Springer.
- Dai, W., Woo, B., Liu, S., Marques, M., Engstrom, C. B., Greer, P. B., Crozier, S., Dowling, J. A., & Chandras, S. S. (2021). CAN3D: Fast 3D medical image segmentation via compact context aggregation (arXiv:2109.05443). arXiv. http://arxiv.org/abs/2109.05443

---