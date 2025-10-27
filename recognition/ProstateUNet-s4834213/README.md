# Improved 3D U-Net with Context Aggregation (CAN) for Prostate MRI Segmentation

This project implements a **3D Improved U-Net with Context Aggregation Network (CAN)** to perform multi-organ semantic segmentation on prostate MRI volumes from the HipMRI dataset. The task focuses on accurately identifying anatomical regions critical for prostate cancer radiotherapy planning. This project explores advanced 3D CNN techniques to push segmentation performance, especially for small and irregularly shaped organs like the prostate.

---

## Overview

- **What:** 3D semantic segmentation of HipMRI prostate volumes (6 classes: background, body outline, bone, bladder, rectum, prostate).
- **Why:** Accurate, context-aware organ masks support downstream clinical/analysis tasks; prostate is small → needs strong multi-scale context.
- **Model:** Improved **3D U-Net** with a **dilated Context Aggregation (CAN) bottleneck** (rates 2/4/8) + **AdaIN/InstanceNorm** for contrast robustness.
- **Training recipe:** PyTorch only (no MONAI), **Dice + Cross-Entropy** loss, Adam (lr=1e-4), patch size **96×96×96**, balanced pos/neg crops, flips, fixed seed (42).
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
