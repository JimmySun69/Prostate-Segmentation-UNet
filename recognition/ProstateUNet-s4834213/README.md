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


---

# Results

**Data Setup.** The training run detected **211** image–label pairs and **6** classes. Training used 3D patches of **96×96×96** with the Dice+Cross-Entropy (DiceCE) objective. Validation was performed at the end of every epoch, and the best checkpoint was selected by **Mean Dice excluding background**.

**Learning Curve.**
<table>
<tr>
<td><img src="./Result Output/training_curve.png" width="60%"/></td>
</tr>
</table>

The training loss decreases smoothly from **≈2.20** at epoch 1 to **≈0.44** by epoch 100 (figure above). Validation accuracy improves in clear phases, with the largest jumps early and smaller, steadier gains after ~60 epochs.

---

## Accuracy Over Time (every ~20 epochs)

| Epoch | Val Mean Dice (no BG) | What’s happening |
|-----:|:-----------------------|:-----------------|
| **20** | **0.7328** | End of warm‑up. The model locks onto large, high‑contrast anatomy; boundaries are still coarse. |
| **40** | **0.8790** | Strong mid‑training consolidation. Context Aggregation + skip paths refine organ boundaries. |
| **60** | **0.8829** | Improvements slow and stabilize. Nearby peaks appear at **0.9015** (≈55) and **0.9053** (≈62). |
| **80** | **0.8991** | Small oscillations but higher local maxima: **0.9108** (71), **0.9113** (82), **0.9136** (83). |
| **89** | **0.9160** | **Best checkpoint** (saved). Recommended for deployment and test-time inference. |
| **100** | **0.9045** | Slight dip vs. the best epoch—consistent with mild late‑epoch overfitting. |

### Phase-by-phase takeaways
- **Epochs 1–20 (rapid fitting).** Dice rises from ~0.25 → **0.73**; the model quickly learns organ localization and shape priors.
- **Epochs 21–40 (context consolidation).** Dice climbs to **~0.88**; boundaries sharpen as deeper context is exploited.
- **Epochs 41–60 (regularization‑limited gains).** Accuracy fluctuates around **0.90** with incremental improvements.
- **Epochs 61–90 (fine‑grained tuning).** Small oscillations with new highs, culminating in the **best** **0.9160** at epoch 89.
- **Epochs 91–100 (saturation).** No new highs; use the saved best checkpoint rather than the final epoch.

---

## Final Validation Breakdown (best model @ epoch 89)

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

## Practical Guidance

- For quick, strong performance, **train to ~40 epochs** (≈**0.88** mean Dice).  
- For maximum accuracy, **train to ~90 epochs** and use the **best checkpoint** (≈**0.916** mean Dice).  
- If VRAM is limited, reduce `--base` or the crop size; this mostly affects speed rather than convergence trend.  
- Maintain identical crop size and overlap between training and sliding‑window inference to avoid tile boundary artifacts.

---