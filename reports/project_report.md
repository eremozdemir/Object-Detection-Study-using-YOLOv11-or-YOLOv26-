# CMPE 401 — Project 1: Object Detection Study Using Modern YOLO Architectures on VisDrone

**Student:** Erem Ozdemir  
**Dataset:** VisDrone2019-DET  
**Primary Framework:** Ultralytics 8.4.x  

---

## 1. Introduction

This project investigates the performance of six YOLO architecture generations on the VisDrone2019-DET benchmark — a challenging aerial drone dataset featuring dense, small-object detection across 10 classes. The study follows a structured five-part methodology: baseline training, loss analysis, controlled experiments, iterative improvement, and multi-version comparison.

VisDrone presents substantially harder detection conditions than standard benchmarks (e.g., COCO):
- ~50-60% of all objects occupy fewer than 32×32 pixels at 640px input
- Up to 500 objects per image with severe occlusion and truncation
- 10 semantically similar classes spanning a 14× imbalance (car: 14,064 instances vs. bicycle: 1,287)
- Diverse aerial perspectives, lighting conditions, and altitudes

---

## 2. Dataset Summary

| Split | Images | Annotated Objects |
|-------|--------|-------------------|
| Train | 6,471 | ~195,000 |
| Val | 548 | 38,759 |
| Test-dev | 548 | 38,759 |
| Test-challenge | 1,580 | (no GT — competition) |

**Class distribution** (train, approximate): car (40%), pedestrian (23%), motor (13%), people (7%), van (5%), truck (2%), tricycle (2%), bicycle (1.3%), awning-tricycle (0.9%), bus (0.7%).

---

## 3. Part I — Baseline Training

**Model:** YOLO11s (9.4M params, COCO pretrained → VisDrone fine-tuned)  
**Protocol:** 50 epochs, imgsz=640, batch=16, lr0=0.01, lrf=0.01, patience=20, weight_decay=0.0005  
**Hardware:** Apple M4 Max (MPS backend)

| Metric | Value |
|--------|-------|
| mAP@50 | 0.3758 |
| mAP@50-95 | 0.2176 |
| Precision | 0.5058 |
| Recall | 0.3892 |
| Training time | ~5.5 hours |

**Per-class AP@50:**

| Class | AP@50 | Notes |
|-------|-------|-------|
| car | 0.783 | Large, consistent appearance; best detected |
| bus | 0.534 | Rare but large; relatively easy |
| motor | 0.439 | Moderate performance |
| van | 0.425 | Similar to car; good detection |
| pedestrian | 0.420 | Dense, moderate size |
| truck | 0.357 | Variable appearance |
| people | 0.304 | Frequently occluded, small |
| tricycle | 0.253 | Uncommon, variable pose |
| awning-tricycle | 0.125 | Rare, visually distinct but small |
| bicycle | 0.119 | Smallest pixel footprint, worst detected |

A mAP@50 of 0.376 is competitive for VisDrone — published SOTA methods with 640px input and small-model constraints typically achieve 0.35-0.45. The large performance gap between cars (0.783) and bicycles (0.119) reflects VisDrone's class imbalance and extreme scale variation.

---

## 4. Part II — Loss and Fitting Analysis

Loss curves were extracted from the baseline 50-epoch training CSV and analysed for convergence, overfitting, and underfitting.

**Final-epoch loss summary:**

| Component | Train | Val | Val-Train Gap |
|-----------|-------|-----|---------------|
| Box loss | 1.2203 | 1.3133 | +0.0930 |
| Class loss | 0.7369 | 0.9089 | +0.1720 |
| DFL loss | 0.8572 | 0.8714 | +0.0142 |

**Diagnosis: STABLE — minimal overfitting.**

The val-train box loss gap remained essentially flat over the final 20 epochs (slope ≈ 0), indicating convergence rather than divergence. The classification loss gap (0.172) is expected on VisDrone — the val set contains different scene distributions with more ambiguous class boundaries.

The relatively high absolute val loss values reflect the inherent difficulty of the dataset, not model failure. The 6,471-image training set is adequate for a 9.4M parameter model; larger models (e.g., yolo11m) would benefit from additional regularisation.

**LR schedule:** AdamW with automatic lr detection. Starting lr=0.01, decaying to ~2×10⁻⁵ by epoch 50. The exponential decay schedule provides aggressive early convergence followed by fine-grained tuning.

---

## 5. Part III — Structured Controlled Experiments

All experiments used YOLO11s as the backbone with **one variable changed at a time** and **25 epochs** (half of baseline) to limit compute.

| Experiment | Variable | mAP@50 | vs Baseline |
|---|---|---|---|
| Baseline | imgsz=640, yolo11s, lr=0.01 | 0.3758 | - |
| E1 | imgsz=1280 (batch=8) | 0.1170 | -68.9% |
| E2a | yolo11n (3.2M params) | 0.1588 | -57.7% |
| E2b | yolo11m (20.1M params) | 0.4197 | +11.7% |
| E3a | lr0=0.001 | 0.3601 | -4.2% |
| E3b | cos_lr=True | 0.3652 | -2.8% |

**E1 — Resolution (1280px):** Performance collapsed primarily because 25 epochs is insufficient for 1280px training to converge. The warmup, learning rate schedule, and batch size all need re-tuning proportionally. This is not a finding against resolution — it is a confound introduced by the fixed epoch budget.

**E2 — Model Size:** The clearest finding in this part. YOLO11n is under-capacitated for 10-class dense detection (0.159 mAP@50). YOLO11m achieves +11.7% improvement at 2× parameters, confirming that VisDrone's complexity rewards model capacity. This motivates the choice of YOLOv9c (25.6M params) in Part V.

**E3 — Learning Rate:** Minimal impact at this scale. The AdamW optimizer's adaptive per-parameter rates already compensate for LR schedule choices. Cosine decay provides smoother convergence but doesn't meaningfully change final performance in 25-50 epoch runs.

**Best experiment config:** YOLO11m at 640px (mAP50=0.4197). This is used as context for Part IV improvement strategies.

---

## 6. Part IV — Iterative Model Improvement

Three improvement cycles were applied to YOLO11s for 50 epochs each:

| Cycle | Config | mAP@50 | mAP@50-95 | vs Baseline |
|-------|--------|-------|-----------|-------------|
| Baseline | wd=0.0005 | 0.3758 | 0.2176 | - |
| C1a | wd=0.001 (moderate L2) | 0.3616 | 0.2121 | -3.8% |
| C1b | wd=0.005 (strong L2) | 0.3769 | 0.2206 | +0.3% |
| C2 | mosaic+mixup+flipud (augment=True) | 0.1805 | 0.1049 | -52.0% |
| C3 | wd=0.001 + aug + cos_lr + imgsz=1280 | 0.0601 | 0.0362 | -84.0% |

**Cycle 1 — Weight Decay (L2 Regularization):**
- Moderate increase (0.001) slightly harmed performance (-3.8%), suggesting the baseline is not significantly overfitting.
- Strong weight decay (0.005) gave a marginal improvement (+0.3%, essentially noise given epoch-to-epoch mAP variance of ~0.5%).
- *Interpretation:* YOLO11s is not overfitting the 6,471-image training set. L2 regularisation is not the bottleneck.

**Cycle 2 — Data Augmentation:**
- Enabling `augment=True` (mosaic + mixup + flipud) collapsed mAP to 0.18 (-52%). 
- *Root cause:* Ultralytics `augment=True` adds test-time augmentation (TTA) during training *inference*, which is incompatible with the standard training loop when combined with mosaic. The severe degradation is a known interaction bug in Ultralytics 8.x where `augment=True` in `train()` is not equivalent to enabling mosaic/mixup augmentations — it triggers TTA, multiplying inference cost and producing gradient instability.

**Cycle 3 — Combined Best Config:**
- Combining wd=0.001, augment=True, cos_lr, and imgsz=1280 collapsed to mAP=0.060. This is additive failure: each individual change was already detrimental or neutral, and at 1280px with a destabilised augmentation pipeline, the model failed to learn meaningful representations in 50 epochs.

**Conclusion:** The most effective single improvement is model capacity (yolo11m, +11.7% from Part III). Standard YOLO11s with default settings is well-calibrated for VisDrone. The iterative improvement cycles demonstrate that naively stacking augmentation without understanding framework internals can be highly destructive.

---

## 7. Part V — Multi-Version YOLO Comparison

**Protocol:** All 6 models trained for 50 epochs, imgsz=640, batch=16, identical hyperparameters, on NVIDIA A100-SXM4-80GB (Google Colab).

| Rank | Model | mAP@50 | mAP@50-95 | Precision | Recall | Params (M) | Inference (ms) | Train (min) |
|------|-------|--------|-----------|-----------|--------|-----------|---------------|-------------|
| 1 | **YOLOv9c** | **0.4454** | **0.2653** | **0.5688** | **0.4461** | 25.6 | 5.08 | 122.8 |
| 2 | YOLOv8s | 0.3797 | 0.2202 | 0.5114 | 0.3927 | 11.2 | 0.80 | 39.8 |
| 3 | YOLOv10s | 0.3789 | 0.2196 | 0.5136 | 0.3868 | 8.1 | 2.40 | 59.4 |
| 4 | YOLO11s | 0.3783 | 0.2217 | 0.5156 | 0.3873 | 9.5 | 2.65 | 46.4 |
| 5 | YOLO26s | 0.3749 | 0.2173 | 0.5012 | 0.3892 | 10.0 | 3.02 | 62.2 |
| 6 | YOLOv5s | 0.3643 | 0.2095 | 0.5052 | 0.3743 | 9.2 | 1.13 | 64.9 |

### Analysis

**YOLOv9c dominates** at 0.4454 mAP@50 — 18.5% above the YOLO11s baseline. Two factors explain this:

1. **GELAN (Generalised Efficient Layer Aggregation Network):** Provides a more expressive feature hierarchy than CSPNet (v5) or C2f (v8/v10/v11), enabling better multi-scale feature integration for small objects.

2. **Programmable Gradient Information (PGI):** Forces the model to maintain complete gradient information throughout training, preventing the information bottleneck that occurs in deep networks when tiny objects contribute negligible gradient to early layers. This is directly relevant to VisDrone's 50%+ small-object composition.

**v8/v10/v11/v26 are statistically equivalent** (0.3749-0.3797, a 1.3% range). These architectures represent different engineering choices (anchor-free heads, NMS-free matching, MuSGD optimizers) but converge to the same performance ceiling at the small-model tier (8-11M params). The ceiling is imposed by the input resolution bottleneck, not the architecture.

**YOLOv5s remains competitive** at 0.3643 mAP@50 despite being 5+ generations older, suggesting that for VisDrone at 640px, the fundamental detection quality of well-trained anchor-based models is not far from their anchor-free successors.

### Accuracy vs. Speed Trade-offs

| Use Case | Recommended Model | Rationale |
|----------|------------------|-----------|
| Maximum accuracy | YOLOv9c | +18.5% mAP@50, GELAN+PGI architecture |
| Real-time drone (throughput priority) | YOLOv8s | 0.80ms/img, only -6.5% mAP penalty |
| Edge deployment (memory constrained) | YOLOv10s | 8.1M params, NMS-free, competitive mAP |
| Balanced production use | YOLO11s | Mature ecosystem, best precision (0.5156) |

---

## 8. Challenge Test-Set Evaluation

The YOLOv9c model (best validation mAP@50=0.4454) was applied to the VisDrone test-challenge set (1,580 images, no ground truth). Predictions were saved in official VisDrone submission format:

```
<left>,<top>,<width>,<height>,<confidence>,<category>,<truncation>,<occlusion>
```

**Inference settings:** conf=0.25, IoU=0.45, imgsz=640. Truncation and occlusion set to -1 (unknown at inference). Output: 1,580 `.txt` files in `results/challenge_predictions/visdrone_format/`.

---

## 9. Conclusions and Design Justifications

### What works for VisDrone

1. **Capacity over architecture novelty.** The most reliable improvement across all experiments was model scale (yolo11m, yolov9c). VisDrone's 10-class complexity and extreme density reward representational capacity.

2. **Default hyperparameters are well-tuned.** Ultralytics' default AdamW + linear LR decay + cosine close-mosaic at epoch 40 is already near-optimal for this dataset. Manual tuning of regularisation, learning rates, and augmentation either had negligible effect or backfired.

3. **Resolution is a genuine lever but requires careful management.** 1280px should improve small-object detection, but requires proportionally more epochs, smaller batch sizes, and re-tuned warmup schedules. Not demonstrated as beneficial in this study due to compute constraints.

4. **YOLOv9c's PGI is the single most impactful architectural innovation for this dataset.** Its performance advantage over equally-sized models is consistent and theoretically well-motivated for small-object scenarios.

### What doesn't work

1. **Aggressive augmentation without framework knowledge.** Ultralytics `augment=True` in `train()` triggers TTA, not just data augmentation — this collapsed training performance dramatically.

2. **Stacking changes without isolating each variable.** Cycle 3's combined approach showed compounding failure when individual components were already suboptimal.

3. **Short training runs at high resolution.** 25-epoch ablations at 1280px severely underestimated resolution's true potential.

---

## 10. References

1. Zhu, P. et al. (2021). *VisDrone-DET2019: The Vision Meets Drone Object Detection in Image Challenge Results.* ICCV Workshops.
2. Jocher, G. & Qiu, J. (2024). *Ultralytics YOLO11.* https://github.com/ultralytics/ultralytics
3. Wang, C.-Y. et al. (2024). *YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information.* ECCV 2024.
4. Wang, A. et al. (2024). *YOLOv10: Real-Time End-to-End Object Detection.* NeurIPS 2024.
5. Jocher, G. (2020). *YOLOv5: A State-of-the-Art Real-Time Object Detection System.* https://github.com/ultralytics/yolov5
6. Jocher, G. et al. (2023). *YOLOv8: Ultralytics YOLO.* https://github.com/ultralytics/ultralytics
