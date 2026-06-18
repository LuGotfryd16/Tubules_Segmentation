---
license: apache-2.0
pipeline_tag: image-segmentation
---


# TubuleSegmentation v1.0.0 — Seminiferous tubule segmentation and morphometry (mouse testis, H&E)

Semantic segmentation model for seminiferous tubules in H&E sections of mouse testis
(*Mus musculus*, CF-1 strain). Segments 3 classes (0 = background, 1 = epithelium,
2 = lumen) and derives calibrated morphometric metrics. Architecture: EfficientNet-B4
encoder + UNet with a dual decoder (segmentation + boundary) and SCSE attention.
Input 512×512, ImageNet normalization, scale 0.32 µm/px.


---

## 1. Segmentation performance (validation, n = 49)

| Class          | IoU   | Dice / F1 | clDice |
|----------------|-------|-----------|--------|
| Background (0) | 0.990 | 0.995     | —      |
| Epithelium (1) | 0.938 | 0.968     | 0.984  |
| Lumen (2)      | 0.935 | 0.967     | 0.968  |
| Mean           | mIoU 0.954 | 0.977 | —    |

Per-image (n = 49): mIoU 0.953 ± 0.013 (min 0.900, median 0.955);
mDice 0.976 ± 0.007 (min 0.946). No image falls below mIoU 0.90.
Metrics computed on the post-processed predicted masks vs. reference masks; validation set, never train.


![Example segmentation output](https://huggingface.co/LuGot16/Tubules_Segmentation/resolve/main/assets/figure_segmentation_examples_hf.png)

---

## 2. Agreement with manual measurement (ImageJ, gold standard) — validation, n = 48

### Tubule
| Metric | Pearson r | CCC (Lin) [95% CI] | Bias | Rel. error (%) | 95% LoA | MAE | Scale factor [95% CI] |
|---|---|---|---|---|---|---|---|
| Area (µm²) | 0.996 | 0.773 [0.701, 0.818] | +4074.6 | 12.2% | [+2530.3, +5618.9] | 4074.6 | 1.126 [1.108, 1.132] |
| Major axis (µm) | 0.996 | 0.831 [0.775, 0.863] | +14.34 | 6.3% | [+9.17, +19.52] | 14.34 | 1.065 [1.058, 1.066] |
| Minor axis (µm) | 0.995 | 0.782 [0.691, 0.838] | +11.91 | 6.3% | [+8.39, +15.44] | 11.91 | 1.066 [1.058, 1.068] |
| Max Feret (µm) | 0.997 | 0.850 [0.794, 0.882] | +13.43 | 5.9% | [+8.54, +18.32] | 13.43 | 1.058 [1.055, 1.063] |
| Min Feret (µm) | 0.995 | 0.796 [0.702, 0.850] | +11.38 | 6.1% | [+7.62, +15.15] | 11.38 | 1.062 [1.055, 1.066] |
| Perimeter (µm) | 0.996 | 0.790 [0.721, 0.833] | +38.26 | 5.8% | [+26.87, +49.64] | 38.26 | 1.059 [1.052, 1.063] |
| Aspect ratio | 0.999 | 0.999 [0.997, 0.999] | −0.0002 | 0.4% | [−0.013, +0.012] | 0.005 | 0.999 [0.997, 1.002] |
| Roundness | 0.996 | 0.993 [0.988, 0.996] | −0.006 | 0.8% | [−0.020, +0.008] | 0.007 | 0.993 [0.992, 0.995] |

### Lumen
| Metric | Pearson r | CCC (Lin) [95% CI] | Bias | Rel. error (%) | 95% LoA | MAE | Scale factor [95% CI] |
|---|---|---|---|---|---|---|---|
| Area (µm²) | 0.993 | 0.948 [0.922, 0.963] | +1114.7 | 8.1% | [+37.7, +2191.8] | 1114.7 | 1.075 [1.067, 1.091] |
| Major axis (µm) | 0.970 | 0.826 [0.746, 0.877] | +11.87 | 8.0% | [+1.84, +21.90] | 11.87 | 1.074 [1.070, 1.081] |
| Minor axis (µm) | 0.989 | 0.880 [0.800, 0.918] | +9.78 | 8.4% | [+3.69, +15.87] | 9.78 | 1.082 [1.072, 1.098] |
| Max Feret (µm) | 0.966 | 0.950 [0.925, 0.965] | +4.07 | 3.7% | [−7.67, +15.82] | 6.41 | 1.033 [1.027, 1.037] |
| Min Feret (µm) | 0.983 | 0.972 [0.953, 0.982] | +3.06 | 3.1% | [−4.41, +10.53] | 4.16 | 1.031 [1.017, 1.034] |
| Perimeter (µm) | 0.934 | 0.931 [0.889, 0.957] | −6.96 | 5.9% | [−115.5, +101.5] | 41.45 | 1.008 [0.984, 1.028] |
| Aspect ratio | 0.974 | 0.971 [0.933, 0.987] | −0.008 | 2.7% | [−0.108, +0.093] | 0.036 | 0.991 [0.982, 1.004] |
| Roundness | 0.939 | 0.818 [0.709, 0.887] | −0.059 | 7.5% | [−0.138, +0.020] | 0.059 | 0.937 [0.921, 0.948] |
---

## 3. Data and training

| Item                  | Value                                                              |
|-----------------------|-------------------------------------------------------------------|
| Tissue / species      | Mouse testis (*Mus musculus*), CF-1 strain; seminiferous tubules, H&E |
| Dataset               | `LuGot16/tubules` (HuggingFace) — 322 images                       |
| Source material       | 11 animals × 2 slides per animal, across 3 experiments            |
| Train / Validation    | 273 / 49 images, random 85/15 split (seed 42)                     |
| Split level           | Image-level random split (not grouped by animal)                  |
| Scale                 | 0.32 µm/px                                                         |
| Input                 | 512×512, ImageNet normalization                                   |
| Classes               | 0 = background, 1 = epithelium, 2 = lumen                         |
| Augmentation          | Macenko (stain normalization) + geometric                         |
| Post-processing       | largest connected component → morphological closing → hole filling → lumen cleanup |
| Inference             | 8× TTA (4 rotations × 2 flips)                                    |

---

## 4. Robustness, known biases, and scope

### Generalization — external validation (other animals, stains, fixatives)

The train/validation split is random at the image level, not grouped by animal, so images
from the same animal may appear in both sets; the validation metrics in Sections 1–2 may be
optimistic for fully unseen animals. The primary evidence for cross-animal generalization is
an external set of **129 images from other animals and experiments**, with different stains and
fixatives (~0.32 µm/px). On a hand-traced subset, the model was evaluated quantitatively.

**Segmentation (per-image mean, n = 29):**

| Class          | IoU   | Dice  | clDice |
|----------------|-------|-------|--------|
| Background     | 0.986 | 0.993 | —      |
| Epithelium     | 0.885 | 0.938 | 0.948  |
| Lumen          | 0.867 | 0.927 | 0.927  |
| **Mean**       | **mIoU 0.913** | **0.953** | — |

Per-image mIoU 0.913 (median 0.924, min 0.811) vs. 0.953 in-domain — a modest, expected
cross-domain drop.

**Boundary accuracy (n = 29):**

| Boundary (µm) | ASSD | HD95  | signed disp. | BF @1.6 µm |
|---------------|------|-------|--------------|------------|
| Tubule        | 0.97 | 3.06  | +0.48        | 0.85       |
| Lumen         | 3.86 | 13.90 | +0.72        | 0.53       |

**Morphometric agreement vs. manual ImageJ (n = 29):**

*Tubule*

| Descriptor | Pearson r | CCC | Rel. error | Scale factor |
|------------|-----------|-----|------------|--------------|
| Area       | 0.998 | 0.769 | 14.3% | 1.142 |
| Perimeter  | 0.966 | 0.819 | 5.6%  | 1.061 |
| Major axis | 0.997 | 0.852 | 6.9%  | 1.069 |
| Minor axis | 0.997 | 0.753 | 7.4%  | 1.072 |
| Max Feret  | 0.996 | 0.863 | 6.3%  | 1.064 |
| Min Feret  | 0.993 | 0.756 | 7.1%  | 1.069 |
| Aspect ratio | 0.995 | 0.994 | 0.8% | 0.997 |
| Roundness  | 0.993 | 0.993 | 0.9%  | 0.999 |

*Lumen*

| Descriptor | Pearson r | CCC | Rel. error | Scale factor |
|------------|-----------|-----|------------|--------------|
| Area       | 0.955 | 0.892 | 14.0% | 1.110 |
| Perimeter  | 0.790 | 0.681 | 13.4% | 0.909 |
| Major axis | 0.941 | 0.842 | 9.1%  | 1.077 |
| Minor axis | 0.939 | 0.759 | 11.1% | 1.101 |
| Max Feret  | 0.924 | 0.917 | 5.1%  | 1.010 |
| Min Feret  | 0.935 | 0.915 | 5.3%  | 1.034 |
| Aspect ratio | 0.925 | 0.881 | 4.1% | 0.983 |
| Roundness  | 0.893 | 0.802 | 6.0%  | 0.954 |

The size bias (+6–14%) and the near-perfect agreement of the shape descriptors
(tubule AR/roundness CCC ≈ 0.99, scale factor ≈ 1.00) reproduce the in-domain pattern,
confirming the offset is a **domain-stable** scaling difference, not out-of-domain distortion.
Lumen perimeter is the only noisy metric (r 0.79), reflecting the scalloped, low-contrast
lumen boundary.

### Known biases

- Boundary accuracy vs. reference masks (validation, n = 49):
  - Tubule: ASSD 0.59 µm, HD95 1.54 µm, mean signed displacement +0.13 µm (Boundary-F1 @1.6 µm = 0.97).
  - Lumen: ASSD 1.47 µm, HD95 4.99 µm, mean signed displacement −0.05 µm (Boundary-F1 @1.6 µm = 0.75).

  The model reproduces the reference masks with sub-pixel mean boundary agreement; the
  external boundary displacement stays small (tubule +0.48 µm ≈ 1.5 px). The ~+12% / +8%
  (tubule / lumen) area difference vs. manual ImageJ **freehand** tracing is therefore a
  convention difference between the reference masks and the freehand protocol — faithfully
  inherited by the model — not error introduced by it. It is systematic (r ≈ 0.99) and
  preserves shape (**tubule** AR/roundness CCC ≈ 0.99; lumen roundness lower, CCC 0.82).

- **Lumen perimeter**: good mean agreement in-domain (CCC 0.93, bias ≈ 0) but wide LoA
  (±110 µm), varying case by case with tracing convention.

### Failure mode

One badly-fixed external tubule was grossly under-segmented: the lumen was missed
entirely and only ~⅓ of the tubule area was captured (tubule boundary displaced
−10 µm inward, a clear outlier). The model's **status output flagged it as CHECK**
(1 of 9 flagged across the 129 images), so the failure is surfaced for review rather
than passed off as a valid measurement.

### Validated scope

Single-tubule crops at 0.32 µm/px, H&E staining. **Not validated** on multi-tubule fields or
other magnifications.

---

## Attribution

Dataset curated by Lucila Gotfryd (image acquisition, annotation, design of the segmentation
and morphometry approach, including the anatomically-motivated containment constraint and the
choice to enforce tubular connectivity). Model implementation carried out with AI coding
assistance ([ML Intern](https://github.com/huggingface/ml-intern)) under the author's direction.

