# Inspection Lab — Experiment Report

## Question
How does the number of normal reference images affect industrial anomaly detection,
and how stable is the largest prespecified model under controlled image perturbations?

## Method
This application uses Anomalib's PatchCore implementation, not a newly invented algorithm.
The backbone is `resnet18` with ImageNet pretrained weights, layer2/layer3
features, 224 × 224 RGB inputs, ImageNet normalization,
and a 1.0% coreset. This compact configuration is not a reproduction
of the original paper's headline benchmark. No gradient-based fine-tuning is performed.

## Evaluation protocol
- Dataset: original MVTec AD, bottle category; 209 normal training images and 83 official test images.
- Hold out 30 normal training images for calibration before selecting nested training budgets.
- Training budgets: [16, 64, 179]; seed: 42.
- Select the upper split-conformal order statistic at alpha=0.05 from normal calibration scores.
- Predict anomaly only when the raw image score strictly exceeds this threshold.
- Test labels and masks do not participate in fitting, calibration, or default-model selection.
- The default model uses the largest prespecified budget, regardless of test scores.
- Thresholds stay fixed for brightness ×0.7 and Gaussian blur radius 1.5 stress tests, when enabled.
- Pixel AUROC uses nearest-neighbor resized ground-truth masks at the input resolution.
- Latency uses a warmed-up model, batch size one, and device synchronization; decoding and preprocessing are excluded.

## Recorded results
| budget | condition | image_auroc | pixel_auroc | recall | false_positive_rate | latency_median_ms |
|---|---|---|---|---|---|---|
| 16 | clean | 0.9968 | 0.9737 | 0.9841 | 0.0500 | 47.6921 |
| 64 | clean | 1.0000 | 0.9762 | 1.0000 | 0.0500 | 45.0020 |
| 179 | clean | 1.0000 | 0.9761 | 1.0000 | 0.0500 | 52.2812 |
| 179 | dim | 1.0000 | 0.9735 | 1.0000 | 0.0000 | 48.6110 |
| 179 | blur | 1.0000 | 0.9765 | 1.0000 | 0.0000 | 41.1308 |

![Budget comparison](budget_comparison.png)

## Interpretation and limitations
AUROC measures ranking across thresholds. Recall and false-positive rate measure the
single threshold calibrated without anomalies; a high AUROC does not guarantee high recall
at this operating point. The nominal false-alarm level relies on exchangeable normal data,
not a guarantee on this small test set or on shifted deployment images.

These are one-category, one-seed observations. Budget comparisons change both reference-image
diversity and memory-bank size. Synthetic brightness/blur tests do not establish real-factory
robustness. The dataset has 20 normal and 63 anomalous test images, so each normal error changes
the observed false-positive rate by five percentage points. Scores are distances, not probabilities.
Heatmap colors use a calibration-derived fixed scale, not a validated pixel decision threshold.

## Reproducibility
See `config.json`, `environment.json`, `splits.json`, `summary.csv`, and the per-budget
calibration/prediction CSV files. Splits include image SHA-256 hashes. Checkpoints contain
tensor state and configuration; load only artifacts from a trusted source.

## Attribution and contribution
Upstream: Anomalib (Apache-2.0), PatchCore (Roth et al., CVPR 2022), and MVTec AD
(Bergmann et al., CVPR 2019; dataset CC BY-NC-SA 4.0).
Project-specific work: split/calibration protocol, nested budget experiments, stress tests,
artifact provenance, English inspection UI, and generated reporting. Development is AI-assisted.
No claim is made that the upstream detector was authored by the project owner.
