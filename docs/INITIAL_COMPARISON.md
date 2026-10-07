# Initial bottle comparison

Both methods use seed 42, 16 normal fitting references, the same 30 held-out normal
calibration images, a 5% false-alarm target and the official 83-image bottle test split.
Thresholds were fixed before test scoring. Input size is 224 pixels.

| Method | Image AUROC | Defect recall | Normal false alarms | Missed defects | False alarms |
|---|---:|---:|---:|---:|---:|
| Pixel template | 0.8865 | 34.9% | 0/20 | 41/63 | 0 |
| PatchCore | 0.9802 | 96.8% | 1/20 | 2/63 | 1 |

PatchCore ranks and detects defects more effectively in this initial comparison. The pixel
template misses many defects at its normal-calibrated threshold. One normal false alarm
changes the observed rate by five percentage points; these small counts limit conclusions.
This is one product, seed and budget. No test result selected a checkpoint or threshold.

Each method directory contains the full predictions, calibration scores, source-image
hashes, model configuration, dependency versions, a failure report and evidence checksums.
The checkpoints and dataset images are excluded. To verify after installing the project:

```bash
python -m inspection verify-results --results results/bottle-baseline
python -m inspection verify-results --results results/bottle-patchcore
```

This verifies consistency of the published evidence. Reproducing inference requires the
original dataset and a freshly fitted model with the recorded configuration. Floating-point
results can vary with dependency versions and hardware.

Dataset: MVTec AD, MVTec Software GmbH; see [attribution](../THIRD_PARTY.md).
