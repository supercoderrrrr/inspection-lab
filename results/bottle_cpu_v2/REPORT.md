# Repeated-seed benchmark

## Prespecified protocol

Seeds: [42, 123, 2026]. Normal-reference budgets: 16, 64, 179. Both methods use identical
training/calibration/test images per seed. Thresholds use only 30 normal calibration images.
The fixed official test set is reused; seed repetitions are not independent test datasets.

## Results (mean ± sample standard deviation across seeds)

| Method | References | Image AUROC | Recall | False-positive rate |
|---|---:|---:|---:|---:|
| PatchCore | 16 | 0.9979 ± 0.0012 | 0.9788 ± 0.0242 | 0.1333 ± 0.1443 |
| PatchCore | 64 | 0.9995 ± 0.0009 | 0.9947 ± 0.0092 | 0.0833 ± 0.0577 |
| PatchCore | 179 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.1333 ± 0.1041 |
| Pixel template | 16 | 0.8833 ± 0.0029 | 0.4127 ± 0.0635 | 0.0167 ± 0.0289 |
| Pixel template | 64 | 0.9045 ± 0.0095 | 0.4762 ± 0.0476 | 0.0000 ± 0.0000 |
| Pixel template | 179 | 0.9011 ± 0.0032 | 0.5344 ± 0.0916 | 0.0000 ± 0.0000 |

## Baseline definition

The pixel-template baseline learns a mean and sample standard deviation at each RGB pixel.
Standard deviations are floored at 0.05 on [0,1] intensities. The anomaly map is the
Gaussian-smoothed (sigma 2) root-mean-square standardized RGB residual; the image score
is its 99th percentile. These choices were fixed before running this study. The baseline
uses CPU NumPy/SciPy; PatchCore device selection is cpu. Timings reflect the recorded
runtime conditions, not an isolated hardware benchmark. Neither method is fine-tuned on defects.

## Limitations

This comparison tests one simple baseline, not all competing anomaly detectors.
The same 83 test images are reused across seeds. Standard deviations describe sensitivity to
normal-reference sampling and calibration, not population confidence intervals or significance.
A larger PatchCore budget also increases memory-bank size. Pixel-template performance depends
on alignment. Synthetic perturbations are not real deployment shifts. No operating threshold
or default model was selected to improve these observed test results.

## Reproduce

`python -m inspection.study --name bottle_cpu_v2 --seeds 42,123,2026 --device cpu`

The plan, per-seed logs, raw predictions, calibrated thresholds, and aggregate CSV are retained.
See THIRD_PARTY.md for upstream attribution. Application and study orchestration are AI-assisted.