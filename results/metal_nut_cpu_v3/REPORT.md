# Repeated-seed benchmark

## Prespecified protocol

Seeds: [42, 123, 2026]. Category: metal_nut. Normal-reference budgets: 16, 64, 190. Both methods use identical
training/calibration/test images per seed. Thresholds use only 30 normal calibration images.
The fixed official test set is reused; seed repetitions are not independent test datasets.

## Results (mean ± sample standard deviation across seeds)

| Method | References | Image AUROC | Recall | False-positive rate |
|---|---:|---:|---:|---:|
| PatchCore | 16 | 0.9158 ± 0.0130 | 0.6487 ± 0.0485 | 0.0303 ± 0.0262 |
| PatchCore | 64 | 0.9829 ± 0.0061 | 0.8638 ± 0.0793 | 0.0455 ± 0.0787 |
| PatchCore | 190 | 0.9891 ± 0.0033 | 0.9247 ± 0.0559 | 0.0455 ± 0.0455 |
| Pixel template | 16 | 0.4989 ± 0.0363 | 0.2509 ± 0.0164 | 0.0000 ± 0.0000 |
| Pixel template | 64 | 0.4899 ± 0.0214 | 0.2509 ± 0.0224 | 0.0303 ± 0.0525 |
| Pixel template | 190 | 0.4904 ± 0.0030 | 0.2509 ± 0.0224 | 0.0152 ± 0.0262 |

## Baseline definition

The pixel-template baseline learns a mean and sample standard deviation at each RGB pixel.
Standard deviations are floored at 0.05 on [0,1] intensities. The anomaly map is the
Gaussian-smoothed (sigma 2) root-mean-square standardized RGB residual; the image score
is its 99th percentile. These choices were fixed before running this study. The baseline
uses CPU NumPy/SciPy; PatchCore device selection is cpu. Timings reflect the recorded
runtime conditions, not an isolated hardware benchmark. Neither method is fine-tuned on defects.

## Limitations

This comparison tests one simple baseline, not all competing anomaly detectors.
The same 115 test images are reused across seeds. Standard deviations describe sensitivity to
normal-reference sampling and calibration, not population confidence intervals or significance.
A larger PatchCore budget also increases memory-bank size. Pixel-template performance depends
on alignment. Synthetic perturbations are not real deployment shifts. No operating threshold
or default model was selected to improve these observed test results.

## Reproduce

`python -m inspection.study --category metal_nut --name metal_nut_cpu_v3 --seeds 42,123,2026 --device cpu`

The plan, per-seed logs, raw predictions, calibrated thresholds, and aggregate CSV are retained.
See THIRD_PARTY.md for upstream attribution. Application and study orchestration are AI-assisted.