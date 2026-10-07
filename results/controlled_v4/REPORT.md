# Controlled comparison

## Protocol

Two categories, three paired seeds, three nested budgets. Every variant uses the exact
same image paths and hashes as the existing PatchCore/pixel-template experiments.
PaDiM: Anomalib 2.2.0, ResNet-18 layer1/2/3, 100 sampled channels, regularized covariance.
PatchCore fixed 125: identical feature extraction and scoring to the existing detector;
exactly 125 coreset patches at every budget. 125 matches the smallest prior 1% bank.
Thresholds are separately calibrated on the same 30 normal images per seed.
These clean-only follow-up experiments were specified after seeing prior results,
before running these variants. They are exploratory, not a preregistered external study.

## Results: mean ± sample standard deviation

| Category | Method | References | AUROC | Recall | False alarms |
|---|---|---:|---:|---:|---:|
| bottle | PaDiM | 16 | 0.9926 ± 0.0059 | 93.7% ± 4.2% | 8.3% ± 10.4% |
| bottle | PaDiM | 64 | 0.9944 ± 0.0044 | 97.9% ± 2.4% | 3.3% ± 2.9% |
| bottle | PaDiM | 179 | 0.9984 ± 0.0016 | 98.4% ± 2.7% | 5.0% ± 5.0% |
| bottle | PatchCore | 16 | 0.9979 ± 0.0012 | 97.9% ± 2.4% | 13.3% ± 14.4% |
| bottle | PatchCore | 64 | 0.9995 ± 0.0009 | 99.5% ± 0.9% | 8.3% ± 5.8% |
| bottle | PatchCore | 179 | 1.0000 ± 0.0000 | 100.0% ± 0.0% | 13.3% ± 10.4% |
| bottle | PatchCore fixed 125 | 16 | 0.9979 ± 0.0012 | 97.9% ± 2.4% | 13.3% ± 14.4% |
| bottle | PatchCore fixed 125 | 64 | 0.9942 ± 0.0041 | 97.4% ± 0.9% | 5.0% ± 5.0% |
| bottle | PatchCore fixed 125 | 179 | 0.9860 ± 0.0129 | 96.8% ± 3.2% | 8.3% ± 2.9% |
| bottle | Pixel template | 16 | 0.8833 ± 0.0029 | 41.3% ± 6.3% | 1.7% ± 2.9% |
| bottle | Pixel template | 64 | 0.9045 ± 0.0095 | 47.6% ± 4.8% | 0.0% ± 0.0% |
| bottle | Pixel template | 179 | 0.9011 ± 0.0032 | 53.4% ± 9.2% | 0.0% ± 0.0% |
| metal_nut | PaDiM | 16 | 0.8900 ± 0.0246 | 46.2% ± 10.9% | 1.5% ± 2.6% |
| metal_nut | PaDiM | 64 | 0.9598 ± 0.0041 | 72.0% ± 21.5% | 3.0% ± 2.6% |
| metal_nut | PaDiM | 190 | 0.9573 ± 0.0076 | 76.0% ± 17.5% | 7.6% ± 6.9% |
| metal_nut | PatchCore | 16 | 0.9158 ± 0.0130 | 64.9% ± 4.8% | 3.0% ± 2.6% |
| metal_nut | PatchCore | 64 | 0.9829 ± 0.0061 | 86.4% ± 7.9% | 4.5% ± 7.9% |
| metal_nut | PatchCore | 190 | 0.9891 ± 0.0033 | 92.5% ± 5.6% | 4.5% ± 4.5% |
| metal_nut | PatchCore fixed 125 | 16 | 0.9158 ± 0.0130 | 64.9% ± 4.8% | 3.0% ± 2.6% |
| metal_nut | PatchCore fixed 125 | 64 | 0.9080 ± 0.0343 | 65.2% ± 11.2% | 0.0% ± 0.0% |
| metal_nut | PatchCore fixed 125 | 190 | 0.8749 ± 0.0233 | 62.7% ± 4.3% | 9.1% ± 7.9% |
| metal_nut | Pixel template | 16 | 0.4989 ± 0.0363 | 25.1% ± 1.6% | 0.0% ± 0.0% |
| metal_nut | Pixel template | 64 | 0.4899 ± 0.0214 | 25.1% ± 2.2% | 3.0% ± 5.2% |
| metal_nut | Pixel template | 190 | 0.4904 ± 0.0030 | 25.1% ± 2.2% | 1.5% ± 2.6% |

## Interpretation boundaries

PaDiM uses different layers, feature dimensions and covariance statistics; this is a
practical method comparison, not an isolation of scoring distance alone. With fewer
training images than channels, covariance regularization is especially important.
The fixed-bank experiment controls final bank capacity but still changes reference
coverage and the candidate pool used by coreset selection. No guarantee of monotonic
improvement follows. Test sets are reused across seeds; standard deviations describe
split sensitivity. Timings from different sessions are not controlled hardware comparisons.

## Reproduce

`python -m inspection.research --name controlled_v4`

See plan.json, runs.csv, aggregate.csv and per-run calibration/prediction CSVs.