# Changelog

## 0.3.0 — 2026-10-06

- Add a local inspection UI backed by the same inference pipeline as the CLI.
- Validate uploads and clear results when the image or model changes.
- Display model responses and dataset annotations separately.
- Export overlay images, raw maps and result JSON without rerunning inspection.
- Show verified saved evaluations and readable failure cases.
- Cover empty source checkouts, input changes and browser workflows with tests.

## 0.2.0 — 2026-10-06

- Hold out normal images for fixed, finite-sample threshold calibration.
- Keep fitting, calibration and test inputs separate.
- Export full test predictions, calibration evidence, metrics and source hashes.
- Verify evidence and recompute reported metrics without a dataset or checkpoint.
- Publish the first single-seed, 16-reference bottle comparison.

## 0.1.0 — 2026-10-04

- Validate MVTec-style data and reject training/test content overlap.
- Fit and serialize a pixel-template baseline and Anomalib PatchCore.
- Export single-image raw scores and response maps from the CLI.
