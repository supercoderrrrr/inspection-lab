# Changelog

## 0.4.2 — 2026-10-07

- Integrate the complete implementation and recorded studies from the earlier workspace.
- Add metal-nut inspection, paired seeds, nested budgets and brightness/blur checks.
- Compare PaDiM, PatchCore, a fixed 125-patch coreset and the pixel-template baseline.
- Publish all 18 run records with calibration scores, split hashes and metric verification.
- Add batch inference, custom product validation, artifact audits and release packaging.
- Complete the five-tab UI with aligned response/annotation views and controlled comparisons.
- Distribute two prespecified PatchCore inference models separately from source and data.
- Add Windows/Linux CPU CI and retain the initial CLI checkpoint format and evidence.

Recorded experiment timestamps identify their original executions. This version packages
the preserved implementation and evidence; it does not represent newly trained studies.

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
