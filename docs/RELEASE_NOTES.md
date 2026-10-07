# Inspection Lab 0.4.2

This release integrates the complete implementation from the earlier project workspace
with the staged Git repository. Existing commits and the initial comparison are retained.
Historical experiments keep their execution timestamps and source hashes.

## Added

- PaDiM with the same splits, budgets and normal-only calibration as existing studies.
- Exact 125-patch PatchCore coreset experiment across all budgets.
- Two categories and three seeds for both new variants; historical runs are preserved.
- Custom annotated datasets with exact-content duplicate detection.
- File/folder inference with per-image error handling, CSV/JSON and optional overlays.
- Public prediction/calibration evidence and an independent verifier.
- Editable installation, CPU instructions, Windows/Linux CI and portable browser setup.
- Separate source/model archives and checksums.
- Aligned input, response and annotation views; response-weighted opacity and numeric raw maps.
- Windows/Linux CPU checks and compatibility with the initial CLI checkpoint format.

## Compatibility and limits

Study-format PatchCore checkpoints remain readable. Default models remain the prespecified
seed42 PatchCore models, independent of new scores. Other variants can be selected from
completed local runs. The compact model bundle includes PatchCore only. No claim of factory
readiness, calibrated pixel segmentation or zero-shot product transfer is made.

Custom evaluation requires PNG images and masks for defective test images. Unannotated
PNG/JPEG images support batch or interactive inference. Controlled follow-up comparisons
are exploratory. Validation details are in `docs/VALIDATION.md`; current hosted status is
available on the repository Actions page.

## Download and run

1. Clone or download the source and follow the CPU installation in README.md.
2. Check `inspection-lab-models-v0.4.2.zip` against `SHA256SUMS.txt` and extract into the checkout.
3. Run `python -m streamlit run app.py --server.address 127.0.0.1` and select Upload image.

The model archive contains two seed42 PatchCore models (bottle and metal nut). It contains
no dataset images. Download a category separately to inspect dataset examples, or reproduce
the study commands to fit PaDiM and the other comparison variants. GitHub hosts source and
downloadable assets; each user runs the application locally.

The earlier `metadata.json` model format remains supported by `python -m inspection`.
Use the release bundle or `inspection.experiment` for the complete UI and batch interface.
