# Inspection Lab

Normal-reference visual inspection, starting with a command-line workflow for MVTec AD.

The command-line workflow validates a category, fits a normal-reference model and exports
single-image scores. The first detector is a standardized pixel-template baseline for
approximately aligned images.

## Install

Use Python 3.11. From this directory:

```bash
python -m venv .venv
```

Activate with `.venv\Scripts\Activate.ps1` in Windows PowerShell, or
`source .venv/bin/activate` on Linux, then install:

```bash
python -m pip install -e ".[dev]"
```

## Check a dataset

Download the bottle category from the [official MVTec AD page](https://www.mvtec.com/research-teaching/datasets/mvtec-ad/downloads).
Dataset files are excluded from Git. Extract them into this structure:

```text
data/mvtec_ad/bottle/
  train/good/*.png
  test/good/*.png
  test/<defect>/*.png
  ground_truth/<defect>/*_mask.png
```

```bash
python -m inspection check-data --root data/mvtec_ad/bottle
python -m pytest -q
```

`--root` can also point to an existing dataset outside this repository. The checker returns
JSON counts and image dimensions; invalid layouts return an error and a nonzero exit code.
It does not download data or modify the dataset.

## Fit the baseline

```bash
python -m inspection fit --method baseline --root data/mvtec_ad/bottle --limit 16 --output artifacts/baseline
python -m inspection predict --model artifacts/baseline --image data/mvtec_ad/bottle/test/broken_large/000.png --output artifacts/baseline-prediction
```

Fitting reads only `train/good`. References are sampled reproducibly using seed 42 by default.
The model directory contains a compressed template and metadata with reference/checkpoint
hashes. Use a new empty output directory for each fit.

Prediction writes `prediction.json` and `anomaly_map.npy`. The image score is the 99th
percentile of smoothed, standardized RGB residuals. This baseline assumes a similar view
and alignment. Raw scores are not probabilities; this stage has no calibrated pass/fail
threshold or benchmark accuracy claim. Generated files are excluded from Git.

## Attribution

Project code is MIT licensed. MVTec AD has separate dataset terms; see
[THIRD_PARTY.md](THIRD_PARTY.md). Development is AI-assisted.
