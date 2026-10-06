# Inspection Lab

Normal-reference visual inspection, starting with a command-line workflow for MVTec AD.

The command-line workflow validates a category, fits a normal-reference model and exports
single-image scores. It includes a standardized pixel-template baseline and Anomalib
PatchCore with frozen ImageNet-pretrained ResNet-18 features.

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

Fitting reads only `train/good`. A seed-42 split holds out 30 normal images for calibration,
then takes 16 references from the remaining pool by default. The two sets cannot share
image content. Neither test images nor defect annotations enter fitting or calibration.
The model directory contains a compressed template and metadata with reference/checkpoint
hashes. Use a new empty output directory for each fit.

Prediction writes `prediction.json` and `anomaly_map.npy`. The image score is the 99th
percentile of smoothed, standardized RGB residuals. This baseline assumes a similar view
and alignment. Raw scores are not probabilities. Generated files are excluded from Git.

## Fit PatchCore on CPU

Install the CPU PyTorch wheels before the optional detector dependencies:

```bash
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[patchcore,dev]"
python -m inspection fit --method patchcore --root data/mvtec_ad/bottle --limit 16 --output artifacts/patchcore
python -m inspection predict --model artifacts/patchcore --image data/mvtec_ad/bottle/test/broken_large/000.png --output artifacts/patchcore-prediction
```

The first fit downloads pretrained features into `.cache/`; later prediction loads the
saved weights without downloading them again. Fitting extracts layers 2 and 3, selects a
1% coreset of reference patches and retains at least nine patches for nine-neighbor scoring.
There is no gradient-based fine-tuning. Input images are resized to 224 pixels and normalized
using ImageNet statistics. `--size`, `--limit` and `--seed` can change the fit configuration.

PatchCore exports the same JSON/NumPy interface as the baseline. Its score uses feature
distances, so the two methods' raw scores must not be compared as if they used the same
scale. Both fit only normal training images. A browser interface will follow in a later stage.

## Calibrated decisions

New fits also write `calibration.csv` with normal-image scores and hashes. The threshold is
the `ceil((n + 1) * (1 - alpha))`-th ordered score; an image is flagged only when its score is
strictly greater than the threshold. Defaults are `n = 30` and `alpha = 0.05`, so the cutoff
is the largest calibration score. Insufficient calibration data is rejected.

This split-conformal rule targets a normal false-alarm rate under exchangeability: future
normal images must come from the same distribution as calibration images. It does not
guarantee a 5% false-alarm rate on a particular small test set or under domain shift.

Use `--calibration-count` and `--alpha` to specify the protocol before inspecting test
results. `--calibration-count 0` retains the earlier raw-score workflow. Existing
uncalibrated checkpoints can still be loaded and return `decision: null`; calibrated
predictions include the fixed threshold and a Boolean decision. Loading verifies the
checkpoint and calibration evidence before inference.

## Evaluate the fixed protocol

```bash
python -m inspection evaluate --model artifacts/baseline --root data/mvtec_ad/bottle --output results/bottle-baseline
python -m inspection evaluate --model artifacts/patchcore --root data/mvtec_ad/bottle --output results/bottle-patchcore
python -m inspection verify-results --results results/bottle-patchcore
```

Evaluation checks the source-image hashes, scores every image in the category's test split
and exports AUROC, average precision, recall, precision, normal false alarms and confusion
counts. It never refits the model or chooses a new threshold. Each result directory contains
per-image predictions, calibration evidence, model metadata, dependency versions and a
failure report. Verification checks file hashes and recomputes metrics from the predictions
without requiring the dataset or a checkpoint. Model weights and dataset images stay out
of the result directory.

The first published comparison uses one product, one seed and 16 fitted references. It is
an initial experiment; repeated-seed and reference-budget studies will follow separately.
The [published comparison](results/README.md) includes all incorrect decisions and
recomputable evidence for both methods.

## Checks

```bash
python -m pytest -q
python -m ruff check .
```

PatchCore tests use random features and synthetic images to check fitting, map dimensions
and checkpoint round trips without downloading weights. They are skipped when the optional
detector dependencies are absent. Real pretrained fitting is checked separately on the
local bottle dataset; see [validation notes](docs/VALIDATION.md).

## Attribution

Project code is MIT licensed. MVTec AD has separate dataset terms; see
[THIRD_PARTY.md](THIRD_PARTY.md). Development is AI-assisted.
