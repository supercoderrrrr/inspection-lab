# Inspection Lab

Normal-reference visual inspection, starting with a command-line workflow for MVTec AD.

The first module validates a category before any model fitting: image readability, expected
directories, defect annotations and separation of training and test image content.

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

## Attribution

Project code is MIT licensed. MVTec AD has separate dataset terms; see
[THIRD_PARTY.md](THIRD_PARTY.md). Development is AI-assisted.
