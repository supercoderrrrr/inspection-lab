# Model card — Inspection Lab

## Intended use

An educational/research workbench for approximately aligned bottle and metal-nut images from
the original MVTec AD dataset. The interface supports inspection, explanation by anomaly
intensity, and reproducible evaluation. It has not been validated for manufacturing acceptance
decisions, unrelated products, or arbitrary camera views.

## Model and provenance

Anomalib 2.2.0 implements PatchCore. ResNet-18 ImageNet-pretrained layer2/layer3 features
are extracted from resized 224x224 RGB images with ImageNet normalization. A 1% k-center
coreset stores reference patches. The feature network is frozen. The application does
not claim algorithmic novelty. See THIRD_PARTY.md for papers, licenses, and AI assistance.

## Data and decision rule

MVTec AD bottle has 209 normal training and 83 test images. Each seed holds out 30 normal
training images for calibration. Nested bottle reference budgets are 16, 64, and 179. Metal nut has 220 normal training
and 115 test images (22 normal, 93 anomalous), with budgets 16, 64, and 190. Each category
has an independent model and calibration split. The image
threshold is the ceil((n+1)*(1-alpha))-th sorted calibration score, with alpha 0.05; a score
strictly above it is flagged. Pixel maps show relative abnormality, not a validated binary mask.
The fixed display scale is derived from calibration maps. Optional per-image maximum scaling
helps inspect spatial variation but prevents color comparison across images; neither scale
is a pixel decision threshold. Uploaded images and annotations
do not update or train the model.

## Validation

The repeated-seed study prespecifies 42, 123, and 2026. A standardized pixel-template model
provides a simple baseline on the exact same splits. Brightness and blur experiments retain
the clean threshold. Means and sample standard deviations summarize split sensitivity.
They are not independent-test confidence intervals. Wilson intervals shown in the interface
apply only to the selected run's binomial false-alarm count, subject to sampling assumptions.

## Known limitations

- Two categories and only 20/22 normal test examples limit external validity and false-alarm precision.
- A high AUROC does not guarantee a useful calibrated operating point.
- Seed comparisons reuse one test set and do not establish statistical superiority.
- Training budget also changes memory-bank size; these effects are not isolated.
- Resizing can hide small defects; synthetic perturbations do not represent all camera shifts.
- Compact ResNet-18 results are not the original paper's full benchmark results.
- GPU/CPU timing uses different hardware paths. Runtime load, warm-up, and power states matter.

## Operating assumptions and maintenance

Run the app on localhost. Decode only bounded PNG/JPEG uploads. Inference outputs remain
in the browser session until explicitly downloaded; no uploaded image is added to the dataset.
Load project-created checkpoints only. Checkpoint digests detect accidental modification,
not malicious replacement of both a file and its manifest. Run `python -m inspection.doctor
--verify-data` after moving or updating artifacts. Treat a new category, camera, or preprocessing
pipeline as a new validation problem, with new normal calibration data and an untouched test set.

## Responsibilities

Human review is required for interpreting uncertain outputs and the suitability of this model
for a new application. Project claims must distinguish imported algorithms, AI-assisted software,
and measured experiments. This is a research application, not an industrial certification.

## Version 0.4 comparisons and custom data

PaDiM is imported from Anomalib with ResNet-18 layer1/2/3 and 100 randomly sampled channels.
It fits regularized patch-location Gaussian distributions; its raw distance scale differs
from PatchCore. Calibration is independently performed on the same held-out normal split.
The fixed-capacity PatchCore variant uses exactly 125 patches at every budget. These
exploratory follow-up experiments preserve the earlier studies and do not select new defaults.
See results/controlled_v4/REPORT.md for every result, not only the best score.

Custom datasets use the documented annotated layout and explicit --data-root. Exact file
hash duplicates are rejected; near-duplicate leakage still requires collection-level review.
Model archives contain prespecified PatchCore snapshots only. Batch timing includes preprocessing
and transfer, with no warm-up exclusion for its first image; it is not a benchmark result.


## Inspection display

The UI defaults to per-image maximum scaling and response-weighted opacity. With
s = clip(raw_map / display_max, 0, 1), blending uses alpha = opacity × s³. This reduces
low-response tint without using dataset annotations or creating a binary defect mask.
Uniform tint remains selectable, and the raw-map plot/export never applies alpha weighting.
The colors and visibility of a region depend on display settings; inspect the raw numeric
map when judging localization. Model parameters, scores and thresholds are unchanged.
