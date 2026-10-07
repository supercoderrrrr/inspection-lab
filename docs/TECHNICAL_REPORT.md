# Inspection Lab: technical report

## 1. Research question
How do normal-reference budget and controlled input perturbations affect a compact
PatchCore detector, compared with an inexpensive aligned-pixel baseline?

## 2. Dataset and separation
Original MVTec AD bottle: 209 normal training images; 83 test images (20 normal, 63 defective).
Metal nut: 220 normal training images; 115 test images (22 normal, 93 defective).
Each category has independent weights and calibration. Test labels and masks are used only
for evaluation and the annotated UI comparison. Dataset source and license: THIRD_PARTY.md.

Per seed, sort normal paths, permute with the recorded seed and hold out 30 for calibration.
Use nested budgets 16, 64 and 179 for bottle; 16, 64 and 190 for metal nut. Seeds are
42, 123 and 2026. A study plan is saved before execution. The default model is the first
planned seed at the largest budget, selected without test-performance optimization.

## 3. Models and preprocessing
PatchCore is imported from Anomalib 2.2.0. Inputs are RGB, resized to 224 × 224 with
ImageNet normalization. Frozen ResNet-18 layer2/layer3 patch features form a 1% k-center
coreset; nearest-neighbor scoring uses nine neighbors. No gradient fine-tuning is performed.
This compact setup differs from the original paper's headline benchmark configuration.

The pixel template fits a mean and sample standard deviation for each RGB pixel, floors
standard deviation at .05 on [0,1] intensities, and smooths RMS standardized residuals
with Gaussian sigma 2. Its image score is the 99th percentile. Both models share splits.

## 4. Calibration and measurement
For n=30 normal scores and alpha=.05, use order statistic ceil((n+1)(1-alpha))=30.
A score strictly above that threshold is anomalous. The exchangeability assumption is
essential; a nominal 5% level does not promise a 5% observed rate on a small shifted test set.
No threshold is fitted to test labels. Raw scores are distances, not probabilities.

Image AUROC measures ranking. Recall and false alarms evaluate the fixed operating point.
Pixel AUROC uses nearest-neighbor resized masks. Model latency excludes decode/preprocess;
the UI separately reports end-to-end inspection timing. CPU execution uses four PyTorch
threads. Timing was collected under ordinary machine load, not isolated performance testing.

## 5. Clean results
Largest prespecified budgets; mean ± sample standard deviation across three seeds.

| Category | Method | Condition | References | Image AUROC | Recall | False alarms |
|---|---|---|---:|---:|---:|---:|
| bottle | PatchCore | clean | 179 | 1.0000 ± 0.0000 | 100.0% ± 0.0% | 13.3% ± 10.4% |
| bottle | Pixel template | clean | 179 | 0.9011 ± 0.0032 | 53.4% ± 9.2% | 0.0% ± 0.0% |
| metal_nut | PatchCore | clean | 190 | 0.9891 ± 0.0033 | 92.5% ± 5.6% | 4.5% ± 4.5% |
| metal_nut | Pixel template | clean | 190 | 0.4904 ± 0.0030 | 25.1% ± 2.2% | 1.5% ± 2.6% |

The aggregate CSVs retain all budgets, pixel AUROC and timing. Review per-image predictions
and failure counts before judging operational suitability. A simple baseline can have few
false alarms while missing many defects; neither a low false-alarm rate nor AUROC alone
is a sufficient summary of inspection quality.

## 6. Controlled perturbations
Brightness is multiplied by .7; Gaussian blur radius is 1.5. Only the largest budget is
tested under perturbations, and the clean calibration threshold remains fixed.

| Category | Method | Condition | References | Image AUROC | Recall | False alarms |
|---|---|---|---:|---:|---:|---:|
| bottle | PatchCore | blur | 179 | 1.0000 ± 0.0000 | 100.0% ± 0.0% | 13.3% ± 12.6% |
| bottle | PatchCore | dim | 179 | 1.0000 ± 0.0000 | 100.0% ± 0.0% | 41.7% ± 52.0% |
| bottle | Pixel template | blur | 179 | 0.9029 ± 0.0012 | 53.4% ± 9.2% | 0.0% ± 0.0% |
| bottle | Pixel template | dim | 179 | 0.6960 ± 0.0021 | 100.0% ± 0.0% | 100.0% ± 0.0% |
| metal_nut | PatchCore | blur | 190 | 0.9945 ± 0.0037 | 91.8% ± 5.0% | 3.0% ± 5.2% |
| metal_nut | PatchCore | dim | 190 | 0.9936 ± 0.0013 | 90.7% ± 5.0% | 1.5% ± 2.6% |
| metal_nut | Pixel template | blur | 190 | 0.4932 ± 0.0047 | 24.4% ± 3.5% | 1.5% ± 2.6% |
| metal_nut | Pixel template | dim | 190 | 0.6479 ± 0.0017 | 9.7% ± 12.4% | 1.5% ± 2.6% |

These reuse the same images and labels. They are sensitivity checks, not new independent
test sets and not evidence of robustness to every camera, lighting or production change.

## 7. UI and interpretation
The overlay blends a colored anomaly map with the image at adjustable opacity. Fixed scaling
uses the 99.9th percentile of normal calibration map values; values above that range saturate.
Per-image maximum scaling helps reveal spatial variation but makes colors incomparable
across images. A numeric color bar, saturation percentage and raw NumPy export expose the
underlying map. Display changes never alter scores, thresholds or predictions. Cyan masks
come from ground-truth annotations, not the detector, and are unavailable for uploads.

## 8. Reproduction and audit
See README.md for Python 3.11 installation, pinned dependencies and study commands.
Run `python -m inspection.doctor --verify-data` to verify artifacts/data and independently
recalculate image metrics and thresholds from stored CSVs. Run `python -m pytest -q` for
protocol and serialization checks. Browser checks exercise inference, upload errors,
result persistence, visualization, downloads and category selection.

Evidence ZIPs include source, reports, split hashes and raw predictions; they exclude dataset
images, weights, caches and environments. Reproduction downloads upstream data and pretrained
weights. Hashes detect accidental changes, not malicious replacement of both file and manifest.
Historical run source hashes describe the actual run, which may precede current UI changes.

## 9. Limitations and next experiments
Three seeds reuse each category's test set. Standard deviations are not population confidence
intervals. One normal error changes false alarms by 5 percentage points for bottle and 4.55
for metal nut. Reference budget changes both image diversity and bank capacity. ImageNet
pretraining, alignment and resized resolution affect results. No real factory trial, calibrated
pixel segmentation, production camera integration or zero-shot cross-category transfer is claimed.

Useful next experiments are new acquisition conditions with independent calibration/test data,
fixed-bank-size budget ablations, and operating-cost analysis on a larger normal test population.

## 10. Attribution and contribution
Upstream detector: Anomalib/PatchCore. Data: MVTec AD, CC BY-NC-SA 4.0. See THIRD_PARTY.md.
Project-specific work covers experimental orchestration, calibration, baseline comparison,
artifact audits, reporting and the UI. Development is AI-assisted; the applicant should describe
their own choices, review and interpretation accurately rather than claim sole algorithm authorship.


## 11. Version 0.4: controlled method and capacity comparisons

The complete follow-up is in [the controlled report](../results/controlled_v4/REPORT.md).
It adds Anomalib PaDiM (100 feature channels, ResNet-18 layers 1/2/3) and PatchCore with an
exact 125-patch coreset at every budget. Both use the same three seed-specific split manifests
as the original studies. These are exploratory follow-ups, planned after inspecting original
results and before running the new variants. No threshold was tuned against test defects.

Largest-budget means (full standard deviations are in the controlled report):

| Category | Method | AUROC | Recall | False alarms |
|---|---|---:|---:|---:|
| bottle | PaDiM | 0.9984 | 98.4% | 5.0% |
| bottle | PatchCore | 1.0000 | 100.0% | 13.3% |
| bottle | PatchCore fixed 125 | 0.9860 | 96.8% | 8.3% |
| bottle | Pixel template | 0.9011 | 53.4% | 0.0% |
| metal_nut | PaDiM | 0.9573 | 76.0% | 7.6% |
| metal_nut | PatchCore | 0.9891 | 92.5% | 4.5% |
| metal_nut | PatchCore fixed 125 | 0.8749 | 62.7% | 9.1% |
| metal_nut | Pixel template | 0.4904 | 25.1% | 1.5% |

Under this configuration, the 1% PatchCore variant has higher metal-nut AUROC and recall
than PaDiM and the fixed-125 variant. The fixed-capacity comparison shows that attributing
all original budget gains to more normal images would be misleading: final bank capacity
also matters. It does not establish statistical superiority across new datasets. PaDiM's
layers and representation differ; its comparison does not isolate the scoring metric alone.
A fixed bank still changes candidate coverage and coreset selection as references increase.

The source release now includes image-level evidence that can be independently recalculated,
and a separate model release supports inference without repeating these studies. Validation
results and the hosted CI link are recorded in docs/VALIDATION.md.


## Inspection display

The UI defaults to per-image maximum scaling and response-weighted opacity. With
s = clip(raw_map / display_max, 0, 1), blending uses alpha = opacity × s³. This reduces
low-response tint without using dataset annotations or creating a binary defect mask.
Uniform tint remains selectable, and the raw-map plot/export never applies alpha weighting.
The colors and visibility of a region depend on display settings; inspect the raw numeric
map when judging localization. Model parameters, scores and thresholds are unchanged.
