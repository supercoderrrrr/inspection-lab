"""Generate an English report exclusively from recorded experiment artifacts."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def build_report(run: Path):
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    splits = json.loads((run / "splits.json").read_text(encoding="utf-8"))
    n_train = len(splits["training_pool"]) + len(splits["calibration"])
    n_test = len(splits["test"])
    n_good = sum("/good/" in item["path"] for item in splits["test"])
    results = pd.read_csv(run / "summary.csv")
    clean = results[results.condition == "clean"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    axes[0].plot(clean.budget, clean.image_auroc, "o-", label="Image AUROC")
    axes[0].plot(clean.budget, clean.recall, "s-", label="Calibrated recall")
    axes[0].set(xlabel="Normal training images", ylabel="Score", ylim=(0, 1.05))
    axes[0].legend()
    axes[1].plot(clean.budget, clean.latency_median_ms, "o-", color="#0891b2")
    axes[1].set(xlabel="Normal training images", ylabel="Median inference latency (ms)")
    fig.savefig(run / "budget_comparison.png", dpi=180)
    plt.close(fig)
    columns = ["budget", "condition", "image_auroc", "pixel_auroc", "recall", "false_positive_rate", "latency_median_ms"]
    header = "| " + " | ".join(columns) + " |\n|" + "---|" * len(columns)
    lines = [header]
    for _, row in results.iterrows():
        lines.append("| " + " | ".join(f"{row[c]:.4f}" if isinstance(row[c], float) else str(row[c]) for c in columns) + " |")
    dataset_label = "Custom annotated dataset" if config.get("data_root") else "Original MVTec AD"
    method = config.get("method", "patchcore")
    representation = ("layer1/layer2/layer3 features, 100 randomly selected dimensions and regularized Gaussian patch distributions"
                      if method == "padim" else f"layer2/layer3 features and {'a fixed ' + str(config['bank_patches']) + '-patch coreset' if config.get('bank_patches') else str(config['coreset_ratio'] * 100) + '% coreset'}")
    budget_note = ("Memory-bank capacity is fixed across budgets." if config.get("bank_patches") else
                   "PaDiM uses the same feature dimensionality across budgets." if method == "padim" else
                   "Budget comparisons change both reference diversity and memory-bank size.")
    text = f"""# Inspection Lab — Experiment Report

## Question
How does the number of normal reference images affect industrial anomaly detection,
and how stable is the largest prespecified model under controlled image perturbations?

## Method
This application uses Anomalib's `{method}` implementation.
The backbone is `{config['backbone']}` with ImageNet pretrained weights,
{representation}, {config['image_size']} × {config['image_size']} RGB inputs and ImageNet normalization. This compact configuration is not a reproduction
of the original paper's headline benchmark. No gradient-based fine-tuning is performed.

## Evaluation protocol
- Dataset: {dataset_label}, {config['category']} category; {n_train} normal training images and {n_test} test images.
- Hold out {config['calibration_count']} normal training images for calibration before selecting nested training budgets.
- Training budgets: {config['budgets']}; seed: {config['seed']}.
- Select the upper split-conformal order statistic at alpha={config['alpha']} from normal calibration scores.
- Predict anomaly only when the raw image score strictly exceeds this threshold.
- Test labels and masks do not participate in fitting, calibration, or default-model selection.
- The default model uses the largest prespecified budget, regardless of test scores.
- Thresholds stay fixed for brightness ×0.7 and Gaussian blur radius 1.5 stress tests, when enabled.
- Pixel AUROC uses nearest-neighbor resized ground-truth masks at the input resolution.
- Latency uses a warmed-up model, batch size one, and device synchronization; decoding and preprocessing are excluded.

## Recorded results
{chr(10).join(lines)}

![Budget comparison](budget_comparison.png)

## Interpretation and limitations
AUROC measures ranking across thresholds. Recall and false-positive rate measure the
single threshold calibrated without anomalies; a high AUROC does not guarantee high recall
at this operating point. The nominal false-alarm level relies on exchangeable normal data,
not a guarantee on this small test set or on shifted deployment images.

These are one-category, one-seed observations. {budget_note} Synthetic brightness/blur tests do not establish real-factory
robustness. The dataset has {n_good} normal and {n_test - n_good} anomalous test images, so each normal error changes
the observed false-positive rate by {100 / n_good:.2f} percentage points. Scores are distances, not probabilities.
Heatmap colors use a calibration-derived fixed scale, not a validated pixel decision threshold.

## Reproducibility
See `config.json`, `environment.json`, `splits.json`, `summary.csv`, and the per-budget
calibration/prediction CSV files. Splits include image SHA-256 hashes. Checkpoints contain
tensor state and configuration; load only artifacts from a trusted source.

## Attribution and contribution
Upstream: Anomalib (Apache-2.0), PatchCore (Roth et al., CVPR 2022), and MVTec AD
(Bergmann et al., CVPR 2019; dataset CC BY-NC-SA 4.0).
Project-specific work: split/calibration protocol, nested budget experiments, stress tests,
artifact provenance, English inspection UI, and generated reporting. Development is AI-assisted.
No claim is made that the upstream detector was authored by the project owner.
"""
    (run / "REPORT.md").write_text(text, encoding="utf-8")
