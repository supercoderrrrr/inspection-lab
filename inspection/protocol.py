"""Normal-only split calibration, independent of detector implementation."""

import hashlib
import math
from pathlib import Path
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

import numpy as np


def split_normal(paths, seed, calibration_count=30):
    ordered = sorted(paths)
    if len(set(ordered)) != len(ordered):
        raise ValueError("Normal image paths must be unique.")
    if calibration_count < 0 or len(ordered) < calibration_count + 2:
        raise ValueError("Leave at least two normal references after reserving calibration images.")
    indices = np.random.default_rng(seed).permutation(len(ordered))
    calibration = [ordered[i] for i in indices[:calibration_count]]
    training = [ordered[i] for i in indices[calibration_count:]]
    return training, calibration


def calibration_rank(count, alpha=0.05):
    if count < 1 or not 0 < alpha < 1:
        raise ValueError("Calibration needs a positive count and alpha between zero and one.")
    rank = math.ceil((count + 1) * (1 - alpha))
    if rank > count:
        raise ValueError("More calibration images are required for this false-alarm target.")
    return rank


def calibrated_threshold(scores, alpha=0.05):
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Calibration requires nonempty finite image scores.")
    return float(np.sort(values)[calibration_rank(len(values), alpha) - 1])


def anomaly_decision(score, threshold):
    if not np.isfinite(score) or not np.isfinite(threshold):
        raise ValueError("Scores and thresholds must be finite.")
    return bool(score > threshold)


def classification_metrics(labels, scores, threshold):
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    if (labels.ndim != 1 or scores.ndim != 1 or len(labels) != len(scores)
            or not len(labels) or not np.isin(labels, [0, 1]).all()
            or not np.isfinite(scores).all() or not np.isfinite(threshold)):
        raise ValueError("Metrics require matching nonempty binary labels and finite scores/threshold.")
    labels = labels.astype(int)
    predicted = scores > threshold
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    both_classes = len(np.unique(labels)) == 2
    return {
        "image_auroc": float(roc_auc_score(labels, scores)) if both_classes else None,
        "average_precision": float(average_precision_score(labels, scores)) if both_classes else None,
        "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else None,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }

def wilson_interval(successes: int, total: int, z: float = 1.959964):
    """A binomial interval for one test split, not an interval across repeated seeds."""
    if total < 1 or not 0 <= successes <= total:
        raise ValueError("Counts must satisfy 0 <= successes <= total and total > 0.")
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)

def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
