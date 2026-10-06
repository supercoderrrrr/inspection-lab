"""Normal-only split calibration, independent of detector implementation."""

import math

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
