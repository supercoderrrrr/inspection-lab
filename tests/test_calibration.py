import csv
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from inspection.protocol import anomaly_decision, calibrated_threshold, split_normal
from inspection.workflow import calibration_records, fit, load_detector


def test_holdout_is_disjoint_repeatable_and_leaves_a_nested_pool():
    paths = [Path(f"{i:03d}.png") for i in range(40)]
    pool, calibration = split_normal(paths, 42, 30)
    assert len(calibration) == 30
    assert len(pool) == 10
    assert not set(pool) & set(calibration)
    assert (pool, calibration) == split_normal(list(reversed(paths)), 42, 30)
    assert set(pool) | set(calibration) == set(paths)


def test_conformal_order_statistic_and_strict_decision():
    scores = np.arange(30, dtype=float)
    threshold = calibrated_threshold(scores, 0.05)
    assert threshold == 29
    assert anomaly_decision(29, threshold) is False
    assert anomaly_decision(30, threshold) is True
    with pytest.raises(ValueError, match="More calibration"):
        calibrated_threshold(scores[:3], 0.05)
    with pytest.raises(ValueError, match="finite"):
        calibrated_threshold([float("nan")], 0.5)


def test_saved_calibration_is_recomputable_and_protected(tmp_path):
    root = tmp_path / "bottle"
    good = root / "train/good"
    good.mkdir(parents=True)
    for i in range(8):
        Image.new("RGB", (64, 64), (70 + i, 80 + i, 90 + i)).save(good / f"{i}.png")
    output = tmp_path / "model"
    metadata = fit(root, output, limit=4, size=64, calibration_count=3, alpha=0.25)
    records = calibration_records(output, metadata)
    assert len(records) == 3
    assert not {r["path"] for r in records} & {r["path"] for r in metadata["references"]}
    expected = calibrated_threshold([float(r["score"]) for r in records], 0.25)
    assert metadata["calibration"]["threshold"] == expected
    load_detector(output)
    with (output / "calibration.csv").open("a", newline="", encoding="utf-8") as stream:
        csv.writer(stream).writerow(["train/good/extra.png", "changed", 999])
    with pytest.raises(ValueError, match="Calibration integrity"):
        load_detector(output)


def test_rejects_duplicate_content_across_normal_partitions(tmp_path):
    root = tmp_path / "bottle"
    good = root / "train/good"
    good.mkdir(parents=True)
    for i in range(6):
        Image.new("RGB", (64, 64), (80, 80, 80)).save(good / f"{i}.png")
    with pytest.raises(ValueError, match="distinct content"):
        fit(root, tmp_path / "model", limit=3, calibration_count=3, alpha=0.25)
