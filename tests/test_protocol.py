from pathlib import Path

import numpy as np
import pytest

from inspection.protocol import calibrated_threshold, classification_metrics, split_normal


def test_disjoint_reproducible_nested_splits():
    paths = [Path(f"{i:03d}.png") for i in range(209)]
    train, calibration = split_normal(paths, 42, 30)
    assert len(train) == 179 and len(calibration) == 30
    assert not set(train) & set(calibration)
    assert set(train) | set(calibration) == set(paths)
    assert split_normal(list(reversed(paths)), 42, 30) == (train, calibration)
    assert set(train[:16]) < set(train[:64]) < set(train)


def test_finite_sample_calibration():
    assert calibrated_threshold(np.arange(30), 0.05) == 29
    assert calibrated_threshold(np.arange(99), 0.1) == 89
    with pytest.raises(ValueError, match="More calibration"):
        calibrated_threshold([1, 2, 3], 0.05)


@pytest.mark.parametrize("values", [[], [np.nan], [np.inf], [[1, 2]]])
def test_invalid_calibration_rejected(values):
    with pytest.raises(ValueError):
        calibrated_threshold(values)


def test_strict_threshold_and_errors():
    result = classification_metrics([0, 0, 1, 1], [1, 3, 2, 4], 2)
    assert (result["tn"], result["fp"], result["fn"], result["tp"]) == (1, 1, 1, 1)
    assert result["recall"] == 0.5
    assert result["false_positive_rate"] == 0.5
    assert result["image_auroc"] == 0.75
