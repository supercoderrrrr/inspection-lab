"""A standardized pixel-template baseline for approximately aligned images."""

from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from inspection.images import pixels, validate_size


class PixelTemplate:
    def __init__(self, size=224, std_floor=0.05):
        validate_size(size)
        if not np.isfinite(std_floor) or std_floor <= 0:
            raise ValueError("Standard-deviation floor must be positive and finite.")
        self.size = size
        self.std_floor = std_floor

    def fit(self, paths):
        if len(paths) < 2:
            raise ValueError("At least two normal references are required.")
        images = np.stack([pixels(path, self.size) for path in paths])
        self.mean = images.mean(axis=0)
        self.std = np.maximum(images.std(axis=0, ddof=1), self.std_floor)
        return self

    def predict(self, path: Path):
        if not hasattr(self, "mean"):
            raise ValueError("Fit or load a template before prediction.")
        residual = np.sqrt(np.mean(((pixels(path, self.size) - self.mean) / self.std) ** 2, axis=2))
        anomaly_map = gaussian_filter(residual, sigma=2)
        return float(np.quantile(anomaly_map, 0.99)), anomaly_map

    def save(self, path: Path):
        np.savez_compressed(path, mean=self.mean, std=self.std, size=self.size, std_floor=self.std_floor)

    @classmethod
    def load(cls, path: Path):
        with np.load(path, allow_pickle=False) as payload:
            model = cls(int(payload["size"]), float(payload["std_floor"]))
            model.mean = payload["mean"]
            model.std = payload["std"]
        expected = (model.size, model.size, 3)
        if model.mean.shape != expected or model.std.shape != expected:
            raise ValueError("Template dimensions do not match the saved image size.")
        if not np.isfinite(model.mean).all() or not np.isfinite(model.std).all() or (model.std <= 0).any():
            raise ValueError("Template contains invalid reference statistics.")
        return model
