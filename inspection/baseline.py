"""A standardized pixel-template baseline for approximately aligned images."""

import time
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter

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
        return self.predict_pixels(pixels(path, self.size))

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

    def pixels(self, path, condition="clean"):
        with Image.open(path) as image:
            image = image.convert("RGB")
            if condition == "dim":
                image = ImageEnhance.Brightness(image).enhance(0.7)
            elif condition == "blur":
                image = image.filter(ImageFilter.GaussianBlur(1.5))
            elif condition != "clean":
                raise ValueError("Unknown perturbation")
            return np.asarray(image.resize((self.size, self.size), Image.Resampling.BILINEAR), dtype=np.float32) / 255

    def predict_pixels(self, pixels):
        residual = np.sqrt(np.mean(((pixels - self.mean) / self.std) ** 2, axis=2))
        anomaly_map = gaussian_filter(residual, sigma=2)
        return float(np.quantile(anomaly_map, 0.99)), anomaly_map

    def evaluate(self, paths, condition="clean"):
        scores, maps, latencies = [], [], []
        self.predict_pixels(self.pixels(paths[0], condition))
        for path in paths:
            pixels = self.pixels(path, condition)
            start = time.perf_counter()
            score, anomaly_map = self.predict_pixels(pixels)
            latencies.append((time.perf_counter() - start) * 1000)
            scores.append(score)
            maps.append(anomaly_map)
        return np.asarray(scores), np.stack(maps), np.asarray(latencies)
