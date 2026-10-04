"""Shared image handling for reference fitting and inference."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def pixels(path: Path, size: int) -> np.ndarray:
    try:
        with Image.open(path) as image:
            if image.width * image.height > 32_000_000:
                raise ValueError("Images must contain at most 32 million pixels.")
            image = ImageOps.exif_transpose(image).convert("RGB")
            image = image.resize((size, size), Image.Resampling.BILINEAR)
            return np.asarray(image, dtype=np.float32) / 255.0
    except (OSError, Image.DecompressionBombError) as error:
        raise ValueError(f"Cannot read image: {Path(path).name}") from error


def validate_size(size: int):
    if not 64 <= size <= 512:
        raise ValueError("Image size must be between 64 and 512 pixels.")
