"""Shared image handling for reference fitting and inference."""

import hashlib
import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 32_000_000


def read_image(source) -> Image.Image:
    try:
        if isinstance(source, Image.Image):
            image = source
            if image.width * image.height > MAX_IMAGE_PIXELS:
                raise ValueError("Images must contain at most 32 million pixels.")
            return ImageOps.exif_transpose(image).convert("RGB")
        with Image.open(source) as image:
            if image.width * image.height > MAX_IMAGE_PIXELS:
                raise ValueError("Images must contain at most 32 million pixels.")
            image.load()
            return ImageOps.exif_transpose(image).convert("RGB")
    except (OSError, Image.DecompressionBombError) as error:
        raise ValueError("Cannot read the supplied image.") from error


def decode_upload(content: bytes) -> Image.Image:
    if not content or len(content) > MAX_UPLOAD_BYTES:
        raise ValueError("Use a nonempty PNG or JPEG no larger than 10 MB.")
    try:
        with Image.open(io.BytesIO(content)) as image:
            if image.format not in {"PNG", "JPEG"}:
                raise ValueError("Only PNG and JPEG content is supported.")
            return read_image(image)
    except (OSError, Image.DecompressionBombError) as error:
        raise ValueError("The uploaded file is not a readable PNG or JPEG.") from error


def image_digest(image: Image.Image) -> str:
    image = read_image(image)
    header = f"RGB:{image.width}:{image.height}:".encode()
    return hashlib.sha256(header + image.tobytes()).hexdigest()


def pixels(source, size: int) -> np.ndarray:
    image = read_image(source).resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0


def validate_size(size: int):
    if not 64 <= size <= 512:
        raise ValueError("Image size must be between 64 and 512 pixels.")
