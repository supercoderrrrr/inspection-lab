"""Bounded image decoding and ground-truth visualization."""

import hashlib
import io

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

MAX_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 20_000_000


def decode_image(content: bytes):
    if not content or len(content) > MAX_BYTES:
        raise ValueError("Use a nonempty PNG or JPEG smaller than 10 MB.")
    try:
        with Image.open(io.BytesIO(content)) as image:
            if image.format not in {"PNG", "JPEG"}:
                raise ValueError("Only PNG and JPEG images are supported.")
            if image.width * image.height > MAX_PIXELS:
                raise ValueError("Use an image below 20 megapixels.")
            image.load()
            return ImageOps.exif_transpose(image).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("The uploaded file is not a readable PNG or JPEG.") from exc


def image_identity(content: bytes):
    return hashlib.sha256(content).hexdigest()


def annotation_overlay(image, mask):
    image = ImageOps.exif_transpose(image).convert("RGB")
    mask = mask.convert("L").resize(image.size, Image.Resampling.NEAREST)
    pixels = np.asarray(image).copy()
    selected = np.asarray(mask) > 0
    pixels[selected] = (pixels[selected] * 0.4 + np.array([34, 211, 238]) * 0.6).astype(np.uint8)
    return Image.fromarray(pixels)


MAX_UPLOAD_BYTES = MAX_BYTES
MAX_IMAGE_PIXELS = MAX_PIXELS

def read_image(source) -> Image.Image:
    try:
        if isinstance(source, Image.Image):
            image = source
            if image.width * image.height > MAX_IMAGE_PIXELS:
                raise ValueError("Images must contain at most 20 million pixels.")
            return ImageOps.exif_transpose(image).convert("RGB")
        with Image.open(source) as image:
            if image.width * image.height > MAX_IMAGE_PIXELS:
                raise ValueError("Images must contain at most 20 million pixels.")
            image.load()
            return ImageOps.exif_transpose(image).convert("RGB")
    except (OSError, Image.DecompressionBombError) as error:
        raise ValueError("Cannot read the supplied image.") from error

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

def decode_upload(content: bytes):
    return decode_image(content)
