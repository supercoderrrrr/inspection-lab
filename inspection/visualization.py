"""Display helpers; visual normalization never changes model outputs."""

import io

import numpy as np
from PIL import Image

from inspection.images import read_image


def response_overlay(image, anomaly_map, opacity=0.6):
    from matplotlib import colormaps

    values = np.asarray(anomaly_map)
    if values.ndim != 2 or not np.isfinite(values).all() or not 0 <= opacity <= 1:
        raise ValueError("Overlay needs a finite 2D response map and opacity between zero and one.")
    source = read_image(image).resize((values.shape[1], values.shape[0]), Image.Resampling.BILINEAR)
    maximum = max(float(values.max()), 1e-8)
    scaled = np.clip(values / maximum, 0, 1)
    colors = colormaps["inferno"](scaled)[..., :3] * 255
    alpha = (opacity * scaled ** 3)[..., None]
    blended = np.asarray(source, dtype=float) * (1 - alpha) + colors * alpha
    return Image.fromarray(np.rint(blended).clip(0, 255).astype(np.uint8))


def annotation_overlay(image, mask):
    source = read_image(image)
    selected = np.asarray(mask.convert("L").resize(source.size, Image.Resampling.NEAREST)) > 0
    values = np.asarray(source).copy()
    values[selected] = (values[selected] * 0.4 + np.array([34, 211, 238]) * 0.6).astype(np.uint8)
    return Image.fromarray(values)


def png_bytes(image):
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


def map_bytes(anomaly_map):
    stream = io.BytesIO()
    np.save(stream, anomaly_map, allow_pickle=False)
    return stream.getvalue()
