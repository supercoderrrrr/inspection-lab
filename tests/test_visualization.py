import io

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("matplotlib")

from inspection.visualization import annotation_overlay, map_bytes, response_overlay


def test_overlay_preserves_low_response_detail_and_raw_map():
    image = Image.new("RGB", (64, 64), (80, 80, 80))
    raw = np.full((64, 64), 0.2, dtype=np.float32)
    raw[:, 32:] = 1
    before = raw.copy()
    overlay = np.asarray(response_overlay(image, raw), dtype=float)
    assert np.mean(abs(overlay[:, :32] - 80)) < 2
    assert np.mean(abs(overlay[:, 32:] - 80)) > 10
    np.testing.assert_array_equal(raw, before)
    np.testing.assert_array_equal(np.load(io.BytesIO(map_bytes(raw)), allow_pickle=False), raw)


def test_annotation_changes_only_marked_pixels():
    image = Image.new("RGB", (20, 20), (80, 80, 80))
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:10, 5:10] = 255
    result = np.asarray(annotation_overlay(image, Image.fromarray(mask)))
    np.testing.assert_array_equal(result[mask == 0], np.asarray(image)[mask == 0])
    assert (result[mask > 0] != np.asarray(image)[mask > 0]).any()
