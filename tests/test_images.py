import hashlib
import io

import numpy as np
import pytest
from PIL import Image

from inspection.images import MAX_UPLOAD_BYTES, decode_upload, pixels
from inspection.workflow import fit, inspect_image, load_detector


def image_bytes(image, format="PNG"):
    stream = io.BytesIO()
    image.save(stream, format=format)
    return stream.getvalue()


def test_rejects_empty_corrupt_oversized_and_disguised_uploads():
    for content in [b"", b"not an image", b"x" * (MAX_UPLOAD_BYTES + 1),
                    image_bytes(Image.new("RGB", (10, 10)), "GIF")]:
        with pytest.raises(ValueError):
            decode_upload(content)


def test_png_upload_and_disk_image_share_preprocessing(tmp_path):
    content = image_bytes(Image.new("RGBA", (100, 80), (80, 120, 160, 255)))
    path = tmp_path / "sample.png"
    path.write_bytes(content)
    decoded = decode_upload(content)
    assert decoded.mode == "RGB"
    np.testing.assert_array_equal(pixels(path, 64), pixels(decoded, 64))


def test_jpeg_orientation_is_applied_before_resize():
    image = Image.new("RGB", (20, 10), (80, 120, 160))
    exif = Image.Exif()
    exif[274] = 6
    stream = io.BytesIO()
    image.save(stream, format="JPEG", exif=exif)
    assert decode_upload(stream.getvalue()).size == (10, 20)


def test_cli_and_uploaded_image_use_identical_inference(tmp_path):
    root = tmp_path / "bottle"
    good = root / "train/good"
    good.mkdir(parents=True)
    for i in range(4):
        Image.new("RGB", (64, 64), (80 + i, 80 + i, 80 + i)).save(good / f"{i}.png")
    model_dir = tmp_path / "model"
    fit(root, model_dir, limit=4, size=64, calibration_count=0)
    model, metadata = load_detector(model_dir)
    path = good / "0.png"
    content = path.read_bytes()
    expected, expected_map = inspect_image(model, metadata, path)
    actual, actual_map = inspect_image(model, metadata, decode_upload(content),
                                       image_name=path.name, image_sha256=hashlib.sha256(content).hexdigest())
    assert actual == expected
    np.testing.assert_array_equal(actual_map, expected_map)
