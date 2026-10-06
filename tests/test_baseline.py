import json

import numpy as np
import pytest
from PIL import Image

from inspection.baseline import PixelTemplate
from inspection.workflow import fit, predict, select_references


@pytest.fixture
def references(tmp_path):
    root = tmp_path / "bottle"
    good = root / "train/good"
    good.mkdir(parents=True)
    for i in range(4):
        Image.new("RGB", (64, 64), (80 + i, 80 + i, 80 + i)).save(good / f"{i:03d}.png")
    return root


def test_sampling_is_repeatable_and_bounded(references):
    assert select_references(references, 2, 42) == select_references(references, 2, 42)
    with pytest.raises(ValueError, match="Reference count"):
        select_references(references, 5, 42)


def test_defect_response_and_round_trip(references, tmp_path):
    normal = tmp_path / "normal.png"
    defect = tmp_path / "defect.png"
    Image.new("RGB", (64, 64), (82, 82, 82)).save(normal)
    pixels = np.full((64, 64, 3), 82, dtype=np.uint8)
    pixels[16:40, 16:40] = 240
    Image.fromarray(pixels).save(defect)
    model = PixelTemplate(size=64).fit(select_references(references, 4, 42))
    normal_score, _ = model.predict(normal)
    expected_score, expected_map = model.predict(defect)
    assert expected_score > normal_score + 5
    assert expected_map[24, 24] > expected_map[0, 0]
    checkpoint = tmp_path / "template.npz"
    model.save(checkpoint)
    restored_score, restored_map = PixelTemplate.load(checkpoint).predict(defect)
    assert restored_score == expected_score
    np.testing.assert_array_equal(restored_map, expected_map)


def test_fit_predict_exports_and_protects_checkpoint(references, tmp_path):
    model_dir = tmp_path / "model"
    metadata = fit(references, model_dir, limit=4, size=64, calibration_count=0)
    assert all(p["path"].startswith("train/good/") for p in metadata["references"])
    assert len(metadata["references"]) == 4
    image = references / "train/good/000.png"
    output = tmp_path / "prediction"
    result = predict(model_dir, image, output)
    assert result["decision"] is None
    assert json.loads((output / "prediction.json").read_text())["raw_anomaly_score"] == result["raw_anomaly_score"]
    assert np.load(output / "anomaly_map.npy", allow_pickle=False).shape == (64, 64)
    with pytest.raises(ValueError, match="must be empty"):
        fit(references, model_dir, limit=4, calibration_count=0)
    with (model_dir / "model.npz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="integrity"):
        predict(model_dir, image, output)
