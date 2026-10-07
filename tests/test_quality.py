import io
import json

import numpy as np
import pytest
from PIL import Image

from inspection.artifacts import atomic_json, completed_runs, seal_run, verify_run
from inspection.baseline import PixelTemplate
from inspection.images import decode_image, image_identity
from inspection.protocol import classification_metrics, split_normal, wilson_interval
from inspection.study import aggregate


def test_tampering_is_detected(tmp_path):
    (tmp_path / "result.csv").write_text("score\n1.0\n")
    seal_run(tmp_path)
    assert verify_run(tmp_path) == []
    (tmp_path / "result.csv").write_text("score\n0.0\n")
    assert "result.csv" in verify_run(tmp_path)[0]


def test_model_rejects_corruption_before_deserialization(tmp_path):
    from inspection.model import load_model
    (tmp_path / "model.pt").write_bytes(b"corrupt checkpoint")
    atomic_json(tmp_path / "checkpoint.json", {"sha256": "incorrect"})
    with pytest.raises(ValueError, match="integrity check failed"):
        load_model(tmp_path / "model.pt", device="cpu")


def test_manifest_cannot_escape_run(tmp_path):
    atomic_json(tmp_path / "integrity.json", {"files": {"../outside": "unused"}})
    assert "Invalid artifact path" in verify_run(tmp_path)[0]


def test_completed_run_requires_actual_artifacts(tmp_path):
    run = tmp_path / "broken"
    run.mkdir()
    atomic_json(run / "complete.json", {})
    atomic_json(run / "config.json", {"budgets": [16]})
    runs, rejected = completed_runs(tmp_path)
    assert not runs and "Required artifact missing" in rejected[0]


def test_image_content_validation():
    with pytest.raises(ValueError):
        decode_image(b"not a png")
    stream = io.BytesIO()
    Image.new("RGB", (8, 8)).save(stream, format="GIF")
    with pytest.raises(ValueError, match="PNG and JPEG"):
        decode_image(stream.getvalue())
    stream = io.BytesIO()
    Image.new("RGBA", (8, 8)).save(stream, format="PNG")
    content = stream.getvalue()
    assert decode_image(content).mode == "RGB"
    assert image_identity(content) != image_identity(content + b"x")


@pytest.mark.parametrize("labels,scores", [([0, 1], [0.1]), ([0, 2], [0.1, 0.2]),
                                           ([], []), ([0], [np.nan])])
def test_metrics_reject_invalid_inputs(labels, scores):
    with pytest.raises(ValueError):
        classification_metrics(labels, scores, 1)


def test_wilson_interval_and_invalid_splits():
    low, high = wilson_interval(2, 20)
    assert 0.027 < low < 0.029 and 0.30 < high < 0.302
    with pytest.raises(ValueError):
        split_normal(["a", "a", "b"], 42, 1)
    with pytest.raises(ValueError):
        split_normal(["a", "b"], 42, 1)


def test_pixel_baseline_detects_a_local_change(tmp_path):
    paths = []
    for i in range(2):
        path = tmp_path / f"normal{i}.png"
        Image.new("RGB", (32, 32), (100, 100, 100)).save(path)
        paths.append(path)
    baseline = PixelTemplate(size=64).fit(paths)
    normal = baseline.pixels(paths[0])
    normal_score, _ = baseline.predict_pixels(normal)
    defect = normal.copy()
    defect[8:20, 8:20] = 1
    defect_score, anomaly_map = baseline.predict_pixels(defect)
    assert defect_score > normal_score
    assert np.isfinite(anomaly_map).all()


def test_aggregation_reports_sample_std_not_pooled_tests():
    rows = []
    for score in [0.8, 0.9, 1.0]:
        rows.append(dict(method="test", budget=16, condition="clean", image_auroc=score,
                         pixel_auroc=score, recall=score, false_positive_rate=0.1, latency_median_ms=1))
    result = aggregate(rows).iloc[0]
    assert result.image_auroc_mean == pytest.approx(0.9)
    assert result.image_auroc_std == pytest.approx(0.1)
