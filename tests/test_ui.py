from pathlib import Path

import pytest
from PIL import Image

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

from inspection.workflow import fit

APP = Path(__file__).resolve().parents[1] / "app.py"


def test_source_checkout_without_models_still_shows_public_evaluation(tmp_path, monkeypatch):
    monkeypatch.setenv("INSPECTION_MODEL_ROOT", str(tmp_path / "models"))
    monkeypatch.setenv("INSPECTION_DATA_ROOT", str(tmp_path / "data"))
    app = AppTest.from_file(str(APP)).run(timeout=20)
    assert not app.exception
    assert any("Fit a reference model" in item.value for item in app.info)
    assert len(app.dataframe) == 3


def test_switching_models_clears_previous_decision(tmp_path, monkeypatch):
    data = tmp_path / "data"
    root = data / "bottle"
    for i in range(8):
        path = root / f"train/good/{i}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 64), (80 + i, 80 + i, 80 + i)).save(path)
    image = root / "test/good/000.png"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (64, 64), (90, 90, 90)).save(image)
    models = tmp_path / "models"
    fit(root, models / "calibrated", limit=4, size=64, calibration_count=3, alpha=0.25)
    fit(root, models / "raw", limit=4, size=64, calibration_count=0)
    monkeypatch.setenv("INSPECTION_MODEL_ROOT", str(models))
    monkeypatch.setenv("INSPECTION_DATA_ROOT", str(data))
    app = AppTest.from_file(str(APP)).run(timeout=20)
    assert not app.exception
    app.button[0].click().run(timeout=20)
    assert not app.exception
    assert any(metric.label == "Raw anomaly score" for metric in app.metric)
    app.sidebar.selectbox[0].set_value(1).run(timeout=20)
    assert not app.exception
    assert not app.metric
