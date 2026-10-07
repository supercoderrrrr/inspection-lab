"""Interactive state checks without weights or dataset downloads."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from streamlit.testing.v1 import AppTest

import inspection.artifacts as artifacts
import inspection.model as models
import inspection.paths as paths

APP = Path(__file__).resolve().parents[1] / "app.py"


def test_source_checkout_without_models_shows_public_evidence(monkeypatch):
    monkeypatch.setattr(artifacts, "completed_runs", lambda _: ([], []))
    app = AppTest.from_file(str(APP)).run(timeout=30)
    assert not app.exception
    assert any("No completed experiment" in item.value for item in app.info)
    assert any("Published benchmark evidence" in item.value for item in app.subheader)


def test_switching_inputs_and_models_clears_previous_decision(tmp_path, monkeypatch):
    data = tmp_path / "data"
    good = data / "bottle/test/good"
    good.mkdir(parents=True)
    for i in range(2):
        Image.new("RGB", (64, 64), (80 + i, 80 + i, 80 + i)).save(good / f"{i}.png")
    runs = []
    for seed in [42, 123]:
        run = tmp_path / "artifacts" / f"example_{seed}"
        folder = run / "n16"
        folder.mkdir(parents=True)
        (folder / "model.pt").write_bytes(b"synthetic UI test fixture")
        (run / "config.json").write_text(json.dumps(dict(
            category="bottle", budgets=[16], backbone="resnet18", seed=seed,
            image_size=64, data_root=str(data))))
        (run / "environment.json").write_text("{}")
        (run / "REPORT.md").write_text("# UI fixture")
        pd.DataFrame([dict(budget=16, condition="clean", image_auroc=0.5, recall=0,
            false_positive_rate=0, latency_median_ms=1, threshold=2,
            tp=0, tn=2, fp=0, fn=1)]).to_csv(run / "summary.csv", index=False)
        pd.DataFrame([dict(path=f"bottle/test/good/{i}.png", defect="good",
            label=0, score=1, prediction=0) for i in range(2)]
        ).to_csv(folder / "predictions_clean.csv", index=False)
        runs.append(run)
    monkeypatch.setattr(artifacts, "completed_runs", lambda _: (runs, []))
    monkeypatch.setattr(paths, "ROOT", tmp_path)
    monkeypatch.setattr(paths, "ARTIFACTS", tmp_path / "artifacts")
    monkeypatch.setattr(models, "load_model", lambda *a, **k: (
        object(), {"threshold": 2, "heatmap_max": 4}, "cpu"))
    monkeypatch.setattr(models, "predict_image", lambda *a: (3.0, np.ones((64, 64))))
    app = AppTest.from_file(str(APP)).run(timeout=30)
    assert not app.exception
    app.button[0].click().run(timeout=30)
    assert not app.exception
    assert any(metric.label == "Raw anomaly score" for metric in app.metric)
    next(box for box in app.selectbox if box.label == "Example").set_value(good / "1.png").run(timeout=30)
    assert not app.exception
    assert all(metric.label != "Raw anomaly score" for metric in app.metric)
    app.button[0].click().run(timeout=30)
    assert any(metric.label == "Raw anomaly score" for metric in app.metric)
    next(box for box in app.sidebar.selectbox if box.label == "Completed run").set_value(runs[1]).run(timeout=30)
    assert not app.exception
    assert all(metric.label != "Raw anomaly score" for metric in app.metric)
