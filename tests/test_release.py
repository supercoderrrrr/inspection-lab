"""Portable behavior checks that do not download weights or benchmark data."""

import json
import random

import numpy as np
import pytest
import torch
from PIL import Image

from inspection.data import validate_custom
from inspection.model import create_model, fit_reference_model, load_model, predict_image
from inspection.predict import inspect_paths


def synthetic_references(tmp_path, count=4):
    rng = np.random.default_rng(42)
    paths = []
    for i in range(count):
        path = tmp_path / f"{i}.png"
        Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)).save(path)
        paths.append(path)
    return paths


@pytest.mark.parametrize("method", ["patchcore", "padim"])
def test_fitted_detector_round_trip_without_pretrained_download(tmp_path, method):
    torch.set_num_threads(2)
    torch.manual_seed(42)
    random.seed(42)
    paths = synthetic_references(tmp_path)
    model = create_model(pretrained=False, method=method)
    fit_reference_model(model, paths, 64, "cpu", bank_patches=15 if method == "patchcore" else None,
                        method=method)
    if method == "patchcore":
        assert model.memory_bank.shape == (15, 384)
    image = Image.open(paths[0]).convert("RGB")
    expected, expected_map = predict_image(model, image, 64, "cpu")
    config = {"backbone": "resnet18", "method": method, "image_size": 64, "category": "synthetic"}
    payload = {"config": config, "state_dict": model.state_dict(), "threshold": expected + 1, "heatmap_max": 10}
    checkpoint = tmp_path / "model.pt"
    torch.save(payload, checkpoint)
    restored, _, _ = load_model(checkpoint, "cpu")
    actual, actual_map = predict_image(restored, image, 64, "cpu")
    assert actual == pytest.approx(expected, rel=1e-5, abs=1e-5)
    assert np.isfinite(actual_map).all()
    assert np.allclose(actual_map, expected_map)


def test_custom_data_rejects_leakage_and_missing_masks(tmp_path):
    for folder in ["train/good", "test/good", "test/scratch", "ground_truth/scratch"]:
        (tmp_path / folder).mkdir(parents=True)
    Image.new("RGB", (8, 8), "red").save(tmp_path / "train/good/0.png")
    Image.new("RGB", (8, 8), "blue").save(tmp_path / "test/good/0.png")
    Image.new("RGB", (8, 8), "green").save(tmp_path / "test/scratch/0.png")
    with pytest.raises(ValueError, match="Missing pixel annotation"):
        validate_custom(tmp_path)
    Image.new("L", (8, 8), 255).save(tmp_path / "ground_truth/scratch/0_mask.png")
    assert validate_custom(tmp_path)["test"] == 2
    (tmp_path / "test/good/0.png").write_bytes((tmp_path / "train/good/0.png").read_bytes())
    with pytest.raises(ValueError, match="Duplicate image"):
        validate_custom(tmp_path)


def test_batch_records_invalid_file_and_preserves_existing_output(tmp_path, monkeypatch):
    import inspection.predict as predictor
    monkeypatch.setattr(predictor, "load_model", lambda *a: (
        object(), {"config": {"category": "synthetic", "image_size": 64}, "threshold": 2, "heatmap_max": 2}, "cpu"))
    monkeypatch.setattr(predictor, "predict_image", lambda *a: (3.0, np.ones((64, 64))))
    inputs = tmp_path / "input"
    inputs.mkdir()
    Image.new("RGB", (8, 8)).save(inputs / "valid.png")
    (inputs / "invalid.png").write_bytes(b"not an image")
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"test double, not a real model")
    output = tmp_path / "output"
    result = inspect_paths(checkpoint, inputs, output, save_overlays=True)
    assert result["errors"] == 1
    assert len(result["images"]) == 2
    good = next(row for row in result["images"] if row["status"] == "ok")
    assert good["flagged"] is True and (output / good["overlay"]).exists()
    assert json.loads((output / "results.json").read_text())["errors"] == 1
    with pytest.raises(ValueError, match="already exists"):
        inspect_paths(checkpoint, inputs, output)
    with pytest.raises(ValueError, match="outside"):
        inspect_paths(checkpoint, inputs, inputs / "nested")


def test_app_without_checkpoints_shows_setup_instead_of_crashing(monkeypatch):
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    import inspection.artifacts as artifacts
    monkeypatch.setattr(artifacts, "completed_runs", lambda _: ([], []))
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run(timeout=30)
    assert not app.exception
    assert any("No completed experiment" in item.value for item in app.info)


def test_model_only_install_does_not_require_failure_images(tmp_path, monkeypatch):
    from pathlib import Path
    import pandas as pd
    from streamlit.testing.v1 import AppTest
    import inspection.artifacts as artifacts
    import inspection.paths as paths
    run = tmp_path / "artifacts/example"
    (run / "n16").mkdir(parents=True)
    config = dict(category="bottle", budgets=[16], backbone="resnet18", seed=42,
                  image_size=64, data_root=str(tmp_path / "missing-data"))
    artifacts.atomic_json(run / "config.json", config)
    artifacts.atomic_json(run / "environment.json", {})
    (run / "REPORT.md").write_text("# Test fixture\n")
    pd.DataFrame([dict(budget=16, condition="clean", image_auroc=0.5, recall=1.0,
                      false_positive_rate=1.0, latency_median_ms=1, threshold=1,
                      tp=1, tn=0, fp=1, fn=0)]).to_csv(run / "summary.csv", index=False)
    pd.DataFrame([dict(path="bottle/test/good/0.png", defect="good", label=0, score=2, prediction=1),
                  dict(path="bottle/test/broken/0.png", defect="broken", label=1, score=3, prediction=1)]
                 ).to_csv(run / "n16/predictions_clean.csv", index=False)
    monkeypatch.setattr(artifacts, "completed_runs", lambda _: ([run], []))
    monkeypatch.setattr(paths, "ROOT", tmp_path)
    monkeypatch.setattr(paths, "ARTIFACTS", tmp_path / "artifacts")
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run(timeout=30)
    assert not app.exception
    assert any("Failure image is not installed" in item.value for item in app.info)
    assert any(item.value == "Upload image" for item in app.radio)
