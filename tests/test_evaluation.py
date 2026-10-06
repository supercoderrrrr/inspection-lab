import json

import numpy as np
import pytest
from PIL import Image

from inspection.data import file_digest
from inspection.evaluation import classification_metrics, evaluate, verify_evaluation
from inspection.workflow import fit


def test_metrics_use_fixed_threshold_and_handle_ties():
    metrics = classification_metrics([0, 0, 1, 1], [0.0, 2.0, 1.0, 3.0], 2.0)
    assert (metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"]) == (2, 0, 1, 1)
    assert metrics["recall"] == 0.5
    assert metrics["normal_false_alarm_rate"] == 0.0
    assert metrics["image_auroc"] == 0.75
    assert classification_metrics([0, 0], [0.0, 1.0], 2.0)["image_auroc"] is None
    with pytest.raises(ValueError, match="binary"):
        classification_metrics([0, 2], [1.0, 2.0], 2.0)


def make_category(tmp_path):
    root = tmp_path / "bottle"
    for i in range(8):
        path = root / f"train/good/{i}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 64), (80 + i, 80 + i, 80 + i)).save(path)
    for label, color in [("good", 84), ("broken", 200)]:
        pixels = np.full((64, 64, 3), color, dtype=np.uint8)
        pixels[0, 0] = [0, 10, 20]
        path = root / f"test/{label}/000.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pixels).save(path)
    mask = root / "ground_truth/broken/000_mask.png"
    mask.parent.mkdir(parents=True)
    Image.new("L", (64, 64), 255).save(mask)
    return root


def test_evaluation_preserves_fitted_state_and_exports_verifiable_evidence(tmp_path):
    root = make_category(tmp_path)
    model_dir = tmp_path / "model"
    metadata = fit(root, model_dir, limit=4, size=64, calibration_count=3, alpha=0.25)
    before = {p.name: file_digest(p) for p in model_dir.iterdir()}
    output = tmp_path / "results"
    report = evaluate(model_dir, root, output)
    assert report["threshold"] == metadata["calibration"]["threshold"]
    assert before == {p.name: file_digest(p) for p in model_dir.iterdir()}
    assert verify_evaluation(output)["images"] == 2
    assert not list(output.glob("*.npz"))
    metrics = json.loads((output / "metrics.json").read_text())
    metrics["threshold"] = 0
    (output / "metrics.json").write_text(json.dumps(metrics))
    with pytest.raises(ValueError, match="evidence changed"):
        verify_evaluation(output)


def test_evaluation_rejects_changed_reference_source(tmp_path):
    root = make_category(tmp_path)
    model_dir = tmp_path / "model"
    metadata = fit(root, model_dir, limit=4, size=64, calibration_count=3, alpha=0.25)
    Image.new("RGB", (64, 64), (1, 2, 3)).save(root / metadata["references"][0]["path"])
    with pytest.raises(ValueError, match="source images changed"):
        evaluate(model_dir, root, tmp_path / "results")
