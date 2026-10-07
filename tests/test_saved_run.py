"""Optional integration checks against locally available real experiment artifacts."""

import pandas as pd
import pytest
import torch
from PIL import Image

from inspection.paths import ARTIFACTS, DATA
from inspection.model import load_model, predict_image


def test_real_checkpoint_reproduces_recorded_examples():
    run = ARTIFACTS / "bottle_verified"
    if not (run / "complete.json").exists():
        pytest.skip("Local benchmark artifacts are not distributed with source code.")
    torch.set_num_threads(2)
    model, payload, device = load_model(run / "n179/model.pt", device="cpu")
    frame = pd.read_csv(run / "n179/predictions_clean.csv")
    examples = pd.concat([frame[frame.label == 0].head(1), frame[frame.label == 1].head(1)])
    for _, row in examples.iterrows():
        with Image.open(DATA / row.path) as image:
            score, anomaly_map = predict_image(model, image, payload["config"]["image_size"], device)
        # CPU/CUDA convolution and nearest-neighbor arithmetic are not bit-identical.
        # Require agreement within 0.1% plus unchanged decisions for these examples.
        assert score == pytest.approx(row.score, rel=1e-3, abs=1e-3)
        assert int(score > payload["threshold"]) == row.prediction
        assert anomaly_map.shape == (224, 224)
