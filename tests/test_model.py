import numpy as np
import torch
from PIL import Image

from inspection.model import create_model, overlay, prepare


def test_preprocessing():
    image = Image.new("RGB", (50, 100), (255, 0, 0))
    tensor = prepare(image, 64)
    assert tensor.shape == (3, 64, 64)
    assert torch.allclose(tensor[0], torch.full((64, 64), (1 - 0.485) / 0.229))
    assert torch.equal(tensor, prepare(image, 64))


def test_fixed_heatmap_scale():
    image = Image.new("RGB", (64, 64))
    low = np.ones((64, 64), dtype=np.float32)
    assert not np.array_equal(np.asarray(overlay(image, low, 10)), np.asarray(overlay(image, low * 5, 10)))


def test_checkpoint_round_trip(tmp_path):
    """Synthetic tensors test serialization only, never model accuracy."""
    torch.set_num_threads(2)
    model = create_model(pretrained=False).eval()
    model.memory_bank = torch.rand(12, 384)
    inputs = torch.rand(1, 3, 64, 64)
    with torch.inference_mode():
        expected = model(inputs).pred_score
    checkpoint = tmp_path / "state.pt"
    torch.save(model.state_dict(), checkpoint)
    restored = create_model(pretrained=False).eval()
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    with torch.inference_mode():
        actual = restored(inputs).pred_score
    assert torch.allclose(expected, actual)


def test_overlay_opacity_preserves_input_and_map():
    image = Image.new("RGB", (64, 64), (20, 40, 60))
    raw = np.arange(4096, dtype=np.float32).reshape(64, 64)
    original = raw.copy()
    assert np.array_equal(np.asarray(overlay(image, raw, 100, opacity=0)), np.asarray(image))
    overlay(image, raw, float(raw.max()), opacity=0.8)
    assert np.array_equal(raw, original)


def test_response_weighting_reduces_low_response_tint_without_changing_map():
    image = Image.new("RGB", (64, 64), (80, 80, 80))
    raw = np.full((64, 64), 0.2, dtype=np.float32)
    raw[:, 32:] = 1
    before = raw.copy()
    weighted = np.asarray(overlay(image, raw, 1, 0.6, response_weighted=True), dtype=float)
    uniform = np.asarray(overlay(image, raw, 1, 0.6), dtype=float)
    assert np.mean(abs(weighted[:, :32] - 80)) < np.mean(abs(uniform[:, :32] - 80)) / 10
    assert np.max(abs(weighted[:, 32:] - uniform[:, 32:])) <= 1
    assert np.array_equal(raw, before)
    zero = np.zeros((64, 64), dtype=np.float32)
    assert np.array_equal(np.asarray(overlay(image, zero, 1, 0.6, True)), np.asarray(image))
