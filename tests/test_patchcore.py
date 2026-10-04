import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")
pytest.importorskip("anomalib")

from inspection.patchcore import PatchCoreDetector


def test_fit_and_checkpoint_round_trip_without_network(tmp_path):
    """Random features exercise the pipeline; they are not an accuracy benchmark."""
    torch.set_num_threads(2)
    rng = np.random.default_rng(42)
    paths = []
    for i in range(3):
        path = tmp_path / f"{i}.png"
        Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)).save(path)
        paths.append(path)
    torch.manual_seed(42)
    model = PatchCoreDetector(size=64, pretrained=False).fit(paths, seed=42, batch_size=2)
    assert model.parameters["memory_bank_patches"] >= 9
    assert model.parameters["feature_initialization"] == "random"
    assert all(not parameter.requires_grad for parameter in model.model.parameters())
    assert not model.model.feature_extractor.training
    expected_score, expected_map = model.predict(paths[0])
    checkpoint = tmp_path / "model.pt"
    model.save(checkpoint)
    restored = PatchCoreDetector.load(checkpoint)
    actual_score, actual_map = restored.predict(paths[0])
    assert np.isfinite(actual_score)
    assert actual_map.shape == (64, 64)
    assert restored.parameters == model.parameters
    np.testing.assert_allclose(actual_score, expected_score, rtol=1e-6)
    np.testing.assert_allclose(actual_map, expected_map, rtol=1e-6)


def test_rejects_unsupported_checkpoint(tmp_path):
    checkpoint = tmp_path / "invalid.pt"
    torch.save({"schema_version": 99}, checkpoint)
    with pytest.raises(ValueError, match="schema"):
        PatchCoreDetector.load(checkpoint)
