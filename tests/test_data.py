import shutil

import pytest
from PIL import Image

from inspection.cli import main
from inspection.data import normal_references, validate_category


@pytest.fixture
def category(tmp_path):
    root = tmp_path / "bottle"
    samples = {
        "train/good/000.png": (20, 40, 60),
        "train/good/001.png": (30, 45, 65),
        "test/good/000.png": (35, 50, 70),
        "test/broken/000.png": (90, 0, 0),
        "ground_truth/broken/000_mask.png": (255, 255, 255),
    }
    for relative, color in samples.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32, 32), color).save(path)
    return root


def test_valid_category(category):
    report = validate_category(category)
    assert report["train_good"] == 2
    assert report["test_by_label"] == {"broken": 1, "good": 1}
    assert report["image_dimensions"] == [[32, 32]]


def test_training_never_uses_test_images(category):
    assert all(p.parent == category / "train/good" for p in normal_references(category))


def test_rejects_cross_partition_duplicates(category):
    shutil.copyfile(category / "train/good/000.png", category / "test/good/000.png")
    with pytest.raises(ValueError, match="overlap"):
        validate_category(category)


def test_rejects_missing_or_wrong_size_mask(category):
    mask = category / "ground_truth/broken/000_mask.png"
    Image.new("RGB", (16, 16)).save(mask)
    with pytest.raises(ValueError, match="dimensions"):
        validate_category(category)
    mask.unlink()
    with pytest.raises(ValueError, match="Missing annotation"):
        validate_category(category)


def test_corrupt_image_returns_cli_error(category, capsys):
    (category / "test/good/000.png").write_bytes(b"not a PNG")
    with pytest.raises(SystemExit) as error:
        main(["check-data", "--root", str(category)])
    assert error.value.code == 2
    assert "Invalid image" in capsys.readouterr().err


def test_rejects_defect_training_folder(category):
    (category / "train/broken").mkdir()
    with pytest.raises(ValueError, match="normal-only"):
        normal_references(category)
