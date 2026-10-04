"""Validate MVTec-style category folders without changing their contents."""

import hashlib
from pathlib import Path

from PIL import Image


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def image_size(path: Path) -> tuple[int, int]:
    try:
        with Image.open(path) as image:
            if image.format != "PNG":
                raise ValueError("Expected PNG content")
            size = image.size
            image.verify()
        return size
    except (OSError, ValueError, SyntaxError) as error:
        raise ValueError(f"Invalid image: {path.name}") from error


def normal_references(root: Path) -> list[Path]:
    root = Path(root)
    paths = sorted((root / "train" / "good").glob("*.png"))
    if len(paths) < 2:
        raise ValueError("Expected at least two normal PNG images in train/good.")
    unexpected = [p.name for p in (root / "train").iterdir() if p.is_dir() and p.name != "good"]
    if unexpected:
        raise ValueError("Fitting requires normal-only training data in train/good.")
    return paths


def validate_category(root: Path) -> dict:
    root = Path(root)
    train = normal_references(root)
    test = sorted((root / "test").glob("*/*.png"))
    good = [p for p in test if p.parent.name == "good"]
    defects = [p for p in test if p.parent.name != "good"]
    if not good or not defects:
        raise ValueError("Expected normal and defect PNG images in test/<label>.")

    dimensions = set()
    train_hashes = set()
    test_counts = {}
    for image in train:
        dimensions.add(image_size(image))
        train_hashes.add(file_digest(image))
    for image in test:
        size = image_size(image)
        dimensions.add(size)
        if file_digest(image) in train_hashes:
            raise ValueError(f"Training/test content overlap: {image.parent.name}/{image.name}")
        label = image.parent.name
        test_counts[label] = test_counts.get(label, 0) + 1
        if label != "good":
            mask = root / "ground_truth" / label / f"{image.stem}_mask.png"
            if not mask.is_file():
                raise ValueError(f"Missing annotation: ground_truth/{label}/{mask.name}")
            if image_size(mask) != size:
                raise ValueError(f"Annotation dimensions differ from image: {mask.name}")
    return {
        "category": root.name,
        "train_good": len(train),
        "test_good": len(good),
        "test_defect": len(defects),
        "test_by_label": test_counts,
        "masks": len(defects),
        "image_dimensions": [list(size) for size in sorted(dimensions)],
    }
