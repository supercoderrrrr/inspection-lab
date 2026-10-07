"""Download an official category archive and validate its expected layout."""

import argparse
import json
import shutil
import tarfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from PIL import Image

from inspection.paths import DATA
from inspection.protocol import file_digest

URL = "https://www.mydrive.ch/shares/150452/132a93367fb17cdf968dfb5c4013f6e7/download/420937370-1629958698/bottle.tar.xz"
SOURCE = "https://www.mvtec.com/research-teaching/datasets/mvtec-ad/downloads"
CATEGORIES = {
    "bottle": {"url": URL, "counts": {"train_good": 209, "test": 83, "masks": 63}},
    "metal_nut": {"url": "https://www.mydrive.ch/shares/150459/c68856a21dca589b0f8ff6d4ee0f18f4/download/420937637-1629959294/metal_nut.tar.xz", "counts": {"train_good": 220, "test": 115, "masks": 93}},
}


def validate(root: Path, category=None):
    category = category or root.name
    counts = {"train_good": len(list((root / "train/good").glob("*.png"))),
              "test": len(list((root / "test").glob("*/*.png"))),
              "masks": len(list((root / "ground_truth").glob("*/*.png")))}
    if counts != CATEGORIES[category]["counts"]:
        raise ValueError(f"Unexpected MVTec AD {category} layout: {counts}")
    for path in sorted((root / "test").glob("*/*.png")):
        if path.parent.name != "good":
            mask = root / "ground_truth" / path.parent.name / f"{path.stem}_mask.png"
            if not mask.is_file():
                raise ValueError(f"Missing annotation: {mask}")
    return counts


def download(category="bottle"):
    url = CATEGORIES[category]["url"]
    DATA.mkdir(parents=True, exist_ok=True)
    root = DATA / category
    if root.exists():
        print(json.dumps(validate(root)), flush=True)
        return
    archive = DATA / f"{category}.tar.xz"
    if not archive.exists():
        partial = archive.with_suffix(".partial")
        print(f"Downloading the official MVTec AD {category} archive...", flush=True)
        request = urllib.request.Request(url, headers={"User-Agent": "InspectionLab/1.0"})
        with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        partial.replace(archive)
    staging = DATA / f"extracting_{category}"
    staging.mkdir(exist_ok=True)
    with tarfile.open(archive, "r:xz") as handle:
        handle.extractall(staging, filter="data")
    candidates = [staging / category, staging]
    extracted = next((p for p in candidates if (p / "train/good").is_dir()), None)
    if extracted is None:
        raise ValueError(f"Archive does not contain the expected {category} layout.")
    counts = validate(extracted, category)
    extracted.rename(root)
    manifest = {"source": SOURCE, "archive_url": url, "sha256": file_digest(archive),
                "downloaded_at": datetime.now(timezone.utc).isoformat(),
                "license": "CC BY-NC-SA 4.0; dataset terms are separate from project code", "counts": counts}
    (DATA / f"{category}_provenance.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", choices=list(CATEGORIES), default="bottle")
    download(parser.parse_args().category)


def validate_custom(root: Path):
    """Validate the documented custom evaluation layout without enforcing MVTec counts."""
    train = sorted((root / "train/good").glob("*.png"))
    test = sorted((root / "test").glob("*/*.png"))
    if not train or not test:
        raise ValueError("Expected train/good/*.png and test/<label>/*.png.")
    normal = [p for p in test if p.parent.name == "good"]
    anomalies = [p for p in test if p.parent.name != "good"]
    if not normal or not anomalies:
        raise ValueError("Evaluation requires both normal and anomalous test images.")
    for path in anomalies:
        if not (root / "ground_truth" / path.parent.name / f"{path.stem}_mask.png").is_file():
            raise ValueError(f"Missing pixel annotation for {path.name}; see docs/CUSTOM_DATA.md.")
    hashes = {}
    for path in train + test:
        digest = file_digest(path)
        partition = "train" if path in train else "test"
        if digest in hashes:
            raise ValueError(f"Duplicate image content ({hashes[digest]}/{partition}): {path}")
        hashes[digest] = partition
    return {"train_good": len(train), "test": len(test), "masks": len(anomalies)}


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
