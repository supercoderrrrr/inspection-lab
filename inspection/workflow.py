"""Fit normal references and export a single-image score and raw response map."""

import csv
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from inspection.baseline import PixelTemplate
from inspection.data import file_digest, normal_references
from PIL import Image

from inspection.images import image_digest, validate_size
from inspection.protocol import anomaly_decision, calibrated_threshold, calibration_rank, split_normal


def select_references(root: Path, limit: int, seed: int) -> list[Path]:
    paths = normal_references(root)
    if not 2 <= limit <= len(paths):
        raise ValueError(f"Reference count must be between 2 and {len(paths)}.")
    return sorted(random.Random(seed).sample(paths, limit))


def write_json(path: Path, value: dict):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def fit(root: Path, output: Path, limit=16, seed=42, size=224, method="baseline",
        calibration_count=30, alpha=0.05) -> dict:
    validate_size(size)
    if not 0 <= seed < 2 ** 32:
        raise ValueError("Seed must be between zero and 2**32 - 1.")
    if method not in {"baseline", "patchcore"}:
        raise ValueError(f"Unknown method: {method}")
    pool, calibration = split_normal(normal_references(root), seed, calibration_count)
    if not 2 <= limit <= len(pool):
        raise ValueError(f"Reference count must be between 2 and {len(pool)} after calibration holdout.")
    if calibration:
        calibration_rank(len(calibration), alpha)
    paths = pool[:limit]
    source_records = [{"path": p.relative_to(root).as_posix(), "sha256": file_digest(p)}
                      for p in paths + calibration]
    if len({record["sha256"] for record in source_records}) != len(source_records):
        raise ValueError("Reference and calibration images must have distinct content.")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Model output directory must be empty; choose a new directory.")
    if method == "baseline":
        model = PixelTemplate(size).fit(paths)
        filename = "model.npz"
        parameters = {"std_floor": model.std_floor, "smoothing_sigma": 2, "score_quantile": 0.99}
    else:
        from inspection.patchcore import PatchCoreDetector

        model = PatchCoreDetector(size).fit(paths, seed=seed)
        filename = "model.pt"
        parameters = model.parameters
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / filename
    model.save(checkpoint)
    calibration_info = None
    if calibration:
        records = []
        for path, source in zip(calibration, source_records[limit:]):
            score, _ = model.predict(path)
            records.append({**source, "score": score})
        threshold = calibrated_threshold([record["score"] for record in records], alpha)
        calibration_file = output / "calibration.csv"
        with calibration_file.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["path", "sha256", "score"])
            writer.writeheader()
            writer.writerows(records)
        calibration_info = {
            "file": calibration_file.name, "sha256": file_digest(calibration_file),
            "count": len(calibration), "alpha": alpha, "threshold": threshold,
            "rank": calibration_rank(len(calibration), alpha), "rule": "score > threshold",
        }
    metadata = {
        "schema_version": 2,
        "method": method,
        "category": root.name,
        "image_size": size,
        "seed": seed,
        "reference_count": len(paths),
        "references": source_records[:limit],
        "calibration": calibration_info,
        "fitted_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": file_digest(checkpoint),
        "parameters": parameters,
    }
    write_json(output / "metadata.json", metadata)
    return metadata


def calibration_records(model_dir: Path, metadata: dict):
    calibration = metadata.get("calibration")
    if not calibration:
        return []
    path = model_dir / "calibration.csv"
    if calibration.get("file") != path.name or file_digest(path) != calibration["sha256"]:
        raise ValueError("Calibration integrity check failed.")
    with path.open(newline="", encoding="utf-8") as stream:
        records = list(csv.DictReader(stream))
    if len(records) != calibration["count"]:
        raise ValueError("Calibration count does not match metadata.")
    if any(not row["path"].startswith("train/good/") or ".." in Path(row["path"]).parts for row in records):
        raise ValueError("Calibration must use normal training images.")
    hashes = [row["sha256"] for row in records + metadata["references"]]
    if len(set(hashes)) != len(hashes):
        raise ValueError("Reference and calibration content overlaps.")
    threshold = calibrated_threshold([float(row["score"]) for row in records], calibration["alpha"])
    if threshold != calibration["threshold"] or calibration["rule"] != "score > threshold":
        raise ValueError("Saved threshold does not match calibration evidence.")
    return records


def load_detector(model_dir: Path):
    metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("schema_version") not in {1, 2} or metadata.get("method") not in {"baseline", "patchcore"}:
        raise ValueError("Unsupported model metadata.")
    method = metadata["method"]
    checkpoint = model_dir / ("model.npz" if method == "baseline" else "model.pt")
    if metadata.get("checkpoint") != checkpoint.name or file_digest(checkpoint) != metadata["checkpoint_sha256"]:
        raise ValueError("Checkpoint integrity check failed.")
    if method == "baseline":
        model = PixelTemplate.load(checkpoint)
    else:
        from inspection.patchcore import PatchCoreDetector

        model = PatchCoreDetector.load(checkpoint)
        if metadata["parameters"] != model.parameters:
            raise ValueError("Metadata parameters do not match the checkpoint.")
    if metadata["image_size"] != model.size:
        raise ValueError("Metadata image size does not match the checkpoint.")
    calibration_records(model_dir, metadata)
    return model, metadata


def inspect_image(model, metadata, image, image_name=None, image_sha256=None):
    score, anomaly_map = model.predict(image)
    if not np.isfinite(score) or not np.isfinite(anomaly_map).all():
        raise ValueError("Prediction contains nonfinite values.")
    calibration = metadata.get("calibration")
    threshold = calibration["threshold"] if calibration else None
    result = {
        "method": metadata["method"],
        "image": image_name or ("image" if isinstance(image, Image.Image) else Path(image).name),
        "image_sha256": image_sha256 or (image_digest(image) if isinstance(image, Image.Image) else file_digest(image)),
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "image_size": model.size,
        "raw_anomaly_score": score,
        "anomaly_map": "anomaly_map.npy",
        "calibrated_threshold": threshold,
        "decision": anomaly_decision(score, threshold) if threshold is not None else None,
        "note": "Raw scores are not probabilities; the decision uses a normal-only calibrated threshold."
                if calibration else "Raw scores are not probabilities. This model has no calibrated threshold.",
    }
    return result, anomaly_map


def predict(model_dir: Path, image: Path, output: Path) -> dict:
    model, metadata = load_detector(model_dir)
    result, anomaly_map = inspect_image(model, metadata, image)
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "anomaly_map.npy", anomaly_map, allow_pickle=False)
    write_json(output / "prediction.json", result)
    return result
