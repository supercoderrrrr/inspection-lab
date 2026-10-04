"""Fit normal references and export a single-image score and raw response map."""

import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from inspection.baseline import PixelTemplate
from inspection.data import file_digest, normal_references
from inspection.images import validate_size


def select_references(root: Path, limit: int, seed: int) -> list[Path]:
    paths = normal_references(root)
    if not 2 <= limit <= len(paths):
        raise ValueError(f"Reference count must be between 2 and {len(paths)}.")
    return sorted(random.Random(seed).sample(paths, limit))


def write_json(path: Path, value: dict):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def fit(root: Path, output: Path, limit=16, seed=42, size=224, method="baseline") -> dict:
    validate_size(size)
    if not 0 <= seed < 2 ** 32:
        raise ValueError("Seed must be between zero and 2**32 - 1.")
    if method not in {"baseline", "patchcore"}:
        raise ValueError(f"Unknown method: {method}")
    paths = select_references(root, limit, seed)
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
    metadata = {
        "schema_version": 1,
        "method": method,
        "category": root.name,
        "image_size": size,
        "seed": seed,
        "reference_count": len(paths),
        "references": [{"path": p.relative_to(root).as_posix(), "sha256": file_digest(p)} for p in paths],
        "fitted_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": file_digest(checkpoint),
        "parameters": parameters,
    }
    write_json(output / "metadata.json", metadata)
    return metadata


def predict(model_dir: Path, image: Path, output: Path) -> dict:
    metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 1 or metadata.get("method") not in {"baseline", "patchcore"}:
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
    score, anomaly_map = model.predict(image)
    if not np.isfinite(score) or not np.isfinite(anomaly_map).all():
        raise ValueError("Prediction contains nonfinite values.")
    result = {
        "method": metadata["method"],
        "image": image.name,
        "image_sha256": file_digest(image),
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "image_size": model.size,
        "raw_anomaly_score": score,
        "anomaly_map": "anomaly_map.npy",
        "decision": None,
        "note": "Raw scores are not probabilities. No decision threshold is calibrated in this stage.",
    }
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "anomaly_map.npy", anomaly_map, allow_pickle=False)
    write_json(output / "prediction.json", result)
    return result
