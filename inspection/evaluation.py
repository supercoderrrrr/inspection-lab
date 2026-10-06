"""Evaluate saved, calibrated models without changing their fitted state."""

import csv
import json
import platform
import shutil
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

from inspection.data import file_digest, validate_category
from inspection.protocol import anomaly_decision
from inspection.workflow import calibration_records, inspect_image, load_detector, write_json


def classification_metrics(labels, scores, threshold):
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    if (labels.ndim != 1 or scores.ndim != 1 or not len(labels) or len(labels) != len(scores)
            or not np.isin(labels, [0, 1]).all() or not np.isfinite(scores).all()
            or not np.isfinite(threshold)):
        raise ValueError("Metrics require binary labels and matching finite scores and threshold.")
    labels = labels.astype(int)
    predicted = scores > threshold
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    both_classes = len(np.unique(labels)) == 2
    return {
        "image_auroc": float(roc_auc_score(labels, scores)) if both_classes else None,
        "average_precision": float(average_precision_score(labels, scores)) if both_classes else None,
        "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "recall": float(tp / (tp + fn)) if tp + fn else None,
        "normal_false_alarm_rate": float(fp / (tn + fp)) if tn + fp else None,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def environment():
    libraries = {}
    for name in ["numpy", "Pillow", "scipy", "scikit-learn", "torch", "torchvision", "anomalib", "timm"]:
        try:
            libraries[name] = version(name)
        except PackageNotFoundError:
            continue
    return {"python": platform.python_version(), "platform": platform.system(), "libraries": libraries}


def evaluate(model_dir: Path, root: Path, output: Path):
    if output.exists() and any(output.iterdir()):
        raise ValueError("Evaluation output directory must be empty; choose a new directory.")
    counts = validate_category(root)
    model, metadata = load_detector(model_dir)
    calibration = metadata.get("calibration")
    if not calibration:
        raise ValueError("Evaluation requires a calibrated model; fit with a positive calibration count.")
    if metadata["category"] != root.name:
        raise ValueError("Evaluation category does not match the saved model.")
    sources = metadata["references"] + calibration_records(model_dir, metadata)
    for record in sources:
        relative = Path(record["path"])
        if relative.is_absolute() or ".." in relative.parts or not record["path"].startswith("train/good/"):
            raise ValueError("Model source paths must stay under train/good.")
        if file_digest(root / relative) != record["sha256"]:
            raise ValueError("Fitting/calibration source images changed since model creation.")

    threshold = calibration["threshold"]
    rows = []
    for path in sorted((root / "test").glob("*/*.png")):
        result, _ = inspect_image(model, metadata, path)
        rows.append({
            "path": path.relative_to(root).as_posix(), "sha256": result["image_sha256"],
            "label": path.parent.name, "is_anomaly": int(path.parent.name != "good"),
            "score": result["raw_anomaly_score"],
            "predicted_anomaly": int(anomaly_decision(result["raw_anomaly_score"], threshold)),
        })
    metrics = classification_metrics([r["is_anomaly"] for r in rows], [r["score"] for r in rows], threshold)
    report = {
        "schema_version": 1, "method": metadata["method"], "category": root.name,
        "reference_count": metadata["reference_count"], "calibration_count": calibration["count"],
        "seed": metadata["seed"], "threshold": threshold,
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": counts, "metrics": metrics,
    }
    output.mkdir(parents=True, exist_ok=True)
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(output / "metrics.json", report)
    write_json(output / "environment.json", environment())
    shutil.copyfile(model_dir / "metadata.json", output / "model_metadata.json")
    shutil.copyfile(model_dir / "calibration.csv", output / "calibration.csv")
    failures = [r for r in rows if r["is_anomaly"] != r["predicted_anomaly"]]
    lines = [f"# {root.name}: {metadata['method']}", "",
             f"Seed {metadata['seed']}; {metadata['reference_count']} fitted references; "
             f"{calibration['count']} normal calibration images; {len(rows)} test images.", "",
             "The threshold was fixed using normal calibration images before test scoring.", "",
             "| Metric | Value |", "|---|---:|"]
    lines.extend(f"| {name} | {value} |" for name, value in metrics.items())
    lines.extend(["", "## Incorrect decisions", "",
                  "| Image | True label | Score | Predicted anomaly |", "|---|---|---:|---:|"])
    lines.extend(f"| {r['path']} | {r['label']} | {r['score']:.6f} | {r['predicted_anomaly']} |"
                 for r in failures[:12])
    lines.extend(["", f"{len(failures)} incorrect decisions; up to 12 shown. See predictions.csv for all images.", "",
                  "This is one reference budget and one seed on one product. It does not establish",
                  "performance across products, industrial operating conditions or repeated runs.", "",
                  "Dataset: MVTec AD, MVTec Software GmbH. No images or fitted weights are included."])
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    files = [p for p in output.iterdir() if p.is_file()]
    write_json(output / "MANIFEST.json", {"files": {p.name: file_digest(p) for p in sorted(files)}})
    return report


def verify_evaluation(output: Path):
    manifest = json.loads((output / "MANIFEST.json").read_text(encoding="utf-8"))
    required = {"predictions.csv", "metrics.json", "environment.json", "model_metadata.json",
                "calibration.csv", "REPORT.md"}
    if set(manifest.get("files", {})) != required:
        raise ValueError("Evaluation manifest does not contain the required evidence files.")
    for name, digest in manifest["files"].items():
        if file_digest(output / name) != digest:
            raise ValueError(f"Evaluation evidence changed: {name}")
    metadata = json.loads((output / "model_metadata.json").read_text(encoding="utf-8"))
    calibration_records(output, metadata)
    with (output / "predictions.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    report = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    threshold = metadata["calibration"]["threshold"]
    if report["threshold"] != threshold or report["checkpoint_sha256"] != metadata["checkpoint_sha256"]:
        raise ValueError("Evaluation report does not match the fitted model.")
    seen = set()
    for row in rows:
        path = Path(row["path"])
        if path.is_absolute() or len(path.parts) != 3 or path.parts[0] != "test" or ".." in path.parts:
            raise ValueError("Predictions must identify test images.")
        if row["path"] in seen:
            raise ValueError("Duplicate test image in predictions.")
        seen.add(row["path"])
        if row["label"] != path.parts[1] or int(row["is_anomaly"]) != int(path.parts[1] != "good"):
            raise ValueError("Prediction labels do not match dataset paths.")
        if int(row["predicted_anomaly"]) != int(anomaly_decision(float(row["score"]), threshold)):
            raise ValueError("Predictions do not use the recorded threshold.")
    recomputed = classification_metrics([int(r["is_anomaly"]) for r in rows],
                                       [float(r["score"]) for r in rows], threshold)
    if recomputed != report["metrics"]:
        raise ValueError("Recorded metrics do not match public predictions.")
    return {"verified": True, "images": len(rows), "metrics": recomputed}
