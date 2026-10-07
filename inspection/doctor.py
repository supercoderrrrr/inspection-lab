"""Validate local setup and recompute recorded metrics without loading a network."""

from inspection.paths import ARTIFACTS, DATA

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from inspection.artifacts import completed_runs, verify_run
from inspection.data import validate, CATEGORIES
from inspection.protocol import calibrated_threshold, classification_metrics, file_digest


def audit_run(run, verify_data=False):
    errors = verify_run(run)
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    data_root = Path(config.get("data_root", DATA))
    splits = json.loads((run / "splits.json").read_text(encoding="utf-8"))
    groups = {name: {r["path"] for r in values} for name, values in splits.items()}
    for first, second in [("training_pool", "calibration"), ("training_pool", "test"), ("calibration", "test")]:
        if groups[first] & groups[second]:
            errors.append(f"Split overlap: {first}/{second}")
    if verify_data:
        for items in splits.values():
            for item in items:
                path = (data_root / item["path"]).resolve()
                if not path.is_relative_to(data_root.resolve()) or not path.is_file() or file_digest(path) != item["sha256"]:
                    errors.append(f"Dataset mismatch: {item['path']}")
    for _, row in pd.read_csv(run / "summary.csv").iterrows():
        folder = run / f"n{int(row.budget)}"
        calibration = pd.read_csv(folder / "calibration.csv")
        threshold = calibrated_threshold(calibration.score, config["alpha"])
        predictions = pd.read_csv(folder / f"predictions_{row.condition}.csv")
        if set(predictions.path) != groups["test"] or len(predictions) != len(groups["test"]):
            errors.append(f"Test coverage mismatch: {folder.name}/{row.condition}")
        if not np.isclose(threshold, row.threshold, rtol=1e-6):
            errors.append(f"Calibration mismatch: {folder.name}")
        expected = classification_metrics(predictions.label, predictions.score, threshold)
        for metric, value in expected.items():
            if value is not None and not np.isclose(value, row[metric], atol=1e-8):
                errors.append(f"Metric mismatch: {folder.name}/{row.condition}/{metric}")
        if not np.array_equal(predictions.prediction, (predictions.score > threshold).astype(int)):
            errors.append(f"Decision mismatch: {folder.name}/{row.condition}")
    baseline = run / "baseline"
    if (baseline / "summary.csv").exists():
        for _, row in pd.read_csv(baseline / "summary.csv").iterrows():
            budget = int(row.budget)
            calibration = pd.read_csv(baseline / f"n{budget}_calibration.csv")
            if set(calibration.path) != groups["calibration"]:
                errors.append(f"Baseline calibration coverage mismatch: n{budget}")
            threshold = calibrated_threshold(calibration.score, config["alpha"])
            predictions = pd.read_csv(baseline / f"n{budget}_{row.condition}.csv")
            if set(predictions.path) != groups["test"] or len(predictions) != len(groups["test"]):
                errors.append(f"Baseline test coverage mismatch: n{budget}")
            if not np.isclose(threshold, row.threshold, rtol=1e-6):
                errors.append(f"Baseline calibration threshold mismatch: n{budget}")
            values = classification_metrics(predictions.label, predictions.score, threshold)
            for metric, value in values.items():
                if value is not None and not np.isclose(value, row[metric], atol=1e-8):
                    errors.append(f"Baseline metric mismatch: n{budget}/{row.condition}/{metric}")
            if not np.array_equal(predictions.prediction, (predictions.score > threshold).astype(int)):
                errors.append(f"Baseline decision mismatch: n{budget}/{row.condition}")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run")
    parser.add_argument("--verify-data", action="store_true")
    args = parser.parse_args()
    result = {"cuda_available": torch.cuda.is_available(), "torch": torch.__version__}
    result["dataset"] = {c: validate(DATA / c) for c in CATEGORIES if (DATA / c).exists()}
    runs, rejected = completed_runs(ARTIFACTS)
    result["rejected_runs"] = rejected
    if args.run:
        runs = [p for p in runs if p.name == args.run]
        if not runs:
            raise SystemExit("The requested completed run was not found.")
    else:
        runs = [p for p in runs if (p / "integrity.json").exists()]
    result["runs"] = {p.name: audit_run(p, args.verify_data) for p in runs}
    result["passed"] = bool(runs) and not rejected and not any(result["runs"].values())
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
