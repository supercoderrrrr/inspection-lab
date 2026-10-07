"""Verify the public evidence manifest and independently recompute image metrics."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from inspection.paths import ROOT
from inspection.protocol import calibrated_threshold, classification_metrics, file_digest


def verify_evidence(root: Path):
    root = root.resolve()
    manifest = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
    errors = []
    for relative, expected in manifest["files"].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file() or file_digest(path) != expected:
            errors.append(f"Missing, modified or invalid evidence: {relative}")
    if errors:
        return errors
    for run in (root / "runs").iterdir():
        config = json.loads((run / "config.json").read_text(encoding="utf-8"))
        splits = json.loads((run / "splits.json").read_text(encoding="utf-8"))
        expected_cal = {item["path"] for item in splits["calibration"]}
        expected_test = {item["path"] for item in splits["test"]}
        train = {item["path"] for item in splits["training_pool"]}
        if train & expected_cal or train & expected_test or expected_cal & expected_test:
            errors.append(f"Overlapping splits: {run.name}")
        for baseline in [False, True]:
            summary = run / "baseline/summary.csv" if baseline else run / "summary.csv"
            if not summary.exists():
                continue
            for _, row in pd.read_csv(summary).iterrows():
                folder = run / "baseline" if baseline else run / f"n{int(row.budget)}"
                prefix = f"n{int(row.budget)}_" if baseline else ""
                cal = pd.read_csv(folder / f"{prefix}calibration.csv")
                pred = pd.read_csv(folder / (f"{prefix}{row.condition}.csv" if baseline else f"predictions_{row.condition}.csv"))
                if set(cal.path) != expected_cal or len(cal) != len(expected_cal):
                    errors.append(f"Calibration coverage: {run.name}")
                if set(pred.path) != expected_test or len(pred) != len(expected_test):
                    errors.append(f"Test coverage: {run.name}")
                threshold = calibrated_threshold(cal.score, config["alpha"])
                if not np.isclose(threshold, row.threshold, rtol=1e-6):
                    errors.append(f"Threshold: {run.name}")
                if not np.array_equal(pred.prediction, (pred.score > threshold).astype(int)):
                    errors.append(f"Decisions: {run.name}")
                for metric, value in classification_metrics(pred.label, pred.score, threshold).items():
                    if value is not None and not np.isclose(value, row[metric], atol=1e-8):
                        errors.append(f"Metric {metric}: {run.name}")
    return errors


def main():
    errors = verify_evidence(ROOT / "results")
    print(json.dumps({"passed": not errors, "errors": errors,
                      "scope": "Public hashes, split coverage, image decisions and metrics. Pixel AUROC requires raw maps."}, indent=2))
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
