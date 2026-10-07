"""Run discovery and explicit integrity checks for local experiment artifacts."""

import json
import csv
import math
from pathlib import Path

from inspection.protocol import file_digest


def atomic_json(path: Path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def seal_run(root: Path):
    files = sorted(p for p in root.rglob("*") if p.is_file()
                   and p.name not in {"integrity.json", "complete.json"})
    atomic_json(root / "integrity.json", {
        "schema_version": 1,
        "files": {p.relative_to(root).as_posix(): file_digest(p) for p in files},
    })


def verify_run(root: Path):
    manifest = root / "integrity.json"
    if not manifest.exists():
        return ["This legacy run has no integrity manifest."]
    errors = []
    for relative, expected in json.loads(manifest.read_text(encoding="utf-8"))["files"].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()):
            errors.append(f"Invalid artifact path: {relative}")
        elif not path.is_file() or file_digest(path) != expected:
            errors.append(f"Missing or modified artifact: {relative}")
    return errors


def completed_runs(root: Path):
    runs, rejected = [], []
    for path in root.iterdir() if root.exists() else []:
        if not path.is_dir() or not (path / "complete.json").exists():
            continue
        try:
            config = json.loads((path / "config.json").read_text(encoding="utf-8"))
            if not config["budgets"] or not all(isinstance(n, int) and n > 0 for n in config["budgets"]):
                raise ValueError("Invalid model budgets")
            required = ["summary.csv", "REPORT.md", "splits.json", "environment.json"]
            required += [f"n{n}/model.pt" for n in config["budgets"]]
            required += [f"n{n}/predictions_clean.csv" for n in config["budgets"]]
            if not all((path / name).is_file() for name in required):
                raise ValueError("Required artifact missing")
            if not all(k in config for k in ["image_size", "backbone", "seed", "category"]):
                raise ValueError("Run configuration is incomplete")
            with (path / "summary.csv").open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            for budget in config["budgets"]:
                matches = [r for r in rows if int(r["budget"]) == budget and r["condition"] == "clean"]
                if len(matches) != 1:
                    raise ValueError("Run summary has missing or duplicate clean rows")
                if not all(math.isfinite(float(matches[0][m])) for m in
                           ["image_auroc", "recall", "false_positive_rate", "latency_median_ms", "threshold"]):
                    raise ValueError("Run summary contains nonfinite metrics")
            runs.append(path)
        except (OSError, KeyError, ValueError, TypeError) as exc:
            rejected.append(f"{path.name}: {exc}")
    return sorted(runs, key=lambda p: (p / "complete.json").stat().st_mtime, reverse=True), rejected
