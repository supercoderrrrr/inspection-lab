"""Prespecified repeated-seed comparison of PatchCore and a pixel-template baseline."""

from inspection.paths import ARTIFACTS, DATA, ROOT

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from inspection.artifacts import atomic_json, verify_run
from inspection.baseline import PixelTemplate
from inspection.data import CATEGORIES, validate
from inspection.protocol import calibrated_threshold, classification_metrics, file_digest


def baseline_run(run):
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    splits = json.loads((run / "splits.json").read_text(encoding="utf-8"))
    paths = {k: [DATA / item["path"] for item in v] for k, v in splits.items()}
    from inspection.experiment import pixel_labels
    masks = pixel_labels(paths["test"], config["image_size"], DATA / config["category"])
    labels = [int(p.parent.name != "good") for p in paths["test"]]
    output = run / "baseline"
    output.mkdir(exist_ok=True)
    results = []
    for budget in config["budgets"]:
        model = PixelTemplate(config["image_size"])
        start = time.perf_counter()
        model.fit(paths["training_pool"][:budget])
        fit_seconds = time.perf_counter() - start
        scores, _, _ = model.evaluate(paths["calibration"])
        threshold = calibrated_threshold(scores, config["alpha"])
        np.savez_compressed(output / f"n{budget}.npz", mean=model.mean, std=model.std,
                            threshold=threshold, size=model.size)
        pd.DataFrame({"path": [p.relative_to(DATA).as_posix() for p in paths["calibration"]],
                      "score": scores}).to_csv(output / f"n{budget}_calibration.csv", index=False)
        conditions = ["clean", "dim", "blur"] if config["robustness"] and budget == max(config["budgets"]) else ["clean"]
        for condition in conditions:
            scores, maps, latency = model.evaluate(paths["test"], condition)
            metrics = classification_metrics(labels, scores, threshold)
            metrics.update({"budget": budget, "condition": condition, "threshold": threshold,
                            "pixel_auroc": float(roc_auc_score(masks.ravel(), maps.ravel())),
                            "latency_median_ms": float(np.median(latency)),
                            "fit_seconds": fit_seconds, "seed": config["seed"], "method": "Pixel template"})
            results.append(metrics)
            pd.DataFrame({"path": [p.relative_to(DATA).as_posix() for p in paths["test"]],
                          "label": labels, "score": scores, "prediction": (scores > threshold).astype(int),
                          "latency_ms": latency}).to_csv(output / f"n{budget}_{condition}.csv", index=False)
    pd.DataFrame(results).to_csv(output / "summary.csv", index=False)
    return results


def aggregate(rows):
    frame = pd.DataFrame(rows)
    metrics = ["image_auroc", "pixel_auroc", "recall", "false_positive_rate", "latency_median_ms"]
    grouped = frame.groupby(["method", "budget", "condition"])[metrics].agg(["mean", "std", "min", "max"])
    grouped.columns = ["_".join(c) for c in grouped.columns]
    return grouped.reset_index()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", choices=list(CATEGORIES), default="bottle")
    parser.add_argument("--name", default="bottle_cpu_v2")
    parser.add_argument("--seeds", default="42,123,2026")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    if not args.name.replace("_", "").replace("-", "").isalnum():
        raise ValueError("Invalid study name.")
    seeds = [int(s) for s in args.seeds.split(",")]
    if len(set(seeds)) != len(seeds) or len(seeds) < 2 or min(seeds) < 0:
        raise ValueError("Use at least two distinct nonnegative seeds.")
    counts = validate(DATA / args.category)
    largest = counts["train_good"] - 30
    output = ARTIFACTS / "studies" / args.name
    output.mkdir(parents=True, exist_ok=True)
    plan = {"schema_version": 1, "seeds": seeds, "budgets": [16, 64, largest], "device": args.device,
            "category": args.category, "calibration_count": 30, "alpha": 0.05,
            "default_seed": seeds[0], "default_budget": largest,
            "baseline": {"std_floor": 0.05, "blur_sigma": 2, "score_quantile": 0.99},
            "selection": "First prespecified seed and largest budget; not selected by test metrics."}
    if (output / "plan.json").exists():
        if json.loads((output / "plan.json").read_text()) != plan:
            raise ValueError("Existing study has a different plan. Use a new study name.")
    else:
        atomic_json(output / "plan.json", plan)
    all_rows = []
    run_names = []
    for seed in seeds:
        run_name = f"{args.name}_seed{seed}"
        run = ARTIFACTS / run_name
        if not (run / "complete.json").exists():
            if run.exists():
                raise ValueError(f"Incomplete run exists: {run}. Preserve it and choose a new study name.")
            print(f"Starting PatchCore seed {seed}", flush=True)
            with (output / f"seed{seed}.log").open("w", encoding="utf-8") as log:
                subprocess.run([sys.executable, "-m", "inspection.experiment", "--seed", str(seed),
                                "--category", args.category, "--robustness", "--name", run_name, "--device", args.device], cwd=ROOT, stdout=log,
                               stderr=subprocess.STDOUT, check=True)
        issues = verify_run(run)
        if issues:
            raise ValueError("; ".join(issues))
        patch_rows = pd.read_csv(run / "summary.csv")
        patch_rows["method"] = "PatchCore"
        patch_rows["seed"] = seed
        all_rows.extend(patch_rows.to_dict("records"))
        print(f"Evaluating pixel-template baseline, seed {seed}", flush=True)
        all_rows.extend(baseline_run(run))
        run_names.append(run_name)
        pd.DataFrame(all_rows).to_csv(output / "runs.csv", index=False)
    aggregated = aggregate(all_rows)
    aggregated.to_csv(output / "aggregate.csv", index=False)
    clean = aggregated[aggregated.condition == "clean"]
    lines = ["# Repeated-seed benchmark", "", "## Prespecified protocol", "",
             f"Seeds: {seeds}. Category: {args.category}. Normal-reference budgets: 16, 64, {largest}. Both methods use identical",
             "training/calibration/test images per seed. Thresholds use only 30 normal calibration images.",
             "The fixed official test set is reused; seed repetitions are not independent test datasets.", "",
             "## Results (mean ± sample standard deviation across seeds)", "",
             "| Method | References | Image AUROC | Recall | False-positive rate |",
             "|---|---:|---:|---:|---:|"]
    for _, r in clean.iterrows():
        lines.append(f"| {r.method} | {r.budget} | {r.image_auroc_mean:.4f} ± {r.image_auroc_std:.4f} | "
                     f"{r.recall_mean:.4f} ± {r.recall_std:.4f} | {r.false_positive_rate_mean:.4f} ± {r.false_positive_rate_std:.4f} |")
    lines.extend(["", "## Baseline definition", "",
        "The pixel-template baseline learns a mean and sample standard deviation at each RGB pixel.",
        "Standard deviations are floored at 0.05 on [0,1] intensities. The anomaly map is the",
        "Gaussian-smoothed (sigma 2) root-mean-square standardized RGB residual; the image score",
        "is its 99th percentile. These choices were fixed before running this study. The baseline",
        f"uses CPU NumPy/SciPy; PatchCore device selection is {args.device}. Timings reflect the recorded",
        "runtime conditions, not an isolated hardware benchmark. Neither method is fine-tuned on defects.", "",
        "## Limitations", "", "This comparison tests one simple baseline, not all competing anomaly detectors.",
        f"The same {counts['test']} test images are reused across seeds. Standard deviations describe sensitivity to",
        "normal-reference sampling and calibration, not population confidence intervals or significance.",
        "A larger PatchCore budget also increases memory-bank size. Pixel-template performance depends",
        "on alignment. Synthetic perturbations are not real deployment shifts. No operating threshold",
        "or default model was selected to improve these observed test results.", "",
        "## Reproduce", "", f"`python -m inspection.study --category {args.category} --name {args.name} --seeds {args.seeds} --device {args.device}`", "",
        "The plan, per-seed logs, raw predictions, calibrated thresholds, and aggregate CSV are retained.",
        "See THIRD_PARTY.md for upstream attribution. Application and study orchestration are AI-assisted."])
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    from inspection.artifacts import seal_run
    for run_name in run_names:
        seal_run(ARTIFACTS / run_name)
    atomic_json(output / "complete.json", {"runs": run_names, "completed_utc": datetime.now(timezone.utc).isoformat(),
               "source_sha256": file_digest(Path(__file__)), "default_run": run_names[0]})
    (ARTIFACTS / "latest.txt").write_text(run_names[0], encoding="utf-8")
    print(f"Completed study: {output}", flush=True)


if __name__ == "__main__":
    main()
