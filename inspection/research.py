"""Run the prespecified PaDiM comparison and fixed-capacity PatchCore ablation."""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone

import pandas as pd

from inspection.paths import ARTIFACTS, ROOT
from inspection.artifacts import atomic_json, seal_run, verify_run
from inspection.doctor import audit_run
from inspection.study import aggregate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="controlled_v4")
    args = parser.parse_args()
    if not args.name.replace("_", "").replace("-", "").isalnum():
        raise ValueError("Invalid study name.")
    output = ARTIFACTS / "research" / args.name
    output.mkdir(parents=True, exist_ok=True)
    plan = {
        "categories": {"bottle": "bottle_cpu_v2", "metal_nut": "metal_nut_cpu_v3"},
        "seeds": [42, 123, 2026], "budgets": "16,64,all", "fixed_bank_patches": 125,
        "methods": ["PaDiM", "PatchCore fixed 125"], "device": "cpu",
        "padim_features": 100, "padim_layers": ["layer1", "layer2", "layer3"],
        "alpha": 0.05, "calibration_count": 30, "image_size": 224,
        "scope": "Clean images; identical splits to prior studies. No tuning against test labels.",
        "selection": "125 equals the smallest prior 1% bank; selected before new results.",
    }
    if (output / "plan.json").exists():
        if json.loads((output / "plan.json").read_text(encoding="utf-8")) != plan:
            raise ValueError("Plan differs; use a new name.")
    else:
        atomic_json(output / "plan.json", plan)
    if (output / "complete.json").exists():
        issues = verify_run(output)
        if issues:
            raise ValueError("; ".join(issues))
        print(f"Already complete: {output}")
        return
    all_rows, names = [], []
    for category, previous_study in plan["categories"].items():
        previous = pd.read_csv(ARTIFACTS / "studies" / previous_study / "runs.csv")
        previous = previous[previous.condition == "clean"].copy()
        previous["category"] = category
        all_rows.extend(previous.to_dict("records"))
        for seed in plan["seeds"]:
            reference = ARTIFACTS / f"{previous_study}_seed{seed}"
            if audit_run(reference):
                raise ValueError(f"Reference run failed audit: {reference}")
            expected_splits = json.loads((reference / "splits.json").read_text(encoding="utf-8"))
            for variant, label in [("padim", "PaDiM"), ("fixed125", "PatchCore fixed 125")]:
                name = f"{args.name}_{category}_{variant}_seed{seed}"
                run = ARTIFACTS / name
                if not (run / "complete.json").exists():
                    if run.exists():
                        raise ValueError(f"Incomplete run preserved: {name}; use a new research name.")
                    command = [sys.executable, "-m", "inspection.experiment", "--category", category,
                               "--seed", str(seed), "--device", "cpu", "--name", name]
                    command += ["--method", "padim"] if variant == "padim" else ["--bank-patches", "125"]
                    print(f"Running {name}", flush=True)
                    with (output / f"{name}.log").open("w", encoding="utf-8") as log:
                        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
                if audit_run(run):
                    raise ValueError(f"Run failed audit: {name}")
                if json.loads((run / "splits.json").read_text(encoding="utf-8")) != expected_splits:
                    raise ValueError("The comparison does not use identical splits.")
                frame = pd.read_csv(run / "summary.csv")
                if variant == "fixed125" and not (frame.memory_patches == 125).all():
                    raise ValueError("Fixed-capacity experiment changed capacity.")
                frame["method"], frame["seed"], frame["category"] = label, seed, category
                all_rows.extend(frame.to_dict("records"))
                names.append(name)
                pd.DataFrame(all_rows).to_csv(output / "runs.csv", index=False)
    frame = pd.DataFrame(all_rows)
    summaries = []
    for category, group in frame.groupby("category"):
        summary = aggregate(group.to_dict("records"))
        summary.insert(0, "category", category)
        summaries.append(summary)
    result = pd.concat(summaries, ignore_index=True)
    result.to_csv(output / "aggregate.csv", index=False)
    lines = ["# Controlled comparison", "", "## Protocol", "",
             "Two categories, three paired seeds, three nested budgets. Every variant uses the exact",
             "same image paths and hashes as the existing PatchCore/pixel-template experiments.",
             "PaDiM: Anomalib 2.2.0, ResNet-18 layer1/2/3, 100 sampled channels, regularized covariance.",
             "PatchCore fixed 125: identical feature extraction and scoring to the existing detector;",
             "exactly 125 coreset patches at every budget. 125 matches the smallest prior 1% bank.",
             "Thresholds are separately calibrated on the same 30 normal images per seed.",
             "These clean-only follow-up experiments were specified after seeing prior results,",
             "before running these variants. They are exploratory, not a preregistered external study.", "",
             "## Results: mean ± sample standard deviation", "",
             "| Category | Method | References | AUROC | Recall | False alarms |",
             "|---|---|---:|---:|---:|---:|"]
    for _, row in result.iterrows():
        lines.append(f"| {row.category} | {row.method} | {int(row.budget)} | "
                     f"{row.image_auroc_mean:.4f} ± {row.image_auroc_std:.4f} | "
                     f"{row.recall_mean:.1%} ± {row.recall_std:.1%} | "
                     f"{row.false_positive_rate_mean:.1%} ± {row.false_positive_rate_std:.1%} |")
    lines += ["", "## Interpretation boundaries", "",
              "PaDiM uses different layers, feature dimensions and covariance statistics; this is a",
              "practical method comparison, not an isolation of scoring distance alone. With fewer",
              "training images than channels, covariance regularization is especially important.",
              "The fixed-bank experiment controls final bank capacity but still changes reference",
              "coverage and the candidate pool used by coreset selection. No guarantee of monotonic",
              "improvement follows. Test sets are reused across seeds; standard deviations describe",
              "split sensitivity. Timings from different sessions are not controlled hardware comparisons.", "",
              "## Reproduce", "", f"`python -m inspection.research --name {args.name}`", "",
              "See plan.json, runs.csv, aggregate.csv and per-run calibration/prediction CSVs."]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    seal_run(output)
    atomic_json(output / "complete.json", {"runs": names, "completed_utc": datetime.now(timezone.utc).isoformat()})
    print(f"Completed: {output}", flush=True)


if __name__ == "__main__":
    main()
