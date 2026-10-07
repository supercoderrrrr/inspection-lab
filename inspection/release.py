"""Assemble public evidence, source archive and two compact inference bundles."""

import argparse
import json
import shutil
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from inspection.paths import ROOT, ARTIFACTS
from inspection.artifacts import atomic_json, seal_run, verify_run
from inspection.doctor import audit_run
from inspection.evidence import verify_evidence
from inspection.protocol import file_digest


PUBLIC_DOCS = ["CUSTOM_DATA.md", "PROJECT_BRIEF.md", "TECHNICAL_REPORT.md", "REPRODUCIBILITY.md", "RELEASE_NOTES.md", "VALIDATION.md", "PUBLISHING.md", "STAGED_VALIDATION.md", "INITIAL_COMPARISON.md"]


def write_archive(destination, files, base):
    files = sorted(set(files))
    manifest_name = "MODEL_MANIFEST.json" if "models" in destination.name else "SOURCE_MANIFEST.json"
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(base).as_posix())
        archive.writestr(manifest_name, json.dumps({
            "files": {p.relative_to(base).as_posix(): file_digest(p) for p in files}}, indent=2))
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None:
            raise ValueError("Archive CRC validation failed.")
        import hashlib
        manifest = json.loads(archive.read(manifest_name))
        if not all(hashlib.sha256(archive.read(p)).hexdigest() == h for p, h in manifest["files"].items()):
            raise ValueError("Archive hash validation failed.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, default=ARTIFACTS,
                        help="Directory containing the sealed experiments to distribute")
    args = parser.parse_args(argv)
    source_root = args.experiment_root.resolve()
    public = ROOT / "results"
    public.mkdir(exist_ok=True)
    source_studies = [source_root / "studies/bottle_cpu_v2", source_root / "studies/metal_nut_cpu_v3",
                      source_root / "research/controlled_v4"]
    names = []
    for study in source_studies:
        complete = json.loads((study / "complete.json").read_text(encoding="utf-8"))
        names.extend(complete["runs"])
        destination = public / study.name
        destination.mkdir(exist_ok=True)
        for filename in ["plan.json", "runs.csv", "aggregate.csv", "REPORT.md", "complete.json"]:
            shutil.copy2(study / filename, destination / filename)
    for name in names:
        run = source_root / name
        issues = audit_run(run)
        if issues:
            raise ValueError("; ".join(issues))
        for path in run.rglob("*"):
            if (path.suffix not in {".csv", ".json", ".md"} and path.name != "budget_comparison.png") or path.name in {"integrity.json", "complete.json"}:
                continue
            target = public / "runs" / name / path.relative_to(run)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    frame = pd.read_csv(public / "controlled_v4/aggregate.csv")
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout="constrained")
    for row_index, (category, group) in enumerate(frame.groupby("category")):
        for col, metric in enumerate(["image_auroc", "recall", "false_positive_rate"]):
            ax = axes[row_index, col]
            for method, subset in group.groupby("method"):
                ax.errorbar(subset.budget, subset[f"{metric}_mean"], yerr=subset[f"{metric}_std"],
                            marker="o", capsize=3, label=method)
            ax.set(title=f"{category.replace('_', ' ').title()} · {metric.replace('_', ' ')}",
                   xlabel="Normal reference images", ylim=(-0.05, 1.05))
            ax.grid(alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4)
    fig.savefig(public / "comparison.png", dpi=160)
    plt.close(fig)
    (public / "README.md").write_text("""# Recorded evidence

The comparison figure shows mean ± sample standard deviation across three paired seeds.
Read controlled_v4/REPORT.md for the PaDiM comparison and fixed-capacity ablation.
The original two studies also include clean, dim and blur conditions and a pixel baseline.

`runs/` preserves configuration, split hashes, calibration scores and every recorded image
prediction for these studies. Model weights, dataset images and raw pixel maps are excluded.
Pixel AUROC cannot be independently recalculated from these image-level CSVs.

Run `python -m inspection.evidence` to verify this public subset and recalculate image
metrics. The manifest covers the distributed subset; checkpoint hashes refer to separately
stored weights. Source hashes in each environment.json identify code at experiment time.
Historical benchmark timings were recorded under varying ordinary machine load.

The earlier single-budget comparison is retained in `bottle-baseline/` and
`bottle-patchcore/`; see `../docs/INITIAL_COMPARISON.md`. Its separate manifests
can be checked with `python -m inspection verify-results --results <directory>`.
The larger studies were developed in the earlier workspace before this staged
repository was assembled. Their recorded timestamps and source hashes are retained;
this release packages that evidence rather than claiming newly executed studies.
""", encoding="utf-8")
    atomic_json(public / "MANIFEST.json", {"files": {
        p.relative_to(public).as_posix(): file_digest(p) for p in public.rglob("*")
        if p.is_file() and p != public / "MANIFEST.json"}})
    issues = verify_evidence(public)
    if issues:
        raise ValueError("; ".join(issues))

    release = ARTIFACTS / "release-v0.4.2"
    release.mkdir(exist_ok=True)
    bundle = release / "inference"
    for category, original, budget in [("bottle", "bottle_cpu_v2_seed42", 179),
                                        ("metal_nut", "metal_nut_cpu_v3_seed42", 190)]:
        source = source_root / original
        target = bundle / "artifacts" / f"{category}_demo_v4"
        target.mkdir(parents=True, exist_ok=True)
        config = json.loads((source / "config.json").read_text(encoding="utf-8"))
        config.update(budgets=[budget], origin_run=original, method="patchcore")
        atomic_json(target / "config.json", config)
        for filename in ["splits.json", "environment.json"]:
            shutil.copy2(source / filename, target / filename)
        folder = target / f"n{budget}"
        folder.mkdir(exist_ok=True)
        for path in (source / f"n{budget}").iterdir():
            if path.is_file():
                shutil.copy2(path, folder / path.name)
        summary = pd.read_csv(source / "summary.csv")
        summary[summary.budget == budget].to_csv(target / "summary.csv", index=False)
        (target / "REPORT.md").write_text(
            f"# {category}: inference snapshot\n\nThis is the prespecified seed42, largest-budget model from `{original}`.\n"
            "Checkpoint bytes are unchanged. The bundle includes this budget only; see the public results directory\n"
            "for the complete study, failures and other seeds. Data images are downloaded separately.\n"
            "Model features and weights derive from Anomalib/timm/ImageNet; their terms remain separate.\n",
            encoding="utf-8")
        seal_run(target)
        atomic_json(target / "complete.json", {"default_budget": budget, "origin_run": original})
        if verify_run(target) or audit_run(target):
            raise ValueError("Inference bundle audit failed.")
    (bundle / "artifacts/latest.txt").write_text("bottle_demo_v4", encoding="utf-8")
    shutil.copy2(ROOT / "THIRD_PARTY.md", bundle / "MODEL_PROVENANCE.md")
    write_archive(release / "inspection-lab-models-v0.4.2.zip", [p for p in bundle.rglob("*") if p.is_file()], bundle)

    files = [ROOT / filename for filename in ["README.md", "LICENSE", "THIRD_PARTY.md", "MODEL_CARD.md",
             "CHANGELOG.md", "CONTRIBUTING.md", "pyproject.toml", "requirements.txt", "requirements-lock.txt",
             "requirements-cpu-lock.txt", "pytest.ini", ".gitignore", ".gitattributes", "app.py", "launch.ps1", "Launch.cmd", "package.json", ".streamlit/config.toml"]]
    files += [p for folder in ["inspection", "tests", ".github", "results", "docs/assets"]
              for p in (ROOT / folder).rglob("*") if p.is_file() and p.suffix in {".py", ".cjs", ".json", ".csv", ".md", ".png", ".yml", ".toml"}]
    files += [ROOT / "docs" / name for name in PUBLIC_DOCS]
    missing = [str(p) for p in files if not p.is_file()]
    if missing:
        raise ValueError(f"Release files missing: {missing}")
    write_archive(release / "inspection-lab-source-v0.4.2.zip", files, ROOT)
    archives = sorted(release.glob("*.zip"))
    (release / "SHA256SUMS.txt").write_text("\n".join(f"{file_digest(p)}  {p.name}" for p in archives) + "\n", encoding="utf-8")
    print(json.dumps({"archives": [str(p) for p in archives], "public_runs": len(names),
                      "public_evidence_verified": True}, indent=2))


if __name__ == "__main__":
    main()
