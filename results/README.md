# Recorded evidence

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
