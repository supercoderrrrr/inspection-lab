# Reproducibility and verification

## Verification levels

1. **Source tests:** `python -m pytest -q`. No pretrained weights or benchmark downloads.
2. **Public evidence:** `python -m inspection.evidence`. Verify distributed hashes, split
   coverage, thresholds, decisions and image metrics from the committed CSVs.
3. **Local audit:** `python -m inspection.doctor --verify-data`. Also verify complete local
   artifacts and dataset hashes. Requires downloaded data and local models.
4. **Re-run training:** use README study commands. Exact bits can differ across hardware,
   library versions and CPU/CUDA kernels; compare scores with tolerances and inspect decisions.

Pixel AUROC requires anomaly maps and masks. It is recorded in the public tables but is not
independently recomputed by the image-CSV verifier.

## Environments

Python 3.11 is supported. `pyproject.toml` pins direct dependencies; `requirements-lock.txt`
records the original Windows CUDA environment. `requirements-cpu-lock.txt` records a separately
tested Windows CPU environment. Platform-specific snapshots are not universal lockfiles.
For the exact Windows CPU snapshot, install CPU Torch wheels first, then the CPU requirements
file and the editable project. Do not upgrade detector dependencies without rerunning tests.

Set `INSPECTION_HOME` to an absolute workspace if needed. Otherwise editable installation
uses the repository root. `INSPECTION_DATA_ROOT` overrides the MVTec data directory;
`INSPECTION_MODEL_ROOT` overrides the artifact directory. Custom `--data-root` is the
parent of category folders. The server and browser test must share the data setting.

## Browser checks

Start the app with both categories and the model bundle present, plus the
`controlled_v4_metal_nut_padim_seed42` run. Public study tables supply benchmark views.
Install Node.js and run
`npm install`, `npx playwright install chromium`, then `npm run test:browser`.
Set `BROWSER_CHANNEL=msedge` to use an installed Edge browser instead. Checks cover inference,
category changes, result persistence, numeric-score stability under display changes,
invalid uploads, downloads and integrity verification. This suite uses real local
models and is separate from the download-free CPU CI workflow.

## Release artifacts

- Source archive: code, docs, public evidence and screenshots.
- Model archive: two seed42 largest-budget PatchCore checkpoints with metadata, without
  dataset images. Checkpoint bytes match original runs; bundle run names end in `_demo_v4`.
- SHA256SUMS.txt and internal manifests: archive and content checksums.

Extract models into the source checkout. Load only trusted checkpoints. Hashes detect changes;
they do not authenticate a publisher. Run `python -m inspection.release` after all studies
finish to rebuild local release artifacts. `--experiment-root` can select another sealed
experiment workspace. Earlier CSV evidence remains under `results/bottle-baseline` and
`results/bottle-patchcore` and uses the original `inspection verify-results` command.

GitHub Actions runs Windows and Linux CPU tests. See the repository Actions page for
hosted status and `docs/VALIDATION.md` for the validation record.
