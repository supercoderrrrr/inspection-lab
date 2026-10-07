# Release validation — 0.4.2

Validated on 2026-10-07 after integrating the preserved complete implementation with
the staged Git repository. Runtime: Windows, Python 3.11.13, Torch 2.6.0+cpu.

| Check | Observed result |
|---|---|
| Combined unit/integration and UI state suite | 59 passed, 1 expected skip |
| Independently extracted source archive | 59 passed, 1 expected skip; Ruff and evidence verification passed |
| CPU dependency compatibility and wheel build | Passed |
| Ruff correctness checks | Passed |
| Public evidence verification | Passed for 18 full runs and both initial method records |
| Installed model and dataset audit | Passed for both release models and the local PaDiM run |
| Browser acceptance | Passed for inference, uploads, exports, categories, PaDiM and benchmark tabs |
| Extracted models with source and no local dataset | Upload UI passed; real batch score and decision matched the recorded bottle example |
| Release packaging | Archive CRC and per-file SHA-256 checked by the packager |

The skip requires `artifacts/bottle_verified`, a historical checkpoint excluded from
source. Synthetic fitting and checkpoint round trips for both detectors still run offline.
One deprecation warning comes from the Anomalib coreset API used by the initial adapter.

The CI workflow checks Windows and Linux using CPU Torch, without benchmark or pretrained
downloads. Actual hosted status is on the
[Actions page](https://github.com/supercoderrrrr/inspection-lab/actions/workflows/ci.yml).
Browser checks use real local models and datasets and run separately from source-only CI.
No benchmark training was repeated when assembling this release; historical timestamps,
configuration and source hashes are retained. Earlier staged checks are in
[STAGED_VALIDATION.md](STAGED_VALIDATION.md).

## Earlier workspace validation — display snapshot 0.4.1

Validated locally on Windows after the inspection interface and heatmap display update.

| Check | Observed result |
|---|---|
| Working environment unit/integration suite | 31 passed |
| Extracted source ZIP in the separate CPU environment | 30 passed, 1 expected skip |
| Ruff correctness checks on extracted source | Passed |
| Public evidence verification on extracted source | Passed |
| Browser workflow | Passed, including uploads, category/model changes, display controls and downloads |

The new display test verifies that response weighting reduces low-response tint, preserves
the strongest response and leaves the raw anomaly map unchanged. Browser checks verify that
changing the display scale leaves the image score unchanged. These are display checks;
the update does not establish an improvement in defect localization accuracy.

The source-only skip still concerns the historical checkpoint described below. The CPU
environment was reused from the 0.4.0 clean installation; the 0.4.1 source ZIP was extracted
into a new directory. Model fitting and the recorded experiments were not repeated for
this display update. Hosted GitHub Actions have not yet run.

## Earlier workspace validation — comparison snapshot 0.4.0

Validated locally on Windows with Python 3.11.13.

| Check | Observed result |
|---|---|
| Working environment unit/integration suite | 30 passed |
| Source ZIP extracted to an independent directory, fresh CPU environment | 29 passed, 1 expected skip |
| CPU dependency compatibility | Passed |
| Ruff correctness checks | Passed |
| Public evidence hashes and image-metric recomputation | Passed for 18 published runs |
| Full local artifact and dataset audit | Passed |
| Browser workflow | Passed, including PaDiM inference and controlled-comparison tab |
| Release archive CRC and per-file SHA-256 checks | Passed |
| Extracted model bundle, real defect image, fresh CPU runtime | Passed; score and decision match the original run |

The skipped test requires a historical local checkpoint deliberately excluded from the
source release. Synthetic PatchCore and PaDiM fit/serialization tests still run in the fresh
environment without network downloads. Browser checks include category/model identity,
upload validation, stale-result invalidation, display-scale score stability and downloads.

These checks verify the supplied implementation and evidence. Linux is configured in the
GitHub Actions matrix but was not executed on this Windows host. Hosted CI status remains
pending until the repository is pushed. Full local XML/JSON test logs are retained under
artifacts/; public image-level evidence is under results/.

The actual extracted source plus model bundle was also checked without any dataset images: the app, public benchmarks and upload mode passed.
