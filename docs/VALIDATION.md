# Local validation

Checked on Windows on 2026-10-04 using Python 3.11.13 and a previously installed CPU runtime
with PyTorch 2.6.0+cpu and Anomalib 2.2.0.

| Check | Result |
|---|---|
| Dataset checker on the local MVTec AD bottle category | 209 normal training images, 20 normal test images, 63 defect test images and 63 matching masks |
| Unit and integration tests | 11 passed |
| Ruff correctness checks | Passed |
| Pixel-template baseline on 16 normal references | Fit, checkpoint reload and single-image inference passed |
| ImageNet-pretrained PatchCore on 16 normal references | Fit, checkpoint reload and single-image inference passed; 125 reference patches retained |

Both real fitting checks used size 224 and seed 42. Pretrained weights were reused from a
local cache with network downloads disabled. The smoke inference image was
`test/broken_large/000.png`; only `train/good` was used for fitting.

Tests cover corrupt data, missing/mismatched annotations, training/test content overlap,
deterministic reference sampling, template response to a synthetic defect, checkpoint
integrity and PatchCore serialization. Random-feature PatchCore tests exercise the pipeline
without assessing detection accuracy. Anomalib emits an internal deprecation warning during
coreset construction; fitting and the checks complete successfully.

This record establishes that the current workflow runs. It is not a held-out accuracy
evaluation or a fresh-environment installation check. The stage exports uncalibrated raw
scores and has no pass/fail threshold. Dataset files, caches, fitted models and generated
predictions stay outside Git history.
