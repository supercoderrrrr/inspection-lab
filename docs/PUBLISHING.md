# Publish to GitHub

Repository: https://github.com/supercoderrrrr/inspection-lab

The repository contains code, docs, tests, CI, screenshots and public evidence.
Environments, caches, dataset images, model checkpoints and release archives are ignored.
Inference models are distributed as release assets rather than through Git history.

## Prepare a release

Run tests and both evidence checks from README.md. Build assets with
`python -m inspection.release` after completing the documented studies locally.
`--experiment-root <directory>` accepts an existing sealed experiment workspace.
The packager audits runs, preserves checkpoint bytes, seals the bundles and verifies
archive CRC and per-file SHA-256 hashes. Inspect `git diff --cached` before committing.

For a versioned release, attach these separately:

- `inspection-lab-models-v0.4.2.zip`
- `SHA256SUMS.txt`
- Optionally the curated source ZIP, alongside GitHub's generated source archives.

Use `docs/RELEASE_NOTES.md` as the release description with GitHub CLI's `--notes-file`.
Push the commit and version tag, then attach the assets from `artifacts/release-v0.4.2/`.
Inspect both CPU jobs on the Actions page. Validation results are in `docs/VALIDATION.md`.

The root license covers project-authored code; dataset-derived screenshots and upstream
weights have separate terms documented in THIRD_PARTY.md. Keep those notices with the release.
