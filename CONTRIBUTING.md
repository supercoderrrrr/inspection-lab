# Contributing

Use Python 3.11 and install the editable project with the `dev` extra. Run
`python -m ruff check .` and `python -m pytest -q` before submitting a change.
Tests must not download benchmark images or pretrained weights. The optional saved-run
test skips when local artifacts are absent. New protocol or model changes need tests that
check behavior and compatibility, not assertions that merely repeat implementation details.

Keep fitting, calibration and test data disjoint. Write experiment plans before executing
new comparisons. Never select a threshold or a default checkpoint using test metrics.
Preserve historical results and create new run names after method changes. Explain upstream
reuse and AI assistance accurately. Keep dataset files and checkpoints outside Git history.

For a bug report, include the command, Python version, operating system, error and minimal
reproduction. Remove private paths and images before posting publicly.
