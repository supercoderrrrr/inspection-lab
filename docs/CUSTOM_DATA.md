# Custom datasets and batch inspection

## Train and evaluate

Use a parent directory containing a category folder with this structure:

```text
my-data/
  part_a/
    train/good/*.png
    test/good/*.png
    test/scratch/*.png
    ground_truth/scratch/*_mask.png
```

The evaluator currently requires PNG images, both test classes, and a matching grayscale
mask for every anomalous image. Nonzero mask pixels represent defects. It rejects identical
image bytes across the collection. Independently check for near-duplicate images, frames
from the same acquisition, and other leakage that byte hashes cannot detect.

Provide more normal training images than the calibration count (default 30) plus your
largest requested budget. For a small custom dataset, explicitly choose feasible budgets:

```bash
python -m inspection.experiment --data-root /path/to/my-data --category part_a --budgets 16,all --device cpu --name part_a_v1
```

Thirty normal training images are held out for calibration. Test labels never set the
threshold. Custom paths are saved in local run configuration; redact them before sharing.
Select the new category in the UI when training finishes. This path supports either
`--method patchcore` or `--method padim`. The built-in repeated-seed study commands currently
target the two supported MVTec categories.

## Inspect without labels

Single-file and batch inference accept PNG/JPEG images and do not require annotations:

```bash
python -m inspection.predict --checkpoint artifacts/bottle_cpu_v2_seed42/n179/model.pt --input /path/to/images --output artifacts/batch_001 --overlays
```

The output contains CSV and JSON records, checkpoint/image hashes and optional overlays.
Use a new output directory each time. Invalid files are reported individually; remaining
images are processed and the command exits with status 1 if any image failed. Output must
be outside the input directory. Scores are raw distances; use the checkpoint's category
and acquisition conditions. Uploaded or batch images never update the model.
