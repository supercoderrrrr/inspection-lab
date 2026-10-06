# bottle: patchcore

Seed 42; 16 fitted references; 30 normal calibration images; 83 test images.

The threshold was fixed using normal calibration images before test scoring.

| Metric | Value |
|---|---:|
| image_auroc | 0.9801587301587301 |
| average_precision | 0.994990623474709 |
| precision | 0.9838709677419355 |
| recall | 0.9682539682539683 |
| normal_false_alarm_rate | 0.05 |
| tn | 19 |
| fp | 1 |
| fn | 2 |
| tp | 61 |

## Incorrect decisions

| Image | True label | Score | Predicted anomaly |
|---|---|---:|---:|
| test/contamination/003.png | contamination | 12.599132 | 0 |
| test/contamination/019.png | contamination | 11.970406 | 0 |
| test/good/006.png | good | 13.463331 | 1 |

3 incorrect decisions; up to 12 shown. See predictions.csv for all images.

This is one reference budget and one seed on one product. It does not establish
performance across products, industrial operating conditions or repeated runs.

Dataset: MVTec AD, MVTec Software GmbH. No images or fitted weights are included.
