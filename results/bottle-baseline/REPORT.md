# bottle: baseline

Seed 42; 16 fitted references; 30 normal calibration images; 83 test images.

The threshold was fixed using normal calibration images before test scoring.

| Metric | Value |
|---|---:|
| image_auroc | 0.8865079365079365 |
| average_precision | 0.9623720812996472 |
| precision | 1.0 |
| recall | 0.3492063492063492 |
| normal_false_alarm_rate | 0.0 |
| tn | 20 |
| fp | 0 |
| fn | 41 |
| tp | 22 |

## Incorrect decisions

| Image | True label | Score | Predicted anomaly |
|---|---|---:|---:|
| test/broken_large/000.png | broken_large | 2.937652 | 0 |
| test/broken_large/001.png | broken_large | 2.677120 | 0 |
| test/broken_large/002.png | broken_large | 2.443757 | 0 |
| test/broken_large/003.png | broken_large | 3.201754 | 0 |
| test/broken_large/005.png | broken_large | 2.176302 | 0 |
| test/broken_large/007.png | broken_large | 2.771027 | 0 |
| test/broken_large/008.png | broken_large | 2.640508 | 0 |
| test/broken_large/009.png | broken_large | 1.780430 | 0 |
| test/broken_large/010.png | broken_large | 2.125925 | 0 |
| test/broken_large/011.png | broken_large | 1.756339 | 0 |
| test/broken_large/015.png | broken_large | 2.509020 | 0 |
| test/broken_large/019.png | broken_large | 2.955209 | 0 |

41 incorrect decisions; up to 12 shown. See predictions.csv for all images.

This is one reference budget and one seed on one product. It does not establish
performance across products, industrial operating conditions or repeated runs.

Dataset: MVTec AD, MVTec Software GmbH. No images or fitted weights are included.
