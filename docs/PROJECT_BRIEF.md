# Inspection Lab

## Applied visual inspection with auditable experiments

**Problem.** Detect visual defects using normal references, while exposing the operating
threshold, failure cases and experimental evidence behind an inspection decision.

**Implementation.** Python/PyTorch, imported Anomalib PatchCore and PaDiM, an independent
pixel-template baseline, a Streamlit interface and batch JSON/CSV inference. Two MVTec AD
categories are fitted separately. Each uses three paired seeds, nested reference budgets
and 30 disjoint normal calibration images. A fixed-capacity PatchCore experiment controls
the number of reference patches. Test labels never set operating thresholds.

**Largest-budget mean results.** Seeds 42, 123, 2026; reused official test sets.

| Category | Method | AUROC | Recall | False alarms |
|---|---|---:|---:|---:|
| bottle | PaDiM | 0.9984 | 98.4% | 5.0% |
| bottle | PatchCore | 1.0000 | 100.0% | 13.3% |
| bottle | PatchCore fixed 125 | 0.9860 | 96.8% | 8.3% |
| bottle | Pixel template | 0.9011 | 53.4% | 0.0% |
| metal_nut | PaDiM | 0.9573 | 76.0% | 7.6% |
| metal_nut | PatchCore | 0.9891 | 92.5% | 4.5% |
| metal_nut | PatchCore fixed 125 | 0.8749 | 62.7% | 9.1% |
| metal_nut | Pixel template | 0.4904 | 25.1% | 1.5% |

**Project-specific work.** Experimental orchestration, normal-only calibration, controlled
comparisons, custom data validation, failure inspection, source/data/checkpoint provenance,
public metric verification, automated tests and portable release packaging.

**Interpretation.** Compact PatchCore performs well here, but high AUROC does not ensure low
false alarms at a normal-calibrated threshold. The fixed-bank experiment identifies memory
capacity as an important contributor. Twenty/twenty-two normal test examples limit precision.
These are public-benchmark observations, not industrial acceptance certification.

**Ownership.** Detectors are imported from Anomalib; development is AI-assisted. The project
claims applied engineering and evaluation contributions rather than a new detection algorithm.

See the README, technical report, controlled comparison and public results for reproduction.
