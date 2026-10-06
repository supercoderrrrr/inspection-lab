# Attribution

## MVTec AD

- Owner: MVTec Software GmbH.
- Source: https://www.mvtec.com/research-teaching/datasets/mvtec-ad
- License: Creative Commons Attribution-NonCommercial-ShareAlike 4.0.
- Citation: Bergmann et al., *MVTec AD: A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection*, CVPR 2019.

No dataset images are distributed in this repository. The project code's MIT license does
not replace dataset terms. Software development is AI-assisted.

## Anomalib and PatchCore

- Implementation: https://github.com/open-edge-platform/anomalib, version 2.2.0, Apache-2.0.
- Library citation: Akcay et al., *Anomalib: A Deep Learning Library for Anomaly Detection*, ICIP 2022.
- Algorithm: Roth et al., *Towards Total Recall in Industrial Anomaly Detection*, CVPR 2022.
- Paper: https://arxiv.org/abs/2106.08265

The application imports Anomalib's detector and coreset implementation. No upstream model
source is copied into the package. The ResNet-18 configuration is smaller than the paper's
full benchmark configuration. Pretrained weights resolved through timm retain upstream terms.
The root MIT license covers project-authored code.

## Interface screenshot

The screenshot in docs/assets contains MVTec AD imagery and annotations. It retains the
dataset's CC BY-NC-SA 4.0 terms, with attribution beside the image and in docs/assets/README.md.
The screenshot is a capture of actual model output; display colors are not segmentation labels.
