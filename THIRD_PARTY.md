# Attribution and provenance

## Anomalib

- Repository: https://github.com/open-edge-platform/anomalib
- Version: 2.2.0; license: Apache License 2.0 (see upstream and the installed package).
- Use: imported PatchCore, feature extraction, coreset selection, and anomaly maps.
- Citation: Akcay et al., *Anomalib: A Deep Learning Library for Anomaly Detection*, ICIP 2022.

No upstream detector source is copied into the application modules.

## PatchCore

Roth et al., *Towards Total Recall in Industrial Anomaly Detection*, CVPR 2022.
https://arxiv.org/abs/2106.08265
This project defaults to compact ResNet-18, not the paper's full benchmark configuration.

## MVTec AD

- Owner: MVTec Software GmbH.
- Source: https://www.mvtec.com/research-teaching/datasets/mvtec-ad
- Downloads: https://www.mvtec.com/research-teaching/datasets/mvtec-ad/downloads
- License: Creative Commons Attribution-NonCommercial-ShareAlike 4.0.
- Citation: Bergmann et al., *MVTec AD: A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection*, CVPR 2019.
- Use: original bottle and metal_nut images and masks; official test split retained.

The downloader records the archive URL and SHA-256. This documents the retrieved archive;
it is not verified against an independently published official checksum. Dataset terms are
separate from code licenses. Do not assume commercial usage rights.

## Weights and dependencies

ImageNet-pretrained ResNet-18 is resolved through Anomalib/timm. Model weights and dependencies
retain their respective terms. PyTorch, torchvision, timm, scikit-learn, Streamlit, Pillow,
NumPy, pandas, Matplotlib, and Plotly are dependencies. Tested versions are recorded in the
environment lock and experiment metadata. Application development is AI-assisted.

## PaDiM

- Implementation: imported from Anomalib 2.2.0 (Apache-2.0).
- Paper: Defard et al., *PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization*.
- Reference: https://arxiv.org/abs/2011.08785
- Configuration: ResNet-18, layer1/layer2/layer3, 100 sampled channels, Anomalib covariance regularization.

## Distribution

The root MIT license covers project-authored code. It does not relicense Anomalib,
pretrained weights, MVTec images, or image-containing screenshots. The separate inference
archive retains model provenance. Screenshots in docs/assets containing MVTec imagery
remain under the dataset's CC BY-NC-SA 4.0 terms; attribution appears beside those assets.
