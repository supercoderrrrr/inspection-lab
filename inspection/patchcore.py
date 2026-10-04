"""CPU PatchCore fitting and inference through Anomalib."""

import os
from pathlib import Path

import numpy as np
import torch

from inspection.images import pixels, validate_size


def create_model(pretrained=True):
    cache = Path(__file__).resolve().parents[1] / ".cache"
    os.environ.setdefault("HF_HOME", str(cache / "huggingface"))
    os.environ.setdefault("TORCH_HOME", str(cache / "torch"))
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    from anomalib.models.image.patchcore.torch_model import PatchcoreModel

    model = PatchcoreModel(
        backbone="resnet18", layers=["layer2", "layer3"],
        pre_trained=pretrained, num_neighbors=9,
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


class PatchCoreDetector:
    def __init__(self, size=224, ratio=0.01, pretrained=True):
        validate_size(size)
        if not 0 < ratio <= 1:
            raise ValueError("Coreset ratio must be greater than zero and at most one.")
        self.size = size
        self.ratio = ratio
        self.feature_initialization = "imagenet" if pretrained else "random"
        self.model = create_model(pretrained)

    def tensor(self, path):
        values = torch.from_numpy(pixels(path, self.size).transpose(2, 0, 1)).contiguous()
        mean = torch.tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        return (values - mean) / std

    def fit(self, paths, seed=42, batch_size=8):
        if len(paths) < 2 or batch_size < 1:
            raise ValueError("Fitting requires at least two references and a positive batch size.")
        torch.set_num_threads(min(torch.get_num_threads(), 8))
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.model.train()
        self.model.feature_extractor.eval()
        self.model.embedding_store.clear()
        with torch.no_grad():
            for offset in range(0, len(paths), batch_size):
                batch = torch.stack([self.tensor(p) for p in paths[offset:offset + batch_size]])
                self.model(batch)
            candidates = sum(len(batch) for batch in self.model.embedding_store)
            if candidates < 9:
                raise ValueError("Too few candidate patches for nine-neighbor scoring.")
            # Small smoke runs still need enough references for nine-neighbor scoring.
            effective_ratio = max(self.ratio, (9 + 1e-6) / candidates)
            self.model.subsample_embedding(min(effective_ratio, 1.0))
        self.model.eval()
        return self

    @property
    def parameters(self):
        return {
            "backbone": "resnet18", "layers": ["layer2", "layer3"],
            "num_neighbors": 9, "coreset_ratio": self.ratio,
            "memory_bank_patches": len(self.model.memory_bank),
            "feature_initialization": self.feature_initialization,
        }

    def predict(self, path):
        if self.model.memory_bank.numel() == 0:
            raise ValueError("Fit or load PatchCore before prediction.")
        self.model.eval()
        with torch.inference_mode():
            result = self.model(self.tensor(path).unsqueeze(0))
        return float(result.pred_score[0]), result.anomaly_map[0, 0].numpy()

    def save(self, path):
        torch.save({
            "schema_version": 1, "size": self.size, "parameters": self.parameters,
            "state_dict": self.model.state_dict(),
        }, path)

    @classmethod
    def load(cls, path):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            raise ValueError("Unsupported PatchCore checkpoint schema.")
        parameters = payload.get("parameters", {})
        if (parameters.get("backbone") != "resnet18"
                or parameters.get("layers") != ["layer2", "layer3"]
                or parameters.get("num_neighbors") != 9
                or "size" not in payload or "state_dict" not in payload):
            raise ValueError("Unsupported PatchCore checkpoint configuration.")
        model = cls(payload["size"], parameters["coreset_ratio"], pretrained=False)
        model.feature_initialization = parameters["feature_initialization"]
        model.model.load_state_dict(payload["state_dict"])
        model.model.eval()
        if len(model.model.memory_bank) < 9:
            raise ValueError("Saved PatchCore memory bank is too small.")
        return model
