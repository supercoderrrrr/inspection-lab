"""Adapters around Anomalib PatchCore and PaDiM with explicit preprocessing."""

from inspection.paths import ROOT  # Configure caches before importing model libraries.

import numpy as np
import json
from pathlib import Path
import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from torchvision import transforms
from inspection.protocol import file_digest


def device_name():
    return "cuda" if torch.cuda.is_available() else "cpu"


def create_model(backbone="resnet18", pretrained=True, method="patchcore"):
    if method == "padim":
        from anomalib.models.image.padim.torch_model import PadimModel
        return PadimModel(backbone=backbone, layers=["layer1", "layer2", "layer3"],
                          pre_trained=pretrained, n_features=100)
    if method != "patchcore":
        raise ValueError(f"Unknown detector: {method}")
    from anomalib.models.image.patchcore.torch_model import PatchcoreModel

    return PatchcoreModel(layers=["layer2", "layer3"], backbone=backbone,
                          pre_trained=pretrained, num_neighbors=9)


def prepare(image: Image.Image, size: int, condition="clean"):
    image = ImageOps.exif_transpose(image).convert("RGB")
    if condition == "dim":
        image = ImageEnhance.Brightness(image).enhance(0.7)
    elif condition == "blur":
        image = image.filter(ImageFilter.GaussianBlur(1.5))
    elif condition != "clean":
        raise ValueError(f"Unknown condition: {condition}")
    transform = transforms.Compose([
        transforms.Resize((size, size), antialias=True), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return transform(image)


def image_batch(paths, size, condition="clean"):
    images = []
    for path in paths:
        with Image.open(path) as image:
            images.append(prepare(image, size, condition))
    return torch.stack(images)


def synchronize(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def load_model(checkpoint, device=None):
    checkpoint = Path(checkpoint)
    metadata = checkpoint.with_name("checkpoint.json")
    if metadata.exists():
        expected = json.loads(metadata.read_text(encoding="utf-8"))["sha256"]
        if file_digest(checkpoint) != expected:
            raise ValueError("Checkpoint integrity check failed. Restore or retrain this model.")
    device = device or device_name()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = create_model(payload["config"]["backbone"], pretrained=False, method=payload["config"].get("method", "patchcore"))
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    return model, payload, device


def predict_image(model, image, size, device):
    with torch.inference_mode():
        output = model(prepare(image, size).unsqueeze(0).to(device))
    return float(output.pred_score[0].cpu()), output.anomaly_map[0, 0].cpu().numpy()


def overlay(image, anomaly_map, heatmap_max, opacity=0.48, response_weighted=False):
    from matplotlib import colormaps

    if not 0 <= opacity <= 1:
        raise ValueError("Opacity must be between zero and one.")
    image = ImageOps.exif_transpose(image).convert("RGB").resize(
        (anomaly_map.shape[1], anomaly_map.shape[0]))
    scaled = np.clip(anomaly_map / max(float(heatmap_max), 1e-8), 0, 1)
    heat = (colormaps["inferno"](scaled)[..., :3] * 255).astype(np.uint8)
    if not response_weighted:
        return Image.blend(image, Image.fromarray(heat), opacity)
    # Display weighting only: keep low-response pixels close to the input image.
    alpha = (opacity * scaled ** 3)[..., None]
    blended = np.asarray(image, dtype=np.float32) * (1 - alpha) + heat * alpha
    return Image.fromarray(np.rint(blended).clip(0, 255).astype(np.uint8))


def fit_reference_model(model, paths, size, device, batch_size=8, ratio=0.01, bank_patches=None, method="patchcore"):
    """Fit imported detectors; an optional exact coreset size controls bank capacity."""
    model.train()
    model.feature_extractor.eval()
    with torch.no_grad():
        for offset in range(0, len(paths), batch_size):
            model(image_batch(paths[offset:offset + batch_size], size).to(device))
        if method == "padim":
            if bank_patches is not None:
                raise ValueError("A fixed patch bank applies only to PatchCore.")
            model.fit()
        elif bank_patches is None:
            model.subsample_embedding(ratio)
        else:
            from anomalib.models.components.sampling import KCenterGreedy
            embeddings = torch.vstack(model.embedding_store)
            model.embedding_store.clear()
            if not 9 <= bank_patches <= len(embeddings):
                raise ValueError("Fixed bank must contain between 9 and the available number of patches.")
            sampler = KCenterGreedy(embedding=embeddings, sampling_ratio=bank_patches / len(embeddings))
            sampler.coreset_size = bank_patches
            model.memory_bank = sampler.sample_coreset()
            if len(model.memory_bank) != bank_patches:
                raise RuntimeError("The coreset does not match the requested fixed capacity.")
    model.eval()
