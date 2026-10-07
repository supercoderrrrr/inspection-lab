"""Run nested training-budget experiments with held-out normal calibration."""

from inspection.paths import ARTIFACTS, DATA, ROOT

import argparse
import gc
import importlib.metadata
import json
import platform
import random
from pathlib import Path
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score

from inspection.data import validate, CATEGORIES, validate_custom
from inspection.model import create_model, device_name, image_batch, overlay, synchronize, fit_reference_model
from inspection.protocol import calibrated_threshold, classification_metrics, file_digest, split_normal
from inspection.artifacts import atomic_json, seal_run


def write_json(path, value):
    atomic_json(path, value)


def evaluate(model, paths, size, device, condition="clean"):
    rows, maps = [], []
    with torch.inference_mode():
        # Warm up the current model and CUDA kernels before measuring batch-size-one latency.
        model(image_batch(paths[:1], size, condition).to(device))
        for path in paths:
            tensor = image_batch([path], size, condition).to(device)
            synchronize(device)
            start = time.perf_counter()
            result = model(tensor)
            synchronize(device)
            latency = (time.perf_counter() - start) * 1000
            score = float(result.pred_score[0].cpu())
            anomaly_map = result.anomaly_map[0, 0].cpu().numpy()
            rows.append({"path": str(path.relative_to(DATA)).replace("\\", "/"),
                         "defect": path.parent.name, "label": int(path.parent.name != "good"),
                         "score": score, "latency_ms": latency})
            maps.append(anomaly_map)
    return pd.DataFrame(rows), np.stack(maps)


def pixel_labels(paths, size, root):
    masks = []
    for path in paths:
        if path.parent.name == "good":
            masks.append(np.zeros((size, size), dtype=np.uint8))
        else:
            mask = root / "ground_truth" / path.parent.name / f"{path.stem}_mask.png"
            with Image.open(mask) as image:
                masks.append((np.asarray(image.convert("L").resize(
                    (size, size), Image.Resampling.NEAREST)) > 0).astype(np.uint8))
    return np.stack(masks)


def run(args):
    global DATA
    if args.data_root:
        DATA = Path(args.data_root).resolve()
    if not args.category.replace("_", "").replace("-", "").isalnum():
        raise ValueError("Invalid category name.")
    if args.bank_patches is not None and (args.method != "patchcore" or args.bank_patches < 9):
        raise ValueError("A fixed bank requires PatchCore and at least 9 patches.")
    if args.size < 64 or args.batch_size < 1 or not 0 < args.coreset <= 1:
        raise ValueError("Invalid size, batch size, or coreset ratio.")
    root = DATA / args.category
    validate_custom(root) if args.data_root else validate(root)
    device = device_name() if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; select --device cpu.")
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    train, calibration = split_normal(list((root / "train/good").glob("*.png")),
                                      args.seed, args.calibration_count)
    calibrated_threshold(np.arange(len(calibration)), args.alpha)
    budgets = sorted(set(len(train) if n == "all" else int(n) for n in args.budgets.split(",")))
    if any(n < 2 or n > len(train) for n in budgets):
        raise ValueError(f"Budgets must be between 2 and {len(train)} or 'all'.")
    if args.method == "patchcore" and args.bank_patches is None and int(min(budgets) * (args.size // 8) ** 2 * args.coreset) < 9:
        raise ValueError("The smallest coreset needs at least 9 patches; increase budget, size, or ratio.")
    test = sorted((root / "test").glob("*/*.png"))
    masks = pixel_labels(test, args.size, root)
    run_id = args.name or datetime.now().strftime(f"{args.category}_%Y%m%d_%H%M%S")
    if not run_id.replace("_", "").replace("-", "").isalnum():
        raise ValueError("Run name must contain only letters, digits, hyphens, and underscores.")
    output = ARTIFACTS / run_id
    output.mkdir(parents=True, exist_ok=False)
    config = {"schema_version": 3, "method": args.method, "bank_patches": args.bank_patches, "category": args.category, "backbone": args.backbone, "image_size": args.size,
              "seed": args.seed, "budgets": budgets, "coreset_ratio": args.coreset,
              "calibration_count": len(calibration), "alpha": args.alpha,
              "device": device, "batch_size": args.batch_size, "robustness": args.robustness}
    if args.data_root:
        config["data_root"] = str(DATA)
    versions = {p: importlib.metadata.version(p) for p in ["torch", "torchvision", "anomalib", "timm", "numpy"]}
    write_json(output / "config.json", config)
    write_json(output / "environment.json", {"python": platform.python_version(), "versions": versions,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_hashes": {str(p.relative_to(ROOT)): file_digest(p) for p in (ROOT / "inspection").glob("*.py")}})
    write_json(output / "splits.json", {name: [{"path": str(p.relative_to(DATA)).replace("\\", "/"),
                                              "sha256": file_digest(p)} for p in paths]
                                      for name, paths in [("training_pool", train), ("calibration", calibration), ("test", test)]})
    all_results = []
    for budget in budgets:
        print(f"Building memory bank: {budget} normal images, {device}", flush=True)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if device == "cuda":
            torch.cuda.manual_seed_all(args.seed)
        model = create_model(args.backbone, method=args.method).to(device)
        start = time.perf_counter()
        fit_reference_model(model, train[:budget], args.size, device, args.batch_size,
                            args.coreset, args.bank_patches, args.method)
        synchronize(device)
        fit_seconds = time.perf_counter() - start
        model.eval()
        calibration_rows, calibration_maps = evaluate(model, calibration, args.size, device)
        threshold = calibrated_threshold(calibration_rows.score.to_numpy(), args.alpha)
        heatmap_max = float(np.quantile(calibration_maps, 0.999))
        budget_dir = output / f"n{budget}"
        budget_dir.mkdir()
        calibration_rows.to_csv(budget_dir / "calibration.csv", index=False)
        payload = {"config": config, "budget": budget, "threshold": threshold,
                   "heatmap_max": heatmap_max,
                   "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}}
        torch.save(payload, budget_dir / "model.pt")
        write_json(budget_dir / "checkpoint.json", {"sha256": file_digest(budget_dir / "model.pt"),
                   "backbone": args.backbone, "budget": budget, "threshold": threshold})
        conditions = ["clean", "dim", "blur"] if args.robustness and budget == max(budgets) else ["clean"]
        for condition in conditions:
            print(f"Evaluating n={budget}, condition={condition}, {len(test)} untouched test images", flush=True)
            rows, maps = evaluate(model, test, args.size, device, condition)
            rows["prediction"] = (rows.score > threshold).astype(int)
            rows["correct"] = rows.prediction == rows.label
            rows.to_csv(budget_dir / f"predictions_{condition}.csv", index=False)
            metrics = classification_metrics(rows.label, rows.score, threshold)
            metrics.update({"budget": budget, "condition": condition, "threshold": threshold,
                "pixel_auroc": float(roc_auc_score(masks.ravel(), maps.ravel())),
                "latency_median_ms": float(rows.latency_ms.median()),
                "latency_p95_ms": float(rows.latency_ms.quantile(0.95)),
                "fit_seconds": fit_seconds, "memory_patches": int(model.memory_bank.shape[0]) if args.method == "patchcore" else 0,
                "test_images": len(test)})
            all_results.append(metrics)
            write_json(budget_dir / f"metrics_{condition}.json", metrics)
            if condition == "clean":
                gallery = budget_dir / "gallery"
                gallery.mkdir()
                for index in [0, *rows.index[~rows.correct].tolist()[:8], int(rows.score.idxmax())]:
                    with Image.open(test[index]) as image:
                        overlay(image, maps[index], heatmap_max).save(gallery / f"{index:03d}_overlay.png")
            print(json.dumps(metrics), flush=True)
        pd.DataFrame(all_results).to_csv(output / "summary.csv", index=False)
        del model, payload, maps, calibration_maps
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    from inspection.report import build_report
    build_report(output)
    seal_run(output)
    write_json(output / "complete.json", {"completed_utc": datetime.now(timezone.utc).isoformat(),
        "default_budget": max(budgets), "selection": "Largest prespecified budget; not selected using test performance."})
    (ARTIFACTS / "latest.txt").write_text(run_id, encoding="utf-8")
    print(f"Complete: {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["patchcore", "padim"], default="patchcore")
    parser.add_argument("--bank-patches", type=int)
    parser.add_argument("--data-root", help="Parent of custom category folders; enables custom validation.")
    parser.add_argument("--category", default="bottle")
    parser.add_argument("--budgets", default="16,64,all")
    parser.add_argument("--backbone", default="resnet18", choices=["resnet18", "wide_resnet50_2"])
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calibration-count", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--coreset", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--robustness", action="store_true")
    parser.add_argument("--name")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    run(parser.parse_args())
