"""Inspect a PNG/JPEG file or directory using a trusted local checkpoint."""

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch

from inspection.artifacts import atomic_json
from inspection.images import decode_image, MAX_BYTES
from inspection.model import load_model, overlay, predict_image, synchronize
from inspection.protocol import file_digest


def inspect_paths(checkpoint, input_path, output, device="cpu", save_overlays=False):
    input_path, output = Path(input_path).resolve(), Path(output).resolve()
    if not input_path.exists():
        raise ValueError("Input path does not exist.")
    if input_path.is_dir() and output.is_relative_to(input_path):
        raise ValueError("Output must be outside the input directory.")
    paths = [input_path] if input_path.is_file() else sorted(
        p for p in input_path.rglob("*") if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if not paths:
        raise ValueError("No PNG/JPEG files found.")
    if output.exists():
        raise ValueError("Output already exists; choose a new directory to preserve prior results.")
    torch.set_num_threads(4)
    model, payload, device = load_model(checkpoint, device)
    output.mkdir(parents=True)
    records = []
    for index, path in enumerate(paths):
        record = {"image": path.name if input_path.is_file() else path.relative_to(input_path).as_posix()}
        try:
            if path.stat().st_size > MAX_BYTES:
                raise ValueError("Image exceeds 10 MB.")
            image = decode_image(path.read_bytes())
            synchronize(device)
            start = time.perf_counter()
            score, anomaly_map = predict_image(model, image, payload["config"]["image_size"], device)
            synchronize(device)
            record.update(status="ok", score=score, threshold=payload["threshold"],
                          flagged=score > payload["threshold"], image_sha256=file_digest(path),
                          inspection_ms=(time.perf_counter() - start) * 1000)
            if save_overlays:
                filename = f"{index:06d}_overlay.png"
                overlay(image, anomaly_map, payload["heatmap_max"]).save(output / filename)
                record["overlay"] = filename
        except (ValueError, OSError, RuntimeError) as error:
            record.update(status="error", error=str(error))
        records.append(record)
    failures = sum(row["status"] == "error" for row in records)
    result = {"schema_version": 1, "checkpoint_sha256": file_digest(Path(checkpoint)),
              "category": payload["config"]["category"], "method": payload["config"].get("method", "patchcore"),
              "device": device, "errors": failures, "images": records,
              "timing": "Preprocessing and transfer included; first image is not warmed up."}
    atomic_json(output / "results.json", result)
    pd.DataFrame(records).to_csv(output / "results.csv", index=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--overlays", action="store_true")
    args = parser.parse_args()
    result = inspect_paths(args.checkpoint, args.input, args.output, args.device, args.overlays)
    print(json.dumps({"images": len(result["images"]), "errors": result["errors"], "output": args.output}))
    raise SystemExit(1 if result["errors"] else 0)


if __name__ == "__main__":
    main()
