"""Local inspection UI over the same inference pipeline used by the CLI."""

import hashlib
import json
import os
import time
from pathlib import Path

import streamlit as st
from PIL import Image

from inspection.data import file_digest
from inspection.evaluation import verify_evaluation
from inspection.images import decode_upload
from inspection.visualization import annotation_overlay, map_bytes, png_bytes, response_overlay
from inspection.workflow import inspect_image, load_detector

ROOT = Path(__file__).resolve().parent
MODEL_ROOT = Path(os.environ.get("INSPECTION_MODEL_ROOT", ROOT / "artifacts"))
DATA_ROOT = Path(os.environ.get("INSPECTION_DATA_ROOT", ROOT / "data" / "mvtec_ad"))

st.set_page_config(page_title="Inspection Lab", layout="wide")
st.markdown("""<style>
.block-container {max-width: 1250px; padding-top: 2.4rem;}
h1 {font-size: 1.8rem !important; font-weight: 600 !important;}
[data-testid="stMetric"] {border-bottom: 1px solid #394047; padding-bottom: .8rem;}
[data-testid="stMetricValue"] {font-size: 1.65rem;}
</style>""", unsafe_allow_html=True)


def available_models():
    models = []
    if MODEL_ROOT.is_dir():
        for path in MODEL_ROOT.rglob("metadata.json"):
            try:
                metadata = json.loads(path.read_text(encoding="utf-8"))
                if (metadata.get("schema_version") in {1, 2}
                        and metadata.get("method") in {"baseline", "patchcore"}
                        and metadata.get("checkpoint") in {"model.npz", "model.pt"}
                        and (path.parent / metadata["checkpoint"]).is_file()):
                    models.append((path.parent, metadata))
            except (OSError, ValueError):
                continue
    # Prefer calibrated PatchCore by method and protocol, independent of test results.
    return sorted(models, key=lambda item: (not bool(item[1].get("calibration")),
                                           item[1]["method"] != "patchcore", str(item[0])))


def model_fingerprint(path, metadata):
    parts = [str(path.resolve()), file_digest(path / "metadata.json"),
             file_digest(path / metadata["checkpoint"])]
    if metadata.get("calibration"):
        parts.append(file_digest(path / "calibration.csv"))
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


@st.cache_resource(max_entries=3)
def cached_detector(path, fingerprint):
    return load_detector(Path(path))


def clear_inspection():
    st.session_state.pop("inspection", None)


def render_inspection(models):
    if not models:
        clear_inspection()
        st.info("Fit a reference model to start inspecting images.")
        st.code("python -m inspection fit --method baseline --root data/mvtec_ad/bottle --output artifacts/baseline")
        st.caption("See README.md for installation and PatchCore fitting. Published results remain available below.")
        return
    selected = st.sidebar.selectbox("Saved model", range(len(models)), format_func=lambda i:
                                   f"{models[i][1]['method']} · {models[i][0].name}")
    model_dir, metadata = models[selected]
    st.sidebar.caption(f"Category: {metadata['category']}\n\nReferences: {metadata['reference_count']}"
                       f"\n\nInput: {metadata['image_size']} px\n\nDevice: CPU")
    calibration = metadata.get("calibration")
    st.sidebar.caption(f"Calibration: {calibration['count']} normal images" if calibration else "Uncalibrated model")
    st.sidebar.caption("Use images with a similar view and background to the fitting references.")
    root = DATA_ROOT / metadata["category"]
    examples = sorted((root / "test").glob("*/*.png"))
    controls = st.columns([1, 1.8, 1])
    with controls[0]:
        source = st.radio("Image source", ["Dataset example", "Upload image"] if examples else ["Upload image"])
    annotation = None
    label = None
    with controls[1]:
        if source == "Dataset example":
            path = st.selectbox("Example", examples, format_func=lambda p: f"{p.parent.name} / {p.name}")
            content = path.read_bytes()
            name = path.name
            label = path.parent.name
            mask_path = root / "ground_truth" / label / f"{path.stem}_mask.png"
            if mask_path.is_file():
                with Image.open(mask_path) as mask:
                    annotation = mask.copy()
        else:
            upload = st.file_uploader("PNG or JPEG", type=["png", "jpg", "jpeg"])
            if upload is None:
                clear_inspection()
                st.caption("Choose an image to inspect.")
                return
            content = upload.getvalue()
            name = Path(upload.name).name
    try:
        image = decode_upload(content)
        fingerprint = model_fingerprint(model_dir, metadata)
    except (OSError, ValueError) as error:
        clear_inspection()
        st.error(str(error))
        return
    digest = hashlib.sha256(content).hexdigest()
    key = (fingerprint, source, label, name, digest)
    if st.session_state.get("inspection", {}).get("key") != key:
        clear_inspection()
    with controls[2]:
        st.write("")
        run = st.button("Run inspection", type="primary", use_container_width=True)
    if run:
        try:
            with st.spinner("Inspecting image..."):
                model, fitted = cached_detector(str(model_dir), fingerprint)
                start = time.perf_counter()
                result, anomaly_map = inspect_image(model, fitted, image, image_name=name, image_sha256=digest)
                result["inspection_ms"] = (time.perf_counter() - start) * 1000
                st.session_state["inspection"] = {"key": key, "result": result, "map": anomaly_map}
        except (OSError, ValueError, RuntimeError, ImportError) as error:
            clear_inspection()
            st.error(f"Inspection could not complete: {error}")
            return
    inspection = st.session_state.get("inspection")
    if inspection is None:
        st.image(image.resize((metadata["image_size"], metadata["image_size"])), caption="Input image", width=350)
        st.caption("Ready to inspect. The saved model and its fixed threshold determine the decision.")
        return
    result = inspection["result"]
    anomaly_map = inspection["map"]
    decision = "Uncalibrated" if result["decision"] is None else (
        "Above threshold" if result["decision"] else "Within threshold")
    metrics = st.columns(4)
    metrics[0].metric("Decision", decision)
    metrics[1].metric("Raw anomaly score", f"{result['raw_anomaly_score']:.4f}")
    metrics[2].metric("Calibrated threshold", f"{result['calibrated_threshold']:.4f}"
                      if result["calibrated_threshold"] is not None else "—")
    metrics[3].metric("Inspection time", f"{result['inspection_ms']:.1f} ms")
    overlay = response_overlay(image, anomaly_map)
    panels = st.columns(3 if annotation is not None else 2)
    panels[0].image(image.resize(overlay.size), caption="Input image", use_container_width=True)
    panels[1].image(overlay, caption="Model response · relative scale", use_container_width=True)
    if annotation is not None:
        panels[2].image(annotation_overlay(image, annotation).resize(overlay.size),
                        caption="Ground truth · cyan annotation", use_container_width=True)
    st.caption("Colors are scaled to this image and are not a segmentation mask. Display colors do not change "
               "the score or decision. Inspection time includes preprocessing and inference after model loading.")
    if label:
        st.caption(f"Dataset label: {label}. Annotations are displayed for reference and never enter inference.")
    downloads = st.columns(3)
    downloads[0].download_button("Download overlay", png_bytes(overlay), "inspection_overlay.png", "image/png")
    downloads[1].download_button("Download raw map", map_bytes(anomaly_map), "anomaly_map.npy", "application/octet-stream")
    downloads[2].download_button("Download result", json.dumps(result, indent=2), "inspection.json", "application/json")


def render_evaluation():
    st.subheader("Initial bottle comparison")
    st.caption("One seed · 16 fitting references · 30 normal calibration images · 83 test images")
    rows = []
    for method in ["baseline", "patchcore"]:
        path = ROOT / "results" / f"bottle-{method}"
        if not path.is_dir():
            continue
        try:
            verified = verify_evaluation(path)
            metrics = verified["metrics"]
            rows.append({"Method": method, "Image AUROC": round(metrics["image_auroc"], 4),
                         "Defect recall": f"{metrics['recall']:.1%}",
                         "Normal false alarms": f"{metrics['fp']}/{metrics['tn'] + metrics['fp']}",
                         "Missed defects": metrics["fn"]})
            with st.expander(f"{method}: predictions and failures"):
                import pandas as pd

                frame = pd.read_csv(path / "predictions.csv")
                st.dataframe(frame[frame.is_anomaly != frame.predicted_anomaly], use_container_width=True, hide_index=True)
                st.download_button("Download predictions", (path / "predictions.csv").read_bytes(),
                                   f"{method}_predictions.csv", "text/csv", key=f"predictions_{method}")
        except (OSError, ValueError) as error:
            st.error(f"Saved {method} evidence could not be verified: {error}")
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)
    st.caption("These are saved test-set results. They do not describe the currently inspected image. "
               "One normal false alarm changes the observed rate by five percentage points.")


st.title("Inspection Lab")
st.sidebar.header("Model")
models = available_models()
inspection_tab, evaluation_tab = st.tabs(["Inspect an image", "Evaluation"])
with inspection_tab:
    render_inspection(models)
with evaluation_tab:
    render_evaluation()
