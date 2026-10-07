"""English-only inspection and experiment browser."""

from inspection.paths import ARTIFACTS, DATA, ROOT

import io
from pathlib import Path
import json
import threading
import time

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

from inspection.model import load_model, overlay, predict_image, synchronize
from inspection.artifacts import completed_runs, verify_run
from inspection.images import decode_image, image_identity, annotation_overlay
from inspection.protocol import wilson_interval, file_digest

st.set_page_config(page_title="Inspection Lab", layout="wide")
st.markdown("""<style>
.block-container {padding-top: 3rem; max-width: 1480px;}
h1 {font-size: 1.8rem !important; font-weight: 600 !important;}
h2,h3 {font-weight: 600 !important;}
[data-testid="stMetric"] {padding: 8px 0; border-bottom: 1px solid #363c43;}
[data-testid="stMetricValue"] {font-size: 1.65rem;}
[data-testid="stSidebar"] {border-right: 1px solid #363c43;}
</style>""", unsafe_allow_html=True)
st.title("Inspection Lab")

runs, rejected_runs = completed_runs(ARTIFACTS)
if not runs:
    st.info("No completed experiment yet. Download the dataset, then run the experiment command below.")
    st.code("python -m inspection.data\npython -m inspection.experiment --robustness")
    public_report = ROOT / "results/controlled_v4/REPORT.md"
    if public_report.exists():
        st.subheader("Published benchmark evidence")
        st.caption("Recorded results are available before installing model artifacts. Extract the model release into this checkout for interactive inference.")
        st.markdown(public_report.read_text(encoding="utf-8"))
    st.stop()

with st.sidebar:
    st.header("Model")
    preferred = (ARTIFACTS / "latest.txt").read_text(encoding="utf-8").strip() if (ARTIFACTS / "latest.txt").exists() else ""
    categories = sorted({json.loads((p / "config.json").read_text(encoding="utf-8"))["category"] for p in runs})
    category = st.selectbox("Product category", categories, format_func=lambda c: c.replace("_", " ").title())
    runs = [p for p in runs if json.loads((p / "config.json").read_text(encoding="utf-8"))["category"] == category]
    category_studies = sorted((ARTIFACTS / "studies").glob("*/complete.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for study_marker in category_studies:
        plan = json.loads((study_marker.parent / "plan.json").read_text(encoding="utf-8"))
        if plan["category"] == category:
            preferred = json.loads(study_marker.read_text(encoding="utf-8"))["default_run"]
            break
    default_index = next((i for i, p in enumerate(runs) if p.name == preferred), 0)
    run = st.selectbox("Completed run", runs, index=default_index, format_func=lambda p: p.name)
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    data_root = Path(config.get("data_root", DATA))
    dataset_name = "Custom dataset" if config.get("data_root") else "MVTec AD"
    budget = st.selectbox("Normal training images", config["budgets"], index=len(config["budgets"]) - 1)
    inference_device = st.selectbox("Inference device", ["cpu", "cuda"], help="CPU avoids contention while GPU experiments or other applications are running. CUDA requires a compatible GPU.")

    st.divider()
    st.markdown(f"**Dataset** · {dataset_name} / {category}\n\n**Method** · {config.get('method', 'patchcore')}\n\n**Backbone** · {config['backbone']}\n\n**Input** · {config['image_size']} px\n\n**Seed** · {config['seed']}")
    st.caption(f"Use {category.replace('_', ' ')} images with a matching camera view and background.")

budget_dir = run / f"n{budget}"
summary = pd.read_csv(run / "summary.csv")
metrics = summary[(summary.budget == budget) & (summary.condition == "clean")].iloc[0]
inspect_tab, experiments_tab, study_tab, research_tab, protocol_tab = st.tabs(["Inspect an image", "Experiments & failures", "Repeated-seed benchmark", "Controlled comparison", "Method & provenance"])


@st.cache_resource(max_entries=2)
def cached_model(path, modified, device):
    import torch
    torch.set_num_threads(4)
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available. Select CPU inference.")
    return (*load_model(path, device=device), threading.Lock())


with inspect_tab:
    controls, selection, action = st.columns([1.2, 2, 1])
    with controls:
        source = st.radio("Input source", ["Dataset example", "Upload image"],
                          index=0 if (data_root / category / "test").exists() else 1)
    image = content = chosen = None
    with selection:
        if source == "Dataset example":
            examples = sorted((data_root / category / "test").glob("*/*.png"))
            if examples:
                chosen = st.selectbox("Example", examples, format_func=lambda p: f"{p.parent.name} / {p.name}")
                content = chosen.read_bytes()
                image = decode_image(content)
                st.caption(f"Dataset label: {chosen.parent.name.replace('_', ' ')}")
            else:
                st.info("Download this category to use dataset examples.")
        else:
            upload = st.file_uploader("PNG or JPEG, up to 10 MB", type=["png", "jpg", "jpeg"])
            if upload:
                try:
                    content = upload.getvalue()
                    image = decode_image(content)
                except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
                    st.error(f"Cannot read image: {exc}")
    with action:
        st.caption(f"{config.get('method', 'patchcore').upper()} · {config['image_size']} px · {inference_device.upper()}")
        inspect = st.button("Run inspection", type="primary", disabled=image is None, use_container_width=True)
    checkpoint = budget_dir / "model.pt"
    input_key = (str(checkpoint), checkpoint.stat().st_mtime_ns, image_identity(content), inference_device) if image is not None else None
    if inspect and image is not None:
        try:
            with st.spinner("Running inspection..."):
                model, payload, device, lock = cached_model(str(checkpoint), checkpoint.stat().st_mtime_ns, inference_device)
                with lock:
                    synchronize(device)
                    start = time.perf_counter()
                    score, anomaly_map = predict_image(model, image, config["image_size"], device)
                    synchronize(device)
                    elapsed = (time.perf_counter() - start) * 1000
            st.session_state["inspection_result"] = {"key": input_key, "score": score,
                "map": anomaly_map, "elapsed": elapsed, "threshold": payload["threshold"],
                "heatmap_max": payload["heatmap_max"], "device": device,
                "checkpoint_sha256": file_digest(checkpoint)}
        except (RuntimeError, OSError, ValueError) as exc:
            st.session_state.pop("inspection_result", None)
            st.error(f"Inspection could not finish: {exc}")
    saved = st.session_state.get("inspection_result")
    if saved and saved["key"] == input_key and image is not None:
        score, anomaly_map, elapsed = saved["score"], saved["map"], saved["elapsed"]
        threshold = saved["threshold"]
        flagged = score > threshold
        status, a, b, c = st.columns(4)
        status.metric("Decision", "Above threshold" if flagged else "Within threshold")
        a.metric("Raw anomaly score", f"{score:.4f}", help="Raw model distance, not a probability.")
        b.metric("Calibrated threshold", f"{threshold:.4f}")
        c.metric("Inspection time", f"{elapsed:.1f} ms", help="Includes preprocessing and output transfer.")

        display, blend, strength = st.columns([1.6, 1.6, 1])
        with display:
            display_scale = st.radio("Color scale", ["Fixed calibration", "Per-image maximum"], index=1,
                                     horizontal=True, help="Fixed: compare images from the same model. Per-image: inspect spatial variation in this image.")
        with blend:
            blend_mode = st.radio("Overlay style", ["Response weighted", "Uniform tint"], horizontal=True,
                                  help="Response weighted uses opacity × (display-normalized response)³. It does not create a defect mask.")
        with strength:
            opacity = st.slider("Overlay opacity", 0.0, 1.0, 0.60, 0.02)
        display_max = saved["heatmap_max"] if display_scale == "Fixed calibration" else max(float(anomaly_map.max()), 1e-8)
        visual = overlay(image, anomaly_map, display_max, opacity, response_weighted=blend_mode == "Response weighted")
        clipped = float(np.mean(anomaly_map > display_max))
        st.caption(f"Distance scale 0–{display_max:.3f} · {clipped:.1%} above color range · "
                   + ("Colors apply to this image only." if display_scale == "Per-image maximum" else "Calibration scale shared across images from this model."))

        original_col, overlay_col, truth_col = st.columns(3)
        preview = ImageOps.exif_transpose(image).convert("RGB").resize((config["image_size"], config["image_size"]))
        original_col.image(preview, caption=f"Input · resized to {config['image_size']} px", use_container_width=True)
        overlay_col.image(visual, caption="Model response", use_container_width=True)
        if chosen is not None and chosen.parent.name != "good":
            mask_path = data_root / category / "ground_truth" / chosen.parent.name / f"{chosen.stem}_mask.png"
            if mask_path.exists():
                with Image.open(mask_path) as mask:
                    truth_col.image(annotation_overlay(preview, mask), caption="Ground truth · cyan annotation", use_container_width=True)
            else:
                truth_col.info("Annotation file missing.")
        elif chosen is not None:
            truth_col.image(preview, caption="Ground truth · no annotated defect", use_container_width=True)
        else:
            truth_col.image(preview, caption="No annotation for uploaded images", use_container_width=True)
        st.caption("Overlay colors show model response, not a segmented defect boundary. Display settings do not change the decision.")

        with st.expander("Raw anomaly map and numeric color scale"):
            raw_chart = px.imshow(anomaly_map, color_continuous_scale="inferno", zmin=0, zmax=display_max,
                                  labels={"color": "Distance"}, aspect="equal")
            raw_chart.update_layout(height=400, margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(raw_chart, use_container_width=True)
            st.caption("This map has no opacity weighting. Fixed scaling can saturate values above its color range.")
            raw_buffer = io.BytesIO()
            np.save(raw_buffer, anomaly_map, allow_pickle=False)
            st.download_button("Download raw map", raw_buffer.getvalue(), "anomaly_map.npy", "application/octet-stream", on_click="ignore")
        encoded = io.BytesIO()
        visual.save(encoded, format="PNG")
        result = {"schema_version": 4, "method": config.get("method", "patchcore"), "category": category,
                  "display_scale": display_scale, "display_max": display_max, "opacity": opacity,
                  "overlay_style": blend_mode, "opacity_exponent": 3 if blend_mode == "Response weighted" else 0,
                  "run": run.name, "budget": budget, "score": score, "threshold": threshold,
                  "flagged": flagged, "latency_ms": elapsed, "device": saved["device"],
                  "image_sha256": input_key[2], "checkpoint_sha256": saved["checkpoint_sha256"],
                  "backbone": config["backbone"], "image_size": config["image_size"], "seed": config["seed"]}
        download_image, download_json, _ = st.columns([1, 1, 2])
        download_image.download_button("Download overlay", encoded.getvalue(), "inspection_overlay.png", "image/png", on_click="ignore")
        download_json.download_button("Download result JSON", json.dumps(result, indent=2), "inspection_result.json", "application/json", on_click="ignore")
    else:
        if image is not None:
            st.image(image, caption="Input image", width=360)
        st.markdown("### Ready to inspect")
        st.caption("Choose an image and run inspection to view its score and response map.")

with experiments_tab:
    clean = summary[summary.condition == "clean"]
    st.caption(f"Recorded clean test set · {category.replace('_', ' ')} · seed {config['seed']} · {budget} references")
    cols = st.columns(4)
    cols[0].metric("Image AUROC", f"{metrics.image_auroc:.3f}")
    cols[1].metric("Defect recall", f"{metrics.recall:.1%}")
    cols[2].metric("Normal false alarms", f"{metrics.false_positive_rate:.1%}")
    cols[3].metric("Model latency · median", f"{metrics.latency_median_ms:.1f} ms")
    st.subheader("Reference budget")
    chart = px.line(clean, x="budget", y=["image_auroc", "recall", "false_positive_rate"], markers=True,
                    labels={"budget": "Normal training images", "value": "Metric", "variable": "Measure"})
    chart.update_yaxes(range=[0, 1.05])
    st.plotly_chart(chart, use_container_width=True)
    st.dataframe(summary, hide_index=True, use_container_width=True)
    st.caption("One seed, one category. More reference images also produce a larger coreset. Perturbation results reuse the clean calibration threshold.")
    st.subheader("Score distribution")
    predictions = pd.read_csv(budget_dir / "predictions_clean.csv")
    histogram = px.histogram(predictions, x="score", color="defect", nbins=25, barmode="overlay")
    histogram.add_vline(x=float(metrics.threshold), line_dash="dash", annotation_text="Calibrated threshold")
    st.plotly_chart(histogram, use_container_width=True)
    st.markdown("#### False positives and missed defects")
    low, high = wilson_interval(int(metrics.fp), int(metrics.fp + metrics.tn))
    st.caption(f"Normal false-alarm 95% Wilson interval for this split: {low:.1%}–{high:.1%}. Only {int(metrics.fp + metrics.tn)} normal test images are available.")
    failures = predictions[predictions.prediction != predictions.label]
    if failures.empty:
        st.info("No image-level errors at this threshold in this test split. This does not establish general deployment accuracy.")
    else:
        st.dataframe(failures, hide_index=True, use_container_width=True)
        failure_path = st.selectbox("Review a failed example", failures.path.tolist())
        if (data_root / failure_path).is_file():
            with Image.open(data_root / failure_path) as failed_image:
                st.image(failed_image, caption=failure_path, width=280)
        else:
            st.info("Failure image is not installed. Recorded scores remain available; download the category dataset to inspect its images.")
    by_defect = predictions.groupby("defect").agg(images=("label", "size"), flagged=("prediction", "sum"))
    st.markdown("#### Breakdown by defect type")
    st.dataframe(by_defect, use_container_width=True)
    st.download_button("Download experiment summary", (run / "summary.csv").read_bytes(), "summary.csv", "text/csv")

with study_tab:
    study_dirs = sorted((ARTIFACTS / "studies").glob("*/complete.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not study_dirs:
        study_dirs = sorted((ROOT / "results").glob("*_cpu_*/complete.json"))
    study_dirs = [p for p in study_dirs if json.loads((p.parent / "plan.json").read_text(encoding="utf-8"))["category"] == category]
    if not study_dirs:
        st.info("No completed repeated-seed study. Run: python -m inspection.study")
    else:
        study = st.selectbox("Benchmark study", [p.parent for p in study_dirs], format_func=lambda p: p.name)
        aggregates = pd.read_csv(study / "aggregate.csv")
        study_clean = aggregates[aggregates.condition == "clean"]
        st.subheader("Stability across normal-reference samples")
        st.caption("Error bars show sample standard deviation across prespecified seeds, not population confidence intervals. Every seed uses the same test images.")
        selected_metric = st.selectbox("Benchmark metric", ["image_auroc", "recall", "false_positive_rate", "pixel_auroc"],
                                       format_func=lambda value: value.replace("_", " ").title())
        chart = px.line(study_clean, x="budget", y=f"{selected_metric}_mean", color="method", error_y=f"{selected_metric}_std", markers=True,
                        labels={"budget": "Normal references", f"{selected_metric}_mean": f"Mean {selected_metric.replace('_', ' ')}"})
        chart.update_yaxes(range=[0, 1.05])
        st.plotly_chart(chart, use_container_width=True)
        shown = ["method", "budget", "condition", "image_auroc_mean", "recall_mean", "false_positive_rate_mean"]
        st.dataframe(aggregates[shown], hide_index=True, use_container_width=True)
        with st.expander("Full statistics and timing"):
            st.dataframe(aggregates, hide_index=True, use_container_width=True)
        with st.expander("Study report"):
            st.markdown((study / "REPORT.md").read_text(encoding="utf-8"))
        st.download_button("Download benchmark CSV", (study / "runs.csv").read_bytes(), "benchmark.csv", "text/csv", on_click="ignore")

with research_tab:
    research = ROOT / "results/controlled_v4"
    if not (research / "aggregate.csv").exists():
        research = ARTIFACTS / "research/controlled_v4"
    if (research / "complete.json").exists():
        comparison = pd.read_csv(research / "aggregate.csv")
        comparison = comparison[comparison.category == category]
        st.subheader("Algorithm and memory-capacity comparison")
        st.caption("PaDiM and fixed-bank PatchCore use identical splits to the original study. Error bars are sample standard deviations across three seeds. These are exploratory follow-up experiments.")
        comparison_metric = st.selectbox("Comparison metric", ["image_auroc", "recall", "false_positive_rate", "pixel_auroc"])
        figure = px.line(comparison, x="budget", y=f"{comparison_metric}_mean", color="method",
                         error_y=f"{comparison_metric}_std", markers=True)
        figure.update_yaxes(range=[0, 1.05])
        st.plotly_chart(figure, use_container_width=True)
        st.dataframe(comparison[["method", "budget", "image_auroc_mean", "recall_mean", "false_positive_rate_mean"]], hide_index=True)
        st.download_button("Download controlled comparison", (research / "runs.csv").read_bytes(), "controlled_comparison.csv", "text/csv")
    else:
        st.info("No completed controlled study yet. Run python -m inspection.research.")

with protocol_tab:
    if rejected_runs:
        st.warning("Some incomplete or invalid runs were excluded: " + "; ".join(rejected_runs))
    if st.button("Verify artifact integrity"):
        with st.spinner("Checking saved artifacts..."):
            issues = verify_run(run)
        if issues:
            st.error("; ".join(issues))
        else:
            st.success("All recorded artifact hashes match.")
    st.markdown((run / "REPORT.md").read_text(encoding="utf-8").replace("![Budget comparison](budget_comparison.png)", ""))
    with st.expander("Environment and source hashes"):
        st.json(json.loads((run / "environment.json").read_text(encoding="utf-8")))
    st.download_button("Download English report", (run / "REPORT.md").read_bytes(), "REPORT.md", "text/markdown")
