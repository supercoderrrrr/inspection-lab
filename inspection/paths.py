"""Keep model caches and generated artifacts inside the project."""

import os
from pathlib import Path

ROOT = Path(os.environ.get("INSPECTION_HOME", Path(__file__).resolve().parents[1])).resolve()
DATA = Path(os.environ.get("INSPECTION_DATA_ROOT", ROOT / "data" / "mvtec_ad")).resolve()
ARTIFACTS = Path(os.environ.get("INSPECTION_MODEL_ROOT", ROOT / "artifacts")).resolve()
os.environ.setdefault("TORCH_HOME", str(ROOT / ".cache" / "torch"))
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
