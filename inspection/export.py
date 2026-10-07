"""Export a compact evidence bundle without dataset images or model weights."""

from inspection.paths import ARTIFACTS, ROOT

import argparse
import json
import zipfile
from pathlib import Path

from inspection.artifacts import atomic_json
from inspection.doctor import audit_run
from inspection.protocol import file_digest


def export_study(name):
    if not name.replace("_", "").replace("-", "").isalnum():
        raise ValueError("Invalid study name.")
    study = ARTIFACTS / "studies" / name
    complete = json.loads((study / "complete.json").read_text(encoding="utf-8"))
    files = [ROOT / p for p in ["README.md", "THIRD_PARTY.md", "MODEL_CARD.md", "CHANGELOG.md",
                               "requirements.txt", "requirements-lock.txt", "pytest.ini"]]
    files += list((ROOT / "inspection").glob("*.py"))
    files += list((ROOT / "tests").glob("*.py"))
    files += [ROOT / "app.py", ROOT / "launch.ps1", ROOT / "Launch.cmd", ROOT / ".streamlit/config.toml"]
    files += list((ROOT / "docs").glob("*.md"))
    files += list((ROOT / "tests").glob("*.cjs"))
    files += [p for p in study.iterdir() if p.suffix in {".json", ".csv", ".md"}]
    for name in complete["runs"]:
        run = (ARTIFACTS / name).resolve()
        if not run.is_relative_to(ARTIFACTS.resolve()):
            raise ValueError("Invalid run path.")
        issues = audit_run(run)
        if issues:
            raise ValueError("; ".join(issues))
        files += [p for p in run.rglob("*") if p.suffix in {".json", ".csv", ".md"}]
    destination = ARTIFACTS / f"{study.name}_evidence.zip"
    manifest = {p.relative_to(ROOT).as_posix(): file_digest(p) for p in files}
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(ROOT).as_posix())
        archive.writestr("EXPORT_MANIFEST.json", json.dumps({"files": manifest,
            "excluded": ["dataset images", "weights", "caches", "environments"],
            "note": "Upstream and dataset licenses remain separate. Export contains evaluation evidence, not a standalone environment."}, indent=2))
    print(destination)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", default="bottle_cpu_v2")
    export_study(parser.parse_args().study)
