from pathlib import Path

import pytest

from inspection.evaluation import verify_evaluation


@pytest.mark.parametrize("method", ["baseline", "patchcore"])
def test_published_initial_comparison_is_recomputable(method):
    root = Path(__file__).resolve().parents[1] / "results" / f"bottle-{method}"
    result = verify_evaluation(root)
    assert result["verified"] is True
    assert result["images"] == 83
    assert sum(result["metrics"][name] for name in ["tn", "fp", "fn", "tp"]) == 83
