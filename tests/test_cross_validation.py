from __future__ import annotations

from hengshui_insar.cross_validation import recalculate_formal_cv


def test_cv_metrics_recalculation() -> None:
    result = recalculate_formal_cv()
    assert result["formal_cv_recalculation_status"] == "passed"
