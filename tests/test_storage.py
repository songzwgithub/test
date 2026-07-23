from __future__ import annotations

from hengshui_insar.storage import recalculate_storage


def test_storage_source_level_metrics_and_structural_envelope() -> None:
    result = recalculate_storage()
    assert result["storage_recalculation_status"] == "passed"
    assert result["delayed_positive_shift_status"] == "passed"
    metrics = result["metrics"]
    assert metrics["local_amplitude_sum_m3"] > metrics["coherent_amplitude_m3"]
