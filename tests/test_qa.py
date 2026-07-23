from __future__ import annotations

from hengshui_insar.constants import RELEASE_ROOT
from hengshui_insar.qa import spatial_qa


def test_no_false_distance_product() -> None:
    result = spatial_qa(RELEASE_ROOT)
    assert result["real_distance_product_status"] == "not_available_not_fabricated"
