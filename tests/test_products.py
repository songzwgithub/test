from __future__ import annotations

from hengshui_insar.products import FORMAL_PRODUCTS


def test_products_use_basis_row_norm_not_leverage() -> None:
    assert "basis_row_norm" in FORMAL_PRODUCTS
    assert "rbf_leverage.tif" not in FORMAL_PRODUCTS.values()
