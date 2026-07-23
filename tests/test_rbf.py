from __future__ import annotations

import numpy as np

from hengshui_insar.constants import RBF_DIMENSION
from hengshui_insar.rbf import basis_row_norm, gaussian_rbf


def test_rbf24_layout_and_basis_row_norm_name() -> None:
    assert RBF_DIMENSION == 24
    basis = gaussian_rbf(np.array([[0.0, 0.0], [1.0, 0.0]]), np.array([[0.0, 0.0], [1.0, 0.0]]), 1.0)
    norms = basis_row_norm(basis)
    assert norms.shape == (2,)
    assert np.all(norms > 0)
