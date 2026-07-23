from __future__ import annotations

import numpy as np
from matplotlib.colors import TwoSlopeNorm

from hengshui_insar.plotting import cyclic_phase_limits, symmetric_real_imag_norm


def test_phase_cyclic_and_real_imag_diverging() -> None:
    assert cyclic_phase_limits(365.2425)[2] == "twilight_shifted"
    assert isinstance(symmetric_real_imag_norm(np.array([-2.0, 1.0])), TwoSlopeNorm)
