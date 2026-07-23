"""RBF release helpers."""

from __future__ import annotations

import numpy as np


def gaussian_rbf(points: np.ndarray, centers: np.ndarray, sigma: float) -> np.ndarray:
    diff = points[:, None, :] - centers[None, :, :]
    return np.exp(-0.5 * np.sum(diff * diff, axis=2) / float(sigma) ** 2)


def apply_orthogonal_transform(raw_basis: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return np.asarray(raw_basis, dtype=float) @ np.asarray(transform, dtype=float)


def basis_row_norm(basis: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.asarray(basis, dtype=float), axis=1)
