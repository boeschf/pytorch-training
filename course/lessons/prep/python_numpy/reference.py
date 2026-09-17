"""Deterministic data and checks for the Python and NumPy refresher."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

Standardizer = Callable[[NDArray[np.float64]], NDArray[np.float64]]


def make_measurements(*, samples: int = 12, seed: int = 7) -> NDArray[np.float64]:
    """Return an offline table whose columns use deliberately different scales."""
    if samples < 4:
        raise ValueError("samples must be at least four")
    generator = np.random.default_rng(seed)
    return generator.normal(
        loc=np.array([10.0, 100.0, -5.0]),
        scale=np.array([2.0, 20.0, 0.5]),
        size=(samples, 3),
    )


def evaluate_standardizer(standardize: Standardizer) -> dict[str, object]:
    """Check a standardization function through observable array properties."""
    values = make_measurements()
    normalized = standardize(values)
    means = normalized.mean(axis=0)
    deviations = normalized.std(axis=0)
    preserved_input = not np.shares_memory(values, normalized)
    correct = (
        normalized.shape == values.shape
        and np.allclose(means, 0.0, atol=1e-12)
        and np.allclose(deviations, 1.0, atol=1e-12)
        and preserved_input
    )
    return {
        "schema_version": 1,
        "samples": len(values),
        "features": values.shape[1],
        "column_means": means.round(12).tolist(),
        "column_standard_deviations": deviations.round(12).tolist(),
        "input_preserved": preserved_input,
        "status": "pass" if correct else "fail",
    }
