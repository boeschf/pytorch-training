"""Reference solution for the Python and NumPy refresher."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from lessons.prep.python_numpy.reference import evaluate_standardizer


def standardize_columns(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return columns with zero mean and unit standard deviation."""
    if values.ndim != 2:
        raise ValueError("values must be a two-dimensional array")
    means = values.mean(axis=0)
    deviations = values.std(axis=0)
    if np.any(deviations == 0):
        raise ValueError("every column must have non-zero variance")
    # Keeping the feature axis allows NumPy to broadcast one value per column.
    return (values - means) / deviations


def python_numpy_report() -> dict[str, object]:
    """Return JSON-safe evidence that the reference solution satisfies the lesson."""
    return evaluate_standardizer(standardize_columns)


def main() -> int:
    """Run the completed solution through the participant contract."""
    report = python_numpy_report()
    print(report)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
