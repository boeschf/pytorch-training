"""Standardize a two-dimensional NumPy array column by column.

Outcome: use axis-aware reductions and broadcasting without Python loops.
Prerequisite: basic Python expressions, functions, and indexing.
Expected runtime: 15 minutes; self-check runs in under one second.
Resources: CPU only; no network or downloaded data.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from lessons.prep.python_numpy.reference import evaluate_standardizer


def standardize_columns(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return columns with zero mean and unit standard deviation."""
    raise NotImplementedError("compute column means and standard deviations, then broadcast")


def main() -> int:
    """Run the bounded participant self-check."""
    try:
        report = evaluate_standardizer(standardize_columns)
    except NotImplementedError as error:
        print(f"Status: INCOMPLETE — {error}")
        return 2
    print(report)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
