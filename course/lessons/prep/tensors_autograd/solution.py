"""Reference solution for the tensors and autograd exercise."""

from __future__ import annotations

import torch

from lessons.prep.tensors_autograd.reference import autograd_report


def squared_error(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Return the sum of squared element-wise errors."""
    errors = inputs - targets
    return errors.square().sum()


def main() -> int:
    """Run the completed solution through the analytical gradient check."""
    report = autograd_report(squared_error)
    print(report)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
