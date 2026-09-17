"""Repair the shape and dtype of a broken image-classification batch.

Outcome: diagnose shape, dtype, device, and train/eval state before optimization.
Prerequisite: tensors, DataLoaders, the MLP, and the compact CNN.
Expected runtime: 20 minutes; self-check runs in under 10 seconds.
Resources: CPU only; no network or downloaded data.
"""

from __future__ import annotations

import torch

from lessons.day1.debugging_evaluation.reference import debugging_report


def repair_batch(
    features: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return `[N, 1, H, W]` float features and one-dimensional int64 targets."""
    raise NotImplementedError("add the channel dimension and convert both dtypes")


def main() -> int:
    """Run the participant repair through the diagnostic contract."""
    try:
        report = debugging_report(repair_batch)
    except NotImplementedError as error:
        print(f"Status: INCOMPLETE — {error}")
        return 2
    print(report)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
