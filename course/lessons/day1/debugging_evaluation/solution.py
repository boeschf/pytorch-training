"""Reference solution for the debugging and evaluation exercise."""

from __future__ import annotations

import torch

from lessons.day1.debugging_evaluation.reference import debugging_report


def repair_batch(
    features: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return `[N, 1, H, W]` float features and one-dimensional int64 targets."""
    return features.unsqueeze(1).float(), targets.long()


def main() -> int:
    """Run the completed batch repair through the diagnostic contract."""
    report = debugging_report(repair_batch)
    print(report)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
