"""Expose paired tensors through the PyTorch Dataset protocol.

Outcome: implement indexable samples and inspect deterministic mini-batches.
Prerequisite: Python classes and tensor indexing.
Expected runtime: 20 minutes; self-check runs in under 10 seconds.
Resources: CPU only; no network or downloaded data.
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

from lessons.prep.datasets_loaders.reference import dataset_loader_report


class PairDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Store paired feature and target tensors."""

    def __init__(self, features: torch.Tensor, targets: torch.Tensor) -> None:
        if len(features) != len(targets):
            raise ValueError("features and targets must contain the same number of samples")
        self.features = features
        self.targets = targets

    def __len__(self) -> int:
        raise NotImplementedError("return the number of paired samples")

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("return the feature and target at index")


def main() -> int:
    """Run the participant dataset through a seeded DataLoader self-check."""
    try:
        report = dataset_loader_report(PairDataset)
    except NotImplementedError as error:
        print(f"Status: INCOMPLETE — {error}")
        return 2
    print(report)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
