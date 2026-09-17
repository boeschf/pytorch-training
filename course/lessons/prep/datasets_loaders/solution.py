"""Reference solution for the Dataset and DataLoader exercise."""

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
        return len(self.features)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.targets[index]


def main() -> int:
    """Run the completed solution through the DataLoader contract."""
    report = dataset_loader_report(PairDataset)
    print(report)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
