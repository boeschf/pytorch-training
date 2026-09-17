"""Deterministic Dataset and DataLoader checks for course preparation."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch.utils.data import DataLoader, Dataset

DatasetFactory = Callable[[torch.Tensor, torch.Tensor], Dataset[tuple[torch.Tensor, torch.Tensor]]]


def make_points(*, samples: int = 20) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a tiny offline regression dataset."""
    if samples < 4:
        raise ValueError("samples must be at least four")
    features = torch.linspace(-1.0, 1.0, samples).unsqueeze(1)
    targets = 3.0 * features.squeeze(1) - 0.5
    return features, targets


def collect_sample_order(dataset: Dataset, *, seed: int) -> list[float]:
    """Return one seeded shuffled epoch as scalar feature values."""
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, generator=generator, num_workers=0)
    return [value for features, _ in loader for value in features.squeeze(1).tolist()]


def dataset_loader_report(factory: DatasetFactory) -> dict[str, object]:
    """Check indexing, batching, complete coverage, and deterministic shuffling."""
    features, targets = make_points()
    dataset = factory(features, targets)
    first_order = collect_sample_order(dataset, seed=7)
    second_order = collect_sample_order(dataset, seed=7)
    expected = sorted(features.squeeze(1).tolist())
    correct = (
        len(dataset) == len(features)
        and torch.equal(dataset[3][0], features[3])
        and torch.equal(dataset[3][1], targets[3])
        and first_order == second_order
        and sorted(first_order) == expected
    )
    return {
        "schema_version": 1,
        "samples": len(dataset),
        "batch_size": 4,
        "batches": (len(dataset) + 3) // 4,
        "deterministic_shuffle": first_order == second_order,
        "complete_epoch": sorted(first_order) == expected,
        "status": "pass" if correct else "fail",
    }
