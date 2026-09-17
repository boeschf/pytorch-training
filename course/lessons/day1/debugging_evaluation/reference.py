"""Observable diagnostics for common Day 1 classification failures."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn

from lessons.day1.cnn.solution import build_cnn

BatchRepair = Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


def classification_issues(
    model: nn.Module,
    features: torch.Tensor,
    targets: torch.Tensor,
) -> list[str]:
    """Return actionable violations of the course classification contract."""
    issues: list[str] = []
    if features.ndim != 4:
        issues.append("features must have shape [batch, channels, height, width]")
    if targets.ndim != 1:
        issues.append("targets must have shape [batch]")
    if len(features) != len(targets):
        issues.append("features and targets must have the same batch size")
    if not features.is_floating_point():
        issues.append("features must use a floating-point dtype")
    if targets.dtype != torch.long:
        issues.append("CrossEntropyLoss targets must use torch.int64")

    parameter = next(model.parameters(), None)
    if parameter is not None and features.device != parameter.device:
        issues.append("features and model parameters must be on the same device")
    if model.training:
        issues.append("evaluation requires model.eval()")
    return issues


def evaluation_accuracy(
    model: nn.Module,
    features: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    """Return accuracy for a valid batch without building an autograd graph."""
    issues = classification_issues(model, features, targets)
    if issues:
        raise ValueError("; ".join(issues))
    with torch.no_grad():
        predictions = model(features).argmax(dim=1)
    return (predictions == targets).float().mean().item()


def debugging_report(repair_batch: BatchRepair) -> dict[str, object]:
    """Repair a deliberately broken batch and verify actionable diagnostics."""
    model = build_cnn()
    model.eval()
    images = torch.zeros(8, 8, 8, dtype=torch.int32)
    targets = torch.arange(8, dtype=torch.float32) % 2
    before = classification_issues(model, images, targets)
    repaired_images, repaired_targets = repair_batch(images, targets)
    after = classification_issues(model, repaired_images, repaired_targets)
    if after:
        accuracy: float | None = None
    else:
        accuracy = evaluation_accuracy(model, repaired_images, repaired_targets)
    correct = len(before) == 3 and not after and accuracy is not None
    return {
        "schema_version": 1,
        "issues_before": before,
        "issues_after": after,
        "evaluation_accuracy": accuracy,
        "status": "pass" if correct else "fail",
    }
