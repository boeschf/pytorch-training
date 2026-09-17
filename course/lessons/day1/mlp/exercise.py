"""Complete one explicit MLP optimization step.

Outcome: implement zero-grad, forward, loss, backward, and optimizer-step operations.
Prerequisite: tensor/device preparation and the introduction to ``torch.nn.Module``.
Expected runtime: under 30 seconds after completing the gaps.
Resources: CPU or one GPU; no network or downloaded dataset.
"""

from __future__ import annotations

import torch
from torch import nn

from pytorch_course.foundations.mlp import MLP, make_dataset
from pytorch_course.foundations.tensors import resolve_device


def optimization_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_function: nn.Module,
    features: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    """Perform one optimization step and return its loss.

    Replace the exception with the five operations listed in the exercise outcome.
    Keep the order explicit; do not introduce a training framework.
    """
    raise NotImplementedError("complete the five operations in optimization_step")


def main() -> int:
    """Train long enough to make a completed exercise self-checking."""
    torch.manual_seed(7)
    device = resolve_device("auto")
    split = make_dataset(samples=512, seed=7)
    features = split.train_features.to(device)
    targets = split.train_targets.to(device)
    model = MLP().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    loss_function = nn.CrossEntropyLoss()

    for _ in range(200):
        optimization_step(model, optimizer, loss_function, features, targets)

    model.eval()
    with torch.no_grad():
        eval_features = split.eval_features.to(device)
        eval_targets = split.eval_targets.to(device)
        accuracy = (model(eval_features).argmax(dim=1) == eval_targets).float().mean().item()
    print(f"evaluation accuracy: {accuracy:.1%}")
    if accuracy < 0.95:
        print("Status: FAIL — revisit the optimization step")
        return 1
    print("Status: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
