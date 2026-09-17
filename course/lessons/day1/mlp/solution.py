"""Complete solution for the adjacent MLP optimization-step exercise."""

from __future__ import annotations

import torch
from torch import nn

from lessons.day1.mlp.exercise import run_exercise


# region optimization-step-solution
def optimization_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_function: nn.Module,
    features: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    """Perform one explicit optimization step and return its loss."""
    optimizer.zero_grad(set_to_none=True)
    logits = model(features)
    loss = loss_function(logits, targets)
    loss.backward()
    optimizer.step()
    return loss.item()


# endregion optimization-step-solution


def main() -> int:
    """Run the solution through the same self-check as the participant exercise."""
    return run_exercise(optimization_step)


if __name__ == "__main__":
    raise SystemExit(main())
